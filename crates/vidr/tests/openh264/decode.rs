//! `Openh264Decoder` against ffmpeg's decode, and its refusals.

use vidr::bitstream::{AvcConfig, ToAnnexB};
use vidr::decode::Decoder;
use vidr::{DecodeError, FrameRoute};

use super::support;
use crate::decode_fixtures::cpu::{check_frames, decode_all, samples};
use crate::decode_fixtures::{Stream, load};

const DECODER: &str = "OpenH264";

fn annexb_units(stream: &Stream) -> Vec<(Vec<u8>, i64)> {
    let name = stream.name;
    let config =
        AvcConfig::parse(&stream.config).unwrap_or_else(|error| panic!("{name}: avcC: {error}"));
    let mut filter = ToAnnexB::h264(&config);
    samples(stream)
        .into_iter()
        .map(|(sample, pts)| {
            let unit = filter
                .filter(&sample)
                .unwrap_or_else(|error| panic!("{name}: filter at pts {pts}: {error}"));
            (unit, pts)
        })
        .collect()
}

/// Constrained Baseline container samples, every byte against ffmpeg's decode.
#[test]
fn baseline_samples_equal_ffmpeg() {
    let name = "h264_baseline";
    let Some(stream) = load(name) else {
        return;
    };
    let Some(decoder) = support::decoder(&stream.config) else {
        return;
    };
    let mut decoder = decoder.unwrap_or_else(|error| panic!("{name}: open: {error}"));
    assert_eq!(decoder.route(), FrameRoute::HostOnly, "{name}: route");
    let frames = decode_all(&mut decoder, samples(&stream), name);
    check_frames(&stream, &frames, None);
}

/// The same stream as Annex-B access units, with no config record.
#[test]
fn baseline_annexb_equals_ffmpeg() {
    let name = "h264_baseline";
    let Some(stream) = load(name) else {
        return;
    };
    let Some(decoder) = support::decoder(&[]) else {
        return;
    };
    let mut decoder = decoder.unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let frames = decode_all(&mut decoder, annexb_units(&stream), name);
    check_frames(&stream, &frames, None);
}

/// Both paths name the refusal: `new` from `avcC`, and `send` from the SPS in
/// the first access unit, which also stays for every later call.
fn refuses(name: &'static str, expected: DecodeError) {
    let Some(stream) = load(name) else {
        return;
    };
    let Some(from_config) = support::decoder(&stream.config) else {
        return;
    };
    assert_eq!(from_config.err(), Some(expected.clone()), "{name}: new");

    let Some(raw) = support::decoder(&[]) else {
        return;
    };
    let mut raw = raw.unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let Some((unit, pts)) = annexb_units(&stream).into_iter().next() else {
        panic!("{name}: the fixture has no samples");
    };
    assert_eq!(raw.send(&unit, pts), Err(expected.clone()), "{name}: send");
    assert_eq!(raw.flush(), Err(expected), "{name}: flush");
}

#[test]
fn ten_bits_are_refused_by_name() {
    refuses(
        "h264_10bit",
        DecodeError::H264BitDepth {
            decoder: DECODER,
            luma: 10,
            chroma: 10,
        },
    );
}

#[test]
fn mbaff_is_refused_by_name() {
    refuses(
        "h264_mbaff_interlaced",
        DecodeError::H264Interlaced { decoder: DECODER },
    );
}

/// `frame_mbs_only_flag` 0 with frame pictures: the C++ decoder refuses the SPS.
#[test]
fn a_flagged_frame_stream_is_refused_by_name() {
    refuses(
        "h264_frame_flagged",
        DecodeError::H264Interlaced { decoder: DECODER },
    );
}

/// High with B-frames. The reason for the refusal is in `b_frame_report`.
#[test]
fn high_profile_is_refused_by_name() {
    refuses(
        "h264_8bit",
        DecodeError::H264Profile {
            decoder: DECODER,
            profile_idc: 100,
        },
    );
}

#[test]
fn calls_out_of_order_are_refused() {
    let Some(decoder) = support::decoder(&[]) else {
        return;
    };
    let mut decoder = decoder.unwrap_or_else(|error| panic!("open: {error}"));
    assert_eq!(
        decoder.send(&[], 5),
        Err(DecodeError::EmptyPacket { pts: 5 })
    );
    assert_eq!(decoder.flush(), Ok(()));
    assert_eq!(decoder.flush(), Ok(()));
    assert_eq!(
        decoder.send(&[0, 0, 1, 9, 0xf0], 6),
        Err(DecodeError::AfterFlush { pts: 6 })
    );
}

/// Data before the first start code stops at the filter, and no pts stays queued.
#[test]
fn a_packet_with_no_start_code_is_a_bitstream_error() {
    let Some(decoder) = support::decoder(&[]) else {
        return;
    };
    let mut decoder = decoder.unwrap_or_else(|error| panic!("open: {error}"));
    assert!(matches!(
        decoder.send(&[0x65, 0x88], 1),
        Err(DecodeError::Bitstream(_))
    ));
}

/// Runs OpenH264 itself on High with B-frames, past vidr's refusal, and prints
/// how many frames equal the reference. The count is the reason for the refusal,
/// and a run on a later OpenH264 shows when it no longer holds. The PSNR floor
/// proves display order: a frame in the wrong position falls far below it.
#[cfg(feature = "openh264")]
#[test]
fn b_frame_report() {
    use ::openh264::OpenH264API;
    use ::openh264::decoder::{Decoder as Library, DecoderConfig, Flush};
    use ::openh264::formats::YUVSource;

    const ORDER_FLOOR_DB: f64 = 30.0;

    let name = "h264_8bit";
    let Some(stream) = load(name) else {
        return;
    };
    let Some(yuv) = &stream.yuv else {
        panic!("{name}: the fixture has no .yuv");
    };
    let (width, height) = (
        stream.number("width") as usize,
        stream.number("height") as usize,
    );
    let frame_bytes = width * height * 3 / 2;
    let settings = DecoderConfig::new().flush_after_decode(Flush::NoFlush);
    let mut library = Library::with_api_config(OpenH264API::from_source(), settings)
        .unwrap_or_else(|error| panic!("{name}: open: {error}"));

    let mut planes: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = Vec::new();
    let copy = |picture: &::openh264::decoder::DecodedYUV<'_>| {
        let (y_stride, chroma_stride, _) = picture.strides();
        let rows = |plane: &[u8], stride: usize, columns: usize, count: usize| -> Vec<u8> {
            plane
                .chunks(stride)
                .take(count)
                .flat_map(|row| row[..columns].iter().copied())
                .collect()
        };
        (
            rows(picture.y(), y_stride, width, height),
            rows(picture.u(), chroma_stride, width / 2, height / 2),
            rows(picture.v(), chroma_stride, width / 2, height / 2),
        )
    };
    for (unit, pts) in annexb_units(&stream) {
        let picture = library
            .decode(&unit)
            .unwrap_or_else(|error| panic!("{name}: decode at pts {pts}: {error}"));
        if let Some(picture) = picture {
            planes.push(copy(&picture));
        }
    }
    let remaining = library
        .flush_remaining()
        .unwrap_or_else(|error| panic!("{name}: flush: {error}"));
    planes.extend(remaining.iter().map(copy));
    assert_eq!(planes.len(), stream.frames.len(), "{name}: frame count");

    let mut exact = 0;
    for (index, (y, u, v)) in planes.iter().enumerate() {
        let reference = &yuv[index * frame_bytes..(index + 1) * frame_bytes];
        let (reference_y, reference_uv) = reference.split_at(width * height);
        let mut uv = Vec::with_capacity(reference_uv.len());
        for (u, v) in u.iter().zip(v) {
            uv.extend([*u, *v]);
        }
        exact += usize::from(y == reference_y && uv == reference_uv);
        let db = support::psnr(y, reference_y);
        assert!(
            db > ORDER_FLOOR_DB,
            "{name}: frame {index} luma PSNR {db:.2} dB: display order is wrong"
        );
    }
    eprintln!(
        "OpenH264 on High with B-frames: {exact} of {} frames equal the reference.",
        planes.len()
    );
}
