//! NVDEC output against ffmpeg's software decode, byte for byte.
//!
//! H.264, HEVC, AV1, and VP9 decoding is bit-exact by specification, so the
//! planes must equal ffmpeg's `nv12` or `p010le` output exactly. A mismatch
//! is an NVDEC or vidr defect, never a tolerance to widen.

use numr::dtype::DType;
use numr::error::Error as NumrError;
use numr::ops::{ChromaSiting, ColorMatrix, ColorOps, ColorRange, ConvLayout, YuvSpec};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;
use vidr::bitstream::av1::Av1Config;
use vidr::bitstream::{
    AvcConfig, HevcConfig, HevcNalType, ToAnnexB, VuiSignal, h264_sps_vui, hevc_sps_vui,
};
use vidr::decode::nvdec::NvdecDecoder;
use vidr::decode::{ChromaLocation, DecodeCodec, DecodedFrame, Decoder, PixelFormat};

use super::fixtures::{self, Stream};

/// Sends every packet, receiving after each, then flushes and drains,
/// through the backend-neutral trait.
fn run(
    decoder: &mut dyn Decoder<CudaRuntime>,
    packets: &[(Vec<u8>, i64)],
    name: &str,
) -> Vec<DecodedFrame<CudaRuntime>> {
    let mut frames = Vec::new();
    for (packet, pts) in packets {
        if let Err(error) = decoder.send(packet, *pts) {
            panic!("{name}: send at pts {pts}: {error}");
        }
        while let Some(frame) = decoder.receive() {
            frames.push(frame);
        }
    }
    if let Err(error) = decoder.flush() {
        panic!("{name}: flush: {error}");
    }
    while let Some(frame) = decoder.receive() {
        frames.push(frame);
    }
    frames
}

fn open(client: &CudaClient, stream: &Stream, config: &[u8]) -> NvdecDecoder {
    NvdecDecoder::new(client, stream.codec(), config)
        .unwrap_or_else(|error| panic!("{}: open: {error}", stream.name))
}

fn samples(stream: &Stream) -> Vec<(Vec<u8>, i64)> {
    stream
        .samples
        .iter()
        .cloned()
        .zip(stream.pts.iter().copied())
        .collect()
}

/// The bytes of a plane in ffmpeg's little-endian raw layout.
fn plane_bytes(plane: &Tensor<CudaRuntime>, format: PixelFormat, name: &str) -> Vec<u8> {
    match format {
        PixelFormat::Nv12 => plane
            .try_to_vec::<u8>()
            .unwrap_or_else(|error| panic!("{name}: reading a plane: {error}")),
        PixelFormat::P010 => plane
            .try_to_vec::<u16>()
            .unwrap_or_else(|error| panic!("{name}: reading a plane: {error}"))
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect(),
        other => panic!("{name}: unexpected format {other}"),
    }
}

/// Checks the frames against ffmpeg's decode and the fixture's facts.
fn check(stream: &Stream, frames: &[DecodedFrame<CudaRuntime>]) {
    let name = stream.name;
    let pts: Vec<i64> = frames.iter().map(DecodedFrame::pts).collect();
    assert_eq!(
        pts, stream.frames,
        "{name}: display-order pts differ from ffmpeg's"
    );

    let (width, height) = (stream.number("width"), stream.number("height"));
    let format = match stream.meta("format") {
        "nv12" => PixelFormat::Nv12,
        "p010" => PixelFormat::P010,
        other => panic!("{name}: format `{other}`"),
    };
    let chroma_rows = height.div_ceil(2) as usize;
    let chroma_columns = (width.div_ceil(2) * 2) as usize;
    let sample = format.bytes_per_sample();
    let frame_bytes = (width as usize * height as usize + chroma_rows * chroma_columns) * sample;
    let Some(yuv) = &stream.yuv else {
        panic!("{name}: the fixture has no .yuv");
    };
    assert_eq!(
        yuv.len(),
        frame_bytes * frames.len(),
        "{name}: ffmpeg decodes {} bytes, {} frames of {frame_bytes} expected",
        yuv.len(),
        frames.len()
    );

    // NVDEC reports no chroma location. vidr reads it from `av1C` for AV1,
    // and from the SPS VUI for H.264 and HEVC: the fixture records the
    // `chroma_sample_loc_type` the encoder wrote. VP9 and streams whose SPS
    // signals none give `None`.
    let location = match stream.codec() {
        DecodeCodec::Av1 => Av1Config::parse(&stream.config)
            .map(|config| ChromaLocation::from_av1(config.chroma_sample_position))
            .unwrap_or_else(|error| panic!("{name}: av1C: {error}")),
        _ => stream.meta_opt("chroma_location").map(|value| {
            let kind: u8 = value
                .parse()
                .unwrap_or_else(|error| panic!("{name}: chroma_location `{value}`: {error}"));
            ChromaLocation::from_h273(kind)
                .unwrap_or_else(|| panic!("{name}: chroma_location {kind} is not 0 to 5"))
        }),
    };

    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(frame.format(), format, "{name} frame {index}: format");
        assert_eq!(
            frame.display().width(),
            width,
            "{name} frame {index}: width"
        );
        assert_eq!(
            frame.display().height(),
            height,
            "{name} frame {index}: height"
        );
        assert_eq!(
            frame.y().shape(),
            &[height as usize, width as usize],
            "{name} frame {index}: luma shape"
        );
        assert_eq!(
            frame.uv().shape(),
            &[chroma_rows, chroma_columns],
            "{name} frame {index}: chroma shape"
        );

        let color = frame.color();
        assert_eq!(
            color.chroma_location, location,
            "{name} frame {index}: chroma location"
        );
        // VP9 carries no primaries or transfer in its bitstream, and the
        // fixture's values come from `vpcC`, which the decoder ignores.
        if stream.codec() != DecodeCodec::Vp9 {
            assert_eq!(
                u32::from(color.matrix_coefficients),
                stream.number("matrix"),
                "{name} frame {index}: matrix"
            );
            assert_eq!(
                u32::from(color.transfer_characteristics),
                stream.number("transfer"),
                "{name} frame {index}: transfer"
            );
            assert_eq!(
                u32::from(color.colour_primaries),
                stream.number("primaries"),
                "{name} frame {index}: primaries"
            );
            assert_eq!(
                color.full_range,
                stream.meta("range") == "pc",
                "{name} frame {index}: range"
            );
        }

        let mut bytes = plane_bytes(frame.y(), format, name);
        bytes.extend(plane_bytes(frame.uv(), format, name));
        let expected = &yuv[index * frame_bytes..(index + 1) * frame_bytes];
        if bytes != expected {
            let first = bytes
                .iter()
                .zip(expected)
                .position(|(ours, theirs)| ours != theirs)
                .unwrap_or(0);
            panic!(
                "{name} frame {index} (pts {}): NVDEC output differs from ffmpeg's at byte \
                 {first} of {frame_bytes}: {} against {}",
                frame.pts(),
                bytes[first],
                expected[first]
            );
        }
    }
}

/// Decodes fixture `name` from its MP4 samples and config record.
fn decodes_samples(name: &'static str) {
    let Some(client) = fixtures::gpu() else {
        return;
    };
    let Some(stream) = fixtures::load(name) else {
        return;
    };
    let mut decoder = open(&client, &stream, &stream.config);
    let frames = run(&mut decoder, &samples(&stream), name);
    check(&stream, &frames);
}

#[test]
fn h264_8bit_matches_ffmpeg() {
    decodes_samples("h264_8bit");
}

#[test]
fn hevc_8bit_matches_ffmpeg() {
    decodes_samples("hevc_8bit");
}

#[test]
fn hevc_10bit_matches_ffmpeg() {
    decodes_samples("hevc_10bit");
}

/// The SPS VUI signals chroma sample location type 1, which is `Center`.
#[test]
fn h264_chroma_location_comes_from_the_sps() {
    decodes_samples("h264_chroma_center");
}

#[test]
fn hevc_chroma_location_comes_from_the_sps() {
    decodes_samples("hevc_chroma_center");
}

#[test]
fn av1_8bit_matches_ffmpeg() {
    decodes_samples("av1_8bit");
}

#[test]
fn vp9_8bit_matches_ffmpeg() {
    decodes_samples("vp9_8bit");
}

/// AV1 and VP9 carry odd display sizes. H.264 and HEVC crop 4:2:0 frames in
/// units of two samples, so they cannot. The fixture is 321x181, and
/// `check` compares both planes with ffmpeg's `nv12` output, whose chroma
/// plane is `ceil(H / 2)` rows of `2 * ceil(W / 2)` bytes.
#[test]
fn av1_odd_size_matches_ffmpeg() {
    decodes_samples("av1_odd");
}

#[test]
fn vp9_odd_size_matches_ffmpeg() {
    decodes_samples("vp9_odd");
}

/// numr converts even sizes only, and says so for an odd frame.
fn odd_size_is_refused_by_numr(name: &'static str) {
    let Some(client) = fixtures::gpu() else {
        return;
    };
    let Some(stream) = fixtures::load(name) else {
        return;
    };
    let mut decoder = open(&client, &stream, &stream.config);
    let frames = run(&mut decoder, &samples(&stream), name);
    let Some(frame) = frames.first() else {
        panic!("{name}: no frame decoded");
    };
    assert_eq!(frame.display().width() % 2, 1, "{name}: width is not odd");
    assert_eq!(frame.display().height() % 2, 1, "{name}: height is not odd");
    let spec = YuvSpec::new(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left);
    match client.yuv_to_rgb(frame.planes(), &spec, DType::F32, ConvLayout::Nchw) {
        Err(NumrError::InvalidArgument { arg, reason }) => {
            assert_eq!(arg, "planes", "{name}: argument");
            assert!(!reason.is_empty(), "{name}: empty reason");
        }
        Err(other) => panic!("{name}: expected InvalidArgument for planes, got {other}"),
        Ok(_) => panic!("{name}: numr converted an odd-size frame"),
    }
}

#[test]
fn av1_odd_size_is_refused_by_numr() {
    odd_size_is_refused_by_numr("av1_odd");
}

#[test]
fn vp9_odd_size_is_refused_by_numr() {
    odd_size_is_refused_by_numr("vp9_odd");
}

/// Annex-B packets with an empty config give the same frames as samples
/// with the config record.
fn decodes_annexb(name: &'static str) {
    let Some(client) = fixtures::gpu() else {
        return;
    };
    let Some(stream) = fixtures::load(name) else {
        return;
    };
    let mut filter = match stream.codec() {
        DecodeCodec::H264 => AvcConfig::parse(&stream.config).map(|c| ToAnnexB::h264(&c)),
        DecodeCodec::Hevc => HevcConfig::parse(&stream.config).map(|c| ToAnnexB::hevc(&c)),
        other => panic!("{name}: {other} has no Annex-B form"),
    }
    .unwrap_or_else(|error| panic!("{name}: config: {error}"));
    let packets: Vec<(Vec<u8>, i64)> = samples(&stream)
        .into_iter()
        .map(|(sample, pts)| {
            let unit = filter
                .filter(&sample)
                .unwrap_or_else(|error| panic!("{name}: filter: {error}"));
            (unit, pts)
        })
        .collect();
    let mut decoder = open(&client, &stream, &[]);
    let frames = run(&mut decoder, &packets, name);
    check(&stream, &frames);
}

#[test]
fn h264_annexb_input_matches_ffmpeg() {
    decodes_annexb("h264_8bit");
}

#[test]
fn hevc_annexb_input_matches_ffmpeg() {
    decodes_annexb("hevc_10bit");
}

/// With an empty config the location comes from the in-band SPS.
#[test]
fn h264_annexb_chroma_location_comes_from_the_in_band_sps() {
    decodes_annexb("h264_chroma_center");
}

#[test]
fn hevc_annexb_chroma_location_comes_from_the_in_band_sps() {
    decodes_annexb("hevc_chroma_center");
}

/// The SPS in each config record signals the location the generator wrote.
/// The check needs no GPU.
#[test]
fn config_sps_carries_the_chroma_location() {
    for (name, expected) in [
        ("h264_8bit", None),
        ("h264_chroma_center", Some((1, 1))),
        ("hevc_8bit", None),
        ("hevc_chroma_center", Some((1, 1))),
    ] {
        let Some(stream) = fixtures::load(name) else {
            continue;
        };
        let signal: VuiSignal = match stream.codec() {
            DecodeCodec::H264 => {
                let config = AvcConfig::parse(&stream.config)
                    .unwrap_or_else(|error| panic!("{name}: avcC: {error}"));
                h264_sps_vui(&config.sps[0])
            }
            _ => {
                let config = HevcConfig::parse(&stream.config)
                    .unwrap_or_else(|error| panic!("{name}: hvcC: {error}"));
                let sps = config
                    .arrays
                    .iter()
                    .find(|array| array.nal_unit_type == HevcNalType::SPS)
                    .unwrap_or_else(|| panic!("{name}: hvcC has no SPS array"));
                hevc_sps_vui(&sps.nal_units[0])
            }
        }
        .unwrap_or_else(|error| panic!("{name}: SPS VUI: {error}"));
        let found = signal
            .chroma_sample_loc
            .map(|loc| (loc.top_field, loc.bottom_field));
        assert_eq!(found, expected, "{name}: chroma sample location types");
    }
}
