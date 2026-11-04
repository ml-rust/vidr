//! AV1 against ffmpeg's libdav1d decode, byte for byte. AV1 decoding is bit-exact by
//! specification, so a mismatch is a defect, never a tolerance to widen. Film grain is exact
//! when the driver follows the specification's process, and a stream with grain tests that.
//!
//! `tests/fixtures/gen_nvdec.py` writes each stream, and exits when an encoder stops writing the
//! tool the stream's name states.

use numr::runtime::wgpu::WgpuRuntime;
use vidr::DecodeError;
use vidr::FrameRoute;
use vidr::bitstream::av1::{Av1Config, Av1ToDecoder};
use vidr::decode::{DecodeCodec, DecodedFrame, Decoder, PixelFormat, PlaneForm};
use vidr::vulkan::VulkanDecoder;

use super::support::{
    assert_no_validation_errors, assert_same_bytes, codec_present, decode_all, decode_in_each_mode,
    device, frame_bytes, samples,
};
use crate::decode_fixtures::{Stream, load};

const CODEC: DecodeCodec = DecodeCodec::Av1;

fn check_stream(stream: &Stream, frames: &[DecodedFrame<WgpuRuntime>], label: &str) {
    let pts: Vec<i64> = frames.iter().map(DecodedFrame::pts).collect();
    assert_eq!(pts, stream.frames, "{label}: pts");
    let format = match stream.meta("format") {
        "nv12" => PixelFormat::Nv12,
        "p010le" | "p010" => PixelFormat::P010,
        other => panic!("{label}: format `{other}`"),
    };
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(frame.format(), format, "{label} frame {index}");
        assert_eq!(
            frame.plane_form(),
            PlaneForm::Words,
            "{label} frame {index}"
        );
    }
    let Some(yuv) = &stream.yuv else {
        panic!("{label}: the fixture has no .yuv");
    };
    assert_same_bytes(&frame_bytes(frames, label), yuv, label);
}

fn decodes_exactly(name: &'static str) {
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, CODEC) {
        return;
    }
    let Some(stream) = load(name) else {
        return;
    };
    let packets = samples(&stream);
    for (preference, frames) in decode_in_each_mode(&device, CODEC, &stream.config, &packets, name)
    {
        check_stream(&stream, &frames, &format!("{name} {preference:?}"));
    }
    assert_no_validation_errors(&device, name);
}

#[test]
fn eight_bits_equal_libdav1d() {
    decodes_exactly("av1_8bit");
}

#[test]
fn ten_bits_equal_libdav1d() {
    decodes_exactly("av1_10bit");
}

#[test]
fn an_odd_size_equals_libdav1d() {
    decodes_exactly("av1_odd");
}

/// Hidden alt-ref frames and `show_existing_frame`: a sample holds several frames, one shown.
#[test]
fn hidden_frames_and_frames_shown_again_equal_libdav1d() {
    decodes_exactly("av1_altref");
}

#[test]
fn hidden_frames_at_ten_bits_equal_libdav1d() {
    decodes_exactly("av1_altref_10bit");
}

/// A hidden key frame, shown later, refreshes every name.
#[test]
fn a_forward_key_frame_equals_libdav1d() {
    decodes_exactly("av1_forward_key");
}

#[test]
fn tiles_equal_libdav1d() {
    decodes_exactly("av1_tiles");
}

/// Tiles across tile group OBUs, each with its offset in the joined bytes.
#[test]
fn tile_groups_equal_libdav1d() {
    decodes_exactly("av1_tile_groups");
}

#[test]
fn cdef_and_loop_restoration_equal_libdav1d() {
    decodes_exactly("av1_restoration");
}

/// The frame keeps its session's width, and decodes at `FrameWidth` under `UpscaledWidth`.
#[test]
fn superres_equals_libdav1d() {
    decodes_exactly("av1_superres");
}

/// Each frame reads references of another size.
#[test]
fn a_size_change_with_reference_scaling_equals_libdav1d() {
    decodes_exactly("av1_resize");
}

/// Grain lands in the output image, and the slot stays free of it. A frame shown again repeats
/// the grained planes of its slot.
#[test]
fn film_grain_equals_libdav1d() {
    decodes_exactly("av1_film_grain");
}

#[test]
fn film_grain_at_ten_bits_equals_libdav1d() {
    decodes_exactly("av1_film_grain_10bit");
}

#[test]
fn screen_content_tools_equal_libdav1d() {
    decodes_exactly("av1_screen");
}

#[test]
fn a_128_superblock_equals_libdav1d() {
    decodes_exactly("av1_sb128");
}

#[test]
fn segmentation_and_quantizer_matrices_equal_libdav1d() {
    decodes_exactly("av1_segmentation");
    decodes_exactly("av1_qm");
}

/// Frame ids, order hints in the header, and no `primary_ref_frame`.
#[test]
fn error_resilient_mode_equals_libdav1d() {
    decodes_exactly("av1_error_resilient");
}

#[test]
fn a_switch_frame_equals_libdav1d() {
    decodes_exactly("av1_sframe");
}

#[test]
fn other_encoders_equal_libdav1d() {
    decodes_exactly("av1_svt");
    decodes_exactly("av1_svt_10bit");
    decodes_exactly("av1_svt_film_grain");
    decodes_exactly("av1_rav1e");
}

/// A bare OBU stream, with the sequence header in band and no `av1C`.
#[test]
fn bare_temporal_units_equal_samples_with_a_config() {
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, CODEC) {
        return;
    }
    let name = "av1_altref";
    let Some(stream) = load(name) else {
        return;
    };
    let config =
        Av1Config::parse(&stream.config).unwrap_or_else(|error| panic!("{name}: av1C: {error}"));
    let mut filter =
        Av1ToDecoder::new(&config).unwrap_or_else(|error| panic!("{name}: filter: {error}"));
    let units: Vec<(Vec<u8>, i64)> = samples(&stream)
        .into_iter()
        .map(|(sample, pts)| {
            let unit = filter
                .filter(&sample)
                .unwrap_or_else(|error| panic!("{name}: filter at pts {pts}: {error}"));
            (unit, pts)
        })
        .collect();
    let mut decoder = VulkanDecoder::new(&device, CODEC, &[])
        .unwrap_or_else(|error| panic!("{name}: open: {error}"));
    assert_eq!(decoder.route(), FrameRoute::Device, "{name}: route");
    let frames = decode_all(&mut decoder, units, name);
    check_stream(&stream, &frames, name);
    assert_no_validation_errors(&device, name);
}

/// A key frame under a new sequence header opens a new session, and both parts stay byte-exact.
#[test]
fn a_new_sequence_header_opens_a_new_session() {
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, CODEC) {
        return;
    }
    let (Some(first), Some(second)) = (load("av1_8bit"), load("av1_tiles")) else {
        return;
    };
    assert_ne!(
        first.number("width"),
        second.number("width"),
        "the second stream changes the size"
    );
    let name = "av1_8bit then av1_tiles";
    let last = first.pts.iter().copied().max().unwrap_or(0);
    // `configOBUs` follows the 4 bytes of the record. So the first sample of the second stream
    // takes its sequence header in band.
    let mut header_first = true;
    let packets = samples(&first)
        .into_iter()
        .chain(samples(&second).into_iter().map(|(sample, pts)| {
            let packet = if header_first {
                [&second.config[4..], &sample[..]].concat()
            } else {
                sample
            };
            header_first = false;
            (packet, pts + last + 1)
        }));
    let mut decoder = VulkanDecoder::new(&device, CODEC, &first.config)
        .unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let frames = decode_all(&mut decoder, packets, name);
    let (Some(first_yuv), Some(second_yuv)) = (&first.yuv, &second.yuv) else {
        panic!("{name}: a fixture has no .yuv");
    };
    let mut yuv = first_yuv.clone();
    yuv.extend_from_slice(second_yuv);
    assert_same_bytes(&frame_bytes(&frames, name), &yuv, name);
    assert_no_validation_errors(&device, name);
}

#[test]
fn other_layouts_are_refused_by_name_at_open() {
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, CODEC) {
        return;
    }
    for name in ["av1_12bit", "av1_444", "av1_422", "av1_mono"] {
        let Some(stream) = load(name) else {
            continue;
        };
        let error = VulkanDecoder::new(&device, CODEC, &stream.config)
            .err()
            .unwrap_or_else(|| panic!("{name}: opens"));
        assert!(
            matches!(error, DecodeError::UnsupportedLayout { .. }),
            "{name}: {error}"
        );
    }
}

#[test]
fn an_inter_frame_first_is_refused() {
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, CODEC) {
        return;
    }
    let name = "av1_8bit";
    let Some(stream) = load(name) else {
        return;
    };
    let mut decoder = VulkanDecoder::new(&device, CODEC, &stream.config)
        .unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let (packet, pts) = &samples(&stream)[1];
    assert!(decoder.send(packet, *pts).is_err(), "{name}: refused");
    assert_no_validation_errors(&device, name);
}

/// A packet in Annex B framing: temporal unit size, frame unit size, a delimiter, and a frame.
#[test]
fn annex_b_framing_is_refused_by_name() {
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, CODEC) {
        return;
    }
    let mut decoder =
        VulkanDecoder::new(&device, CODEC, &[]).unwrap_or_else(|error| panic!("open: {error}"));
    let error = decoder
        .send(&[7, 6, 1, 0x10, 3, 0x30, 0xaa, 0xbb], 0)
        .expect_err("refused");
    let text = error.to_string();
    assert!(matches!(error, DecodeError::VulkanRefused { .. }), "{text}");
    assert!(text.contains("Annex B"), "{text}");
}

/// A cut through a sample reaches a named error or fewer frames, with the device still fine.
#[test]
fn a_cut_sample_is_a_named_error_and_the_error_repeats() {
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, CODEC) {
        return;
    }
    let name = "av1_8bit";
    let Some(stream) = load(name) else {
        return;
    };
    let (packet, pts) = &samples(&stream)[0];
    let mut decoder = VulkanDecoder::new(&device, CODEC, &stream.config)
        .unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let end = packet.len() / 2;
    let error = decoder.send(&packet[..end], *pts).expect_err("refused");
    assert!(
        matches!(error, DecodeError::Bitstream(_)),
        "{name}: {error}"
    );
    assert_eq!(decoder.send(packet, *pts), Err(error), "the error repeats");
    assert_no_validation_errors(&device, name);
}
