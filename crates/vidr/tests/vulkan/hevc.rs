//! HEVC against its reference decode, byte for byte, in output order. HEVC decoding is
//! bit-exact by specification, so a mismatch is a defect, never a tolerance to widen.
//!
//! Each `_vk` stream is sized for a driver's minimum coded size. ffmpeg's decode is the
//! reference, except where `gen_hevc.py` names x265's reconstruction. There ffmpeg's chroma
//! differs from the hash SEI message. `gen_hevc.py` writes each `.yuv`.
//!
//! x265 writes no tiles, so tiles run only through the conformance sweep.

use numr::dtype::DType;
use numr::ops::{ChromaSiting, ConvLayout};
use numr::runtime::wgpu::WgpuRuntime;
use numr::tensor::Tensor;
use vidr::DecodeError;
use vidr::FrameRoute;
use vidr::decode::{ChromaLocation, DecodeCodec, DecodedFrame, Decoder, PixelFormat, PlaneForm};
use vidr::frame::to_rgb;
use vidr::vulkan::VulkanDecoder;

use super::support::{
    assert_no_validation_errors, assert_same_bytes, codec_present, decode_all, decode_in_each_mode,
    device, frame_bytes, hevc_units, planar_bytes, samples,
};
use crate::decode_fixtures::{Stream, load, raw_hevc};

/// 8 and 10 bits: the prediction and filter tools, a B pyramid, several slices, and WPP. Also
/// weighted prediction, scaling lists, conformance windows, and CRA pictures with decoded
/// leading pictures. Also an end of sequence unit, POC wrap, long GOPs, and lossless blocks.
const STREAMS: [&str; 28] = [
    "inter_8_vk",
    "bframes_8_vk",
    "bframes_10_vk",
    "b_pyramid_8_vk",
    "b_pyramid_10_vk",
    "p_only_8_vk",
    "p_only_10_vk",
    "slices_inter_8_vk",
    "wavefront_8_vk",
    "wavefront_inter_8_vk",
    "weightp_8_vk",
    "weightp_10_vk",
    "weightb_8_vk",
    "weightb_10_vk",
    "scaling_default_8_vk",
    "scaling_default_10_vk",
    "window_inter_8_vk",
    "open_gop_8_vk",
    "end_of_sequence_8_vk",
    "poc_wrap_8_vk",
    "long_gop_8_vk",
    "refs_4_8_vk",
    "amp_8_vk",
    "amp_10_vk",
    "tmvp_off_8_vk",
    "cu_lossless_inter_8_vk",
    "rect_8_vk",
    "ctu16_inter_8_vk",
];

#[test]
fn every_stream_equals_its_reference_in_each_dpb_mode() {
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, DecodeCodec::Hevc) {
        return;
    }
    for name in STREAMS {
        let Some((stream, reference)) = raw_hevc(name) else {
            continue;
        };
        let packets = hevc_units(&stream, name);
        let runs = decode_in_each_mode(&device, DecodeCodec::Hevc, &[], &packets, name);
        for (preference, frames) in runs {
            for frame in &frames {
                assert_eq!(frame.plane_form(), PlaneForm::Words, "{name}");
            }
            let label = format!("{name} {preference:?}");
            assert_same_bytes(&planar_bytes(&frames, name), &reference, &label);
        }
        assert_no_validation_errors(&device, name);
    }
}

/// The first CRA picture has `NoRaslOutputFlag` 1, so its RASL pictures do not decode and
/// give no frame. The reference holds what remains.
#[test]
fn rasl_pictures_of_a_first_cra_picture_are_skipped() {
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, DecodeCodec::Hevc) {
        return;
    }
    let name = "rasl_cut_8_vk";
    let Some((stream, reference)) = raw_hevc(name) else {
        return;
    };
    let packets = hevc_units(&stream, name);
    let count = packets.len();
    for (preference, frames) in decode_in_each_mode(&device, DecodeCodec::Hevc, &[], &packets, name)
    {
        assert!(frames.len() < count, "{name}: no picture is skipped");
        let label = format!("{name} {preference:?}");
        assert_same_bytes(&planar_bytes(&frames, name), &reference, &label);
    }
    assert_no_validation_errors(&device, name);
}

/// `no_output_of_prior_pics_flag` 1 drops each frame that waits, as C.5.2.2 states.
#[test]
fn no_output_of_prior_pics_drops_what_waits() {
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, DecodeCodec::Hevc) {
        return;
    }
    let name = "no_output_prior_8_vk";
    let Some((stream, reference)) = raw_hevc(name) else {
        return;
    };
    let packets = hevc_units(&stream, name);
    let count = packets.len();
    for (preference, frames) in decode_in_each_mode(&device, DecodeCodec::Hevc, &[], &packets, name)
    {
        assert!(frames.len() < count, "{name}: the fixture drops no frame");
        let label = format!("{name} {preference:?}");
        assert_same_bytes(&planar_bytes(&frames, name), &reference, &label);
    }
    assert_no_validation_errors(&device, name);
}

fn check_stream(stream: &Stream, frames: &[DecodedFrame<WgpuRuntime>]) {
    let name = stream.name;
    let pts: Vec<i64> = frames.iter().map(DecodedFrame::pts).collect();
    assert_eq!(pts, stream.frames, "{name}: output-order pts");
    let format = match stream.meta("format") {
        "nv12" => PixelFormat::Nv12,
        "p010le" | "p010" => PixelFormat::P010,
        other => panic!("{name}: format `{other}`"),
    };
    let (width, height) = (stream.number("width"), stream.number("height"));
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(frame.format(), format, "{name} frame {index}");
        assert_eq!(
            (frame.display().width(), frame.display().height()),
            (width, height),
            "{name} frame {index}: display size"
        );
        assert_eq!(
            frame.color().chroma_location,
            (name == "hevc_chroma_center").then_some(ChromaLocation::Center),
            "{name} frame {index}: chroma location"
        );
    }
    let Some(yuv) = &stream.yuv else {
        panic!("{name}: the fixture has no .yuv");
    };
    assert_same_bytes(&frame_bytes(frames, name), yuv, name);
}

/// Container samples with an `hvcC`: pts, facts, and bytes.
#[test]
fn container_samples_equal_ffmpeg() {
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, DecodeCodec::Hevc) {
        return;
    }
    for name in ["hevc_8bit", "hevc_10bit", "hevc_chroma_center"] {
        let Some(stream) = load(name) else {
            continue;
        };
        let mut decoder = VulkanDecoder::new(&device, DecodeCodec::Hevc, &stream.config)
            .unwrap_or_else(|error| panic!("{name}: open: {error}"));
        assert_eq!(decoder.route(), FrameRoute::Device, "{name}: route");
        let frames = decode_all(&mut decoder, samples(&stream), name);
        check_stream(&stream, &frames);
        assert_no_validation_errors(&device, name);
    }
}

/// A kernel reads a 10-bit frame on numr's queue after the raw copy, and sees what `to_bytes`
/// sees. This fails if the P010 layout is wrong or wgpu zeroes a buffer.
#[test]
fn numr_kernels_read_copied_p010_planes() {
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, DecodeCodec::Hevc) {
        return;
    }
    let name = "hevc_10bit";
    let Some(stream) = load(name) else {
        return;
    };
    let mut decoder = VulkanDecoder::new(&device, DecodeCodec::Hevc, &stream.config)
        .unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let frames = decode_all(&mut decoder, samples(&stream), name);
    let frame = frames.first().unwrap_or_else(|| panic!("{name}: no frame"));
    let spec = frame
        .color()
        .yuv_spec(ChromaSiting::Left)
        .unwrap_or_else(|| panic!("{name}: no colour matrix"));
    let client = device.client();
    let got = to_rgb(client, frame, &spec, DType::F32, ConvLayout::Nchw)
        .unwrap_or_else(|error| panic!("{name}: to_rgb: {error}"));
    let uploaded = |plane: &Tensor<_>| {
        let bytes = plane
            .to_bytes()
            .unwrap_or_else(|error| panic!("{name}: read: {error}"));
        Tensor::from_bytes(&bytes, plane.shape(), DType::U32, device.numr_device())
            .unwrap_or_else(|error| panic!("{name}: upload: {error}"))
    };
    let reference = DecodedFrame::from_packed_planes(
        uploaded(frame.y()),
        uploaded(frame.uv()),
        frame.format(),
        frame.pts(),
        frame.display(),
        *frame.color(),
    );
    let want = to_rgb(client, &reference, &spec, DType::F32, ConvLayout::Nchw)
        .unwrap_or_else(|error| panic!("{name}: to_rgb on the upload: {error}"));
    let read = |tensor: &Tensor<_>| {
        tensor
            .to_bytes()
            .unwrap_or_else(|error| panic!("{name}: read: {error}"))
    };
    assert!(read(&got) == read(&want), "{name}: RGB differs");
    assert_no_validation_errors(&device, name);
}

#[test]
fn a_444_stream_is_refused_at_open() {
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, DecodeCodec::Hevc) {
        return;
    }
    let name = "hevc_444";
    let Some(stream) = load(name) else {
        return;
    };
    let error = VulkanDecoder::new(&device, DecodeCodec::Hevc, &stream.config)
        .err()
        .unwrap_or_else(|| panic!("{name}: opens"));
    assert!(
        matches!(error, DecodeError::UnsupportedLayout { .. }),
        "{name}: {error}"
    );
}

/// A trailing picture first has no reference, and is refused rather than decoded wrong.
#[test]
fn a_stream_that_starts_on_a_trailing_picture_is_refused() {
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, DecodeCodec::Hevc) {
        return;
    }
    let name = "inter_8";
    let Some((stream, _)) = raw_hevc(name) else {
        return;
    };
    let packets = hevc_units(&stream, name);
    let mut decoder = VulkanDecoder::new(&device, DecodeCodec::Hevc, &[])
        .unwrap_or_else(|error| panic!("{name}: open: {error}"));
    // The parameter sets travel with the first picture, so a cut needs them on their own.
    let (first, _) = &packets[0];
    let cut = first
        .windows(5)
        .position(|window| window == [0, 0, 0, 1, 0x26] || window == [0, 0, 0, 1, 0x28])
        .unwrap_or_else(|| panic!("{name}: no IDR slice in the first packet"));
    decoder
        .send(&first[..cut], 0)
        .unwrap_or_else(|error| panic!("{name}: parameter sets: {error}"));
    let (second, pts) = &packets[1];
    let error = decoder.send(second, *pts).expect_err("refused");
    assert!(
        matches!(error, DecodeError::VulkanDecodeRequest { .. }),
        "{name}: {error}"
    );
    assert_no_validation_errors(&device, name);
}
