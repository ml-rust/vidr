//! Each stream against ffmpeg's decode, byte for byte, in output order.
//!
//! H.264 decoding is bit-exact by specification, so a mismatch is a defect, never a
//! tolerance to widen.

use numr::dtype::DType;
use numr::ops::{ChromaSiting, ConvLayout};
use numr::tensor::Tensor;
use vidr::DecodeError;
use vidr::FrameRoute;
use vidr::decode::{ChromaLocation, DecodeCodec, DecodedFrame, Decoder, PixelFormat, PlaneForm};
use vidr::frame::to_rgb;
use vidr::vulkan::{DpbPreference, VulkanDecoder};

use super::support::{
    annexb_units, assert_no_validation_errors, assert_same_bytes, decode_all, device, nv12_bytes,
    samples,
};
use crate::decode_fixtures::{Stream, load};

/// Baseline, Main, High, B-frames, a B pyramid, reference marking commands, several
/// slices per picture, scaling lists, a VUI chroma location, and CAVLC with POC type 2.
const STREAMS: [&str; 8] = [
    "h264_baseline",
    "h264_main",
    "h264_8bit",
    "h264_bpyramid",
    "h264_mmco",
    "h264_slices",
    "h264_cqm",
    "h264_chroma_center",
];

fn check_stream(stream: &Stream, frames: &[DecodedFrame<numr::runtime::wgpu::WgpuRuntime>]) {
    let name = stream.name;
    let pts: Vec<i64> = frames.iter().map(DecodedFrame::pts).collect();
    assert_eq!(pts, stream.frames, "{name}: display-order pts");
    let (width, height) = (stream.number("width"), stream.number("height"));
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(frame.format(), PixelFormat::Nv12, "{name} frame {index}");
        assert_eq!(frame.plane_form(), PlaneForm::Words, "{name} frame {index}");
        assert_eq!(
            (frame.display().width(), frame.display().height()),
            (width, height),
            "{name} frame {index}: display size"
        );
        assert!(frame.progressive(), "{name} frame {index}: progressive");
        assert_eq!(
            frame.color().chroma_location,
            stream
                .meta_opt("chroma_location")
                .and_then(|value| value.parse().ok())
                .and_then(ChromaLocation::from_h273),
            "{name} frame {index}: chroma location"
        );
    }
    let Some(yuv) = &stream.yuv else {
        panic!("{name}: the fixture has no .yuv");
    };
    assert_same_bytes(&nv12_bytes(frames, name), yuv, name);
}

/// Container samples in each DPB mode the driver lists.
#[test]
fn every_stream_equals_ffmpeg_in_each_dpb_mode() {
    let Some((device, _guard)) = device() else {
        return;
    };
    for name in STREAMS {
        let Some(stream) = load(name) else {
            continue;
        };
        let mut ran = 0;
        for preference in [DpbPreference::Coincide, DpbPreference::Distinct] {
            let opened = VulkanDecoder::with_dpb_preference(
                &device,
                DecodeCodec::H264,
                &stream.config,
                preference,
            );
            let mut decoder = match opened {
                Ok(decoder) => decoder,
                Err(DecodeError::VulkanRefused { reason, .. }) if reason.contains("mode") => {
                    eprintln!("notice: {name}: {preference:?} does not run here: {reason}");
                    continue;
                }
                Err(error) => panic!("{name}: open: {error}"),
            };
            assert_eq!(decoder.route(), FrameRoute::Device, "{name}: route");
            let frames = decode_all(&mut decoder, samples(&stream), name);
            check_stream(&stream, &frames);
            ran += 1;
        }
        assert!(ran > 0, "{name}: no DPB mode ran");
        assert_no_validation_errors(&device, name);
    }
}

/// Annex-B access units with no config record take their parameter sets in band.
#[test]
fn annexb_input_equals_ffmpeg() {
    let Some((device, _guard)) = device() else {
        return;
    };
    for name in ["h264_8bit", "h264_baseline"] {
        let Some(stream) = load(name) else {
            continue;
        };
        let mut decoder = VulkanDecoder::new(&device, DecodeCodec::H264, &[])
            .unwrap_or_else(|error| panic!("{name}: open: {error}"));
        let frames = decode_all(&mut decoder, annexb_units(&stream), name);
        check_stream(&stream, &frames);
        assert_no_validation_errors(&device, name);
    }
}

/// Two streams at an IDR picture, as Annex-B access units, with the second's pts after
/// the first's. Returns the decoded frames.
fn decode_joined(
    device: &std::sync::Arc<vidr::vulkan::VulkanDevice>,
    first: &Stream,
    second: &Stream,
    name: &str,
) -> (Vec<DecodedFrame<numr::runtime::wgpu::WgpuRuntime>>, i64) {
    let last = first.pts.iter().copied().max().unwrap_or(0);
    let packets = annexb_units(first).into_iter().chain(
        annexb_units(second)
            .into_iter()
            .map(|(unit, pts)| (unit, pts + last + 1)),
    );
    let mut decoder = VulkanDecoder::new(device, DecodeCodec::H264, &[])
        .unwrap_or_else(|error| panic!("{name}: open: {error}"));
    (decode_all(&mut decoder, packets, name), last + 1)
}

/// High, then Main at an IDR picture, with one size and one `max_dec_frame_buffering`.
/// The SPS changes the profile, so a new session opens, every frame of the first sequence
/// leaves first, and both stay byte-exact.
#[test]
fn a_parameter_set_change_at_idr_opens_a_new_session() {
    let Some((device, _guard)) = device() else {
        return;
    };
    let (Some(first), Some(second)) = (load("h264_8bit"), load("h264_main")) else {
        return;
    };
    let name = "h264_8bit then h264_main";
    let (frames, shift) = decode_joined(&device, &first, &second, name);

    let mut pts = first.frames.clone();
    pts.extend(second.frames.iter().map(|pts| pts + shift));
    let decoded: Vec<i64> = frames.iter().map(DecodedFrame::pts).collect();
    assert_eq!(decoded, pts, "{name}: display-order pts");
    let (Some(first_yuv), Some(second_yuv)) = (&first.yuv, &second.yuv) else {
        panic!("{name}: a fixture has no .yuv");
    };
    let mut yuv = first_yuv.clone();
    yuv.extend_from_slice(second_yuv);
    assert_same_bytes(&nv12_bytes(&frames, name), &yuv, name);
    assert_no_validation_errors(&device, name);
}

/// High, then Baseline at an IDR picture: `max_dec_frame_buffering` goes from 4 to 3. The
/// flag stays 0, so every waiting frame leaves first.
#[test]
fn a_new_buffer_size_at_idr_keeps_each_waiting_frame() {
    let Some((device, _guard)) = device() else {
        return;
    };
    let (Some(first), Some(second)) = (load("h264_8bit"), load("h264_baseline")) else {
        return;
    };
    let name = "h264_8bit then h264_baseline";
    let (frames, shift) = decode_joined(&device, &first, &second, name);

    let mut pts = first.frames.clone();
    pts.extend(second.frames.iter().map(|pts| pts + shift));
    let decoded: Vec<i64> = frames.iter().map(DecodedFrame::pts).collect();
    assert_eq!(decoded, pts, "{name}: display-order pts");
    let (Some(first_yuv), Some(second_yuv)) = (&first.yuv, &second.yuv) else {
        panic!("{name}: a fixture has no .yuv");
    };
    let mut yuv = first_yuv.clone();
    yuv.extend_from_slice(second_yuv);
    assert_same_bytes(&nv12_bytes(&frames, name), &yuv, name);
    assert_no_validation_errors(&device, name);
}

/// `no_output_of_prior_pics_flag` 1 on every IDR picture but the first. The fixture holds
/// ffmpeg's bytes for each frame C.4.4 keeps, in each DPB mode the driver lists.
#[test]
fn no_output_of_prior_pics_drops_what_waits() {
    let Some((device, _guard)) = device() else {
        return;
    };
    let name = "h264_no_output_prior";
    let Some(stream) = load(name) else {
        return;
    };
    assert!(
        stream.frames.len() < stream.samples.len(),
        "{name}: the fixture drops no frame"
    );
    let mut ran = 0;
    for preference in [DpbPreference::Coincide, DpbPreference::Distinct] {
        let opened = VulkanDecoder::with_dpb_preference(
            &device,
            DecodeCodec::H264,
            &stream.config,
            preference,
        );
        let mut decoder = match opened {
            Ok(decoder) => decoder,
            Err(DecodeError::VulkanRefused { reason, .. }) if reason.contains("mode") => {
                eprintln!("notice: {name}: {preference:?} does not run here: {reason}");
                continue;
            }
            Err(error) => panic!("{name}: open: {error}"),
        };
        let frames = decode_all(&mut decoder, samples(&stream), name);
        check_stream(&stream, &frames);
        ran += 1;
    }
    assert!(ran > 0, "{name}: no DPB mode ran");
    assert_no_validation_errors(&device, name);
}

/// A kernel reads a frame on numr's queue after the raw copy, and sees what `to_bytes`
/// sees. This fails if wgpu zeroes a buffer it does not know holds data.
#[test]
fn numr_kernels_read_the_copied_planes() {
    let Some((device, _guard)) = device() else {
        return;
    };
    let name = "h264_8bit";
    let Some(stream) = load(name) else {
        return;
    };
    let mut decoder = VulkanDecoder::new(&device, DecodeCodec::H264, &stream.config)
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
