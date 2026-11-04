//! VP9 against ffmpeg's decode, byte for byte. VP9 decoding is bit-exact by specification, so
//! a mismatch is a defect, never a tolerance to widen.
//!
//! ffmpeg splits a superframe and returns each shown frame, and libvpx returns the last one.
//! `vp9_two_shown` holds what libvpx returns.

use numr::runtime::wgpu::WgpuRuntime;
use vidr::DecodeError;
use vidr::FrameRoute;
use vidr::decode::{DecodeCodec, DecodedFrame, Decoder, PixelFormat, PlaneForm};
use vidr::vulkan::VulkanDecoder;

use super::support::{
    assert_no_validation_errors, assert_same_bytes, codec_present, decode_all, decode_in_each_mode,
    device, frame_bytes, samples,
};
use crate::decode_fixtures::{Stream, load};

/// A one-byte `show_existing_frame` header of profile 0: frame marker 2, profile 0, show
/// existing 1, slot 0.
const SHOW_SLOT_0: u8 = 0x88;

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
    if !codec_present(&device, DecodeCodec::Vp9) {
        return;
    }
    let Some(stream) = load(name) else {
        return;
    };
    let packets = samples(&stream);
    for (preference, frames) in decode_in_each_mode(&device, DecodeCodec::Vp9, &[], &packets, name)
    {
        check_stream(&stream, &frames, &format!("{name} {preference:?}"));
    }
    assert_no_validation_errors(&device, name);
}

#[test]
fn profile_0_equals_ffmpeg() {
    decodes_exactly("vp9_8bit");
}

#[test]
fn profile_2_equals_ffmpeg() {
    decodes_exactly("vp9_10bit");
}

#[test]
fn an_odd_size_equals_ffmpeg() {
    decodes_exactly("vp9_odd");
}

#[test]
fn a_full_range_header_equals_ffmpeg() {
    decodes_exactly("vp9_bt470bg_pc");
}

/// Superframes hold hidden alt-ref frames. Each hidden frame fills a slot with no copy, and
/// each packet gives one frame, at its pts.
#[test]
fn hidden_frames_in_superframes_equal_ffmpeg() {
    let name = "vp9_altref";
    decodes_exactly(name);
    let Some(stream) = load(name) else {
        return;
    };
    let superframes = stream
        .samples
        .iter()
        .filter(|sample| sample.last().is_some_and(|byte| byte & 0xe0 == 0xc0))
        .count();
    assert!(superframes > 0, "{name}: the fixture holds no superframe");
    assert_eq!(stream.frames.len(), stream.samples.len(), "{name}");
}

/// Each sample is a superframe of two shown frames. The last one of a chunk leaves, so the
/// frame count stays one per packet, as libvpx gives.
#[test]
fn each_superframe_gives_its_last_shown_frame() {
    decodes_exactly("vp9_two_shown");
}

/// Spatial resize changes the size at frames that keep references, so each frame reads
/// references of another size. A session sized to the first frame takes a stream that only
/// shrinks and comes back.
#[test]
fn a_stream_that_resizes_equals_ffmpeg() {
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, DecodeCodec::Vp9) {
        return;
    }
    let name = "vp9_resize";
    let Some(stream) = load(name) else {
        return;
    };
    let mut decoder = VulkanDecoder::with_max_coded_size(
        &device,
        DecodeCodec::Vp9,
        &[],
        (stream.number("width"), stream.number("height")),
    )
    .unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let frames = decode_all(&mut decoder, samples(&stream), name);
    let mut sizes: Vec<(u32, u32)> = frames
        .iter()
        .map(|frame| (frame.display().width(), frame.display().height()))
        .collect();
    sizes.dedup();
    assert!(
        sizes.len() >= 2,
        "{name}: the fixture never resizes: {sizes:?}. Regenerate it with a lower rate."
    );
    check_stream(&stream, &frames, name);
    assert_no_validation_errors(&device, name);
}

/// A larger key frame stays byte-exact. Inside the session floor it keeps its session, and past that
/// floor it opens a new one.
#[test]
fn a_larger_key_frame_stays_exact_in_either_session() {
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, DecodeCodec::Vp9) {
        return;
    }
    let (Some(first), Some(second)) = (load("vp9_altref"), load("vp9_resize")) else {
        return;
    };
    assert!(
        first.number("width") < second.number("width"),
        "the second stream grows the frame"
    );
    let name = "vp9_altref then vp9_resize";
    let last = first.pts.iter().copied().max().unwrap_or(0);
    let packets = samples(&first).into_iter().chain(
        samples(&second)
            .into_iter()
            .map(|(sample, pts)| (sample, pts + last + 1)),
    );
    let mut decoder = VulkanDecoder::new(&device, DecodeCodec::Vp9, &[])
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

/// A key frame fills every slot, so showing slot 0 repeats it from the slot. The frame stays
/// a reference, so the next frame still decodes, and a second show repeats the same bytes.
#[test]
fn show_existing_frame_copies_the_slot_again() {
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, DecodeCodec::Vp9) {
        return;
    }
    let name = "vp9_8bit";
    let Some(stream) = load(name) else {
        return;
    };
    let Some(key) = stream.samples.first() else {
        panic!("{name}: no samples");
    };
    let show = [SHOW_SLOT_0];
    let packets = vec![(key.clone(), 0), (show.to_vec(), 1), (show.to_vec(), 2)];
    for (preference, frames) in decode_in_each_mode(&device, DecodeCodec::Vp9, &[], &packets, name)
    {
        let label = format!("{name} {preference:?}");
        assert_eq!(frames.len(), 3, "{label}: frames");
        let key_bytes = frame_bytes(&frames[..1], &label);
        assert_same_bytes(&frame_bytes(&frames[1..2], &label), &key_bytes, &label);
        assert_same_bytes(&frame_bytes(&frames[2..3], &label), &key_bytes, &label);
        let pts: Vec<i64> = frames.iter().map(DecodedFrame::pts).collect();
        assert_eq!(pts, [0, 1, 2], "{label}: pts");
        assert_eq!(frames[1].color(), frames[0].color(), "{label}: colour");
    }
    assert_no_validation_errors(&device, name);
}

#[test]
fn other_profiles_and_depths_are_refused_by_name() {
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, DecodeCodec::Vp9) {
        return;
    }
    for name in ["vp9_12bit", "vp9_444", "vp9_422_10bit"] {
        let Some(stream) = load(name) else {
            continue;
        };
        let mut decoder = VulkanDecoder::new(&device, DecodeCodec::Vp9, &[])
            .unwrap_or_else(|error| panic!("{name}: open: {error}"));
        assert_eq!(decoder.route(), FrameRoute::Device, "{name}: route");
        let (packet, pts) = &samples(&stream)[0];
        let error = decoder.send(packet, *pts).expect_err("refused");
        assert!(
            matches!(error, DecodeError::Vp9Layout { .. }),
            "{name}: {error}"
        );
        assert_no_validation_errors(&device, name);
    }
}

#[test]
fn an_inter_frame_first_is_refused() {
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, DecodeCodec::Vp9) {
        return;
    }
    let name = "vp9_8bit";
    let Some(stream) = load(name) else {
        return;
    };
    let mut decoder = VulkanDecoder::new(&device, DecodeCodec::Vp9, &[])
        .unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let (packet, pts) = &samples(&stream)[1];
    assert!(decoder.send(packet, *pts).is_err(), "{name}: refused");
    assert_no_validation_errors(&device, name);
}
