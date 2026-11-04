//! HEVC output against ffmpeg's decode, byte for byte. The tolerance is zero: HEVC
//! decoding is bit-exact by specification.

use std::collections::HashSet;

use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use vidr::bitstream::{HevcConfig, ToAnnexB};
use vidr::decode::hevc::HevcDecoder;
use vidr::decode::{ChromaFormat, ChromaLocation, DecodeCodec, DecodedFrame, Decoder};
use vidr::{DecodeError, FrameRoute};

use super::packets::nal_packets;
use crate::decode_fixtures::cpu::{check_frames, decode_all, samples};
use crate::decode_fixtures::{Stream, load, raw_hevc as raw_fixture};

fn open(config: &[u8], name: &str) -> HevcDecoder<CpuRuntime> {
    HevcDecoder::new(&CpuDevice::new(), config)
        .unwrap_or_else(|error| panic!("{name}: open: {error}"))
}

fn location(stream: &Stream) -> Option<ChromaLocation> {
    (stream.name == "hevc_chroma_center").then_some(ChromaLocation::Center)
}

/// Decodes container samples, and checks pts, facts, and bytes.
fn samples_equal_ffmpeg(name: &'static str) {
    let Some(stream) = load(name) else {
        return;
    };
    let mut decoder = open(&stream.config, name);
    assert_eq!(decoder.route(), FrameRoute::HostOnly, "{name}: route");
    let frames = decode_all(&mut decoder, samples(&stream), name);
    check_frames(&stream, &frames, location(&stream));
}

#[test]
fn hevc_8bit_equals_ffmpeg() {
    samples_equal_ffmpeg("hevc_8bit");
}

#[test]
fn hevc_10bit_equals_ffmpeg() {
    samples_equal_ffmpeg("hevc_10bit");
}

#[test]
fn chroma_location_comes_from_the_sps() {
    samples_equal_ffmpeg("hevc_chroma_center");
}

/// With an empty config the parameter sets come in band, and the facts stay equal.
#[test]
fn annexb_input_equals_ffmpeg() {
    for name in ["hevc_8bit", "hevc_10bit", "hevc_chroma_center"] {
        let Some(stream) = load(name) else {
            continue;
        };
        let config = HevcConfig::parse(&stream.config)
            .unwrap_or_else(|error| panic!("{name}: config: {error}"));
        let mut filter = ToAnnexB::hevc(&config);
        let packets = samples(&stream).into_iter().map(|(sample, pts)| {
            let unit = filter
                .filter(&sample)
                .unwrap_or_else(|error| panic!("{name}: filter: {error}"));
            (unit, pts)
        });
        let mut decoder = open(&[], name);
        let frames = decode_all(&mut decoder, packets, name);
        check_frames(&stream, &frames, location(&stream));
    }
}

/// A stream with no hash SEI message or with matching ones decodes the same.
#[test]
fn the_hash_check_changes_no_output() {
    let name = "hevc_8bit";
    let Some(stream) = load(name) else {
        return;
    };
    let mut decoder =
        HevcDecoder::<CpuRuntime>::with_picture_hash_check(&CpuDevice::new(), &stream.config)
            .unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let frames = decode_all(&mut decoder, samples(&stream), name);
    check_frames(&stream, &frames, location(&stream));
}

#[test]
fn a_444_config_is_refused_at_open() {
    let name = "hevc_444";
    let Some(stream) = load(name) else {
        return;
    };
    let error = HevcDecoder::<CpuRuntime>::new(&CpuDevice::new(), &stream.config)
        .err()
        .unwrap_or_else(|| panic!("{name}: the decoder opens"));
    assert_eq!(
        error,
        DecodeError::UnsupportedLayout {
            codec: DecodeCodec::Hevc,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            chroma: ChromaFormat::Yuv444,
        }
    );
}

/// A length field past the sample's end is the filter's refusal, so decoding goes on.
#[test]
fn a_sample_that_does_not_split_leaves_the_decoder_usable() {
    let name = "hevc_8bit";
    let Some(stream) = load(name) else {
        return;
    };
    let mut decoder = open(&stream.config, name);
    let bad = [0xff, 0xff, 0xff, 0xff, 0x02];
    assert!(matches!(
        decoder.send(&bad, 0),
        Err(DecodeError::Bitstream(_))
    ));
    let frames = decode_all(&mut decoder, samples(&stream), name);
    check_frames(&stream, &frames, location(&stream));
}

/// The planar bytes one frame holds in the fixture's `.yuv`.
fn yuv_bytes(frame: &DecodedFrame<CpuRuntime>) -> usize {
    let (width, height) = (
        frame.display().width() as usize,
        frame.display().height() as usize,
    );
    (width * height + 2 * width.div_ceil(2) * height.div_ceil(2))
        * frame.format().bytes_per_sample()
}

/// Every output picture carries a pts its stream started, once, and flush finds no
/// picture lost. Each name states a case where a picture gives no frame or reorders.
fn pts_follow_the_pictures(name: &str) {
    let Some((stream, reference)) = raw_fixture(name) else {
        return;
    };
    let packets = nal_packets(&stream).unwrap_or_else(|error| panic!("{name}: {error}"));
    let mut decoder = open(&[], name);
    let frames = decode_all(&mut decoder, packets, name);
    assert_eq!(
        frames.iter().map(yuv_bytes).sum::<usize>(),
        reference.len(),
        "{name}: frames"
    );
    let unique: HashSet<i64> = frames.iter().map(DecodedFrame::pts).collect();
    assert_eq!(unique.len(), frames.len(), "{name}: a pts repeats");
    assert!(unique.iter().all(|&pts| pts > 0), "{name}: a pts is unset");
}

#[test]
fn reordered_pictures_keep_their_pts() {
    pts_follow_the_pictures("bframes_8");
    pts_follow_the_pictures("b_pyramid_10");
}

#[test]
fn skipped_rasl_pictures_take_no_pts() {
    pts_follow_the_pictures("rasl_cut_8");
}

#[test]
fn dropped_pictures_leave_no_pending_pts() {
    pts_follow_the_pictures("end_of_sequence_8");
    pts_follow_the_pictures("no_output_prior_8");
}

#[test]
fn each_sequence_of_a_scene_cut_keeps_its_pts() {
    pts_follow_the_pictures("scene_cut_8");
}
