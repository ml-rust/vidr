//! Small H.264 and HEVC sources through `VulkanDecoder`, against ffmpeg's decode, byte for byte.
//!
//! NVIDIA's driver refuses some small H.264 decode sessions though it lists them, so each session
//! opens at a floor on each side. These streams keep that path covered. `gen_nvdec.py` writes them.
//! A size below the driver's minimum prints a notice, because that refusal is correct.

use vidr::DecodeError;
use vidr::decode::DecodeCodec;
use vidr::vulkan::VulkanDecoder;

use super::support::{
    assert_no_validation_errors, assert_same_bytes, codec_present, decode_all, device, frame_bytes,
    samples,
};
use crate::decode_fixtures::load;

fn decodes_where_the_driver_allows(name: &'static str, codec: DecodeCodec) {
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, codec) {
        return;
    }
    let Some(stream) = load(name) else {
        return;
    };
    let mut decoder = match VulkanDecoder::new(&device, codec, &stream.config) {
        Ok(decoder) => decoder,
        Err(DecodeError::VulkanRefused { what, reason, .. }) if what.contains("coded size") => {
            eprintln!("skipping {name}: {what}: {reason}");
            return;
        }
        Err(error) => panic!("{name}: open: {error}"),
    };
    let frames = decode_all(&mut decoder, samples(&stream), name);
    let Some(yuv) = &stream.yuv else {
        panic!("{name}: the fixture has no .yuv");
    };
    assert_same_bytes(&frame_bytes(&frames, name), yuv, name);
    assert_no_validation_errors(&device, name);
}

#[test]
fn h264_at_64x64_equals_ffmpeg() {
    decodes_where_the_driver_allows("h264_small_64", DecodeCodec::H264);
}

#[test]
fn h264_at_128x128_equals_ffmpeg() {
    decodes_where_the_driver_allows("h264_small_128", DecodeCodec::H264);
}

#[test]
fn hevc_at_64x64_equals_ffmpeg_where_the_driver_allows() {
    decodes_where_the_driver_allows("hevc_small_64", DecodeCodec::Hevc);
}

#[test]
fn hevc_at_128x128_equals_ffmpeg_where_the_driver_allows() {
    decodes_where_the_driver_allows("hevc_small_128", DecodeCodec::Hevc);
}

/// `gen_hevc.py` sizes its Vulkan streams for this side, so this one runs on each driver it names.
#[test]
fn hevc_at_144x144_equals_ffmpeg() {
    decodes_where_the_driver_allows("hevc_small_144", DecodeCodec::Hevc);
}
