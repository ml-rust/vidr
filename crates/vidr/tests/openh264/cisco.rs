//! Cisco's binary: the hash check, and a decode when the binary is present.

use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use vidr::decode::openh264::Openh264Decoder;
use vidr::encode::openh264::Openh264Encoder;
use vidr::encode::{EncodeCodec, EncodeConfig, PixelFormat};
use vidr::{DecodeError, EncodeError, Rational};

use super::support::cisco_binary;
use crate::decode_fixtures::cpu::{check_frames, decode_all, samples};
use crate::decode_fixtures::load;

fn config() -> EncodeConfig {
    let rate = Rational::new(30, 1).unwrap_or_else(|error| panic!("rate: {error}"));
    let base = Rational::new(1, 30).unwrap_or_else(|error| panic!("base: {error}"));
    EncodeConfig::new(EncodeCodec::H264, 64, 64, PixelFormat::Nv12, rate, base)
}

/// A real file with a hash off the crate's list never loads.
#[test]
fn a_file_off_the_hash_list_is_refused() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
    let decoder = Openh264Decoder::<CpuRuntime>::with_cisco_binary(&CpuDevice::new(), &[], path);
    assert!(matches!(
        decoder.err(),
        Some(DecodeError::Openh264Library { .. })
    ));
    let encoder = Openh264Encoder::<CpuRuntime>::with_cisco_binary(config(), path);
    assert!(matches!(
        encoder.err(),
        Some(EncodeError::Openh264Library { .. })
    ));
}

#[test]
fn a_missing_file_names_its_path() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/no-such-openh264.so");
    let decoder = Openh264Decoder::<CpuRuntime>::with_cisco_binary(&CpuDevice::new(), &[], path);
    let Some(DecodeError::Openh264Library { path: named, .. }) = decoder.err() else {
        panic!("expected Openh264Library");
    };
    assert_eq!(named, path);
}

/// The binary at `VIDR_OPENH264_CISCO`, every byte against ffmpeg's decode.
#[test]
fn the_binary_decodes_baseline_exactly() {
    let name = "h264_baseline";
    let (Some(path), Some(stream)) = (cisco_binary(), load(name)) else {
        return;
    };
    let mut decoder =
        Openh264Decoder::<CpuRuntime>::with_cisco_binary(&CpuDevice::new(), &stream.config, path)
            .unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let frames = decode_all(&mut decoder, samples(&stream), name);
    check_frames(&stream, &frames, None);
}
