//! `avcC` parse and write on hand-built records.

use vidr::BitstreamError;
use vidr::bitstream::{AvcConfig, AvcHighProfile, LengthSize};

/// A Main profile SPS: profile 77, constraints 0x40, level 30, id 0.
const MAIN_SPS: [u8; 5] = [0x67, 77, 0x40, 30, 0x80];
/// A High profile SPS: profile 100, level 31, id 0, 4:2:0, 8 bits.
const HIGH_SPS: [u8; 6] = [0x67, 100, 0, 31, 0b1010_1100, 0x80];
const PPS: [u8; 3] = [0x68, 0xce, 0x38];

fn main_record() -> Vec<u8> {
    let mut bytes = vec![1, 77, 0x40, 30, 0xff, 0xe1, 0, 5];
    bytes.extend_from_slice(&MAIN_SPS);
    bytes.extend_from_slice(&[1, 0, 3]);
    bytes.extend_from_slice(&PPS);
    bytes
}

fn field_of<T: std::fmt::Debug>(result: Result<T, BitstreamError>) -> &'static str {
    match result {
        Err(BitstreamError::InvalidField { field, .. }) => field,
        other => panic!("expected an invalid field, got {other:?}"),
    }
}

#[test]
fn legacy_high_profile_record_without_extension_round_trips() {
    let mut bytes = vec![1, 100, 0, 31, 0xff, 0xe1, 0, 6];
    bytes.extend_from_slice(&HIGH_SPS);
    bytes.extend_from_slice(&[1, 0, 3]);
    bytes.extend_from_slice(&PPS);
    let config = AvcConfig::parse(&bytes).unwrap();
    assert!(config.high_profile.is_none());
    assert_eq!(config.to_bytes().unwrap(), bytes);
}

#[test]
fn parse_errors_name_the_field() {
    let mut bytes = main_record();
    bytes[0] = 0;
    assert_eq!(field_of(AvcConfig::parse(&bytes)), "configurationVersion");

    let mut bytes = main_record();
    bytes[4] = 0xfe;
    assert_eq!(field_of(AvcConfig::parse(&bytes)), "lengthSizeMinusOne");

    let mut bytes = main_record();
    bytes[8] = 0x68;
    assert_eq!(field_of(AvcConfig::parse(&bytes)), "nal_unit_type");

    let mut bytes = main_record();
    bytes[7] = 0;
    assert_eq!(
        field_of(AvcConfig::parse(&bytes)),
        "sequenceParameterSetLength"
    );

    let mut bytes = main_record();
    bytes.push(0);
    assert!(matches!(
        AvcConfig::parse(&bytes),
        Err(BitstreamError::TrailingBytes { .. })
    ));
}

#[test]
fn every_truncation_is_an_error() {
    let bytes = main_record();
    for end in 0..bytes.len() {
        assert!(
            AvcConfig::parse(&bytes[..end]).is_err(),
            "prefix of {end} bytes"
        );
    }
}

#[test]
fn write_refuses_what_the_record_cannot_hold() {
    let mut config = AvcConfig::parse(&main_record()).unwrap();
    config.sps = vec![MAIN_SPS.to_vec(); 32];
    assert_eq!(field_of(config.to_bytes()), "numOfSequenceParameterSets");
    config.sps = vec![PPS.to_vec()];
    assert_eq!(field_of(config.to_bytes()), "nal_unit_type");
    config.sps = vec![MAIN_SPS.to_vec()];
    config.high_profile = Some(AvcHighProfile {
        chroma_format: 1,
        bit_depth_luma_minus8: 0,
        bit_depth_chroma_minus8: 0,
        sps_ext: Vec::new(),
    });
    assert_eq!(field_of(config.to_bytes()), "AVCProfileIndication");
}

#[test]
fn building_needs_sps_and_pps() {
    let build = |sps: &[&[u8]], pps: &[&[u8]], ext: &[&[u8]]| {
        AvcConfig::from_parameter_sets(sps, pps, ext, LengthSize::Four)
    };
    assert_eq!(
        field_of(build(&[], &[&PPS], &[])),
        "numOfSequenceParameterSets"
    );
    assert_eq!(
        field_of(build(&[&MAIN_SPS], &[], &[])),
        "numOfPictureParameterSets"
    );
    assert_eq!(
        field_of(build(&[&MAIN_SPS], &[&PPS], &[&[0x6d, 1]])),
        "numOfSequenceParameterSetExt"
    );
}

#[test]
fn real_x264_parameter_sets_build_the_ffmpeg_record() {
    // From an x264 128x96 High profile MP4 that ffmpeg wrote.
    let sps = [
        0x67, 0x64, 0x00, 0x0a, 0xac, 0xd9, 0x42, 0x0d, 0xb0, 0x11, 0x00, 0x00, 0x03, 0x00, 0x01,
        0x00, 0x00, 0x03, 0x00, 0x32, 0x0f, 0x12, 0x25, 0x96,
    ];
    let pps = [0x68, 0xeb, 0xe3, 0xcb, 0x22, 0xc0];
    let mut expected = vec![0x01, 0x64, 0x00, 0x0a, 0xff, 0xe1, 0x00, 0x18];
    expected.extend_from_slice(&sps);
    expected.extend_from_slice(&[0x01, 0x00, 0x06]);
    expected.extend_from_slice(&pps);
    expected.extend_from_slice(&[0xfd, 0xf8, 0xf8, 0x00]);
    let built = AvcConfig::from_parameter_sets(&[&sps], &[&pps], &[], LengthSize::Four).unwrap();
    assert_eq!(built.to_bytes().unwrap(), expected);
}
