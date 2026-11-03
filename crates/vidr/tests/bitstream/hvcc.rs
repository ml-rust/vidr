//! `hvcC` parse, write, and build on a real x265 record.

use vidr::BitstreamError;
use vidr::bitstream::{HevcConfig, LengthSize};

/// The VPS, SPS, and PPS of an x265 128x96 Main profile stream at level 1.
const VPS: [u8; 24] = [
    0x40, 0x01, 0x0c, 0x01, 0xff, 0xff, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03,
    0x00, 0x00, 0x03, 0x00, 0x1e, 0x95, 0x90, 0x09,
];
const SPS: [u8; 41] = [
    0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03,
    0x00, 0x1e, 0xa0, 0x10, 0x20, 0x61, 0x65, 0x95, 0x92, 0x49, 0x32, 0xbc, 0x05, 0xa0, 0x20, 0x00,
    0x00, 0x03, 0x00, 0x20, 0x00, 0x00, 0x03, 0x03, 0x21,
];
const PPS: [u8; 7] = [0x44, 0x01, 0xc1, 0x72, 0xb4, 0x22, 0x40];

/// The `hvcC` that ffmpeg writes for that stream with `-tag:v hvc1`.
fn x265_record() -> Vec<u8> {
    let mut bytes = vec![
        0x01, 0x01, 0x60, 0x00, 0x00, 0x00, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1e, 0xf0, 0x00,
        0xfc, 0xfd, 0xf8, 0xf8, 0x00, 0x00, 0x0f, 0x03,
    ];
    for (kind, unit) in [(0xa0, &VPS[..]), (0xa1, &SPS[..]), (0xa2, &PPS[..])] {
        bytes.extend_from_slice(&[kind, 0, 1, 0, unit.len() as u8]);
        bytes.extend_from_slice(unit);
    }
    bytes
}

fn field_of<T: std::fmt::Debug>(result: Result<T, BitstreamError>) -> &'static str {
    match result {
        Err(BitstreamError::InvalidField { field, .. }) => field,
        other => panic!("expected an invalid field, got {other:?}"),
    }
}

#[test]
fn built_record_matches_ffmpeg() {
    let built =
        HevcConfig::from_parameter_sets(&[&VPS], &[&SPS], &[&PPS], LengthSize::Four).unwrap();
    assert_eq!(built.to_bytes().unwrap(), x265_record());
    assert_eq!(HevcConfig::parse(&x265_record()).unwrap(), built);
}

#[test]
fn parse_errors_name_the_field() {
    let mut bytes = x265_record();
    bytes[21] = 0x0e;
    assert_eq!(field_of(HevcConfig::parse(&bytes)), "lengthSizeMinusOne");

    let mut bytes = x265_record();
    bytes[23] = 0x81;
    assert_eq!(field_of(HevcConfig::parse(&bytes)), "NAL_unit_type");

    let mut bytes = x265_record();
    bytes[23] = 0xa1;
    assert_eq!(field_of(HevcConfig::parse(&bytes)), "nal_unit_type");

    let mut bytes = x265_record();
    bytes[0] = 0;
    assert_eq!(field_of(HevcConfig::parse(&bytes)), "configurationVersion");

    let mut bytes = x265_record();
    bytes.push(0);
    assert!(matches!(
        HevcConfig::parse(&bytes),
        Err(BitstreamError::TrailingBytes { .. })
    ));
}

#[test]
fn every_truncation_is_an_error() {
    let bytes = x265_record();
    for end in 0..bytes.len() {
        assert!(
            HevcConfig::parse(&bytes[..end]).is_err(),
            "prefix of {end} bytes"
        );
    }
}

#[test]
fn write_refuses_out_of_range_fields() {
    let mut config = HevcConfig::parse(&x265_record()).unwrap();
    config.general_profile_idc = 32;
    assert_eq!(field_of(config.to_bytes()), "general_profile_idc");
    config.general_profile_idc = 1;
    config.arrays[0].nal_units.push(vec![0x40]);
    assert_eq!(field_of(config.to_bytes()), "nalUnitLength");
}

#[test]
fn building_needs_every_parameter_set_kind() {
    let build = |vps: &[&[u8]], sps: &[&[u8]], pps: &[&[u8]]| {
        HevcConfig::from_parameter_sets(vps, sps, pps, LengthSize::Four)
    };
    assert_eq!(field_of(build(&[], &[&SPS], &[&PPS])), "VPS numNalus");
    assert_eq!(field_of(build(&[&VPS], &[], &[&PPS])), "SPS numNalus");
    assert_eq!(field_of(build(&[&VPS], &[&SPS], &[])), "PPS numNalus");
    // An SPS in the VPS slot.
    assert_eq!(field_of(build(&[&SPS], &[&SPS], &[&PPS])), "nal_unit_type");
}
