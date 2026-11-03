//! Unit tests of the sequence callback's decisions.

use std::ffi::c_ulong;

use super::ffi::{
    CHROMA_420, CHROMA_444, CODEC_H264, CODEC_HEVC, CuVideoFormat, CudaVideoCodec, CuvidDecodeCaps,
    DEINTERLACE_WEAVE, IntRect, SURFACE_NV12, SURFACE_P016, VideoSignalDescription,
};
use super::sequence::{OUTPUT_SURFACES, Sequence};
use crate::decode::{ChromaFormat, DecodeCodec, PixelFormat};
use crate::error::DecodeError;
use crate::geometry::Dimensions;

fn format(codec: CudaVideoCodec, depth_minus8: u8, area: IntRect) -> CuVideoFormat {
    CuVideoFormat {
        codec,
        bit_depth_luma_minus8: depth_minus8,
        bit_depth_chroma_minus8: depth_minus8,
        min_num_decode_surfaces: 6,
        coded_width: 320,
        coded_height: 192,
        display_area: area,
        chroma_format: CHROMA_420,
        video_signal_description: VideoSignalDescription {
            flags: 0,
            color_primaries: 1,
            transfer_characteristics: 1,
            matrix_coefficients: 1,
        },
        ..CuVideoFormat::default()
    }
}

fn area(right: i32, bottom: i32) -> IntRect {
    IntRect {
        left: 0,
        top: 0,
        right,
        bottom,
    }
}

fn read(format: &CuVideoFormat) -> Result<Sequence, DecodeError> {
    Sequence::read(DecodeCodec::H264, format, None)
}

#[test]
fn h264_crop_keeps_the_display_size() {
    let sequence = read(&format(CODEC_H264, 0, area(320, 180)));
    let Ok(sequence) = sequence else {
        panic!("read failed: {sequence:?}");
    };
    assert_eq!(sequence.format, PixelFormat::Nv12);
    assert_eq!(sequence.target(), (320, 180));
    assert_eq!(
        sequence.display,
        Dimensions::new(320, 180).unwrap_or_else(|e| panic!("{e}"))
    );
    assert_eq!(sequence.decode_surfaces, 6);
    assert!(!sequence.color.full_range);
    assert_eq!(sequence.color.matrix_coefficients, 1);
}

#[test]
fn odd_display_sides_round_only_the_surface() {
    let Ok(sequence) = read(&format(CODEC_H264, 0, area(319, 179))) else {
        panic!("read failed");
    };
    assert_eq!(sequence.target(), (320, 180));
    assert_eq!(sequence.display.width(), 319);
    assert_eq!(sequence.display.height(), 179);
}

#[test]
fn odd_side_at_the_coded_edge_keeps_the_coded_size() {
    let mut header = format(CODEC_H264, 0, area(320, 192));
    header.coded_width = 321;
    header.display_area = area(321, 192);
    let Ok(sequence) = read(&header) else {
        panic!("read failed");
    };
    assert_eq!(sequence.coded, (321, 192));
    assert_eq!(sequence.target(), (322, 192));
    assert_eq!(sequence.display.width(), 321);
    let info = sequence.create_info(std::ptr::null_mut());
    assert_eq!(
        (info.display_area.right, info.display_area.bottom),
        (322, 192)
    );
    assert_eq!(info.width, 321);
}

#[test]
fn a_rectangle_past_an_odd_coded_edge_is_an_error() {
    let mut header = format(CODEC_H264, 0, area(320, 192));
    header.coded_width = 321;
    header.display_area = area(322, 192);
    assert!(matches!(
        read(&header),
        Err(DecodeError::DisplayArea { .. })
    ));
}

#[test]
fn rectangles_outside_the_frame_are_errors() {
    assert!(matches!(
        read(&format(CODEC_H264, 0, area(330, 180))),
        Err(DecodeError::DisplayArea { .. })
    ));
    assert!(matches!(
        read(&format(CODEC_H264, 0, area(0, 180))),
        Err(DecodeError::DisplayArea { .. })
    ));
}

#[test]
fn ten_bit_is_p010_and_twelve_bit_is_refused() {
    let Ok(sequence) = Sequence::read(
        DecodeCodec::Hevc,
        &format(CODEC_HEVC, 2, area(320, 180)),
        None,
    ) else {
        panic!("read failed");
    };
    assert_eq!(sequence.format, PixelFormat::P010);
    assert!(matches!(
        Sequence::read(
            DecodeCodec::Hevc,
            &format(CODEC_HEVC, 4, area(320, 180)),
            None
        ),
        Err(DecodeError::UnsupportedLayout {
            bit_depth_luma: 12,
            ..
        })
    ));
}

#[test]
fn non_420_and_mixed_depths_are_refused() {
    let mut header = format(CODEC_HEVC, 0, area(320, 180));
    header.chroma_format = CHROMA_444;
    assert!(matches!(
        Sequence::read(DecodeCodec::Hevc, &header, None),
        Err(DecodeError::UnsupportedLayout {
            chroma: ChromaFormat::Yuv444,
            ..
        })
    ));
    let mut header = format(CODEC_HEVC, 2, area(320, 180));
    header.bit_depth_chroma_minus8 = 0;
    assert!(matches!(
        Sequence::read(DecodeCodec::Hevc, &header, None),
        Err(DecodeError::UnsupportedLayout { .. })
    ));
}

#[test]
fn another_codec_is_a_mismatch() {
    assert!(matches!(
        read(&format(CODEC_HEVC, 0, area(320, 180))),
        Err(DecodeError::CodecMismatch {
            expected: DecodeCodec::H264,
            found: CODEC_HEVC
        })
    ));
}

#[test]
fn decoder_is_kept_until_the_size_or_depth_changes() {
    let Ok(first) = read(&format(CODEC_H264, 0, area(320, 180))) else {
        panic!("read failed");
    };
    let mut recolored = first;
    recolored.color.full_range = true;
    assert!(first.same_decoder(&recolored));
    let Ok(resized) = read(&format(CODEC_H264, 0, area(320, 176))) else {
        panic!("read failed");
    };
    assert!(!first.same_decoder(&resized));
    let mut deeper = first;
    deeper.decode_surfaces += 1;
    assert!(!first.same_decoder(&deeper));
}

fn caps(supported: u8, mask: u16) -> CuvidDecodeCaps {
    CuvidDecodeCaps {
        is_supported: supported,
        output_format_mask: mask,
        max_width: 4096,
        max_height: 4096,
        max_mb_count: 65536,
        min_width: 48,
        min_height: 16,
        ..CuvidDecodeCaps::default()
    }
}

#[test]
fn caps_refusals_name_the_reason() {
    let Ok(sequence) = read(&format(CODEC_H264, 0, area(320, 180))) else {
        panic!("read failed");
    };
    assert_eq!(sequence.check_caps(&caps(1, 1 << SURFACE_NV12)), Ok(()));
    let refused = sequence.check_caps(&caps(0, 1));
    assert!(matches!(
        &refused,
        Err(DecodeError::Unsupported { reason, bit_depth: 8, .. })
            if reason.contains("bIsSupported 0")
    ));
    let no_surface = sequence.check_caps(&caps(1, 1 << SURFACE_P016));
    assert!(matches!(
        &no_surface,
        Err(DecodeError::Unsupported { reason, .. }) if reason.contains("NV12")
    ));
    let mut small = caps(1, 1);
    small.max_width = 256;
    assert!(matches!(
        sequence.check_caps(&small),
        Err(DecodeError::CodedSize { width: 320, .. })
    ));
}

#[test]
fn create_info_crops_without_scaling() {
    let Ok(sequence) = read(&format(CODEC_H264, 0, area(319, 179))) else {
        panic!("read failed");
    };
    let info = sequence.create_info(std::ptr::null_mut());
    assert_eq!(info.width, 320);
    assert_eq!(info.height, 192);
    assert_eq!(info.target_width, 320);
    assert_eq!(info.target_height, 180);
    assert_eq!(info.display_area.right, 320);
    assert_eq!(info.display_area.bottom, 180);
    assert_eq!(info.output_format, SURFACE_NV12);
    assert_eq!(info.deinterlace_mode, DEINTERLACE_WEAVE);
    assert_eq!(info.num_output_surfaces, c_ulong::from(OUTPUT_SURFACES));
    let query = sequence.caps_query();
    assert_eq!(query.codec_type, CODEC_H264);
    assert_eq!(query.bit_depth_minus8, 0);
}
