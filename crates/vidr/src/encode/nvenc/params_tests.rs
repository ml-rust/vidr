//! Config mapping tests of `params`.

use super::ffi::{
    AV1_CHROMA_FORMAT_SHIFT, AV1_FLAG_REPEAT_SEQ_HDR, BFRAME_REF_DISABLED,
    BUFFER_FORMAT_YUV420_10BIT, CODEC_H264, CODEC_HEVC, H264_FLAG_OUTPUT_AUD,
    H264_FLAG_REPEAT_SPSPPS, HEVC_CHROMA_FORMAT_SHIFT, INITIALIZE_PARAMS_VER, NvEncConfig, PRESETS,
    PROFILE_AV1_MAIN, PROFILE_H264_HIGH, PROFILE_HEVC_MAIN10, RC_CONSTQP, RC_FLAG_LOOKAHEAD,
    RC_VBR,
};
use super::params::{apply, codec_names, initialize_params};
use crate::decode::ChromaLocation;
use crate::decode::{ColorFacts, PixelFormat};
use crate::encode::{EncodeCodec, EncodeConfig, RateControl};
use crate::time::Rational;

fn config(codec: EncodeCodec, format: PixelFormat) -> EncodeConfig {
    let rate = Rational::new(30000, 1001).unwrap_or(Rational::ONE);
    let base = Rational::new(1, 30000).unwrap_or(Rational::ONE);
    let mut config = EncodeConfig::new(codec, 640, 360, format, rate, base);
    config.rate_control = RateControl::VbrQuality { cq: 20 };
    config.color = ColorFacts {
        matrix_coefficients: 9,
        transfer_characteristics: 16,
        colour_primaries: 9,
        full_range: true,
        chroma_location: Some(ChromaLocation::TopLeft),
    };
    config
}

#[test]
fn h264_maps_profile_rate_control_and_vui() {
    let mut cfg = NvEncConfig::zeroed();
    // The preset asks for B-frame references and repeated headers.
    // SAFETY: `NvEncConfig::zeroed` makes every union member all zeros.
    let mut preset = unsafe { cfg.codec.h264 };
    preset.use_b_frames_as_ref = 2;
    preset.flags = H264_FLAG_REPEAT_SPSPPS | H264_FLAG_OUTPUT_AUD;
    cfg.codec.h264 = preset;
    cfg.rc_params.flags = RC_FLAG_LOOKAHEAD;
    cfg.rc_params.max_bit_rate = 99;

    apply(&config(EncodeCodec::H264, PixelFormat::Nv12), &mut cfg);

    assert_eq!(cfg.profile_guid, PROFILE_H264_HIGH);
    assert_eq!(cfg.gop_length, 250);
    assert_eq!(cfg.frame_interval_p, 1);
    assert_eq!(cfg.rc_params.rate_control_mode, RC_VBR);
    assert_eq!(cfg.rc_params.target_quality, 20);
    assert_eq!(cfg.rc_params.max_bit_rate, 0);
    assert_eq!(cfg.rc_params.flags & RC_FLAG_LOOKAHEAD, 0);
    // SAFETY: `apply` wrote this member.
    let h264 = unsafe { cfg.codec.h264 };
    assert_eq!(h264.flags, 0);
    assert_eq!(h264.use_b_frames_as_ref, BFRAME_REF_DISABLED);
    assert_eq!(h264.idr_period, 250);
    assert_eq!(h264.chroma_format_idc, 1);
    assert_eq!(h264.input_bit_depth, 8);
    assert_eq!(h264.vui.video_full_range_flag, 1);
    assert_eq!(h264.vui.colour_matrix, 9);
    assert_eq!(h264.vui.transfer_characteristics, 16);
    assert_eq!(h264.vui.colour_primaries, 9);
    assert_eq!(h264.vui.chroma_sample_location_flag, 1);
    assert_eq!(h264.vui.chroma_sample_location_top, 2);
    assert_eq!(h264.vui.bitstream_restriction_flag, 1);
}

#[test]
fn hevc_10_bit_sets_main10_and_both_depths() {
    let mut cfg = NvEncConfig::zeroed();
    let mut encode = config(EncodeCodec::Hevc, PixelFormat::P010);
    encode.b_frames = 2;
    apply(&encode, &mut cfg);
    assert_eq!(cfg.profile_guid, PROFILE_HEVC_MAIN10);
    assert_eq!(cfg.frame_interval_p, 3);
    // SAFETY: `apply` wrote this member.
    let hevc = unsafe { cfg.codec.hevc };
    assert_eq!(hevc.input_bit_depth, 10);
    assert_eq!(hevc.output_bit_depth, 10);
    assert_eq!(hevc.flags, 1 << HEVC_CHROMA_FORMAT_SHIFT);
    assert_eq!(hevc.vui.colour_matrix, 9);
}

#[test]
fn av1_writes_colour_to_the_sequence_header_fields() {
    let mut cfg = NvEncConfig::zeroed();
    apply(&config(EncodeCodec::Av1, PixelFormat::Nv12), &mut cfg);
    assert_eq!(cfg.profile_guid, PROFILE_AV1_MAIN);
    // SAFETY: `apply` wrote this member.
    let av1 = unsafe { cfg.codec.av1 };
    assert_eq!(av1.color_range, 1);
    assert_eq!(av1.matrix_coefficients, 9);
    assert_eq!(av1.chroma_sample_position, 2);
    assert_eq!(
        av1.flags,
        AV1_FLAG_REPEAT_SEQ_HDR | (1 << AV1_CHROMA_FORMAT_SHIFT)
    );
}

#[test]
fn const_qp_and_bitrate_map_their_fields() {
    let mut encode = config(EncodeCodec::H264, PixelFormat::Nv12);
    encode.rate_control = RateControl::ConstQp {
        intra: 18,
        inter_p: 20,
        inter_b: 22,
    };
    let mut cfg = NvEncConfig::zeroed();
    apply(&encode, &mut cfg);
    assert_eq!(cfg.rc_params.rate_control_mode, RC_CONSTQP);
    assert_eq!(cfg.rc_params.const_qp.qp_intra, 18);
    assert_eq!(cfg.rc_params.const_qp.qp_inter_p, 20);
    assert_eq!(cfg.rc_params.const_qp.qp_inter_b, 22);

    encode.rate_control = RateControl::Bitrate {
        average: 5_000_000,
        maximum: Some(8_000_000),
    };
    apply(&encode, &mut cfg);
    assert_eq!(cfg.rc_params.rate_control_mode, RC_VBR);
    assert_eq!(cfg.rc_params.target_quality, 0);
    assert_eq!(cfg.rc_params.average_bit_rate, 5_000_000);
    assert_eq!(cfg.rc_params.max_bit_rate, 8_000_000);
}

#[test]
fn intra_only_uses_a_zero_p_interval() {
    let mut encode = config(EncodeCodec::H264, PixelFormat::Nv12);
    encode.gop_length = 1;
    let mut cfg = NvEncConfig::zeroed();
    apply(&encode, &mut cfg);
    assert_eq!(cfg.frame_interval_p, 0);
    // SAFETY: `apply` wrote this member.
    let h264 = unsafe { cfg.codec.h264 };
    assert_eq!(h264.vui.bitstream_restriction_flag, 0);
}

#[test]
fn initialize_params_carry_the_exact_rate_and_synchronous_mode() {
    let encode = config(EncodeCodec::Hevc, PixelFormat::P010);
    let mut cfg = NvEncConfig::zeroed();
    let params = initialize_params(&encode, &mut cfg);
    let Ok(params) = params else {
        panic!("the config is valid");
    };
    assert_eq!(params.version, INITIALIZE_PARAMS_VER);
    assert_eq!(params.encode_guid, CODEC_HEVC);
    assert_eq!(params.preset_guid, PRESETS[3]);
    assert_eq!(
        (params.frame_rate_num, params.frame_rate_den),
        (30000, 1001)
    );
    assert_eq!((params.dar_width, params.dar_height), (16, 9));
    assert_eq!(params.enable_encode_async, 0);
    assert_eq!(params.enable_ptd, 1);
    assert_eq!(params.buffer_format, BUFFER_FORMAT_YUV420_10BIT);
    assert_eq!(params.tuning_info, 1);
}

#[test]
fn codec_names_lists_what_the_gpu_reports() {
    assert_eq!(codec_names(&[CODEC_H264, CODEC_HEVC]), "H.264, HEVC");
    assert_eq!(codec_names(&[]), "no codec vidr knows");
}
