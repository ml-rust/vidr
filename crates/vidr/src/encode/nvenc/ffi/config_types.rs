//! `#[repr(C)]` mirrors of `NV_ENC_CONFIG` and the types it holds.
//!
//! The layout reference is as `types.rs` states. C bit fields become one `u32`
//! with the masks below, which `layout.rs` and a GCC probe tie to the header.

// The driver reads or writes every field, including the ones vidr never
// touches, so each mirror keeps all of them.
#![allow(dead_code, reason = "FFI mirrors keep every header field and constant")]

use std::ffi::{c_int, c_void};

use super::guid::Guid;
use super::types::{long_struct_version, struct_version};

/// `NV_ENC_CONFIG_VER`.
pub(crate) const CONFIG_VER: u32 = long_struct_version(9);
/// `NV_ENC_RC_PARAMS_VER`.
pub(crate) const RC_PARAMS_VER: u32 = struct_version(1);

/// `NV_ENC_PARAMS_RC_CONSTQP`.
pub(crate) const RC_CONSTQP: c_int = 0;
/// `NV_ENC_PARAMS_RC_VBR`.
pub(crate) const RC_VBR: c_int = 1;
/// `NV_ENC_BFRAME_REF_MODE_DISABLED`.
pub(crate) const BFRAME_REF_DISABLED: c_int = 0;
/// `NV_ENC_VUI_VIDEO_FORMAT` unspecified.
pub(crate) const VIDEO_FORMAT_UNSPECIFIED: c_int = 5;

/// `NV_ENC_RC_PARAMS::enableLookahead`.
pub(crate) const RC_FLAG_LOOKAHEAD: u32 = 1 << 5;

/// `NV_ENC_CONFIG_H264::outputAUD`.
pub(crate) const H264_FLAG_OUTPUT_AUD: u32 = 1 << 6;
/// `NV_ENC_CONFIG_H264::disableSPSPPS`.
pub(crate) const H264_FLAG_DISABLE_SPSPPS: u32 = 1 << 7;
/// `NV_ENC_CONFIG_H264::repeatSPSPPS`.
pub(crate) const H264_FLAG_REPEAT_SPSPPS: u32 = 1 << 12;

/// `NV_ENC_CONFIG_HEVC::outputAUD`.
pub(crate) const HEVC_FLAG_OUTPUT_AUD: u32 = 1 << 4;
/// `NV_ENC_CONFIG_HEVC::disableSPSPPS`.
pub(crate) const HEVC_FLAG_DISABLE_SPSPPS: u32 = 1 << 6;
/// `NV_ENC_CONFIG_HEVC::repeatSPSPPS`.
pub(crate) const HEVC_FLAG_REPEAT_SPSPPS: u32 = 1 << 7;
/// Bit of `NV_ENC_CONFIG_HEVC::chromaFormatIDC`, 2 bits wide.
pub(crate) const HEVC_CHROMA_FORMAT_SHIFT: u32 = 9;

/// `NV_ENC_CONFIG_AV1::outputAnnexBFormat`.
pub(crate) const AV1_FLAG_ANNEXB: u32 = 1;
/// `NV_ENC_CONFIG_AV1::disableSeqHdr`.
pub(crate) const AV1_FLAG_DISABLE_SEQ_HDR: u32 = 1 << 4;
/// `NV_ENC_CONFIG_AV1::repeatSeqHdr`.
pub(crate) const AV1_FLAG_REPEAT_SEQ_HDR: u32 = 1 << 5;
/// Bit of `NV_ENC_CONFIG_AV1::chromaFormatIDC`, 2 bits wide.
pub(crate) const AV1_CHROMA_FORMAT_SHIFT: u32 = 7;

/// `chromaFormatIDC` for 4:2:0.
pub(crate) const CHROMA_FORMAT_420: u32 = 1;

/// `NV_ENC_QP`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct NvEncQp {
    pub(crate) qp_inter_p: u32,
    pub(crate) qp_inter_b: u32,
    pub(crate) qp_intra: u32,
}

/// `NV_ENC_RC_PARAMS`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct NvEncRcParams {
    pub(crate) version: u32,
    pub(crate) rate_control_mode: c_int,
    pub(crate) const_qp: NvEncQp,
    pub(crate) average_bit_rate: u32,
    pub(crate) max_bit_rate: u32,
    pub(crate) vbv_buffer_size: u32,
    pub(crate) vbv_initial_delay: u32,
    /// `enableMinQP` to `reservedBitFields`.
    pub(crate) flags: u32,
    pub(crate) min_qp: NvEncQp,
    pub(crate) max_qp: NvEncQp,
    pub(crate) initial_rc_qp: NvEncQp,
    pub(crate) temporal_layer_idx_mask: u32,
    pub(crate) temporal_layer_qp: [u8; 8],
    pub(crate) target_quality: u8,
    pub(crate) target_quality_lsb: u8,
    pub(crate) lookahead_depth: u16,
    pub(crate) low_delay_key_frame_scale: u8,
    pub(crate) y_dc_qp_index_offset: i8,
    pub(crate) u_dc_qp_index_offset: i8,
    pub(crate) v_dc_qp_index_offset: i8,
    pub(crate) qp_map_mode: c_int,
    pub(crate) multi_pass: c_int,
    pub(crate) alpha_layer_bitrate_ratio: u32,
    pub(crate) cb_qp_index_offset: i8,
    pub(crate) cr_qp_index_offset: i8,
    pub(crate) reserved2: u16,
    pub(crate) lookahead_level: c_int,
    pub(crate) view_bitrate_ratios: [u8; 7],
    pub(crate) reserved3: u8,
    pub(crate) reserved1: u32,
}

/// `NV_ENC_CONFIG_H264_VUI_PARAMETERS`, which HEVC shares.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct NvEncVuiParameters {
    pub(crate) overscan_info_present_flag: u32,
    pub(crate) overscan_info: u32,
    pub(crate) video_signal_type_present_flag: u32,
    pub(crate) video_format: c_int,
    pub(crate) video_full_range_flag: u32,
    pub(crate) colour_description_present_flag: u32,
    pub(crate) colour_primaries: c_int,
    pub(crate) transfer_characteristics: c_int,
    pub(crate) colour_matrix: c_int,
    pub(crate) chroma_sample_location_flag: u32,
    pub(crate) chroma_sample_location_top: u32,
    pub(crate) chroma_sample_location_bot: u32,
    pub(crate) bitstream_restriction_flag: u32,
    pub(crate) timing_info_present_flag: u32,
    pub(crate) num_unit_in_ticks: u32,
    pub(crate) time_scale: u32,
    pub(crate) reserved: [u32; 12],
}

/// `NV_ENC_CONFIG_H264`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct NvEncConfigH264 {
    /// `enableTemporalSVC` to `reservedBitFields`.
    pub(crate) flags: u32,
    pub(crate) level: u32,
    pub(crate) idr_period: u32,
    pub(crate) separate_colour_plane_flag: u32,
    pub(crate) disable_deblocking_filter_idc: u32,
    pub(crate) num_temporal_layers: u32,
    pub(crate) sps_id: u32,
    pub(crate) pps_id: u32,
    pub(crate) adaptive_transform_mode: c_int,
    pub(crate) fmo_mode: c_int,
    pub(crate) bdirect_mode: c_int,
    pub(crate) entropy_coding_mode: c_int,
    pub(crate) stereo_mode: c_int,
    pub(crate) intra_refresh_period: u32,
    pub(crate) intra_refresh_cnt: u32,
    pub(crate) max_num_ref_frames: u32,
    pub(crate) slice_mode: u32,
    pub(crate) slice_mode_data: u32,
    pub(crate) vui: NvEncVuiParameters,
    pub(crate) ltr_num_frames: u32,
    pub(crate) ltr_trust_mode: u32,
    pub(crate) chroma_format_idc: u32,
    pub(crate) max_temporal_layers: u32,
    pub(crate) use_b_frames_as_ref: c_int,
    pub(crate) num_ref_l0: c_int,
    pub(crate) num_ref_l1: c_int,
    pub(crate) output_bit_depth: c_int,
    pub(crate) input_bit_depth: c_int,
    pub(crate) tf_level: c_int,
    pub(crate) reserved1: [u32; 264],
    pub(crate) reserved2: [*mut c_void; 64],
}

/// `NV_ENC_CONFIG_HEVC`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct NvEncConfigHevc {
    pub(crate) level: u32,
    pub(crate) tier: u32,
    pub(crate) min_cu_size: c_int,
    pub(crate) max_cu_size: c_int,
    /// `useConstrainedIntraPred` to `reserved`.
    pub(crate) flags: u32,
    pub(crate) idr_period: u32,
    pub(crate) intra_refresh_period: u32,
    pub(crate) intra_refresh_cnt: u32,
    pub(crate) max_num_ref_frames_in_dpb: u32,
    pub(crate) ltr_num_frames: u32,
    pub(crate) vps_id: u32,
    pub(crate) sps_id: u32,
    pub(crate) pps_id: u32,
    pub(crate) slice_mode: u32,
    pub(crate) slice_mode_data: u32,
    pub(crate) max_temporal_layers_minus1: u32,
    pub(crate) vui: NvEncVuiParameters,
    pub(crate) ltr_trust_mode: u32,
    pub(crate) use_b_frames_as_ref: c_int,
    pub(crate) num_ref_l0: c_int,
    pub(crate) num_ref_l1: c_int,
    pub(crate) tf_level: c_int,
    pub(crate) disable_deblocking_filter_idc: u32,
    pub(crate) output_bit_depth: c_int,
    pub(crate) input_bit_depth: c_int,
    pub(crate) num_temporal_layers: u32,
    pub(crate) num_views: u32,
    pub(crate) reserved1: [u32; 208],
    pub(crate) reserved2: [*mut c_void; 64],
}

/// `NV_ENC_CONFIG_AV1`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct NvEncConfigAv1 {
    pub(crate) level: u32,
    pub(crate) tier: u32,
    pub(crate) min_part_size: c_int,
    pub(crate) max_part_size: c_int,
    /// `outputAnnexBFormat` to `reserved`.
    pub(crate) flags: u32,
    pub(crate) idr_period: u32,
    pub(crate) intra_refresh_period: u32,
    pub(crate) intra_refresh_cnt: u32,
    pub(crate) max_num_ref_frames_in_dpb: u32,
    pub(crate) num_tile_columns: u32,
    pub(crate) num_tile_rows: u32,
    pub(crate) reserved2: u32,
    pub(crate) tile_widths: *mut u32,
    pub(crate) tile_heights: *mut u32,
    pub(crate) max_temporal_layers_minus1: u32,
    pub(crate) color_primaries: c_int,
    pub(crate) transfer_characteristics: c_int,
    pub(crate) matrix_coefficients: c_int,
    pub(crate) color_range: u32,
    pub(crate) chroma_sample_position: u32,
    pub(crate) use_b_frames_as_ref: c_int,
    pub(crate) film_grain_params: *mut c_void,
    pub(crate) num_fwd_refs: c_int,
    pub(crate) num_bwd_refs: c_int,
    pub(crate) output_bit_depth: c_int,
    pub(crate) input_bit_depth: c_int,
    pub(crate) ltr_num_frames: u32,
    pub(crate) num_temporal_layers: u32,
    pub(crate) tf_level: c_int,
    pub(crate) reserved1: [u32; 230],
    pub(crate) reserved3: [*mut c_void; 62],
}

/// `NV_ENC_CODEC_CONFIG`. The motion-estimation-only members fit inside these
/// and vidr never uses them.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) union NvEncCodecConfig {
    pub(crate) h264: NvEncConfigH264,
    pub(crate) hevc: NvEncConfigHevc,
    pub(crate) av1: NvEncConfigAv1,
    pub(crate) reserved: [u32; 320],
}

/// `NV_ENC_CONFIG`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct NvEncConfig {
    pub(crate) version: u32,
    pub(crate) profile_guid: Guid,
    pub(crate) gop_length: u32,
    pub(crate) frame_interval_p: i32,
    pub(crate) mono_chrome_encoding: u32,
    pub(crate) frame_field_mode: c_int,
    pub(crate) mv_precision: c_int,
    pub(crate) rc_params: NvEncRcParams,
    pub(crate) codec: NvEncCodecConfig,
    pub(crate) reserved: [u32; 278],
    pub(crate) reserved2: [*mut c_void; 64],
}

/// `NV_ENC_PRESET_CONFIG`.
#[repr(C)]
pub(crate) struct NvEncPresetConfig {
    pub(crate) version: u32,
    pub(crate) reserved: u32,
    pub(crate) preset_cfg: NvEncConfig,
    pub(crate) reserved1: [u32; 256],
    pub(crate) reserved2: [*mut c_void; 64],
}

impl_zeroed!(NvEncConfig, NvEncPresetConfig);
