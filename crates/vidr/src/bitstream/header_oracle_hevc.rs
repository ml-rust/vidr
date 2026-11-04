//! HEVC parameter sets and slice headers against `trace_headers`.

use std::collections::BTreeMap;

use super::super::hevc_pps::PicParameterSet;
use super::super::hevc_ptl::ProfileTierLevel;
use super::super::hevc_rps::ShortTermRefPicSet;
use super::super::hevc_scaling::ScalingListData;
use super::super::hevc_slice::{
    SLICE_TYPE_B, SLICE_TYPE_I, SliceSegmentHeader, slice_pic_parameter_set_id,
};
use super::super::hevc_sps_full::SeqParameterSet;
use super::super::hevc_vps::VideoParameterSet;
use super::*;

/// NAL unit framing, trailing bits, reserved bits, and extension payloads.
const IGNORED_FRAMING: &[&str] = &[
    "forbidden_zero_bit",
    "nal_unit_type",
    "nuh_layer_id",
    "nuh_temporal_id_plus1",
    "rbsp_stop_one_bit",
    "rbsp_alignment_zero_bit",
    "alignment_bit_equal_to_one",
    "alignment_bit_equal_to_zero",
    "extension_data",
    "reserved_zero_2bits",
    "vps_reserved_0xffff_16bits",
    "general_reserved_zero_7bits",
    "general_reserved_zero_33bits",
    "general_reserved_zero_34bits",
    "general_reserved_zero_35bits",
    "general_reserved_zero_43bits",
    "general_reserved_zero_bit",
    "sub_layer_reserved_zero_7bits",
    "sub_layer_reserved_zero_33bits",
    "sub_layer_reserved_zero_34bits",
    "sub_layer_reserved_zero_35bits",
    "sub_layer_reserved_zero_43bits",
    "sub_layer_reserved_zero_bit",
];
/// `check_constraint_flags` compares these against the kept bit field.
const CONSTRAINT_FLAGS: &[(&str, u32)] = &[
    ("progressive_source_flag", 47),
    ("interlaced_source_flag", 46),
    ("non_packed_constraint_flag", 45),
    ("frame_only_constraint_flag", 44),
    ("max_12bit_constraint_flag", 43),
    ("max_10bit_constraint_flag", 42),
    ("max_8bit_constraint_flag", 41),
    ("max_422chroma_constraint_flag", 40),
    ("max_420chroma_constraint_flag", 39),
    ("max_monochrome_constraint_flag", 38),
    ("intra_constraint_flag", 37),
    ("one_picture_only_constraint_flag", 36),
    ("lower_bit_rate_constraint_flag", 35),
    ("max_14bit_constraint_flag", 34),
    ("inbld_flag", 0),
];
/// `hrd_parameters()`, read and dropped.
const IGNORED_HRD: &[&str] = &[
    "nal_hrd_parameters_present_flag",
    "vcl_hrd_parameters_present_flag",
    "sub_pic_hrd_params_present_flag",
    "tick_divisor_minus2",
    "du_cpb_removal_delay_increment_length_minus1",
    "sub_pic_cpb_params_in_pic_timing_sei_flag",
    "dpb_output_delay_du_length_minus1",
    "bit_rate_scale",
    "cpb_size_scale",
    "cpb_size_du_scale",
    "initial_cpb_removal_delay_length_minus1",
    "au_cpb_removal_delay_length_minus1",
    "dpb_output_delay_length_minus1",
    "fixed_pic_rate_general_flag",
    "fixed_pic_rate_within_cvs_flag",
    "elemental_duration_in_tc_minus1",
    "low_delay_hrd_flag",
    "cpb_cnt_minus1",
    "bit_rate_value_minus1",
    "cpb_size_value_minus1",
    "cpb_size_du_value_minus1",
    "bit_rate_du_value_minus1",
    "cbr_flag",
];
/// `vui_parameters()` fields that vidr does not keep.
const IGNORED_VUI: &[&str] = &[
    "aspect_ratio_info_present_flag",
    "aspect_ratio_idc",
    "sar_width",
    "sar_height",
    "overscan_info_present_flag",
    "overscan_appropriate_flag",
    "video_signal_type_present_flag",
    "colour_description_present_flag",
    "chroma_loc_info_present_flag",
    "neutral_chroma_indication_flag",
    "field_seq_flag",
    "frame_field_info_present_flag",
    "default_display_window_flag",
    "def_disp_win_left_offset",
    "def_disp_win_right_offset",
    "def_disp_win_top_offset",
    "def_disp_win_bottom_offset",
    "vui_timing_info_present_flag",
    "vui_num_units_in_tick",
    "vui_time_scale",
    "vui_poc_proportional_to_timing_flag",
    "vui_num_ticks_poc_diff_one_minus1",
    "vui_hrd_parameters_present_flag",
    "bitstream_restriction_flag",
    "tiles_fixed_structure_flag",
    "motion_vectors_over_pic_boundaries_flag",
    "restricted_ref_pic_lists_flag",
    "max_bytes_per_pic_denom",
    "max_bits_per_min_cu_denom",
    "log2_max_mv_length_horizontal",
    "log2_max_mv_length_vertical",
];
/// VPS elements with no decoding use.
const IGNORED_VPS: &[&str] = &[
    "layer_id_included_flag",
    "hrd_layer_set_idx",
    "cprms_present_flag",
];
/// Slice header bytes a decoder skips.
const IGNORED_SLICE: &[&str] = &["slice_segment_header_extension_data_byte"];

/// The names of [`CONSTRAINT_FLAGS`], under their prefix and index.
fn constraint_names(prefix: &str, suffix: &str) -> Vec<(String, u32)> {
    CONSTRAINT_FLAGS
        .iter()
        .map(|(name, bit)| (format!("{prefix}{name}{suffix}"), *bit))
        .collect()
}

/// Compares each constraint flag ffmpeg logs against its bit in `flags`.
fn check_constraint_flags(
    label: &str,
    oracle: &[(String, i64)],
    names: &[(String, u32)],
    flags: u64,
) {
    for (element, value) in oracle {
        if let Some((_, bit)) = names.iter().find(|(name, _)| name == element) {
            assert_eq!(((flags >> bit) & 1) as i64, *value, "{label}: {element}");
        }
    }
}

fn ptl_elements(out: &mut Elements, ptl: &ProfileTierLevel, sub_layers: usize) {
    let general = &ptl.general;
    out.put("general_profile_space", general.profile_space);
    out.put("general_tier_flag", general.tier_flag);
    out.put("general_profile_idc", general.profile_idc);
    for j in 0..32 {
        out.put(
            format!("general_profile_compatibility_flag[{j}]"),
            (general.compatibility_flags >> (31 - j)) & 1,
        );
    }
    out.put("general_level_idc", general.level_idc);
    for (i, layer) in ptl.sub_layers.iter().enumerate().take(sub_layers) {
        out.put(
            format!("sub_layer_profile_present_flag[{i}]"),
            layer.sub_layer_profile_present_flag,
        );
        out.put(
            format!("sub_layer_level_present_flag[{i}]"),
            layer.sub_layer_level_present_flag,
        );
        if layer.sub_layer_profile_present_flag {
            out.put(
                format!("sub_layer_profile_space[{i}]"),
                layer.sub_layer_profile_space,
            );
            out.put(
                format!("sub_layer_tier_flag[{i}]"),
                layer.sub_layer_tier_flag,
            );
            out.put(
                format!("sub_layer_profile_idc[{i}]"),
                layer.sub_layer_profile_idc,
            );
            for j in 0..32 {
                out.put(
                    format!("sub_layer_profile_compatibility_flag[{i}][{j}]"),
                    (layer.compatibility_flags >> (31 - j)) & 1,
                );
            }
        }
        if layer.sub_layer_level_present_flag {
            out.put(
                format!("sub_layer_level_idc[{i}]"),
                layer.sub_layer_level_idc,
            );
        }
    }
}

/// The constraint flags of `ptl`, ignored by name in [`compare`].
fn ignored_constraint_names() -> Vec<String> {
    let mut names = Vec::new();
    for (name, _) in CONSTRAINT_FLAGS {
        names.push(format!("general_{name}"));
        names.push(format!("sub_layer_{name}"));
    }
    names
}

fn check_ptl_flags(
    label: &str,
    oracle: &[(String, i64)],
    ptl: &ProfileTierLevel,
    sub_layers: usize,
) {
    check_constraint_flags(
        label,
        oracle,
        &constraint_names("general_", ""),
        ptl.general.constraint_flags,
    );
    for (i, layer) in ptl.sub_layers.iter().enumerate().take(sub_layers) {
        check_constraint_flags(
            label,
            oracle,
            &constraint_names("sub_layer_", &format!("[{i}]")),
            layer.constraint_flags,
        );
    }
}

fn vps_elements(vps: &VideoParameterSet) -> Elements {
    let mut out = Elements::default();
    out.put("vps_video_parameter_set_id", vps.vps_video_parameter_set_id);
    out.put(
        "vps_base_layer_internal_flag",
        vps.vps_base_layer_internal_flag,
    );
    out.put(
        "vps_base_layer_available_flag",
        vps.vps_base_layer_available_flag,
    );
    out.put("vps_max_layers_minus1", vps.vps_max_layers_minus1);
    out.put("vps_max_sub_layers_minus1", vps.vps_max_sub_layers_minus1);
    out.put(
        "vps_temporal_id_nesting_flag",
        vps.vps_temporal_id_nesting_flag,
    );
    let top = usize::from(vps.vps_max_sub_layers_minus1);
    ptl_elements(&mut out, &vps.profile_tier_level, top);
    out.put(
        "vps_sub_layer_ordering_info_present_flag",
        vps.vps_sub_layer_ordering_info_present_flag,
    );
    let first = if vps.vps_sub_layer_ordering_info_present_flag {
        0
    } else {
        top
    };
    for i in first..=top {
        out.put(
            format!("vps_max_dec_pic_buffering_minus1[{i}]"),
            vps.vps_max_dec_pic_buffering_minus1[i],
        );
        out.put(
            format!("vps_max_num_reorder_pics[{i}]"),
            vps.vps_max_num_reorder_pics[i],
        );
        out.put(
            format!("vps_max_latency_increase_plus1[{i}]"),
            vps.vps_max_latency_increase_plus1[i],
        );
    }
    out.put("vps_max_layer_id", vps.vps_max_layer_id);
    out.put("vps_num_layer_sets_minus1", vps.vps_num_layer_sets_minus1);
    out.put(
        "vps_timing_info_present_flag",
        vps.vps_timing_info_present_flag,
    );
    if vps.vps_timing_info_present_flag {
        out.put("vps_num_units_in_tick", vps.vps_num_units_in_tick);
        out.put("vps_time_scale", vps.vps_time_scale);
        out.put(
            "vps_poc_proportional_to_timing_flag",
            vps.vps_poc_proportional_to_timing_flag,
        );
        if vps.vps_poc_proportional_to_timing_flag {
            out.put(
                "vps_num_ticks_poc_diff_one_minus1",
                vps.vps_num_ticks_poc_diff_one_minus1,
            );
        }
        out.put("vps_num_hrd_parameters", vps.vps_num_hrd_parameters);
    }
    out.put("vps_extension_flag", vps.vps_extension_flag);
    out
}

fn scaling_elements(out: &mut Elements, data: &ScalingListData) {
    for size_id in 0..4usize {
        let step = if size_id == 3 { 3 } else { 1 };
        for matrix_id in (0..6usize).step_by(step) {
            let explicit = data.scaling_list_pred_mode_flag[size_id][matrix_id];
            out.put(
                format!("scaling_list_pred_mode_flag[{size_id}][{matrix_id}]"),
                explicit,
            );
            if !explicit {
                out.put(
                    format!("scaling_list_pred_matrix_id_delta[{size_id}][{matrix_id}]"),
                    data.scaling_list_pred_matrix_id_delta[size_id][matrix_id],
                );
                continue;
            }
            let mut previous = 8i64;
            if size_id > 1 {
                let dc = data.scaling_list_dc_coef_minus8[size_id - 2][matrix_id];
                out.put(
                    format!("scaling_list_dc_coef_minus8[{}][{matrix_id}]", size_id - 2),
                    dc,
                );
                previous = i64::from(dc) + 8;
            }
            let count = if size_id == 0 { 16 } else { 64 };
            for (i, coefficient) in data.scaling_list[size_id][matrix_id]
                .iter()
                .take(count)
                .enumerate()
            {
                // The delta range covers each residue modulo 256 once.
                let current = i64::from(*coefficient);
                let delta = (current - previous + 128).rem_euclid(256) - 128;
                out.put(
                    format!("scaling_list_delta_coeff[{size_id}][{matrix_id}][{i}]"),
                    delta,
                );
                previous = current;
            }
        }
    }
}

fn rps_elements(out: &mut Elements, set: &ShortTermRefPicSet, index: usize, in_slice: bool) {
    if index != 0 {
        out.put(
            "inter_ref_pic_set_prediction_flag",
            set.inter_ref_pic_set_prediction_flag,
        );
    }
    if set.inter_ref_pic_set_prediction_flag {
        if in_slice {
            out.put("delta_idx_minus1", set.delta_idx_minus1);
        }
        out.put("delta_rps_sign", set.delta_rps_sign);
        out.put("abs_delta_rps_minus1", set.abs_delta_rps_minus1);
        for j in 0..u32::from(set.num_prediction_flags) {
            let used = (set.used_by_curr_pic_flag >> j) & 1;
            out.put(format!("used_by_curr_pic_flag[{j}]"), used);
            if used == 0 {
                out.put(
                    format!("use_delta_flag[{j}]"),
                    (set.use_delta_flag >> j) & 1,
                );
            }
        }
    } else {
        out.put("num_negative_pics", set.num_negative_pics);
        out.put("num_positive_pics", set.num_positive_pics);
        for i in 0..usize::from(set.num_negative_pics) {
            out.put(
                format!("delta_poc_s0_minus1[{i}]"),
                set.delta_poc_s0_minus1[i],
            );
            out.put(
                format!("used_by_curr_pic_s0_flag[{i}]"),
                set.used_by_curr_pic_s0[i],
            );
        }
        for i in 0..usize::from(set.num_positive_pics) {
            out.put(
                format!("delta_poc_s1_minus1[{i}]"),
                set.delta_poc_s1_minus1[i],
            );
            out.put(
                format!("used_by_curr_pic_s1_flag[{i}]"),
                set.used_by_curr_pic_s1[i],
            );
        }
    }
}

fn sps_elements(sps: &SeqParameterSet) -> Elements {
    let mut out = Elements::default();
    out.put("sps_video_parameter_set_id", sps.sps_video_parameter_set_id);
    out.put("sps_max_sub_layers_minus1", sps.sps_max_sub_layers_minus1);
    out.put(
        "sps_temporal_id_nesting_flag",
        sps.sps_temporal_id_nesting_flag,
    );
    let top = usize::from(sps.sps_max_sub_layers_minus1);
    ptl_elements(&mut out, &sps.profile_tier_level, top);
    out.put("sps_seq_parameter_set_id", sps.sps_seq_parameter_set_id);
    out.put("chroma_format_idc", sps.chroma_format_idc);
    if sps.chroma_format_idc == 3 {
        out.put("separate_colour_plane_flag", sps.separate_colour_plane_flag);
    }
    out.put("pic_width_in_luma_samples", sps.pic_width_in_luma_samples);
    out.put("pic_height_in_luma_samples", sps.pic_height_in_luma_samples);
    out.put("conformance_window_flag", sps.conformance_window_flag);
    if sps.conformance_window_flag {
        out.put("conf_win_left_offset", sps.conf_win_left_offset);
        out.put("conf_win_right_offset", sps.conf_win_right_offset);
        out.put("conf_win_top_offset", sps.conf_win_top_offset);
        out.put("conf_win_bottom_offset", sps.conf_win_bottom_offset);
    }
    out.put("bit_depth_luma_minus8", sps.bit_depth_luma_minus8);
    out.put("bit_depth_chroma_minus8", sps.bit_depth_chroma_minus8);
    out.put(
        "log2_max_pic_order_cnt_lsb_minus4",
        sps.log2_max_pic_order_cnt_lsb_minus4,
    );
    out.put(
        "sps_sub_layer_ordering_info_present_flag",
        sps.sps_sub_layer_ordering_info_present_flag,
    );
    let first = if sps.sps_sub_layer_ordering_info_present_flag {
        0
    } else {
        top
    };
    for i in first..=top {
        out.put(
            format!("sps_max_dec_pic_buffering_minus1[{i}]"),
            sps.sps_max_dec_pic_buffering_minus1[i],
        );
        out.put(
            format!("sps_max_num_reorder_pics[{i}]"),
            sps.sps_max_num_reorder_pics[i],
        );
        out.put(
            format!("sps_max_latency_increase_plus1[{i}]"),
            sps.sps_max_latency_increase_plus1[i],
        );
    }
    out.put(
        "log2_min_luma_coding_block_size_minus3",
        sps.log2_min_luma_coding_block_size_minus3,
    );
    out.put(
        "log2_diff_max_min_luma_coding_block_size",
        sps.log2_diff_max_min_luma_coding_block_size,
    );
    out.put(
        "log2_min_luma_transform_block_size_minus2",
        sps.log2_min_luma_transform_block_size_minus2,
    );
    out.put(
        "log2_diff_max_min_luma_transform_block_size",
        sps.log2_diff_max_min_luma_transform_block_size,
    );
    out.put(
        "max_transform_hierarchy_depth_inter",
        sps.max_transform_hierarchy_depth_inter,
    );
    out.put(
        "max_transform_hierarchy_depth_intra",
        sps.max_transform_hierarchy_depth_intra,
    );
    out.put("scaling_list_enabled_flag", sps.scaling_list_enabled_flag);
    if sps.scaling_list_enabled_flag {
        out.put(
            "sps_scaling_list_data_present_flag",
            sps.sps_scaling_list_data_present_flag,
        );
    }
    if let Some(data) = &sps.scaling_list_data {
        scaling_elements(&mut out, data);
    }
    out.put("amp_enabled_flag", sps.amp_enabled_flag);
    out.put(
        "sample_adaptive_offset_enabled_flag",
        sps.sample_adaptive_offset_enabled_flag,
    );
    out.put("pcm_enabled_flag", sps.pcm_enabled_flag);
    if sps.pcm_enabled_flag {
        out.put(
            "pcm_sample_bit_depth_luma_minus1",
            sps.pcm_sample_bit_depth_luma_minus1,
        );
        out.put(
            "pcm_sample_bit_depth_chroma_minus1",
            sps.pcm_sample_bit_depth_chroma_minus1,
        );
        out.put(
            "log2_min_pcm_luma_coding_block_size_minus3",
            sps.log2_min_pcm_luma_coding_block_size_minus3,
        );
        out.put(
            "log2_diff_max_min_pcm_luma_coding_block_size",
            sps.log2_diff_max_min_pcm_luma_coding_block_size,
        );
        out.put(
            "pcm_loop_filter_disabled_flag",
            sps.pcm_loop_filter_disabled_flag,
        );
    }
    out.put(
        "num_short_term_ref_pic_sets",
        sps.st_ref_pic_set.len() as u32,
    );
    for (index, set) in sps.st_ref_pic_set.iter().enumerate() {
        rps_elements(&mut out, set, index, false);
    }
    out.put(
        "long_term_ref_pics_present_flag",
        sps.long_term_ref_pics_present_flag,
    );
    if sps.long_term_ref_pics_present_flag {
        out.put("num_long_term_ref_pics_sps", sps.num_long_term_ref_pics_sps);
        for i in 0..sps.num_long_term_ref_pics_sps as usize {
            out.put(
                format!("lt_ref_pic_poc_lsb_sps[{i}]"),
                sps.lt_ref_pic_poc_lsb_sps[i],
            );
            out.put(
                format!("used_by_curr_pic_lt_sps_flag[{i}]"),
                sps.used_by_curr_pic_lt_sps_flag[i],
            );
        }
    }
    out.put(
        "sps_temporal_mvp_enabled_flag",
        sps.sps_temporal_mvp_enabled_flag,
    );
    out.put(
        "strong_intra_smoothing_enabled_flag",
        sps.strong_intra_smoothing_enabled_flag,
    );
    out.put(
        "vui_parameters_present_flag",
        sps.vui_parameters_present_flag,
    );
    if let Some(vui) = &sps.vui {
        if let Some(signal) = vui.signal.video_signal_type {
            out.put("video_format", signal.video_format);
            out.put("video_full_range_flag", signal.full_range);
            if let Some(colour) = signal.colour {
                out.put("colour_primaries", colour.colour_primaries);
                out.put("transfer_characteristics", colour.transfer_characteristics);
                out.put("matrix_coefficients", colour.matrix_coefficients);
            }
        }
        if let Some(location) = vui.signal.chroma_sample_loc {
            out.put("chroma_sample_loc_type_top_field", location.top_field);
            out.put("chroma_sample_loc_type_bottom_field", location.bottom_field);
        }
        if let Some(idc) = vui.min_spatial_segmentation_idc {
            out.put("min_spatial_segmentation_idc", idc);
        }
    }
    out.put("sps_extension_present_flag", sps.sps_extension_present_flag);
    if sps.sps_extension_present_flag {
        out.put("sps_range_extension_flag", sps.sps_range_extension_flag);
        out.put(
            "sps_multilayer_extension_flag",
            sps.sps_multilayer_extension_flag,
        );
        out.put("sps_3d_extension_flag", sps.sps_3d_extension_flag);
        out.put("sps_scc_extension_flag", false);
        out.put("sps_extension_4bits", sps.sps_extension_4bits);
    }
    if sps.sps_range_extension_flag {
        let range = &sps.sps_range_extension;
        out.put(
            "transform_skip_rotation_enabled_flag",
            range.transform_skip_rotation_enabled_flag,
        );
        out.put(
            "transform_skip_context_enabled_flag",
            range.transform_skip_context_enabled_flag,
        );
        out.put(
            "implicit_rdpcm_enabled_flag",
            range.implicit_rdpcm_enabled_flag,
        );
        out.put(
            "explicit_rdpcm_enabled_flag",
            range.explicit_rdpcm_enabled_flag,
        );
        out.put(
            "extended_precision_processing_flag",
            range.extended_precision_processing_flag,
        );
        out.put(
            "intra_smoothing_disabled_flag",
            range.intra_smoothing_disabled_flag,
        );
        out.put(
            "high_precision_offsets_enabled_flag",
            range.high_precision_offsets_enabled_flag,
        );
        out.put(
            "persistent_rice_adaptation_enabled_flag",
            range.persistent_rice_adaptation_enabled_flag,
        );
        out.put(
            "cabac_bypass_alignment_enabled_flag",
            range.cabac_bypass_alignment_enabled_flag,
        );
    }
    if sps.sps_multilayer_extension_flag {
        out.put(
            "inter_view_mv_vert_constraint_flag",
            sps.inter_view_mv_vert_constraint_flag,
        );
    }
    out
}

fn pps_elements(pps: &PicParameterSet) -> Elements {
    let mut out = Elements::default();
    out.put("pps_pic_parameter_set_id", pps.pps_pic_parameter_set_id);
    out.put("pps_seq_parameter_set_id", pps.pps_seq_parameter_set_id);
    out.put(
        "dependent_slice_segments_enabled_flag",
        pps.dependent_slice_segments_enabled_flag,
    );
    out.put("output_flag_present_flag", pps.output_flag_present_flag);
    out.put(
        "num_extra_slice_header_bits",
        pps.num_extra_slice_header_bits,
    );
    out.put(
        "sign_data_hiding_enabled_flag",
        pps.sign_data_hiding_enabled_flag,
    );
    out.put("cabac_init_present_flag", pps.cabac_init_present_flag);
    out.put(
        "num_ref_idx_l0_default_active_minus1",
        pps.num_ref_idx_l0_default_active_minus1,
    );
    out.put(
        "num_ref_idx_l1_default_active_minus1",
        pps.num_ref_idx_l1_default_active_minus1,
    );
    out.put("init_qp_minus26", pps.init_qp_minus26);
    out.put(
        "constrained_intra_pred_flag",
        pps.constrained_intra_pred_flag,
    );
    out.put(
        "transform_skip_enabled_flag",
        pps.transform_skip_enabled_flag,
    );
    out.put("cu_qp_delta_enabled_flag", pps.cu_qp_delta_enabled_flag);
    if pps.cu_qp_delta_enabled_flag {
        out.put("diff_cu_qp_delta_depth", pps.diff_cu_qp_delta_depth);
    }
    out.put("pps_cb_qp_offset", pps.pps_cb_qp_offset);
    out.put("pps_cr_qp_offset", pps.pps_cr_qp_offset);
    out.put(
        "pps_slice_chroma_qp_offsets_present_flag",
        pps.pps_slice_chroma_qp_offsets_present_flag,
    );
    out.put("weighted_pred_flag", pps.weighted_pred_flag);
    out.put("weighted_bipred_flag", pps.weighted_bipred_flag);
    out.put(
        "transquant_bypass_enabled_flag",
        pps.transquant_bypass_enabled_flag,
    );
    out.put("tiles_enabled_flag", pps.tiles_enabled_flag);
    out.put(
        "entropy_coding_sync_enabled_flag",
        pps.entropy_coding_sync_enabled_flag,
    );
    if pps.tiles_enabled_flag {
        out.put("num_tile_columns_minus1", pps.num_tile_columns_minus1);
        out.put("num_tile_rows_minus1", pps.num_tile_rows_minus1);
        out.put("uniform_spacing_flag", pps.uniform_spacing_flag);
        for (i, size) in pps.column_width_minus1.iter().enumerate() {
            out.put(format!("column_width_minus1[{i}]"), *size);
        }
        for (i, size) in pps.row_height_minus1.iter().enumerate() {
            out.put(format!("row_height_minus1[{i}]"), *size);
        }
        out.put(
            "loop_filter_across_tiles_enabled_flag",
            pps.loop_filter_across_tiles_enabled_flag,
        );
    }
    out.put(
        "pps_loop_filter_across_slices_enabled_flag",
        pps.pps_loop_filter_across_slices_enabled_flag,
    );
    out.put(
        "deblocking_filter_control_present_flag",
        pps.deblocking_filter_control_present_flag,
    );
    if pps.deblocking_filter_control_present_flag {
        out.put(
            "deblocking_filter_override_enabled_flag",
            pps.deblocking_filter_override_enabled_flag,
        );
        out.put(
            "pps_deblocking_filter_disabled_flag",
            pps.pps_deblocking_filter_disabled_flag,
        );
        if !pps.pps_deblocking_filter_disabled_flag {
            out.put("pps_beta_offset_div2", pps.pps_beta_offset_div2);
            out.put("pps_tc_offset_div2", pps.pps_tc_offset_div2);
        }
    }
    out.put(
        "pps_scaling_list_data_present_flag",
        pps.pps_scaling_list_data_present_flag,
    );
    if let Some(data) = &pps.scaling_list_data {
        scaling_elements(&mut out, data);
    }
    out.put(
        "lists_modification_present_flag",
        pps.lists_modification_present_flag,
    );
    out.put(
        "log2_parallel_merge_level_minus2",
        pps.log2_parallel_merge_level_minus2,
    );
    out.put(
        "slice_segment_header_extension_present_flag",
        pps.slice_segment_header_extension_present_flag,
    );
    out.put("pps_extension_present_flag", pps.pps_extension_present_flag);
    if pps.pps_extension_present_flag {
        out.put("pps_range_extension_flag", pps.pps_range_extension_flag);
        out.put(
            "pps_multilayer_extension_flag",
            pps.pps_multilayer_extension_flag,
        );
        out.put("pps_3d_extension_flag", pps.pps_3d_extension_flag);
        out.put("pps_scc_extension_flag", false);
        out.put("pps_extension_4bits", pps.pps_extension_4bits);
    }
    if pps.pps_range_extension_flag {
        let range = &pps.pps_range_extension;
        if pps.transform_skip_enabled_flag {
            out.put(
                "log2_max_transform_skip_block_size_minus2",
                range.log2_max_transform_skip_block_size_minus2,
            );
        }
        out.put(
            "cross_component_prediction_enabled_flag",
            range.cross_component_prediction_enabled_flag,
        );
        out.put(
            "chroma_qp_offset_list_enabled_flag",
            range.chroma_qp_offset_list_enabled_flag,
        );
        if range.chroma_qp_offset_list_enabled_flag {
            out.put(
                "diff_cu_chroma_qp_offset_depth",
                range.diff_cu_chroma_qp_offset_depth,
            );
            out.put(
                "chroma_qp_offset_list_len_minus1",
                range.chroma_qp_offset_list_len_minus1,
            );
            for i in 0..=range.chroma_qp_offset_list_len_minus1 as usize {
                out.put(
                    format!("cb_qp_offset_list[{i}]"),
                    range.cb_qp_offset_list[i],
                );
                out.put(
                    format!("cr_qp_offset_list[{i}]"),
                    range.cr_qp_offset_list[i],
                );
            }
        }
        out.put(
            "log2_sao_offset_scale_luma",
            range.log2_sao_offset_scale_luma,
        );
        out.put(
            "log2_sao_offset_scale_chroma",
            range.log2_sao_offset_scale_chroma,
        );
    }
    out
}

fn weight_elements(
    out: &mut Elements,
    list: &super::super::hevc_pred_weight::PredWeightList,
    count: usize,
    suffix: &str,
    chroma: bool,
) {
    for i in 0..count {
        out.put(
            format!("luma_weight_{suffix}_flag[{i}]"),
            list.luma_weight_flag[i],
        );
        if chroma {
            out.put(
                format!("chroma_weight_{suffix}_flag[{i}]"),
                list.chroma_weight_flag[i],
            );
        }
        if list.luma_weight_flag[i] {
            out.put(
                format!("delta_luma_weight_{suffix}[{i}]"),
                list.delta_luma_weight[i],
            );
            out.put(format!("luma_offset_{suffix}[{i}]"), list.luma_offset[i]);
        }
        if list.chroma_weight_flag[i] {
            for j in 0..2 {
                out.put(
                    format!("delta_chroma_weight_{suffix}[{i}][{j}]"),
                    list.delta_chroma_weight[i][j],
                );
                // ffmpeg logs `delta_chroma_offset_l0` under this name.
                out.put(
                    format!("chroma_offset_{suffix}[{i}][{j}]"),
                    list.delta_chroma_offset[i][j],
                );
            }
        }
    }
}

fn slice_elements(
    header: &SliceSegmentHeader,
    pps: &PicParameterSet,
    sps: &SeqParameterSet,
) -> Elements {
    let mut out = Elements::default();
    out.put(
        "first_slice_segment_in_pic_flag",
        header.first_slice_segment_in_pic_flag,
    );
    if (16..=23).contains(&header.nal_unit_type) {
        out.put(
            "no_output_of_prior_pics_flag",
            header.no_output_of_prior_pics_flag,
        );
    }
    out.put(
        "slice_pic_parameter_set_id",
        header.slice_pic_parameter_set_id,
    );
    if !header.first_slice_segment_in_pic_flag {
        if pps.dependent_slice_segments_enabled_flag {
            out.put(
                "dependent_slice_segment_flag",
                header.dependent_slice_segment_flag,
            );
        }
        out.put("slice_segment_address", header.slice_segment_address);
    }
    if !header.dependent_slice_segment_flag {
        independent_elements(&mut out, header, pps, sps);
    }
    if pps.tiles_enabled_flag || pps.entropy_coding_sync_enabled_flag {
        out.put(
            "num_entry_point_offsets",
            header.entry_point_offset_minus1.len() as u32,
        );
        if !header.entry_point_offset_minus1.is_empty() {
            out.put("offset_len_minus1", header.offset_len_minus1);
            for (i, offset) in header.entry_point_offset_minus1.iter().enumerate() {
                out.put(format!("entry_point_offset_minus1[{i}]"), *offset);
            }
        }
    }
    if pps.slice_segment_header_extension_present_flag {
        out.put(
            "slice_segment_header_extension_length",
            header.slice_segment_header_extension_length,
        );
    }
    out
}

fn independent_elements(
    out: &mut Elements,
    header: &SliceSegmentHeader,
    pps: &PicParameterSet,
    sps: &SeqParameterSet,
) {
    let extra = u32::from(pps.num_extra_slice_header_bits);
    for i in 0..extra {
        out.put(
            format!("slice_reserved_flag[{i}]"),
            (u32::from(header.slice_reserved_flag) >> (extra - 1 - i)) & 1,
        );
    }
    out.put("slice_type", header.slice_type);
    if pps.output_flag_present_flag {
        out.put("pic_output_flag", header.pic_output_flag);
    }
    if sps.separate_colour_plane_flag {
        out.put("colour_plane_id", header.colour_plane_id);
    }
    if !matches!(header.nal_unit_type, 19 | 20) {
        out.put("slice_pic_order_cnt_lsb", header.slice_pic_order_cnt_lsb);
        out.put(
            "short_term_ref_pic_set_sps_flag",
            header.short_term_ref_pic_set_sps_flag,
        );
        if let Some(set) = &header.short_term_ref_pic_set {
            rps_elements(out, set, sps.st_ref_pic_set.len(), true);
        } else if sps.st_ref_pic_set.len() > 1 {
            out.put(
                "short_term_ref_pic_set_idx",
                header.short_term_ref_pic_set_idx,
            );
        }
        if sps.long_term_ref_pics_present_flag {
            long_term_elements(out, header, sps);
        }
        if sps.sps_temporal_mvp_enabled_flag {
            out.put(
                "slice_temporal_mvp_enabled_flag",
                header.slice_temporal_mvp_enabled_flag,
            );
        }
    }
    if sps.sample_adaptive_offset_enabled_flag {
        out.put("slice_sao_luma_flag", header.slice_sao_luma_flag);
        if sps.chroma_array_type() != 0 {
            out.put("slice_sao_chroma_flag", header.slice_sao_chroma_flag);
        }
    }
    if header.slice_type != SLICE_TYPE_I {
        inter_elements(out, header, pps, sps);
    }
    out.put("slice_qp_delta", header.slice_qp_delta);
    if pps.pps_slice_chroma_qp_offsets_present_flag {
        out.put("slice_cb_qp_offset", header.slice_cb_qp_offset);
        out.put("slice_cr_qp_offset", header.slice_cr_qp_offset);
    }
    if pps.pps_range_extension.chroma_qp_offset_list_enabled_flag {
        out.put(
            "cu_chroma_qp_offset_enabled_flag",
            header.cu_chroma_qp_offset_enabled_flag,
        );
    }
    if pps.deblocking_filter_override_enabled_flag {
        out.put(
            "deblocking_filter_override_flag",
            header.deblocking_filter_override_flag,
        );
    }
    if header.deblocking_filter_override_flag {
        out.put(
            "slice_deblocking_filter_disabled_flag",
            header.slice_deblocking_filter_disabled_flag,
        );
        if !header.slice_deblocking_filter_disabled_flag {
            out.put("slice_beta_offset_div2", header.slice_beta_offset_div2);
            out.put("slice_tc_offset_div2", header.slice_tc_offset_div2);
        }
    }
    if pps.pps_loop_filter_across_slices_enabled_flag
        && (header.slice_sao_luma_flag
            || header.slice_sao_chroma_flag
            || !header.slice_deblocking_filter_disabled_flag)
    {
        out.put(
            "slice_loop_filter_across_slices_enabled_flag",
            header.slice_loop_filter_across_slices_enabled_flag,
        );
    }
}

fn long_term_elements(out: &mut Elements, header: &SliceSegmentHeader, sps: &SeqParameterSet) {
    if sps.num_long_term_ref_pics_sps > 0 {
        out.put("num_long_term_sps", header.num_long_term_sps);
    }
    out.put("num_long_term_pics", header.num_long_term_pics);
    let from_sps = header.num_long_term_sps as usize;
    for i in 0..from_sps + header.num_long_term_pics as usize {
        if i < from_sps {
            if sps.num_long_term_ref_pics_sps > 1 {
                out.put(format!("lt_idx_sps[{i}]"), header.lt_idx_sps[i]);
            }
        } else {
            out.put(format!("poc_lsb_lt[{i}]"), header.poc_lsb_lt[i]);
            out.put(
                format!("used_by_curr_pic_lt_flag[{i}]"),
                header.used_by_curr_pic_lt[i],
            );
        }
        out.put(
            format!("delta_poc_msb_present_flag[{i}]"),
            header.delta_poc_msb_present_flag[i],
        );
        if header.delta_poc_msb_present_flag[i] {
            out.put(
                format!("delta_poc_msb_cycle_lt[{i}]"),
                header.delta_poc_msb_cycle_lt[i],
            );
        }
    }
}

fn inter_elements(
    out: &mut Elements,
    header: &SliceSegmentHeader,
    pps: &PicParameterSet,
    sps: &SeqParameterSet,
) {
    let is_b = header.slice_type == SLICE_TYPE_B;
    let l0 = header.num_ref_idx_l0_active_minus1 as usize + 1;
    let l1 = header.num_ref_idx_l1_active_minus1 as usize + 1;
    out.put(
        "num_ref_idx_active_override_flag",
        header.num_ref_idx_active_override_flag,
    );
    if header.num_ref_idx_active_override_flag {
        out.put(
            "num_ref_idx_l0_active_minus1",
            header.num_ref_idx_l0_active_minus1,
        );
        if is_b {
            out.put(
                "num_ref_idx_l1_active_minus1",
                header.num_ref_idx_l1_active_minus1,
            );
        }
    }
    if pps.lists_modification_present_flag && header.num_pic_total_curr > 1 {
        out.put(
            "ref_pic_list_modification_flag_l0",
            header.ref_pic_list_modification_flag_l0,
        );
        if header.ref_pic_list_modification_flag_l0 {
            for (i, entry) in header.list_entry_l0.iter().enumerate().take(l0) {
                out.put(format!("list_entry_l0[{i}]"), *entry);
            }
        }
        if is_b {
            out.put(
                "ref_pic_list_modification_flag_l1",
                header.ref_pic_list_modification_flag_l1,
            );
            if header.ref_pic_list_modification_flag_l1 {
                for (i, entry) in header.list_entry_l1.iter().enumerate().take(l1) {
                    out.put(format!("list_entry_l1[{i}]"), *entry);
                }
            }
        }
    }
    if is_b {
        out.put("mvd_l1_zero_flag", header.mvd_l1_zero_flag);
    }
    if pps.cabac_init_present_flag {
        out.put("cabac_init_flag", header.cabac_init_flag);
    }
    if header.slice_temporal_mvp_enabled_flag {
        if is_b {
            out.put("collocated_from_l0_flag", header.collocated_from_l0_flag);
        }
        let last = if header.collocated_from_l0_flag {
            l0
        } else {
            l1
        };
        if last > 1 {
            out.put("collocated_ref_idx", header.collocated_ref_idx);
        }
    }
    if let Some(table) = &header.pred_weight_table {
        let chroma = sps.chroma_array_type() != 0;
        out.put("luma_log2_weight_denom", table.luma_log2_weight_denom);
        if chroma {
            out.put(
                "delta_chroma_log2_weight_denom",
                table.delta_chroma_log2_weight_denom,
            );
        }
        weight_elements(out, &table.l0, l0, "l0", chroma);
        if is_b {
            weight_elements(out, &table.l1, l1, "l1", chroma);
        }
    }
    out.put(
        "five_minus_max_num_merge_cand",
        header.five_minus_max_num_merge_cand,
    );
}

const STREAMS: &[&str] = &[
    "headers_synth_full",
    "headers_synth_vps",
    "headers_synth_mono",
    "headers_synth_planes",
    "headers_synth_444_scaling",
    "headers_synth_high_precision",
    "headers_synth_short_poc",
    "headers_x265_default",
    "headers_x265_wpp",
    "headers_x265_weighted",
    "headers_x265_scaling_default",
    "headers_x265_scaling_file",
    "headers_x265_amp",
    "headers_x265_refs",
    "headers_x265_temporal",
    "headers_x265_slices",
    "headers_x265_hrd",
    "headers_x265_lossless",
    "headers_x265_qp_deblock",
    "headers_x265_10bit",
    "headers_x265_422",
];

/// The active parameter sets and slice state while a stream runs.
#[derive(Default)]
struct Stream {
    sps: BTreeMap<u32, SeqParameterSet>,
    pps: BTreeMap<u32, PicParameterSet>,
    independent: Option<SliceSegmentHeader>,
}

impl Stream {
    fn run(&mut self, label: &str, section: &Section, constraint_names: &[String]) {
        let names: Vec<&str> = constraint_names.iter().map(String::as_str).collect();
        let lists: [&[&str]; 6] = [
            IGNORED_FRAMING,
            IGNORED_HRD,
            IGNORED_VUI,
            IGNORED_VPS,
            IGNORED_SLICE,
            &names,
        ];
        let oracle = &section.elements;
        match section.kind.as_str() {
            "vps" => {
                let vps = VideoParameterSet::parse(&section.raw)
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
                compare(label, oracle, &lists, vps_elements(&vps));
                check_ptl_flags(
                    label,
                    oracle,
                    &vps.profile_tier_level,
                    usize::from(vps.vps_max_sub_layers_minus1),
                );
            }
            "sps" => {
                let sps = SeqParameterSet::parse(&section.raw)
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
                compare(label, oracle, &lists, sps_elements(&sps));
                check_ptl_flags(
                    label,
                    oracle,
                    &sps.profile_tier_level,
                    usize::from(sps.sps_max_sub_layers_minus1),
                );
                self.sps.insert(sps.sps_seq_parameter_set_id, sps);
            }
            "pps" => {
                let pps = PicParameterSet::parse(&section.raw)
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
                compare(label, oracle, &lists, pps_elements(&pps));
                self.pps.insert(pps.pps_pic_parameter_set_id, pps);
            }
            "slice" => self.run_slice(label, section, &lists),
            other => panic!("{label}: unknown section kind `{other}`"),
        }
    }

    fn run_slice(&mut self, label: &str, section: &Section, lists: &[&[&str]]) {
        let id = slice_pic_parameter_set_id(&section.raw)
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        let pps = self
            .pps
            .get(&id)
            .unwrap_or_else(|| panic!("{label}: no PPS {id}"));
        let sps = self
            .sps
            .get(&pps.pps_seq_parameter_set_id)
            .unwrap_or_else(|| panic!("{label}: no SPS {}", pps.pps_seq_parameter_set_id));
        let header = SliceSegmentHeader::parse(&section.raw, pps, sps, self.independent.as_ref())
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        compare(
            label,
            &section.elements,
            lists,
            slice_elements(&header, pps, sps),
        );
        if !header.dependent_slice_segment_flag {
            self.independent = Some(header);
        }
    }

    /// Runs every cut and flips a byte, and needs only the absence of a panic.
    fn stress(&self, section: &Section) {
        let cuts = section.raw.len().min(160);
        let parse = |bytes: &[u8]| match section.kind.as_str() {
            "vps" => drop(VideoParameterSet::parse(bytes)),
            "sps" => drop(SeqParameterSet::parse(bytes)),
            "pps" => drop(PicParameterSet::parse(bytes)),
            _ => {
                if let Ok(id) = slice_pic_parameter_set_id(bytes)
                    && let Some(pps) = self.pps.get(&id)
                    && let Some(sps) = self.sps.get(&pps.pps_seq_parameter_set_id)
                {
                    drop(SliceSegmentHeader::parse(
                        bytes,
                        pps,
                        sps,
                        self.independent.as_ref(),
                    ));
                }
            }
        };
        for end in 0..cuts {
            parse(&section.raw[..end]);
            let mut flipped = section.raw[..cuts].to_vec();
            flipped[end] ^= 0x55;
            parse(&flipped);
        }
    }
}

#[test]
fn hevc_headers_match_trace_headers() {
    let constraint_names = ignored_constraint_names();
    for stream in STREAMS {
        let Some(sections) = load(stream) else {
            continue;
        };
        let mut state = Stream::default();
        for (index, section) in sections.iter().enumerate() {
            let label = format!("{stream} section {index} ({})", section.kind);
            state.run(&label, section, &constraint_names);
        }
        assert!(!sections.is_empty(), "{stream} holds no sections");
    }
}

#[test]
fn hevc_headers_survive_truncation_and_corruption() {
    for stream in STREAMS {
        let Some(sections) = load(stream) else {
            continue;
        };
        let constraint_names = ignored_constraint_names();
        let mut state = Stream::default();
        for (index, section) in sections.iter().enumerate() {
            let label = format!("{stream} section {index}");
            state.stress(section);
            state.run(&label, section, &constraint_names);
        }
    }
}
