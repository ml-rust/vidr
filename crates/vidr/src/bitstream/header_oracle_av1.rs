//! AV1 sequence and frame headers against `trace_headers`, over whole temporal units, so the
//! state that crosses frames counts too. `frame_header_bits` is the header's length.

use std::collections::VecDeque;

use super::super::av1::frame_header::{
    AFFINE, FrameHeader, IDENTITY, INTER_FRAME, KEY_FRAME, LAST_FRAME, MAX_SEGMENTS,
    PRIMARY_REF_NONE, REFS_PER_FRAME, ROTZOOM, SEG_LVL_MAX, SWITCH_FRAME, SWITCHABLE, TRANSLATION,
    WARPEDMODEL_PREC_BITS,
};
use super::super::av1::seq_header::{
    SELECT_INTEGER_MV, SELECT_SCREEN_CONTENT_TOOLS, SequenceHeader, parse_sequence_header,
};
use super::super::av1::temporal_unit::{Av1Parser, ParsedFrame};
use super::*;

/// Bits with a fixed value, and the codes behind `gm_params`, which the parser does not keep.
const IGNORED: &[&str] = &[
    "trailing_one_bit",
    "trailing_zero_bit",
    "zero_bit",
    "subexp_more_bits",
    "subexp_bits",
    "subexp_final_bits",
];

const STREAMS: &[&str] = &[
    "headers_synth_av1",
    "headers_av1_default",
    "headers_av1_tiles",
    "headers_av1_tile_groups",
    "headers_av1_film_grain",
    "headers_av1_superres",
    "headers_av1_resize",
    "headers_av1_error_resilient",
    "headers_av1_sframe",
    "headers_av1_forward_key",
    "headers_av1_sb128",
    "headers_av1_qm",
    "headers_av1_segmentation",
    "headers_av1_screen",
    "headers_av1_still",
    "headers_av1_no_order_hint",
    "headers_av1_10bit",
    "headers_av1_mono",
    "headers_av1_svt",
    "headers_av1_svt_grain",
    "headers_av1_rav1e",
];

/// `Segmentation_Feature_Bits` restated.
const FEATURE_BITS: [u8; SEG_LVL_MAX] = [8, 6, 6, 6, 6, 3, 0, 0];

fn sequence_elements(header: &SequenceHeader) -> Elements {
    let mut out = Elements::default();
    out.put("seq_profile", header.seq_profile);
    out.put("still_picture", header.still_picture);
    out.put(
        "reduced_still_picture_header",
        header.reduced_still_picture_header,
    );
    if header.reduced_still_picture_header {
        out.put("seq_level_idx[0]", header.operating_points[0].seq_level_idx);
    } else {
        timing_elements(&mut out, header);
        operating_point_elements(&mut out, header);
    }
    out.put("frame_width_bits_minus_1", header.frame_width_bits_minus_1);
    out.put(
        "frame_height_bits_minus_1",
        header.frame_height_bits_minus_1,
    );
    out.put("max_frame_width_minus_1", header.max_frame_width_minus_1);
    out.put("max_frame_height_minus_1", header.max_frame_height_minus_1);
    if !header.reduced_still_picture_header {
        out.put(
            "frame_id_numbers_present_flag",
            header.frame_id_numbers_present_flag,
        );
    }
    if header.frame_id_numbers_present_flag {
        out.put(
            "delta_frame_id_length_minus_2",
            header.delta_frame_id_length_minus_2,
        );
        out.put(
            "additional_frame_id_length_minus_1",
            header.additional_frame_id_length_minus_1,
        );
    }
    out.put("use_128x128_superblock", header.use_128x128_superblock);
    out.put("enable_filter_intra", header.enable_filter_intra);
    out.put("enable_intra_edge_filter", header.enable_intra_edge_filter);
    if !header.reduced_still_picture_header {
        tool_elements(&mut out, header);
    }
    out.put("enable_superres", header.enable_superres);
    out.put("enable_cdef", header.enable_cdef);
    out.put("enable_restoration", header.enable_restoration);
    color_elements(&mut out, header);
    out.put(
        "film_grain_params_present",
        header.film_grain_params_present,
    );
    out
}

fn timing_elements(out: &mut Elements, header: &SequenceHeader) {
    out.put("timing_info_present_flag", header.timing_info.is_some());
    let Some(timing) = header.timing_info else {
        return;
    };
    out.put(
        "num_units_in_display_tick",
        timing.num_units_in_display_tick,
    );
    out.put("time_scale", timing.time_scale);
    out.put("equal_picture_interval", timing.equal_picture_interval);
    if timing.equal_picture_interval {
        out.put(
            "num_ticks_per_picture_minus_1",
            timing.num_ticks_per_picture_minus_1,
        );
    }
    out.put(
        "decoder_model_info_present_flag",
        header.decoder_model_info.is_some(),
    );
    if let Some(model) = header.decoder_model_info {
        out.put(
            "buffer_delay_length_minus_1",
            model.buffer_delay_length_minus_1,
        );
        out.put(
            "num_units_in_decoding_tick",
            model.num_units_in_decoding_tick,
        );
        out.put(
            "buffer_removal_time_length_minus_1",
            model.buffer_removal_time_length_minus_1,
        );
        out.put(
            "frame_presentation_time_length_minus_1",
            model.frame_presentation_time_length_minus_1,
        );
    }
}

fn operating_point_elements(out: &mut Elements, header: &SequenceHeader) {
    out.put(
        "initial_display_delay_present_flag",
        header.initial_display_delay_present_flag,
    );
    out.put(
        "operating_points_cnt_minus_1",
        (header.operating_points.len() - 1) as u32,
    );
    for (index, point) in header.operating_points.iter().enumerate() {
        out.put(format!("operating_point_idc[{index}]"), point.idc);
        out.put(format!("seq_level_idx[{index}]"), point.seq_level_idx);
        if point.seq_level_idx > 7 {
            out.put(format!("seq_tier[{index}]"), point.seq_tier);
        }
        if header.decoder_model_info.is_some() {
            out.put(
                format!("decoder_model_present_for_this_op[{index}]"),
                point.decoder_model_present,
            );
            if point.decoder_model_present {
                out.put(
                    format!("decoder_buffer_delay[{index}]"),
                    point.decoder_buffer_delay,
                );
                out.put(
                    format!("encoder_buffer_delay[{index}]"),
                    point.encoder_buffer_delay,
                );
                out.put(
                    format!("low_delay_mode_flag[{index}]"),
                    point.low_delay_mode_flag,
                );
            }
        }
        if header.initial_display_delay_present_flag {
            out.put(
                format!("initial_display_delay_present_for_this_op[{index}]"),
                point.initial_display_delay_present,
            );
            if point.initial_display_delay_present {
                out.put(
                    format!("initial_display_delay_minus_1[{index}]"),
                    point.initial_display_delay_minus_1,
                );
            }
        }
    }
}

fn tool_elements(out: &mut Elements, header: &SequenceHeader) {
    out.put(
        "enable_interintra_compound",
        header.enable_interintra_compound,
    );
    out.put("enable_masked_compound", header.enable_masked_compound);
    out.put("enable_warped_motion", header.enable_warped_motion);
    out.put("enable_dual_filter", header.enable_dual_filter);
    out.put("enable_order_hint", header.enable_order_hint);
    if header.enable_order_hint {
        out.put("enable_jnt_comp", header.enable_jnt_comp);
        out.put("enable_ref_frame_mvs", header.enable_ref_frame_mvs);
    }
    out.put(
        "seq_choose_screen_content_tools",
        header.seq_choose_screen_content_tools,
    );
    if !header.seq_choose_screen_content_tools {
        out.put(
            "seq_force_screen_content_tools",
            header.seq_force_screen_content_tools,
        );
    }
    if header.seq_force_screen_content_tools > 0 {
        out.put("seq_choose_integer_mv", header.seq_choose_integer_mv);
        if !header.seq_choose_integer_mv {
            out.put("seq_force_integer_mv", header.seq_force_integer_mv);
        }
    }
    if header.enable_order_hint {
        out.put("order_hint_bits_minus_1", header.order_hint_bits - 1);
    }
}

fn color_elements(out: &mut Elements, header: &SequenceHeader) {
    let color = &header.color;
    out.put("high_bitdepth", color.high_bitdepth);
    if header.seq_profile == 2 && color.high_bitdepth {
        out.put("twelve_bit", color.twelve_bit);
    }
    if header.seq_profile != 1 {
        out.put("mono_chrome", color.mono_chrome);
    }
    out.put(
        "color_description_present_flag",
        color.color_description_present_flag,
    );
    if color.color_description_present_flag {
        out.put("color_primaries", color.color_primaries);
        out.put("transfer_characteristics", color.transfer_characteristics);
        out.put("matrix_coefficients", color.matrix_coefficients);
    }
    if color.mono_chrome {
        out.put("color_range", color.color_range);
        return;
    }
    let srgb = (
        color.color_primaries,
        color.transfer_characteristics,
        color.matrix_coefficients,
    ) == (1, 13, 0);
    if !srgb {
        out.put("color_range", color.color_range);
        if header.seq_profile == 2 && color.twelve_bit {
            out.put("subsampling_x", color.subsampling_x);
            if color.subsampling_x {
                out.put("subsampling_y", color.subsampling_y);
            }
        }
        if color.subsampling_x && color.subsampling_y {
            out.put("chroma_sample_position", color.chroma_sample_position);
        }
    }
    out.put("separate_uv_delta_q", color.separate_uv_delta_q);
}

fn frame_kind_elements(out: &mut Elements, sequence: &SequenceHeader, header: &FrameHeader) {
    if !sequence.reduced_still_picture_header {
        out.put("show_existing_frame", false);
        out.put("frame_type", header.frame_type);
        out.put("show_frame", header.show_frame);
        if let Some(time) = header.frame_presentation_time {
            out.put("frame_presentation_time", time);
        }
        if !header.show_frame {
            out.put("showable_frame", header.showable_frame);
        }
        let key_shown = header.frame_type == KEY_FRAME && header.show_frame;
        if header.frame_type != SWITCH_FRAME && !key_shown {
            out.put("error_resilient_mode", header.error_resilient_mode);
        }
    }
    out.put("disable_cdf_update", header.disable_cdf_update);
    if sequence.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS {
        out.put(
            "allow_screen_content_tools",
            header.allow_screen_content_tools,
        );
    }
    if header.allow_screen_content_tools && sequence.seq_force_integer_mv == SELECT_INTEGER_MV {
        out.put("force_integer_mv", header.coded_force_integer_mv);
    }
    if sequence.frame_id_numbers_present_flag {
        out.put("current_frame_id", header.current_frame_id);
    }
    if header.frame_type != SWITCH_FRAME && !sequence.reduced_still_picture_header {
        out.put("frame_size_override_flag", header.frame_size_override_flag);
    }
    if sequence.enable_order_hint {
        out.put("order_hint", header.order_hint);
    }
    if !header.frame_is_intra && !header.error_resilient_mode {
        out.put("primary_ref_frame", header.primary_ref_frame);
    }
    if sequence.decoder_model_info.is_some() {
        out.put(
            "buffer_removal_time_present_flag",
            header.buffer_removal_time_present_flag,
        );
        for (point, time) in &header.buffer_removal_times {
            out.put(format!("buffer_removal_time[{point}]"), *time);
        }
    }
    let key_shown = header.frame_type == KEY_FRAME && header.show_frame;
    if header.frame_type != SWITCH_FRAME && !key_shown {
        out.put("refresh_frame_flags", header.refresh_frame_flags);
    }
    if let Some(hints) = header.ref_order_hint {
        for (index, hint) in hints.into_iter().enumerate() {
            out.put(format!("ref_order_hint[{index}]"), hint);
        }
    }
}

fn size_elements(out: &mut Elements, sequence: &SequenceHeader, header: &FrameHeader) {
    if let Some((count, found)) = header.found_ref {
        for index in 0..count {
            out.put(format!("found_ref[{index}]"), found && index + 1 == count);
        }
    }
    if header.frame_size_coded {
        out.put("frame_width_minus_1", header.upscaled_width - 1);
        out.put("frame_height_minus_1", header.frame_height - 1);
    }
    if sequence.enable_superres {
        out.put("use_superres", header.use_superres);
    }
    if header.use_superres {
        out.put("coded_denom", header.coded_denom);
    }
    let from_reference = header.found_ref.is_some_and(|(_, found)| found);
    if !from_reference {
        out.put(
            "render_and_frame_size_different",
            header.render_and_frame_size_different,
        );
        if header.render_and_frame_size_different {
            out.put("render_width_minus_1", header.render_width - 1);
            out.put("render_height_minus_1", header.render_height - 1);
        }
    }
}

fn inter_elements(out: &mut Elements, sequence: &SequenceHeader, header: &FrameHeader) {
    if sequence.enable_order_hint {
        out.put(
            "frame_refs_short_signaling",
            header.frame_refs_short_signaling,
        );
    }
    if header.frame_refs_short_signaling {
        out.put("last_frame_idx", header.last_frame_idx);
        out.put("golden_frame_idx", header.gold_frame_idx);
    }
    for index in 0..REFS_PER_FRAME {
        if !header.frame_refs_short_signaling {
            out.put(
                format!("ref_frame_idx[{index}]"),
                header.ref_frame_idx[index],
            );
        }
        if sequence.frame_id_numbers_present_flag {
            out.put(
                format!("delta_frame_id_minus1[{index}]"),
                header.delta_frame_id_minus_1[index],
            );
        }
    }
    size_elements(out, sequence, header);
    if !header.force_integer_mv {
        out.put("allow_high_precision_mv", header.allow_high_precision_mv);
    }
    out.put("is_filter_switchable", header.is_filter_switchable);
    if header.interpolation_filter != SWITCHABLE {
        out.put("interpolation_filter", header.interpolation_filter);
    }
    out.put(
        "is_motion_mode_switchable",
        header.is_motion_mode_switchable,
    );
    if !header.error_resilient_mode && sequence.enable_ref_frame_mvs {
        out.put("use_ref_frame_mvs", header.use_ref_frame_mvs);
    }
}

fn tile_elements(out: &mut Elements, header: &FrameHeader) {
    let info = &header.tile_info;
    out.put("uniform_tile_spacing_flag", info.uniform_tile_spacing_flag);
    if info.uniform_tile_spacing_flag {
        out.put("tile_cols_log2", info.tile_cols_log2);
        out.put("tile_rows_log2", info.tile_rows_log2);
    } else {
        for (index, width) in info.width_in_sbs_minus_1.iter().enumerate() {
            out.put(format!("width_in_sbs_minus_1[{index}]"), *width);
        }
        for (index, height) in info.height_in_sbs_minus_1.iter().enumerate() {
            out.put(format!("height_in_sbs_minus_1[{index}]"), *height);
        }
    }
    if info.tile_cols_log2 > 0 || info.tile_rows_log2 > 0 {
        out.put("context_update_tile_id", info.context_update_tile_id);
        out.put("tile_size_bytes_minus1", info.tile_size_bytes_minus_1);
    }
}

fn quant_elements(out: &mut Elements, sequence: &SequenceHeader, header: &FrameHeader) {
    let q = &header.quantization;
    out.put("base_q_idx", q.base_q_idx);
    let mut delta = |name: &str, index: usize, value: i8| {
        out.put(format!("{name}.delta_coded"), q.delta_coded[index]);
        if q.delta_coded[index] {
            out.put(format!("{name}.delta_q"), value);
        }
    };
    delta("delta_q_y_dc", 0, q.delta_q_y_dc);
    if !sequence.color.mono_chrome {
        delta("delta_q_u_dc", 1, q.delta_q_u_dc);
        delta("delta_q_u_ac", 2, q.delta_q_u_ac);
        if q.diff_uv_delta {
            delta("delta_q_v_dc", 3, q.delta_q_v_dc);
            delta("delta_q_v_ac", 4, q.delta_q_v_ac);
        }
        if sequence.color.separate_uv_delta_q {
            out.put("diff_uv_delta", q.diff_uv_delta);
        }
    }
    out.put("using_qmatrix", q.using_qmatrix);
    if q.using_qmatrix {
        out.put("qm_y", q.qm_y);
        out.put("qm_u", q.qm_u);
        if sequence.color.separate_uv_delta_q {
            out.put("qm_v", q.qm_v);
        }
    }
}

fn segmentation_elements(out: &mut Elements, header: &FrameHeader) {
    let seg = &header.segmentation;
    out.put("segmentation_enabled", seg.enabled);
    if !seg.enabled {
        return;
    }
    if header.primary_ref_frame != PRIMARY_REF_NONE {
        out.put("segmentation_update_map", seg.update_map);
        if seg.update_map {
            out.put("segmentation_temporal_update", seg.temporal_update);
        }
        out.put("segmentation_update_data", seg.update_data);
    }
    if !seg.update_data {
        return;
    }
    for segment in 0..MAX_SEGMENTS {
        for (feature, &bits) in FEATURE_BITS.iter().enumerate().take(SEG_LVL_MAX) {
            let enabled = seg.feature_enabled[segment][feature];
            out.put(format!("feature_enabled[{segment}][{feature}]"), enabled);
            if enabled && bits > 0 {
                out.put(
                    format!("feature_value[{segment}][{feature}]"),
                    seg.feature_data[segment][feature],
                );
            }
        }
    }
}

fn delta_elements(out: &mut Elements, header: &FrameHeader) {
    if header.quantization.base_q_idx > 0 {
        out.put("delta_q_present", header.delta_q_present);
    }
    if header.delta_q_present {
        out.put("delta_q_res", header.delta_q_res);
        if !header.allow_intrabc {
            out.put("delta_lf_present", header.delta_lf_present);
        }
        if header.delta_lf_present {
            out.put("delta_lf_res", header.delta_lf_res);
            out.put("delta_lf_multi", header.delta_lf_multi);
        }
    }
}

fn loop_filter_elements(out: &mut Elements, sequence: &SequenceHeader, header: &FrameHeader) {
    if header.coded_lossless || header.allow_intrabc {
        return;
    }
    let filter = &header.loop_filter;
    out.put("loop_filter_level[0]", filter.level[0]);
    out.put("loop_filter_level[1]", filter.level[1]);
    if !sequence.color.mono_chrome && (filter.level[0] != 0 || filter.level[1] != 0) {
        out.put("loop_filter_level[2]", filter.level[2]);
        out.put("loop_filter_level[3]", filter.level[3]);
    }
    out.put("loop_filter_sharpness", filter.sharpness);
    out.put("loop_filter_delta_enabled", filter.delta_enabled);
    if !filter.delta_enabled {
        return;
    }
    out.put("loop_filter_delta_update", filter.delta_update);
    if !filter.delta_update {
        return;
    }
    for index in 0..filter.ref_deltas.len() {
        out.put(
            format!("update_ref_delta[{index}]"),
            filter.update_ref_delta[index],
        );
        if filter.update_ref_delta[index] {
            out.put(
                format!("loop_filter_ref_deltas[{index}]"),
                filter.ref_deltas[index],
            );
        }
    }
    for index in 0..filter.mode_deltas.len() {
        out.put(
            format!("update_mode_delta[{index}]"),
            filter.update_mode_delta[index],
        );
        if filter.update_mode_delta[index] {
            out.put(
                format!("loop_filter_mode_deltas[{index}]"),
                filter.mode_deltas[index],
            );
        }
    }
}

fn cdef_lr_elements(out: &mut Elements, sequence: &SequenceHeader, header: &FrameHeader) {
    let planes = sequence.color.num_planes();
    if !header.coded_lossless && !header.allow_intrabc && sequence.enable_cdef {
        let cdef = &header.cdef;
        out.put("cdef_damping_minus_3", cdef.damping_minus_3);
        out.put("cdef_bits", cdef.bits);
        for index in 0..(1_usize << cdef.bits) {
            out.put(
                format!("cdef_y_pri_strength[{index}]"),
                cdef.y_pri_strength[index],
            );
            out.put(
                format!("cdef_y_sec_strength[{index}]"),
                cdef.y_sec_strength[index],
            );
            if planes > 1 {
                out.put(
                    format!("cdef_uv_pri_strength[{index}]"),
                    cdef.uv_pri_strength[index],
                );
                out.put(
                    format!("cdef_uv_sec_strength[{index}]"),
                    cdef.uv_sec_strength[index],
                );
            }
        }
    }
    if header.all_lossless || header.allow_intrabc || !sequence.enable_restoration {
        return;
    }
    let lr = &header.loop_restoration;
    for plane in 0..planes {
        out.put(format!("lr_type[{plane}]"), lr.lr_type[plane]);
    }
    if lr.uses_lr {
        out.put("lr_unit_shift", lr.lr_unit_shift);
        if sequence.color.subsampling_x && sequence.color.subsampling_y && lr.uses_chroma_lr {
            out.put("lr_uv_shift", lr.lr_uv_shift);
        }
    }
}

/// The value `trace_headers` logs for one parameter: the decoded code, before its shift.
fn coded_gm_param(kind: u8, idx: usize, param: i32, high_precision: bool) -> i32 {
    let prec_bits = if idx >= 2 {
        15
    } else if kind == TRANSLATION {
        3 - i32::from(!high_precision)
    } else {
        6
    };
    let round = if idx % 3 == 2 {
        1 << WARPEDMODEL_PREC_BITS
    } else {
        0
    };
    (param - round) >> (WARPEDMODEL_PREC_BITS as i32 - prec_bits)
}

fn global_motion_elements(out: &mut Elements, header: &FrameHeader) {
    if header.frame_is_intra {
        return;
    }
    for reference in LAST_FRAME..LAST_FRAME + REFS_PER_FRAME {
        let kind = header.gm_type[reference];
        out.put(format!("is_global[{reference}]"), kind != IDENTITY);
        if kind == IDENTITY {
            continue;
        }
        out.put(format!("is_rot_zoom[{reference}]"), kind == ROTZOOM);
        if kind != ROTZOOM {
            out.put(format!("is_translation[{reference}]"), kind == TRANSLATION);
        }
        let mut indices = Vec::new();
        if kind >= ROTZOOM {
            indices.extend([2, 3]);
        }
        if kind == AFFINE {
            indices.extend([4, 5]);
        }
        indices.extend([0, 1]);
        for idx in indices {
            let param = header.gm_params[reference][idx];
            let coded = coded_gm_param(kind, idx, param, header.allow_high_precision_mv);
            out.put(format!("gm_params[{reference}][{idx}]"), coded);
        }
    }
}

fn film_grain_elements(out: &mut Elements, sequence: &SequenceHeader, header: &FrameHeader) {
    if !sequence.film_grain_params_present || (!header.show_frame && !header.showable_frame) {
        return;
    }
    let grain = &header.film_grain;
    let loaded = grain.grain_seed != 0 && !grain.update_grain && header.frame_type == INTER_FRAME;
    if !grain.apply_grain && !loaded {
        out.put("apply_grain", false);
        return;
    }
    out.put("apply_grain", true);
    out.put("grain_seed", grain.grain_seed);
    if header.frame_type == INTER_FRAME {
        out.put("update_grain", grain.update_grain);
    }
    if !grain.update_grain {
        out.put("film_grain_params_ref_idx", grain.film_grain_params_ref_idx);
        return;
    }
    let lists = [
        (
            "y",
            grain.num_y_points,
            &grain.point_y_value[..],
            &grain.point_y_scaling[..],
        ),
        (
            "cb",
            grain.num_cb_points,
            &grain.point_cb_value[..],
            &grain.point_cb_scaling[..],
        ),
        (
            "cr",
            grain.num_cr_points,
            &grain.point_cr_value[..],
            &grain.point_cr_scaling[..],
        ),
    ];
    let subsampled = sequence.color.subsampling_x && sequence.color.subsampling_y;
    let mono = sequence.color.mono_chrome;
    let no_chroma_points =
        mono || grain.chroma_scaling_from_luma || (subsampled && grain.num_y_points == 0);
    for (plane, count, values, scalings) in lists {
        if plane != "y" && no_chroma_points {
            continue;
        }
        out.put(format!("num_{plane}_points"), count);
        for index in 0..usize::from(count) {
            out.put(format!("point_{plane}_value[{index}]"), values[index]);
            out.put(format!("point_{plane}_scaling[{index}]"), scalings[index]);
        }
    }
    if !mono {
        out.put("chroma_scaling_from_luma", grain.chroma_scaling_from_luma);
    }
    out.put("grain_scaling_minus_8", grain.grain_scaling_minus_8);
    out.put("ar_coeff_lag", grain.ar_coeff_lag);
    let lag = usize::from(grain.ar_coeff_lag);
    let num_pos_luma = 2 * lag * (lag + 1);
    let num_pos_chroma = num_pos_luma + usize::from(grain.num_y_points > 0);
    if grain.num_y_points > 0 {
        for index in 0..num_pos_luma {
            out.put(
                format!("ar_coeffs_y_plus_128[{index}]"),
                grain.ar_coeffs_y_plus_128[index],
            );
        }
    }
    if grain.chroma_scaling_from_luma || grain.num_cb_points > 0 {
        for index in 0..num_pos_chroma {
            out.put(
                format!("ar_coeffs_cb_plus_128[{index}]"),
                grain.ar_coeffs_cb_plus_128[index],
            );
        }
    }
    if grain.chroma_scaling_from_luma || grain.num_cr_points > 0 {
        for index in 0..num_pos_chroma {
            out.put(
                format!("ar_coeffs_cr_plus_128[{index}]"),
                grain.ar_coeffs_cr_plus_128[index],
            );
        }
    }
    out.put("ar_coeff_shift_minus_6", grain.ar_coeff_shift_minus_6);
    out.put("grain_scale_shift", grain.grain_scale_shift);
    if grain.num_cb_points > 0 {
        out.put("cb_mult", grain.cb_mult);
        out.put("cb_luma_mult", grain.cb_luma_mult);
        out.put("cb_offset", grain.cb_offset);
    }
    if grain.num_cr_points > 0 {
        out.put("cr_mult", grain.cr_mult);
        out.put("cr_luma_mult", grain.cr_luma_mult);
        out.put("cr_offset", grain.cr_offset);
    }
    out.put("overlap_flag", grain.overlap_flag);
    out.put("clip_to_restricted_range", grain.clip_to_restricted_range);
}

fn frame_elements(sequence: &SequenceHeader, header: &FrameHeader) -> Elements {
    let mut out = Elements::default();
    out.put("frame_header_bits", header.header_bits as i64);
    if header.show_existing_frame {
        out.put("show_existing_frame", true);
        out.put("frame_to_show_map_idx", header.frame_to_show_map_idx);
        if let Some(time) = header.frame_presentation_time {
            out.put("frame_presentation_time", time);
        }
        if sequence.frame_id_numbers_present_flag {
            out.put("display_frame_id", header.display_frame_id);
        }
        return out;
    }
    frame_kind_elements(&mut out, sequence, header);
    if header.frame_is_intra {
        size_elements(&mut out, sequence, header);
        let can_read =
            header.allow_screen_content_tools && header.upscaled_width == header.frame_width;
        if can_read {
            out.put("allow_intrabc", header.allow_intrabc);
        }
    } else {
        inter_elements(&mut out, sequence, header);
    }
    if !sequence.reduced_still_picture_header && !header.disable_cdf_update {
        out.put(
            "disable_frame_end_update_cdf",
            header.disable_frame_end_update_cdf,
        );
    }
    tile_elements(&mut out, header);
    quant_elements(&mut out, sequence, header);
    segmentation_elements(&mut out, header);
    delta_elements(&mut out, header);
    loop_filter_elements(&mut out, sequence, header);
    cdef_lr_elements(&mut out, sequence, header);
    if !header.coded_lossless {
        out.put("tx_mode", header.tx_mode);
    }
    if !header.frame_is_intra {
        out.put("reference_select", header.reference_select);
    }
    if header.skip_mode_allowed {
        out.put("skip_mode_present", header.skip_mode_present);
    }
    if !header.frame_is_intra && !header.error_resilient_mode && sequence.enable_warped_motion {
        out.put("allow_warped_motion", header.allow_warped_motion);
    }
    out.put("reduced_tx_set", header.reduced_tx_set);
    global_motion_elements(&mut out, header);
    film_grain_elements(&mut out, sequence, header);
    out
}

/// The frame a section compares, with the index of its next tile group.
struct Current<'a> {
    frame: ParsedFrame<'a>,
    group: usize,
}

fn check_stream(name: &str) -> Option<usize> {
    let sections = load(name)?;
    let mut parser = Av1Parser::default();
    let mut frames: VecDeque<ParsedFrame<'_>> = VecDeque::new();
    let mut current: Option<Current<'_>> = None;
    let mut compared = 0;
    for (index, section) in sections.iter().enumerate() {
        let label = format!("{name} section {index} ({})", section.kind);
        match section.kind.as_str() {
            "av1_unit" => {
                let parsed = parser
                    .parse_packet(&section.raw)
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
                frames.extend(parsed);
            }
            "av1_seq" => {
                let header = parse_sequence_header(&section.raw)
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
                compare(
                    &label,
                    &section.elements,
                    &[IGNORED],
                    sequence_elements(&header),
                );
                compared += 1;
            }
            "av1_frame" => {
                let frame = frames
                    .pop_front()
                    .unwrap_or_else(|| panic!("{label}: the parser gives fewer frames"));
                let emitted = frame_elements(&frame.sequence, &frame.header);
                compare(&label, &section.elements, &[IGNORED], emitted);
                current = Some(Current { frame, group: 0 });
                compared += 1;
            }
            "av1_group" => {
                let Some(open) = &mut current else {
                    panic!("{label}: a tile group before any frame header");
                };
                let facts = open.frame.groups[open.group];
                open.group += 1;
                let num_tiles = open.frame.header.tile_info.num_tiles();
                let mut emitted = Elements::default();
                if num_tiles > 1 {
                    emitted.put(
                        "tile_start_and_end_present_flag",
                        facts.start_and_end_present,
                    );
                }
                if facts.start_and_end_present {
                    emitted.put("tg_start", facts.tg_start);
                    emitted.put("tg_end", facts.tg_end);
                }
                compare(&label, &section.elements, &[IGNORED], emitted);
            }
            other => panic!("{label}: section kind `{other}`"),
        }
    }
    assert!(frames.is_empty(), "{name}: the parser gives more frames");
    Some(compared)
}

#[test]
fn headers_match_trace_headers() {
    for name in STREAMS {
        if let Some(compared) = check_stream(name) {
            assert!(compared > 0, "{name}: no header section");
        }
    }
}

/// The parser walks each generated stream cut at every byte of its first units. A cut gives a
/// named error or fewer frames, and never a panic.
#[test]
fn a_real_stream_cut_at_every_byte_never_panics() {
    for name in STREAMS {
        let Some(sections) = load(name) else {
            continue;
        };
        let mut parser = Av1Parser::default();
        for section in sections
            .iter()
            .filter(|section| section.kind == "av1_unit")
            .take(4)
        {
            for end in 0..section.raw.len().min(4096) {
                let mut cut = parser.clone();
                let _ = cut.parse_packet(&section.raw[..end]);
            }
            parser
                .parse_packet(&section.raw)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
        }
    }
}

/// `gm_params` round trips through what `trace_headers` logs.
#[test]
fn the_logged_gm_param_undoes_the_shift_and_the_round() {
    assert_eq!(coded_gm_param(ROTZOOM, 2, (1 << 16) + 80, true), 40);
    assert_eq!(coded_gm_param(ROTZOOM, 3, -80, true), -40);
    assert_eq!(coded_gm_param(ROTZOOM, 0, 1 << 10, true), 1);
    assert_eq!(coded_gm_param(TRANSLATION, 0, 1 << 13, true), 1);
    assert_eq!(coded_gm_param(TRANSLATION, 1, 1 << 14, false), 1);
}
