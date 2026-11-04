//! VP9 uncompressed headers against `trace_headers`.

use super::super::vp9::split_superframe;
use super::super::vp9_header::{
    KEY_FRAME, MAX_SEGMENTS, RefFrameTable, SEG_LVL_MAX, UncompressedHeader, Vp9HeaderParser,
};
use super::*;

/// Bits with a fixed value that the parser checks but does not keep.
const IGNORED: &[&str] = &[
    "frame_marker",
    "frame_sync_byte_0",
    "frame_sync_byte_1",
    "frame_sync_byte_2",
    "reserved_zero",
    "zero_bit",
];

const STREAMS: &[&str] = &[
    "headers_synth_vp9_p0",
    "headers_synth_vp9_p1",
    "headers_synth_vp9_p2",
    "headers_synth_vp9_p3",
    "headers_synth_vp9_resilient",
    "headers_vp9_default",
    "headers_vp9_tiles",
    "headers_vp9_segmentation",
    "headers_vp9_hidden",
    "headers_vp9_resize",
    "headers_vp9_error_resilient",
];

/// `segmentation_feature_bits` and `segmentation_feature_signed` restated.
const FEATURES: [(bool, bool); SEG_LVL_MAX] =
    [(true, true), (true, true), (true, false), (false, false)];

fn color_elements(out: &mut Elements, header: &UncompressedHeader) {
    let color = &header.color_config;
    if header.profile >= 2 {
        out.put("ten_or_twelve_bit", color.bit_depth == 12);
    }
    out.put("color_space", color.color_space);
    if color.color_space != 7 {
        out.put("color_range", color.color_range);
        if header.profile & 1 == 1 {
            out.put("subsampling_x", color.subsampling_x);
            out.put("subsampling_y", color.subsampling_y);
        }
    }
}

fn size_elements(out: &mut Elements, header: &UncompressedHeader) {
    if header.found_ref.is_none() {
        out.put("frame_width_minus_1", header.frame_width - 1);
        out.put("frame_height_minus_1", header.frame_height - 1);
    }
}

fn render_elements(out: &mut Elements, header: &UncompressedHeader) {
    out.put(
        "render_and_frame_size_different",
        header.render_and_frame_size_different,
    );
    if header.render_and_frame_size_different {
        out.put("render_width_minus_1", header.render_width - 1);
        out.put("render_height_minus_1", header.render_height - 1);
    }
}

fn loop_filter_elements(out: &mut Elements, header: &UncompressedHeader) {
    let filter = &header.loop_filter;
    out.put("loop_filter_level", filter.loop_filter_level);
    out.put("loop_filter_sharpness", filter.loop_filter_sharpness);
    out.put(
        "loop_filter_delta_enabled",
        filter.loop_filter_delta_enabled,
    );
    if !filter.loop_filter_delta_enabled {
        return;
    }
    out.put("loop_filter_delta_update", filter.loop_filter_delta_update);
    if !filter.loop_filter_delta_update {
        return;
    }
    for i in 0..4 {
        out.put(format!("update_ref_delta[{i}]"), filter.update_ref_delta[i]);
        if filter.update_ref_delta[i] {
            out.put(
                format!("loop_filter_ref_deltas[{i}]"),
                filter.loop_filter_ref_deltas[i],
            );
        }
    }
    for i in 0..2 {
        out.put(
            format!("update_mode_delta[{i}]"),
            filter.update_mode_delta[i],
        );
        if filter.update_mode_delta[i] {
            out.put(
                format!("loop_filter_mode_deltas[{i}]"),
                filter.loop_filter_mode_deltas[i],
            );
        }
    }
}

fn quantization_elements(out: &mut Elements, header: &UncompressedHeader) {
    let q = &header.quantization;
    out.put("base_q_idx", q.base_q_idx);
    let deltas = [
        ("delta_q_y_dc", q.delta_q_y_dc),
        ("delta_q_uv_dc", q.delta_q_uv_dc),
        ("delta_q_uv_ac", q.delta_q_uv_ac),
    ];
    for (index, (name, delta)) in deltas.into_iter().enumerate() {
        out.put(format!("{name}.delta_coded"), q.delta_coded[index]);
        if q.delta_coded[index] {
            out.put(format!("{name}.delta_q"), delta);
        }
    }
}

fn segmentation_elements(out: &mut Elements, header: &UncompressedHeader) {
    let params = &header.segmentation;
    out.put("segmentation_enabled", params.segmentation_enabled);
    if !params.segmentation_enabled {
        return;
    }
    out.put("segmentation_update_map", params.segmentation_update_map);
    if params.segmentation_update_map {
        for i in 0..7 {
            let coded = params.segmentation_tree_prob_coded[i];
            out.put(format!("segmentation_tree_probs[{i}].prob_coded"), coded);
            if coded {
                out.put(
                    format!("segmentation_tree_probs[{i}].prob"),
                    params.segmentation_tree_probs[i],
                );
            }
        }
        out.put(
            "segmentation_temporal_update",
            params.segmentation_temporal_update,
        );
        if params.segmentation_temporal_update {
            for i in 0..3 {
                let coded = params.segmentation_pred_prob_coded[i];
                out.put(format!("segmentation_pred_prob[{i}].prob_coded"), coded);
                if coded {
                    out.put(
                        format!("segmentation_pred_prob[{i}].prob"),
                        params.segmentation_pred_prob[i],
                    );
                }
            }
        }
    }
    out.put("segmentation_update_data", params.segmentation_update_data);
    if !params.segmentation_update_data {
        return;
    }
    out.put(
        "segmentation_abs_or_delta_update",
        params.segmentation_abs_or_delta_update,
    );
    for i in 0..MAX_SEGMENTS {
        for (j, (has_value, signed)) in FEATURES.into_iter().enumerate() {
            let enabled = params.feature_enabled[i][j];
            out.put(format!("feature_enabled[{i}][{j}]"), enabled);
            if enabled && has_value {
                out.put(
                    format!("feature_value[{i}][{j}]"),
                    params.feature_data[i][j].abs(),
                );
            }
            if enabled && signed {
                out.put(format!("feature_sign[{i}][{j}]"), params.feature_sign[i][j]);
            }
        }
    }
}

fn header_elements(header: &UncompressedHeader) -> Elements {
    let mut out = Elements::default();
    out.put("profile_low_bit", header.profile & 1);
    out.put("profile_high_bit", header.profile >> 1);
    out.put("show_existing_frame", header.show_existing_frame);
    if header.show_existing_frame {
        out.put("frame_to_show_map_idx", header.frame_to_show_map_idx);
        return out;
    }
    out.put("frame_type", header.frame_type);
    out.put("show_frame", header.show_frame);
    out.put("error_resilient_mode", header.error_resilient_mode);
    if header.frame_type == KEY_FRAME {
        color_elements(&mut out, header);
        size_elements(&mut out, header);
        render_elements(&mut out, header);
    } else {
        if !header.show_frame {
            out.put("intra_only", header.intra_only);
        }
        if !header.error_resilient_mode {
            out.put("reset_frame_context", header.reset_frame_context);
        }
        if header.intra_only {
            if header.profile > 0 {
                color_elements(&mut out, header);
            }
            out.put("refresh_frame_flags", header.refresh_frame_flags);
            size_elements(&mut out, header);
            render_elements(&mut out, header);
        } else {
            out.put("refresh_frame_flags", header.refresh_frame_flags);
            for i in 0..3 {
                out.put(format!("ref_frame_idx[{i}]"), header.ref_frame_idx[i]);
                out.put(
                    format!("ref_frame_sign_bias[{}]", i + 1),
                    header.ref_frame_sign_bias[i + 1],
                );
            }
            for i in 0..3u8 {
                let found = header.found_ref == Some(i);
                out.put(format!("found_ref[{i}]"), found);
                if found {
                    break;
                }
            }
            size_elements(&mut out, header);
            render_elements(&mut out, header);
            out.put("allow_high_precision_mv", header.allow_high_precision_mv);
            out.put("is_filter_switchable", header.is_filter_switchable);
            if !header.is_filter_switchable {
                out.put(
                    "raw_interpolation_filter_type",
                    header.raw_interpolation_filter,
                );
            }
        }
    }
    if !header.error_resilient_mode {
        out.put("refresh_frame_context", header.refresh_frame_context);
        out.put(
            "frame_parallel_decoding_mode",
            header.frame_parallel_decoding_mode,
        );
    }
    out.put("frame_context_idx", header.coded_frame_context_idx);
    loop_filter_elements(&mut out, header);
    quantization_elements(&mut out, header);
    segmentation_elements(&mut out, header);
    out.put("tile_cols_log2", header.tile_cols_log2);
    out.put("tile_rows_log2", header.tile_rows_log2);
    out.put("header_size_in_bytes", header.header_size_in_bytes);
    out
}

#[test]
fn vp9_headers_match_trace_headers() {
    for stream in STREAMS {
        let Some(sections) = load(stream) else {
            continue;
        };
        let mut parser = Vp9HeaderParser::new();
        let mut refs = RefFrameTable::default();
        for (index, section) in sections.iter().enumerate() {
            let label = format!("{stream} frame {index}");
            assert_eq!(section.kind, "vp9", "{label}");
            let header = parser
                .parse(&section.raw, &refs)
                .unwrap_or_else(|error| panic!("{label}: {error}"));
            refs.update(&header);
            compare(
                &label,
                &section.elements,
                &[IGNORED],
                header_elements(&header),
            );
        }
        assert!(!sections.is_empty(), "{stream} holds no sections");
    }
}

#[test]
fn vp9_headers_survive_truncation_and_corruption() {
    for stream in STREAMS {
        let Some(sections) = load(stream) else {
            continue;
        };
        let mut parser = Vp9HeaderParser::new();
        let mut refs = RefFrameTable::default();
        for section in &sections {
            let cuts = section.raw.len().min(120);
            for end in 0..cuts {
                let mut attempt = parser.clone();
                let _ = attempt.parse(&section.raw[..end], &refs);
                let _ = split_superframe(&section.raw[..end]);
                let mut flipped = section.raw.clone();
                flipped[end] ^= 0x55;
                let mut attempt = parser.clone();
                let _ = attempt.parse(&flipped, &refs);
            }
            let header = parser
                .parse(&section.raw, &refs)
                .unwrap_or_else(|error| panic!("{stream}: {error}"));
            refs.update(&header);
        }
    }
}
