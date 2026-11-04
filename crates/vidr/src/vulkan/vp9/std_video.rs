//! The mapping of one VP9 frame header into `StdVideoDecodeVP9*` structures. It touches no
//! device, so tests read each field back.

use crate::bitstream::{
    ColorConfig, KEY_FRAME, LoopFilterParams, SegmentationParams, UncompressedHeader,
};
use crate::error::DecodeError;
use crate::vulkan::gap::{
    StdVideoDecodeVp9PictureInfo, StdVideoVp9ColorConfig, StdVideoVp9LoopFilter,
    StdVideoVp9Segmentation,
};

/// A bit mask with bit `index` set for each true entry.
fn mask(flags: impl IntoIterator<Item = bool>) -> u32 {
    flags
        .into_iter()
        .enumerate()
        .fold(0, |bits, (index, flag)| bits | (u32::from(flag) << index))
}

pub(super) fn std_color(color: &ColorConfig) -> StdVideoVp9ColorConfig {
    StdVideoVp9ColorConfig {
        flags: u32::from(color.color_range),
        bit_depth: color.bit_depth,
        subsampling_x: u8::from(color.subsampling_x),
        subsampling_y: u8::from(color.subsampling_y),
        reserved1: 0,
        color_space: u32::from(color.color_space),
    }
}

pub(super) fn std_loop_filter(filter: &LoopFilterParams) -> StdVideoVp9LoopFilter {
    StdVideoVp9LoopFilter {
        flags: mask([
            filter.loop_filter_delta_enabled,
            filter.loop_filter_delta_update,
        ]),
        loop_filter_level: filter.loop_filter_level,
        loop_filter_sharpness: filter.loop_filter_sharpness,
        update_ref_delta: mask(filter.update_ref_delta) as u8,
        loop_filter_ref_deltas: filter.loop_filter_ref_deltas,
        update_mode_delta: mask(filter.update_mode_delta) as u8,
        loop_filter_mode_deltas: filter.loop_filter_mode_deltas,
    }
}

pub(super) fn std_segmentation(params: &SegmentationParams) -> StdVideoVp9Segmentation {
    StdVideoVp9Segmentation {
        flags: mask([
            params.segmentation_update_map,
            params.segmentation_temporal_update,
            params.segmentation_update_data,
            params.segmentation_abs_or_delta_update,
        ]),
        segmentation_tree_probs: params.segmentation_tree_probs,
        segmentation_pred_prob: params.segmentation_pred_prob,
        feature_enabled: params.feature_enabled.map(|segment| mask(segment) as u8),
        feature_data: params.feature_data,
    }
}

/// The picture info of `header`, with null pointers. The caller points them at what the
/// functions above build.
///
/// `frame_context_idx` is the coded value, because Vulkan names a derived value in CamelCase.
/// The driver resets and selects contexts from it and `reset_frame_context`.
pub(super) fn std_picture_info(header: &UncompressedHeader) -> StdVideoDecodeVp9PictureInfo {
    let quantization = &header.quantization;
    StdVideoDecodeVp9PictureInfo {
        flags: mask([
            header.error_resilient_mode,
            header.intra_only,
            header.allow_high_precision_mv,
            header.refresh_frame_context,
            header.frame_parallel_decoding_mode,
            header.segmentation.segmentation_enabled,
            header.show_frame,
            header.use_prev_frame_mvs,
        ]),
        profile: u32::from(header.profile),
        frame_type: u32::from(header.frame_type != KEY_FRAME),
        frame_context_idx: header.coded_frame_context_idx,
        reset_frame_context: header.reset_frame_context,
        refresh_frame_flags: header.refresh_frame_flags,
        ref_frame_sign_bias_mask: mask(header.ref_frame_sign_bias) as u8,
        // The parser's numbering and `StdVideoVP9InterpolationFilter` both give EIGHTTAP 0
        // and EIGHTTAP_SMOOTH 1.
        interpolation_filter: u32::from(header.interpolation_filter),
        base_q_idx: quantization.base_q_idx,
        delta_q_y_dc: quantization.delta_q_y_dc,
        delta_q_uv_dc: quantization.delta_q_uv_dc,
        delta_q_uv_ac: quantization.delta_q_uv_ac,
        tile_cols_log2: header.tile_cols_log2,
        tile_rows_log2: header.tile_rows_log2,
        reserved1: [0; 3],
        p_color_config: std::ptr::null(),
        p_loop_filter: std::ptr::null(),
        p_segmentation: std::ptr::null(),
    }
}

/// `compressedHeaderOffset` and `tilesOffset` of a frame at offset 0.
pub(super) fn offsets(header: &UncompressedHeader) -> (u32, u32) {
    let compressed = header.uncompressed_header_size as u32;
    (
        compressed,
        compressed + u32::from(header.header_size_in_bytes),
    )
}

/// Refuses a frame that ends at or before its tile data, on what the driver takes as
/// `VUID-vkCmdDecodeVideoKHR-tilesOffset-10808`. The shared parser takes that frame, so the CPU
/// decoder keeps its own answer.
pub(super) fn check_tile_bytes(
    header: &UncompressedHeader,
    frame_bytes: usize,
    pts: i64,
) -> Result<(), DecodeError> {
    let (_, tiles_offset) = offsets(header);
    if tiles_offset as usize >= frame_bytes {
        return Err(DecodeError::VulkanNoTileBytes {
            pts,
            tiles_offset,
            frame_bytes,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_with_no_tile_bytes_is_refused_by_name() {
        let header = UncompressedHeader {
            uncompressed_header_size: 31,
            header_size_in_bytes: 200,
            ..UncompressedHeader::default()
        };
        assert_eq!(check_tile_bytes(&header, 232, 7), Ok(()));
        for frame_bytes in [231, 100] {
            let error = check_tile_bytes(&header, frame_bytes, 7).expect_err("refused");
            assert_eq!(
                error,
                DecodeError::VulkanNoTileBytes {
                    pts: 7,
                    tiles_offset: 231,
                    frame_bytes,
                }
            );
            let text = error.to_string();
            assert!(text.contains("pts 7"), "{text}");
            assert!(text.contains("byte 231"), "{text}");
            assert!(text.contains("re-encode"), "{text}");
        }
    }

    #[test]
    fn a_header_maps_field_by_field() {
        let mut header = UncompressedHeader {
            profile: 2,
            frame_type: 1,
            show_frame: true,
            allow_high_precision_mv: true,
            refresh_frame_context: true,
            reset_frame_context: 2,
            coded_frame_context_idx: 3,
            frame_context_idx: 0,
            refresh_frame_flags: 0b0100_0001,
            interpolation_filter: 4,
            tile_cols_log2: 1,
            tile_rows_log2: 2,
            uncompressed_header_size: 31,
            header_size_in_bytes: 200,
            ..UncompressedHeader::default()
        };
        header.ref_frame_sign_bias = [false, false, true, true];
        header.quantization.base_q_idx = 60;
        header.quantization.delta_q_y_dc = -3;
        header.quantization.delta_q_uv_dc = 2;
        header.quantization.delta_q_uv_ac = -1;
        let info = std_picture_info(&header);
        assert_eq!(info.flags, 0b0100_1100);
        assert_eq!(info.profile, 2);
        assert_eq!(info.frame_type, 1, "STD_VIDEO_VP9_FRAME_TYPE_NON_KEY");
        assert_eq!(info.frame_context_idx, 3);
        assert_eq!(info.reset_frame_context, 2);
        assert_eq!(info.refresh_frame_flags, 0b0100_0001);
        assert_eq!(info.ref_frame_sign_bias_mask, 0b1100);
        assert_eq!(info.interpolation_filter, 4);
        assert_eq!(info.base_q_idx, 60);
        assert_eq!(
            (info.delta_q_y_dc, info.delta_q_uv_dc, info.delta_q_uv_ac),
            (-3, 2, -1)
        );
        assert_eq!((info.tile_cols_log2, info.tile_rows_log2), (1, 2));
        assert_eq!(offsets(&header), (31, 231));
    }

    #[test]
    fn a_key_frame_is_type_0() {
        let header = UncompressedHeader {
            frame_type: KEY_FRAME,
            ..UncompressedHeader::default()
        };
        assert_eq!(std_picture_info(&header).frame_type, 0);
    }

    #[test]
    fn color_maps_each_field() {
        let color = ColorConfig {
            bit_depth: 10,
            color_space: 2,
            color_range: true,
            subsampling_x: true,
            subsampling_y: true,
        };
        let std = std_color(&color);
        assert_eq!(std.flags, 1);
        assert_eq!(std.bit_depth, 10);
        assert_eq!((std.subsampling_x, std.subsampling_y), (1, 1));
        assert_eq!(std.color_space, 2);
    }

    #[test]
    fn a_loop_filter_keeps_its_deltas_and_update_masks() {
        let filter = LoopFilterParams {
            loop_filter_level: 12,
            loop_filter_sharpness: 3,
            loop_filter_delta_enabled: true,
            loop_filter_delta_update: true,
            update_ref_delta: [true, false, false, true],
            update_mode_delta: [false, true],
            loop_filter_ref_deltas: [1, 0, -1, -2],
            loop_filter_mode_deltas: [0, 4],
        };
        let std = std_loop_filter(&filter);
        assert_eq!(std.flags, 0b11);
        assert_eq!(std.update_ref_delta, 0b1001);
        assert_eq!(std.update_mode_delta, 0b10);
        assert_eq!(std.loop_filter_ref_deltas, [1, 0, -1, -2]);
        assert_eq!(std.loop_filter_mode_deltas, [0, 4]);
    }

    #[test]
    fn segmentation_turns_features_into_one_mask_per_segment() {
        let mut params = SegmentationParams {
            segmentation_update_data: true,
            segmentation_abs_or_delta_update: true,
            segmentation_tree_probs: [255; 7],
            segmentation_pred_prob: [255, 7, 255],
            ..SegmentationParams::default()
        };
        params.feature_enabled[2] = [true, false, true, false];
        params.feature_data[2] = [-9, 0, 3, 0];
        let std = std_segmentation(&params);
        assert_eq!(std.flags, 0b1100);
        assert_eq!(std.feature_enabled[2], 0b0101);
        assert_eq!(std.feature_data[2], [-9, 0, 3, 0]);
        assert_eq!(std.segmentation_pred_prob, [255, 7, 255]);
    }
}
