//! The mapping of one frame header into `StdVideoDecodeAV1*` structures. It touches no device, so
//! tests read each field back.

use ash::vk::native::{
    StdVideoAV1CDEF, StdVideoAV1FilmGrain, StdVideoAV1FilmGrainFlags, StdVideoAV1GlobalMotion,
    StdVideoAV1LoopFilter, StdVideoAV1LoopFilterFlags, StdVideoAV1LoopRestoration,
    StdVideoAV1Quantization, StdVideoAV1QuantizationFlags, StdVideoAV1Segmentation,
    StdVideoAV1TileInfo, StdVideoAV1TileInfoFlags, StdVideoDecodeAV1PictureInfo,
    StdVideoDecodeAV1PictureInfoFlags, StdVideoDecodeAV1ReferenceInfo,
    StdVideoDecodeAV1ReferenceInfoFlags,
};

use super::refs::SlotFacts;
use crate::bitstream::av1::frame_header::{
    Cdef, FilmGrain, FrameHeader, LoopFilter, LoopRestoration, Quantization, Segmentation, TileInfo,
};
use crate::vulkan::zeroed::{AllZero, zeroed};

// SAFETY: integers and bitfield units only.
unsafe impl AllZero for StdVideoAV1LoopFilterFlags {}
// SAFETY: as above.
unsafe impl AllZero for StdVideoAV1QuantizationFlags {}
// SAFETY: as above.
unsafe impl AllZero for StdVideoAV1TileInfoFlags {}
// SAFETY: as above.
unsafe impl AllZero for StdVideoAV1FilmGrainFlags {}
// SAFETY: integers, integer arrays, and bitfield units only.
unsafe impl AllZero for StdVideoAV1FilmGrain {}
// SAFETY: integers and bitfield units only.
unsafe impl AllZero for StdVideoDecodeAV1PictureInfoFlags {}
// SAFETY: as above.
unsafe impl AllZero for StdVideoDecodeAV1ReferenceInfoFlags {}

/// A bit mask with bit `index` set for each true entry.
fn mask(flags: impl IntoIterator<Item = bool>) -> u32 {
    flags
        .into_iter()
        .enumerate()
        .fold(0, |bits, (index, flag)| bits | (u32::from(flag) << index))
}

/// The tile info with null pointers. The caller points them at the arrays of `tiles`.
pub(super) fn std_tile_info(tiles: &TileInfo) -> StdVideoAV1TileInfo {
    let mut flags: StdVideoAV1TileInfoFlags = zeroed();
    flags.set_uniform_tile_spacing_flag(tiles.uniform_tile_spacing_flag.into());
    StdVideoAV1TileInfo {
        flags,
        TileCols: tiles.tile_cols as u8,
        TileRows: tiles.tile_rows as u8,
        context_update_tile_id: tiles.context_update_tile_id,
        tile_size_bytes_minus_1: tiles.tile_size_bytes_minus_1,
        reserved1: [0; 7],
        pMiColStarts: tiles.mi_col_starts.as_ptr(),
        pMiRowStarts: tiles.mi_row_starts.as_ptr(),
        pWidthInSbsMinus1: tiles.width_in_sbs_minus_1.as_ptr(),
        pHeightInSbsMinus1: tiles.height_in_sbs_minus_1.as_ptr(),
    }
}

pub(super) fn std_quantization(q: &Quantization) -> StdVideoAV1Quantization {
    let mut flags: StdVideoAV1QuantizationFlags = zeroed();
    flags.set_using_qmatrix(q.using_qmatrix.into());
    flags.set_diff_uv_delta(q.diff_uv_delta.into());
    StdVideoAV1Quantization {
        flags,
        base_q_idx: q.base_q_idx,
        DeltaQYDc: q.delta_q_y_dc,
        DeltaQUDc: q.delta_q_u_dc,
        DeltaQUAc: q.delta_q_u_ac,
        DeltaQVDc: q.delta_q_v_dc,
        DeltaQVAc: q.delta_q_v_ac,
        qm_y: q.qm_y,
        qm_u: q.qm_u,
        qm_v: q.qm_v,
    }
}

pub(super) fn std_segmentation(segmentation: &Segmentation) -> StdVideoAV1Segmentation {
    StdVideoAV1Segmentation {
        FeatureEnabled: segmentation
            .feature_enabled
            .map(|segment| mask(segment) as u8),
        FeatureData: segmentation.feature_data,
    }
}

pub(super) fn std_loop_filter(filter: &LoopFilter) -> StdVideoAV1LoopFilter {
    let mut flags: StdVideoAV1LoopFilterFlags = zeroed();
    flags.set_loop_filter_delta_enabled(filter.delta_enabled.into());
    flags.set_loop_filter_delta_update(filter.delta_update.into());
    StdVideoAV1LoopFilter {
        flags,
        loop_filter_level: filter.level,
        loop_filter_sharpness: filter.sharpness,
        update_ref_delta: mask(filter.update_ref_delta) as u8,
        loop_filter_ref_deltas: filter.ref_deltas,
        update_mode_delta: mask(filter.update_mode_delta) as u8,
        loop_filter_mode_deltas: filter.mode_deltas,
    }
}

/// Secondary strengths stay bitstream values, as the Vulkan specification states.
pub(super) const fn std_cdef(cdef: &Cdef) -> StdVideoAV1CDEF {
    StdVideoAV1CDEF {
        cdef_damping_minus_3: cdef.damping_minus_3,
        cdef_bits: cdef.bits,
        cdef_y_pri_strength: cdef.y_pri_strength,
        cdef_y_sec_strength: cdef.y_sec_strength,
        cdef_uv_pri_strength: cdef.uv_pri_strength,
        cdef_uv_sec_strength: cdef.uv_sec_strength,
    }
}

/// `FrameRestorationType` holds the numbering of the AV1 specification, which
/// `StdVideoAV1FrameRestorationType` shares. `LoopRestorationSize` is `log2(size) - 5`, as the
/// Vulkan specification states.
pub(super) fn std_loop_restoration(lr: &LoopRestoration) -> StdVideoAV1LoopRestoration {
    StdVideoAV1LoopRestoration {
        FrameRestorationType: lr.frame_restoration_type.map(u32::from),
        LoopRestorationSize: [
            lr.log2_size_minus_5(0),
            lr.log2_size_minus_5(1),
            lr.log2_size_minus_5(2),
        ],
    }
}

pub(super) const fn std_global_motion(header: &FrameHeader) -> StdVideoAV1GlobalMotion {
    StdVideoAV1GlobalMotion {
        GmType: header.gm_type,
        gm_params: header.gm_params,
    }
}

pub(super) fn std_film_grain(grain: &FilmGrain) -> StdVideoAV1FilmGrain {
    let mut std: StdVideoAV1FilmGrain = zeroed();
    let mut flags: StdVideoAV1FilmGrainFlags = zeroed();
    flags.set_chroma_scaling_from_luma(grain.chroma_scaling_from_luma.into());
    flags.set_overlap_flag(grain.overlap_flag.into());
    flags.set_clip_to_restricted_range(grain.clip_to_restricted_range.into());
    flags.set_update_grain(grain.update_grain.into());
    std.flags = flags;
    std.grain_scaling_minus_8 = grain.grain_scaling_minus_8;
    std.ar_coeff_lag = grain.ar_coeff_lag;
    std.ar_coeff_shift_minus_6 = grain.ar_coeff_shift_minus_6;
    std.grain_scale_shift = grain.grain_scale_shift;
    std.grain_seed = grain.grain_seed;
    std.film_grain_params_ref_idx = grain.film_grain_params_ref_idx;
    std.num_y_points = grain.num_y_points;
    std.point_y_value = grain.point_y_value;
    std.point_y_scaling = grain.point_y_scaling;
    std.num_cb_points = grain.num_cb_points;
    std.point_cb_value = grain.point_cb_value;
    std.point_cb_scaling = grain.point_cb_scaling;
    std.num_cr_points = grain.num_cr_points;
    std.point_cr_value = grain.point_cr_value;
    std.point_cr_scaling = grain.point_cr_scaling;
    std.ar_coeffs_y_plus_128 = grain.ar_coeffs_y_plus_128.map(|value| value as i8);
    std.ar_coeffs_cb_plus_128 = grain.ar_coeffs_cb_plus_128.map(|value| value as i8);
    std.ar_coeffs_cr_plus_128 = grain.ar_coeffs_cr_plus_128.map(|value| value as i8);
    std.cb_mult = grain.cb_mult;
    std.cb_luma_mult = grain.cb_luma_mult;
    std.cb_offset = grain.cb_offset;
    std.cr_mult = grain.cr_mult;
    std.cr_luma_mult = grain.cr_luma_mult;
    std.cr_offset = grain.cr_offset;
    std
}

fn picture_flags(header: &FrameHeader) -> StdVideoDecodeAV1PictureInfoFlags {
    let mut flags: StdVideoDecodeAV1PictureInfoFlags = zeroed();
    flags.set_error_resilient_mode(header.error_resilient_mode.into());
    flags.set_disable_cdf_update(header.disable_cdf_update.into());
    flags.set_use_superres(header.use_superres.into());
    flags.set_render_and_frame_size_different(header.render_and_frame_size_different.into());
    flags.set_allow_screen_content_tools(header.allow_screen_content_tools.into());
    flags.set_is_filter_switchable(header.is_filter_switchable.into());
    flags.set_force_integer_mv(header.force_integer_mv.into());
    flags.set_frame_size_override_flag(header.frame_size_override_flag.into());
    flags.set_buffer_removal_time_present_flag(header.buffer_removal_time_present_flag.into());
    flags.set_allow_intrabc(header.allow_intrabc.into());
    flags.set_frame_refs_short_signaling(header.frame_refs_short_signaling.into());
    flags.set_allow_high_precision_mv(header.allow_high_precision_mv.into());
    flags.set_is_motion_mode_switchable(header.is_motion_mode_switchable.into());
    flags.set_use_ref_frame_mvs(header.use_ref_frame_mvs.into());
    flags.set_disable_frame_end_update_cdf(header.disable_frame_end_update_cdf.into());
    flags.set_allow_warped_motion(header.allow_warped_motion.into());
    flags.set_reduced_tx_set(header.reduced_tx_set.into());
    flags.set_reference_select(header.reference_select.into());
    flags.set_skip_mode_present(header.skip_mode_present.into());
    flags.set_delta_q_present(header.delta_q_present.into());
    flags.set_delta_lf_present(header.delta_lf_present.into());
    flags.set_delta_lf_multi(header.delta_lf_multi.into());
    flags.set_segmentation_enabled(header.segmentation.enabled.into());
    flags.set_segmentation_update_map(header.segmentation.update_map.into());
    flags.set_segmentation_temporal_update(header.segmentation.temporal_update.into());
    flags.set_segmentation_update_data(header.segmentation.update_data.into());
    flags.set_UsesLr(header.loop_restoration.uses_lr.into());
    flags.set_usesChromaLr(header.loop_restoration.uses_chroma_lr.into());
    flags.set_apply_grain(header.film_grain.apply_grain.into());
    flags
}

/// The picture info of `header`, with null pointers. The caller points them at what the
/// functions above build. `frame_type`, `interpolation_filter`, and `TxMode` hold the numbering
/// of the AV1 specification, which the Vulkan enumerations share.
pub(super) fn std_picture_info(header: &FrameHeader) -> StdVideoDecodeAV1PictureInfo {
    let mut expected = [0; 8];
    expected[..7].copy_from_slice(&header.expected_frame_id);
    StdVideoDecodeAV1PictureInfo {
        flags: picture_flags(header),
        frame_type: header.frame_type.into(),
        current_frame_id: header.current_frame_id,
        OrderHint: header.order_hint,
        primary_ref_frame: header.primary_ref_frame,
        refresh_frame_flags: header.refresh_frame_flags,
        reserved1: 0,
        interpolation_filter: header.interpolation_filter.into(),
        TxMode: header.tx_mode.into(),
        delta_q_res: header.delta_q_res,
        delta_lf_res: header.delta_lf_res,
        SkipModeFrame: header.skip_mode_frame,
        coded_denom: header.coded_denom,
        reserved2: [0; 3],
        OrderHints: header.order_hints,
        expectedFrameId: expected,
        pTileInfo: std::ptr::null(),
        pQuantization: std::ptr::null(),
        pSegmentation: std::ptr::null(),
        pLoopFilter: std::ptr::null(),
        pCDEF: std::ptr::null(),
        pLoopRestoration: std::ptr::null(),
        pGlobalMotion: std::ptr::null(),
        pFilmGrain: std::ptr::null(),
    }
}

/// What a DPB slot keeps of `header`.
pub(super) fn slot_facts(header: &FrameHeader) -> SlotFacts {
    SlotFacts {
        disable_frame_end_update_cdf: header.disable_frame_end_update_cdf,
        segmentation_enabled: header.segmentation.enabled,
        frame_type: header.frame_type,
        sign_bias: mask(header.ref_frame_sign_bias) as u8,
        order_hint: header.order_hint,
        saved_order_hints: header.order_hints,
    }
}

/// The reference info of a slot, or of the frame in decode.
pub(super) fn std_reference_info(facts: &SlotFacts) -> StdVideoDecodeAV1ReferenceInfo {
    let mut flags: StdVideoDecodeAV1ReferenceInfoFlags = zeroed();
    flags.set_disable_frame_end_update_cdf(facts.disable_frame_end_update_cdf.into());
    flags.set_segmentation_enabled(facts.segmentation_enabled.into());
    StdVideoDecodeAV1ReferenceInfo {
        flags,
        frame_type: facts.frame_type,
        RefFrameSignBias: facts.sign_bias,
        OrderHint: facts.order_hint,
        SavedOrderHints: facts.saved_order_hints,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::av1::frame_header::{
        INTRA_ONLY_FRAME, RESTORE_SGRPROJ, RESTORE_SWITCHABLE, RESTORE_WIENER, TX_MODE_LARGEST,
    };

    #[test]
    fn a_header_maps_field_by_field() {
        let mut header = FrameHeader {
            frame_type: INTRA_ONLY_FRAME,
            error_resilient_mode: true,
            force_integer_mv: true,
            use_superres: true,
            coded_denom: 3,
            current_frame_id: 77,
            order_hint: 9,
            primary_ref_frame: 2,
            refresh_frame_flags: 0x41,
            interpolation_filter: 4,
            is_filter_switchable: true,
            tx_mode: TX_MODE_LARGEST,
            delta_q_res: 2,
            delta_lf_res: 1,
            delta_lf_multi: true,
            skip_mode_frame: [2, 5],
            order_hints: [0, 1, 2, 3, 4, 5, 6, 7],
            reference_select: true,
            ..FrameHeader::default()
        };
        header.expected_frame_id[6] = 55;
        header.segmentation.enabled = true;
        header.segmentation.update_data = true;
        header.loop_restoration.uses_lr = true;
        header.film_grain.apply_grain = true;
        let info = std_picture_info(&header);
        assert_eq!(info.frame_type, 2, "STD_VIDEO_AV1_FRAME_TYPE_INTRA_ONLY");
        assert_eq!(info.flags.error_resilient_mode(), 1);
        assert_eq!(info.flags.force_integer_mv(), 1);
        assert_eq!(info.flags.use_superres(), 1);
        assert_eq!(info.flags.is_filter_switchable(), 1);
        assert_eq!(info.flags.delta_lf_multi(), 1);
        assert_eq!(info.flags.reference_select(), 1);
        assert_eq!(info.flags.segmentation_enabled(), 1);
        assert_eq!(info.flags.segmentation_update_data(), 1);
        assert_eq!(info.flags.UsesLr(), 1);
        assert_eq!(info.flags.usesChromaLr(), 0);
        assert_eq!(info.flags.apply_grain(), 1);
        assert_eq!(info.coded_denom, 3);
        assert_eq!(info.current_frame_id, 77);
        assert_eq!(info.OrderHint, 9);
        assert_eq!(info.primary_ref_frame, 2);
        assert_eq!(info.refresh_frame_flags, 0x41);
        assert_eq!(info.interpolation_filter, 4, "SWITCHABLE");
        assert_eq!(info.TxMode, 1, "STD_VIDEO_AV1_TX_MODE_LARGEST");
        assert_eq!((info.delta_q_res, info.delta_lf_res), (2, 1));
        assert_eq!(info.SkipModeFrame, [2, 5]);
        assert_eq!(info.OrderHints, [0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(info.expectedFrameId[6], 55);
    }

    #[test]
    fn tile_arrays_point_at_the_starts_and_the_sizes() {
        let tiles = TileInfo {
            uniform_tile_spacing_flag: true,
            tile_cols_log2: 1,
            tile_cols: 2,
            tile_rows: 1,
            mi_col_starts: vec![0, 80, 160],
            mi_row_starts: vec![0, 90],
            width_in_sbs_minus_1: vec![4, 4],
            height_in_sbs_minus_1: vec![5],
            context_update_tile_id: 1,
            tile_size_bytes_minus_1: 3,
            ..TileInfo::default()
        };
        let std = std_tile_info(&tiles);
        assert_eq!(std.flags.uniform_tile_spacing_flag(), 1);
        assert_eq!((std.TileCols, std.TileRows), (2, 1));
        assert_eq!(std.context_update_tile_id, 1);
        assert_eq!(std.tile_size_bytes_minus_1, 3);
        assert_eq!(std.pMiColStarts, tiles.mi_col_starts.as_ptr());
        assert_eq!(std.pMiRowStarts, tiles.mi_row_starts.as_ptr());
        assert_eq!(std.pWidthInSbsMinus1, tiles.width_in_sbs_minus_1.as_ptr());
        assert_eq!(std.pHeightInSbsMinus1, tiles.height_in_sbs_minus_1.as_ptr());
    }

    #[test]
    fn quantization_keeps_each_delta_in_its_place() {
        let std = std_quantization(&Quantization {
            base_q_idx: 90,
            delta_q_y_dc: -1,
            delta_q_u_dc: -2,
            delta_q_u_ac: -3,
            delta_q_v_dc: -4,
            delta_q_v_ac: -5,
            diff_uv_delta: true,
            using_qmatrix: true,
            qm_y: 6,
            qm_u: 7,
            qm_v: 8,
            ..Quantization::default()
        });
        assert_eq!(std.base_q_idx, 90);
        assert_eq!(std.flags.diff_uv_delta(), 1);
        assert_eq!(std.flags.using_qmatrix(), 1);
        assert_eq!(
            (
                std.DeltaQYDc,
                std.DeltaQUDc,
                std.DeltaQUAc,
                std.DeltaQVDc,
                std.DeltaQVAc
            ),
            (-1, -2, -3, -4, -5)
        );
        assert_eq!((std.qm_y, std.qm_u, std.qm_v), (6, 7, 8));
    }

    /// `FeatureEnabled` holds one bit per feature for each segment.
    #[test]
    fn segmentation_turns_features_into_one_mask_per_segment() {
        let mut segmentation = Segmentation::default();
        segmentation.feature_enabled[3] = [true, false, true, false, false, false, false, true];
        segmentation.feature_data[3][2] = -40;
        let std = std_segmentation(&segmentation);
        assert_eq!(std.FeatureEnabled[3], 0b1000_0101);
        assert_eq!(std.FeatureEnabled[0], 0);
        assert_eq!(std.FeatureData[3][2], -40);
    }

    #[test]
    fn a_loop_filter_keeps_its_deltas_and_update_masks() {
        let filter = LoopFilter {
            level: [1, 2, 3, 4],
            sharpness: 5,
            delta_enabled: true,
            delta_update: true,
            update_ref_delta: [true, false, false, true, false, false, false, true],
            update_mode_delta: [false, true],
            ref_deltas: [1, 0, 0, 0, -1, 0, -1, -1],
            mode_deltas: [0, 4],
        };
        let std = std_loop_filter(&filter);
        assert_eq!(std.flags.loop_filter_delta_enabled(), 1);
        assert_eq!(std.flags.loop_filter_delta_update(), 1);
        assert_eq!(std.loop_filter_level, [1, 2, 3, 4]);
        assert_eq!(std.loop_filter_sharpness, 5);
        assert_eq!(std.update_ref_delta, 0b1000_1001);
        assert_eq!(std.update_mode_delta, 0b10);
        assert_eq!(std.loop_filter_ref_deltas, [1, 0, 0, 0, -1, 0, -1, -1]);
        assert_eq!(std.loop_filter_mode_deltas, [0, 4]);
    }

    /// The bitstream value 3 stays 3.
    #[test]
    fn cdef_keeps_the_bitstream_secondary_strength() {
        let std = std_cdef(&Cdef {
            damping_minus_3: 2,
            bits: 1,
            y_pri_strength: [7, 0, 0, 0, 0, 0, 0, 0],
            y_sec_strength: [3, 0, 0, 0, 0, 0, 0, 0],
            uv_pri_strength: [0, 9, 0, 0, 0, 0, 0, 0],
            uv_sec_strength: [0, 3, 0, 0, 0, 0, 0, 0],
        });
        assert_eq!(std.cdef_damping_minus_3, 2);
        assert_eq!(std.cdef_bits, 1);
        assert_eq!(std.cdef_y_sec_strength[0], 3);
        assert_eq!(std.cdef_uv_sec_strength[1], 3);
        assert_eq!(std.cdef_uv_pri_strength[1], 9);
    }

    /// A worked case: `lr_unit_shift` 1 gives 128 luma samples, or 2 in `log2(size) - 5`.
    /// `lr_uv_shift` 1 gives 64 chroma samples, or 1.
    #[test]
    fn restoration_states_each_type_and_the_log2_size() {
        let std = std_loop_restoration(&LoopRestoration {
            frame_restoration_type: [RESTORE_SWITCHABLE, RESTORE_WIENER, RESTORE_SGRPROJ],
            uses_lr: true,
            uses_chroma_lr: true,
            lr_unit_shift: 1,
            lr_uv_shift: 1,
            ..LoopRestoration::default()
        });
        // STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_SWITCHABLE, WIENER, and SGRPROJ.
        assert_eq!(std.FrameRestorationType, [3, 1, 2]);
        assert_eq!(std.LoopRestorationSize, [2, 1, 1]);
    }

    #[test]
    fn global_motion_keeps_each_reference() {
        let mut header = FrameHeader::default();
        header.gm_type[2] = 3;
        header.gm_params[2] = [1, 2, 3, 4, 5, 6];
        let std = std_global_motion(&header);
        assert_eq!(std.GmType[2], 3);
        assert_eq!(std.gm_params[2], [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn film_grain_keeps_each_field() {
        let grain = FilmGrain {
            apply_grain: true,
            grain_seed: 4242,
            update_grain: true,
            film_grain_params_ref_idx: 3,
            num_y_points: 2,
            point_y_value: [10, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            point_y_scaling: [20, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            chroma_scaling_from_luma: true,
            num_cb_points: 1,
            point_cb_value: [30; 10],
            num_cr_points: 1,
            point_cr_scaling: [51; 10],
            grain_scaling_minus_8: 3,
            ar_coeff_lag: 1,
            ar_coeffs_y_plus_128: [200; 24],
            ar_coeffs_cb_plus_128: [100; 25],
            ar_coeff_shift_minus_6: 2,
            grain_scale_shift: 1,
            cb_mult: 128,
            cb_offset: 256,
            cr_luma_mult: 191,
            cr_offset: 255,
            overlap_flag: true,
            clip_to_restricted_range: true,
            ..FilmGrain::default()
        };
        let std = std_film_grain(&grain);
        assert_eq!(std.flags.chroma_scaling_from_luma(), 1);
        assert_eq!(std.flags.overlap_flag(), 1);
        assert_eq!(std.flags.clip_to_restricted_range(), 1);
        assert_eq!(std.flags.update_grain(), 1);
        assert_eq!(std.grain_seed, 4242);
        assert_eq!(std.film_grain_params_ref_idx, 3);
        assert_eq!(std.num_y_points, 2);
        assert_eq!(std.point_y_value[1], 200);
        assert_eq!(std.point_y_scaling[1], 40);
        assert_eq!(std.point_cb_value[0], 30);
        assert_eq!(std.point_cr_scaling[9], 51);
        assert_eq!(std.grain_scaling_minus_8, 3);
        assert_eq!(std.ar_coeff_lag, 1);
        assert_eq!(std.ar_coeffs_y_plus_128[0] as u8, 200, "the bits stay");
        assert_eq!(std.ar_coeffs_cb_plus_128[0], 100);
        assert_eq!(std.ar_coeff_shift_minus_6, 2);
        assert_eq!(std.grain_scale_shift, 1);
        assert_eq!(std.cb_mult, 128);
        assert_eq!(std.cb_offset, 256);
        assert_eq!(std.cr_luma_mult, 191);
        assert_eq!(std.cr_offset, 255);
    }

    /// `RefFrameSignBias` is a mask, and `SavedOrderHints` is what the frame's `OrderHints` hold.
    #[test]
    fn a_reference_keeps_what_its_frame_held() {
        let mut header = FrameHeader {
            frame_type: 1,
            disable_frame_end_update_cdf: true,
            order_hint: 12,
            order_hints: [0, 11, 10, 9, 8, 13, 14, 15],
            ..FrameHeader::default()
        };
        header.segmentation.enabled = true;
        header.ref_frame_sign_bias[5] = true;
        header.ref_frame_sign_bias[7] = true;
        let std = std_reference_info(&slot_facts(&header));
        assert_eq!(std.flags.disable_frame_end_update_cdf(), 1);
        assert_eq!(std.flags.segmentation_enabled(), 1);
        assert_eq!(std.frame_type, 1);
        assert_eq!(std.RefFrameSignBias, 0b1010_0000);
        assert_eq!(std.OrderHint, 12);
        assert_eq!(std.SavedOrderHints, [0, 11, 10, 9, 8, 13, 14, 15]);
    }
}
