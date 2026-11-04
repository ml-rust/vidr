//! The frame header, its parts, and the constants of section 3 of the AV1 specification.

/// `NUM_REF_FRAMES`.
pub(crate) const NUM_REF_FRAMES: usize = 8;
/// `REFS_PER_FRAME`.
pub(crate) const REFS_PER_FRAME: usize = 7;
/// `TOTAL_REFS_PER_FRAME`.
pub(crate) const TOTAL_REFS_PER_FRAME: usize = 8;
/// `PRIMARY_REF_NONE`.
pub(crate) const PRIMARY_REF_NONE: u8 = 7;
/// `MAX_SEGMENTS`.
pub(crate) const MAX_SEGMENTS: usize = 8;
/// `SEG_LVL_MAX`.
pub(crate) const SEG_LVL_MAX: usize = 8;
/// `LAST_FRAME`.
pub(crate) const LAST_FRAME: usize = 1;
/// `WARPEDMODEL_PREC_BITS`.
pub(crate) const WARPEDMODEL_PREC_BITS: u32 = 16;
/// `KEY_FRAME`.
pub(crate) const KEY_FRAME: u8 = 0;
/// `INTER_FRAME`.
pub(crate) const INTER_FRAME: u8 = 1;
/// `INTRA_ONLY_FRAME`.
pub(crate) const INTRA_ONLY_FRAME: u8 = 2;
/// `SWITCH_FRAME`.
pub(crate) const SWITCH_FRAME: u8 = 3;
/// Every bit of `refresh_frame_flags`: `allFrames`.
pub(crate) const ALL_FRAMES: u8 = 0xff;
/// `SWITCHABLE`, the value of `interpolation_filter` with `is_filter_switchable` 1.
pub(crate) const SWITCHABLE: u8 = 4;
/// `IDENTITY`.
pub(crate) const IDENTITY: u8 = 0;
/// `TRANSLATION`.
pub(crate) const TRANSLATION: u8 = 1;
/// `ROTZOOM`.
pub(crate) const ROTZOOM: u8 = 2;
/// `AFFINE`.
pub(crate) const AFFINE: u8 = 3;
/// `SEG_LVL_ALT_Q`.
pub(crate) const SEG_LVL_ALT_Q: usize = 0;
/// `ONLY_4X4`.
pub(crate) const ONLY_4X4: u8 = 0;
/// `TX_MODE_LARGEST`.
pub(crate) const TX_MODE_LARGEST: u8 = 1;
/// `TX_MODE_SELECT`.
pub(crate) const TX_MODE_SELECT: u8 = 2;

/// The `FrameRestorationType` of each `lr_type`: `Remap_Lr_Type`.
pub(crate) const REMAP_LR_TYPE: [u8; 4] = [
    RESTORE_NONE,
    RESTORE_SWITCHABLE,
    RESTORE_WIENER,
    RESTORE_SGRPROJ,
];
/// `RESTORE_NONE`.
pub(crate) const RESTORE_NONE: u8 = 0;
/// `RESTORE_WIENER`.
pub(crate) const RESTORE_WIENER: u8 = 1;
/// `RESTORE_SGRPROJ`.
pub(crate) const RESTORE_SGRPROJ: u8 = 2;
/// `RESTORE_SWITCHABLE`.
pub(crate) const RESTORE_SWITCHABLE: u8 = 3;

/// The default `gm_params` of one reference: the identity warp.
pub(crate) const IDENTITY_WARP: [i32; 6] = [
    0,
    0,
    1 << WARPEDMODEL_PREC_BITS,
    0,
    0,
    1 << WARPEDMODEL_PREC_BITS,
];

/// The `loop_filter_ref_deltas` that `setup_past_independence` states.
pub(crate) const DEFAULT_REF_DELTAS: [i8; TOTAL_REFS_PER_FRAME] = [1, 0, 0, 0, -1, 0, -1, -1];

/// `tile_info()`, with its derived values.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TileInfo {
    pub(crate) uniform_tile_spacing_flag: bool,
    pub(crate) tile_cols_log2: u8,
    pub(crate) tile_rows_log2: u8,
    /// `TileCols`.
    pub(crate) tile_cols: u16,
    /// `TileRows`.
    pub(crate) tile_rows: u16,
    /// `MiColStarts`: `TileCols + 1` entries.
    pub(crate) mi_col_starts: Vec<u16>,
    /// `MiRowStarts`: `TileRows + 1` entries.
    pub(crate) mi_row_starts: Vec<u16>,
    /// One entry per tile column, coded or derived.
    pub(crate) width_in_sbs_minus_1: Vec<u16>,
    /// One entry per tile row, coded or derived.
    pub(crate) height_in_sbs_minus_1: Vec<u16>,
    pub(crate) context_update_tile_id: u16,
    pub(crate) tile_size_bytes_minus_1: u8,
}

impl TileInfo {
    /// `NumTiles`.
    pub(crate) const fn num_tiles(&self) -> u32 {
        self.tile_cols as u32 * self.tile_rows as u32
    }
}

/// `quantization_params()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Quantization {
    pub(crate) base_q_idx: u8,
    /// `delta_coded` of `DeltaQYDc`, `DeltaQUDc`, `DeltaQUAc`, `DeltaQVDc`, and `DeltaQVAc`.
    pub(crate) delta_coded: [bool; 5],
    pub(crate) delta_q_y_dc: i8,
    pub(crate) diff_uv_delta: bool,
    pub(crate) delta_q_u_dc: i8,
    pub(crate) delta_q_u_ac: i8,
    pub(crate) delta_q_v_dc: i8,
    pub(crate) delta_q_v_ac: i8,
    pub(crate) using_qmatrix: bool,
    pub(crate) qm_y: u8,
    pub(crate) qm_u: u8,
    pub(crate) qm_v: u8,
}

/// `segmentation_params()`, with the values it loads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Segmentation {
    pub(crate) enabled: bool,
    pub(crate) update_map: bool,
    pub(crate) temporal_update: bool,
    pub(crate) update_data: bool,
    pub(crate) feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub(crate) feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
}

/// `loop_filter_params()`, with the deltas it loads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LoopFilter {
    pub(crate) level: [u8; 4],
    pub(crate) sharpness: u8,
    pub(crate) delta_enabled: bool,
    pub(crate) delta_update: bool,
    pub(crate) update_ref_delta: [bool; TOTAL_REFS_PER_FRAME],
    pub(crate) update_mode_delta: [bool; 2],
    pub(crate) ref_deltas: [i8; TOTAL_REFS_PER_FRAME],
    pub(crate) mode_deltas: [i8; 2],
}

/// `cdef_params()`. Each secondary strength holds its bitstream value, before 3 turns into 4.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Cdef {
    pub(crate) damping_minus_3: u8,
    pub(crate) bits: u8,
    pub(crate) y_pri_strength: [u8; 8],
    pub(crate) y_sec_strength: [u8; 8],
    pub(crate) uv_pri_strength: [u8; 8],
    pub(crate) uv_sec_strength: [u8; 8],
}

/// `lr_params()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LoopRestoration {
    /// The coded `lr_type` of each plane.
    pub(crate) lr_type: [u8; 3],
    /// `FrameRestorationType`.
    pub(crate) frame_restoration_type: [u8; 3],
    pub(crate) uses_lr: bool,
    pub(crate) uses_chroma_lr: bool,
    /// `lr_unit_shift` after both of its reads.
    pub(crate) lr_unit_shift: u8,
    pub(crate) lr_uv_shift: u8,
}

impl LoopRestoration {
    /// `LoopRestorationSize` of one plane, in the form `log2(size) - 5`.
    pub(crate) const fn log2_size_minus_5(&self, plane: usize) -> u16 {
        let shift = if plane == 0 { 0 } else { self.lr_uv_shift };
        1 + self.lr_unit_shift as u16 - shift as u16
    }
}

/// `film_grain_params()`, after its load.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FilmGrain {
    pub(crate) apply_grain: bool,
    pub(crate) grain_seed: u16,
    pub(crate) update_grain: bool,
    pub(crate) film_grain_params_ref_idx: u8,
    pub(crate) num_y_points: u8,
    pub(crate) point_y_value: [u8; 14],
    pub(crate) point_y_scaling: [u8; 14],
    pub(crate) chroma_scaling_from_luma: bool,
    pub(crate) num_cb_points: u8,
    pub(crate) point_cb_value: [u8; 10],
    pub(crate) point_cb_scaling: [u8; 10],
    pub(crate) num_cr_points: u8,
    pub(crate) point_cr_value: [u8; 10],
    pub(crate) point_cr_scaling: [u8; 10],
    pub(crate) grain_scaling_minus_8: u8,
    pub(crate) ar_coeff_lag: u8,
    pub(crate) ar_coeffs_y_plus_128: [u8; 24],
    pub(crate) ar_coeffs_cb_plus_128: [u8; 25],
    pub(crate) ar_coeffs_cr_plus_128: [u8; 25],
    pub(crate) ar_coeff_shift_minus_6: u8,
    pub(crate) grain_scale_shift: u8,
    pub(crate) cb_mult: u8,
    pub(crate) cb_luma_mult: u8,
    pub(crate) cb_offset: u16,
    pub(crate) cr_mult: u8,
    pub(crate) cr_luma_mult: u8,
    pub(crate) cr_offset: u16,
    pub(crate) overlap_flag: bool,
    pub(crate) clip_to_restricted_range: bool,
}

/// `uncompressed_header()`, with its derived values. A `show_existing_frame` header holds what
/// the reference frame loading process of section 7.21 loads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct FrameHeader {
    pub(crate) temporal_id: u8,
    pub(crate) spatial_id: u8,
    pub(crate) show_existing_frame: bool,
    pub(crate) frame_to_show_map_idx: u8,
    pub(crate) frame_presentation_time: Option<u32>,
    pub(crate) display_frame_id: u32,
    pub(crate) frame_type: u8,
    /// `FrameIsIntra`.
    pub(crate) frame_is_intra: bool,
    pub(crate) show_frame: bool,
    pub(crate) showable_frame: bool,
    pub(crate) error_resilient_mode: bool,
    pub(crate) disable_cdf_update: bool,
    pub(crate) allow_screen_content_tools: bool,
    /// `force_integer_mv` as coded, before an intra frame sets it to 1.
    pub(crate) coded_force_integer_mv: bool,
    pub(crate) force_integer_mv: bool,
    pub(crate) current_frame_id: u32,
    pub(crate) frame_size_override_flag: bool,
    /// `OrderHint`.
    pub(crate) order_hint: u8,
    pub(crate) primary_ref_frame: u8,
    pub(crate) buffer_removal_time_present_flag: bool,
    /// Each coded `buffer_removal_time`, with its operating point.
    pub(crate) buffer_removal_times: Vec<(u8, u32)>,
    pub(crate) allow_high_precision_mv: bool,
    pub(crate) use_ref_frame_mvs: bool,
    pub(crate) allow_intrabc: bool,
    pub(crate) refresh_frame_flags: u8,
    /// Each coded `ref_order_hint`.
    pub(crate) ref_order_hint: Option<[u8; NUM_REF_FRAMES]>,
    /// Each name, as a bit, whose frame this header's `ref_order_hint` states as lost.
    pub(crate) lost_names: u8,
    pub(crate) frame_refs_short_signaling: bool,
    pub(crate) last_frame_idx: u8,
    pub(crate) gold_frame_idx: u8,
    pub(crate) ref_frame_idx: [u8; REFS_PER_FRAME],
    pub(crate) delta_frame_id_minus_1: [u32; REFS_PER_FRAME],
    /// `expectedFrameId`.
    pub(crate) expected_frame_id: [u32; REFS_PER_FRAME],
    /// Whether `frame_width_minus_1` and `frame_height_minus_1` hold coded values.
    pub(crate) frame_size_coded: bool,
    /// How many `found_ref` bits the header holds, and whether the last one is 1.
    pub(crate) found_ref: Option<(u8, bool)>,
    pub(crate) use_superres: bool,
    pub(crate) coded_denom: u8,
    /// `UpscaledWidth`.
    pub(crate) upscaled_width: u32,
    /// `FrameWidth`.
    pub(crate) frame_width: u32,
    /// `FrameHeight`.
    pub(crate) frame_height: u32,
    pub(crate) render_and_frame_size_different: bool,
    /// `RenderWidth`.
    pub(crate) render_width: u32,
    /// `RenderHeight`.
    pub(crate) render_height: u32,
    /// `MiCols`.
    pub(crate) mi_cols: u32,
    /// `MiRows`.
    pub(crate) mi_rows: u32,
    pub(crate) is_filter_switchable: bool,
    pub(crate) interpolation_filter: u8,
    pub(crate) is_motion_mode_switchable: bool,
    /// `OrderHints`, by reference frame.
    pub(crate) order_hints: [u8; TOTAL_REFS_PER_FRAME],
    /// `RefFrameSignBias`, by reference frame.
    pub(crate) ref_frame_sign_bias: [bool; TOTAL_REFS_PER_FRAME],
    pub(crate) disable_frame_end_update_cdf: bool,
    pub(crate) tile_info: TileInfo,
    pub(crate) quantization: Quantization,
    pub(crate) segmentation: Segmentation,
    pub(crate) delta_q_present: bool,
    pub(crate) delta_q_res: u8,
    pub(crate) delta_lf_present: bool,
    pub(crate) delta_lf_res: u8,
    pub(crate) delta_lf_multi: bool,
    /// `CodedLossless`.
    pub(crate) coded_lossless: bool,
    /// `AllLossless`.
    pub(crate) all_lossless: bool,
    pub(crate) loop_filter: LoopFilter,
    pub(crate) cdef: Cdef,
    pub(crate) loop_restoration: LoopRestoration,
    /// `TxMode`.
    pub(crate) tx_mode: u8,
    pub(crate) reference_select: bool,
    /// `skipModeAllowed`.
    pub(crate) skip_mode_allowed: bool,
    pub(crate) skip_mode_present: bool,
    /// `SkipModeFrame`.
    pub(crate) skip_mode_frame: [u8; 2],
    pub(crate) allow_warped_motion: bool,
    pub(crate) reduced_tx_set: bool,
    /// `GmType`, by reference frame.
    pub(crate) gm_type: [u8; TOTAL_REFS_PER_FRAME],
    /// `gm_params`, by reference frame.
    pub(crate) gm_params: [[i32; 6]; TOTAL_REFS_PER_FRAME],
    pub(crate) film_grain: FilmGrain,
    /// The bits `uncompressed_header()` reads.
    pub(crate) header_bits: u64,
}

impl FrameHeader {
    /// The size of the decoded frame: `UpscaledWidth` by `FrameHeight`.
    pub(crate) const fn upscaled_size(&self) -> (u32, u32) {
        (self.upscaled_width, self.frame_height)
    }

    /// The bytes of the header with its `byte_alignment()`.
    pub(crate) const fn header_bytes(&self) -> usize {
        self.header_bits.div_ceil(8) as usize
    }
}
