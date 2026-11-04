//! VP9 `uncompressed_header()` (section 6.2 of the VP9 bitstream specification,
//! v0.7).
//!
//! [`Vp9HeaderParser`] holds the state the syntax carries from frame to frame:
//! colour, loop filter deltas, segmentation features, and the values of the
//! last `compute_image_size()`. The caller holds [`RefFrameTable`], and updates
//! it when a frame finishes decoding (8.10).

use super::bits::BitReader;
use super::vp9::{CS_RGB, STRUCTURE, sync_code};
use crate::error::BitstreamError;

/// `frame_type` of a key frame.
pub(crate) const KEY_FRAME: u8 = 0;
/// `color_space` 1: `CS_BT_601`.
const CS_BT_601: u8 = 1;
/// `interpolation_filter` 4: `SWITCHABLE`.
pub(crate) const SWITCHABLE: u8 = 4;
/// `literal_to_type` (6.2.7): `EIGHTTAP_SMOOTH`, `EIGHTTAP`, `EIGHTTAP_SHARP`,
/// and `BILINEAR`.
const LITERAL_TO_TYPE: [u8; 4] = [1, 0, 2, 3];
/// `NUM_REF_FRAMES`.
pub(crate) const NUM_REF_FRAMES: usize = 8;
/// `MAX_SEGMENTS`.
pub(crate) const MAX_SEGMENTS: usize = 8;
/// `SEG_LVL_MAX`.
pub(crate) const SEG_LVL_MAX: usize = 4;
/// `segmentation_feature_bits` (6.2.11).
const SEGMENTATION_FEATURE_BITS: [u32; SEG_LVL_MAX] = [8, 6, 2, 0];
/// `segmentation_feature_signed` (6.2.11).
const SEGMENTATION_FEATURE_SIGNED: [bool; SEG_LVL_MAX] = [true, true, false, false];
/// `MIN_TILE_WIDTH_B64`.
const MIN_TILE_WIDTH_B64: u32 = 4;
/// `MAX_TILE_WIDTH_B64`.
const MAX_TILE_WIDTH_B64: u32 = 64;

/// `color_config()` (6.2.2). Inter frames inherit it from the parser state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ColorConfig {
    /// `BitDepth`: 8, 10, or 12.
    pub(crate) bit_depth: u8,
    pub(crate) color_space: u8,
    pub(crate) color_range: bool,
    pub(crate) subsampling_x: bool,
    pub(crate) subsampling_y: bool,
}

/// `loop_filter_params()` (6.2.8). The deltas hold the values in force after
/// this frame's updates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LoopFilterParams {
    pub(crate) loop_filter_level: u8,
    pub(crate) loop_filter_sharpness: u8,
    pub(crate) loop_filter_delta_enabled: bool,
    pub(crate) loop_filter_delta_update: bool,
    pub(crate) update_ref_delta: [bool; 4],
    pub(crate) update_mode_delta: [bool; 2],
    pub(crate) loop_filter_ref_deltas: [i8; 4],
    pub(crate) loop_filter_mode_deltas: [i8; 2],
}

/// `quantization_params()` (6.2.9).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct QuantizationParams {
    pub(crate) base_q_idx: u8,
    pub(crate) delta_q_y_dc: i8,
    pub(crate) delta_q_uv_dc: i8,
    pub(crate) delta_q_uv_ac: i8,
    /// `delta_coded` of `delta_q_y_dc`, `delta_q_uv_dc`, and `delta_q_uv_ac`.
    pub(crate) delta_coded: [bool; 3],
    /// `Lossless`.
    pub(crate) lossless: bool,
}

/// `segmentation_params()` (6.2.11). The feature arrays hold the values in
/// force after this frame's updates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SegmentationParams {
    pub(crate) segmentation_enabled: bool,
    pub(crate) segmentation_update_map: bool,
    pub(crate) segmentation_tree_probs: [u8; 7],
    pub(crate) segmentation_tree_prob_coded: [bool; 7],
    pub(crate) segmentation_temporal_update: bool,
    pub(crate) segmentation_pred_prob: [u8; 3],
    pub(crate) segmentation_pred_prob_coded: [bool; 3],
    pub(crate) segmentation_update_data: bool,
    pub(crate) segmentation_abs_or_delta_update: bool,
    /// `FeatureEnabled`, indexed by segment and feature.
    pub(crate) feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    /// `FeatureData`, indexed by segment and feature.
    pub(crate) feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
    /// `feature_sign` as coded, indexed by segment and feature.
    pub(crate) feature_sign: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
}

/// One `uncompressed_header()`. With `show_existing_frame`, only `profile`,
/// `show_existing_frame`, and `frame_to_show_map_idx` hold values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct UncompressedHeader {
    /// `Profile`.
    pub(crate) profile: u8,
    pub(crate) show_existing_frame: bool,
    pub(crate) frame_to_show_map_idx: u8,
    /// `LastFrameType`. `None` before the first frame that sets it.
    pub(crate) last_frame_type: Option<u8>,
    pub(crate) frame_type: u8,
    pub(crate) show_frame: bool,
    pub(crate) error_resilient_mode: bool,
    pub(crate) intra_only: bool,
    /// `FrameIsIntra`.
    pub(crate) frame_is_intra: bool,
    pub(crate) reset_frame_context: u8,
    pub(crate) color_config: ColorConfig,
    pub(crate) refresh_frame_flags: u8,
    pub(crate) ref_frame_idx: [u8; 3],
    /// Indexed by reference frame: `INTRA_FRAME` 0 stays false.
    pub(crate) ref_frame_sign_bias: [bool; 4],
    /// The `i` whose `found_ref` is 1. `None` when the size is explicit.
    pub(crate) found_ref: Option<u8>,
    /// `FrameWidth`.
    pub(crate) frame_width: u32,
    /// `FrameHeight`.
    pub(crate) frame_height: u32,
    pub(crate) render_and_frame_size_different: bool,
    /// `renderWidth`.
    pub(crate) render_width: u32,
    /// `renderHeight`.
    pub(crate) render_height: u32,
    /// `MiCols`.
    pub(crate) mi_cols: u32,
    /// `MiRows`.
    pub(crate) mi_rows: u32,
    /// `Sb64Cols`.
    pub(crate) sb64_cols: u32,
    /// `Sb64Rows`.
    pub(crate) sb64_rows: u32,
    /// True when the size differs from the last `compute_image_size()`, or
    /// none ran. The segmentation map clears in this case (7.2.6).
    pub(crate) size_changed: bool,
    /// `UsePrevFrameMvs` (7.2.6).
    pub(crate) use_prev_frame_mvs: bool,
    pub(crate) allow_high_precision_mv: bool,
    pub(crate) is_filter_switchable: bool,
    /// `raw_interpolation_filter`. 0 with a switchable filter.
    pub(crate) raw_interpolation_filter: u8,
    pub(crate) interpolation_filter: u8,
    pub(crate) refresh_frame_context: bool,
    pub(crate) frame_parallel_decoding_mode: bool,
    /// `frame_context_idx` as coded, which `save_probs` reads (6.2).
    pub(crate) coded_frame_context_idx: u8,
    /// `frame_context_idx` after 6.2 sets it to 0 for an independent frame.
    pub(crate) frame_context_idx: u8,
    /// True when `setup_past_independence()` runs for this frame.
    pub(crate) past_independence: bool,
    pub(crate) loop_filter: LoopFilterParams,
    pub(crate) quantization: QuantizationParams,
    pub(crate) segmentation: SegmentationParams,
    pub(crate) tile_cols_log2: u8,
    pub(crate) tile_rows_log2: u8,
    pub(crate) header_size_in_bytes: u16,
    /// Bytes of the uncompressed header with its trailing bits: the offset of
    /// the compressed header in the frame.
    pub(crate) uncompressed_header_size: usize,
}

/// `RefFrameWidth`, `RefFrameHeight`, `RefSubsamplingX`, `RefSubsamplingY`, and
/// `RefBitDepth`. A width of 0 marks a slot no frame has filled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RefFrameTable {
    pub(crate) ref_frame_width: [u32; NUM_REF_FRAMES],
    pub(crate) ref_frame_height: [u32; NUM_REF_FRAMES],
    pub(crate) ref_subsampling_x: [bool; NUM_REF_FRAMES],
    pub(crate) ref_subsampling_y: [bool; NUM_REF_FRAMES],
    pub(crate) ref_bit_depth: [u8; NUM_REF_FRAMES],
}

impl RefFrameTable {
    /// Applies step 1 of the reference frame update process (8.10). Call it
    /// when a frame finishes decoding.
    pub(crate) fn update(&mut self, header: &UncompressedHeader) {
        for slot in 0..NUM_REF_FRAMES {
            if (header.refresh_frame_flags >> slot) & 1 == 1 {
                self.ref_frame_width[slot] = header.frame_width;
                self.ref_frame_height[slot] = header.frame_height;
                self.ref_subsampling_x[slot] = header.color_config.subsampling_x;
                self.ref_subsampling_y[slot] = header.color_config.subsampling_y;
                self.ref_bit_depth[slot] = header.color_config.bit_depth;
            }
        }
    }
}

/// The values the syntax keeps between frames.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Vp9HeaderParser {
    /// The `frame_type` of the previous frame.
    frame_type: Option<u8>,
    color_config: Option<ColorConfig>,
    /// `LastWidth`, the frame width at the last `compute_image_size()`.
    last_width: u32,
    /// `LastHeight`, the frame height at the last `compute_image_size()`.
    last_height: u32,
    /// `show_frame` at the last `compute_image_size()`.
    last_show_frame: bool,
    loop_filter_ref_deltas: [i8; 4],
    loop_filter_mode_deltas: [i8; 2],
    segmentation_abs_or_delta_update: bool,
    feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
    feature_sign: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
}

impl Vp9HeaderParser {
    /// Creates a parser with no frame seen.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Reads the header of one frame, not a superframe, and updates the state.
    /// `refs` is the table as it stands before this frame. A failed read leaves
    /// the state unchanged.
    pub(crate) fn parse(
        &mut self,
        frame: &[u8],
        refs: &RefFrameTable,
    ) -> Result<UncompressedHeader, BitstreamError> {
        let mut next = self.clone();
        let header = next.read_header(frame, refs)?;
        *self = next;
        Ok(header)
    }

    fn read_header(
        &mut self,
        frame: &[u8],
        refs: &RefFrameTable,
    ) -> Result<UncompressedHeader, BitstreamError> {
        let mut reader = BitReader::new(frame, STRUCTURE);
        let reader = &mut reader;
        let marker = reader.u8(2, "frame_marker")?;
        if marker != 2 {
            return Err(reader.invalid("frame_marker", marker.into(), "2"));
        }
        let low = reader.u8(1, "profile_low_bit")?;
        let high = reader.u8(1, "profile_high_bit")?;
        let profile = (high << 1) | low;
        if profile == 3 && reader.bit("reserved_zero")? {
            return Err(reader.invalid("reserved_zero", 1, "0"));
        }
        let mut header = UncompressedHeader {
            profile,
            ..UncompressedHeader::default()
        };
        header.show_existing_frame = reader.bit("show_existing_frame")?;
        if header.show_existing_frame {
            header.frame_to_show_map_idx = reader.u8(3, "frame_to_show_map_idx")?;
            return Ok(header);
        }
        header.last_frame_type = self.frame_type;
        header.frame_type = reader.u8(1, "frame_type")?;
        self.frame_type = Some(header.frame_type);
        header.show_frame = reader.bit("show_frame")?;
        header.error_resilient_mode = reader.bit("error_resilient_mode")?;
        if header.frame_type == KEY_FRAME {
            sync_code(reader)?;
            self.read_color_config(reader, profile)?;
            header.frame_is_intra = true;
            self.read_frame_size(reader, &mut header)?;
            read_render_size(reader, &mut header)?;
            header.refresh_frame_flags = 0xff;
        } else {
            self.read_non_key_frame(reader, refs, &mut header)?;
        }
        header.color_config = self.color_config.unwrap_or_default();
        if !header.error_resilient_mode {
            header.refresh_frame_context = reader.bit("refresh_frame_context")?;
            header.frame_parallel_decoding_mode = reader.bit("frame_parallel_decoding_mode")?;
        } else {
            header.frame_parallel_decoding_mode = true;
        }
        header.coded_frame_context_idx = reader.u8(2, "frame_context_idx")?;
        header.frame_context_idx = header.coded_frame_context_idx;
        if header.frame_is_intra || header.error_resilient_mode {
            self.setup_past_independence();
            header.past_independence = true;
            header.frame_context_idx = 0;
        }
        self.read_loop_filter(reader, &mut header)?;
        read_quantization(reader, &mut header)?;
        self.read_segmentation(reader, &mut header)?;
        read_tile_info(reader, &mut header)?;
        header.header_size_in_bytes = reader.bits(16, "header_size_in_bytes")? as u16;
        // `trailing_bits()` (6.1.1).
        while !reader.position().is_multiple_of(8) {
            if reader.bit("zero_bit")? {
                return Err(reader.invalid("zero_bit", 1, "0"));
            }
        }
        header.uncompressed_header_size = (reader.position() / 8) as usize;
        if usize::from(header.header_size_in_bytes) > frame.len() - header.uncompressed_header_size
        {
            return Err(reader.invalid(
                "header_size_in_bytes",
                header.header_size_in_bytes.into(),
                "at most the bytes left in the frame after the uncompressed header",
            ));
        }
        Ok(header)
    }

    /// Reads the part of 6.2 for a frame with `frame_type` `NON_KEY_FRAME`.
    fn read_non_key_frame(
        &mut self,
        reader: &mut BitReader<'_>,
        refs: &RefFrameTable,
        header: &mut UncompressedHeader,
    ) -> Result<(), BitstreamError> {
        header.intra_only = !header.show_frame && reader.bit("intra_only")?;
        header.frame_is_intra = header.intra_only;
        if !header.error_resilient_mode {
            header.reset_frame_context = reader.u8(2, "reset_frame_context")?;
        }
        if header.intra_only {
            sync_code(reader)?;
            if header.profile > 0 {
                self.read_color_config(reader, header.profile)?;
            } else {
                // Profile 0 implies these values. `color_range` stays as it was.
                let color_range = self.color_config.is_some_and(|color| color.color_range);
                self.color_config = Some(ColorConfig {
                    bit_depth: 8,
                    color_space: CS_BT_601,
                    color_range,
                    subsampling_x: true,
                    subsampling_y: true,
                });
            }
            header.refresh_frame_flags = reader.u8(8, "refresh_frame_flags")?;
            self.read_frame_size(reader, header)?;
            read_render_size(reader, header)?;
            return Ok(());
        }
        let Some(color) = self.color_config else {
            return Err(reader.invalid(
                "frame_type",
                header.frame_type.into(),
                "a key or intra-only frame before the first inter frame (8.2)",
            ));
        };
        header.refresh_frame_flags = reader.u8(8, "refresh_frame_flags")?;
        for index in 0..3 {
            header.ref_frame_idx[index] = reader.u8(3, "ref_frame_idx")?;
            header.ref_frame_sign_bias[index + 1] = reader.bit("ref_frame_sign_bias")?;
        }
        self.read_frame_size_with_refs(reader, refs, header)?;
        check_reference_format(reader, refs, header, &color)?;
        header.allow_high_precision_mv = reader.bit("allow_high_precision_mv")?;
        header.is_filter_switchable = reader.bit("is_filter_switchable")?;
        if header.is_filter_switchable {
            header.interpolation_filter = SWITCHABLE;
        } else {
            header.raw_interpolation_filter = reader.u8(2, "raw_interpolation_filter")?;
            header.interpolation_filter =
                LITERAL_TO_TYPE[usize::from(header.raw_interpolation_filter)];
        }
        Ok(())
    }

    /// Reads `color_config()` (6.2.2) into the parser state.
    fn read_color_config(
        &mut self,
        reader: &mut BitReader<'_>,
        profile: u8,
    ) -> Result<(), BitstreamError> {
        let bit_depth = if profile >= 2 {
            if reader.bit("ten_or_twelve_bit")? {
                12
            } else {
                10
            }
        } else {
            8
        };
        let odd_profile = profile & 1 == 1;
        let color_space = reader.u8(3, "color_space")?;
        let mut color = ColorConfig {
            bit_depth,
            color_space,
            color_range: true,
            subsampling_x: true,
            subsampling_y: true,
        };
        if color_space != CS_RGB {
            color.color_range = reader.bit("color_range")?;
            if odd_profile {
                color.subsampling_x = reader.bit("subsampling_x")?;
                color.subsampling_y = reader.bit("subsampling_y")?;
                if color.subsampling_x && color.subsampling_y {
                    return Err(reader.invalid(
                        "subsampling_y",
                        1,
                        "0 when subsampling_x is 1 in profile 1 or 3 (7.2.2)",
                    ));
                }
                if reader.bit("reserved_zero")? {
                    return Err(reader.invalid("reserved_zero", 1, "0"));
                }
            }
        } else {
            if !odd_profile {
                return Err(reader.invalid(
                    "color_space",
                    color_space.into(),
                    "0 to 6 in profile 0 or 2 (7.2.2)",
                ));
            }
            color.subsampling_x = false;
            color.subsampling_y = false;
            if reader.bit("reserved_zero")? {
                return Err(reader.invalid("reserved_zero", 1, "0"));
            }
        }
        self.color_config = Some(color);
        Ok(())
    }

    /// Reads `frame_size()` (6.2.3).
    fn read_frame_size(
        &mut self,
        reader: &mut BitReader<'_>,
        header: &mut UncompressedHeader,
    ) -> Result<(), BitstreamError> {
        header.frame_width = reader.bits(16, "frame_width_minus_1")? as u32 + 1;
        header.frame_height = reader.bits(16, "frame_height_minus_1")? as u32 + 1;
        self.compute_image_size(header);
        Ok(())
    }

    /// Reads `frame_size_with_refs()` (6.2.5).
    fn read_frame_size_with_refs(
        &mut self,
        reader: &mut BitReader<'_>,
        refs: &RefFrameTable,
        header: &mut UncompressedHeader,
    ) -> Result<(), BitstreamError> {
        for index in 0..3u8 {
            if reader.bit("found_ref")? {
                let slot = usize::from(header.ref_frame_idx[usize::from(index)]);
                if refs.ref_frame_width[slot] == 0 {
                    return Err(reader.invalid(
                        "ref_frame_idx",
                        slot as u64,
                        "a slot that a previous frame refreshed",
                    ));
                }
                header.found_ref = Some(index);
                header.frame_width = refs.ref_frame_width[slot];
                header.frame_height = refs.ref_frame_height[slot];
                break;
            }
        }
        if header.found_ref.is_some() {
            self.compute_image_size(header);
        } else {
            self.read_frame_size(reader, header)?;
        }
        read_render_size(reader, header)
    }

    /// Runs `compute_image_size()` (6.2.6 and 7.2.6).
    fn compute_image_size(&mut self, header: &mut UncompressedHeader) {
        header.mi_cols = (header.frame_width + 7) >> 3;
        header.mi_rows = (header.frame_height + 7) >> 3;
        header.sb64_cols = (header.mi_cols + 7) >> 3;
        header.sb64_rows = (header.mi_rows + 7) >> 3;
        // A width of 0 means this is the first call: a frame width is above 0.
        header.size_changed =
            header.frame_width != self.last_width || header.frame_height != self.last_height;
        header.use_prev_frame_mvs = !header.size_changed
            && self.last_show_frame
            && !header.error_resilient_mode
            && !header.frame_is_intra;
        self.last_width = header.frame_width;
        self.last_height = header.frame_height;
        self.last_show_frame = header.show_frame;
    }

    /// Applies the parts of `setup_past_independence()` (7.2) that touch this
    /// parser's state.
    fn setup_past_independence(&mut self) {
        self.feature_enabled = [[false; SEG_LVL_MAX]; MAX_SEGMENTS];
        self.feature_data = [[0; SEG_LVL_MAX]; MAX_SEGMENTS];
        self.feature_sign = [[false; SEG_LVL_MAX]; MAX_SEGMENTS];
        self.segmentation_abs_or_delta_update = false;
        self.loop_filter_ref_deltas = [1, 0, -1, -1];
        self.loop_filter_mode_deltas = [0, 0];
    }

    /// Reads `loop_filter_params()` (6.2.8).
    fn read_loop_filter(
        &mut self,
        reader: &mut BitReader<'_>,
        header: &mut UncompressedHeader,
    ) -> Result<(), BitstreamError> {
        let filter = &mut header.loop_filter;
        filter.loop_filter_level = reader.u8(6, "loop_filter_level")?;
        filter.loop_filter_sharpness = reader.u8(3, "loop_filter_sharpness")?;
        filter.loop_filter_delta_enabled = reader.bit("loop_filter_delta_enabled")?;
        if filter.loop_filter_delta_enabled {
            filter.loop_filter_delta_update = reader.bit("loop_filter_delta_update")?;
            if filter.loop_filter_delta_update {
                for index in 0..4 {
                    filter.update_ref_delta[index] = reader.bit("update_ref_delta")?;
                    if filter.update_ref_delta[index] {
                        self.loop_filter_ref_deltas[index] =
                            read_signed(reader, 6, "loop_filter_ref_deltas")? as i8;
                    }
                }
                for index in 0..2 {
                    filter.update_mode_delta[index] = reader.bit("update_mode_delta")?;
                    if filter.update_mode_delta[index] {
                        self.loop_filter_mode_deltas[index] =
                            read_signed(reader, 6, "loop_filter_mode_deltas")? as i8;
                    }
                }
            }
        }
        filter.loop_filter_ref_deltas = self.loop_filter_ref_deltas;
        filter.loop_filter_mode_deltas = self.loop_filter_mode_deltas;
        Ok(())
    }

    /// Reads `segmentation_params()` (6.2.11).
    fn read_segmentation(
        &mut self,
        reader: &mut BitReader<'_>,
        header: &mut UncompressedHeader,
    ) -> Result<(), BitstreamError> {
        let params = &mut header.segmentation;
        params.segmentation_tree_probs = [255; 7];
        params.segmentation_pred_prob = [255; 3];
        params.segmentation_enabled = reader.bit("segmentation_enabled")?;
        if params.segmentation_enabled {
            params.segmentation_update_map = reader.bit("segmentation_update_map")?;
            if params.segmentation_update_map {
                for index in 0..7 {
                    params.segmentation_tree_prob_coded[index] = reader.bit("prob_coded")?;
                    if params.segmentation_tree_prob_coded[index] {
                        params.segmentation_tree_probs[index] =
                            reader.u8(8, "segmentation_tree_probs")?;
                    }
                }
                params.segmentation_temporal_update = reader.bit("segmentation_temporal_update")?;
                if params.segmentation_temporal_update {
                    for index in 0..3 {
                        params.segmentation_pred_prob_coded[index] = reader.bit("prob_coded")?;
                        if params.segmentation_pred_prob_coded[index] {
                            params.segmentation_pred_prob[index] =
                                reader.u8(8, "segmentation_pred_prob")?;
                        }
                    }
                }
            }
            params.segmentation_update_data = reader.bit("segmentation_update_data")?;
            if params.segmentation_update_data {
                self.segmentation_abs_or_delta_update =
                    reader.bit("segmentation_abs_or_delta_update")?;
                self.read_segmentation_features(reader)?;
            }
        }
        params.segmentation_abs_or_delta_update = self.segmentation_abs_or_delta_update;
        params.feature_enabled = self.feature_enabled;
        params.feature_data = self.feature_data;
        params.feature_sign = self.feature_sign;
        Ok(())
    }

    fn read_segmentation_features(
        &mut self,
        reader: &mut BitReader<'_>,
    ) -> Result<(), BitstreamError> {
        for segment in 0..MAX_SEGMENTS {
            for feature in 0..SEG_LVL_MAX {
                let enabled = reader.bit("feature_enabled")?;
                let mut value = 0i16;
                let mut sign = false;
                if enabled {
                    value =
                        reader.bits(SEGMENTATION_FEATURE_BITS[feature], "feature_value")? as i16;
                    if SEGMENTATION_FEATURE_SIGNED[feature] {
                        sign = reader.bit("feature_sign")?;
                        // 7.2.10: absolute values carry no sign.
                        if sign && self.segmentation_abs_or_delta_update {
                            return Err(reader.invalid(
                                "feature_sign",
                                1,
                                "0 when segmentation_abs_or_delta_update is 1",
                            ));
                        }
                        if sign {
                            value = -value;
                        }
                    }
                }
                self.feature_enabled[segment][feature] = enabled;
                self.feature_data[segment][feature] = value;
                self.feature_sign[segment][feature] = sign;
            }
        }
        Ok(())
    }
}

/// Reads `render_size()` (6.2.4).
fn read_render_size(
    reader: &mut BitReader<'_>,
    header: &mut UncompressedHeader,
) -> Result<(), BitstreamError> {
    header.render_and_frame_size_different = reader.bit("render_and_frame_size_different")?;
    if header.render_and_frame_size_different {
        header.render_width = reader.bits(16, "render_width_minus_1")? as u32 + 1;
        header.render_height = reader.bits(16, "render_height_minus_1")? as u32 + 1;
    } else {
        header.render_width = header.frame_width;
        header.render_height = header.frame_height;
    }
    Ok(())
}

/// Checks the rules of 6.2 and 7.2.5 on the three references of an inter frame:
/// at least one has a valid scale, and each filled slot matches this frame's
/// bit depth and subsampling.
fn check_reference_format(
    reader: &BitReader<'_>,
    refs: &RefFrameTable,
    header: &UncompressedHeader,
    color: &ColorConfig,
) -> Result<(), BitstreamError> {
    let width = u64::from(header.frame_width);
    let height = u64::from(header.frame_height);
    let mut any_valid_scale = false;
    for &slot in &header.ref_frame_idx {
        let slot = usize::from(slot);
        let ref_width = u64::from(refs.ref_frame_width[slot]);
        let ref_height = u64::from(refs.ref_frame_height[slot]);
        if ref_width == 0 {
            continue;
        }
        any_valid_scale |= 2 * width >= ref_width
            && 2 * height >= ref_height
            && width <= 16 * ref_width
            && height <= 16 * ref_height;
        if refs.ref_bit_depth[slot] != color.bit_depth
            || refs.ref_subsampling_x[slot] != color.subsampling_x
            || refs.ref_subsampling_y[slot] != color.subsampling_y
        {
            return Err(reader.invalid(
                "ref_frame_idx",
                slot as u64,
                "a slot with the bit depth and subsampling of this frame",
            ));
        }
    }
    if !any_valid_scale {
        return Err(reader.invalid(
            "frame_width_minus_1",
            u64::from(header.frame_width) - 1,
            "a size within the scale limits of at least one reference (7.2.5)",
        ));
    }
    Ok(())
}

/// Reads a VP9 `s(n)` (4.9.2): `count` magnitude bits, then a sign bit.
fn read_signed(
    reader: &mut BitReader<'_>,
    count: u32,
    field: &'static str,
) -> Result<i16, BitstreamError> {
    let magnitude = reader.bits(count, field)? as i16;
    Ok(if reader.bit(field)? {
        -magnitude
    } else {
        magnitude
    })
}

/// Reads `read_delta_q()` (6.2.10) and returns `delta_coded` and `delta_q`.
fn read_delta_q(reader: &mut BitReader<'_>) -> Result<(bool, i8), BitstreamError> {
    if reader.bit("delta_coded")? {
        Ok((true, read_signed(reader, 4, "delta_q")? as i8))
    } else {
        Ok((false, 0))
    }
}

/// Reads `quantization_params()` (6.2.9).
fn read_quantization(
    reader: &mut BitReader<'_>,
    header: &mut UncompressedHeader,
) -> Result<(), BitstreamError> {
    let params = &mut header.quantization;
    params.base_q_idx = reader.u8(8, "base_q_idx")?;
    let (y_dc_coded, y_dc) = read_delta_q(reader)?;
    let (uv_dc_coded, uv_dc) = read_delta_q(reader)?;
    let (uv_ac_coded, uv_ac) = read_delta_q(reader)?;
    params.delta_q_y_dc = y_dc;
    params.delta_q_uv_dc = uv_dc;
    params.delta_q_uv_ac = uv_ac;
    params.delta_coded = [y_dc_coded, uv_dc_coded, uv_ac_coded];
    params.lossless = params.base_q_idx == 0 && y_dc == 0 && uv_dc == 0 && uv_ac == 0;
    Ok(())
}

/// Reads `tile_info()` (6.2.13). `compute_image_size()` runs before this.
fn read_tile_info(
    reader: &mut BitReader<'_>,
    header: &mut UncompressedHeader,
) -> Result<(), BitstreamError> {
    let sb64_cols = header.sb64_cols;
    let mut min_log2 = 0u32;
    while (MAX_TILE_WIDTH_B64 << min_log2) < sb64_cols {
        min_log2 += 1;
    }
    let mut max_log2 = 1u32;
    while (sb64_cols >> max_log2) >= MIN_TILE_WIDTH_B64 {
        max_log2 += 1;
    }
    max_log2 -= 1;
    let mut tile_cols_log2 = min_log2;
    while tile_cols_log2 < max_log2 {
        if reader.bit("increment_tile_cols_log2")? {
            tile_cols_log2 += 1;
        } else {
            break;
        }
    }
    // 7.2.11.
    if tile_cols_log2 > 6 {
        return Err(reader.invalid("tile_cols_log2", tile_cols_log2.into(), "0 to 6"));
    }
    header.tile_cols_log2 = tile_cols_log2 as u8;
    header.tile_rows_log2 = u8::from(reader.bit("tile_rows_log2")?);
    if header.tile_rows_log2 == 1 {
        header.tile_rows_log2 += u8::from(reader.bit("increment_tile_rows_log2")?);
    }
    Ok(())
}

#[cfg(test)]
#[path = "vp9_header_tests.rs"]
mod tests;
