//! HEVC `pic_parameter_set_rbsp()` (ITU-T H.265 7.3.2.3) for `nuh_layer_id` 0.
//!
//! A PPS can precede the SPS it refers to. [`PicParameterSet::parse`] reads the
//! syntax and checks the ranges that need no SPS.
//! [`PicParameterSet::check_with_sps`] checks the rest when a slice activates
//! the PPS.

use super::bits::BitReader;
use super::hevc_params;
use super::hevc_scaling::ScalingListData;
use super::hevc_sps_full::SeqParameterSet;
use super::nal::HevcNalType;
use crate::error::BitstreamError;

const STRUCTURE: &str = "HEVC PPS";
/// The largest `MaxTileCols` of any level (Table A.8).
const MAX_TILE_COLS: u32 = 40;
/// The largest `MaxTileRows` of any level (Table A.8).
const MAX_TILE_ROWS: u32 = 44;
/// The lowest `init_qp_minus26` at any bit depth: `QpBdOffsetY` is at most 48.
const MIN_INIT_QP_MINUS26: i64 = -74;

/// `pps_range_extension()` (7.3.2.3.2). The parser reads it. A decoder without
/// these tools calls [`PpsRangeExtension::refuse_tools`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PpsRangeExtension {
    pub(crate) log2_max_transform_skip_block_size_minus2: u32,
    pub(crate) cross_component_prediction_enabled_flag: bool,
    pub(crate) chroma_qp_offset_list_enabled_flag: bool,
    pub(crate) diff_cu_chroma_qp_offset_depth: u32,
    pub(crate) chroma_qp_offset_list_len_minus1: u32,
    pub(crate) cb_qp_offset_list: [i8; 6],
    pub(crate) cr_qp_offset_list: [i8; 6],
    pub(crate) log2_sao_offset_scale_luma: u32,
    pub(crate) log2_sao_offset_scale_chroma: u32,
}

impl PpsRangeExtension {
    fn parse(
        reader: &mut BitReader<'_>,
        transform_skip_enabled_flag: bool,
    ) -> Result<Self, BitstreamError> {
        let mut extension = Self::default();
        if transform_skip_enabled_flag {
            extension.log2_max_transform_skip_block_size_minus2 =
                reader.ue("log2_max_transform_skip_block_size_minus2")?;
        }
        extension.cross_component_prediction_enabled_flag =
            reader.bit("cross_component_prediction_enabled_flag")?;
        extension.chroma_qp_offset_list_enabled_flag =
            reader.bit("chroma_qp_offset_list_enabled_flag")?;
        if extension.chroma_qp_offset_list_enabled_flag {
            extension.diff_cu_chroma_qp_offset_depth =
                reader.ue("diff_cu_chroma_qp_offset_depth")?;
            extension.chroma_qp_offset_list_len_minus1 =
                reader.ue_max("chroma_qp_offset_list_len_minus1", 5, "0 to 5")?;
            for index in 0..=extension.chroma_qp_offset_list_len_minus1 as usize {
                extension.cb_qp_offset_list[index] =
                    reader.se_range("cb_qp_offset_list", -12, 12, "-12 to 12")? as i8;
                extension.cr_qp_offset_list[index] =
                    reader.se_range("cr_qp_offset_list", -12, 12, "-12 to 12")? as i8;
            }
        }
        extension.log2_sao_offset_scale_luma = reader.ue("log2_sao_offset_scale_luma")?;
        extension.log2_sao_offset_scale_chroma = reader.ue("log2_sao_offset_scale_chroma")?;
        Ok(extension)
    }

    /// Refuses a tool that changes decoding, for a decoder without them.
    pub(crate) fn refuse_tools(&self) -> Result<(), BitstreamError> {
        let values = [
            (
                "log2_max_transform_skip_block_size_minus2",
                self.log2_max_transform_skip_block_size_minus2,
            ),
            (
                "cross_component_prediction_enabled_flag",
                u32::from(self.cross_component_prediction_enabled_flag),
            ),
            (
                "chroma_qp_offset_list_enabled_flag",
                u32::from(self.chroma_qp_offset_list_enabled_flag),
            ),
            (
                "log2_sao_offset_scale_luma",
                self.log2_sao_offset_scale_luma,
            ),
            (
                "log2_sao_offset_scale_chroma",
                self.log2_sao_offset_scale_chroma,
            ),
        ];
        match values.into_iter().find(|(_, value)| *value != 0) {
            Some((field, value)) => Err(invalid(
                field,
                value.into(),
                "0, because this decoder has no range extension tools",
            )),
            None => Ok(()),
        }
    }
}

/// One `pic_parameter_set_rbsp()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PicParameterSet {
    pub(crate) pps_pic_parameter_set_id: u32,
    pub(crate) pps_seq_parameter_set_id: u32,
    pub(crate) dependent_slice_segments_enabled_flag: bool,
    pub(crate) output_flag_present_flag: bool,
    pub(crate) num_extra_slice_header_bits: u8,
    pub(crate) sign_data_hiding_enabled_flag: bool,
    pub(crate) cabac_init_present_flag: bool,
    pub(crate) num_ref_idx_l0_default_active_minus1: u32,
    pub(crate) num_ref_idx_l1_default_active_minus1: u32,
    pub(crate) init_qp_minus26: i32,
    pub(crate) constrained_intra_pred_flag: bool,
    pub(crate) transform_skip_enabled_flag: bool,
    pub(crate) cu_qp_delta_enabled_flag: bool,
    pub(crate) diff_cu_qp_delta_depth: u32,
    pub(crate) pps_cb_qp_offset: i32,
    pub(crate) pps_cr_qp_offset: i32,
    pub(crate) pps_slice_chroma_qp_offsets_present_flag: bool,
    pub(crate) weighted_pred_flag: bool,
    pub(crate) weighted_bipred_flag: bool,
    pub(crate) transquant_bypass_enabled_flag: bool,
    pub(crate) tiles_enabled_flag: bool,
    pub(crate) entropy_coding_sync_enabled_flag: bool,
    pub(crate) num_tile_columns_minus1: u32,
    pub(crate) num_tile_rows_minus1: u32,
    /// Inferred as 1 without tiles.
    pub(crate) uniform_spacing_flag: bool,
    /// `num_tile_columns_minus1` entries without uniform spacing, else empty.
    pub(crate) column_width_minus1: Vec<u32>,
    /// `num_tile_rows_minus1` entries without uniform spacing, else empty.
    pub(crate) row_height_minus1: Vec<u32>,
    /// Inferred as 1 without tiles.
    pub(crate) loop_filter_across_tiles_enabled_flag: bool,
    pub(crate) pps_loop_filter_across_slices_enabled_flag: bool,
    pub(crate) deblocking_filter_control_present_flag: bool,
    pub(crate) deblocking_filter_override_enabled_flag: bool,
    pub(crate) pps_deblocking_filter_disabled_flag: bool,
    pub(crate) pps_beta_offset_div2: i32,
    pub(crate) pps_tc_offset_div2: i32,
    pub(crate) pps_scaling_list_data_present_flag: bool,
    /// `None` without `pps_scaling_list_data_present_flag`.
    pub(crate) scaling_list_data: Option<Box<ScalingListData>>,
    pub(crate) lists_modification_present_flag: bool,
    pub(crate) log2_parallel_merge_level_minus2: u32,
    pub(crate) slice_segment_header_extension_present_flag: bool,
    pub(crate) pps_extension_present_flag: bool,
    pub(crate) pps_range_extension_flag: bool,
    pub(crate) pps_multilayer_extension_flag: bool,
    pub(crate) pps_3d_extension_flag: bool,
    pub(crate) pps_extension_4bits: u8,
    pub(crate) pps_range_extension: PpsRangeExtension,
}

fn invalid(field: &'static str, value: u64, allowed: &'static str) -> BitstreamError {
    BitstreamError::InvalidField {
        structure: STRUCTURE,
        field,
        value,
        allowed,
    }
}

fn invalid_signed(field: &'static str, value: i64, allowed: &'static str) -> BitstreamError {
    BitstreamError::InvalidSignedField {
        structure: STRUCTURE,
        field,
        value,
        allowed,
    }
}

impl PicParameterSet {
    /// Reads a PPS NAL unit, 2-byte header included.
    pub(crate) fn parse(nal: &[u8]) -> Result<Self, BitstreamError> {
        let rbsp = hevc_params::payload(nal, HevcNalType::PPS, STRUCTURE)?;
        let mut reader = BitReader::new(&rbsp, STRUCTURE);
        let reader = &mut reader;
        let pps_pic_parameter_set_id = reader.ue_max("pps_pic_parameter_set_id", 63, "0 to 63")?;
        let pps_seq_parameter_set_id = reader.ue_max("pps_seq_parameter_set_id", 15, "0 to 15")?;
        let dependent_slice_segments_enabled_flag =
            reader.bit("dependent_slice_segments_enabled_flag")?;
        let output_flag_present_flag = reader.bit("output_flag_present_flag")?;
        // A decoder accepts any value of this field (7.4.3.3.1).
        let num_extra_slice_header_bits = reader.u8(3, "num_extra_slice_header_bits")?;
        let sign_data_hiding_enabled_flag = reader.bit("sign_data_hiding_enabled_flag")?;
        let cabac_init_present_flag = reader.bit("cabac_init_present_flag")?;
        let num_ref_idx_l0_default_active_minus1 =
            reader.ue_max("num_ref_idx_l0_default_active_minus1", 14, "0 to 14")?;
        let num_ref_idx_l1_default_active_minus1 =
            reader.ue_max("num_ref_idx_l1_default_active_minus1", 14, "0 to 14")?;
        let init_qp_minus26 = reader.se_range(
            "init_qp_minus26",
            MIN_INIT_QP_MINUS26,
            25,
            "-( 26 + QpBdOffsetY ) to 25",
        )? as i32;
        let constrained_intra_pred_flag = reader.bit("constrained_intra_pred_flag")?;
        let transform_skip_enabled_flag = reader.bit("transform_skip_enabled_flag")?;
        let cu_qp_delta_enabled_flag = reader.bit("cu_qp_delta_enabled_flag")?;
        let diff_cu_qp_delta_depth = if cu_qp_delta_enabled_flag {
            reader.ue("diff_cu_qp_delta_depth")?
        } else {
            0
        };
        let pps_cb_qp_offset = reader.se_range("pps_cb_qp_offset", -12, 12, "-12 to 12")? as i32;
        let pps_cr_qp_offset = reader.se_range("pps_cr_qp_offset", -12, 12, "-12 to 12")? as i32;
        let pps_slice_chroma_qp_offsets_present_flag =
            reader.bit("pps_slice_chroma_qp_offsets_present_flag")?;
        let weighted_pred_flag = reader.bit("weighted_pred_flag")?;
        let weighted_bipred_flag = reader.bit("weighted_bipred_flag")?;
        let transquant_bypass_enabled_flag = reader.bit("transquant_bypass_enabled_flag")?;
        let tiles_enabled_flag = reader.bit("tiles_enabled_flag")?;
        let entropy_coding_sync_enabled_flag = reader.bit("entropy_coding_sync_enabled_flag")?;

        let mut pps = Self {
            pps_pic_parameter_set_id,
            pps_seq_parameter_set_id,
            dependent_slice_segments_enabled_flag,
            output_flag_present_flag,
            num_extra_slice_header_bits,
            sign_data_hiding_enabled_flag,
            cabac_init_present_flag,
            num_ref_idx_l0_default_active_minus1,
            num_ref_idx_l1_default_active_minus1,
            init_qp_minus26,
            constrained_intra_pred_flag,
            transform_skip_enabled_flag,
            cu_qp_delta_enabled_flag,
            diff_cu_qp_delta_depth,
            pps_cb_qp_offset,
            pps_cr_qp_offset,
            pps_slice_chroma_qp_offsets_present_flag,
            weighted_pred_flag,
            weighted_bipred_flag,
            transquant_bypass_enabled_flag,
            tiles_enabled_flag,
            entropy_coding_sync_enabled_flag,
            num_tile_columns_minus1: 0,
            num_tile_rows_minus1: 0,
            uniform_spacing_flag: true,
            column_width_minus1: Vec::new(),
            row_height_minus1: Vec::new(),
            loop_filter_across_tiles_enabled_flag: true,
            pps_loop_filter_across_slices_enabled_flag: false,
            deblocking_filter_control_present_flag: false,
            deblocking_filter_override_enabled_flag: false,
            pps_deblocking_filter_disabled_flag: false,
            pps_beta_offset_div2: 0,
            pps_tc_offset_div2: 0,
            pps_scaling_list_data_present_flag: false,
            scaling_list_data: None,
            lists_modification_present_flag: false,
            log2_parallel_merge_level_minus2: 0,
            slice_segment_header_extension_present_flag: false,
            pps_extension_present_flag: false,
            pps_range_extension_flag: false,
            pps_multilayer_extension_flag: false,
            pps_3d_extension_flag: false,
            pps_extension_4bits: 0,
            pps_range_extension: PpsRangeExtension::default(),
        };
        if tiles_enabled_flag {
            pps.read_tiles(reader)?;
        }
        pps.pps_loop_filter_across_slices_enabled_flag =
            reader.bit("pps_loop_filter_across_slices_enabled_flag")?;
        pps.deblocking_filter_control_present_flag =
            reader.bit("deblocking_filter_control_present_flag")?;
        if pps.deblocking_filter_control_present_flag {
            pps.deblocking_filter_override_enabled_flag =
                reader.bit("deblocking_filter_override_enabled_flag")?;
            pps.pps_deblocking_filter_disabled_flag =
                reader.bit("pps_deblocking_filter_disabled_flag")?;
            if !pps.pps_deblocking_filter_disabled_flag {
                pps.pps_beta_offset_div2 =
                    reader.se_range("pps_beta_offset_div2", -6, 6, "-6 to 6")? as i32;
                pps.pps_tc_offset_div2 =
                    reader.se_range("pps_tc_offset_div2", -6, 6, "-6 to 6")? as i32;
            }
        }
        pps.pps_scaling_list_data_present_flag =
            reader.bit("pps_scaling_list_data_present_flag")?;
        if pps.pps_scaling_list_data_present_flag {
            pps.scaling_list_data = Some(Box::new(ScalingListData::parse(reader)?));
        }
        pps.lists_modification_present_flag = reader.bit("lists_modification_present_flag")?;
        pps.log2_parallel_merge_level_minus2 = reader.ue("log2_parallel_merge_level_minus2")?;
        pps.slice_segment_header_extension_present_flag =
            reader.bit("slice_segment_header_extension_present_flag")?;
        pps.pps_extension_present_flag = reader.bit("pps_extension_present_flag")?;
        if pps.pps_extension_present_flag {
            pps.read_extensions(reader)?;
        }
        Ok(pps)
    }

    fn read_tiles(&mut self, reader: &mut BitReader<'_>) -> Result<(), BitstreamError> {
        self.num_tile_columns_minus1 =
            reader.ue_max("num_tile_columns_minus1", MAX_TILE_COLS - 1, "0 to 39")?;
        self.num_tile_rows_minus1 =
            reader.ue_max("num_tile_rows_minus1", MAX_TILE_ROWS - 1, "0 to 43")?;
        if self.num_tile_columns_minus1 == 0 && self.num_tile_rows_minus1 == 0 {
            return Err(invalid(
                "num_tile_rows_minus1",
                0,
                "above 0 when num_tile_columns_minus1 is 0 and tiles_enabled_flag is 1",
            ));
        }
        self.uniform_spacing_flag = reader.bit("uniform_spacing_flag")?;
        if !self.uniform_spacing_flag {
            self.column_width_minus1 =
                read_list(reader, self.num_tile_columns_minus1, "column_width_minus1")?;
            self.row_height_minus1 =
                read_list(reader, self.num_tile_rows_minus1, "row_height_minus1")?;
        }
        self.loop_filter_across_tiles_enabled_flag =
            reader.bit("loop_filter_across_tiles_enabled_flag")?;
        Ok(())
    }

    fn read_extensions(&mut self, reader: &mut BitReader<'_>) -> Result<(), BitstreamError> {
        self.pps_range_extension_flag = reader.bit("pps_range_extension_flag")?;
        self.pps_multilayer_extension_flag = reader.bit("pps_multilayer_extension_flag")?;
        self.pps_3d_extension_flag = reader.bit("pps_3d_extension_flag")?;
        if reader.bit("pps_scc_extension_flag")? {
            // Screen content coding changes the slice header syntax.
            return Err(invalid(
                "pps_scc_extension_flag",
                1,
                "0, because vidr does not read screen content coding streams",
            ));
        }
        self.pps_extension_4bits = reader.u8(4, "pps_extension_4bits")?;
        if self.pps_range_extension_flag {
            self.pps_range_extension =
                PpsRangeExtension::parse(reader, self.transform_skip_enabled_flag)?;
        }
        // The multilayer and 3D extensions and extension data do not affect
        // layer 0.
        Ok(())
    }

    /// Checks the ranges that depend on the SPS this PPS refers to.
    pub(crate) fn check_with_sps(&self, sps: &SeqParameterSet) -> Result<(), BitstreamError> {
        if self.pps_seq_parameter_set_id != sps.sps_seq_parameter_set_id {
            return Err(invalid(
                "pps_seq_parameter_set_id",
                self.pps_seq_parameter_set_id.into(),
                "the sps_seq_parameter_set_id of the SPS passed in",
            ));
        }
        let lowest_qp = -(26 + i64::from(sps.qp_bd_offset_y()));
        if i64::from(self.init_qp_minus26) < lowest_qp {
            return Err(invalid_signed(
                "init_qp_minus26",
                self.init_qp_minus26.into(),
                "-( 26 + QpBdOffsetY ) to 25",
            ));
        }
        let depth_limit = sps.log2_diff_max_min_luma_coding_block_size;
        if self.diff_cu_qp_delta_depth > depth_limit {
            return Err(invalid(
                "diff_cu_qp_delta_depth",
                self.diff_cu_qp_delta_depth.into(),
                "0 to log2_diff_max_min_luma_coding_block_size",
            ));
        }
        let chroma = sps.chroma_array_type();
        if chroma == 0 && self.pps_slice_chroma_qp_offsets_present_flag {
            return Err(invalid(
                "pps_slice_chroma_qp_offsets_present_flag",
                1,
                "0 when ChromaArrayType is 0",
            ));
        }
        if self.pps_scaling_list_data_present_flag && !sps.scaling_list_enabled_flag {
            return Err(invalid(
                "pps_scaling_list_data_present_flag",
                1,
                "0 when scaling_list_enabled_flag is 0",
            ));
        }
        if self.log2_parallel_merge_level_minus2 > sps.ctb_log2_size_y() - 2 {
            return Err(invalid(
                "log2_parallel_merge_level_minus2",
                self.log2_parallel_merge_level_minus2.into(),
                "0 to CtbLog2SizeY - 2",
            ));
        }
        self.check_tiles(sps)?;
        self.check_range_extension(sps)
    }

    fn check_tiles(&self, sps: &SeqParameterSet) -> Result<(), BitstreamError> {
        let width = sps.pic_width_in_ctbs_y();
        let height = sps.pic_height_in_ctbs_y();
        if self.num_tile_columns_minus1 >= width {
            return Err(invalid(
                "num_tile_columns_minus1",
                self.num_tile_columns_minus1.into(),
                "0 to PicWidthInCtbsY - 1",
            ));
        }
        if self.num_tile_rows_minus1 >= height {
            return Err(invalid(
                "num_tile_rows_minus1",
                self.num_tile_rows_minus1.into(),
                "0 to PicHeightInCtbsY - 1",
            ));
        }
        // Each explicit size leaves at least one CTB for the last tile (6-3).
        if explicit_total(&self.column_width_minus1) >= u64::from(width) {
            return Err(invalid(
                "column_width_minus1",
                explicit_total(&self.column_width_minus1),
                "sizes that leave at least one CTB for the last column",
            ));
        }
        if explicit_total(&self.row_height_minus1) >= u64::from(height) {
            return Err(invalid(
                "row_height_minus1",
                explicit_total(&self.row_height_minus1),
                "sizes that leave at least one CTB for the last row",
            ));
        }
        Ok(())
    }

    fn check_range_extension(&self, sps: &SeqParameterSet) -> Result<(), BitstreamError> {
        let extension = &self.pps_range_extension;
        if extension.log2_max_transform_skip_block_size_minus2 > sps.max_tb_log2_size_y() - 2 {
            return Err(invalid(
                "log2_max_transform_skip_block_size_minus2",
                extension.log2_max_transform_skip_block_size_minus2.into(),
                "0 to MaxTbLog2SizeY - 2",
            ));
        }
        let chroma = sps.chroma_array_type();
        if extension.cross_component_prediction_enabled_flag && chroma != 3 {
            return Err(invalid(
                "cross_component_prediction_enabled_flag",
                1,
                "0 when ChromaArrayType is not 3",
            ));
        }
        if extension.chroma_qp_offset_list_enabled_flag && chroma == 0 {
            return Err(invalid(
                "chroma_qp_offset_list_enabled_flag",
                1,
                "0 when ChromaArrayType is 0",
            ));
        }
        if extension.diff_cu_chroma_qp_offset_depth > sps.log2_diff_max_min_luma_coding_block_size {
            return Err(invalid(
                "diff_cu_chroma_qp_offset_depth",
                extension.diff_cu_chroma_qp_offset_depth.into(),
                "0 to log2_diff_max_min_luma_coding_block_size",
            ));
        }
        if extension.log2_sao_offset_scale_luma > sps.bit_depth_y().saturating_sub(10) {
            return Err(invalid(
                "log2_sao_offset_scale_luma",
                extension.log2_sao_offset_scale_luma.into(),
                "0 to Max( 0, BitDepthY - 10 )",
            ));
        }
        if extension.log2_sao_offset_scale_chroma > sps.bit_depth_c().saturating_sub(10) {
            return Err(invalid(
                "log2_sao_offset_scale_chroma",
                extension.log2_sao_offset_scale_chroma.into(),
                "0 to Max( 0, BitDepthC - 10 )",
            ));
        }
        Ok(())
    }

    /// `colWidth` (6-3), in CTBs. Call after [`Self::check_with_sps`].
    pub(crate) fn column_widths(&self, sps: &SeqParameterSet) -> Vec<u32> {
        tile_sizes(
            self.uniform_spacing_flag,
            self.num_tile_columns_minus1,
            &self.column_width_minus1,
            sps.pic_width_in_ctbs_y(),
        )
    }

    /// `rowHeight` (6-4), in CTBs. Call after [`Self::check_with_sps`].
    pub(crate) fn row_heights(&self, sps: &SeqParameterSet) -> Vec<u32> {
        tile_sizes(
            self.uniform_spacing_flag,
            self.num_tile_rows_minus1,
            &self.row_height_minus1,
            sps.pic_height_in_ctbs_y(),
        )
    }
}

/// Reads `count` `ue(v)` values. Each code takes at least one bit, so a count
/// above the bits left is a truncation. This check runs before the allocation.
fn read_list(
    reader: &mut BitReader<'_>,
    count: u32,
    field: &'static str,
) -> Result<Vec<u32>, BitstreamError> {
    if u64::from(count) > reader.bits_left() {
        return Err(reader.truncated(field));
    }
    let mut list = Vec::with_capacity(count as usize);
    for _ in 0..count {
        list.push(reader.ue(field)?);
    }
    Ok(list)
}

/// The CTBs that explicit tile sizes take.
fn explicit_total(minus1: &[u32]) -> u64 {
    minus1.iter().map(|size| u64::from(*size) + 1).sum()
}

/// (6-3) and (6-4).
fn tile_sizes(uniform: bool, count_minus1: u32, explicit: &[u32], total: u32) -> Vec<u32> {
    let count = count_minus1 + 1;
    if uniform {
        return (0..count)
            .map(|index| (index + 1) * total / count - index * total / count)
            .collect();
    }
    let mut sizes: Vec<u32> = explicit.iter().map(|size| size + 1).collect();
    let used: u32 = sizes.iter().sum();
    sizes.push(total - used);
    sizes
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::hevc_params::tests::X265_PPS;
    use super::super::hevc_sps_full::tests::{Spec as SpsSpec, valid as valid_sps};
    use super::super::test_bits::{BitWriter, hevc_nal};
    use super::*;

    /// Columns minus 1, rows minus 1, and explicit sizes.
    pub(crate) type TileSpec = (u32, u32, Option<(Vec<u32>, Vec<u32>)>);

    /// The values a test varies. The defaults make a valid plain PPS.
    #[derive(Clone)]
    pub(crate) struct Spec {
        pub(crate) pps_id: u32,
        pub(crate) sps_id: u32,
        pub(crate) dependent_slices: bool,
        pub(crate) output_flag: bool,
        pub(crate) extra_bits: u64,
        pub(crate) cabac_init: bool,
        pub(crate) l0_default: u32,
        pub(crate) l1_default: u32,
        pub(crate) init_qp: i64,
        pub(crate) transform_skip: bool,
        pub(crate) qp_delta_depth: Option<u32>,
        pub(crate) cb_offset: i64,
        pub(crate) cr_offset: i64,
        pub(crate) slice_chroma_offsets: bool,
        pub(crate) weighted: (bool, bool),
        pub(crate) wavefront: bool,
        pub(crate) tiles: Option<TileSpec>,
        pub(crate) loop_filter_across_slices: bool,
        /// Override enabled, disabled, beta, and tc.
        pub(crate) deblock: Option<(bool, bool, i64, i64)>,
        pub(crate) scaling: bool,
        pub(crate) lists_modification: bool,
        pub(crate) merge_level: u32,
        pub(crate) header_extension: bool,
        pub(crate) range_extension: Option<RangeSpec>,
        pub(crate) scc: bool,
    }

    /// The values of `pps_range_extension()`.
    #[derive(Clone, Default)]
    pub(crate) struct RangeSpec {
        pub(crate) transform_skip_size: u32,
        pub(crate) cross_component: bool,
        /// Depth, then the Cb and Cr offsets.
        pub(crate) chroma_list: Option<(u32, Vec<(i64, i64)>)>,
        pub(crate) sao_luma: u32,
        pub(crate) sao_chroma: u32,
    }

    impl Default for Spec {
        fn default() -> Self {
            Self {
                pps_id: 0,
                sps_id: 0,
                dependent_slices: false,
                output_flag: false,
                extra_bits: 0,
                cabac_init: false,
                l0_default: 0,
                l1_default: 0,
                init_qp: 0,
                transform_skip: false,
                qp_delta_depth: None,
                cb_offset: 0,
                cr_offset: 0,
                slice_chroma_offsets: false,
                weighted: (false, false),
                wavefront: false,
                tiles: None,
                loop_filter_across_slices: true,
                deblock: None,
                scaling: false,
                lists_modification: false,
                merge_level: 0,
                header_extension: false,
                range_extension: None,
                scc: false,
            }
        }
    }

    pub(crate) fn build(spec: &Spec) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.ue(spec.pps_id)
            .ue(spec.sps_id)
            .flag(spec.dependent_slices)
            .flag(spec.output_flag)
            .u(3, spec.extra_bits)
            .flag(true)
            .flag(spec.cabac_init)
            .ue(spec.l0_default)
            .ue(spec.l1_default)
            .se(spec.init_qp)
            .flag(false)
            .flag(spec.transform_skip)
            .flag(spec.qp_delta_depth.is_some());
        if let Some(depth) = spec.qp_delta_depth {
            w.ue(depth);
        }
        w.se(spec.cb_offset)
            .se(spec.cr_offset)
            .flag(spec.slice_chroma_offsets)
            .flag(spec.weighted.0)
            .flag(spec.weighted.1)
            .flag(false)
            .flag(spec.tiles.is_some())
            .flag(spec.wavefront);
        if let Some((columns, rows, explicit)) = &spec.tiles {
            w.ue(*columns).ue(*rows).flag(explicit.is_none());
            if let Some((widths, heights)) = explicit {
                for size in widths.iter().chain(heights) {
                    w.ue(*size);
                }
            }
            w.flag(false);
        }
        w.flag(spec.loop_filter_across_slices);
        w.flag(spec.deblock.is_some());
        if let Some((override_enabled, disabled, beta, tc)) = spec.deblock {
            w.flag(override_enabled).flag(disabled);
            if !disabled {
                w.se(beta).se(tc);
            }
        }
        w.flag(spec.scaling);
        if spec.scaling {
            for _ in 0..20 {
                w.flag(false).ue(0);
            }
        }
        w.flag(spec.lists_modification)
            .ue(spec.merge_level)
            .flag(spec.header_extension);
        let extension = spec.range_extension.is_some() || spec.scc;
        w.flag(extension);
        if extension {
            w.flag(spec.range_extension.is_some())
                .flag(false)
                .flag(false)
                .flag(spec.scc)
                .u(4, 0);
        }
        if let Some(range) = &spec.range_extension {
            if spec.transform_skip {
                w.ue(range.transform_skip_size);
            }
            w.flag(range.cross_component)
                .flag(range.chroma_list.is_some());
            if let Some((depth, offsets)) = &range.chroma_list {
                w.ue(*depth).ue(offsets.len() as u32 - 1);
                for (cb, cr) in offsets {
                    w.se(*cb).se(*cr);
                }
            }
            w.ue(range.sao_luma).ue(range.sao_chroma);
        }
        w.trailing();
        hevc_nal(34, &w.bytes())
    }

    pub(crate) fn valid(spec: &Spec) -> PicParameterSet {
        PicParameterSet::parse(&build(spec)).unwrap_or_else(|error| panic!("{error}"))
    }

    #[test]
    fn reads_a_real_pps() {
        let pps = PicParameterSet::parse(&X265_PPS).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(pps.pps_pic_parameter_set_id, 0);
        assert!(pps.sign_data_hiding_enabled_flag);
        assert!(pps.pps_loop_filter_across_slices_enabled_flag);
        assert!(!pps.tiles_enabled_flag);
        assert!(pps.uniform_spacing_flag);
    }

    #[test]
    fn reads_tiles_deblock_scaling_and_extensions() {
        let pps = valid(&Spec {
            pps_id: 7,
            sps_id: 2,
            dependent_slices: true,
            output_flag: true,
            extra_bits: 5,
            l0_default: 14,
            l1_default: 3,
            init_qp: -26,
            transform_skip: true,
            qp_delta_depth: Some(1),
            cb_offset: 12,
            cr_offset: -12,
            slice_chroma_offsets: true,
            weighted: (true, false),
            wavefront: true,
            tiles: Some((2, 1, Some((vec![1, 0], vec![0])))),
            deblock: Some((true, false, -6, 6)),
            scaling: true,
            lists_modification: true,
            merge_level: 2,
            header_extension: true,
            range_extension: Some(RangeSpec {
                transform_skip_size: 2,
                chroma_list: Some((1, vec![(3, -3), (-12, 12)])),
                ..RangeSpec::default()
            }),
            ..Spec::default()
        });
        assert_eq!(pps.pps_pic_parameter_set_id, 7);
        assert_eq!(pps.pps_seq_parameter_set_id, 2);
        assert_eq!(pps.num_extra_slice_header_bits, 5);
        assert_eq!(pps.num_ref_idx_l0_default_active_minus1, 14);
        assert_eq!(pps.init_qp_minus26, -26);
        assert_eq!(pps.diff_cu_qp_delta_depth, 1);
        assert_eq!((pps.pps_cb_qp_offset, pps.pps_cr_qp_offset), (12, -12));
        assert!(pps.weighted_pred_flag && !pps.weighted_bipred_flag);
        assert!(pps.tiles_enabled_flag && pps.entropy_coding_sync_enabled_flag);
        assert!(!pps.uniform_spacing_flag);
        assert_eq!(pps.column_width_minus1, vec![1, 0]);
        assert_eq!(pps.row_height_minus1, vec![0]);
        assert!(!pps.loop_filter_across_tiles_enabled_flag);
        assert!(pps.deblocking_filter_override_enabled_flag);
        assert_eq!((pps.pps_beta_offset_div2, pps.pps_tc_offset_div2), (-6, 6));
        assert!(pps.scaling_list_data.is_some());
        assert!(pps.lists_modification_present_flag);
        assert_eq!(pps.log2_parallel_merge_level_minus2, 2);
        assert!(pps.slice_segment_header_extension_present_flag);
        let range = &pps.pps_range_extension;
        assert_eq!(range.log2_max_transform_skip_block_size_minus2, 2);
        assert_eq!(range.diff_cu_chroma_qp_offset_depth, 1);
        assert_eq!(range.chroma_qp_offset_list_len_minus1, 1);
        assert_eq!(range.cb_qp_offset_list[..2], [3, -12]);
        assert_eq!(range.cr_qp_offset_list[..2], [-3, 12]);
        assert!(matches!(
            range.refuse_tools(),
            Err(BitstreamError::InvalidField {
                field: "log2_max_transform_skip_block_size_minus2",
                value: 2,
                ..
            })
        ));
    }

    #[test]
    fn tile_sizes_follow_6_3_and_6_4() {
        let sps = valid_sps(&SpsSpec::default());
        // 7 by 4 CTBs. Uniform spacing: 7 over 3 columns, 4 over 3 rows.
        let uniform = valid(&Spec {
            tiles: Some((2, 2, None)),
            ..Spec::default()
        });
        assert_eq!(uniform.check_with_sps(&sps), Ok(()));
        assert_eq!(uniform.column_widths(&sps), vec![2, 2, 3]);
        assert_eq!(uniform.row_heights(&sps), vec![1, 1, 2]);
        let explicit = valid(&Spec {
            tiles: Some((2, 1, Some((vec![2, 1], vec![0])))),
            ..Spec::default()
        });
        assert_eq!(explicit.check_with_sps(&sps), Ok(()));
        assert_eq!(explicit.column_widths(&sps), vec![3, 2, 2]);
        assert_eq!(explicit.row_heights(&sps), vec![1, 3]);
        let none = valid(&Spec::default());
        assert_eq!(none.column_widths(&sps), vec![7]);
        assert_eq!(none.row_heights(&sps), vec![4]);
    }

    #[test]
    fn out_of_range_values_are_refused() {
        type Change = fn(&mut Spec);
        let unsigned: [(&str, u64, Change); 9] = [
            ("pps_pic_parameter_set_id", 64, |s| s.pps_id = 64),
            ("pps_seq_parameter_set_id", 16, |s| s.sps_id = 16),
            ("num_ref_idx_l0_default_active_minus1", 15, |s| {
                s.l0_default = 15
            }),
            ("num_ref_idx_l1_default_active_minus1", 15, |s| {
                s.l1_default = 15
            }),
            ("num_tile_columns_minus1", 40, |s| {
                s.tiles = Some((40, 0, None))
            }),
            ("num_tile_rows_minus1", 44, |s| {
                s.tiles = Some((0, 44, None))
            }),
            ("num_tile_rows_minus1", 0, |s| s.tiles = Some((0, 0, None))),
            ("chroma_qp_offset_list_len_minus1", 6, |s| {
                s.range_extension = Some(RangeSpec {
                    chroma_list: Some((0, vec![(0, 0); 7])),
                    ..RangeSpec::default()
                });
            }),
            ("pps_scc_extension_flag", 1, |s| s.scc = true),
        ];
        for (name, expected, change) in unsigned {
            let mut spec = Spec::default();
            change(&mut spec);
            let result = PicParameterSet::parse(&build(&spec));
            assert!(
                matches!(
                    &result,
                    Err(BitstreamError::InvalidField { field, value, .. })
                        if *field == name && *value == expected
                ),
                "{name} {expected}: {result:?}"
            );
        }
        let signed: [(&str, i64, Change); 8] = [
            ("init_qp_minus26", 26, |s| s.init_qp = 26),
            ("init_qp_minus26", -75, |s| s.init_qp = -75),
            ("pps_cb_qp_offset", 13, |s| s.cb_offset = 13),
            ("pps_cr_qp_offset", -13, |s| s.cr_offset = -13),
            ("pps_beta_offset_div2", 7, |s| {
                s.deblock = Some((false, false, 7, 0))
            }),
            ("pps_tc_offset_div2", -7, |s| {
                s.deblock = Some((false, false, 0, -7))
            }),
            ("cb_qp_offset_list", 13, |s| {
                s.range_extension = Some(RangeSpec {
                    chroma_list: Some((0, vec![(13, 0)])),
                    ..RangeSpec::default()
                });
            }),
            ("cr_qp_offset_list", -13, |s| {
                s.range_extension = Some(RangeSpec {
                    chroma_list: Some((0, vec![(0, -13)])),
                    ..RangeSpec::default()
                });
            }),
        ];
        for (name, expected, change) in signed {
            let mut spec = Spec::default();
            change(&mut spec);
            let result = PicParameterSet::parse(&build(&spec));
            assert!(
                matches!(
                    &result,
                    Err(BitstreamError::InvalidSignedField { field, value, .. })
                        if *field == name && *value == expected
                ),
                "{name} {expected}: {result:?}"
            );
        }
    }

    #[test]
    fn sps_dependent_ranges_are_refused_at_activation() {
        type Change = fn(&mut Spec);
        let sps = valid_sps(&SpsSpec::default());
        let cases: [(&str, u64, Change); 11] = [
            ("pps_seq_parameter_set_id", 1, |s| s.sps_id = 1),
            ("diff_cu_qp_delta_depth", 4, |s| s.qp_delta_depth = Some(4)),
            ("log2_parallel_merge_level_minus2", 5, |s| s.merge_level = 5),
            ("pps_scaling_list_data_present_flag", 1, |s| {
                s.scaling = true
            }),
            ("num_tile_columns_minus1", 7, |s| {
                s.tiles = Some((7, 0, None))
            }),
            ("num_tile_rows_minus1", 4, |s| s.tiles = Some((0, 4, None))),
            ("column_width_minus1", 7, |s| {
                s.tiles = Some((1, 0, Some((vec![6], vec![]))));
            }),
            ("row_height_minus1", 4, |s| {
                s.tiles = Some((0, 1, Some((vec![], vec![3]))));
            }),
            ("log2_max_transform_skip_block_size_minus2", 4, |s| {
                s.transform_skip = true;
                s.range_extension = Some(RangeSpec {
                    transform_skip_size: 4,
                    ..RangeSpec::default()
                });
            }),
            ("cross_component_prediction_enabled_flag", 1, |s| {
                s.range_extension = Some(RangeSpec {
                    cross_component: true,
                    ..RangeSpec::default()
                });
            }),
            ("log2_sao_offset_scale_luma", 1, |s| {
                s.range_extension = Some(RangeSpec {
                    sao_luma: 1,
                    ..RangeSpec::default()
                });
            }),
        ];
        for (name, expected, change) in cases {
            let mut spec = Spec::default();
            change(&mut spec);
            let result = valid(&spec).check_with_sps(&sps);
            assert!(
                matches!(
                    &result,
                    Err(BitstreamError::InvalidField { field, value, .. })
                        if *field == name && *value == expected
                ),
                "{name} {expected}: {result:?}"
            );
        }
        // An init_qp valid only at a higher bit depth.
        let low = valid(&Spec {
            init_qp: -27,
            ..Spec::default()
        });
        assert!(matches!(
            low.check_with_sps(&sps),
            Err(BitstreamError::InvalidSignedField {
                field: "init_qp_minus26",
                value: -27,
                ..
            })
        ));
        let deep = valid_sps(&SpsSpec {
            bit_depth_luma_minus8: 2,
            ..SpsSpec::default()
        });
        assert_eq!(low.check_with_sps(&deep), Ok(()));
        // Chroma rules at monochrome.
        let mono = valid_sps(&SpsSpec {
            chroma_format_idc: 0,
            ..SpsSpec::default()
        });
        let offsets = valid(&Spec {
            slice_chroma_offsets: true,
            ..Spec::default()
        });
        assert!(matches!(
            offsets.check_with_sps(&mono),
            Err(BitstreamError::InvalidField {
                field: "pps_slice_chroma_qp_offsets_present_flag",
                ..
            })
        ));
        let chroma_list = valid(&Spec {
            range_extension: Some(RangeSpec {
                chroma_list: Some((4, vec![(0, 0)])),
                ..RangeSpec::default()
            }),
            ..Spec::default()
        });
        assert!(matches!(
            chroma_list.check_with_sps(&mono),
            Err(BitstreamError::InvalidField {
                field: "chroma_qp_offset_list_enabled_flag",
                ..
            })
        ));
        assert!(matches!(
            chroma_list.check_with_sps(&sps),
            Err(BitstreamError::InvalidField {
                field: "diff_cu_chroma_qp_offset_depth",
                value: 4,
                ..
            })
        ));
    }

    #[test]
    fn truncation_at_every_byte_never_panics() {
        let nal = build(&Spec {
            transform_skip: true,
            qp_delta_depth: Some(1),
            tiles: Some((2, 1, Some((vec![1, 0], vec![0])))),
            deblock: Some((true, false, -6, 6)),
            scaling: true,
            range_extension: Some(RangeSpec {
                chroma_list: Some((1, vec![(3, -3), (-12, 12)])),
                ..RangeSpec::default()
            }),
            ..Spec::default()
        });
        for end in 0..nal.len() {
            let result = PicParameterSet::parse(&nal[..end]);
            if end + 1 < nal.len() {
                assert!(result.is_err(), "cut at byte {end}");
            }
        }
        assert!(PicParameterSet::parse(&nal).is_ok());
    }
}
