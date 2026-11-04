//! HEVC `seq_parameter_set_rbsp()` (ITU-T H.265 7.3.2.2) for `nuh_layer_id` 0,
//! every field that decoding reads, and the variables 7.4.3.2 derives.
//!
//! [`read_through_vui`] walks the syntax with size guards only, so `hvcC` work
//! keeps accepting a stream with odd but readable values.
//! [`SeqParameterSet::parse`] adds the extensions and every range check.

use super::bits::BitReader;
use super::hevc_params;
use super::hevc_ptl::{self, ProfileTierLevel};
use super::hevc_rps::ShortTermRefPicSet;
use super::hevc_scaling::ScalingListData;
use super::hevc_sps::STRUCTURE;
use super::hevc_vui::{self, Vui};
use super::nal::HevcNalType;
use crate::error::BitstreamError;

/// The largest `sps_max_dec_pic_buffering_minus1`. `MaxDpbSize` is at most 16
/// (A.4.2).
const MAX_DEC_PIC_BUFFERING_MINUS1: u32 = 15;
/// A guard vidr sets on each picture side, so later allocations stay bounded.
/// Level 7.2 allows 33776 (A.4.1), and level 8.5 sets no limit.
const MAX_PICTURE_SIDE: u32 = 65_535;

/// `sps_range_extension()` (7.3.2.2.2). The parser reads the flags. A decoder
/// without these tools calls [`SpsRangeExtension::refuse_tools`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SpsRangeExtension {
    pub(crate) transform_skip_rotation_enabled_flag: bool,
    pub(crate) transform_skip_context_enabled_flag: bool,
    pub(crate) implicit_rdpcm_enabled_flag: bool,
    pub(crate) explicit_rdpcm_enabled_flag: bool,
    pub(crate) extended_precision_processing_flag: bool,
    pub(crate) intra_smoothing_disabled_flag: bool,
    pub(crate) high_precision_offsets_enabled_flag: bool,
    pub(crate) persistent_rice_adaptation_enabled_flag: bool,
    pub(crate) cabac_bypass_alignment_enabled_flag: bool,
}

impl SpsRangeExtension {
    fn parse(reader: &mut BitReader<'_>) -> Result<Self, BitstreamError> {
        Ok(Self {
            transform_skip_rotation_enabled_flag: reader
                .bit("transform_skip_rotation_enabled_flag")?,
            transform_skip_context_enabled_flag: reader
                .bit("transform_skip_context_enabled_flag")?,
            implicit_rdpcm_enabled_flag: reader.bit("implicit_rdpcm_enabled_flag")?,
            explicit_rdpcm_enabled_flag: reader.bit("explicit_rdpcm_enabled_flag")?,
            extended_precision_processing_flag: reader.bit("extended_precision_processing_flag")?,
            intra_smoothing_disabled_flag: reader.bit("intra_smoothing_disabled_flag")?,
            high_precision_offsets_enabled_flag: reader
                .bit("high_precision_offsets_enabled_flag")?,
            persistent_rice_adaptation_enabled_flag: reader
                .bit("persistent_rice_adaptation_enabled_flag")?,
            cabac_bypass_alignment_enabled_flag: reader
                .bit("cabac_bypass_alignment_enabled_flag")?,
        })
    }

    /// Refuses a set flag, for a decoder without range extension tools.
    pub(crate) fn refuse_tools(&self) -> Result<(), BitstreamError> {
        let flags = [
            (
                "transform_skip_rotation_enabled_flag",
                self.transform_skip_rotation_enabled_flag,
            ),
            (
                "transform_skip_context_enabled_flag",
                self.transform_skip_context_enabled_flag,
            ),
            (
                "implicit_rdpcm_enabled_flag",
                self.implicit_rdpcm_enabled_flag,
            ),
            (
                "explicit_rdpcm_enabled_flag",
                self.explicit_rdpcm_enabled_flag,
            ),
            (
                "extended_precision_processing_flag",
                self.extended_precision_processing_flag,
            ),
            (
                "intra_smoothing_disabled_flag",
                self.intra_smoothing_disabled_flag,
            ),
            (
                "high_precision_offsets_enabled_flag",
                self.high_precision_offsets_enabled_flag,
            ),
            (
                "persistent_rice_adaptation_enabled_flag",
                self.persistent_rice_adaptation_enabled_flag,
            ),
            (
                "cabac_bypass_alignment_enabled_flag",
                self.cabac_bypass_alignment_enabled_flag,
            ),
        ];
        match flags.into_iter().find(|(_, set)| *set) {
            Some((field, _)) => Err(invalid(
                field,
                1,
                "0, because this decoder has no range extension tools",
            )),
            None => Ok(()),
        }
    }
}

/// One `seq_parameter_set_rbsp()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SeqParameterSet {
    pub(crate) sps_video_parameter_set_id: u8,
    pub(crate) sps_max_sub_layers_minus1: u8,
    pub(crate) sps_temporal_id_nesting_flag: bool,
    pub(crate) profile_tier_level: ProfileTierLevel,
    pub(crate) sps_seq_parameter_set_id: u32,
    pub(crate) chroma_format_idc: u32,
    pub(crate) separate_colour_plane_flag: bool,
    pub(crate) pic_width_in_luma_samples: u32,
    pub(crate) pic_height_in_luma_samples: u32,
    pub(crate) conformance_window_flag: bool,
    pub(crate) conf_win_left_offset: u32,
    pub(crate) conf_win_right_offset: u32,
    pub(crate) conf_win_top_offset: u32,
    pub(crate) conf_win_bottom_offset: u32,
    pub(crate) bit_depth_luma_minus8: u32,
    pub(crate) bit_depth_chroma_minus8: u32,
    pub(crate) log2_max_pic_order_cnt_lsb_minus4: u32,
    pub(crate) sps_sub_layer_ordering_info_present_flag: bool,
    /// Indexed by sub-layer, with the 7.4.3.2.1 inference for absent entries.
    pub(crate) sps_max_dec_pic_buffering_minus1: [u32; 7],
    /// Indexed by sub-layer, with the 7.4.3.2.1 inference for absent entries.
    pub(crate) sps_max_num_reorder_pics: [u32; 7],
    /// Indexed by sub-layer, with the 7.4.3.2.1 inference for absent entries.
    pub(crate) sps_max_latency_increase_plus1: [u32; 7],
    pub(crate) log2_min_luma_coding_block_size_minus3: u32,
    pub(crate) log2_diff_max_min_luma_coding_block_size: u32,
    pub(crate) log2_min_luma_transform_block_size_minus2: u32,
    pub(crate) log2_diff_max_min_luma_transform_block_size: u32,
    pub(crate) max_transform_hierarchy_depth_inter: u32,
    pub(crate) max_transform_hierarchy_depth_intra: u32,
    pub(crate) scaling_list_enabled_flag: bool,
    pub(crate) sps_scaling_list_data_present_flag: bool,
    /// `None` without `sps_scaling_list_data_present_flag`.
    pub(crate) scaling_list_data: Option<Box<ScalingListData>>,
    pub(crate) amp_enabled_flag: bool,
    pub(crate) sample_adaptive_offset_enabled_flag: bool,
    pub(crate) pcm_enabled_flag: bool,
    pub(crate) pcm_sample_bit_depth_luma_minus1: u8,
    pub(crate) pcm_sample_bit_depth_chroma_minus1: u8,
    pub(crate) log2_min_pcm_luma_coding_block_size_minus3: u32,
    pub(crate) log2_diff_max_min_pcm_luma_coding_block_size: u32,
    pub(crate) pcm_loop_filter_disabled_flag: bool,
    /// One entry per `st_ref_pic_set( i )`: `num_short_term_ref_pic_sets`.
    pub(crate) st_ref_pic_set: Vec<ShortTermRefPicSet>,
    pub(crate) long_term_ref_pics_present_flag: bool,
    pub(crate) num_long_term_ref_pics_sps: u32,
    pub(crate) lt_ref_pic_poc_lsb_sps: [u16; 32],
    pub(crate) used_by_curr_pic_lt_sps_flag: [bool; 32],
    pub(crate) sps_temporal_mvp_enabled_flag: bool,
    pub(crate) strong_intra_smoothing_enabled_flag: bool,
    pub(crate) vui_parameters_present_flag: bool,
    /// `None` without `vui_parameters_present_flag`.
    pub(crate) vui: Option<Vui>,
    pub(crate) sps_extension_present_flag: bool,
    pub(crate) sps_range_extension_flag: bool,
    pub(crate) sps_multilayer_extension_flag: bool,
    pub(crate) sps_3d_extension_flag: bool,
    pub(crate) sps_extension_4bits: u8,
    pub(crate) sps_range_extension: SpsRangeExtension,
    pub(crate) inter_view_mv_vert_constraint_flag: bool,
}

fn invalid(field: &'static str, value: u64, allowed: &'static str) -> BitstreamError {
    BitstreamError::InvalidField {
        structure: STRUCTURE,
        field,
        value,
        allowed,
    }
}

/// Reads an SPS RBSP from its first field to the end of `vui_parameters()` as
/// far as `min_spatial_segmentation_idc`. Only limits that keep the walk sound
/// apply here.
pub(crate) fn read_through_vui(
    reader: &mut BitReader<'_>,
) -> Result<SeqParameterSet, BitstreamError> {
    let sps_video_parameter_set_id = reader.u8(4, "sps_video_parameter_set_id")?;
    let sps_max_sub_layers_minus1 = reader.u8(3, "sps_max_sub_layers_minus1")?;
    if sps_max_sub_layers_minus1 > 6 {
        return Err(invalid(
            "sps_max_sub_layers_minus1",
            sps_max_sub_layers_minus1.into(),
            "0 to 6",
        ));
    }
    let sps_temporal_id_nesting_flag = reader.bit("sps_temporal_id_nesting_flag")?;
    let profile_tier_level = hevc_ptl::parse_full(reader, sps_max_sub_layers_minus1)?;
    let sps_seq_parameter_set_id = reader.ue("sps_seq_parameter_set_id")?;
    let chroma_format_idc = reader.ue_max("chroma_format_idc", 3, "0 to 3")?;
    let separate_colour_plane_flag =
        chroma_format_idc == 3 && reader.bit("separate_colour_plane_flag")?;
    let pic_width_in_luma_samples = reader.ue("pic_width_in_luma_samples")?;
    let pic_height_in_luma_samples = reader.ue("pic_height_in_luma_samples")?;
    let conformance_window_flag = reader.bit("conformance_window_flag")?;
    let mut conf_win = [0u32; 4];
    if conformance_window_flag {
        conf_win = [
            reader.ue("conf_win_left_offset")?,
            reader.ue("conf_win_right_offset")?,
            reader.ue("conf_win_top_offset")?,
            reader.ue("conf_win_bottom_offset")?,
        ];
    }
    let bit_depth_luma_minus8 = reader.ue("bit_depth_luma_minus8")?;
    let bit_depth_chroma_minus8 = reader.ue("bit_depth_chroma_minus8")?;
    let log2_max_pic_order_cnt_lsb_minus4 =
        reader.ue_max("log2_max_pic_order_cnt_lsb_minus4", 12, "0 to 12")?;

    let top = usize::from(sps_max_sub_layers_minus1);
    let sps_sub_layer_ordering_info_present_flag =
        reader.bit("sps_sub_layer_ordering_info_present_flag")?;
    let first = if sps_sub_layer_ordering_info_present_flag {
        0
    } else {
        top
    };
    let mut sps_max_dec_pic_buffering_minus1 = [0u32; 7];
    let mut sps_max_num_reorder_pics = [0u32; 7];
    let mut sps_max_latency_increase_plus1 = [0u32; 7];
    for layer in first..=top {
        sps_max_dec_pic_buffering_minus1[layer] = reader.ue("sps_max_dec_pic_buffering_minus1")?;
        sps_max_num_reorder_pics[layer] = reader.ue("sps_max_num_reorder_pics")?;
        sps_max_latency_increase_plus1[layer] = reader.ue("sps_max_latency_increase_plus1")?;
    }
    for layer in 0..first {
        sps_max_dec_pic_buffering_minus1[layer] = sps_max_dec_pic_buffering_minus1[top];
        sps_max_num_reorder_pics[layer] = sps_max_num_reorder_pics[top];
        sps_max_latency_increase_plus1[layer] = sps_max_latency_increase_plus1[top];
    }

    let log2_min_luma_coding_block_size_minus3 =
        reader.ue("log2_min_luma_coding_block_size_minus3")?;
    let log2_diff_max_min_luma_coding_block_size =
        reader.ue("log2_diff_max_min_luma_coding_block_size")?;
    let log2_min_luma_transform_block_size_minus2 =
        reader.ue("log2_min_luma_transform_block_size_minus2")?;
    let log2_diff_max_min_luma_transform_block_size =
        reader.ue("log2_diff_max_min_luma_transform_block_size")?;
    let max_transform_hierarchy_depth_inter = reader.ue("max_transform_hierarchy_depth_inter")?;
    let max_transform_hierarchy_depth_intra = reader.ue("max_transform_hierarchy_depth_intra")?;

    let scaling_list_enabled_flag = reader.bit("scaling_list_enabled_flag")?;
    let sps_scaling_list_data_present_flag =
        scaling_list_enabled_flag && reader.bit("sps_scaling_list_data_present_flag")?;
    let scaling_list_data = if sps_scaling_list_data_present_flag {
        Some(Box::new(ScalingListData::parse(reader)?))
    } else {
        None
    };
    let amp_enabled_flag = reader.bit("amp_enabled_flag")?;
    let sample_adaptive_offset_enabled_flag = reader.bit("sample_adaptive_offset_enabled_flag")?;

    let pcm_enabled_flag = reader.bit("pcm_enabled_flag")?;
    let mut pcm_sample_bit_depth_luma_minus1 = 0;
    let mut pcm_sample_bit_depth_chroma_minus1 = 0;
    let mut log2_min_pcm_luma_coding_block_size_minus3 = 0;
    let mut log2_diff_max_min_pcm_luma_coding_block_size = 0;
    let mut pcm_loop_filter_disabled_flag = false;
    if pcm_enabled_flag {
        pcm_sample_bit_depth_luma_minus1 = reader.u8(4, "pcm_sample_bit_depth_luma_minus1")?;
        pcm_sample_bit_depth_chroma_minus1 = reader.u8(4, "pcm_sample_bit_depth_chroma_minus1")?;
        log2_min_pcm_luma_coding_block_size_minus3 =
            reader.ue("log2_min_pcm_luma_coding_block_size_minus3")?;
        log2_diff_max_min_pcm_luma_coding_block_size =
            reader.ue("log2_diff_max_min_pcm_luma_coding_block_size")?;
        pcm_loop_filter_disabled_flag = reader.bit("pcm_loop_filter_disabled_flag")?;
    }

    let num_short_term_ref_pic_sets =
        reader.ue_max("num_short_term_ref_pic_sets", 64, "0 to 64")?;
    let mut st_ref_pic_set = Vec::with_capacity(num_short_term_ref_pic_sets as usize);
    for _ in 0..num_short_term_ref_pic_sets {
        let set = ShortTermRefPicSet::parse(reader, &st_ref_pic_set, false)?;
        st_ref_pic_set.push(set);
    }

    let long_term_ref_pics_present_flag = reader.bit("long_term_ref_pics_present_flag")?;
    let mut num_long_term_ref_pics_sps = 0;
    let mut lt_ref_pic_poc_lsb_sps = [0u16; 32];
    let mut used_by_curr_pic_lt_sps_flag = [false; 32];
    if long_term_ref_pics_present_flag {
        num_long_term_ref_pics_sps = reader.ue_max("num_long_term_ref_pics_sps", 32, "0 to 32")?;
        let lsb_bits = log2_max_pic_order_cnt_lsb_minus4 + 4;
        for (lsb, used) in lt_ref_pic_poc_lsb_sps
            .iter_mut()
            .zip(&mut used_by_curr_pic_lt_sps_flag)
            .take(num_long_term_ref_pics_sps as usize)
        {
            *lsb = reader.bits(lsb_bits, "lt_ref_pic_poc_lsb_sps")? as u16;
            *used = reader.bit("used_by_curr_pic_lt_sps_flag")?;
        }
    }
    let sps_temporal_mvp_enabled_flag = reader.bit("sps_temporal_mvp_enabled_flag")?;
    let strong_intra_smoothing_enabled_flag = reader.bit("strong_intra_smoothing_enabled_flag")?;
    let vui_parameters_present_flag = reader.bit("vui_parameters_present_flag")?;
    let vui = if vui_parameters_present_flag {
        Some(hevc_vui::parse(reader, sps_max_sub_layers_minus1)?)
    } else {
        None
    };
    let [
        conf_win_left_offset,
        conf_win_right_offset,
        conf_win_top_offset,
        conf_win_bottom_offset,
    ] = conf_win;
    Ok(SeqParameterSet {
        sps_video_parameter_set_id,
        sps_max_sub_layers_minus1,
        sps_temporal_id_nesting_flag,
        profile_tier_level,
        sps_seq_parameter_set_id,
        chroma_format_idc,
        separate_colour_plane_flag,
        pic_width_in_luma_samples,
        pic_height_in_luma_samples,
        conformance_window_flag,
        conf_win_left_offset,
        conf_win_right_offset,
        conf_win_top_offset,
        conf_win_bottom_offset,
        bit_depth_luma_minus8,
        bit_depth_chroma_minus8,
        log2_max_pic_order_cnt_lsb_minus4,
        sps_sub_layer_ordering_info_present_flag,
        sps_max_dec_pic_buffering_minus1,
        sps_max_num_reorder_pics,
        sps_max_latency_increase_plus1,
        log2_min_luma_coding_block_size_minus3,
        log2_diff_max_min_luma_coding_block_size,
        log2_min_luma_transform_block_size_minus2,
        log2_diff_max_min_luma_transform_block_size,
        max_transform_hierarchy_depth_inter,
        max_transform_hierarchy_depth_intra,
        scaling_list_enabled_flag,
        sps_scaling_list_data_present_flag,
        scaling_list_data,
        amp_enabled_flag,
        sample_adaptive_offset_enabled_flag,
        pcm_enabled_flag,
        pcm_sample_bit_depth_luma_minus1,
        pcm_sample_bit_depth_chroma_minus1,
        log2_min_pcm_luma_coding_block_size_minus3,
        log2_diff_max_min_pcm_luma_coding_block_size,
        pcm_loop_filter_disabled_flag,
        st_ref_pic_set,
        long_term_ref_pics_present_flag,
        num_long_term_ref_pics_sps,
        lt_ref_pic_poc_lsb_sps,
        used_by_curr_pic_lt_sps_flag,
        sps_temporal_mvp_enabled_flag,
        strong_intra_smoothing_enabled_flag,
        vui_parameters_present_flag,
        vui,
        sps_extension_present_flag: false,
        sps_range_extension_flag: false,
        sps_multilayer_extension_flag: false,
        sps_3d_extension_flag: false,
        sps_extension_4bits: 0,
        sps_range_extension: SpsRangeExtension::default(),
        inter_view_mv_vert_constraint_flag: false,
    })
}

impl SeqParameterSet {
    /// Reads an SPS NAL unit, 2-byte header included, and checks every range
    /// the semantics of 7.4.3.2 state.
    pub(crate) fn parse(nal: &[u8]) -> Result<Self, BitstreamError> {
        let rbsp = hevc_params::payload(nal, HevcNalType::SPS, STRUCTURE)?;
        let mut reader = BitReader::new(&rbsp, STRUCTURE);
        let mut sps = read_through_vui(&mut reader)?;
        if let Some(vui) = &sps.vui {
            hevc_vui::read_restriction_tail(&mut reader, vui)?;
        }
        sps.read_extensions(&mut reader)?;
        sps.check_ranges()?;
        Ok(sps)
    }

    fn read_extensions(&mut self, reader: &mut BitReader<'_>) -> Result<(), BitstreamError> {
        self.sps_extension_present_flag = reader.bit("sps_extension_present_flag")?;
        if !self.sps_extension_present_flag {
            return Ok(());
        }
        self.sps_range_extension_flag = reader.bit("sps_range_extension_flag")?;
        self.sps_multilayer_extension_flag = reader.bit("sps_multilayer_extension_flag")?;
        self.sps_3d_extension_flag = reader.bit("sps_3d_extension_flag")?;
        if reader.bit("sps_scc_extension_flag")? {
            // Screen content coding changes the slice header syntax.
            return Err(invalid(
                "sps_scc_extension_flag",
                1,
                "0, because vidr does not read screen content coding streams",
            ));
        }
        self.sps_extension_4bits = reader.u8(4, "sps_extension_4bits")?;
        if self.sps_range_extension_flag {
            self.sps_range_extension = SpsRangeExtension::parse(reader)?;
        }
        if self.sps_multilayer_extension_flag {
            self.inter_view_mv_vert_constraint_flag =
                reader.bit("inter_view_mv_vert_constraint_flag")?;
        }
        // The 3D extension and extension data do not affect layer 0.
        Ok(())
    }

    fn check_ranges(&self) -> Result<(), BitstreamError> {
        if self.sps_max_sub_layers_minus1 == 0 && !self.sps_temporal_id_nesting_flag {
            return Err(invalid(
                "sps_temporal_id_nesting_flag",
                0,
                "1 when sps_max_sub_layers_minus1 is 0",
            ));
        }
        if self.sps_seq_parameter_set_id > 15 {
            return Err(invalid(
                "sps_seq_parameter_set_id",
                self.sps_seq_parameter_set_id.into(),
                "0 to 15",
            ));
        }
        if self.bit_depth_luma_minus8 > 8 {
            return Err(invalid(
                "bit_depth_luma_minus8",
                self.bit_depth_luma_minus8.into(),
                "0 to 8",
            ));
        }
        if self.bit_depth_chroma_minus8 > 8 {
            return Err(invalid(
                "bit_depth_chroma_minus8",
                self.bit_depth_chroma_minus8.into(),
                "0 to 8",
            ));
        }
        self.check_ordering()?;
        self.check_block_sizes()?;
        self.check_picture_size()?;
        self.check_pcm()?;
        let top = usize::from(self.sps_max_sub_layers_minus1);
        for set in &self.st_ref_pic_set {
            set.check_dpb_limit(STRUCTURE, self.sps_max_dec_pic_buffering_minus1[top])?;
        }
        Ok(())
    }

    fn check_ordering(&self) -> Result<(), BitstreamError> {
        let top = usize::from(self.sps_max_sub_layers_minus1);
        let mut previous = (0, 0);
        for layer in 0..=top {
            let buffering = self.sps_max_dec_pic_buffering_minus1[layer];
            let reorder = self.sps_max_num_reorder_pics[layer];
            if buffering > MAX_DEC_PIC_BUFFERING_MINUS1 || buffering < previous.0 {
                return Err(invalid(
                    "sps_max_dec_pic_buffering_minus1",
                    buffering.into(),
                    "0 to 15, and no lower than the sub-layer before it",
                ));
            }
            if reorder > buffering || reorder < previous.1 {
                return Err(invalid(
                    "sps_max_num_reorder_pics",
                    reorder.into(),
                    "0 to sps_max_dec_pic_buffering_minus1, and no lower than the sub-layer \
                     before it",
                ));
            }
            previous = (buffering, reorder);
        }
        Ok(())
    }

    /// Checks the block size fields. Every later derivation relies on these
    /// limits to stay free of overflow.
    fn check_block_sizes(&self) -> Result<(), BitstreamError> {
        // CtbLog2SizeY is 4 to 6 in every video profile (A.3).
        if self.log2_min_luma_coding_block_size_minus3 > 3 {
            return Err(invalid(
                "log2_min_luma_coding_block_size_minus3",
                self.log2_min_luma_coding_block_size_minus3.into(),
                "0 to 3",
            ));
        }
        let ctb_log2 = u64::from(self.log2_min_luma_coding_block_size_minus3)
            + 3
            + u64::from(self.log2_diff_max_min_luma_coding_block_size);
        if !(4..=6).contains(&ctb_log2) {
            return Err(invalid(
                "log2_diff_max_min_luma_coding_block_size",
                self.log2_diff_max_min_luma_coding_block_size.into(),
                "a value that gives CtbLog2SizeY 4 to 6",
            ));
        }
        let min_tb_log2 = u64::from(self.log2_min_luma_transform_block_size_minus2) + 2;
        if min_tb_log2 >= u64::from(self.log2_min_luma_coding_block_size_minus3) + 3 {
            return Err(invalid(
                "log2_min_luma_transform_block_size_minus2",
                self.log2_min_luma_transform_block_size_minus2.into(),
                "a value that gives MinTbLog2SizeY below MinCbLog2SizeY",
            ));
        }
        let max_tb_log2 = min_tb_log2 + u64::from(self.log2_diff_max_min_luma_transform_block_size);
        if max_tb_log2 > ctb_log2.min(5) {
            return Err(invalid(
                "log2_diff_max_min_luma_transform_block_size",
                self.log2_diff_max_min_luma_transform_block_size.into(),
                "a value that gives MaxTbLog2SizeY at most Min( CtbLog2SizeY, 5 )",
            ));
        }
        let max_depth = ctb_log2 - min_tb_log2;
        if u64::from(self.max_transform_hierarchy_depth_inter) > max_depth {
            return Err(invalid(
                "max_transform_hierarchy_depth_inter",
                self.max_transform_hierarchy_depth_inter.into(),
                "0 to CtbLog2SizeY - MinTbLog2SizeY",
            ));
        }
        if u64::from(self.max_transform_hierarchy_depth_intra) > max_depth {
            return Err(invalid(
                "max_transform_hierarchy_depth_intra",
                self.max_transform_hierarchy_depth_intra.into(),
                "0 to CtbLog2SizeY - MinTbLog2SizeY",
            ));
        }
        Ok(())
    }

    /// Runs after [`Self::check_block_sizes`], which bounds `MinCbSizeY`.
    fn check_picture_size(&self) -> Result<(), BitstreamError> {
        let min_cb = 1u32 << self.min_cb_log2_size_y();
        let width = self.pic_width_in_luma_samples;
        if width == 0 || width > MAX_PICTURE_SIDE || !width.is_multiple_of(min_cb) {
            return Err(invalid(
                "pic_width_in_luma_samples",
                width.into(),
                "a multiple of MinCbSizeY from 1 to 65535",
            ));
        }
        let height = self.pic_height_in_luma_samples;
        if height == 0 || height > MAX_PICTURE_SIDE || !height.is_multiple_of(min_cb) {
            return Err(invalid(
                "pic_height_in_luma_samples",
                height.into(),
                "a multiple of MinCbSizeY from 1 to 65535",
            ));
        }
        let horizontal =
            u64::from(self.conf_win_left_offset) + u64::from(self.conf_win_right_offset);
        if u64::from(self.sub_width_c()) * horizontal >= u64::from(width) {
            return Err(invalid(
                "conf_win_right_offset",
                self.conf_win_right_offset.into(),
                "a value that leaves SubWidthC * ( left + right ) below the picture width",
            ));
        }
        let vertical = u64::from(self.conf_win_top_offset) + u64::from(self.conf_win_bottom_offset);
        if u64::from(self.sub_height_c()) * vertical >= u64::from(height) {
            return Err(invalid(
                "conf_win_bottom_offset",
                self.conf_win_bottom_offset.into(),
                "a value that leaves SubHeightC * ( top + bottom ) below the picture height",
            ));
        }
        Ok(())
    }

    /// Runs after [`Self::check_block_sizes`].
    fn check_pcm(&self) -> Result<(), BitstreamError> {
        if !self.pcm_enabled_flag {
            return Ok(());
        }
        if u32::from(self.pcm_sample_bit_depth_luma_minus1) + 1 > self.bit_depth_y() {
            return Err(invalid(
                "pcm_sample_bit_depth_luma_minus1",
                self.pcm_sample_bit_depth_luma_minus1.into(),
                "a value that gives PcmBitDepthY at most BitDepthY",
            ));
        }
        // A decoder ignores this value with `ChromaArrayType` 0 (7.4.3.2.1).
        if self.chroma_array_type() != 0
            && u32::from(self.pcm_sample_bit_depth_chroma_minus1) + 1 > self.bit_depth_c()
        {
            return Err(invalid(
                "pcm_sample_bit_depth_chroma_minus1",
                self.pcm_sample_bit_depth_chroma_minus1.into(),
                "a value that gives PcmBitDepthC at most BitDepthC",
            ));
        }
        let log2_min = u64::from(self.log2_min_pcm_luma_coding_block_size_minus3) + 3;
        let ctb_limit = u64::from(self.ctb_log2_size_y().min(5));
        let lowest = u64::from(self.min_cb_log2_size_y().min(5));
        if log2_min < lowest || log2_min > ctb_limit {
            return Err(invalid(
                "log2_min_pcm_luma_coding_block_size_minus3",
                self.log2_min_pcm_luma_coding_block_size_minus3.into(),
                "a value that gives Log2MinIpcmCbSizeY from Min( MinCbLog2SizeY, 5 ) to \
                 Min( CtbLog2SizeY, 5 )",
            ));
        }
        if log2_min + u64::from(self.log2_diff_max_min_pcm_luma_coding_block_size) > ctb_limit {
            return Err(invalid(
                "log2_diff_max_min_pcm_luma_coding_block_size",
                self.log2_diff_max_min_pcm_luma_coding_block_size.into(),
                "a value that gives Log2MaxIpcmCbSizeY at most Min( CtbLog2SizeY, 5 )",
            ));
        }
        Ok(())
    }

    /// `ChromaArrayType`.
    pub(crate) const fn chroma_array_type(&self) -> u32 {
        if self.separate_colour_plane_flag {
            0
        } else {
            self.chroma_format_idc
        }
    }

    /// `SubWidthC` (Table 6-1).
    pub(crate) const fn sub_width_c(&self) -> u32 {
        if self.chroma_format_idc == 1 || self.chroma_format_idc == 2 {
            2
        } else {
            1
        }
    }

    /// `SubHeightC` (Table 6-1).
    pub(crate) const fn sub_height_c(&self) -> u32 {
        if self.chroma_format_idc == 1 { 2 } else { 1 }
    }

    /// `BitDepthY`.
    pub(crate) const fn bit_depth_y(&self) -> u32 {
        8 + self.bit_depth_luma_minus8
    }

    /// `BitDepthC`.
    pub(crate) const fn bit_depth_c(&self) -> u32 {
        8 + self.bit_depth_chroma_minus8
    }

    /// `QpBdOffsetY`.
    pub(crate) const fn qp_bd_offset_y(&self) -> u32 {
        6 * self.bit_depth_luma_minus8
    }

    /// `MinCbLog2SizeY`.
    pub(crate) const fn min_cb_log2_size_y(&self) -> u32 {
        self.log2_min_luma_coding_block_size_minus3 + 3
    }

    /// `CtbLog2SizeY`.
    pub(crate) const fn ctb_log2_size_y(&self) -> u32 {
        self.min_cb_log2_size_y() + self.log2_diff_max_min_luma_coding_block_size
    }

    /// `MaxTbLog2SizeY`.
    pub(crate) const fn max_tb_log2_size_y(&self) -> u32 {
        self.log2_min_luma_transform_block_size_minus2
            + 2
            + self.log2_diff_max_min_luma_transform_block_size
    }

    /// `PicWidthInCtbsY`.
    pub(crate) const fn pic_width_in_ctbs_y(&self) -> u32 {
        self.pic_width_in_luma_samples
            .div_ceil(1 << self.ctb_log2_size_y())
    }

    /// `PicHeightInCtbsY`.
    pub(crate) const fn pic_height_in_ctbs_y(&self) -> u32 {
        self.pic_height_in_luma_samples
            .div_ceil(1 << self.ctb_log2_size_y())
    }

    /// `PicSizeInCtbsY`.
    pub(crate) const fn pic_size_in_ctbs_y(&self) -> u32 {
        self.pic_width_in_ctbs_y() * self.pic_height_in_ctbs_y()
    }

    /// `WpOffsetHalfRangeY`.
    pub(crate) const fn wp_offset_half_range_y(&self) -> i64 {
        if self.sps_range_extension.high_precision_offsets_enabled_flag {
            1 << (self.bit_depth_y() - 1)
        } else {
            1 << 7
        }
    }

    /// `WpOffsetHalfRangeC`.
    pub(crate) const fn wp_offset_half_range_c(&self) -> i64 {
        if self.sps_range_extension.high_precision_offsets_enabled_flag {
            1 << (self.bit_depth_c() - 1)
        } else {
            1 << 7
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::hevc_sps::tests::X265_SPS;
    use super::super::test_bits::{BitWriter, hevc_nal};
    use super::*;

    /// The values a test varies. The defaults make a valid 416x240 Main SPS
    /// with a 64x64 CTB.
    #[derive(Clone)]
    pub(crate) struct Spec {
        pub(crate) sub_layers_minus1: u64,
        pub(crate) nesting: bool,
        pub(crate) sps_id: u32,
        pub(crate) chroma_format_idc: u32,
        pub(crate) separate_colour_plane: bool,
        pub(crate) width: u32,
        pub(crate) height: u32,
        pub(crate) conf_win: Option<[u32; 4]>,
        pub(crate) bit_depth_luma_minus8: u32,
        pub(crate) bit_depth_chroma_minus8: u32,
        pub(crate) poc_lsb_minus4: u32,
        pub(crate) dec_pic_buffering: [u32; 2],
        pub(crate) num_reorder: [u32; 2],
        pub(crate) min_cb_minus3: u32,
        pub(crate) diff_cb: u32,
        pub(crate) min_tb_minus2: u32,
        pub(crate) diff_tb: u32,
        pub(crate) depth_inter: u32,
        pub(crate) depth_intra: u32,
        pub(crate) scaling: bool,
        pub(crate) sao: bool,
        pub(crate) pcm: Option<(u64, u64, u32, u32)>,
        /// `(num_negative, num_positive)` of each explicit set.
        pub(crate) sets: Vec<(u32, u32)>,
        pub(crate) long_term: Vec<(u64, bool)>,
        pub(crate) temporal_mvp: bool,
        pub(crate) range_extension: Option<[bool; 9]>,
        pub(crate) multilayer: bool,
        pub(crate) scc: bool,
    }

    impl Default for Spec {
        fn default() -> Self {
            Self {
                sub_layers_minus1: 0,
                nesting: true,
                sps_id: 0,
                chroma_format_idc: 1,
                separate_colour_plane: false,
                width: 416,
                height: 240,
                conf_win: None,
                bit_depth_luma_minus8: 0,
                bit_depth_chroma_minus8: 0,
                poc_lsb_minus4: 4,
                dec_pic_buffering: [5, 5],
                num_reorder: [2, 2],
                min_cb_minus3: 0,
                diff_cb: 3,
                min_tb_minus2: 0,
                diff_tb: 3,
                depth_inter: 2,
                depth_intra: 1,
                scaling: false,
                sao: true,
                pcm: None,
                sets: vec![(2, 1)],
                long_term: Vec::new(),
                temporal_mvp: true,
                range_extension: None,
                multilayer: false,
                scc: false,
            }
        }
    }

    pub(crate) fn build(spec: &Spec) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.u(4, 0).u(3, spec.sub_layers_minus1).flag(spec.nesting);
        w.u(8, 1)
            .u(32, 0x6000_0000)
            .u(48, 0x9000_0000_0000)
            .u(8, 120);
        let layers = spec.sub_layers_minus1.min(6) as usize;
        for _ in 0..layers {
            w.flag(false).flag(false);
        }
        if layers > 0 {
            w.u(2 * (8 - layers as u32), 0);
        }
        w.ue(spec.sps_id).ue(spec.chroma_format_idc);
        if spec.chroma_format_idc == 3 {
            w.flag(spec.separate_colour_plane);
        }
        w.ue(spec.width).ue(spec.height);
        w.flag(spec.conf_win.is_some());
        if let Some(window) = spec.conf_win {
            for offset in window {
                w.ue(offset);
            }
        }
        w.ue(spec.bit_depth_luma_minus8)
            .ue(spec.bit_depth_chroma_minus8)
            .ue(spec.poc_lsb_minus4);
        w.flag(true);
        for layer in 0..=layers {
            let at = layer.min(1);
            w.ue(spec.dec_pic_buffering[at])
                .ue(spec.num_reorder[at])
                .ue(0);
        }
        w.ue(spec.min_cb_minus3)
            .ue(spec.diff_cb)
            .ue(spec.min_tb_minus2)
            .ue(spec.diff_tb)
            .ue(spec.depth_inter)
            .ue(spec.depth_intra);
        w.flag(spec.scaling);
        if spec.scaling {
            w.flag(false);
        }
        w.flag(true).flag(spec.sao);
        w.flag(spec.pcm.is_some());
        if let Some((luma, chroma, min, diff)) = spec.pcm {
            w.u(4, luma).u(4, chroma).ue(min).ue(diff).flag(true);
        }
        w.ue(spec.sets.len() as u32);
        for (index, (negative, positive)) in spec.sets.iter().enumerate() {
            if index > 0 {
                w.flag(false);
            }
            w.ue(*negative).ue(*positive);
            for _ in 0..negative + positive {
                w.ue(0).flag(true);
            }
        }
        w.flag(!spec.long_term.is_empty());
        if !spec.long_term.is_empty() {
            w.ue(spec.long_term.len() as u32);
            for (lsb, used) in &spec.long_term {
                w.u(spec.poc_lsb_minus4 + 4, *lsb).flag(*used);
            }
        }
        w.flag(spec.temporal_mvp).flag(true);
        // No VUI.
        w.flag(false);
        let extension = spec.range_extension.is_some() || spec.multilayer || spec.scc;
        w.flag(extension);
        if extension {
            w.flag(spec.range_extension.is_some())
                .flag(spec.multilayer)
                .flag(false)
                .flag(spec.scc)
                .u(4, 0);
            if let Some(flags) = spec.range_extension {
                for flag in flags {
                    w.flag(flag);
                }
            }
            if spec.multilayer {
                w.flag(true);
            }
        }
        w.trailing();
        hevc_nal(33, &w.bytes())
    }

    pub(crate) fn valid(spec: &Spec) -> SeqParameterSet {
        SeqParameterSet::parse(&build(spec)).unwrap_or_else(|error| panic!("{error}"))
    }

    #[test]
    fn reads_a_real_sps() {
        let sps = SeqParameterSet::parse(&X265_SPS).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(sps.pic_width_in_luma_samples, 128);
        assert_eq!(sps.pic_height_in_luma_samples, 96);
        assert_eq!(sps.chroma_array_type(), 1);
        assert_eq!(sps.ctb_log2_size_y(), 6);
        assert_eq!(sps.pic_width_in_ctbs_y(), 2);
        assert_eq!(sps.pic_height_in_ctbs_y(), 2);
        assert_eq!(sps.pic_size_in_ctbs_y(), 4);
        assert!(sps.sample_adaptive_offset_enabled_flag);
        assert!(!sps.sps_extension_present_flag);
        assert!(sps.vui.is_some());
    }

    #[test]
    fn reads_the_optional_structures() {
        let sps = valid(&Spec {
            conf_win: Some([1, 2, 0, 3]),
            pcm: Some((6, 5, 0, 2)),
            scaling: true,
            sets: vec![(2, 1), (0, 0)],
            long_term: vec![(17, true), (200, false)],
            range_extension: Some([true; 9]),
            multilayer: true,
            ..Spec::default()
        });
        assert_eq!(sps.conf_win_left_offset, 1);
        assert_eq!(sps.conf_win_right_offset, 2);
        assert_eq!(sps.conf_win_bottom_offset, 3);
        assert!(sps.scaling_list_enabled_flag && !sps.sps_scaling_list_data_present_flag);
        assert!(sps.scaling_list_data.is_none());
        assert!(sps.amp_enabled_flag && sps.pcm_enabled_flag);
        assert_eq!(sps.pcm_sample_bit_depth_luma_minus1, 6);
        assert_eq!(sps.pcm_sample_bit_depth_chroma_minus1, 5);
        assert_eq!(sps.log2_diff_max_min_pcm_luma_coding_block_size, 2);
        assert!(sps.pcm_loop_filter_disabled_flag);
        assert_eq!(sps.st_ref_pic_set.len(), 2);
        assert_eq!(sps.st_ref_pic_set[0].num_delta_pocs(), 3);
        assert_eq!(sps.num_long_term_ref_pics_sps, 2);
        assert_eq!(sps.lt_ref_pic_poc_lsb_sps[..2], [17, 200]);
        assert_eq!(sps.used_by_curr_pic_lt_sps_flag[..2], [true, false]);
        assert!(sps.sps_range_extension.cabac_bypass_alignment_enabled_flag);
        assert!(sps.inter_view_mv_vert_constraint_flag);
    }

    #[test]
    fn range_tools_are_read_and_refused_on_request() {
        let sps = valid(&Spec {
            range_extension: Some([false, false, true, false, false, false, false, false, false]),
            ..Spec::default()
        });
        assert!(sps.sps_range_extension.implicit_rdpcm_enabled_flag);
        assert!(matches!(
            sps.sps_range_extension.refuse_tools(),
            Err(BitstreamError::InvalidField {
                field: "implicit_rdpcm_enabled_flag",
                value: 1,
                ..
            })
        ));
        assert_eq!(
            valid(&Spec::default()).sps_range_extension.refuse_tools(),
            Ok(())
        );
    }

    #[test]
    fn offset_ranges_follow_high_precision_offsets() {
        let plain = valid(&Spec {
            bit_depth_luma_minus8: 2,
            bit_depth_chroma_minus8: 4,
            ..Spec::default()
        });
        assert_eq!(plain.wp_offset_half_range_y(), 128);
        assert_eq!(plain.wp_offset_half_range_c(), 128);
        assert_eq!(plain.qp_bd_offset_y(), 12);
        let wide = valid(&Spec {
            bit_depth_luma_minus8: 2,
            bit_depth_chroma_minus8: 4,
            range_extension: Some([false, false, false, false, false, false, true, false, false]),
            ..Spec::default()
        });
        assert_eq!(wide.wp_offset_half_range_y(), 512);
        assert_eq!(wide.wp_offset_half_range_c(), 2048);
    }

    #[test]
    fn out_of_range_values_are_refused() {
        type Change = fn(&mut Spec);
        let cases: [(&str, u64, Change); 26] = [
            ("sps_max_sub_layers_minus1", 7, |s| s.sub_layers_minus1 = 7),
            ("sps_temporal_id_nesting_flag", 0, |s| s.nesting = false),
            ("sps_seq_parameter_set_id", 16, |s| s.sps_id = 16),
            ("chroma_format_idc", 4, |s| s.chroma_format_idc = 4),
            ("pic_width_in_luma_samples", 0, |s| s.width = 0),
            ("pic_width_in_luma_samples", 417, |s| s.width = 417),
            ("pic_width_in_luma_samples", 65_536, |s| s.width = 65_536),
            ("pic_height_in_luma_samples", 0, |s| s.height = 0),
            ("pic_height_in_luma_samples", 241, |s| s.height = 241),
            ("conf_win_right_offset", 208, |s| {
                s.conf_win = Some([0, 208, 0, 0])
            }),
            ("conf_win_bottom_offset", 120, |s| {
                s.conf_win = Some([0, 0, 0, 120])
            }),
            ("bit_depth_luma_minus8", 9, |s| s.bit_depth_luma_minus8 = 9),
            ("bit_depth_chroma_minus8", 9, |s| {
                s.bit_depth_chroma_minus8 = 9
            }),
            ("log2_max_pic_order_cnt_lsb_minus4", 13, |s| {
                s.poc_lsb_minus4 = 13
            }),
            ("sps_max_dec_pic_buffering_minus1", 16, |s| {
                s.dec_pic_buffering = [16, 16]
            }),
            ("sps_max_num_reorder_pics", 6, |s| s.num_reorder = [6, 6]),
            ("log2_min_luma_coding_block_size_minus3", 4, |s| {
                s.min_cb_minus3 = 4
            }),
            ("log2_diff_max_min_luma_coding_block_size", 4, |s| {
                s.diff_cb = 4
            }),
            ("log2_diff_max_min_luma_coding_block_size", 0, |s| {
                s.diff_cb = 0
            }),
            ("log2_min_luma_transform_block_size_minus2", 1, |s| {
                s.min_tb_minus2 = 1
            }),
            ("log2_diff_max_min_luma_transform_block_size", 4, |s| {
                s.diff_tb = 4
            }),
            ("max_transform_hierarchy_depth_inter", 5, |s| {
                s.depth_inter = 5
            }),
            ("max_transform_hierarchy_depth_intra", 5, |s| {
                s.depth_intra = 5
            }),
            ("num_long_term_ref_pics_sps", 33, |s| {
                s.long_term = vec![(1, true); 33]
            }),
            ("num_negative_pics", 6, |s| s.sets = vec![(6, 0)]),
            ("num_positive_pics", 4, |s| s.sets = vec![(2, 4)]),
        ];
        for (name, expected, change) in cases {
            let mut spec = Spec::default();
            change(&mut spec);
            let result = SeqParameterSet::parse(&build(&spec));
            assert!(
                matches!(
                    &result,
                    Err(BitstreamError::InvalidField { field, value, .. })
                        if *field == name && *value == expected
                ),
                "{name} {expected}: {result:?}"
            );
        }
    }

    #[test]
    fn pcm_limits_are_refused() {
        type Case<'a> = (&'a str, (u64, u64, u32, u32));
        let cases: [Case<'_>; 4] = [
            ("pcm_sample_bit_depth_luma_minus1", (8, 0, 0, 0)),
            ("pcm_sample_bit_depth_chroma_minus1", (0, 8, 0, 0)),
            ("log2_min_pcm_luma_coding_block_size_minus3", (0, 0, 3, 0)),
            ("log2_diff_max_min_pcm_luma_coding_block_size", (0, 0, 0, 3)),
        ];
        for (name, pcm) in cases {
            let result = SeqParameterSet::parse(&build(&Spec {
                pcm: Some(pcm),
                ..Spec::default()
            }));
            assert!(
                matches!(&result, Err(BitstreamError::InvalidField { field, .. }) if *field == name),
                "{name}: {result:?}"
            );
        }
        // Monochrome ignores the chroma depth.
        let mono = valid(&Spec {
            chroma_format_idc: 0,
            pcm: Some((0, 15, 0, 0)),
            ..Spec::default()
        });
        assert_eq!(mono.pcm_sample_bit_depth_chroma_minus1, 15);
    }

    #[test]
    fn scc_is_refused_and_sub_layers_fall_no_lower() {
        assert!(matches!(
            SeqParameterSet::parse(&build(&Spec {
                scc: true,
                ..Spec::default()
            })),
            Err(BitstreamError::InvalidField {
                field: "sps_scc_extension_flag",
                ..
            })
        ));
        assert!(matches!(
            SeqParameterSet::parse(&build(&Spec {
                sub_layers_minus1: 1,
                nesting: false,
                dec_pic_buffering: [5, 4],
                num_reorder: [0, 0],
                ..Spec::default()
            })),
            Err(BitstreamError::InvalidField {
                field: "sps_max_dec_pic_buffering_minus1",
                value: 4,
                ..
            })
        ));
        assert!(matches!(
            SeqParameterSet::parse(&build(&Spec {
                sub_layers_minus1: 1,
                nesting: false,
                num_reorder: [2, 1],
                ..Spec::default()
            })),
            Err(BitstreamError::InvalidField {
                field: "sps_max_num_reorder_pics",
                value: 1,
                ..
            })
        ));
    }

    #[test]
    fn chroma_formats_set_the_derived_variables() {
        let separate = valid(&Spec {
            chroma_format_idc: 3,
            separate_colour_plane: true,
            ..Spec::default()
        });
        assert_eq!(separate.chroma_array_type(), 0);
        assert_eq!((separate.sub_width_c(), separate.sub_height_c()), (1, 1));
        let packed = valid(&Spec {
            chroma_format_idc: 2,
            ..Spec::default()
        });
        assert_eq!(packed.chroma_array_type(), 2);
        assert_eq!((packed.sub_width_c(), packed.sub_height_c()), (2, 1));
    }

    #[test]
    fn truncation_at_every_byte_never_panics() {
        let nal = build(&Spec {
            conf_win: Some([1, 2, 0, 3]),
            pcm: Some((6, 5, 0, 2)),
            sets: vec![(2, 1), (0, 0)],
            long_term: vec![(17, true), (200, false)],
            range_extension: Some([true; 9]),
            multilayer: true,
            ..Spec::default()
        });
        for end in 0..nal.len() {
            let result = SeqParameterSet::parse(&nal[..end]);
            // Only the last byte can hold nothing but the trailing bits.
            if end + 1 < nal.len() {
                assert!(result.is_err(), "cut at byte {end}");
            }
        }
        assert!(SeqParameterSet::parse(&nal).is_ok());
    }
}
