//! The mapping from parsed HEVC parameter sets into Vulkan's `StdVideoH265*` structures. It
//! touches no device, so tests read each field back.
//!
//! An owner holds what a structure points to, in stable heap storage. `as_std` builds the
//! structure on demand, so the pointers stay valid while the owner lives, across moves too.
//!
//! The crate's VUI parser keeps the colour fields and `min_spatial_segmentation_idc`. So a
//! `StdVideoH265SequenceParameterSetVui` states those and leaves timing and HRD unset. A VPS
//! states no HRD parameters either. A decode reads none of them.

use std::ptr;

use ash::vk::native::{
    StdVideoH265DecPicBufMgr, StdVideoH265LongTermRefPicsSps, StdVideoH265PictureParameterSet,
    StdVideoH265PpsFlags, StdVideoH265ProfileTierLevel, StdVideoH265ScalingLists,
    StdVideoH265SequenceParameterSet, StdVideoH265SequenceParameterSetVui,
    StdVideoH265ShortTermRefPicSet, StdVideoH265ShortTermRefPicSetFlags, StdVideoH265SpsFlags,
    StdVideoH265SpsVuiFlags, StdVideoH265VideoParameterSet, StdVideoH265VpsFlags,
};

use super::std_common::{std_dec_pic_buf_mgr, std_profile_tier_level, std_scaling_lists};
use crate::bitstream::VuiSignal;
use crate::bitstream::hevc_pps::PicParameterSet;
use crate::bitstream::hevc_ptl::GeneralPtl;
use crate::bitstream::hevc_rps::ShortTermRefPicSet;
use crate::bitstream::hevc_scaling::ScalingListData;
use crate::bitstream::hevc_sps_full::SeqParameterSet;
use crate::bitstream::hevc_vps::VideoParameterSet;
use crate::vulkan::retained::{Retained, RetainedList};
use crate::vulkan::zeroed::{AllZero, zeroed};

// SAFETY: integers and bitfield units only.
unsafe impl AllZero for StdVideoH265VpsFlags {}
// SAFETY: integers, bitfield units, and raw pointers only.
unsafe impl AllZero for StdVideoH265VideoParameterSet {}
// SAFETY: integers and bitfield units only.
unsafe impl AllZero for StdVideoH265SpsFlags {}
// SAFETY: integers, bitfield units, and raw pointers only.
unsafe impl AllZero for StdVideoH265SequenceParameterSet {}
// SAFETY: integers and bitfield units only.
unsafe impl AllZero for StdVideoH265SpsVuiFlags {}
// SAFETY: integers, bitfield units, and raw pointers only.
unsafe impl AllZero for StdVideoH265SequenceParameterSetVui {}
// SAFETY: integers and bitfield units only.
unsafe impl AllZero for StdVideoH265PpsFlags {}
// SAFETY: integers, bitfield units, and raw pointers only.
unsafe impl AllZero for StdVideoH265PictureParameterSet {}
// SAFETY: integers and bitfield units only.
unsafe impl AllZero for StdVideoH265ShortTermRefPicSetFlags {}
// SAFETY: integers and bitfield units only.
unsafe impl AllZero for StdVideoH265ShortTermRefPicSet {}

/// `aspect_ratio_idc` of an extended ratio.
const EXTENDED_SAR: u32 = 255;

/// A bit mask with bit `index` set for each true entry of `flags`.
fn flag_mask(flags: &[bool]) -> u32 {
    flags
        .iter()
        .enumerate()
        .fold(0, |bits, (index, &flag)| bits | (u32::from(flag) << index))
}

fn ptl_of(general: &GeneralPtl) -> StdVideoH265ProfileTierLevel {
    std_profile_tier_level(
        general.profile_idc,
        general.tier_flag,
        general.level_idc,
        general.constraint_flags,
    )
}

/// A VPS as a `StdVideoH265VideoParameterSet`, with what it points to.
pub(super) struct StdVps {
    value: Retained<StdVideoH265VideoParameterSet>,
    ptl: Retained<StdVideoH265ProfileTierLevel>,
    buffers: Retained<StdVideoH265DecPicBufMgr>,
}

impl StdVps {
    pub(super) fn new(vps: &VideoParameterSet) -> Self {
        let mut flags: StdVideoH265VpsFlags = zeroed();
        flags.set_vps_temporal_id_nesting_flag(vps.vps_temporal_id_nesting_flag.into());
        flags.set_vps_sub_layer_ordering_info_present_flag(
            vps.vps_sub_layer_ordering_info_present_flag.into(),
        );
        flags.set_vps_timing_info_present_flag(vps.vps_timing_info_present_flag.into());
        flags.set_vps_poc_proportional_to_timing_flag(
            vps.vps_poc_proportional_to_timing_flag.into(),
        );
        let mut value: StdVideoH265VideoParameterSet = zeroed();
        value.flags = flags;
        value.vps_video_parameter_set_id = vps.vps_video_parameter_set_id;
        value.vps_max_sub_layers_minus1 = vps.vps_max_sub_layers_minus1;
        value.vps_num_units_in_tick = vps.vps_num_units_in_tick;
        value.vps_time_scale = vps.vps_time_scale;
        value.vps_num_ticks_poc_diff_one_minus1 = vps.vps_num_ticks_poc_diff_one_minus1;
        Self {
            value: Retained::new(value),
            ptl: Retained::new(ptl_of(&vps.profile_tier_level.general)),
            buffers: Retained::new(std_dec_pic_buf_mgr(
                &vps.vps_max_latency_increase_plus1,
                &vps.vps_max_dec_pic_buffering_minus1,
                &vps.vps_max_num_reorder_pics,
            )),
        }
    }

    pub(super) fn as_std(&self) -> StdVideoH265VideoParameterSet {
        StdVideoH265VideoParameterSet {
            pDecPicBufMgr: self.buffers.as_ptr(),
            pHrdParameters: ptr::null(),
            pProfileTierLevel: self.ptl.as_ptr(),
            ..self.value.get()
        }
    }
}

/// An SPS as a `StdVideoH265SequenceParameterSet`, with what it points to.
pub(super) struct StdSps {
    value: Retained<StdVideoH265SequenceParameterSet>,
    ptl: Retained<StdVideoH265ProfileTierLevel>,
    buffers: Retained<StdVideoH265DecPicBufMgr>,
    scaling: Option<Retained<StdVideoH265ScalingLists>>,
    short_term: RetainedList<StdVideoH265ShortTermRefPicSet>,
    long_term: Retained<StdVideoH265LongTermRefPicsSps>,
    vui: Option<Retained<StdVideoH265SequenceParameterSetVui>>,
}

fn sps_flags(sps: &SeqParameterSet) -> StdVideoH265SpsFlags {
    let mut flags: StdVideoH265SpsFlags = zeroed();
    flags.set_sps_temporal_id_nesting_flag(sps.sps_temporal_id_nesting_flag.into());
    flags.set_separate_colour_plane_flag(sps.separate_colour_plane_flag.into());
    flags.set_conformance_window_flag(sps.conformance_window_flag.into());
    flags.set_sps_sub_layer_ordering_info_present_flag(
        sps.sps_sub_layer_ordering_info_present_flag.into(),
    );
    flags.set_scaling_list_enabled_flag(sps.scaling_list_enabled_flag.into());
    flags.set_sps_scaling_list_data_present_flag(sps.sps_scaling_list_data_present_flag.into());
    flags.set_amp_enabled_flag(sps.amp_enabled_flag.into());
    flags.set_sample_adaptive_offset_enabled_flag(sps.sample_adaptive_offset_enabled_flag.into());
    flags.set_pcm_enabled_flag(sps.pcm_enabled_flag.into());
    flags.set_pcm_loop_filter_disabled_flag(sps.pcm_loop_filter_disabled_flag.into());
    flags.set_long_term_ref_pics_present_flag(sps.long_term_ref_pics_present_flag.into());
    flags.set_sps_temporal_mvp_enabled_flag(sps.sps_temporal_mvp_enabled_flag.into());
    flags.set_strong_intra_smoothing_enabled_flag(sps.strong_intra_smoothing_enabled_flag.into());
    flags.set_vui_parameters_present_flag(sps.vui_parameters_present_flag.into());
    flags.set_sps_extension_present_flag(sps.sps_extension_present_flag.into());
    flags.set_sps_range_extension_flag(sps.sps_range_extension_flag.into());
    let range = &sps.sps_range_extension;
    flags.set_transform_skip_rotation_enabled_flag(
        range.transform_skip_rotation_enabled_flag.into(),
    );
    flags.set_transform_skip_context_enabled_flag(range.transform_skip_context_enabled_flag.into());
    flags.set_implicit_rdpcm_enabled_flag(range.implicit_rdpcm_enabled_flag.into());
    flags.set_explicit_rdpcm_enabled_flag(range.explicit_rdpcm_enabled_flag.into());
    flags.set_extended_precision_processing_flag(range.extended_precision_processing_flag.into());
    flags.set_intra_smoothing_disabled_flag(range.intra_smoothing_disabled_flag.into());
    flags.set_high_precision_offsets_enabled_flag(range.high_precision_offsets_enabled_flag.into());
    flags.set_persistent_rice_adaptation_enabled_flag(
        range.persistent_rice_adaptation_enabled_flag.into(),
    );
    flags.set_cabac_bypass_alignment_enabled_flag(range.cabac_bypass_alignment_enabled_flag.into());
    flags
}

/// One `st_ref_pic_set()`. A set in the SPS has `delta_idx_minus1` 0.
fn std_short_term(set: &ShortTermRefPicSet) -> StdVideoH265ShortTermRefPicSet {
    let mut flags: StdVideoH265ShortTermRefPicSetFlags = zeroed();
    flags.set_inter_ref_pic_set_prediction_flag(set.inter_ref_pic_set_prediction_flag.into());
    flags.set_delta_rps_sign(set.delta_rps_sign.into());
    let mut value: StdVideoH265ShortTermRefPicSet = zeroed();
    value.flags = flags;
    value.delta_idx_minus1 = set.delta_idx_minus1;
    value.use_delta_flag = set.use_delta_flag as u16;
    value.abs_delta_rps_minus1 = set.abs_delta_rps_minus1 as u16;
    value.used_by_curr_pic_flag = set.used_by_curr_pic_flag as u16;
    value.used_by_curr_pic_s0_flag = flag_mask(&set.used_by_curr_pic_s0) as u16;
    value.used_by_curr_pic_s1_flag = flag_mask(&set.used_by_curr_pic_s1) as u16;
    value.num_negative_pics = set.num_negative_pics;
    value.num_positive_pics = set.num_positive_pics;
    value.delta_poc_s0_minus1 = set.delta_poc_s0_minus1;
    value.delta_poc_s1_minus1 = set.delta_poc_s1_minus1;
    value
}

fn std_long_term(sps: &SeqParameterSet) -> StdVideoH265LongTermRefPicsSps {
    StdVideoH265LongTermRefPicsSps {
        used_by_curr_pic_lt_sps_flag: flag_mask(&sps.used_by_curr_pic_lt_sps_flag),
        lt_ref_pic_poc_lsb_sps: sps.lt_ref_pic_poc_lsb_sps.map(u32::from),
    }
}

/// The VUI fields the parser keeps.
fn std_vui(
    signal: &VuiSignal,
    min_spatial_segmentation_idc: Option<u32>,
) -> StdVideoH265SequenceParameterSetVui {
    let mut flags: StdVideoH265SpsVuiFlags = zeroed();
    let mut value: StdVideoH265SequenceParameterSetVui = zeroed();
    if let Some((width, height)) = signal.sample_aspect_ratio {
        flags.set_aspect_ratio_info_present_flag(1);
        value.aspect_ratio_idc = EXTENDED_SAR;
        value.sar_width = width;
        value.sar_height = height;
    }
    if let Some(video) = signal.video_signal_type {
        flags.set_video_signal_type_present_flag(1);
        flags.set_video_full_range_flag(video.full_range.into());
        value.video_format = video.video_format;
        if let Some(colour) = video.colour {
            flags.set_colour_description_present_flag(1);
            value.colour_primaries = colour.colour_primaries;
            value.transfer_characteristics = colour.transfer_characteristics;
            value.matrix_coeffs = colour.matrix_coefficients;
        }
    }
    if let Some(location) = signal.chroma_sample_loc {
        flags.set_chroma_loc_info_present_flag(1);
        value.chroma_sample_loc_type_top_field = location.top_field;
        value.chroma_sample_loc_type_bottom_field = location.bottom_field;
    }
    if let Some(idc) = min_spatial_segmentation_idc {
        flags.set_bitstream_restriction_flag(1);
        value.min_spatial_segmentation_idc = idc as u16;
    }
    value.flags = flags;
    value
}

impl StdSps {
    pub(super) fn new(sps: &SeqParameterSet) -> Self {
        let mut value: StdVideoH265SequenceParameterSet = zeroed();
        value.flags = sps_flags(sps);
        value.chroma_format_idc = sps.chroma_format_idc;
        value.pic_width_in_luma_samples = sps.pic_width_in_luma_samples;
        value.pic_height_in_luma_samples = sps.pic_height_in_luma_samples;
        value.sps_video_parameter_set_id = sps.sps_video_parameter_set_id;
        value.sps_max_sub_layers_minus1 = sps.sps_max_sub_layers_minus1;
        value.sps_seq_parameter_set_id = sps.sps_seq_parameter_set_id as u8;
        value.bit_depth_luma_minus8 = sps.bit_depth_luma_minus8 as u8;
        value.bit_depth_chroma_minus8 = sps.bit_depth_chroma_minus8 as u8;
        value.log2_max_pic_order_cnt_lsb_minus4 = sps.log2_max_pic_order_cnt_lsb_minus4 as u8;
        value.log2_min_luma_coding_block_size_minus3 =
            sps.log2_min_luma_coding_block_size_minus3 as u8;
        value.log2_diff_max_min_luma_coding_block_size =
            sps.log2_diff_max_min_luma_coding_block_size as u8;
        value.log2_min_luma_transform_block_size_minus2 =
            sps.log2_min_luma_transform_block_size_minus2 as u8;
        value.log2_diff_max_min_luma_transform_block_size =
            sps.log2_diff_max_min_luma_transform_block_size as u8;
        value.max_transform_hierarchy_depth_inter = sps.max_transform_hierarchy_depth_inter as u8;
        value.max_transform_hierarchy_depth_intra = sps.max_transform_hierarchy_depth_intra as u8;
        value.num_short_term_ref_pic_sets = sps.st_ref_pic_set.len() as u8;
        value.num_long_term_ref_pics_sps = sps.num_long_term_ref_pics_sps as u8;
        value.pcm_sample_bit_depth_luma_minus1 = sps.pcm_sample_bit_depth_luma_minus1;
        value.pcm_sample_bit_depth_chroma_minus1 = sps.pcm_sample_bit_depth_chroma_minus1;
        value.log2_min_pcm_luma_coding_block_size_minus3 =
            sps.log2_min_pcm_luma_coding_block_size_minus3 as u8;
        value.log2_diff_max_min_pcm_luma_coding_block_size =
            sps.log2_diff_max_min_pcm_luma_coding_block_size as u8;
        value.conf_win_left_offset = sps.conf_win_left_offset;
        value.conf_win_right_offset = sps.conf_win_right_offset;
        value.conf_win_top_offset = sps.conf_win_top_offset;
        value.conf_win_bottom_offset = sps.conf_win_bottom_offset;
        let vui = sps
            .vui
            .as_ref()
            .map(|vui| Retained::new(std_vui(&vui.signal, vui.min_spatial_segmentation_idc)));
        Self {
            value: Retained::new(value),
            ptl: Retained::new(ptl_of(&sps.profile_tier_level.general)),
            buffers: Retained::new(std_dec_pic_buf_mgr(
                &sps.sps_max_latency_increase_plus1,
                &sps.sps_max_dec_pic_buffering_minus1,
                &sps.sps_max_num_reorder_pics,
            )),
            scaling: scaling_of(
                sps.sps_scaling_list_data_present_flag,
                &sps.scaling_list_data,
            ),
            short_term: RetainedList::new(sps.st_ref_pic_set.iter().map(std_short_term).collect()),
            long_term: Retained::new(std_long_term(sps)),
            vui,
        }
    }

    pub(super) fn as_std(&self) -> StdVideoH265SequenceParameterSet {
        StdVideoH265SequenceParameterSet {
            pProfileTierLevel: self.ptl.as_ptr(),
            pDecPicBufMgr: self.buffers.as_ptr(),
            pScalingLists: self.scaling.as_ref().map_or(ptr::null(), Retained::as_ptr),
            pShortTermRefPicSet: if self.short_term.is_empty() {
                ptr::null()
            } else {
                self.short_term.as_ptr()
            },
            pLongTermRefPicsSps: self.long_term.as_ptr(),
            pSequenceParameterSetVui: self.vui.as_ref().map_or(ptr::null(), Retained::as_ptr),
            pPredictorPaletteEntries: ptr::null(),
            ..self.value.get()
        }
    }
}

/// The lists of a parameter set whose data-present flag is 1. With the flag 0, a null pointer
/// states the inference: a PPS takes the SPS's lists, and an SPS the default lists.
fn scaling_of(
    present: bool,
    data: &Option<Box<ScalingListData>>,
) -> Option<Retained<StdVideoH265ScalingLists>> {
    data.as_deref()
        .filter(|_| present)
        .map(|data| Retained::new(std_scaling_lists(data)))
}

fn pps_flags(pps: &PicParameterSet) -> StdVideoH265PpsFlags {
    let mut flags: StdVideoH265PpsFlags = zeroed();
    flags.set_dependent_slice_segments_enabled_flag(
        pps.dependent_slice_segments_enabled_flag.into(),
    );
    flags.set_output_flag_present_flag(pps.output_flag_present_flag.into());
    flags.set_sign_data_hiding_enabled_flag(pps.sign_data_hiding_enabled_flag.into());
    flags.set_cabac_init_present_flag(pps.cabac_init_present_flag.into());
    flags.set_constrained_intra_pred_flag(pps.constrained_intra_pred_flag.into());
    flags.set_transform_skip_enabled_flag(pps.transform_skip_enabled_flag.into());
    flags.set_cu_qp_delta_enabled_flag(pps.cu_qp_delta_enabled_flag.into());
    flags.set_pps_slice_chroma_qp_offsets_present_flag(
        pps.pps_slice_chroma_qp_offsets_present_flag.into(),
    );
    flags.set_weighted_pred_flag(pps.weighted_pred_flag.into());
    flags.set_weighted_bipred_flag(pps.weighted_bipred_flag.into());
    flags.set_transquant_bypass_enabled_flag(pps.transquant_bypass_enabled_flag.into());
    flags.set_tiles_enabled_flag(pps.tiles_enabled_flag.into());
    flags.set_entropy_coding_sync_enabled_flag(pps.entropy_coding_sync_enabled_flag.into());
    flags.set_uniform_spacing_flag(pps.uniform_spacing_flag.into());
    flags.set_loop_filter_across_tiles_enabled_flag(
        pps.loop_filter_across_tiles_enabled_flag.into(),
    );
    flags.set_pps_loop_filter_across_slices_enabled_flag(
        pps.pps_loop_filter_across_slices_enabled_flag.into(),
    );
    flags.set_deblocking_filter_control_present_flag(
        pps.deblocking_filter_control_present_flag.into(),
    );
    flags.set_deblocking_filter_override_enabled_flag(
        pps.deblocking_filter_override_enabled_flag.into(),
    );
    flags.set_pps_deblocking_filter_disabled_flag(pps.pps_deblocking_filter_disabled_flag.into());
    flags.set_pps_scaling_list_data_present_flag(pps.pps_scaling_list_data_present_flag.into());
    flags.set_lists_modification_present_flag(pps.lists_modification_present_flag.into());
    flags.set_slice_segment_header_extension_present_flag(
        pps.slice_segment_header_extension_present_flag.into(),
    );
    flags.set_pps_extension_present_flag(pps.pps_extension_present_flag.into());
    flags.set_pps_range_extension_flag(pps.pps_range_extension_flag.into());
    let range = &pps.pps_range_extension;
    flags.set_cross_component_prediction_enabled_flag(
        range.cross_component_prediction_enabled_flag.into(),
    );
    flags.set_chroma_qp_offset_list_enabled_flag(range.chroma_qp_offset_list_enabled_flag.into());
    flags
}

/// A PPS as a `StdVideoH265PictureParameterSet`, with what it points to.
pub(super) struct StdPps {
    value: Retained<StdVideoH265PictureParameterSet>,
    scaling: Option<Retained<StdVideoH265ScalingLists>>,
}

impl StdPps {
    /// Tile sizes come from (6-3) and (6-4) on `sps`, for uniform spacing too.
    pub(super) fn new(pps: &PicParameterSet, sps: &SeqParameterSet) -> Self {
        let mut value: StdVideoH265PictureParameterSet = zeroed();
        value.flags = pps_flags(pps);
        value.pps_pic_parameter_set_id = pps.pps_pic_parameter_set_id as u8;
        value.pps_seq_parameter_set_id = pps.pps_seq_parameter_set_id as u8;
        value.sps_video_parameter_set_id = sps.sps_video_parameter_set_id;
        value.num_extra_slice_header_bits = pps.num_extra_slice_header_bits;
        value.num_ref_idx_l0_default_active_minus1 = pps.num_ref_idx_l0_default_active_minus1 as u8;
        value.num_ref_idx_l1_default_active_minus1 = pps.num_ref_idx_l1_default_active_minus1 as u8;
        value.init_qp_minus26 = pps.init_qp_minus26 as i8;
        value.diff_cu_qp_delta_depth = pps.diff_cu_qp_delta_depth as u8;
        value.pps_cb_qp_offset = pps.pps_cb_qp_offset as i8;
        value.pps_cr_qp_offset = pps.pps_cr_qp_offset as i8;
        value.pps_beta_offset_div2 = pps.pps_beta_offset_div2 as i8;
        value.pps_tc_offset_div2 = pps.pps_tc_offset_div2 as i8;
        value.log2_parallel_merge_level_minus2 = pps.log2_parallel_merge_level_minus2 as u8;
        let range = &pps.pps_range_extension;
        value.log2_max_transform_skip_block_size_minus2 =
            range.log2_max_transform_skip_block_size_minus2 as u8;
        value.diff_cu_chroma_qp_offset_depth = range.diff_cu_chroma_qp_offset_depth as u8;
        value.chroma_qp_offset_list_len_minus1 = range.chroma_qp_offset_list_len_minus1 as u8;
        value.cb_qp_offset_list = range.cb_qp_offset_list;
        value.cr_qp_offset_list = range.cr_qp_offset_list;
        value.log2_sao_offset_scale_luma = range.log2_sao_offset_scale_luma as u8;
        value.log2_sao_offset_scale_chroma = range.log2_sao_offset_scale_chroma as u8;
        value.num_tile_columns_minus1 = pps.num_tile_columns_minus1 as u8;
        value.num_tile_rows_minus1 = pps.num_tile_rows_minus1 as u8;
        if pps.tiles_enabled_flag {
            let columns = pps.column_widths(sps);
            for (target, width) in value.column_width_minus1.iter_mut().zip(&columns) {
                *target = width.saturating_sub(1) as u16;
            }
            let rows = pps.row_heights(sps);
            for (target, height) in value.row_height_minus1.iter_mut().zip(&rows) {
                *target = height.saturating_sub(1) as u16;
            }
        }
        Self {
            value: Retained::new(value),
            scaling: scaling_of(
                pps.pps_scaling_list_data_present_flag,
                &pps.scaling_list_data,
            ),
        }
    }

    pub(super) fn as_std(&self) -> StdVideoH265PictureParameterSet {
        StdVideoH265PictureParameterSet {
            pScalingLists: self.scaling.as_ref().map_or(ptr::null(), Retained::as_ptr),
            pPredictorPaletteEntries: ptr::null(),
            ..self.value.get()
        }
    }
}

/// The parameter sets of one session parameters object, with each array the create call names.
pub(super) struct StdParameterSets {
    _vps: StdVps,
    _sps: StdSps,
    _pps: Vec<StdPps>,
    vps_list: RetainedList<StdVideoH265VideoParameterSet>,
    sps_list: RetainedList<StdVideoH265SequenceParameterSet>,
    pps_list: RetainedList<StdVideoH265PictureParameterSet>,
}

impl StdParameterSets {
    pub(super) fn new(vps: StdVps, sps: StdSps, pps: Vec<StdPps>) -> Self {
        let vps_list = RetainedList::new(vec![vps.as_std()]);
        let sps_list = RetainedList::new(vec![sps.as_std()]);
        let pps_list = RetainedList::new(pps.iter().map(StdPps::as_std).collect());
        Self {
            _vps: vps,
            _sps: sps,
            _pps: pps,
            vps_list,
            sps_list,
            pps_list,
        }
    }

    pub(super) fn vps_list(&self) -> &[StdVideoH265VideoParameterSet] {
        self.vps_list.as_slice()
    }

    pub(super) fn sps_list(&self) -> &[StdVideoH265SequenceParameterSet] {
        self.sps_list.as_slice()
    }

    pub(super) fn pps_list(&self) -> &[StdVideoH265PictureParameterSet] {
        self.pps_list.as_slice()
    }
}
