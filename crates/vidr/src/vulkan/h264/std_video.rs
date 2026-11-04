//! The mapping from parsed H.264 headers into Vulkan's `StdVideoH264*` structures. It
//! touches no device, so tests read each field back.
//!
//! An owner holds what a structure points to. `as_std` builds the structure on demand,
//! so the pointers stay valid while the owner lives.

use std::ptr;

use ash::vk::native::{
    StdVideoDecodeH264PictureInfo, StdVideoDecodeH264PictureInfoFlags,
    StdVideoDecodeH264ReferenceInfo, StdVideoDecodeH264ReferenceInfoFlags,
    StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_INVALID, StdVideoH264PictureParameterSet,
    StdVideoH264PpsFlags, StdVideoH264ScalingLists, StdVideoH264SequenceParameterSet,
    StdVideoH264SequenceParameterSetVui, StdVideoH264SpsFlags, StdVideoH264SpsVuiFlags,
};
use h264_reader::nal::pps::PicParameterSet;
use h264_reader::nal::sps::{
    AspectRatioInfo, FrameMbsFlags, OverscanAppropriate, PicOrderCntType, SeqParameterSet,
    VuiParameters,
};

use super::dpb::RefFrame;
use crate::vulkan::zeroed::{AllZero, zeroed};

// SAFETY: integers, bitfield units, and raw pointers only.
unsafe impl AllZero for StdVideoH264SequenceParameterSet {}
// SAFETY: integers and bitfield units only.
unsafe impl AllZero for StdVideoH264SpsFlags {}
// SAFETY: as above.
unsafe impl AllZero for StdVideoH264SpsVuiFlags {}
// SAFETY: integers, bitfield units, and raw pointers only.
unsafe impl AllZero for StdVideoH264SequenceParameterSetVui {}
// SAFETY: integers and bitfield units only.
unsafe impl AllZero for StdVideoH264PpsFlags {}
// SAFETY: integers, bitfield units, and raw pointers only.
unsafe impl AllZero for StdVideoH264PictureParameterSet {}
// SAFETY: integers and bitfield units only.
unsafe impl AllZero for StdVideoDecodeH264PictureInfoFlags {}
// SAFETY: as above.
unsafe impl AllZero for StdVideoDecodeH264ReferenceInfoFlags {}

/// The `StdVideoH264LevelIdc` of `level_idc`, or `None` for a level with no value.
/// Level 1b, which `level_idc` 9 states in High profiles, takes level 1.1's value.
pub(super) const fn std_level(level_idc: u8) -> Option<u32> {
    Some(match level_idc {
        10 => 0,
        9 | 11 => 1,
        12 => 2,
        13 => 3,
        20 => 4,
        21 => 5,
        22 => 6,
        30 => 7,
        31 => 8,
        32 => 9,
        40 => 10,
        41 => 11,
        42 => 12,
        50 => 13,
        51 => 14,
        52 => 15,
        60 => 16,
        61 => 17,
        62 => 18,
        _ => return None,
    })
}

/// Every list resolved, so each present bit is set and no default bit is. Lists stay in
/// the scan order the headers state them in, which is the order Vulkan takes.
fn scaling_lists(
    lists_4x4: [[u8; 16]; 6],
    lists_8x8: &[[u8; 64]],
) -> Box<StdVideoH264ScalingLists> {
    let mut lists = Box::new(StdVideoH264ScalingLists {
        scaling_list_present_mask: 0b11_1111,
        use_default_scaling_matrix_mask: 0,
        ScalingList4x4: lists_4x4,
        ScalingList8x8: [[16; 64]; 6],
    });
    for (index, list) in lists_8x8.iter().take(6).enumerate() {
        lists.ScalingList8x8[index] = *list;
        lists.scaling_list_present_mask |= 1 << (6 + index);
    }
    lists
}

/// An SPS as a `StdVideoH264SequenceParameterSet`, with what it points to.
pub(super) struct StdSps {
    value: StdVideoH264SequenceParameterSet,
    offsets: Vec<i32>,
    scaling: Option<Box<StdVideoH264ScalingLists>>,
    vui: Option<Box<StdVideoH264SequenceParameterSetVui>>,
}

impl StdSps {
    pub(super) fn new(sps: &SeqParameterSet) -> Self {
        let mut flags: StdVideoH264SpsFlags = zeroed();
        let constraints = sps.constraint_flags;
        flags.set_constraint_set0_flag(constraints.flag0().into());
        flags.set_constraint_set1_flag(constraints.flag1().into());
        flags.set_constraint_set2_flag(constraints.flag2().into());
        flags.set_constraint_set3_flag(constraints.flag3().into());
        flags.set_constraint_set4_flag(constraints.flag4().into());
        flags.set_constraint_set5_flag(constraints.flag5().into());
        flags.set_direct_8x8_inference_flag(sps.direct_8x8_inference_flag.into());
        flags.set_frame_mbs_only_flag(matches!(sps.frame_mbs_flags, FrameMbsFlags::Frames).into());
        flags.set_gaps_in_frame_num_value_allowed_flag(
            sps.gaps_in_frame_num_value_allowed_flag.into(),
        );
        flags.set_qpprime_y_zero_transform_bypass_flag(
            sps.chroma_info.qpprime_y_zero_transform_bypass_flag.into(),
        );
        flags.set_frame_cropping_flag(sps.frame_cropping.is_some().into());
        flags.set_seq_scaling_matrix_present_flag(sps.chroma_info.scaling_matrix.is_some().into());
        flags.set_vui_parameters_present_flag(sps.vui_parameters.is_some().into());

        let mut value: StdVideoH264SequenceParameterSet = zeroed();
        let mut offsets = Vec::new();
        match &sps.pic_order_cnt {
            PicOrderCntType::TypeZero {
                log2_max_pic_order_cnt_lsb_minus4,
            } => {
                value.pic_order_cnt_type = 0;
                value.log2_max_pic_order_cnt_lsb_minus4 = *log2_max_pic_order_cnt_lsb_minus4;
            }
            PicOrderCntType::TypeOne {
                delta_pic_order_always_zero_flag,
                offset_for_non_ref_pic,
                offset_for_top_to_bottom_field,
                offsets_for_ref_frame,
            } => {
                value.pic_order_cnt_type = 1;
                flags.set_delta_pic_order_always_zero_flag(
                    (*delta_pic_order_always_zero_flag).into(),
                );
                value.offset_for_non_ref_pic = *offset_for_non_ref_pic;
                value.offset_for_top_to_bottom_field = *offset_for_top_to_bottom_field;
                value.num_ref_frames_in_pic_order_cnt_cycle = offsets_for_ref_frame.len() as u8;
                offsets.clone_from(offsets_for_ref_frame);
            }
            PicOrderCntType::TypeTwo => value.pic_order_cnt_type = 2,
        }
        let profile_idc: u8 = sps.profile_idc.into();
        value.flags = flags;
        value.profile_idc = u32::from(profile_idc);
        value.level_idc = std_level(sps.level_idc)
            .unwrap_or(StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_INVALID);
        value.chroma_format_idc = sps.chroma_info.chroma_format.to_u32();
        value.seq_parameter_set_id = sps.seq_parameter_set_id.id();
        value.bit_depth_luma_minus8 = sps.chroma_info.bit_depth_luma_minus8;
        value.bit_depth_chroma_minus8 = sps.chroma_info.bit_depth_chroma_minus8;
        value.log2_max_frame_num_minus4 = sps.log2_max_frame_num_minus4;
        value.max_num_ref_frames = sps.max_num_ref_frames as u8;
        value.pic_width_in_mbs_minus1 = sps.pic_width_in_mbs_minus1;
        value.pic_height_in_map_units_minus1 = sps.pic_height_in_map_units_minus1;
        if let Some(crop) = &sps.frame_cropping {
            value.frame_crop_left_offset = crop.left_offset;
            value.frame_crop_right_offset = crop.right_offset;
            value.frame_crop_top_offset = crop.top_offset;
            value.frame_crop_bottom_offset = crop.bottom_offset;
        }
        let scaling = sps.chroma_info.scaling_matrix.as_ref().map(|matrix| {
            let lists_8x8 = matrix.scaling_lists_8x8();
            scaling_lists(matrix.scaling_lists_4x4(), lists_8x8.as_slice())
        });
        Self {
            value,
            offsets,
            scaling,
            vui: sps.vui_parameters.as_ref().map(std_vui),
        }
    }

    /// The structure, valid while `self` lives.
    pub(super) fn as_std(&self) -> StdVideoH264SequenceParameterSet {
        let mut value = self.value;
        if !self.offsets.is_empty() {
            value.pOffsetForRefFrame = self.offsets.as_ptr();
        }
        if let Some(scaling) = &self.scaling {
            value.pScalingLists = ptr::from_ref(&**scaling);
        }
        if let Some(vui) = &self.vui {
            value.pSequenceParameterSetVui = ptr::from_ref(&**vui);
        }
        value
    }
}

/// The VUI, without HRD parameters. A decode reads none of them, so both present flags
/// stay 0 and `pHrdParameters` stays null.
fn std_vui(vui: &VuiParameters) -> Box<StdVideoH264SequenceParameterSetVui> {
    let mut flags: StdVideoH264SpsVuiFlags = zeroed();
    let mut value: StdVideoH264SequenceParameterSetVui = zeroed();
    if let Some(aspect) = &vui.aspect_ratio_info {
        flags.set_aspect_ratio_info_present_flag(1);
        value.aspect_ratio_idc = u32::from(aspect.to_u8());
        if let AspectRatioInfo::Extended(width, height) = aspect {
            value.sar_width = *width;
            value.sar_height = *height;
        }
    }
    match vui.overscan_appropriate {
        OverscanAppropriate::Unspecified => {}
        OverscanAppropriate::Appropriate => {
            flags.set_overscan_info_present_flag(1);
            flags.set_overscan_appropriate_flag(1);
        }
        OverscanAppropriate::Inappropriate => flags.set_overscan_info_present_flag(1),
    }
    // H.264 E.2.1 infers 5, unspecified, when the VUI omits video_format.
    value.video_format = 5;
    if let Some(signal) = &vui.video_signal_type {
        flags.set_video_signal_type_present_flag(1);
        flags.set_video_full_range_flag(signal.video_full_range_flag.into());
        value.video_format = signal.video_format.to_u8();
        if let Some(description) = &signal.colour_description {
            flags.set_color_description_present_flag(1);
            value.colour_primaries = description.colour_primaries;
            value.transfer_characteristics = description.transfer_characteristics;
            value.matrix_coefficients = description.matrix_coefficients;
        }
    }
    if let Some(location) = &vui.chroma_loc_info {
        flags.set_chroma_loc_info_present_flag(1);
        value.chroma_sample_loc_type_top_field = location.chroma_sample_loc_type_top_field as u8;
        value.chroma_sample_loc_type_bottom_field =
            location.chroma_sample_loc_type_bottom_field as u8;
    }
    if let Some(timing) = &vui.timing_info {
        flags.set_timing_info_present_flag(1);
        flags.set_fixed_frame_rate_flag(timing.fixed_frame_rate_flag.into());
        value.num_units_in_tick = timing.num_units_in_tick;
        value.time_scale = timing.time_scale;
    }
    if let Some(restrictions) = &vui.bitstream_restrictions {
        flags.set_bitstream_restriction_flag(1);
        value.max_num_reorder_frames = restrictions.max_num_reorder_frames as u8;
        value.max_dec_frame_buffering = restrictions.max_dec_frame_buffering as u8;
    }
    value.flags = flags;
    Box::new(value)
}

/// A PPS as a `StdVideoH264PictureParameterSet`, with what it points to.
pub(super) struct StdPps {
    value: StdVideoH264PictureParameterSet,
    scaling: Option<Box<StdVideoH264ScalingLists>>,
}

impl StdPps {
    pub(super) fn new(pps: &PicParameterSet, sps: &SeqParameterSet) -> Self {
        let mut flags: StdVideoH264PpsFlags = zeroed();
        let extension = pps.extension.as_ref();
        let matrix = extension.and_then(|extra| extra.pic_scaling_matrix.as_ref());
        flags.set_transform_8x8_mode_flag(
            extension
                .is_some_and(|extra| extra.transform_8x8_mode_flag)
                .into(),
        );
        flags.set_redundant_pic_cnt_present_flag(pps.redundant_pic_cnt_present_flag.into());
        flags.set_constrained_intra_pred_flag(pps.constrained_intra_pred_flag.into());
        flags.set_deblocking_filter_control_present_flag(
            pps.deblocking_filter_control_present_flag.into(),
        );
        flags.set_weighted_pred_flag(pps.weighted_pred_flag.into());
        flags.set_bottom_field_pic_order_in_frame_present_flag(
            pps.bottom_field_pic_order_in_frame_present_flag.into(),
        );
        flags.set_entropy_coding_mode_flag(pps.entropy_coding_mode_flag.into());
        flags.set_pic_scaling_matrix_present_flag(matrix.is_some().into());

        let mut value: StdVideoH264PictureParameterSet = zeroed();
        value.flags = flags;
        value.seq_parameter_set_id = pps.seq_parameter_set_id.id();
        value.pic_parameter_set_id = pps.pic_parameter_set_id.id();
        value.num_ref_idx_l0_default_active_minus1 = pps.num_ref_idx_l0_default_active_minus1 as u8;
        value.num_ref_idx_l1_default_active_minus1 = pps.num_ref_idx_l1_default_active_minus1 as u8;
        value.weighted_bipred_idc = u32::from(pps.weighted_bipred_idc);
        value.pic_init_qp_minus26 = pps.pic_init_qp_minus26 as i8;
        value.pic_init_qs_minus26 = pps.pic_init_qs_minus26 as i8;
        value.chroma_qp_index_offset = pps.chroma_qp_index_offset as i8;
        // 7.4.2.2 infers the second offset equal to the first when the PPS omits it.
        value.second_chroma_qp_index_offset = extension
            .map_or(pps.chroma_qp_index_offset as i8, |extra| {
                extra.second_chroma_qp_index_offset as i8
            });
        let scaling = matrix.map(|matrix| {
            let lists_8x8 = matrix.scaling_lists_8x8(&sps.chroma_info);
            let lists_8x8 = lists_8x8.as_ref().map_or(&[][..], |lists| lists.as_slice());
            scaling_lists(matrix.scaling_lists_4x4(&sps.chroma_info), lists_8x8)
        });
        Self { value, scaling }
    }

    /// The structure, valid while `self` lives.
    pub(super) fn as_std(&self) -> StdVideoH264PictureParameterSet {
        let mut value = self.value;
        if let Some(scaling) = &self.scaling {
            value.pScalingLists = ptr::from_ref(&**scaling);
        }
        value
    }
}

/// What the picture info of one frame states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PictureFacts {
    pub(super) sps_id: u8,
    pub(super) pps_id: u8,
    pub(super) frame_num: u16,
    pub(super) idr_pic_id: u16,
    pub(super) poc: [i32; 2],
    pub(super) idr: bool,
    pub(super) intra: bool,
    pub(super) reference: bool,
}

/// The `StdVideoDecodeH264PictureInfo` of a frame. Field flags stay 0.
pub(super) fn std_picture_info(facts: &PictureFacts) -> StdVideoDecodeH264PictureInfo {
    let mut flags: StdVideoDecodeH264PictureInfoFlags = zeroed();
    flags.set_is_intra(facts.intra.into());
    flags.set_IdrPicFlag(facts.idr.into());
    flags.set_is_reference(facts.reference.into());
    StdVideoDecodeH264PictureInfo {
        flags,
        seq_parameter_set_id: facts.sps_id,
        pic_parameter_set_id: facts.pps_id,
        reserved1: 0,
        reserved2: 0,
        frame_num: facts.frame_num,
        idr_pic_id: facts.idr_pic_id,
        PicOrderCnt: facts.poc,
    }
}

/// The `StdVideoDecodeH264ReferenceInfo` of a frame in the DPB. `FrameNum` holds
/// `LongTermFrameIdx` for a long-term reference. Both field flags stay 0 for a frame.
pub(super) fn std_reference_info(frame: &RefFrame) -> StdVideoDecodeH264ReferenceInfo {
    let mut flags: StdVideoDecodeH264ReferenceInfoFlags = zeroed();
    flags.set_used_for_long_term_reference(frame.long_term.is_some().into());
    StdVideoDecodeH264ReferenceInfo {
        flags,
        FrameNum: frame.long_term.unwrap_or(frame.frame_num) as u16,
        reserved: 0,
        PicOrderCnt: frame.poc,
    }
}
