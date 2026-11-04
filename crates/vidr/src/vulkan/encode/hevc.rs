//! HEVC parameter sets and picture structures for encode. Main or Main 10, one reference, one slice
//! segment. Every syntax element here follows ITU-T H.265.

use std::ptr;

use ash::vk;
use ash::vk::native::{
    StdVideoEncodeH265PictureInfo, StdVideoEncodeH265PictureInfoFlags,
    StdVideoEncodeH265ReferenceInfo, StdVideoEncodeH265ReferenceInfoFlags,
    StdVideoEncodeH265ReferenceListsInfo, StdVideoEncodeH265ReferenceListsInfoFlags,
    StdVideoEncodeH265SliceSegmentHeader, StdVideoEncodeH265SliceSegmentHeaderFlags,
    StdVideoH265AspectRatioIdc_STD_VIDEO_H265_ASPECT_RATIO_IDC_EXTENDED_SAR,
    StdVideoH265ChromaFormatIdc_STD_VIDEO_H265_CHROMA_FORMAT_IDC_420, StdVideoH265DecPicBufMgr,
    StdVideoH265PictureParameterSet, StdVideoH265PictureType,
    StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_IDR,
    StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_P, StdVideoH265ProfileTierLevel,
    StdVideoH265SequenceParameterSet, StdVideoH265SequenceParameterSetVui,
    StdVideoH265ShortTermRefPicSet, StdVideoH265SliceType_STD_VIDEO_H265_SLICE_TYPE_I,
    StdVideoH265SliceType_STD_VIDEO_H265_SLICE_TYPE_P, StdVideoH265VideoParameterSet,
};

use crate::decode::PixelFormat;
use crate::encode::{EncodeCodec, EncodeConfig};
use crate::error::EncodeError;
use crate::vulkan::hevc::std_common::{std_dec_pic_buf_mgr, std_profile_tier_level};
use crate::vulkan::retained::Retained;
use crate::vulkan::session::ParameterOwner;
use crate::vulkan::zeroed::{AllZero, zeroed};

use super::caps::{CodecCaps, EncodeCaps};
use super::engine::{EncodeEngine, EncodeRequest, SlotInfo};
use super::gop::PicturePlan;
use super::h264::{VIDEO_FORMAT_UNSPECIFIED, sar_parts};
use super::level::{check_driver_level, hevc_level};
use super::rate::RatePlan;

// SAFETY: integers and bitfield units only.
unsafe impl AllZero for StdVideoEncodeH265SliceSegmentHeaderFlags {}
// SAFETY: as above.
unsafe impl AllZero for StdVideoEncodeH265PictureInfoFlags {}
// SAFETY: as above.
unsafe impl AllZero for StdVideoEncodeH265ReferenceInfoFlags {}
// SAFETY: as above.
unsafe impl AllZero for StdVideoEncodeH265ReferenceListsInfoFlags {}

/// `STD_VIDEO_H265_NO_REFERENCE_PICTURE`.
const NO_REFERENCE: u8 = 0xFF;
/// `general_profile_idc` of Main.
const PROFILE_MAIN: u8 = 1;
/// `general_profile_idc` of Main 10.
const PROFILE_MAIN_10: u8 = 2;
/// The 48 constraint bits with `general_progressive_source_flag` and
/// `general_frame_only_constraint_flag` set.
const PROGRESSIVE_FRAMES: u64 = 0b1001 << 44;
/// `log2_max_pic_order_cnt_lsb_minus4`: the POC's low part holds 8 bits.
const LOG2_MAX_POC_LSB_MINUS4: u8 = 4;
/// `MinCbLog2SizeY`: an 8 by 8 smallest coding block.
const MIN_CB_LOG2: u8 = 3;
/// `MinTbLog2SizeY`: a 4 by 4 smallest transform block.
const MIN_TB_LOG2: u8 = 2;
/// `MaxNumMergeCand` at its top of 5, which `five_minus_max_num_merge_cand` 0 states.
const MAX_MERGE_CANDIDATES: u8 = 5;

/// The largest CTB the driver lists, as `CtbLog2SizeY`, or the refusal.
fn ctb_log2(caps: &EncodeCaps) -> Result<u8, EncodeError> {
    let CodecCaps::Hevc { ctb_sizes, .. } = caps.codec else {
        return Err(not_hevc());
    };
    use vk::VideoEncodeH265CtbSizeFlagsKHR as Ctb;
    [(Ctb::TYPE_64, 6), (Ctb::TYPE_32, 5), (Ctb::TYPE_16, 4)]
        .into_iter()
        .find(|(flag, _)| ctb_sizes.contains(*flag))
        .map(|(_, log2)| log2)
        .ok_or_else(|| EncodeError::VulkanRefused {
            codec: EncodeCodec::Hevc,
            what: "a coding tree block".to_owned(),
            reason: format!("the driver lists {ctb_sizes:?}"),
        })
}

/// `MaxTbLog2SizeY`: the largest transform the driver lists, at most the CTB and 32 by 32, or the
/// refusal when 4 by 4 is missing. `MinCbLog2SizeY` must stay above `MinTbLog2SizeY`.
fn max_tb_log2(caps: &EncodeCaps, ctb: u8) -> Result<u8, EncodeError> {
    let CodecCaps::Hevc {
        transform_sizes, ..
    } = caps.codec
    else {
        return Err(not_hevc());
    };
    use vk::VideoEncodeH265TransformBlockSizeFlagsKHR as Tb;
    if !transform_sizes.contains(Tb::TYPE_4) {
        return Err(EncodeError::VulkanRefused {
            codec: EncodeCodec::Hevc,
            what: "a 4 by 4 transform".to_owned(),
            reason: format!("the driver lists {transform_sizes:?}"),
        });
    }
    let largest = [
        (Tb::TYPE_32, 5),
        (Tb::TYPE_16, 4),
        (Tb::TYPE_8, 3),
        (Tb::TYPE_4, 2),
    ]
    .into_iter()
    .find(|(flag, _)| transform_sizes.contains(*flag))
    .map_or(MIN_TB_LOG2, |(_, log2)| log2);
    Ok(largest.min(ctb.min(5)))
}

fn not_hevc() -> EncodeError {
    EncodeError::VulkanCore {
        reason: "HEVC parameter sets read capabilities of another codec".to_owned(),
    }
}

/// Refuses a CTB of 16 at level 5 and up. ITU-T H.265 A.4.1 d): for level 5 and higher levels, the
/// value of `CtbSizeY` shall be equal to 32 or 64.
fn check_ctb_at_level(level_idc: u8, caps: &EncodeCaps) -> Result<(), EncodeError> {
    const LEVEL_5: u8 = 150;
    const CTB_32_LOG2: u8 = 5;
    let ctb = ctb_log2(caps)?;
    if level_idc < LEVEL_5 || ctb >= CTB_32_LOG2 {
        return Ok(());
    }
    Err(EncodeError::VulkanRefused {
        codec: EncodeCodec::Hevc,
        what: format!(
            "a coding tree block of {} at general_level_idc {level_idc}",
            1_u32 << ctb
        ),
        reason: "ITU-T H.265 A.4.1 d) needs 32 or 64 at level 5 and up, and the driver lists \
                 neither"
            .to_owned(),
    })
}

const fn sao_listed(caps: &EncodeCaps) -> bool {
    match caps.codec {
        CodecCaps::Hevc { std_syntax, .. } => std_syntax
            .contains(vk::VideoEncodeH265StdFlagsKHR::SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG_SET),
        CodecCaps::H264 { .. } | CodecCaps::Av1 { .. } => false,
    }
}

/// The profile tier level at the lowest level that holds the frame, the rate, and `top_bitrate`.
/// Main tier, progressive frames only. The driver must list that level and a CTB it allows.
pub(super) fn profile_tier_level(
    config: &EncodeConfig,
    coded: (u32, u32),
    caps: &EncodeCaps,
    top_bitrate: Option<u64>,
) -> Result<StdVideoH265ProfileTierLevel, EncodeError> {
    let profile = match config.format {
        PixelFormat::Nv12 => PROFILE_MAIN,
        PixelFormat::P010 => PROFILE_MAIN_10,
    };
    let level = hevc_level(coded, config.frame_rate_parts()?, top_bitrate)?;
    check_ctb_at_level(level, caps)?;
    let ptl = std_profile_tier_level(profile, false, level, PROGRESSIVE_FRAMES);
    check_driver_level(
        EncodeCodec::Hevc,
        level,
        ptl.general_level_idc,
        caps.max_level,
    )?;
    Ok(ptl)
}

/// One picture held for the current one, and no reorder.
fn dec_pic_buf_mgr() -> StdVideoH265DecPicBufMgr {
    let mut buffering = [0_u32; 7];
    buffering[0] = 1;
    std_dec_pic_buf_mgr(&[0; 7], &buffering, &[0; 7])
}

/// The VPS with null pointers: one layer, no timing.
pub(super) fn std_vps() -> StdVideoH265VideoParameterSet {
    let mut vps: StdVideoH265VideoParameterSet = zeroed();
    vps.flags.set_vps_temporal_id_nesting_flag(1);
    vps.flags.set_vps_sub_layer_ordering_info_present_flag(1);
    vps
}

/// The VUI of `config`: colour as NVENC writes it, and the aspect ratio.
pub(super) fn std_vui(
    config: &EncodeConfig,
) -> Result<StdVideoH265SequenceParameterSetVui, EncodeError> {
    let color = &config.color;
    let mut vui: StdVideoH265SequenceParameterSetVui = zeroed();
    vui.flags.set_video_signal_type_present_flag(1);
    vui.flags.set_video_full_range_flag(color.full_range.into());
    vui.flags.set_colour_description_present_flag(1);
    vui.video_format = VIDEO_FORMAT_UNSPECIFIED;
    vui.colour_primaries = color.colour_primaries;
    vui.transfer_characteristics = color.transfer_characteristics;
    vui.matrix_coeffs = color.matrix_coefficients;
    if let Some(location) = color.chroma_location {
        vui.flags.set_chroma_loc_info_present_flag(1);
        vui.chroma_sample_loc_type_top_field = location.to_h273();
        vui.chroma_sample_loc_type_bottom_field = location.to_h273();
    }
    if let Some((width, height)) = sar_parts(config)? {
        vui.flags.set_aspect_ratio_info_present_flag(1);
        vui.aspect_ratio_idc =
            StdVideoH265AspectRatioIdc_STD_VIDEO_H265_ASPECT_RATIO_IDC_EXTENDED_SAR;
        vui.sar_width = width;
        vui.sar_height = height;
    }
    Ok(vui)
}

/// The SPS with null pointers. A conformance window maps the coded size back to the frame's, in
/// units of two samples for 4:2:0.
pub(super) fn std_sps(
    config: &EncodeConfig,
    coded: (u32, u32),
    caps: &EncodeCaps,
) -> Result<StdVideoH265SequenceParameterSet, EncodeError> {
    let ctb = ctb_log2(caps)?;
    let max_tb = max_tb_log2(caps, ctb)?;
    let depth = config.format.bit_depth() - 8;
    let mut sps: StdVideoH265SequenceParameterSet = zeroed();
    sps.flags.set_sps_temporal_id_nesting_flag(1);
    sps.flags.set_sps_sub_layer_ordering_info_present_flag(1);
    sps.flags.set_vui_parameters_present_flag(1);
    sps.flags
        .set_sample_adaptive_offset_enabled_flag(sao_listed(caps).into());
    sps.chroma_format_idc = StdVideoH265ChromaFormatIdc_STD_VIDEO_H265_CHROMA_FORMAT_IDC_420;
    sps.pic_width_in_luma_samples = coded.0;
    sps.pic_height_in_luma_samples = coded.1;
    sps.bit_depth_luma_minus8 = depth;
    sps.bit_depth_chroma_minus8 = depth;
    sps.log2_max_pic_order_cnt_lsb_minus4 = LOG2_MAX_POC_LSB_MINUS4;
    sps.log2_min_luma_coding_block_size_minus3 = MIN_CB_LOG2 - 3;
    sps.log2_diff_max_min_luma_coding_block_size = ctb - MIN_CB_LOG2;
    sps.log2_min_luma_transform_block_size_minus2 = MIN_TB_LOG2 - 2;
    sps.log2_diff_max_min_luma_transform_block_size = max_tb - MIN_TB_LOG2;
    // 7.4.3.2.1: at most CtbLog2SizeY - MinTbLog2SizeY.
    let hierarchy = (ctb - MIN_TB_LOG2).min(3);
    sps.max_transform_hierarchy_depth_inter = hierarchy;
    sps.max_transform_hierarchy_depth_intra = hierarchy;
    let (right, bottom) = ((coded.0 - config.width) / 2, (coded.1 - config.height) / 2);
    if right > 0 || bottom > 0 {
        sps.flags.set_conformance_window_flag(1);
        sps.conf_win_right_offset = right;
        sps.conf_win_bottom_offset = bottom;
    }
    Ok(sps)
}

/// The PPS. A rate-controlled stream needs `cu_qp_delta_enabled_flag`, and a constant quantiser
/// does not.
pub(super) fn std_pps(constant_quantiser: bool) -> StdVideoH265PictureParameterSet {
    let mut pps: StdVideoH265PictureParameterSet = zeroed();
    pps.flags.set_pps_loop_filter_across_slices_enabled_flag(1);
    pps.flags
        .set_cu_qp_delta_enabled_flag((!constant_quantiser).into());
    pps
}

/// One stream's parameter sets, at stable addresses for the life of the encoder.
pub(super) struct HevcStream {
    ptl: Retained<StdVideoH265ProfileTierLevel>,
    buffering: Retained<StdVideoH265DecPicBufMgr>,
    vui: Retained<StdVideoH265SequenceParameterSetVui>,
    vps: Retained<StdVideoH265VideoParameterSet>,
    sps: Retained<StdVideoH265SequenceParameterSet>,
    pps: Retained<StdVideoH265PictureParameterSet>,
    sao: bool,
}

const fn picture_type(idr: bool) -> StdVideoH265PictureType {
    if idr {
        StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_IDR
    } else {
        StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_P
    }
}

/// `PicOrderCntVal`: an IDR picture holds 0, and each picture after it counts up.
const fn pic_order_cnt(since_idr: u32) -> i32 {
    since_idr as i32
}

/// The short-term set of a picture. A P picture keeps and uses the one before it, at a POC delta of
/// -1. An IDR picture keeps none.
fn short_term_set(idr: bool) -> StdVideoH265ShortTermRefPicSet {
    let mut set: StdVideoH265ShortTermRefPicSet = zeroed();
    if !idr {
        set.num_negative_pics = 1;
        set.used_by_curr_pic_s0_flag = 1;
        set.delta_poc_s0_minus1[0] = 0;
    }
    set
}

/// The reference lists: the one slot this picture reads in list 0, and nothing else.
fn reference_lists(reference: Option<u32>) -> StdVideoEncodeH265ReferenceListsInfo {
    let mut list0 = [NO_REFERENCE; 15];
    if let Some(slot) = reference {
        list0[0] = slot as u8;
    }
    StdVideoEncodeH265ReferenceListsInfo {
        flags: zeroed(),
        num_ref_idx_l0_active_minus1: 0,
        num_ref_idx_l1_active_minus1: 0,
        RefPicList0: list0,
        RefPicList1: [NO_REFERENCE; 15],
        list_entry_l0: [0; 15],
        list_entry_l1: [0; 15],
    }
}

/// The picture structure. Its pointers stay null for the caller to set.
fn picture_info(plan: &PicturePlan) -> StdVideoEncodeH265PictureInfo {
    let mut flags: StdVideoEncodeH265PictureInfoFlags = zeroed();
    flags.set_is_reference(1);
    flags.set_IrapPicFlag(plan.idr.into());
    flags.set_pic_output_flag(1);
    StdVideoEncodeH265PictureInfo {
        flags,
        pic_type: picture_type(plan.idr),
        sps_video_parameter_set_id: 0,
        pps_seq_parameter_set_id: 0,
        pps_pic_parameter_set_id: 0,
        short_term_ref_pic_set_idx: 0,
        PicOrderCntVal: pic_order_cnt(plan.since_idr),
        TemporalId: 0,
        reserved1: [0; 7],
        pRefLists: ptr::null(),
        pShortTermRefPicSet: ptr::null(),
        pLongTermRefPics: ptr::null(),
    }
}

/// The slice segment header. One segment covers the picture.
fn slice_segment_header(idr: bool, sao: bool) -> StdVideoEncodeH265SliceSegmentHeader {
    let mut flags: StdVideoEncodeH265SliceSegmentHeaderFlags = zeroed();
    flags.set_first_slice_segment_in_pic_flag(1);
    flags.set_slice_sao_luma_flag(sao.into());
    flags.set_slice_sao_chroma_flag(sao.into());
    flags.set_slice_loop_filter_across_slices_enabled_flag(1);
    StdVideoEncodeH265SliceSegmentHeader {
        flags,
        slice_type: if idr {
            StdVideoH265SliceType_STD_VIDEO_H265_SLICE_TYPE_I
        } else {
            StdVideoH265SliceType_STD_VIDEO_H265_SLICE_TYPE_P
        },
        slice_segment_address: 0,
        collocated_ref_idx: 0,
        MaxNumMergeCand: MAX_MERGE_CANDIDATES,
        slice_cb_qp_offset: 0,
        slice_cr_qp_offset: 0,
        slice_beta_offset_div2: 0,
        slice_tc_offset_div2: 0,
        slice_act_y_qp_offset: 0,
        slice_act_cb_qp_offset: 0,
        slice_act_cr_qp_offset: 0,
        slice_qp_delta: 0,
        reserved1: 0,
        pWeightTable: ptr::null(),
    }
}

/// The structure of the picture `since_idr` pictures after an IDR picture, for its slot.
fn reference_info(since_idr: u32) -> StdVideoEncodeH265ReferenceInfo {
    StdVideoEncodeH265ReferenceInfo {
        flags: zeroed(),
        pic_type: picture_type(since_idr == 0),
        PicOrderCntVal: pic_order_cnt(since_idr),
        TemporalId: 0,
    }
}

impl HevcStream {
    pub(super) fn new(
        config: &EncodeConfig,
        coded: (u32, u32),
        caps: &EncodeCaps,
        plan: RatePlan,
    ) -> Result<Self, EncodeError> {
        let constant_quantiser = matches!(plan, RatePlan::Constant { .. });
        let ptl = Retained::new(profile_tier_level(config, coded, caps, plan.top_bitrate())?);
        let buffering = Retained::new(dec_pic_buf_mgr());
        let vui = Retained::new(std_vui(config)?);
        let mut vps = std_vps();
        vps.pDecPicBufMgr = buffering.as_ptr();
        vps.pProfileTierLevel = ptl.as_ptr();
        let mut sps = std_sps(config, coded, caps)?;
        sps.pProfileTierLevel = ptl.as_ptr();
        sps.pDecPicBufMgr = buffering.as_ptr();
        sps.pSequenceParameterSetVui = vui.as_ptr();
        Ok(Self {
            ptl,
            buffering,
            vui,
            vps: Retained::new(vps),
            sps: Retained::new(sps),
            pps: Retained::new(std_pps(constant_quantiser)),
            sao: sao_listed(caps),
        })
    }

    /// Creates the engine's session parameters and returns the driver's VPS, SPS, and PPS bytes.
    pub(super) fn parameters(&self, engine: &mut EncodeEngine) -> Result<Vec<u8>, EncodeError> {
        let add = vk::VideoEncodeH265SessionParametersAddInfoKHR {
            std_vps_count: 1,
            p_std_vp_ss: self.vps.as_ptr(),
            std_sps_count: 1,
            p_std_sp_ss: self.sps.as_ptr(),
            std_pps_count: 1,
            p_std_pp_ss: self.pps.as_ptr(),
            ..Default::default()
        };
        let create = vk::VideoEncodeH265SessionParametersCreateInfoKHR {
            max_std_vps_count: 1,
            max_std_sps_count: 1,
            max_std_pps_count: 1,
            p_parameters_add_info: &add,
            ..Default::default()
        };
        let points_to: ParameterOwner = Box::new((
            self.ptl.clone(),
            self.buffering.clone(),
            self.vui.clone(),
            self.vps.clone(),
            self.sps.clone(),
            self.pps.clone(),
        ));
        // SAFETY: `create` and `add` live through the call, and `points_to` owns what they point to.
        unsafe { engine.create_parameters(ptr::from_ref(&create).cast(), points_to) }?;
        let get = vk::VideoEncodeH265SessionParametersGetInfoKHR {
            write_std_vps: vk::TRUE,
            write_std_sps: vk::TRUE,
            write_std_pps: vk::TRUE,
            std_vps_id: 0,
            std_sps_id: 0,
            std_pps_id: 0,
            ..Default::default()
        };
        // SAFETY: `get` lives through the call.
        unsafe { engine.encoded_parameters(ptr::from_ref(&get).cast()) }
    }

    /// Encodes one picture from the engine's input image.
    pub(super) fn encode(
        &self,
        engine: &mut EncodeEngine,
        plan: &PicturePlan,
        pts: i64,
    ) -> Result<Vec<u8>, EncodeError> {
        let lists = reference_lists(plan.reference_slot);
        let set = short_term_set(plan.idr);
        let mut picture = picture_info(plan);
        picture.pRefLists = &lists;
        picture.pShortTermRefPicSet = &set;
        let header = slice_segment_header(plan.idr, self.sao);
        let segment = vk::VideoEncodeH265NaluSliceSegmentInfoKHR {
            constant_qp: engine.rate().plan().quantiser(plan.idr),
            p_std_slice_segment_header: &header,
            ..Default::default()
        };
        let info = vk::VideoEncodeH265PictureInfoKHR {
            nalu_slice_segment_entry_count: 1,
            p_nalu_slice_segment_entries: &segment,
            p_std_picture_info: &picture,
            ..Default::default()
        };
        let setup_std = reference_info(plan.since_idr);
        let setup = vk::VideoEncodeH265DpbSlotInfoKHR {
            p_std_reference_info: &setup_std,
            ..Default::default()
        };
        let reference_std = reference_info(plan.since_idr.wrapping_sub(1));
        let reference = vk::VideoEncodeH265DpbSlotInfoKHR {
            p_std_reference_info: &reference_std,
            ..Default::default()
        };
        let request = EncodeRequest {
            picture_info: ptr::from_ref(&info).cast(),
            setup: SlotInfo {
                slot: plan.setup_slot,
                codec_info: ptr::from_ref(&setup).cast(),
            },
            reference: plan.reference_slot.map(|slot| SlotInfo {
                slot,
                codec_info: ptr::from_ref(&reference).cast(),
            }),
            pts,
        };
        // SAFETY: each chain above is whole and lives through the call, and the encoder loads the
        // frame first.
        unsafe { engine.encode(&request) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{ChromaLocation, ColorFacts};
    use crate::geometry::SampleAspectRatio;
    use crate::time::Rational;
    use crate::vulkan::encode::caps::test_caps::{h264, hevc};
    use crate::vulkan::encode::gop::GopState;

    fn config(width: u32, height: u32, format: PixelFormat) -> EncodeConfig {
        let rate = Rational::new(30, 1).unwrap_or(Rational::ONE);
        let base = Rational::new(1, 30).unwrap_or(Rational::ONE);
        EncodeConfig::new(EncodeCodec::Hevc, width, height, format, rate, base)
    }

    #[test]
    fn the_sps_states_the_size_depth_and_window_to_the_frame() {
        let source = config(640, 360, PixelFormat::P010);
        let sps = std_sps(&source, (640, 384), &hevc()).expect("the SPS");
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.pic_width_in_luma_samples, 640);
        assert_eq!(sps.pic_height_in_luma_samples, 384);
        assert_eq!(sps.bit_depth_luma_minus8, 2);
        assert_eq!(sps.bit_depth_chroma_minus8, 2);
        assert_eq!(sps.flags.conformance_window_flag(), 1);
        assert_eq!(sps.conf_win_right_offset, 0);
        assert_eq!(sps.conf_win_bottom_offset, 12);
        assert_eq!(sps.flags.sample_adaptive_offset_enabled_flag(), 1);
        assert_eq!(sps.num_short_term_ref_pic_sets, 0);
    }

    #[test]
    fn block_sizes_follow_what_the_driver_lists() {
        let sps =
            std_sps(&config(640, 384, PixelFormat::Nv12), (640, 384), &hevc()).expect("the SPS");
        assert_eq!(sps.flags.conformance_window_flag(), 0);
        assert_eq!(sps.log2_min_luma_coding_block_size_minus3, 0);
        assert_eq!(sps.log2_diff_max_min_luma_coding_block_size, 2, "CTB 32");
        assert_eq!(sps.log2_min_luma_transform_block_size_minus2, 0);
        assert_eq!(sps.log2_diff_max_min_luma_transform_block_size, 3, "TB 32");
        assert_eq!(sps.max_transform_hierarchy_depth_intra, 3);

        let large = EncodeCaps {
            codec: CodecCaps::Hevc {
                std_syntax: vk::VideoEncodeH265StdFlagsKHR::empty(),
                ctb_sizes: vk::VideoEncodeH265CtbSizeFlagsKHR::TYPE_16
                    | vk::VideoEncodeH265CtbSizeFlagsKHR::TYPE_64,
                transform_sizes: vk::VideoEncodeH265TransformBlockSizeFlagsKHR::TYPE_4
                    | vk::VideoEncodeH265TransformBlockSizeFlagsKHR::TYPE_16,
            },
            ..hevc()
        };
        let sps =
            std_sps(&config(640, 384, PixelFormat::Nv12), (640, 384), &large).expect("the SPS");
        assert_eq!(sps.log2_diff_max_min_luma_coding_block_size, 3, "CTB 64");
        assert_eq!(sps.log2_diff_max_min_luma_transform_block_size, 2, "TB 16");
        assert_eq!(sps.flags.sample_adaptive_offset_enabled_flag(), 0);
    }

    #[test]
    fn a_driver_without_a_usable_block_size_is_refused_by_name() {
        let no_ctb = EncodeCaps {
            codec: CodecCaps::Hevc {
                std_syntax: vk::VideoEncodeH265StdFlagsKHR::empty(),
                ctb_sizes: vk::VideoEncodeH265CtbSizeFlagsKHR::empty(),
                transform_sizes: vk::VideoEncodeH265TransformBlockSizeFlagsKHR::TYPE_4,
            },
            ..hevc()
        };
        let source = config(640, 384, PixelFormat::Nv12);
        let text = std_sps(&source, (640, 384), &no_ctb)
            .expect_err("refused")
            .to_string();
        assert!(text.contains("coding tree block"), "{text}");
        let no_small = EncodeCaps {
            codec: CodecCaps::Hevc {
                std_syntax: vk::VideoEncodeH265StdFlagsKHR::empty(),
                ctb_sizes: vk::VideoEncodeH265CtbSizeFlagsKHR::TYPE_32,
                transform_sizes: vk::VideoEncodeH265TransformBlockSizeFlagsKHR::TYPE_8,
            },
            ..hevc()
        };
        let text = std_sps(&source, (640, 384), &no_small)
            .expect_err("refused")
            .to_string();
        assert!(text.contains("4 by 4"), "{text}");
        assert!(std_sps(&source, (640, 384), &h264()).is_err());
    }

    #[test]
    fn the_profile_follows_the_depth_and_a_level_past_the_driver_is_refused() {
        let main = profile_tier_level(
            &config(640, 360, PixelFormat::Nv12),
            (640, 384),
            &hevc(),
            None,
        )
        .expect("the PTL");
        assert_eq!(main.general_profile_idc, 1);
        assert_eq!(main.general_level_idc, 2, "level 2.1");
        assert_eq!(main.flags.general_tier_flag(), 0);
        assert_eq!(main.flags.general_progressive_source_flag(), 1);
        assert_eq!(main.flags.general_frame_only_constraint_flag(), 1);
        let main10 = profile_tier_level(
            &config(640, 360, PixelFormat::P010),
            (640, 384),
            &hevc(),
            None,
        )
        .expect("the PTL");
        assert_eq!(main10.general_profile_idc, 2);
        let capped = EncodeCaps {
            max_level: 1,
            ..hevc()
        };
        let text = profile_tier_level(
            &config(640, 360, PixelFormat::Nv12),
            (640, 384),
            &capped,
            None,
        )
        .expect_err("refused")
        .to_string();
        assert!(text.contains("level_idc 63"), "{text}");
    }

    #[test]
    fn the_level_follows_the_target_bitrate() {
        let source = config(1920, 1088, PixelFormat::Nv12);
        let plain = profile_tier_level(&source, (1920, 1088), &hevc(), None).expect("the PTL");
        assert_eq!(plain.general_level_idc, 5, "level 4");
        let high =
            profile_tier_level(&source, (1920, 1088), &hevc(), Some(15_000_000)).expect("the PTL");
        assert_eq!(high.general_level_idc, 6, "level 4.1");
    }

    /// A.4.1 d). 3840x2176 at 30 needs level 5, where a CTB of 16 is out. 1920x1088 at 30 needs
    /// level 4, where it stays in.
    #[test]
    fn a_ctb_of_16_alone_is_refused_at_level_5_and_up() {
        let small_ctb = EncodeCaps {
            codec: CodecCaps::Hevc {
                std_syntax: vk::VideoEncodeH265StdFlagsKHR::empty(),
                ctb_sizes: vk::VideoEncodeH265CtbSizeFlagsKHR::TYPE_16,
                transform_sizes: vk::VideoEncodeH265TransformBlockSizeFlagsKHR::TYPE_4
                    | vk::VideoEncodeH265TransformBlockSizeFlagsKHR::TYPE_16,
            },
            ..hevc()
        };
        let four_k = config(3840, 2176, PixelFormat::Nv12);
        let text = profile_tier_level(&four_k, (3840, 2176), &small_ctb, None)
            .expect_err("refused")
            .to_string();
        assert!(text.contains("A.4.1 d)"), "{text}");
        assert!(text.contains("general_level_idc 150"), "{text}");
        let full_hd = config(1920, 1088, PixelFormat::Nv12);
        assert!(profile_tier_level(&full_hd, (1920, 1088), &small_ctb, None).is_ok());
        assert!(profile_tier_level(&four_k, (3840, 2176), &hevc(), None).is_ok());
    }

    #[test]
    fn the_vui_carries_each_colour_tag_and_the_aspect_ratio() {
        let mut source = config(720, 576, PixelFormat::Nv12);
        source.color = ColorFacts {
            matrix_coefficients: 1,
            transfer_characteristics: 14,
            colour_primaries: 5,
            full_range: false,
            chroma_location: Some(ChromaLocation::Left),
        };
        source.sample_aspect_ratio = SampleAspectRatio::new(16, 15);
        let vui = std_vui(&source).expect("the VUI");
        assert_eq!(vui.flags.video_signal_type_present_flag(), 1);
        assert_eq!(vui.flags.video_full_range_flag(), 0);
        assert_eq!(vui.flags.colour_description_present_flag(), 1);
        assert_eq!(vui.colour_primaries, 5);
        assert_eq!(vui.transfer_characteristics, 14);
        assert_eq!(vui.matrix_coeffs, 1);
        assert_eq!(vui.chroma_sample_loc_type_top_field, 0);
        assert_eq!(vui.flags.chroma_loc_info_present_flag(), 1);
        assert_eq!(vui.aspect_ratio_idc, 255);
        assert_eq!((vui.sar_width, vui.sar_height), (16, 15));
    }

    /// ITU-T H.265 7.4.8: with `inter_ref_pic_set_prediction_flag` 0, DeltaPocS0[0] is
    /// -(delta_poc_s0_minus1[0] + 1), and `used_by_curr_pic_s0_flag` holds one bit per entry.
    #[test]
    fn a_p_picture_keeps_and_uses_the_picture_before_it() {
        let set = short_term_set(false);
        assert_eq!(set.num_negative_pics, 1);
        assert_eq!(set.num_positive_pics, 0);
        assert_eq!(set.delta_poc_s0_minus1[0], 0);
        assert_eq!(set.used_by_curr_pic_s0_flag, 1);
        assert_eq!(set.flags.inter_ref_pic_set_prediction_flag(), 0);
        let idr = short_term_set(true);
        assert_eq!(idr.num_negative_pics, 0);
        assert_eq!(idr.used_by_curr_pic_s0_flag, 0);
    }

    #[test]
    fn an_idr_picture_and_a_p_picture_state_their_types_and_lists() {
        let mut gop = GopState::new(4);
        let idr = gop.next();
        let picture = picture_info(&idr);
        assert_eq!(picture.flags.IrapPicFlag(), 1);
        assert_eq!(picture.flags.is_reference(), 1);
        assert_eq!(picture.flags.pic_output_flag(), 1);
        assert_eq!(picture.pic_type, 3);
        assert_eq!(picture.PicOrderCntVal, 0);
        assert_eq!(
            reference_lists(idr.reference_slot).RefPicList0[0],
            NO_REFERENCE
        );
        assert_eq!(slice_segment_header(true, true).slice_type, 2);

        let p = gop.next();
        let picture = picture_info(&p);
        assert_eq!(picture.flags.IrapPicFlag(), 0);
        assert_eq!(picture.pic_type, 0);
        assert_eq!(picture.PicOrderCntVal, 1);
        let lists = reference_lists(p.reference_slot);
        assert_eq!(lists.RefPicList0[0], 0);
        assert_eq!(lists.RefPicList0[1], NO_REFERENCE);
        assert_eq!(lists.RefPicList1[0], NO_REFERENCE);
        let header = slice_segment_header(false, true);
        assert_eq!(header.slice_type, 1);
        assert_eq!(header.flags.first_slice_segment_in_pic_flag(), 1);
        assert_eq!(header.flags.slice_sao_luma_flag(), 1);
        assert_eq!(header.MaxNumMergeCand, 5);
    }

    #[test]
    fn a_slot_states_the_picture_it_holds() {
        let first = reference_info(0);
        assert_eq!((first.pic_type, first.PicOrderCntVal), (3, 0));
        let later = reference_info(9);
        assert_eq!((later.pic_type, later.PicOrderCntVal), (0, 9));
    }

    #[test]
    fn the_pps_asks_for_quantiser_deltas_only_under_rate_control() {
        assert_eq!(std_pps(true).flags.cu_qp_delta_enabled_flag(), 0);
        assert_eq!(std_pps(false).flags.cu_qp_delta_enabled_flag(), 1);
        assert_eq!(
            std_pps(true)
                .flags
                .pps_loop_filter_across_slices_enabled_flag(),
            1
        );
        assert_eq!(std_vps().flags.vps_temporal_id_nesting_flag(), 1);
    }
}
