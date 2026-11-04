//! H.264 parameter sets and picture structures for encode. High profile, CABAC where the driver
//! lists it, POC type 2, one reference.

use std::ptr;

use ash::vk;
use ash::vk::native::{
    StdVideoEncodeH264PictureInfo, StdVideoEncodeH264PictureInfoFlags,
    StdVideoEncodeH264ReferenceInfo, StdVideoEncodeH264ReferenceInfoFlags,
    StdVideoEncodeH264ReferenceListsInfo, StdVideoEncodeH264ReferenceListsInfoFlags,
    StdVideoEncodeH264SliceHeader, StdVideoEncodeH264SliceHeaderFlags,
    StdVideoH264AspectRatioIdc_STD_VIDEO_H264_ASPECT_RATIO_IDC_EXTENDED_SAR,
    StdVideoH264CabacInitIdc_STD_VIDEO_H264_CABAC_INIT_IDC_0,
    StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_420,
    StdVideoH264DisableDeblockingFilterIdc_STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_DISABLED,
    StdVideoH264PictureParameterSet, StdVideoH264PictureType,
    StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_IDR,
    StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P,
    StdVideoH264PocType_STD_VIDEO_H264_POC_TYPE_2,
    StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH, StdVideoH264SequenceParameterSet,
    StdVideoH264SequenceParameterSetVui, StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_I,
    StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_P,
    StdVideoH264WeightedBipredIdc_STD_VIDEO_H264_WEIGHTED_BIPRED_IDC_DEFAULT,
};

use crate::encode::EncodeConfig;
use crate::error::EncodeError;
use crate::geometry::SampleAspectRatio;
use crate::vulkan::h264::std_video::std_level;
use crate::vulkan::retained::Retained;
use crate::vulkan::zeroed::{AllZero, zeroed};

use super::caps::{CodecCaps, EncodeCaps};
use super::engine::{EncodeEngine, EncodeRequest, SlotInfo};
use super::gop::PicturePlan;
use super::level::h264_level;

// SAFETY: integers and bitfield units only.
unsafe impl AllZero for StdVideoEncodeH264SliceHeaderFlags {}
// SAFETY: as above.
unsafe impl AllZero for StdVideoEncodeH264PictureInfoFlags {}
// SAFETY: as above.
unsafe impl AllZero for StdVideoEncodeH264ReferenceInfoFlags {}
// SAFETY: as above.
unsafe impl AllZero for StdVideoEncodeH264ReferenceListsInfoFlags {}

/// `STD_VIDEO_H264_NO_REFERENCE_PICTURE`.
const NO_REFERENCE: u8 = 0xFF;
/// `log2_max_frame_num_minus4`: `frame_num` runs modulo 256.
const LOG2_MAX_FRAME_NUM_MINUS4: u8 = 4;
/// `MaxFrameNum`.
const MAX_FRAME_NUM: u32 = 1 << (LOG2_MAX_FRAME_NUM_MINUS4 + 4);
/// `video_format` 5: unspecified, in H.264 and HEVC.
pub(super) const VIDEO_FORMAT_UNSPECIFIED: u8 = 5;

/// The sample aspect ratio as `(sar_width, sar_height)`, or `None` for square samples.
///
/// # Errors
///
/// [`EncodeError::InvalidConfig`] for a part past 16 bits.
pub(super) fn sar_parts(config: &EncodeConfig) -> Result<Option<(u16, u16)>, EncodeError> {
    let sar = config.sample_aspect_ratio;
    if sar == SampleAspectRatio::SQUARE {
        return Ok(None);
    }
    let ratio = sar.ratio();
    let parts = u16::try_from(ratio.num())
        .ok()
        .zip(u16::try_from(ratio.den()).ok());
    parts.map(Some).ok_or_else(|| EncodeError::InvalidConfig {
        field: "sample_aspect_ratio",
        value: sar.to_string(),
        reason: "each part must fit 16 bits",
    })
}

/// The VUI of `config`: colour as NVENC writes it, the aspect ratio, and no reorder.
pub(super) fn std_vui(
    config: &EncodeConfig,
) -> Result<StdVideoH264SequenceParameterSetVui, EncodeError> {
    let color = &config.color;
    let mut vui: StdVideoH264SequenceParameterSetVui = zeroed();
    vui.flags.set_video_signal_type_present_flag(1);
    vui.flags.set_video_full_range_flag(color.full_range.into());
    vui.flags.set_color_description_present_flag(1);
    vui.flags.set_bitstream_restriction_flag(1);
    vui.video_format = VIDEO_FORMAT_UNSPECIFIED;
    vui.colour_primaries = color.colour_primaries;
    vui.transfer_characteristics = color.transfer_characteristics;
    vui.matrix_coefficients = color.matrix_coefficients;
    vui.max_num_reorder_frames = 0;
    vui.max_dec_frame_buffering = 1;
    if let Some(location) = color.chroma_location {
        vui.flags.set_chroma_loc_info_present_flag(1);
        vui.chroma_sample_loc_type_top_field = location.to_h273();
        vui.chroma_sample_loc_type_bottom_field = location.to_h273();
    }
    if let Some((width, height)) = sar_parts(config)? {
        vui.flags.set_aspect_ratio_info_present_flag(1);
        vui.aspect_ratio_idc =
            StdVideoH264AspectRatioIdc_STD_VIDEO_H264_ASPECT_RATIO_IDC_EXTENDED_SAR;
        vui.sar_width = width;
        vui.sar_height = height;
    }
    Ok(vui)
}

/// The SPS with a null VUI pointer: High, 4:2:0, 8 bits, frames only, POC type 2, one reference.
/// A crop maps the coded size back to the frame's.
pub(super) fn std_sps(
    config: &EncodeConfig,
    coded: (u32, u32),
    caps: &EncodeCaps,
) -> Result<StdVideoH264SequenceParameterSet, EncodeError> {
    let rate = config.frame_rate_parts()?;
    let mut sps: StdVideoH264SequenceParameterSet = zeroed();
    sps.flags.set_direct_8x8_inference_flag(1);
    sps.flags.set_frame_mbs_only_flag(1);
    sps.flags.set_vui_parameters_present_flag(1);
    sps.profile_idc = StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH;
    let level = std_level(h264_level(coded, rate)).unwrap_or(caps.max_level);
    sps.level_idc = level.min(caps.max_level);
    sps.chroma_format_idc = StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_420;
    sps.log2_max_frame_num_minus4 = LOG2_MAX_FRAME_NUM_MINUS4;
    sps.pic_order_cnt_type = StdVideoH264PocType_STD_VIDEO_H264_POC_TYPE_2;
    sps.max_num_ref_frames = 1;
    sps.pic_width_in_mbs_minus1 = coded.0.div_ceil(16) - 1;
    sps.pic_height_in_map_units_minus1 = coded.1.div_ceil(16) - 1;
    let (crop_right, crop_bottom) = ((coded.0 - config.width) / 2, (coded.1 - config.height) / 2);
    if crop_right > 0 || crop_bottom > 0 {
        sps.flags.set_frame_cropping_flag(1);
        sps.frame_crop_right_offset = crop_right;
        sps.frame_crop_bottom_offset = crop_bottom;
    }
    Ok(sps)
}

/// The PPS. `transform_8x8_mode_flag` and CABAC follow what the driver lists. One active reference
/// is the default, so a slice needs no override.
pub(super) fn std_pps(caps: &EncodeCaps) -> StdVideoH264PictureParameterSet {
    let mut pps: StdVideoH264PictureParameterSet = zeroed();
    if let CodecCaps::H264 { std_syntax } = caps.codec {
        pps.flags.set_entropy_coding_mode_flag(
            std_syntax
                .contains(vk::VideoEncodeH264StdFlagsKHR::ENTROPY_CODING_MODE_FLAG_SET)
                .into(),
        );
        pps.flags.set_transform_8x8_mode_flag(
            std_syntax
                .contains(vk::VideoEncodeH264StdFlagsKHR::TRANSFORM_8X8_MODE_FLAG_SET)
                .into(),
        );
    }
    pps.weighted_bipred_idc =
        StdVideoH264WeightedBipredIdc_STD_VIDEO_H264_WEIGHTED_BIPRED_IDC_DEFAULT;
    pps
}

const fn picture_type(idr: bool) -> StdVideoH264PictureType {
    if idr {
        StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_IDR
    } else {
        StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P
    }
}

/// `frame_num` of the picture `since_idr` pictures after an IDR picture. Each picture is a
/// reference, so it counts up by one.
const fn frame_num(since_idr: u32) -> u32 {
    since_idr % MAX_FRAME_NUM
}

/// `PicOrderCnt` under POC type 2, for reference pictures: twice the pictures since the IDR picture.
const fn pic_order_cnt(since_idr: u32) -> i32 {
    since_idr.wrapping_mul(2) as i32
}

/// The structure of the picture `since_idr` pictures after an IDR picture, for its slot.
fn reference_info(since_idr: u32) -> StdVideoEncodeH264ReferenceInfo {
    StdVideoEncodeH264ReferenceInfo {
        flags: zeroed(),
        primary_pic_type: picture_type(since_idr == 0),
        FrameNum: frame_num(since_idr),
        PicOrderCnt: pic_order_cnt(since_idr),
        long_term_pic_num: 0,
        long_term_frame_idx: 0,
        temporal_id: 0,
    }
}

/// The reference lists: the one slot this picture reads in list 0, and nothing else.
fn reference_lists(reference: Option<u32>) -> StdVideoEncodeH264ReferenceListsInfo {
    let mut list0 = [NO_REFERENCE; 32];
    if let Some(slot) = reference {
        list0[0] = slot as u8;
    }
    StdVideoEncodeH264ReferenceListsInfo {
        flags: zeroed(),
        num_ref_idx_l0_active_minus1: 0,
        num_ref_idx_l1_active_minus1: 0,
        RefPicList0: list0,
        RefPicList1: [NO_REFERENCE; 32],
        refList0ModOpCount: 0,
        refList1ModOpCount: 0,
        refPicMarkingOpCount: 0,
        reserved1: [0; 7],
        pRefList0ModOperations: ptr::null(),
        pRefList1ModOperations: ptr::null(),
        pRefPicMarkingOperations: ptr::null(),
    }
}

/// The picture structure. Its `pRefLists` stays null for the caller to set.
fn picture_info(plan: &PicturePlan) -> StdVideoEncodeH264PictureInfo {
    let mut flags: StdVideoEncodeH264PictureInfoFlags = zeroed();
    flags.set_IdrPicFlag(plan.idr.into());
    flags.set_is_reference(1);
    StdVideoEncodeH264PictureInfo {
        flags,
        seq_parameter_set_id: 0,
        pic_parameter_set_id: 0,
        // Consecutive IDR pictures need different values, and the range is 16 bits.
        idr_pic_id: (plan.idr_count % 2) as u16,
        primary_pic_type: picture_type(plan.idr),
        frame_num: frame_num(plan.since_idr),
        PicOrderCnt: pic_order_cnt(plan.since_idr),
        temporal_id: 0,
        reserved1: [0; 3],
        pRefLists: ptr::null(),
    }
}

/// The slice header. One slice covers the picture.
fn slice_header(idr: bool) -> StdVideoEncodeH264SliceHeader {
    StdVideoEncodeH264SliceHeader {
        flags: zeroed(),
        first_mb_in_slice: 0,
        slice_type: if idr {
            StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_I
        } else {
            StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_P
        },
        slice_alpha_c0_offset_div2: 0,
        slice_beta_offset_div2: 0,
        slice_qp_delta: 0,
        reserved1: 0,
        cabac_init_idc: StdVideoH264CabacInitIdc_STD_VIDEO_H264_CABAC_INIT_IDC_0,
        // The name states the filter's own disable flag at 0: the filter runs.
        disable_deblocking_filter_idc:
            StdVideoH264DisableDeblockingFilterIdc_STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_DISABLED,
        pWeightTable: ptr::null(),
    }
}

/// One stream's parameter sets, at stable addresses for the life of the encoder.
pub(super) struct H264Stream {
    _vui: Retained<StdVideoH264SequenceParameterSetVui>,
    sps: Retained<StdVideoH264SequenceParameterSet>,
    pps: Retained<StdVideoH264PictureParameterSet>,
}

impl H264Stream {
    pub(super) fn new(
        config: &EncodeConfig,
        coded: (u32, u32),
        caps: &EncodeCaps,
    ) -> Result<Self, EncodeError> {
        let vui = Retained::new(std_vui(config)?);
        let mut sps = std_sps(config, coded, caps)?;
        sps.pSequenceParameterSetVui = vui.as_ptr();
        Ok(Self {
            _vui: vui,
            sps: Retained::new(sps),
            pps: Retained::new(std_pps(caps)),
        })
    }

    /// Creates the engine's session parameters and returns the driver's SPS and PPS bytes.
    pub(super) fn parameters(&self, engine: &mut EncodeEngine) -> Result<Vec<u8>, EncodeError> {
        let add = vk::VideoEncodeH264SessionParametersAddInfoKHR {
            std_sps_count: 1,
            p_std_sp_ss: self.sps.as_ptr(),
            std_pps_count: 1,
            p_std_pp_ss: self.pps.as_ptr(),
            ..Default::default()
        };
        let create = vk::VideoEncodeH264SessionParametersCreateInfoKHR {
            max_std_sps_count: 1,
            max_std_pps_count: 1,
            p_parameters_add_info: &add,
            ..Default::default()
        };
        // SAFETY: `create` and `add` live through the call, and `self` holds what they point to.
        unsafe { engine.create_parameters(ptr::from_ref(&create).cast()) }?;
        let get = vk::VideoEncodeH264SessionParametersGetInfoKHR {
            write_std_sps: vk::TRUE,
            write_std_pps: vk::TRUE,
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
        let mut picture = picture_info(plan);
        picture.pRefLists = &lists;
        let header = slice_header(plan.idr);
        let slice = vk::VideoEncodeH264NaluSliceInfoKHR {
            constant_qp: engine.rate().plan().quantiser(plan.idr),
            p_std_slice_header: &header,
            ..Default::default()
        };
        let info = vk::VideoEncodeH264PictureInfoKHR {
            nalu_slice_entry_count: 1,
            p_nalu_slice_entries: &slice,
            p_std_picture_info: &picture,
            generate_prefix_nalu: vk::FALSE,
            ..Default::default()
        };
        let setup_std = reference_info(plan.since_idr);
        let setup = vk::VideoEncodeH264DpbSlotInfoKHR {
            p_std_reference_info: &setup_std,
            ..Default::default()
        };
        let reference_std = reference_info(plan.since_idr.wrapping_sub(1));
        let reference = vk::VideoEncodeH264DpbSlotInfoKHR {
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
    use crate::decode::{ChromaLocation, ColorFacts, PixelFormat};
    use crate::encode::EncodeCodec;
    use crate::time::Rational;
    use crate::vulkan::encode::caps::test_caps::h264;
    use crate::vulkan::encode::gop::GopState;

    fn config(width: u32, height: u32) -> EncodeConfig {
        let rate = Rational::new(30, 1).unwrap_or(Rational::ONE);
        let base = Rational::new(1, 30).unwrap_or(Rational::ONE);
        EncodeConfig::new(
            EncodeCodec::H264,
            width,
            height,
            PixelFormat::Nv12,
            rate,
            base,
        )
    }

    #[test]
    fn the_sps_states_high_poc_2_and_the_crop_to_the_frame() {
        let sps = std_sps(&config(640, 360), (640, 368), &h264()).expect("the SPS");
        assert_eq!(sps.profile_idc, 100);
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.pic_order_cnt_type, 2);
        assert_eq!(sps.max_num_ref_frames, 1);
        assert_eq!(sps.pic_width_in_mbs_minus1, 39);
        assert_eq!(sps.pic_height_in_map_units_minus1, 22);
        assert_eq!(sps.flags.frame_mbs_only_flag(), 1);
        assert_eq!(sps.flags.frame_cropping_flag(), 1);
        assert_eq!(sps.frame_crop_right_offset, 0);
        assert_eq!(sps.frame_crop_bottom_offset, 4);
        assert_eq!(sps.level_idc, 7, "level 3.0");
    }

    #[test]
    fn an_aligned_size_states_no_crop_and_a_level_past_the_driver_takes_its_top() {
        let sps = std_sps(&config(640, 368), (640, 368), &h264()).expect("the SPS");
        assert_eq!(sps.flags.frame_cropping_flag(), 0);
        let capped = EncodeCaps {
            max_level: 3,
            ..h264()
        };
        let sps = std_sps(&config(1920, 1088), (1920, 1088), &capped).expect("the SPS");
        assert_eq!(sps.level_idc, 3);
    }

    #[test]
    fn the_vui_carries_each_colour_tag_and_no_reorder() {
        let mut source = config(640, 360);
        source.color = ColorFacts {
            matrix_coefficients: 9,
            transfer_characteristics: 16,
            colour_primaries: 9,
            full_range: true,
            chroma_location: Some(ChromaLocation::TopLeft),
        };
        let vui = std_vui(&source).expect("the VUI");
        assert_eq!(vui.flags.video_signal_type_present_flag(), 1);
        assert_eq!(vui.flags.video_full_range_flag(), 1);
        assert_eq!(vui.flags.color_description_present_flag(), 1);
        assert_eq!(vui.colour_primaries, 9);
        assert_eq!(vui.transfer_characteristics, 16);
        assert_eq!(vui.matrix_coefficients, 9);
        assert_eq!(vui.flags.chroma_loc_info_present_flag(), 1);
        assert_eq!(vui.chroma_sample_loc_type_top_field, 2);
        assert_eq!(vui.chroma_sample_loc_type_bottom_field, 2);
        assert_eq!(vui.flags.bitstream_restriction_flag(), 1);
        assert_eq!(vui.max_num_reorder_frames, 0);
        assert_eq!(vui.flags.aspect_ratio_info_present_flag(), 0);
    }

    #[test]
    fn a_sample_aspect_ratio_goes_in_as_extended_and_refuses_past_16_bits() {
        let mut source = config(720, 576);
        source.sample_aspect_ratio = SampleAspectRatio::new(16, 15);
        let vui = std_vui(&source).expect("the VUI");
        assert_eq!(vui.flags.aspect_ratio_info_present_flag(), 1);
        assert_eq!(vui.aspect_ratio_idc, 255);
        assert_eq!((vui.sar_width, vui.sar_height), (16, 15));
        source.sample_aspect_ratio = SampleAspectRatio::new(70_000, 1);
        assert!(matches!(
            std_vui(&source),
            Err(EncodeError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn the_pps_follows_what_the_driver_lists() {
        let pps = std_pps(&h264());
        assert_eq!(pps.flags.entropy_coding_mode_flag(), 1);
        assert_eq!(pps.flags.transform_8x8_mode_flag(), 1);
        let plain = EncodeCaps {
            codec: CodecCaps::H264 {
                std_syntax: vk::VideoEncodeH264StdFlagsKHR::empty(),
            },
            ..h264()
        };
        assert_eq!(std_pps(&plain).flags.entropy_coding_mode_flag(), 0);
        assert_eq!(pps.num_ref_idx_l0_default_active_minus1, 0);
    }

    #[test]
    fn frame_num_counts_each_reference_and_wraps_and_poc_counts_in_twos() {
        assert_eq!(frame_num(0), 0);
        assert_eq!(frame_num(255), 255);
        assert_eq!(frame_num(256), 0);
        assert_eq!(pic_order_cnt(0), 0);
        assert_eq!(pic_order_cnt(1), 2);
        assert_eq!(pic_order_cnt(300), 600);
    }

    #[test]
    fn an_idr_picture_and_a_p_picture_state_their_types_and_lists() {
        let mut gop = GopState::new(3);
        let idr = gop.next();
        let picture = picture_info(&idr);
        assert_eq!(picture.flags.IdrPicFlag(), 1);
        assert_eq!(picture.flags.is_reference(), 1);
        assert_eq!(picture.primary_pic_type, 5);
        assert_eq!(picture.frame_num, 0);
        assert_eq!(picture.PicOrderCnt, 0);
        let lists = reference_lists(idr.reference_slot);
        assert_eq!(lists.RefPicList0[0], NO_REFERENCE);
        assert_eq!(slice_header(true).slice_type, 2);

        let p = gop.next();
        let picture = picture_info(&p);
        assert_eq!(picture.flags.IdrPicFlag(), 0);
        assert_eq!(picture.primary_pic_type, 0);
        assert_eq!(picture.frame_num, 1);
        assert_eq!(picture.PicOrderCnt, 2);
        let lists = reference_lists(p.reference_slot);
        assert_eq!(lists.RefPicList0[0], 0);
        assert_eq!(lists.RefPicList0[1], NO_REFERENCE);
        assert_eq!(lists.RefPicList1[0], NO_REFERENCE);
        assert_eq!(lists.num_ref_idx_l0_active_minus1, 0);
        assert_eq!(slice_header(false).slice_type, 0);
        assert_eq!(slice_header(false).disable_deblocking_filter_idc, 0);
    }

    #[test]
    fn a_new_idr_picture_changes_its_id_and_restarts_the_counters() {
        let mut gop = GopState::new(2);
        let first = picture_info(&gop.next());
        let _ = gop.next();
        let second_plan = gop.next();
        let second = picture_info(&second_plan);
        assert_ne!(first.idr_pic_id, second.idr_pic_id);
        assert_eq!(second.frame_num, 0);
        assert_eq!(second.PicOrderCnt, 0);
    }

    #[test]
    fn a_slot_states_the_picture_it_holds() {
        let first = reference_info(0);
        assert_eq!(first.primary_pic_type, 5);
        assert_eq!((first.FrameNum, first.PicOrderCnt), (0, 0));
        let later = reference_info(3);
        assert_eq!(later.primary_pic_type, 0);
        assert_eq!((later.FrameNum, later.PicOrderCnt), (3, 6));
    }
}
