//! Unit tests of the mapping into `StdVideoH264*` structures.

use ash::vk::native::{
    StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_0,
    StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_1,
    StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_2,
};
use h264_reader::Context;

use super::dpb::{CurrentPicture, Marking, Mmco, RefFrame, marked_frame};
use super::params::{parse_pps, parse_sps};
use super::std_video::{
    PictureFacts, StdParameterSets, StdPps, StdSps, std_level, std_picture_info, std_reference_info,
};
use super::test_nal::{NalWriter, REAL_PPS, REAL_SPS, TestSps, sps_nal};

#[test]
fn levels_follow_the_header_values() {
    assert_eq!(
        std_level(10),
        Some(StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_0)
    );
    assert_eq!(
        std_level(31),
        Some(StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_1)
    );
    assert_eq!(
        std_level(62),
        Some(StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_2)
    );
    assert_eq!(std_level(9), std_level(11));
    assert_eq!(std_level(99), None);
}

/// Every value here comes from `ffmpeg -bsf:v trace_headers` on the stream.
#[test]
fn a_real_sps_maps_field_by_field() {
    let sps = parse_sps(&REAL_SPS).expect("parses");
    let owner = StdSps::new(&sps);
    let std = owner.as_std();
    assert_eq!(std.profile_idc, 100);
    assert_eq!(std.level_idc, std_level(12).expect("a level"));
    assert_eq!(std.chroma_format_idc, 1);
    assert_eq!(std.seq_parameter_set_id, 0);
    assert_eq!(
        (std.bit_depth_luma_minus8, std.bit_depth_chroma_minus8),
        (0, 0)
    );
    assert_eq!(std.log2_max_frame_num_minus4, 0);
    assert_eq!(std.pic_order_cnt_type, 0);
    assert_eq!(std.log2_max_pic_order_cnt_lsb_minus4, 2);
    assert_eq!(std.max_num_ref_frames, 4);
    assert_eq!(std.pic_width_in_mbs_minus1, 19);
    assert_eq!(std.pic_height_in_map_units_minus1, 11);
    assert_eq!(std.frame_crop_bottom_offset, 6);
    assert_eq!(std.frame_crop_top_offset, 0);
    assert_eq!(std.flags.frame_mbs_only_flag(), 1);
    assert_eq!(std.flags.direct_8x8_inference_flag(), 1);
    assert_eq!(std.flags.frame_cropping_flag(), 1);
    assert_eq!(std.flags.vui_parameters_present_flag(), 1);
    assert_eq!(std.flags.seq_scaling_matrix_present_flag(), 0);
    assert_eq!(std.flags.mb_adaptive_frame_field_flag(), 0);
    assert!(std.pScalingLists.is_null());
    assert!(std.pOffsetForRefFrame.is_null());

    // SAFETY: the flag above says the pointer is set, and `owner` still lives.
    let vui = unsafe { &*std.pSequenceParameterSetVui };
    assert_eq!(vui.aspect_ratio_idc, 1);
    assert_eq!(vui.video_format, 5);
    assert_eq!(
        (
            vui.colour_primaries,
            vui.transfer_characteristics,
            vui.matrix_coefficients
        ),
        (1, 1, 1)
    );
    assert_eq!((vui.num_units_in_tick, vui.time_scale), (1, 50));
    assert_eq!(
        (vui.max_num_reorder_frames, vui.max_dec_frame_buffering),
        (2, 4)
    );
    assert_eq!(vui.flags.bitstream_restriction_flag(), 1);
    assert_eq!(vui.flags.nal_hrd_parameters_present_flag(), 0);
    assert!(vui.pHrdParameters.is_null());
}

#[test]
fn a_real_pps_maps_field_by_field() {
    let sps = parse_sps(&REAL_SPS).expect("parses");
    let mut context = Context::new();
    context.put_seq_param_set(sps.clone());
    let pps = parse_pps(&context, &REAL_PPS).expect("parses");
    let owner = StdPps::new(&pps, &sps);
    let std = owner.as_std();
    assert_eq!((std.seq_parameter_set_id, std.pic_parameter_set_id), (0, 0));
    assert_eq!(std.flags.entropy_coding_mode_flag(), 1);
    assert_eq!(std.flags.weighted_pred_flag(), 1);
    assert_eq!(std.weighted_bipred_idc, 2);
    assert_eq!(std.num_ref_idx_l0_default_active_minus1, 2);
    assert_eq!(std.num_ref_idx_l1_default_active_minus1, 0);
    assert_eq!(std.pic_init_qp_minus26, -3);
    assert_eq!(std.pic_init_qs_minus26, 0);
    assert_eq!(std.chroma_qp_index_offset, -2);
    assert_eq!(std.second_chroma_qp_index_offset, -2);
    assert_eq!(std.flags.transform_8x8_mode_flag(), 1);
    assert_eq!(std.flags.deblocking_filter_control_present_flag(), 1);
    assert_eq!(std.flags.pic_scaling_matrix_present_flag(), 0);
    assert!(std.pScalingLists.is_null());
}

#[test]
fn poc_type_1_keeps_its_cycle() {
    let type1 = TestSps {
        poc_type: 1,
        ..TestSps::default()
    };
    let sps = parse_sps(&sps_nal(&type1)).expect("parses");
    let owner = StdSps::new(&sps);
    let std = owner.as_std();
    assert_eq!(std.pic_order_cnt_type, 1);
    assert_eq!(std.offset_for_non_ref_pic, -1);
    assert_eq!(std.offset_for_top_to_bottom_field, 1);
    assert_eq!(std.num_ref_frames_in_pic_order_cnt_cycle, 2);
    assert_eq!(std.flags.delta_pic_order_always_zero_flag(), 0);
    // SAFETY: the count above says two offsets are set, and `owner` still lives.
    let offsets = unsafe { std::slice::from_raw_parts(std.pOffsetForRefFrame, 2) };
    assert_eq!(offsets, [2, 4]);
}

/// The driver reads through these pointers after the create call, so a move keeps them good.
#[test]
fn moved_parameter_sets_keep_what_their_pointers_name() {
    let type1 = TestSps {
        poc_type: 1,
        ..TestSps::default()
    };
    let sps = parse_sps(&sps_nal(&type1)).expect("parses");
    let sets = StdParameterSets::new(StdSps::new(&sps), Vec::new());
    let list = sets.sps_list().as_ptr();
    let moved = Some(sets);
    let moved = moved.as_ref().expect("the sets");
    assert_eq!(moved.sps_list().as_ptr(), list);
    assert!(moved.pps_list().is_empty());
    // SAFETY: `moved` owns the list and the offsets, and the count below says two are set.
    let offsets = unsafe {
        assert_eq!((*list).num_ref_frames_in_pic_order_cnt_cycle, 2);
        std::slice::from_raw_parts((*list).pOffsetForRefFrame, 2)
    };
    assert_eq!(offsets, [2, 4]);
}

/// An SPS with a scaling matrix: list 0 explicit, list 1 absent, list 2 default, and
/// the rest absent. Table 7-2 rule set A gives each fall-back.
#[test]
fn scaling_lists_resolve_and_set_each_present_bit() {
    let mut nal = NalWriter::new(0x67);
    nal.bits(100, 8)
        .bits(0, 8)
        .bits(30, 8)
        .ue(0)
        .ue(1)
        .ue(0)
        .ue(0)
        .flag(false);
    nal.flag(true);
    // List 0: a delta of 12 from the start of 8 gives 20, and deltas of 0 keep it.
    nal.flag(true).se(12);
    for _ in 1..16 {
        nal.se(0);
    }
    nal.flag(false);
    // List 2: a delta of -8 from 8 gives next_scale 0 at j 0, the default-list form.
    nal.flag(true).se(-8);
    // Lists 3, 4, 5, and both 8x8 lists are absent.
    for _ in 3..8 {
        nal.flag(false);
    }
    nal.ue(0)
        .ue(0)
        .ue(2)
        .ue(4)
        .flag(false)
        .ue(19)
        .ue(11)
        .flag(true)
        .flag(true);
    nal.flag(false).flag(false);
    let sps = parse_sps(&nal.finish()).expect("parses");
    let owner = StdSps::new(&sps);
    let std = owner.as_std();
    assert_eq!(std.flags.seq_scaling_matrix_present_flag(), 1);
    // SAFETY: the flag above says the pointer is set, and `owner` still lives.
    let lists = unsafe { &*std.pScalingLists };
    assert_eq!(lists.scaling_list_present_mask, 0b1111_1111);
    assert_eq!(lists.use_default_scaling_matrix_mask, 0);
    assert_eq!(lists.ScalingList4x4[0], [20; 16]);
    assert_eq!(
        lists.ScalingList4x4[1], [20; 16],
        "list 1 falls back to list 0"
    );
    // Default_4x4_Intra in scan order, Table 7-3.
    let intra = [
        6, 13, 13, 20, 20, 20, 28, 28, 28, 28, 32, 32, 32, 37, 37, 42,
    ];
    assert_eq!(lists.ScalingList4x4[2], intra);
    // Default_4x4_Inter in scan order, Table 7-3.
    let inter = [
        10, 14, 14, 20, 20, 20, 24, 24, 24, 24, 27, 27, 27, 30, 30, 34,
    ];
    assert_eq!(
        lists.ScalingList4x4[3], inter,
        "list 3 takes the inter default"
    );
    assert_eq!(lists.ScalingList4x4[5], inter);
    assert_eq!(lists.ScalingList8x8[0][0], 6, "the intra 8x8 default");
    assert_eq!(lists.ScalingList8x8[1][0], 9, "the inter 8x8 default");
}

#[test]
fn picture_and_reference_info_carry_each_value() {
    let picture = std_picture_info(&PictureFacts {
        sps_id: 2,
        pps_id: 3,
        frame_num: 7,
        idr_pic_id: 9,
        poc: [10, 11],
        idr: true,
        intra: true,
        reference: true,
    });
    assert_eq!(picture.flags.IdrPicFlag(), 1);
    assert_eq!(picture.flags.is_intra(), 1);
    assert_eq!(picture.flags.is_reference(), 1);
    assert_eq!(picture.flags.field_pic_flag(), 0);
    assert_eq!(
        (picture.seq_parameter_set_id, picture.pic_parameter_set_id),
        (2, 3)
    );
    assert_eq!((picture.frame_num, picture.idr_pic_id), (7, 9));
    assert_eq!(picture.PicOrderCnt, [10, 11]);

    let short = RefFrame {
        slot: 1,
        id: 5,
        frame_num: 12,
        long_term: None,
        poc: [4, 5],
    };
    let info = std_reference_info(&short);
    assert_eq!(info.FrameNum, 12);
    assert_eq!(info.flags.used_for_long_term_reference(), 0);
    assert_eq!(info.flags.top_field_flag(), 0);
    assert_eq!(info.flags.bottom_field_flag(), 0);
    assert_eq!(info.PicOrderCnt, [4, 5]);

    let long = RefFrame {
        long_term: Some(3),
        ..short
    };
    let info = std_reference_info(&long);
    assert_eq!(info.FrameNum, 3);
    assert_eq!(info.flags.used_for_long_term_reference(), 1);
}

/// The setup slot of each marking holds what the Vulkan H.264 decode text states: the
/// long-term mark of 8.2.5.1, and the values 8.2 gives the picture after its marking.
#[test]
fn setup_slot_info_states_the_marked_picture() {
    let current = CurrentPicture {
        slot: 2,
        id: 8,
        frame_num: 6,
        poc: [24, 23],
    };
    let setup = |marking: &Marking| std_reference_info(&marked_frame(current, marking));

    let short = setup(&Marking::SlidingWindow);
    assert_eq!(short.flags.used_for_long_term_reference(), 0);
    assert_eq!((short.FrameNum, short.PicOrderCnt), (6, [24, 23]));

    let idr = setup(&Marking::Idr { long_term: true });
    assert_eq!(idr.flags.used_for_long_term_reference(), 1);
    assert_eq!(idr.FrameNum, 0, "LongTermFrameIdx 0");

    let current_to_long = setup(&Marking::Adaptive(vec![Mmco::CurrentToLong {
        long_term_frame_idx: 3,
    }]));
    assert_eq!(current_to_long.flags.used_for_long_term_reference(), 1);
    assert_eq!(current_to_long.FrameNum, 3, "LongTermFrameIdx 3");
    assert_eq!(current_to_long.PicOrderCnt, [24, 23]);

    let reset = setup(&Marking::Adaptive(vec![Mmco::UnmarkAll]));
    assert_eq!(reset.flags.used_for_long_term_reference(), 0);
    assert_eq!(reset.FrameNum, 0);
    assert_eq!(reset.PicOrderCnt, [1, 0]);
}
