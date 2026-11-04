//! Unit tests of the mapping into `StdVideoH265*` structures, on the parameter sets of a
//! stream from x265. Each expected value comes from `ffmpeg -bsf:v trace_headers`.

use ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN;

use super::std_params::{StdParameterSets, StdPps, StdSps, StdVps};
use crate::bitstream::hevc_pps::PicParameterSet;
use crate::bitstream::hevc_sps_full::SeqParameterSet;
use crate::bitstream::hevc_vps::VideoParameterSet;

const REAL_VPS: [u8; 24] = [
    0x40, 0x01, 0x0c, 0x01, 0xff, 0xff, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03,
    0x00, 0x00, 0x03, 0x00, 0x1e, 0x95, 0x98, 0x09,
];
const REAL_SPS: [u8; 38] = [
    0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03,
    0x00, 0x1e, 0xa0, 0x10, 0x20, 0x61, 0x65, 0x95, 0x9a, 0x49, 0x32, 0xb8, 0x04, 0x00, 0x00, 0x0f,
    0xa0, 0x00, 0x01, 0x86, 0xa0, 0x20,
];
const REAL_PPS: [u8; 7] = [0x44, 0x01, 0xc1, 0x72, 0xb4, 0x22, 0x40];

fn sps() -> SeqParameterSet {
    SeqParameterSet::parse(&REAL_SPS).expect("parses")
}

fn pps() -> PicParameterSet {
    PicParameterSet::parse(&REAL_PPS).expect("parses")
}

#[test]
fn a_real_vps_maps_field_by_field() {
    let vps = VideoParameterSet::parse(&REAL_VPS).expect("parses");
    let owner = StdVps::new(&vps);
    let std = owner.as_std();
    assert_eq!(std.vps_video_parameter_set_id, 0);
    assert_eq!(std.vps_max_sub_layers_minus1, 0);
    assert!(std.pHrdParameters.is_null());
    // SAFETY: `owner` still lives, and `as_std` sets the pointer.
    let buffers = unsafe { &*std.pDecPicBufMgr };
    assert_eq!(buffers.max_dec_pic_buffering_minus1[0], 4);
    assert_eq!(buffers.max_num_reorder_pics[0], 2);
    assert_eq!(buffers.max_latency_increase_plus1[0], 5);
    // SAFETY: as above.
    let ptl = unsafe { &*std.pProfileTierLevel };
    assert_eq!(
        ptl.general_profile_idc,
        StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN
    );
    assert_eq!(ptl.general_level_idc, 0, "general_level_idc 30 is level 1");
}

#[test]
fn a_real_sps_maps_field_by_field() {
    let parsed = sps();
    let owner = StdSps::new(&parsed);
    let std = owner.as_std();
    assert_eq!(std.chroma_format_idc, 1);
    assert_eq!(std.pic_width_in_luma_samples, 128);
    assert_eq!(std.pic_height_in_luma_samples, 96);
    assert_eq!(std.sps_seq_parameter_set_id, 0);
    assert_eq!(
        (std.bit_depth_luma_minus8, std.bit_depth_chroma_minus8),
        (0, 0)
    );
    assert_eq!(std.log2_max_pic_order_cnt_lsb_minus4, 4);
    assert_eq!(std.log2_min_luma_coding_block_size_minus3, 0);
    assert_eq!(std.log2_diff_max_min_luma_coding_block_size, 3);
    assert_eq!(std.log2_min_luma_transform_block_size_minus2, 0);
    assert_eq!(std.log2_diff_max_min_luma_transform_block_size, 3);
    assert_eq!(std.num_short_term_ref_pic_sets, 0);
    assert_eq!(std.flags.conformance_window_flag(), 0);
    assert_eq!(std.flags.scaling_list_enabled_flag(), 0);
    assert_eq!(std.flags.amp_enabled_flag(), 0);
    assert_eq!(std.flags.sample_adaptive_offset_enabled_flag(), 1);
    assert_eq!(std.flags.long_term_ref_pics_present_flag(), 0);
    assert_eq!(std.flags.sps_temporal_mvp_enabled_flag(), 1);
    assert_eq!(std.flags.strong_intra_smoothing_enabled_flag(), 1);
    assert_eq!(std.flags.vui_parameters_present_flag(), 1);
    assert!(!std.pSequenceParameterSetVui.is_null());
    assert!(std.pScalingLists.is_null());
    assert!(std.pShortTermRefPicSet.is_null());
    // SAFETY: `owner` still lives, and `as_std` sets the pointer.
    let buffers = unsafe { &*std.pDecPicBufMgr };
    assert_eq!(buffers.max_dec_pic_buffering_minus1[0], 4);
    assert_eq!(buffers.max_num_reorder_pics[0], 2);
    assert_eq!(buffers.max_latency_increase_plus1[0], 5);
}

#[test]
fn a_real_pps_maps_field_by_field() {
    let parsed_sps = sps();
    let owner = StdPps::new(&pps(), &parsed_sps);
    let std = owner.as_std();
    assert_eq!(std.pps_pic_parameter_set_id, 0);
    assert_eq!(std.pps_seq_parameter_set_id, 0);
    assert_eq!(std.sps_video_parameter_set_id, 0);
    assert_eq!(std.init_qp_minus26, 0);
    assert_eq!(std.diff_cu_qp_delta_depth, 1);
    assert_eq!(std.log2_parallel_merge_level_minus2, 0);
    assert_eq!(std.flags.sign_data_hiding_enabled_flag(), 1);
    assert_eq!(std.flags.cu_qp_delta_enabled_flag(), 1);
    assert_eq!(std.flags.weighted_pred_flag(), 1);
    assert_eq!(std.flags.weighted_bipred_flag(), 0);
    assert_eq!(std.flags.tiles_enabled_flag(), 0);
    assert_eq!(std.flags.entropy_coding_sync_enabled_flag(), 0);
    assert_eq!(std.flags.pps_loop_filter_across_slices_enabled_flag(), 1);
    assert_eq!(std.flags.pps_scaling_list_data_present_flag(), 0);
    assert!(std.pScalingLists.is_null());
}

/// The driver reads through these pointers after the create call, so a move keeps them good.
#[test]
fn moved_parameter_sets_keep_what_their_pointers_name() {
    let vps = VideoParameterSet::parse(&REAL_VPS).expect("parses");
    let parsed_sps = sps();
    let sets = StdParameterSets::new(
        StdVps::new(&vps),
        StdSps::new(&parsed_sps),
        vec![StdPps::new(&pps(), &parsed_sps)],
    );
    let (vps_list, sps_list) = (sets.vps_list().as_ptr(), sets.sps_list().as_ptr());
    let moved = Some(sets);
    let moved = moved.as_ref().expect("the sets");
    assert_eq!(moved.vps_list().as_ptr(), vps_list);
    assert_eq!(moved.sps_list().as_ptr(), sps_list);
    assert_eq!(moved.pps_list().len(), 1);
    // SAFETY: `moved` owns each list and each structure the lists point to.
    let (vps_buffers, sps_buffers) =
        unsafe { (*(*vps_list).pDecPicBufMgr, *(*sps_list).pDecPicBufMgr) };
    assert_eq!(vps_buffers.max_num_reorder_pics[0], 2);
    assert_eq!(sps_buffers.max_num_reorder_pics[0], 2);
}

/// Each tile size comes from the parser's (6-3) and (6-4), with uniform spacing too.
#[test]
fn tile_sizes_come_from_6_3_and_6_4_for_uniform_spacing() {
    let parsed_sps = sps();
    let mut parsed = pps();
    parsed.tiles_enabled_flag = true;
    parsed.uniform_spacing_flag = true;
    parsed.num_tile_columns_minus1 = 1;
    parsed.num_tile_rows_minus1 = 1;
    let columns = parsed.column_widths(&parsed_sps);
    let rows = parsed.row_heights(&parsed_sps);
    let std = StdPps::new(&parsed, &parsed_sps).as_std();
    assert_eq!(std.num_tile_columns_minus1, 1);
    assert_eq!(std.num_tile_rows_minus1, 1);
    assert_eq!(std.flags.uniform_spacing_flag(), 1);
    for (index, width) in columns.iter().enumerate() {
        assert_eq!(u32::from(std.column_width_minus1[index]), width - 1);
    }
    for (index, height) in rows.iter().enumerate() {
        assert_eq!(u32::from(std.row_height_minus1[index]), height - 1);
    }
}
