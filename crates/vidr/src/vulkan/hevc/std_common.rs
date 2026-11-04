//! The `StdVideoH265*` structures that a VPS and an SPS both point to. It also holds the
//! scaling lists of an SPS and a PPS.

use ash::vk::native::{
    StdVideoH265DecPicBufMgr, StdVideoH265ProfileIdc, StdVideoH265ProfileTierLevel,
    StdVideoH265ProfileTierLevelFlags, StdVideoH265ScalingLists,
};

use super::sequence::std_level;
use crate::bitstream::hevc_scaling::ScalingListData;
use crate::vulkan::zeroed::{AllZero, zeroed};

// SAFETY: integers and bitfield units only.
unsafe impl AllZero for StdVideoH265ProfileTierLevelFlags {}

/// `STD_VIDEO_H265_LEVEL_IDC_INVALID` and `STD_VIDEO_H265_PROFILE_IDC_INVALID`.
const INVALID: u32 = 0x7FFF_FFFF;
/// The top bit of the 48 constraint bits: `general_progressive_source_flag`.
const PROGRESSIVE_SOURCE_BIT: u32 = 47;

/// The profile tier level the parser reads. `constraint_flags` holds 48 bits, with
/// `general_progressive_source_flag` on top.
pub(super) fn std_profile_tier_level(
    profile_idc: u8,
    tier_flag: bool,
    level_idc: u8,
    constraint_flags: u64,
) -> StdVideoH265ProfileTierLevel {
    let bit =
        |index: u32| u32::from((constraint_flags >> (PROGRESSIVE_SOURCE_BIT - index)) & 1 == 1);
    let mut flags: StdVideoH265ProfileTierLevelFlags = zeroed();
    flags.set_general_tier_flag(tier_flag.into());
    flags.set_general_progressive_source_flag(bit(0));
    flags.set_general_interlaced_source_flag(bit(1));
    flags.set_general_non_packed_constraint_flag(bit(2));
    flags.set_general_frame_only_constraint_flag(bit(3));
    StdVideoH265ProfileTierLevel {
        flags,
        general_profile_idc: match profile_idc {
            1..=4 | 9 => StdVideoH265ProfileIdc::from(profile_idc),
            _ => INVALID,
        },
        general_level_idc: std_level(level_idc).unwrap_or(INVALID),
    }
}

/// The per-sub-layer buffering of a VPS or an SPS.
pub(super) fn std_dec_pic_buf_mgr(
    latency_plus1: &[u32; 7],
    buffering_minus1: &[u32; 7],
    reorder: &[u32; 7],
) -> StdVideoH265DecPicBufMgr {
    StdVideoH265DecPicBufMgr {
        max_latency_increase_plus1: *latency_plus1,
        max_dec_pic_buffering_minus1: buffering_minus1.map(|value| value as u8),
        max_num_reorder_pics: reorder.map(|value| value as u8),
    }
}

/// `ScalingList`, in the coefficient order the headers state it. `scaling_list_dc_coef_minus8`
/// goes back to the coded DC value. Size 3 holds `matrixId` 0 and 3.
pub(super) fn std_scaling_lists(data: &ScalingListData) -> StdVideoH265ScalingLists {
    let mut lists = StdVideoH265ScalingLists {
        ScalingList4x4: [[0; 16]; 6],
        ScalingList8x8: data.scaling_list[1],
        ScalingList16x16: data.scaling_list[2],
        ScalingList32x32: [data.scaling_list[3][0], data.scaling_list[3][3]],
        ScalingListDCCoef16x16: [0; 6],
        ScalingListDCCoef32x32: [0; 2],
    };
    for (target, list) in lists.ScalingList4x4.iter_mut().zip(&data.scaling_list[0]) {
        target.copy_from_slice(&list[..16]);
    }
    let dc = |value: i16| (value + 8) as u8;
    lists.ScalingListDCCoef16x16 = data.scaling_list_dc_coef_minus8[0].map(dc);
    lists.ScalingListDCCoef32x32 = [
        dc(data.scaling_list_dc_coef_minus8[1][0]),
        dc(data.scaling_list_dc_coef_minus8[1][3]),
    ];
    lists
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constraint_flags_come_from_the_top_of_the_48_bits() {
        // Progressive and frame only.
        let flags = 0b1001u64 << 44;
        let ptl = std_profile_tier_level(2, true, 93, flags);
        assert_eq!(ptl.flags.general_tier_flag(), 1);
        assert_eq!(ptl.flags.general_progressive_source_flag(), 1);
        assert_eq!(ptl.flags.general_interlaced_source_flag(), 0);
        assert_eq!(ptl.flags.general_non_packed_constraint_flag(), 0);
        assert_eq!(ptl.flags.general_frame_only_constraint_flag(), 1);
        assert_eq!(ptl.general_profile_idc, 2);
        assert_eq!(ptl.general_level_idc, 4);
    }

    #[test]
    fn an_unknown_profile_or_level_is_the_invalid_value() {
        let ptl = std_profile_tier_level(7, false, 33, 0);
        assert_eq!(ptl.general_profile_idc, INVALID);
        assert_eq!(ptl.general_level_idc, INVALID);
    }

    #[test]
    fn scaling_lists_hold_the_lists_and_the_coded_dc() {
        let mut data = ScalingListData::default_lists();
        data.scaling_list[0][2][15] = 77;
        data.scaling_list[1][4][63] = 78;
        data.scaling_list[2][5][0] = 79;
        data.scaling_list[3][3][1] = 80;
        data.scaling_list_dc_coef_minus8[0][1] = -7;
        data.scaling_list_dc_coef_minus8[1][3] = 247;
        let lists = std_scaling_lists(&data);
        assert_eq!(lists.ScalingList4x4[2][15], 77);
        assert_eq!(lists.ScalingList8x8[4][63], 78);
        assert_eq!(lists.ScalingList16x16[5][0], 79);
        assert_eq!(lists.ScalingList32x32[1][1], 80);
        assert_eq!(lists.ScalingListDCCoef16x16[1], 1);
        assert_eq!(lists.ScalingListDCCoef32x32[1], 255);
    }
}
