//! The mapping of one picture into `StdVideoDecodeH265*` structures. It touches no device,
//! so tests read each field back.

use ash::vk::native::{
    StdVideoDecodeH265PictureInfo, StdVideoDecodeH265PictureInfoFlags,
    StdVideoDecodeH265ReferenceInfo, StdVideoDecodeH265ReferenceInfoFlags,
};

use super::slots::{ActiveReference, SlotLists};
use crate::bitstream::hevc_slice::SliceSegmentHeader;
use crate::decode::hevc_order::poc::{is_idr, is_irap};
use crate::vulkan::zeroed::{AllZero, zeroed};

// SAFETY: integers and bitfield units only.
unsafe impl AllZero for StdVideoDecodeH265PictureInfoFlags {}
// SAFETY: as above.
unsafe impl AllZero for StdVideoDecodeH265ReferenceInfoFlags {}

/// `NumDeltaPocsOfRefRpsIdx`: `NumDeltaPocs[ RefRpsIdx ]` of a predicted set in the slice
/// header, and 0 otherwise.
fn delta_pocs_of_ref(header: &SliceSegmentHeader) -> u8 {
    header
        .short_term_ref_pic_set
        .as_ref()
        .filter(|set| {
            !header.short_term_ref_pic_set_sps_flag && set.inter_ref_pic_set_prediction_flag
        })
        .map_or(0, |set| set.num_prediction_flags.saturating_sub(1))
}

/// The picture info of the first slice segment `header`.
///
/// `IsReference` is always 1. Each decoded picture stays available, because a sub-layer
/// non-reference picture can serve a picture of a higher sub-layer.
pub(super) fn std_picture_info(
    header: &SliceSegmentHeader,
    vps_id: u8,
    sps_id: u8,
    poc: i32,
    lists: &SlotLists,
) -> StdVideoDecodeH265PictureInfo {
    let mut flags: StdVideoDecodeH265PictureInfoFlags = zeroed();
    flags.set_IrapPicFlag(is_irap(header.nal_unit_type).into());
    flags.set_IdrPicFlag(is_idr(header.nal_unit_type).into());
    flags.set_IsReference(1);
    flags.set_short_term_ref_pic_set_sps_flag(header.short_term_ref_pic_set_sps_flag.into());
    StdVideoDecodeH265PictureInfo {
        flags,
        sps_video_parameter_set_id: vps_id,
        pps_seq_parameter_set_id: sps_id,
        pps_pic_parameter_set_id: header.slice_pic_parameter_set_id as u8,
        NumDeltaPocsOfRefRpsIdx: delta_pocs_of_ref(header),
        PicOrderCntVal: poc,
        NumBitsForSTRefPicSetInSlice: header.short_term_ref_pic_set_size as u16,
        reserved: 0,
        RefPicSetStCurrBefore: lists.before,
        RefPicSetStCurrAfter: lists.after,
        RefPicSetLtCurr: lists.long_term,
    }
}

/// The reference info of one active reference, or of the current picture.
pub(super) fn std_reference_info(reference: &ActiveReference) -> StdVideoDecodeH265ReferenceInfo {
    let mut flags: StdVideoDecodeH265ReferenceInfoFlags = zeroed();
    flags.set_used_for_long_term_reference(reference.long_term.into());
    StdVideoDecodeH265ReferenceInfo {
        flags,
        PicOrderCntVal: reference.poc,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::hevc_rps::ShortTermRefPicSet;

    fn lists() -> SlotLists {
        SlotLists {
            before: [3, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
            after: [0xFF; 8],
            long_term: [1, 2, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
            references: Vec::new(),
        }
    }

    #[test]
    fn an_idr_picture_states_both_irap_and_idr() {
        let header = SliceSegmentHeader {
            nal_unit_type: 19,
            slice_pic_parameter_set_id: 2,
            ..SliceSegmentHeader::default()
        };
        let info = std_picture_info(&header, 1, 4, 0, &lists());
        assert_eq!(info.flags.IrapPicFlag(), 1);
        assert_eq!(info.flags.IdrPicFlag(), 1);
        assert_eq!(info.flags.IsReference(), 1);
        assert_eq!(info.sps_video_parameter_set_id, 1);
        assert_eq!(info.pps_seq_parameter_set_id, 4);
        assert_eq!(info.pps_pic_parameter_set_id, 2);
    }

    #[test]
    fn a_cra_picture_is_irap_and_not_idr_and_a_trailing_picture_is_neither() {
        let cra = SliceSegmentHeader {
            nal_unit_type: 21,
            ..SliceSegmentHeader::default()
        };
        let info = std_picture_info(&cra, 0, 0, 40, &lists());
        assert_eq!((info.flags.IrapPicFlag(), info.flags.IdrPicFlag()), (1, 0));
        assert_eq!(info.PicOrderCntVal, 40);
        let trailing = SliceSegmentHeader {
            nal_unit_type: 0,
            ..SliceSegmentHeader::default()
        };
        let info = std_picture_info(&trailing, 0, 0, -3, &lists());
        assert_eq!((info.flags.IrapPicFlag(), info.flags.IdrPicFlag()), (0, 0));
        assert_eq!(
            info.flags.IsReference(),
            1,
            "the sub-layer can serve another"
        );
        assert_eq!(info.PicOrderCntVal, -3);
    }

    #[test]
    fn a_predicted_set_in_the_header_states_the_size_of_its_source() {
        let predicted = ShortTermRefPicSet {
            inter_ref_pic_set_prediction_flag: true,
            num_prediction_flags: 5,
            ..ShortTermRefPicSet::default()
        };
        let header = SliceSegmentHeader {
            nal_unit_type: 1,
            short_term_ref_pic_set: Some(predicted),
            short_term_ref_pic_set_size: 19,
            ..SliceSegmentHeader::default()
        };
        let info = std_picture_info(&header, 0, 0, 1, &lists());
        assert_eq!(info.NumDeltaPocsOfRefRpsIdx, 4);
        assert_eq!(info.NumBitsForSTRefPicSetInSlice, 19);
        assert_eq!(info.flags.short_term_ref_pic_set_sps_flag(), 0);

        let explicit = SliceSegmentHeader {
            short_term_ref_pic_set: Some(ShortTermRefPicSet::default()),
            ..header.clone()
        };
        let info = std_picture_info(&explicit, 0, 0, 1, &lists());
        assert_eq!(info.NumDeltaPocsOfRefRpsIdx, 0);

        let from_sps = SliceSegmentHeader {
            short_term_ref_pic_set_sps_flag: true,
            short_term_ref_pic_set: None,
            short_term_ref_pic_set_size: 0,
            ..header
        };
        let info = std_picture_info(&from_sps, 0, 0, 1, &lists());
        assert_eq!(info.flags.short_term_ref_pic_set_sps_flag(), 1);
        assert_eq!(info.NumDeltaPocsOfRefRpsIdx, 0);
        assert_eq!(info.NumBitsForSTRefPicSetInSlice, 0);
    }

    #[test]
    fn slot_lists_pass_through_in_order() {
        let header = SliceSegmentHeader::default();
        let info = std_picture_info(&header, 0, 0, 0, &lists());
        assert_eq!(info.RefPicSetStCurrBefore[..2], [3, 0xFF]);
        assert_eq!(info.RefPicSetStCurrAfter, [0xFF; 8]);
        assert_eq!(info.RefPicSetLtCurr[..3], [1, 2, 0xFF]);
    }

    #[test]
    fn a_reference_states_its_poc_and_term() {
        let info = std_reference_info(&ActiveReference {
            slot: 1,
            poc: -8,
            long_term: true,
        });
        assert_eq!(info.PicOrderCntVal, -8);
        assert_eq!(info.flags.used_for_long_term_reference(), 1);
        assert_eq!(info.flags.unused_for_reference(), 0);
    }
}
