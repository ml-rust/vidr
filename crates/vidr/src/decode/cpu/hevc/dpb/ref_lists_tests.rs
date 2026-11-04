//! (8-8) to (8-11) and 8.3.5 on hand-made sets.

use super::*;
use crate::bitstream::hevc_pred_weight::PredWeightTable;
use crate::bitstream::hevc_slice::SLICE_TYPE_P;
use crate::bitstream::hevc_sps_full::tests as sps_tests;
use crate::decode::cpu::hevc::motion::col::ColMotion;
use crate::decode::cpu::hevc::picture::Picture;

fn entry(poc: i32, long_term: bool) -> RefEntry<u8> {
    RefEntry {
        frame: Some(Arc::new(Frame {
            picture: Picture::new(8, 8).unwrap(),
            col: ColMotion::new(),
            poc,
        })),
        poc,
        long_term,
    }
}

/// Before: POC 4, 2. After: POC 8. Long-term: POC 1.
fn sets() -> RefPicSets<u8> {
    let mut sets = RefPicSets::empty();
    sets.before = vec![entry(4, false), entry(2, false)];
    sets.after = vec![entry(8, false)];
    sets.long_term = vec![entry(1, true)];
    sets
}

fn header(slice_type: u8, active: [u32; 2]) -> SliceSegmentHeader {
    SliceSegmentHeader {
        slice_type,
        num_ref_idx_l0_active_minus1: active[0] - 1,
        num_ref_idx_l1_active_minus1: active[1] - 1,
        num_pic_total_curr: 4,
        collocated_from_l0_flag: true,
        ..SliceSegmentHeader::default()
    }
}

fn sps() -> SeqParameterSet {
    sps_tests::valid(&sps_tests::Spec::default())
}

fn pocs(refs: &SliceRefs<u8>, list: usize) -> Vec<i32> {
    refs.frames[list].iter().map(|frame| frame.poc).collect()
}

#[test]
fn an_i_slice_gets_no_lists() {
    let refs = SliceRefs::build(&header(SLICE_TYPE_I, [1, 1]), &sets(), 6, &sps()).unwrap();
    assert!(refs.frames[0].is_empty());
    assert_eq!(refs.lists.count, [0, 0]);
}

/// (8-8) orders before, after, long-term. (8-10) orders after, before, long-term.
#[test]
fn the_lists_follow_8_8_and_8_10() {
    let refs = SliceRefs::build(&header(SLICE_TYPE_B, [4, 4]), &sets(), 6, &sps()).unwrap();
    assert_eq!(pocs(&refs, 0), vec![4, 2, 8, 1]);
    assert_eq!(pocs(&refs, 1), vec![8, 4, 2, 1]);
    assert_eq!(refs.lists.count, [4, 4]);
    assert_eq!(refs.lists.poc[1][0], 8);
    assert!(refs.lists.long_term[0][3]);
    assert!(!refs.lists.long_term[0][0]);
}

#[test]
fn a_p_slice_has_one_list() {
    let refs = SliceRefs::build(&header(SLICE_TYPE_P, [2, 1]), &sets(), 6, &sps()).unwrap();
    assert_eq!(pocs(&refs, 0), vec![4, 2]);
    assert!(refs.frames[1].is_empty());
}

/// NumRpsCurrTempList0 is 7 with an active count above the set: the sets repeat.
#[test]
fn the_sets_repeat_past_their_end() {
    let refs = SliceRefs::build(&header(SLICE_TYPE_P, [7, 1]), &sets(), 6, &sps()).unwrap();
    assert_eq!(pocs(&refs, 0), vec![4, 2, 8, 1, 4, 2, 8]);
}

/// (8-9): `list_entry_l0` selects from the temporary list.
#[test]
fn list_entries_select_from_the_temporary_list() {
    let mut request = header(SLICE_TYPE_P, [3, 1]);
    request.ref_pic_list_modification_flag_l0 = true;
    request.list_entry_l0[0] = 3;
    request.list_entry_l0[1] = 0;
    request.list_entry_l0[2] = 3;
    let refs = SliceRefs::build(&request, &sets(), 6, &sps()).unwrap();
    assert_eq!(pocs(&refs, 0), vec![1, 4, 1]);
}

/// A list entry in "no reference picture" is a named error.
#[test]
fn a_missing_reference_is_a_named_error() {
    let mut holes = sets();
    holes.before[0].frame = None;
    let result = SliceRefs::build(&header(SLICE_TYPE_P, [1, 1]), &holes, 6, &sps());
    assert_eq!(
        result.map(|refs| refs.lists.count),
        Err(HevcError::MissingReference { poc: 4 })
    );
}

#[test]
fn an_empty_set_is_a_named_error() {
    let result = SliceRefs::<u8>::build(
        &header(SLICE_TYPE_P, [1, 1]),
        &RefPicSets::empty(),
        6,
        &sps(),
    );
    assert!(matches!(result, Err(HevcError::InvalidSyntax { .. })));
}

/// 8.3.5: POC 8 follows POC 6, and in a P slice of the first two entries no picture
/// does.
#[test]
fn no_backward_pred_reads_every_list() {
    let refs = SliceRefs::build(&header(SLICE_TYPE_B, [4, 4]), &sets(), 6, &sps()).unwrap();
    assert!(!refs.no_backward_pred);
    let refs = SliceRefs::build(&header(SLICE_TYPE_P, [2, 1]), &sets(), 6, &sps()).unwrap();
    assert!(refs.no_backward_pred);
    let refs = SliceRefs::build(&header(SLICE_TYPE_B, [4, 4]), &sets(), 8, &sps()).unwrap();
    assert!(refs.no_backward_pred);
}

/// 8.3.5: a B slice with `collocated_from_l0_flag` 0 reads L1, and otherwise L0.
#[test]
fn the_collocated_picture_follows_its_flag() {
    let mut request = header(SLICE_TYPE_B, [4, 4]);
    let refs = SliceRefs::build(&request, &sets(), 6, &sps()).unwrap();
    assert!(refs.col.is_none());
    request.slice_temporal_mvp_enabled_flag = true;
    request.collocated_ref_idx = 1;
    let refs = SliceRefs::build(&request, &sets(), 6, &sps()).unwrap();
    assert_eq!(refs.col.unwrap().poc, 2);
    request.collocated_from_l0_flag = false;
    let refs = SliceRefs::build(&request, &sets(), 6, &sps()).unwrap();
    assert_eq!(refs.col.unwrap().poc, 4);
}

#[test]
fn a_weight_table_gives_explicit_weights() {
    let mut request = header(SLICE_TYPE_P, [1, 1]);
    let refs = SliceRefs::build(&request, &sets(), 6, &sps()).unwrap();
    assert!(refs.weights.is_none());
    request.pred_weight_table = Some(Box::new(PredWeightTable::default()));
    let refs = SliceRefs::build(&request, &sets(), 6, &sps()).unwrap();
    assert!(refs.weights.is_some());
}

/// A list shares the picture: the slice holds one more `Arc`, and no copy.
#[test]
fn a_list_shares_the_picture() {
    let holder = sets();
    let refs = SliceRefs::build(&header(SLICE_TYPE_P, [1, 1]), &holder, 6, &sps()).unwrap();
    let in_set = holder.before[0].frame.as_ref().unwrap();
    assert!(Arc::ptr_eq(in_set, &refs.frames[0][0]));
    assert_eq!(Arc::strong_count(in_set), 2);
}
