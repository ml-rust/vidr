//! Marking (8.3.2), generated pictures (8.3.3), and bumping (C.5.2).

use super::*;
use crate::bitstream::hevc_rps::ShortTermRefPicSet;
use crate::bitstream::hevc_slice::SliceSegmentHeader;
use crate::bitstream::hevc_sps_full::tests as sps_tests;

const TRAIL_R: u8 = 1;
const MAX_LSB: u32 = 256;

const LIMITS: DpbLimits = DpbLimits {
    max_num_reorder: 2,
    max_latency: None,
    max_dec_buffering: 5,
};

const FORMAT: FrameFormat = FrameFormat {
    width: 16,
    height: 8,
    bit_depth: 8,
    window: DisplayWindow {
        x: 0,
        y: 0,
        width: 16,
        height: 8,
    },
};

fn frame(dpb: &mut Dpb<u8>, poc: i32) -> Frame<u8> {
    let mut frame = dpb.take_frame(&FORMAT).unwrap();
    frame.poc = poc;
    frame
}

fn store(dpb: &mut Dpb<u8>, poc: i32, output: bool) {
    let new_frame = frame(dpb, poc);
    dpb.insert(new_frame, output, &LIMITS);
}

fn outputs(dpb: &mut Dpb<u8>) -> Vec<i32> {
    let mut pocs = Vec::new();
    while let Some(frame) = dpb.pop_output() {
        pocs.push(frame.poc);
    }
    pocs
}

/// POC lists from a short-term set at POC `poc`: entries in `current` are used by the
/// current picture, and entries in `following` are not.
fn rps(poc: i32, current: &[i32], following: &[i32]) -> RpsPocs {
    let mut set = ShortTermRefPicSet::default();
    let mut negative = 0;
    let mut positive = 0;
    for (delta, used) in current
        .iter()
        .map(|&delta| (delta, true))
        .chain(following.iter().map(|&delta| (delta, false)))
    {
        if delta < 0 {
            set.delta_poc_s0[negative] = delta;
            set.used_by_curr_pic_s0[negative] = used;
            negative += 1;
        } else {
            set.delta_poc_s1[positive] = delta;
            set.used_by_curr_pic_s1[positive] = used;
            positive += 1;
        }
    }
    set.num_negative_pics = negative as u8;
    set.num_positive_pics = positive as u8;
    let header = SliceSegmentHeader {
        nal_unit_type: TRAIL_R,
        short_term_ref_pic_set: Some(set),
        ..SliceSegmentHeader::default()
    };
    let sps = sps_tests::valid(&sps_tests::Spec::default());
    RpsPocs::derive(&header, &sps, poc)
}

#[test]
fn the_rps_finds_short_term_pictures_and_marks_the_rest_unused() {
    let mut dpb = Dpb::<u8>::new();
    store(&mut dpb, 0, false);
    store(&mut dpb, 1, false);
    store(&mut dpb, 2, false);
    let sets = dpb.apply_rps(&rps(3, &[-1], &[-3]), MAX_LSB, false);
    assert_eq!(sets.before.len(), 1);
    assert_eq!(sets.before[0].frame.as_ref().unwrap().poc, 2);
    assert!(sets.missing_foll.is_empty());
    // POC 1 is out of the set and waits for no output, so it leaves.
    dpb.before_picture(false, false, &LIMITS);
    let pocs: Vec<i32> = dpb
        .core
        .entries
        .iter()
        .map(|entry| entry.frame.poc)
        .collect();
    assert!(pocs.contains(&0) && pocs.contains(&2) && !pocs.contains(&1));
}

#[test]
fn a_missing_current_entry_holds_no_picture() {
    let mut dpb = Dpb::<u8>::new();
    let sets = dpb.apply_rps(&rps(3, &[-1], &[1]), MAX_LSB, false);
    assert!(sets.before[0].frame.is_none());
    assert_eq!(sets.before[0].poc, 2);
    assert_eq!(sets.missing_foll, vec![(4, false)]);
}

/// An IRAP picture with `NoRaslOutputFlag` 1 marks every picture unused first.
#[test]
fn an_irap_picture_marks_every_picture_unused() {
    let mut dpb = Dpb::<u8>::new();
    store(&mut dpb, 0, false);
    let sets = dpb.apply_rps(&rps(3, &[-3], &[]), MAX_LSB, true);
    assert!(sets.before[0].frame.is_none());
}

/// (8-6): an entry with no MSB compares the LSB alone, and step 2 marks it.
#[test]
fn a_long_term_entry_matches_its_lsb_and_changes_the_marking() {
    let mut dpb = Dpb::<u8>::new();
    store(&mut dpb, 256 + 5, false);
    let mut lists = rps(300, &[], &[]);
    lists.lt_curr.push_for_tests(LongTermPoc {
        poc: 5,
        msb_present: false,
    });
    let sets = dpb.apply_rps(&lists, MAX_LSB, false);
    assert_eq!(sets.long_term[0].frame.as_ref().unwrap().poc, 261);
    assert_eq!(dpb.core.entries[0].marking, Marking::LongTerm);
    // A short-term search no longer finds it.
    let sets = dpb.apply_rps(&rps(300, &[-39], &[]), MAX_LSB, false);
    assert!(sets.before[0].frame.is_none());
}

/// C.5.2.3: pictures wait until the count passes `sps_max_num_reorder_pics`, then
/// leave in POC order.
#[test]
fn pictures_leave_in_poc_order_past_the_reorder_limit() {
    let mut dpb = Dpb::<u8>::new();
    store(&mut dpb, 8, true);
    store(&mut dpb, 4, true);
    assert_eq!(outputs(&mut dpb), Vec::<i32>::new());
    store(&mut dpb, 2, true);
    assert_eq!(outputs(&mut dpb), vec![2]);
    store(&mut dpb, 3, true);
    assert_eq!(outputs(&mut dpb), vec![3]);
    dpb.flush();
    assert_eq!(outputs(&mut dpb), vec![4, 8]);
}

/// `pic_output_flag` 0: the picture stays as a reference, and never leaves.
#[test]
fn a_picture_with_no_output_flag_is_never_output() {
    let mut dpb = Dpb::<u8>::new();
    store(&mut dpb, 0, false);
    store(&mut dpb, 1, true);
    dpb.flush();
    assert_eq!(outputs(&mut dpb), vec![1]);
    assert_eq!(dpb.core.entries.len(), 2);
}

/// C.5.2.2: the DPB size forces a picture out before the reorder limit does.
#[test]
fn a_full_buffer_bumps_before_the_next_picture() {
    let limits = DpbLimits {
        max_num_reorder: 4,
        max_latency: None,
        max_dec_buffering: 2,
    };
    let mut dpb = Dpb::<u8>::new();
    let first = frame(&mut dpb, 1);
    dpb.insert(first, true, &limits);
    let second = frame(&mut dpb, 2);
    dpb.insert(second, true, &limits);
    assert_eq!(outputs(&mut dpb), Vec::<i32>::new());
    dpb.apply_rps(&rps(3, &[], &[]), MAX_LSB, false);
    dpb.before_picture(false, false, &limits);
    assert_eq!(outputs(&mut dpb), vec![1]);
    assert_eq!(dpb.core.entries.len(), 1);
}

/// C.5.2.2: a full buffer with nothing to output stops the bumping loop.
#[test]
fn a_full_buffer_of_references_does_not_loop() {
    let limits = DpbLimits {
        max_num_reorder: 0,
        max_latency: None,
        max_dec_buffering: 1,
    };
    let mut dpb = Dpb::<u8>::new();
    let first = frame(&mut dpb, 1);
    dpb.insert(first, false, &limits);
    dpb.before_picture(false, false, &limits);
    assert_eq!(dpb.core.entries.len(), 1);
}

/// C.5.2.3: `PicLatencyCount` grows only for waiting pictures that follow the current
/// one in output order, and the limit forces them out.
#[test]
fn the_latency_limit_forces_output() {
    let limits = DpbLimits {
        max_num_reorder: 4,
        max_latency: Some(1),
        max_dec_buffering: 8,
    };
    let mut dpb = Dpb::<u8>::new();
    let later = frame(&mut dpb, 8);
    dpb.insert(later, true, &limits);
    assert_eq!(outputs(&mut dpb), Vec::<i32>::new());
    // POC 4 precedes POC 8, so the count of POC 8 reaches 1. The bump takes the
    // smallest POC first, which is POC 4, then POC 8.
    let earlier = frame(&mut dpb, 4);
    dpb.insert(earlier, true, &limits);
    assert_eq!(outputs(&mut dpb), vec![4, 8]);
}

/// C.5.2.2: an IRAP picture with `NoRaslOutputFlag` 1 outputs the waiting pictures
/// with `NoOutputOfPriorPicsFlag` 0, and drops them with 1.
#[test]
fn prior_pictures_follow_the_flag() {
    let mut dpb = Dpb::<u8>::new();
    store(&mut dpb, 5, true);
    store(&mut dpb, 6, true);
    dpb.before_picture(true, false, &LIMITS);
    assert_eq!(outputs(&mut dpb), vec![5, 6]);
    assert!(dpb.core.entries.is_empty());

    store(&mut dpb, 5, true);
    store(&mut dpb, 6, true);
    dpb.before_picture(true, true, &LIMITS);
    assert_eq!(outputs(&mut dpb), Vec::<i32>::new());
    assert!(dpb.core.entries.is_empty());
}

/// 8.3.3: a missing "Foll" entry gets a grey picture with no output and no motion.
#[test]
fn a_missing_foll_entry_gets_a_grey_picture() {
    let mut dpb = Dpb::<u8>::new();
    let sets = dpb.apply_rps(&rps(8, &[], &[-4]), MAX_LSB, true);
    dpb.generate_unavailable(&sets, &FORMAT).unwrap();
    assert_eq!(dpb.core.entries.len(), 1);
    let entry = &dpb.core.entries[0];
    assert_eq!(entry.frame.poc, 4);
    assert_eq!(entry.marking, Marking::ShortTerm);
    assert!(!entry.needed_for_output);
    assert_eq!(entry.frame.picture.luma().row(7).unwrap()[15], 128);
    assert_eq!(entry.frame.picture.cr().row(3).unwrap()[7], 128);
    let unit = entry.frame.col.at(0, 0).unwrap();
    assert!(!unit.motion.uses(0) && !unit.motion.uses(1));
    dpb.flush();
    assert_eq!(outputs(&mut dpb), Vec::<i32>::new());
}

/// A picture nothing else holds returns its buffer for the next picture.
#[test]
fn a_finished_buffer_is_reused() {
    let mut dpb = Dpb::<u8>::new();
    store(&mut dpb, 0, false);
    dpb.apply_rps(&rps(1, &[], &[]), MAX_LSB, false);
    dpb.before_picture(false, false, &LIMITS);
    assert!(dpb.core.entries.is_empty());
    assert_eq!(dpb.free.len(), 1);
    let _reused = dpb.take_frame(&FORMAT).unwrap();
    assert!(dpb.free.is_empty());
}

/// An output that a caller still holds keeps its samples: no buffer reuse takes it.
#[test]
fn a_held_output_is_not_reused() {
    let mut dpb = Dpb::<u8>::new();
    store(&mut dpb, 0, true);
    dpb.apply_rps(&rps(1, &[], &[]), MAX_LSB, false);
    dpb.flush();
    let held = dpb.pop_output();
    assert!(held.is_some());
    assert!(dpb.free.is_empty());
}

#[test]
fn the_limits_read_the_highest_sub_layer() {
    let mut sps = sps_tests::valid(&sps_tests::Spec::default());
    sps.sps_max_num_reorder_pics[0] = 2;
    sps.sps_max_latency_increase_plus1[0] = 3;
    sps.sps_max_dec_pic_buffering_minus1[0] = 4;
    // (7-9): 2 + 3 - 1.
    assert_eq!(
        DpbLimits::from_sps(&sps),
        DpbLimits {
            max_num_reorder: 2,
            max_latency: Some(4),
            max_dec_buffering: 5,
        }
    );
    sps.sps_max_latency_increase_plus1[0] = 0;
    assert_eq!(DpbLimits::from_sps(&sps).max_latency, None);
}
