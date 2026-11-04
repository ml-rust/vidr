//! The set frame refs process (AV1 specification 7.8): `ref_frame_idx` from two coded names and
//! the order hints.

use super::types::{NUM_REF_FRAMES, REFS_PER_FRAME};
use crate::error::BitstreamError;

const STRUCTURE: &str = "AV1 frame header";

/// Index into `ref_frame_idx` of each named reference: the reference frame minus `LAST_FRAME`.
const LAST: usize = 0;
const LAST2: usize = 1;
const LAST3: usize = 2;
const GOLDEN: usize = 3;
const BWDREF: usize = 4;
const ALTREF2: usize = 5;
const ALTREF: usize = 6;

/// `Ref_Frame_List`, as indices into `ref_frame_idx`.
const REF_FRAME_LIST: [usize; REFS_PER_FRAME - 2] = [LAST2, LAST3, BWDREF, ALTREF2, ALTREF];

/// `get_relative_dist()` (5.9.3). With `OrderHintBits` 0, order hints are off and the result is 0.
pub(super) const fn relative_dist(a: i32, b: i32, order_hint_bits: u8) -> i32 {
    if order_hint_bits == 0 {
        return 0;
    }
    let diff = a - b;
    let m = 1 << (order_hint_bits - 1);
    (diff & (m - 1)) - (diff & m)
}

struct Search {
    shifted: [i32; NUM_REF_FRAMES],
    used: [bool; NUM_REF_FRAMES],
    current: i32,
}

impl Search {
    fn take(&mut self, target: &mut [Option<u8>; REFS_PER_FRAME], at: usize, found: Option<usize>) {
        if let Some(name) = found {
            target[at] = Some(name as u8);
            self.used[name] = true;
        }
    }

    /// `find_latest_backward()`.
    fn latest_backward(&self) -> Option<usize> {
        let mut best: Option<(usize, i32)> = None;
        for (name, &hint) in self.shifted.iter().enumerate() {
            let better = best.is_none_or(|(_, latest)| hint >= latest);
            if !self.used[name] && hint >= self.current && better {
                best = Some((name, hint));
            }
        }
        best.map(|(name, _)| name)
    }

    /// `find_earliest_backward()`.
    fn earliest_backward(&self) -> Option<usize> {
        let mut best: Option<(usize, i32)> = None;
        for (name, &hint) in self.shifted.iter().enumerate() {
            let better = best.is_none_or(|(_, earliest)| hint < earliest);
            if !self.used[name] && hint >= self.current && better {
                best = Some((name, hint));
            }
        }
        best.map(|(name, _)| name)
    }

    /// `find_latest_forward()`.
    fn latest_forward(&self) -> Option<usize> {
        let mut best: Option<(usize, i32)> = None;
        for (name, &hint) in self.shifted.iter().enumerate() {
            let better = best.is_none_or(|(_, latest)| hint >= latest);
            if !self.used[name] && hint < self.current && better {
                best = Some((name, hint));
            }
        }
        best.map(|(name, _)| name)
    }

    /// The name with the smallest output order, whether used or not.
    fn earliest(&self) -> u8 {
        let mut best = (0, self.shifted[0]);
        for (name, &hint) in self.shifted.iter().enumerate().skip(1) {
            if hint < best.1 {
                best = (name, hint);
            }
        }
        best.0 as u8
    }
}

/// Computes `ref_frame_idx`. `order_hint_bits` is 1 or more, because this syntax needs
/// `enable_order_hint`.
///
/// # Errors
///
/// [`BitstreamError::InvalidField`] when `last_frame_idx` or `gold_frame_idx` names a frame that
/// does not come before the current one, which section 7.8 forbids.
pub(super) fn set_frame_refs(
    last_frame_idx: u8,
    gold_frame_idx: u8,
    ref_order_hints: &[u8; NUM_REF_FRAMES],
    order_hint: u8,
    order_hint_bits: u8,
) -> Result<[u8; REFS_PER_FRAME], BitstreamError> {
    let current = 1 << (order_hint_bits - 1);
    let shifted = ref_order_hints.map(|hint| {
        current + relative_dist(i32::from(hint), i32::from(order_hint), order_hint_bits)
    });
    let (last, gold) = (
        usize::from(last_frame_idx & 7),
        usize::from(gold_frame_idx & 7),
    );
    if shifted[last] >= current {
        return Err(BitstreamError::InvalidField {
            structure: STRUCTURE,
            field: "last_frame_idx",
            value: u64::from(last_frame_idx),
            allowed: "a name whose frame comes before the current frame",
        });
    }
    if shifted[gold] >= current {
        return Err(BitstreamError::InvalidField {
            structure: STRUCTURE,
            field: "gold_frame_idx",
            value: u64::from(gold_frame_idx),
            allowed: "a name whose frame comes before the current frame",
        });
    }
    let mut chosen: [Option<u8>; REFS_PER_FRAME] = [None; REFS_PER_FRAME];
    chosen[LAST] = Some(last as u8);
    chosen[GOLDEN] = Some(gold as u8);
    let mut used = [false; NUM_REF_FRAMES];
    used[last] = true;
    used[gold] = true;
    let mut search = Search {
        shifted,
        used,
        current,
    };
    let found = search.latest_backward();
    search.take(&mut chosen, ALTREF, found);
    let found = search.earliest_backward();
    search.take(&mut chosen, BWDREF, found);
    let found = search.earliest_backward();
    search.take(&mut chosen, ALTREF2, found);
    for at in REF_FRAME_LIST {
        if chosen[at].is_none() {
            let found = search.latest_forward();
            search.take(&mut chosen, at, found);
        }
    }
    let earliest = search.earliest();
    Ok(chosen.map(|name| name.unwrap_or(earliest)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A worked case with 3 bits of order hint: the current frame is 4. Names hold 0, 1, 2, 3
    /// before it, and 5, 6, 7 after it, and name 7 repeats 2.
    ///
    /// - ALTREF is the latest backward: name 6 holds 7.
    /// - BWDREF is the earliest backward: name 4 holds 5.
    /// - ALTREF2 is the next backward: name 5 holds 6.
    /// - LAST2 and LAST3 are the latest forward left: name 7 holds 2 (a tie takes the higher
    ///   name), then name 2 holds 2.
    #[test]
    fn a_worked_case_with_backward_references_holds() {
        let hints = [0, 1, 2, 3, 5, 6, 7, 2];
        let refs = set_frame_refs(3, 0, &hints, 4, 4).expect("valid");
        assert_eq!(refs[LAST], 3);
        assert_eq!(refs[GOLDEN], 0);
        assert_eq!(refs[ALTREF], 6);
        assert_eq!(refs[BWDREF], 4);
        assert_eq!(refs[ALTREF2], 5);
        assert_eq!(refs[LAST2], 7);
        assert_eq!(refs[LAST3], 2);
    }

    /// With no backward frame every open entry comes from the forward search, and what stays
    /// open takes the earliest frame.
    #[test]
    fn a_forward_only_case_fills_from_the_latest() {
        let hints = [1, 1, 1, 1, 2, 2, 2, 3];
        let refs = set_frame_refs(7, 0, &hints, 4, 4).expect("valid");
        assert_eq!(refs[LAST], 7);
        assert_eq!(refs[GOLDEN], 0);
        // Latest forward with ties to the higher name: 6, 5, 4, 3, 2.
        assert_eq!(refs[LAST2], 6);
        assert_eq!(refs[LAST3], 5);
        assert_eq!(refs[BWDREF], 4);
        assert_eq!(refs[ALTREF2], 3);
        assert_eq!(refs[ALTREF], 2);
    }

    #[test]
    fn a_wrapping_hint_counts_as_before_the_current_frame() {
        // 3 bits of hint: 7 is one before 0.
        let hints = [7; 8];
        let refs = set_frame_refs(0, 1, &hints, 0, 3).expect("valid");
        assert_eq!(refs[LAST], 0);
        assert_eq!(refs[GOLDEN], 1);
    }

    #[test]
    fn a_coded_name_after_the_current_frame_is_refused_by_name() {
        let hints = [6; 8];
        let error = set_frame_refs(0, 1, &hints, 4, 4).expect_err("refused");
        assert!(error.to_string().contains("last_frame_idx"), "{error}");
        let hints = [1, 6, 1, 1, 1, 1, 1, 1];
        let error = set_frame_refs(0, 1, &hints, 4, 4).expect_err("refused");
        assert!(error.to_string().contains("gold_frame_idx"), "{error}");
    }

    #[test]
    fn relative_distance_wraps_by_the_hint_bits() {
        assert_eq!(relative_dist(7, 0, 3), -1);
        assert_eq!(relative_dist(0, 7, 3), 1);
        assert_eq!(relative_dist(3, 1, 3), 2);
        assert_eq!(relative_dist(3, 1, 0), 0);
    }
}
