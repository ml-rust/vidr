//! Luma motion vectors for merge mode (8.5.3.2.2 to 8.5.3.2.5).

use crate::decode::cpu::hevc::params::SliceKind;
use crate::decode::cpu::hevc::syntax::records::PartMode;

use super::context::{MotionContext, PbGeometry};
use super::temporal::TemporalBlock;
use super::types::{Mv, NO_REF, PbMotion};

/// The most candidates (8-119) gives: four spatial, and one temporal.
const MAX_LIST: usize = 5;
/// Table 8-7: `l0CandIdx` and `l1CandIdx` by `combIdx`.
const COMBINATIONS: [(usize, usize); 12] = [
    (0, 1),
    (1, 0),
    (0, 2),
    (2, 0),
    (1, 2),
    (2, 1),
    (0, 3),
    (3, 0),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
];

/// A candidate list that holds at most `N` entries while it grows.
struct CandidateList<const N: usize> {
    entries: [PbMotion; N],
    count: usize,
}

/// `mergeCandList` while it grows.
type MergeList = CandidateList<MAX_LIST>;

/// The combined candidates added to one list. At most four fit: the list holds at
/// least two before them, and at most five after them.
type Extended = CandidateList<4>;

impl<const N: usize> CandidateList<N> {
    const fn new() -> Self {
        Self {
            entries: [PbMotion::NONE; N],
            count: 0,
        }
    }

    fn push(&mut self, candidate: PbMotion) {
        if let Some(slot) = self.entries.get_mut(self.count) {
            *slot = candidate;
            self.count += 1;
        }
    }

    fn get(&self, index: usize) -> PbMotion {
        self.entries.get(index).copied().unwrap_or_default()
    }
}

impl MotionContext<'_> {
    /// The motion for `merge_idx` (8.5.3.2.2).
    pub(crate) fn merge_motion(&self, origin: &PbGeometry, merge_idx: u32) -> PbMotion {
        let pb = self.merge_block(origin);
        let mut list = MergeList::new();
        self.spatial_candidates(&pb, &mut list);
        let wanted = merge_idx as usize;
        if wanted >= list.count {
            if let Some(temporal) = self.temporal_candidate(&pb) {
                list.push(temporal);
            }
            if wanted >= list.count {
                let original = list.count;
                let is_b = self.params.kind == SliceKind::B;
                let extended = if is_b {
                    self.combined_candidates(&list, original)
                } else {
                    Extended::new()
                };
                return self.select_extended(&extended, wanted - original, origin);
            }
        }
        restrict_small_block(list.get(wanted), origin)
    }

    /// (8-110) to (8-113): a coding unit of 8 shares one list above a parallel merge
    /// level of 2.
    fn merge_block(&self, origin: &PbGeometry) -> PbGeometry {
        if self.params.log2_par_mrg_level > 2 && origin.n_cb == 8 {
            PbGeometry {
                x_pb: origin.x_cb,
                y_pb: origin.y_cb,
                width: origin.n_cb,
                height: origin.n_cb,
                part_idx: 0,
                ..*origin
            }
        } else {
            *origin
        }
    }

    /// True when a neighbour lies in the parallel merge region of the block.
    fn same_merge_region(&self, pb: &PbGeometry, x_nb: i32, y_nb: i32) -> bool {
        let level = self.params.log2_par_mrg_level;
        (pb.x_pb as i32) >> level == x_nb >> level && (pb.y_pb as i32) >> level == y_nb >> level
    }

    /// 6.4.2, then the parallel merge check of 8.5.3.2.3.
    fn merge_neighbour(&self, pb: &PbGeometry, x_nb: i32, y_nb: i32) -> Option<PbMotion> {
        if self.same_merge_region(pb, x_nb, y_nb) {
            return None;
        }
        self.neighbour(pb, x_nb, y_nb)
    }

    /// 8.5.3.2.3. The variables named `availableN` stay set for a pruned candidate,
    /// so later comparisons read them, while `availableFlagN` drops it.
    fn spatial_candidates(&self, pb: &PbGeometry, list: &mut MergeList) {
        let (x, y) = (pb.x_pb as i32, pb.y_pb as i32);
        let (width, height) = (pb.width as i32, pb.height as i32);

        let second_of_vertical_pair = pb.part_idx == 1
            && matches!(
                pb.part_mode,
                PartMode::NxTwoN | PartMode::NLxTwoN | PartMode::NRxTwoN
            );
        let a1 = if second_of_vertical_pair {
            None
        } else {
            self.merge_neighbour(pb, x - 1, y + height - 1)
        };

        let second_of_horizontal_pair = pb.part_idx == 1
            && matches!(
                pb.part_mode,
                PartMode::TwoNxN | PartMode::TwoNxnU | PartMode::TwoNxnD
            );
        let b1 = if second_of_horizontal_pair {
            None
        } else {
            self.merge_neighbour(pb, x + width - 1, y - 1)
        };
        let flag_b1 = b1.filter(|motion| a1 != Some(*motion));

        let b0 = self.merge_neighbour(pb, x + width, y - 1);
        let flag_b0 = b0.filter(|motion| b1 != Some(*motion));

        let a0 = self.merge_neighbour(pb, x - 1, y + height);
        let flag_a0 = a0.filter(|motion| a1 != Some(*motion));

        let four_found =
            a1.is_some() && flag_b1.is_some() && flag_b0.is_some() && flag_a0.is_some();
        let b2 = self.merge_neighbour(pb, x - 1, y - 1);
        let flag_b2 = b2.filter(|motion| !four_found && a1 != Some(*motion) && b1 != Some(*motion));

        // (8-119).
        for candidate in [a1, flag_b1, flag_b0, flag_a0, flag_b2]
            .into_iter()
            .flatten()
        {
            list.push(candidate);
        }
    }

    /// Steps 2 to 4 of 8.5.3.2.2: `refIdxLXCol` is 0.
    fn temporal_candidate(&self, pb: &PbGeometry) -> Option<PbMotion> {
        let block = TemporalBlock {
            x: pb.x_pb,
            y: pb.y_pb,
            width: pb.width,
            height: pb.height,
        };
        let l0 = self.temporal.predict(self.refs, block, 0, 0);
        let l1 = if self.params.kind == SliceKind::B {
            self.temporal.predict(self.refs, block, 1, 0)
        } else {
            None
        };
        if l0.is_none() && l1.is_none() {
            return None;
        }
        Some(PbMotion {
            mv: [l0.unwrap_or_default(), l1.unwrap_or_default()],
            ref_idx: [
                if l0.is_some() { 0 } else { NO_REF },
                if l1.is_some() { 0 } else { NO_REF },
            ],
        })
    }

    /// 8.5.3.2.4. `original` is `numOrigMergeCand`.
    fn combined_candidates(&self, list: &MergeList, original: usize) -> Extended {
        let mut extended = Extended::new();
        let max = self.params.max_merge_cand as usize;
        if original <= 1 || original >= max {
            return extended;
        }
        let limit = original * (original - 1);
        for &(l0_index, l1_index) in COMBINATIONS.iter().take(limit) {
            let l0_candidate = list.get(l0_index);
            let l1_candidate = list.get(l1_index);
            if l0_candidate.uses(0) && l1_candidate.uses(1) {
                let different_pictures = self.refs.poc_of(0, l0_candidate.ref_idx[0])
                    != self.refs.poc_of(1, l1_candidate.ref_idx[1]);
                if different_pictures || l0_candidate.mv[0] != l1_candidate.mv[1] {
                    extended.push(PbMotion {
                        mv: [l0_candidate.mv[0], l1_candidate.mv[1]],
                        ref_idx: [l0_candidate.ref_idx[0], l1_candidate.ref_idx[1]],
                    });
                    if original + extended.count == max {
                        break;
                    }
                }
            }
        }
        extended
    }

    /// The candidate `past_original` places after the original list: a combined one,
    /// or one of 8.5.3.2.5.
    fn select_extended(
        &self,
        extended: &Extended,
        past_original: usize,
        origin: &PbGeometry,
    ) -> PbMotion {
        let candidate = if past_original < extended.count {
            extended.get(past_original)
        } else {
            self.zero_candidate(past_original - extended.count)
        };
        restrict_small_block(candidate, origin)
    }

    /// `zeroCandm` with `zeroIdx` equal to `zero_idx` (8.5.3.2.5).
    fn zero_candidate(&self, zero_idx: usize) -> PbMotion {
        let [l0_minus1, l1_minus1] = self.params.num_ref_idx_minus1;
        let is_b = self.params.kind == SliceKind::B;
        let last = if is_b {
            l0_minus1.min(l1_minus1)
        } else {
            l0_minus1
        };
        // numRefIdx is last + 1.
        let ref_idx = if zero_idx <= last as usize {
            zero_idx as i8
        } else {
            0
        };
        PbMotion {
            mv: [Mv::default(); 2],
            ref_idx: [ref_idx, if is_b { ref_idx } else { NO_REF }],
        }
    }
}

/// Step 10 of 8.5.3.2.2: an 8x4 or 4x8 block predicts from L0 alone. The zero vector
/// keeps unused lists equal under `==`.
fn restrict_small_block(mut motion: PbMotion, origin: &PbGeometry) -> PbMotion {
    if motion.uses(0) && motion.uses(1) && origin.width + origin.height == 12 {
        motion.ref_idx[1] = NO_REF;
        motion.mv[1] = Mv::default();
    }
    motion
}

#[cfg(test)]
#[path = "merge_tests.rs"]
mod tests;
