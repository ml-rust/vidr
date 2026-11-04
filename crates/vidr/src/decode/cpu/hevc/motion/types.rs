//! The motion of one prediction block, and the reference facts of one slice.

use crate::bitstream::hevc_pred_weight::MAX_REFS;

/// `refIdxLX` of a list the block does not use.
pub(crate) const NO_REF: i8 = -1;

/// A luma motion vector in quarter samples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Mv {
    /// Horizontal component.
    pub(crate) x: i16,
    /// Vertical component.
    pub(crate) y: i16,
}

impl Mv {
    /// A vector from its components.
    pub(crate) const fn new(x: i16, y: i16) -> Self {
        Self { x, y }
    }
}

/// `MvLX`, `RefIdxLX`, and `PredFlagLX` of one prediction block. `PredFlagLX` is 1
/// when `ref_idx[ X ]` is not negative. An unused list holds a zero vector, so equal
/// motion compares equal with `==`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PbMotion {
    /// `mvL0` and `mvL1`.
    pub(crate) mv: [Mv; 2],
    /// `refIdxL0` and `refIdxL1`.
    pub(crate) ref_idx: [i8; 2],
}

impl PbMotion {
    /// The motion of an intra block: no list in use.
    pub(crate) const NONE: Self = Self {
        mv: [Mv::new(0, 0); 2],
        ref_idx: [NO_REF; 2],
    };

    /// `PredFlagLX` of `list` 0 or 1.
    pub(crate) const fn uses(&self, list: usize) -> bool {
        self.ref_idx[list] >= 0
    }

    /// `PredFlagL0 + PredFlagL1`.
    pub(crate) const fn vector_count(&self) -> u32 {
        self.uses(0) as u32 + self.uses(1) as u32
    }
}

impl Default for PbMotion {
    fn default() -> Self {
        Self::NONE
    }
}

/// What motion derivation reads from `RefPicList0` and `RefPicList1` of one slice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RefLists {
    /// `PicOrderCntVal` of each entry.
    pub(crate) poc: [[i32; MAX_REFS]; 2],
    /// True for an entry marked as "used for long-term reference" while this
    /// picture decodes.
    pub(crate) long_term: [[bool; MAX_REFS]; 2],
    /// `num_ref_idx_lX_active_minus1 + 1`, or 0 for a list the slice does not use.
    pub(crate) count: [usize; 2],
}

impl RefLists {
    /// `PicOrderCnt( RefPicListX[ refIdx ] )`. An index outside the list returns 0.
    pub(crate) fn poc_of(&self, list: usize, ref_idx: i8) -> i32 {
        self.poc[list]
            .get(ref_idx as usize)
            .copied()
            .unwrap_or_default()
    }

    /// True when `RefPicListX[ refIdx ]` is a long-term reference picture.
    pub(crate) fn is_long_term(&self, list: usize, ref_idx: i8) -> bool {
        self.long_term[list]
            .get(ref_idx as usize)
            .copied()
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_list_is_in_use_by_default() {
        let motion = PbMotion::default();
        assert!(!motion.uses(0));
        assert!(!motion.uses(1));
        assert_eq!(motion.vector_count(), 0);
    }

    #[test]
    fn the_count_follows_the_reference_indices() {
        let motion = PbMotion {
            mv: [Mv::new(1, 2), Mv::new(3, 4)],
            ref_idx: [2, 0],
        };
        assert_eq!(motion.vector_count(), 2);
        let single = PbMotion {
            ref_idx: [NO_REF, 1],
            ..motion
        };
        assert_eq!(single.vector_count(), 1);
    }

    #[test]
    fn an_index_outside_the_list_reads_nothing() {
        let mut refs = RefLists::default();
        refs.poc[1][14] = 9;
        refs.long_term[1][14] = true;
        assert_eq!(refs.poc_of(1, 14), 9);
        assert!(refs.is_long_term(1, 14));
        assert_eq!(refs.poc_of(1, NO_REF), 0);
        assert!(!refs.is_long_term(1, NO_REF));
        assert_eq!(refs.poc_of(0, 100), 0);
    }
}
