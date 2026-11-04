//! Temporal luma motion vector prediction (8.5.3.2.8) and collocated motion vectors
//! (8.5.3.2.9).

use super::col::{ColMotion, ColUnit};
use super::scale::scale_mv;
use super::types::{Mv, RefLists};

/// `ColPic` and the slice values 8.5.3.2.8 reads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TemporalContext<'a> {
    /// The motion of `ColPic`. `None` with `slice_temporal_mvp_enabled_flag` 0.
    pub(crate) col: Option<&'a ColMotion>,
    /// `PicOrderCnt( ColPic )`.
    pub(crate) col_poc: i32,
    /// `NoBackwardPredFlag`.
    pub(crate) no_backward_pred: bool,
    /// `collocated_from_l0_flag`.
    pub(crate) collocated_from_l0: bool,
    /// `CtbLog2SizeY`.
    pub(crate) ctb_log2: u32,
    /// `pic_width_in_luma_samples`.
    pub(crate) width: u32,
    /// `pic_height_in_luma_samples`.
    pub(crate) height: u32,
    /// `PicOrderCntVal` of the current picture.
    pub(crate) poc: i32,
}

/// A prediction block as 8.5.3.2.8 takes it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TemporalBlock {
    /// `xPb`.
    pub(crate) x: u32,
    /// `yPb`.
    pub(crate) y: u32,
    /// `nPbW`.
    pub(crate) width: u32,
    /// `nPbH`.
    pub(crate) height: u32,
}

impl TemporalContext<'_> {
    /// `mvLXCol`, or `None` when `availableFlagLXCol` is 0. `refs` holds the current
    /// slice's lists, and `ref_idx` is `refIdxLX`.
    pub(crate) fn predict(
        &self,
        refs: &RefLists,
        block: TemporalBlock,
        list: usize,
        ref_idx: i8,
    ) -> Option<Mv> {
        let col = self.col?;
        let x_br = block.x + block.width;
        let y_br = block.y + block.height;
        // The bottom right candidate stays in the current CTB row and the picture.
        if block.y >> self.ctb_log2 == y_br >> self.ctb_log2
            && y_br < self.height
            && x_br < self.width
            && let Some(mv) = self.collocated(col.at(x_br, y_br), refs, list, ref_idx)
        {
            return Some(mv);
        }
        let x_center = block.x + (block.width >> 1);
        let y_center = block.y + (block.height >> 1);
        self.collocated(col.at(x_center, y_center), refs, list, ref_idx)
    }

    /// 8.5.3.2.9 for one collocated block. A unit outside the collocated picture
    /// occurs only in a corrupt stream, and gives no candidate.
    fn collocated(
        &self,
        unit: Option<ColUnit>,
        refs: &RefLists,
        list: usize,
        ref_idx: i8,
    ) -> Option<Mv> {
        let unit = unit?;
        let list_col = match (unit.motion.uses(0), unit.motion.uses(1)) {
            (false, false) => return None,
            (false, true) => 1,
            (true, false) => 0,
            (true, true) if self.no_backward_pred => list,
            // listCol is LN, with N the value of collocated_from_l0_flag.
            (true, true) => usize::from(self.collocated_from_l0),
        };
        let long_term = refs.is_long_term(list, ref_idx);
        if long_term != unit.long_term[list_col] {
            return None;
        }
        let mv_col = unit.motion.mv[list_col];
        let col_diff = self.col_poc.wrapping_sub(unit.ref_poc[list_col]);
        let curr_diff = self.poc.wrapping_sub(refs.poc_of(list, ref_idx));
        if long_term || col_diff == curr_diff {
            Some(mv_col)
        } else {
            Some(scale_mv(mv_col, col_diff, curr_diff))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::cpu::hevc::motion::types::PbMotion;

    /// Current POC 8 in a 64x64 picture with 16x16 CTBs. Reference index 0 of each
    /// list has POC 4.
    fn refs() -> RefLists {
        let mut refs = RefLists::default();
        refs.poc[0][0] = 4;
        refs.poc[1][0] = 4;
        refs.count = [1, 1];
        refs
    }

    fn context(col: &ColMotion) -> TemporalContext<'_> {
        TemporalContext {
            col: Some(col),
            col_poc: 4,
            no_backward_pred: true,
            collocated_from_l0: true,
            ctb_log2: 4,
            width: 64,
            height: 64,
            poc: 8,
        }
    }

    fn unit(mv_l0: Mv, mv_l1: Mv, ref_idx: [i8; 2], ref_poc: [i32; 2]) -> ColUnit {
        ColUnit {
            motion: PbMotion {
                mv: [mv_l0, mv_l1],
                ref_idx,
            },
            ref_poc,
            long_term: [false; 2],
        }
    }

    fn col_with(at: impl Fn(u32, u32) -> ColUnit) -> ColMotion {
        let mut col = ColMotion::new();
        col.prepare(64, 64).unwrap();
        col.fill_with(at);
        col
    }

    const BLOCK: TemporalBlock = TemporalBlock {
        x: 16,
        y: 16,
        width: 8,
        height: 8,
    };

    /// colPocDiff 4 - 2 = 2, currPocDiff 8 - 4 = 4. td 2, tb 4: tx 8192,
    /// factor ( 32768 + 32 ) >> 6 = 512. 512 * 8 = 4096, ( 4096 + 127 ) >> 8 = 16.
    #[test]
    fn a_bottom_right_candidate_scales_by_distance() {
        let col = col_with(|_, _| unit(Mv::new(8, -8), Mv::default(), [0, -1], [2, 0]));
        assert_eq!(
            context(&col).predict(&refs(), BLOCK, 0, 0),
            Some(Mv::new(16, -16))
        );
    }

    /// (8-204): equal distances take the vector as it is.
    #[test]
    fn equal_distances_keep_the_vector() {
        let col = col_with(|_, _| unit(Mv::new(3, 5), Mv::default(), [0, -1], [0, 0]));
        assert_eq!(
            context(&col).predict(&refs(), BLOCK, 0, 0),
            Some(Mv::new(3, 5))
        );
    }

    /// An intra bottom right unit falls back to the central one. Bottom right is
    /// ( 24, 24 ) in the unit at ( 16, 16 ). Center is ( 20, 20 ) in the same unit,
    /// so this block takes a width that reaches the next unit.
    #[test]
    fn an_intra_bottom_right_falls_back_to_the_center() {
        let col = col_with(|x, _| {
            if x == 32 {
                ColUnit::default()
            } else {
                unit(Mv::new(3, 5), Mv::default(), [0, -1], [0, 0])
            }
        });
        let block = TemporalBlock {
            x: 16,
            y: 16,
            width: 16,
            height: 8,
        };
        assert_eq!(
            context(&col).predict(&refs(), block, 0, 0),
            Some(Mv::new(3, 5))
        );
    }

    /// yColBr falls in the next CTB row, so only the center counts.
    #[test]
    fn the_bottom_right_candidate_stays_in_the_ctb_row() {
        let col = col_with(|_, y| {
            if y == 32 {
                unit(Mv::new(9, 9), Mv::default(), [0, -1], [0, 0])
            } else {
                unit(Mv::new(1, 1), Mv::default(), [0, -1], [0, 0])
            }
        });
        let block = TemporalBlock {
            x: 16,
            y: 16,
            width: 16,
            height: 16,
        };
        assert_eq!(
            context(&col).predict(&refs(), block, 0, 0),
            Some(Mv::new(1, 1))
        );
    }

    /// With `NoBackwardPredFlag` 1 a bi-predicted unit gives the list being derived.
    #[test]
    fn no_backward_prediction_takes_the_same_list() {
        let col = col_with(|_, _| unit(Mv::new(1, 0), Mv::new(2, 0), [0, 0], [0, 0]));
        let context = context(&col);
        assert_eq!(context.predict(&refs(), BLOCK, 0, 0), Some(Mv::new(1, 0)));
        assert_eq!(context.predict(&refs(), BLOCK, 1, 0), Some(Mv::new(2, 0)));
    }

    /// Otherwise a bi-predicted unit gives LN, with N the flag's value.
    #[test]
    fn backward_prediction_follows_collocated_from_l0() {
        let col = col_with(|_, _| unit(Mv::new(1, 0), Mv::new(2, 0), [0, 0], [0, 0]));
        let mut context = context(&col);
        context.no_backward_pred = false;
        assert_eq!(context.predict(&refs(), BLOCK, 0, 0), Some(Mv::new(2, 0)));
        context.collocated_from_l0 = false;
        assert_eq!(context.predict(&refs(), BLOCK, 1, 0), Some(Mv::new(1, 0)));
    }

    /// A unit that uses only L1 gives L1, whichever list is being derived.
    #[test]
    fn a_unit_with_one_list_gives_that_list() {
        let col = col_with(|_, _| unit(Mv::default(), Mv::new(2, 0), [-1, 0], [0, 0]));
        assert_eq!(
            context(&col).predict(&refs(), BLOCK, 0, 0),
            Some(Mv::new(2, 0))
        );
    }

    #[test]
    fn a_long_term_mismatch_gives_no_candidate() {
        let mut stored = unit(Mv::new(3, 5), Mv::default(), [0, -1], [0, 0]);
        stored.long_term[0] = true;
        let col = col_with(|_, _| stored);
        assert_eq!(context(&col).predict(&refs(), BLOCK, 0, 0), None);
    }

    /// (8-204): a long-term reference takes the vector with no scaling.
    #[test]
    fn a_long_term_match_keeps_the_vector() {
        let mut stored = unit(Mv::new(3, 5), Mv::default(), [0, -1], [2, 0]);
        stored.long_term[0] = true;
        let col = col_with(|_, _| stored);
        let mut lists = refs();
        lists.long_term[0][0] = true;
        assert_eq!(
            context(&col).predict(&lists, BLOCK, 0, 0),
            Some(Mv::new(3, 5))
        );
    }

    #[test]
    fn no_collocated_picture_gives_no_candidate() {
        let col = ColMotion::new();
        let mut context = context(&col);
        context.col = None;
        assert_eq!(context.predict(&refs(), BLOCK, 0, 0), None);
    }
}
