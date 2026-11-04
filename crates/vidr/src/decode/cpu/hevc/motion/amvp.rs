//! Luma motion vector prediction (8.5.3.2.6, 8.5.3.2.7).

use super::context::{MotionContext, PbGeometry};
use super::scale::scale_mv;
use super::temporal::TemporalBlock;
use super::types::{Mv, PbMotion};

/// What a candidate vector is compared with.
#[derive(Clone, Copy, Debug)]
struct Target {
    /// `X`.
    list: usize,
    /// `PicOrderCnt( RefPicListX[ refIdxLX ] )`.
    poc: i32,
    /// `LongTermRefPic( currPic, currPb, refIdxLX, RefPicListX )`.
    long_term: bool,
}

impl MotionContext<'_> {
    /// `mvpLX` (8.5.3.2.6).
    pub(crate) fn mv_predictor(
        &self,
        pb: &PbGeometry,
        list: usize,
        ref_idx: i8,
        mvp_flag: bool,
    ) -> Mv {
        let target = Target {
            list,
            poc: self.refs.poc_of(list, ref_idx),
            long_term: self.refs.is_long_term(list, ref_idx),
        };
        let (a, b) = self.spatial_predictors(pb, target);
        // (8-170). The temporal candidate counts only while the list has room.
        let mut candidates = [Mv::default(); 2];
        let mut count = 0;
        for candidate in [a, b.filter(|mv| a != Some(*mv))].into_iter().flatten() {
            candidates[count] = candidate;
            count += 1;
        }
        let wanted = usize::from(mvp_flag);
        if count <= wanted && count < 2 {
            let block = TemporalBlock {
                x: pb.x_pb,
                y: pb.y_pb,
                width: pb.width,
                height: pb.height,
            };
            if let Some(temporal) = self.temporal.predict(self.refs, block, list, ref_idx) {
                candidates[count] = temporal;
            }
        }
        candidates[wanted]
    }

    /// `mvLXA` and `mvLXB` (8.5.3.2.7), each `None` when its flag is 0.
    fn spatial_predictors(&self, pb: &PbGeometry, target: Target) -> (Option<Mv>, Option<Mv>) {
        let (x, y) = (pb.x_pb as i32, pb.y_pb as i32);
        let (width, height) = (pb.width as i32, pb.height as i32);
        let a0 = self.neighbour(pb, x - 1, y + height);
        let a1 = self.neighbour(pb, x - 1, y + height - 1);
        // isScaledFlagLX.
        let is_scaled = a0.is_some() || a1.is_some();
        let same_a = [a0, a1]
            .into_iter()
            .flatten()
            .find_map(|motion| self.same_picture_mv(&motion, target));
        let a = same_a.or_else(|| {
            [a0, a1]
                .into_iter()
                .flatten()
                .find_map(|motion| self.scaled_mv(&motion, target))
        });

        let b_neighbours = [
            self.neighbour(pb, x + width, y - 1),
            self.neighbour(pb, x + width - 1, y - 1),
            self.neighbour(pb, x - 1, y - 1),
        ];
        let same_b = b_neighbours
            .into_iter()
            .flatten()
            .find_map(|motion| self.same_picture_mv(&motion, target));
        if is_scaled {
            return (a, same_b);
        }
        // Steps 4 and 5: with no A neighbour, the unscaled B takes the place of A, and
        // B runs again with scaling.
        let scaled_b = b_neighbours
            .into_iter()
            .flatten()
            .find_map(|motion| self.scaled_mv(&motion, target));
        (same_b, scaled_b)
    }

    /// (8-171), (8-172), (8-184), (8-185): a vector of this block that points at the
    /// target picture, list X first.
    fn same_picture_mv(&self, motion: &PbMotion, target: Target) -> Option<Mv> {
        let other = 1 - target.list;
        [target.list, other].into_iter().find_map(|list| {
            (motion.uses(list) && self.refs.poc_of(list, motion.ref_idx[list]) == target.poc)
                .then(|| motion.mv[list])
        })
    }

    /// (8-173) to (8-183), (8-187) to (8-197): a vector of this block whose long-term
    /// state matches the target, list X first, scaled between short-term pictures.
    fn scaled_mv(&self, motion: &PbMotion, target: Target) -> Option<Mv> {
        let other = 1 - target.list;
        [target.list, other].into_iter().find_map(|list| {
            if !motion.uses(list) {
                return None;
            }
            let ref_idx = motion.ref_idx[list];
            if self.refs.is_long_term(list, ref_idx) != target.long_term {
                return None;
            }
            let poc = self.refs.poc_of(list, ref_idx);
            let mv = motion.mv[list];
            if poc == target.poc || target.long_term {
                return Some(mv);
            }
            Some(scale_mv(
                mv,
                self.poc.wrapping_sub(poc),
                self.poc.wrapping_sub(target.poc),
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::cpu::hevc::motion::col::ColMotion;
    use crate::decode::cpu::hevc::motion::field::MotionField;
    use crate::decode::cpu::hevc::motion::test_support::{Scene, inter, inter_l1, pb};
    use crate::decode::cpu::hevc::motion::types::RefLists;

    const CURRENT: PbGeometry = pb(16, 16, 16, 16);

    /// A1 at ( 15, 31 ) with L0 index 0, and B1 at ( 31, 15 ) with L0 index 0: both
    /// point at the target, POC 4.
    #[test]
    fn matching_neighbours_fill_the_list_in_order() {
        let mut scene = Scene::new();
        scene.set_inter(0, 16, 16, 16, inter(Mv::new(1, 1), 0));
        scene.set_inter(16, 0, 16, 16, inter(Mv::new(2, 2), 0));
        let context = scene.context();
        assert_eq!(context.mv_predictor(&CURRENT, 0, 0, false), Mv::new(1, 1));
        assert_eq!(context.mv_predictor(&CURRENT, 0, 0, true), Mv::new(2, 2));
    }

    /// (8-170): equal A and B count once, and zero fills the second place.
    #[test]
    fn equal_candidates_count_once() {
        let mut scene = Scene::new();
        scene.set_inter(0, 16, 16, 16, inter(Mv::new(1, 1), 0));
        scene.set_inter(16, 0, 16, 16, inter(Mv::new(1, 1), 0));
        let context = scene.context();
        assert_eq!(context.mv_predictor(&CURRENT, 0, 0, false), Mv::new(1, 1));
        assert_eq!(context.mv_predictor(&CURRENT, 0, 0, true), Mv::default());
    }

    /// No neighbour and no temporal picture: both places hold zero.
    #[test]
    fn no_candidates_give_zero() {
        let scene = Scene::new();
        let context = scene.context();
        assert_eq!(context.mv_predictor(&CURRENT, 0, 0, false), Mv::default());
        assert_eq!(context.mv_predictor(&CURRENT, 0, 0, true), Mv::default());
    }

    /// (8-172): the other list of a neighbour counts when it points at the target.
    /// L1 index 1 is POC 16, and L0 of the target is set to POC 16 too.
    #[test]
    fn the_other_list_of_a_neighbour_counts() {
        let mut scene = Scene::new();
        scene.refs.poc[0][0] = 16;
        scene.set_inter(0, 16, 16, 16, inter_l1(Mv::new(7, 7), 1));
        let context = scene.context();
        assert_eq!(context.mv_predictor(&CURRENT, 0, 0, false), Mv::new(7, 7));
    }

    /// A neighbour at another picture scales. Current POC 8, target POC 4 gives tb 4.
    /// The neighbour's L0 index 1 is POC 2, which gives td 6. tx = 16387 / 6 = 2731.
    /// Factor ( 10924 + 32 ) >> 6 = 171. 171 * 12 = 2052, ( 2052 + 127 ) >> 8 = 8.
    #[test]
    fn a_neighbour_at_another_picture_scales() {
        let mut scene = Scene::new();
        scene.set_inter(0, 16, 16, 16, inter(Mv::new(12, -12), 1));
        let context = scene.context();
        assert_eq!(context.mv_predictor(&CURRENT, 0, 0, false), Mv::new(8, -8));
    }

    /// Step 6 runs over A0 and A1 before step 7: a later match beats an earlier
    /// vector that needs scaling. A0 stays out of reach here, so A1 alone proves
    /// the order with its two lists: L1 points at the target, and L0 does not.
    #[test]
    fn an_exact_match_beats_a_scaled_vector() {
        let mut scene = Scene::new();
        scene.refs.poc[1][0] = 4;
        scene.set_inter(
            0,
            16,
            16,
            16,
            PbMotion {
                mv: [Mv::new(20, 0), Mv::new(3, 3)],
                ref_idx: [1, 0],
            },
        );
        let context = scene.context();
        assert_eq!(context.mv_predictor(&CURRENT, 0, 0, false), Mv::new(3, 3));
    }

    /// Steps 4 and 5: with no A neighbour the unscaled B takes place 0. B then runs
    /// again with scaling, and the same exact match comes first, so B equals A and
    /// counts once.
    #[test]
    fn with_no_a_neighbour_b_fills_a() {
        let mut scene = Scene::new();
        scene.set_inter(16, 0, 16, 16, inter(Mv::new(5, 5), 0));
        let context = scene.context();
        assert_eq!(context.mv_predictor(&CURRENT, 0, 0, false), Mv::new(5, 5));
        assert_eq!(context.mv_predictor(&CURRENT, 0, 0, true), Mv::default());
    }

    /// With no A neighbour and no exact B, A stays empty, and B scales.
    /// The B1 vector points at POC 2. Target POC 4: td 6, tb 4, factor 171.
    #[test]
    fn with_no_a_neighbour_a_scaled_b_takes_the_first_place() {
        let mut scene = Scene::new();
        scene.set_inter(16, 0, 16, 16, inter(Mv::new(12, 0), 1));
        let context = scene.context();
        assert_eq!(context.mv_predictor(&CURRENT, 0, 0, false), Mv::new(8, 0));
    }

    /// With an A neighbour that is available but gives nothing, isScaledFlagLX is 1
    /// and B takes no scaling. A long-term target refuses the short-term neighbours.
    #[test]
    fn a_long_term_target_refuses_short_term_neighbours() {
        let mut scene = Scene::new();
        scene.refs.long_term[0][0] = true;
        scene.set_inter(0, 16, 16, 16, inter(Mv::new(12, 0), 1));
        scene.set_inter(16, 0, 16, 16, inter(Mv::new(9, 0), 1));
        let context = scene.context();
        assert_eq!(context.mv_predictor(&CURRENT, 0, 0, false), Mv::default());
    }

    /// (8.5.3.2.6, step 2): the temporal candidate fills a free place after A. The
    /// collocated vector has equal distances: POC 6 to POC 2, and POC 8 to POC 4.
    #[test]
    fn the_temporal_candidate_fills_a_free_place() {
        let mut scene = Scene::new();
        scene.set_inter(0, 16, 16, 16, inter(Mv::new(1, 1), 0));
        let mut field = MotionField::new(64, 64, &scene.layout).unwrap();
        let mut refs = RefLists::default();
        refs.poc[0][0] = 2;
        field.begin_slice(refs).unwrap();
        for rs in 0..16 {
            field.set_ctb(rs);
        }
        field.fill(0, 0, 64, 64, inter(Mv::new(6, 6), 0));
        let mut col = ColMotion::new();
        field.compress(64, 64, &mut col).unwrap();
        scene.col = Some(col);
        scene.col_poc = 6;
        let context = scene.context();
        assert_eq!(context.mv_predictor(&CURRENT, 0, 0, false), Mv::new(1, 1));
        assert_eq!(context.mv_predictor(&CURRENT, 0, 0, true), Mv::new(6, 6));
    }
}
