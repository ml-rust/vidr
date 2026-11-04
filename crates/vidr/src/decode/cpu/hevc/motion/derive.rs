//! The motion of one prediction block (8.5.3.2.1).

use crate::decode::cpu::hevc::syntax::records::{InterPredIdc, PuRecord};

use super::context::{MotionContext, PbGeometry};
use super::types::{Mv, PbMotion};

impl MotionContext<'_> {
    /// `mvL0`, `mvL1`, `refIdxL0`, `refIdxL1`, and the flags of one parsed unit.
    pub(crate) fn pb_motion(&self, pb: &PbGeometry, pu: &PuRecord) -> PbMotion {
        if pu.merge_flag {
            return self.merge_motion(pb, pu.merge_idx);
        }
        let mut motion = PbMotion::NONE;
        for list in 0..2 {
            let used = match pu.inter_pred_idc {
                InterPredIdc::L0 => list == 0,
                InterPredIdc::L1 => list == 1,
                InterPredIdc::Bi => true,
            };
            if !used {
                continue;
            }
            // The parser holds ref_idx at most 14.
            let ref_idx = pu.ref_idx[list].min(i8::MAX as u32) as i8;
            let predictor = self.mv_predictor(pb, list, ref_idx, pu.mvp_flag[list]);
            motion.ref_idx[list] = ref_idx;
            motion.mv[list] = add_mvd(predictor, pu.mvd[list]);
        }
        motion
    }
}

/// (8-94) to (8-97): the sum wraps to 16 bits.
fn add_mvd(predictor: Mv, mvd: [i32; 2]) -> Mv {
    Mv::new(
        i32::from(predictor.x).wrapping_add(mvd[0]) as i16,
        i32::from(predictor.y).wrapping_add(mvd[1]) as i16,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::cpu::hevc::motion::test_support::{Scene, inter, pb};
    use crate::decode::cpu::hevc::motion::types::NO_REF;

    fn explicit(idc: InterPredIdc) -> PuRecord {
        PuRecord {
            x: 16,
            y: 16,
            width: 16,
            height: 16,
            merge_flag: false,
            merge_idx: 0,
            inter_pred_idc: idc,
            ref_idx: [0, 1],
            mvd: [[3, -3], [10, 20]],
            mvp_flag: [false, false],
        }
    }

    #[test]
    fn a_vector_wraps_at_16_bits() {
        assert_eq!(add_mvd(Mv::new(32_767, 0), [1, 0]), Mv::new(-32_768, 0));
        assert_eq!(add_mvd(Mv::new(-32_768, 5), [-1, -5]), Mv::new(32_767, 0));
    }

    /// (8-90), (8-91): a list out of `inter_pred_idc` stays unused.
    #[test]
    fn one_list_leaves_the_other_unused() {
        let scene = Scene::new();
        let motion = scene
            .context()
            .pb_motion(&pb(16, 16, 16, 16), &explicit(InterPredIdc::L1));
        assert_eq!(motion.ref_idx, [NO_REF, 1]);
        assert_eq!(motion.mv, [Mv::default(), Mv::new(10, 20)]);
    }

    #[test]
    fn two_lists_each_take_their_own_values() {
        let scene = Scene::new();
        let motion = scene
            .context()
            .pb_motion(&pb(16, 16, 16, 16), &explicit(InterPredIdc::Bi));
        assert_eq!(motion.ref_idx, [0, 1]);
        assert_eq!(motion.mv, [Mv::new(3, -3), Mv::new(10, 20)]);
    }

    #[test]
    fn the_predictor_adds_to_the_difference() {
        let mut scene = Scene::new();
        scene.set_inter(0, 16, 16, 16, inter(Mv::new(100, 200), 0));
        let motion = scene
            .context()
            .pb_motion(&pb(16, 16, 16, 16), &explicit(InterPredIdc::L0));
        assert_eq!(motion.mv[0], Mv::new(103, 197));
    }

    #[test]
    fn merge_mode_takes_the_merge_list() {
        let mut scene = Scene::new();
        let neighbour = inter(Mv::new(100, 200), 0);
        scene.set_inter(0, 16, 16, 16, neighbour);
        let mut unit = explicit(InterPredIdc::Bi);
        unit.merge_flag = true;
        assert_eq!(
            scene.context().pb_motion(&pb(16, 16, 16, 16), &unit),
            neighbour
        );
    }
}
