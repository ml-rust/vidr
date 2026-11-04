//! The motion conditions of `bS` (8.7.2.4).

use crate::decode::cpu::hevc::motion::types::{Mv, PbMotion};

/// The motion of one side of an edge, with the POC of each reference picture. POC
/// names a picture, because no two pictures of one CVS share a POC (8.3.1).
#[derive(Clone, Copy, Debug)]
pub(crate) struct EdgeSide {
    pub(crate) motion: PbMotion,
    /// `PicOrderCnt` of the reference of each list.
    pub(crate) ref_poc: [i32; 2],
}

/// `bS` 1 when the two blocks use another picture, another vector count, or vectors
/// at least one full sample apart. Otherwise 0.
pub(crate) fn motion_strength(p: &EdgeSide, q: &EdgeSide) -> u8 {
    let count = p.motion.vector_count();
    if count != q.motion.vector_count() {
        return 1;
    }
    let different = match count {
        0 => false,
        1 => {
            let (p_list, q_list) = (first_list(&p.motion), first_list(&q.motion));
            p.ref_poc[p_list] != q.ref_poc[q_list] || far(p.motion.mv[p_list], q.motion.mv[q_list])
        }
        _ => two_vectors_differ(p, q),
    };
    u8::from(different)
}

/// The list a block with one vector uses.
const fn first_list(motion: &PbMotion) -> usize {
    if motion.uses(0) { 0 } else { 1 }
}

/// True when a component differs by at least 4 quarter samples.
fn far(a: Mv, b: Mv) -> bool {
    (i32::from(a.x) - i32::from(b.x)).abs() >= 4 || (i32::from(a.y) - i32::from(b.y)).abs() >= 4
}

fn two_vectors_differ(p: &EdgeSide, q: &EdgeSide) -> bool {
    let [p0, p1] = p.ref_poc;
    let [q0, q1] = q.ref_poc;
    let [p_l0, p_l1] = p.motion.mv;
    let [q_l0, q_l1] = q.motion.mv;
    let same_pictures = (p0 == q0 && p1 == q1) || (p0 == q1 && p1 == q0);
    if !same_pictures {
        return true;
    }
    if p0 != p1 {
        // Two different pictures: compare the vectors for each picture.
        if p0 == q0 {
            far(p_l0, q_l0) || far(p_l1, q_l1)
        } else {
            far(p_l0, q_l1) || far(p_l1, q_l0)
        }
    } else {
        // One picture twice: both pairings have to differ.
        (far(p_l0, q_l0) || far(p_l1, q_l1)) && (far(p_l0, q_l1) || far(p_l1, q_l0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::cpu::hevc::motion::types::NO_REF;

    fn one(mv: Mv, poc: i32) -> EdgeSide {
        EdgeSide {
            motion: PbMotion {
                mv: [mv, Mv::default()],
                ref_idx: [0, NO_REF],
            },
            ref_poc: [poc, 0],
        }
    }

    fn one_l1(mv: Mv, poc: i32) -> EdgeSide {
        EdgeSide {
            motion: PbMotion {
                mv: [Mv::default(), mv],
                ref_idx: [NO_REF, 0],
            },
            ref_poc: [0, poc],
        }
    }

    fn two(mv: [Mv; 2], poc: [i32; 2]) -> EdgeSide {
        EdgeSide {
            motion: PbMotion {
                mv,
                ref_idx: [0, 0],
            },
            ref_poc: poc,
        }
    }

    const A: Mv = Mv::new(0, 0);
    const NEAR: Mv = Mv::new(3, -3);
    const FAR_X: Mv = Mv::new(4, 0);
    const FAR_Y: Mv = Mv::new(0, -4);

    #[test]
    fn another_vector_count_gives_1() {
        assert_eq!(motion_strength(&one(A, 1), &two([A, A], [1, 1])), 1);
    }

    #[test]
    fn one_vector_compares_picture_and_distance() {
        assert_eq!(motion_strength(&one(A, 1), &one(NEAR, 1)), 0);
        assert_eq!(motion_strength(&one(A, 1), &one(FAR_X, 1)), 1);
        assert_eq!(motion_strength(&one(A, 1), &one(FAR_Y, 1)), 1);
        assert_eq!(motion_strength(&one(A, 1), &one(A, 2)), 1);
    }

    /// NOTE 1 of 8.7.2.4: the list a block uses does not matter.
    #[test]
    fn one_vector_ignores_the_list() {
        assert_eq!(motion_strength(&one(A, 1), &one_l1(NEAR, 1)), 0);
        assert_eq!(motion_strength(&one_l1(A, 1), &one(FAR_X, 1)), 1);
    }

    #[test]
    fn two_pictures_compare_each_picture_s_vectors() {
        let p = two([A, FAR_X], [1, 2]);
        // The same order and near vectors.
        assert_eq!(motion_strength(&p, &two([NEAR, FAR_X], [1, 2])), 0);
        // The swapped order pairs L0 of p with L1 of q.
        assert_eq!(motion_strength(&p, &two([FAR_X, NEAR], [2, 1])), 0);
        // One vector is far.
        assert_eq!(motion_strength(&p, &two([FAR_Y, FAR_X], [1, 2])), 1);
        // Another picture.
        assert_eq!(motion_strength(&p, &two([A, FAR_X], [1, 3])), 1);
    }

    #[test]
    fn one_picture_twice_takes_the_better_pairing() {
        let p = two([A, FAR_X], [1, 1]);
        // Direct pairing near.
        assert_eq!(motion_strength(&p, &two([NEAR, FAR_X], [1, 1])), 0);
        // Crossed pairing near, direct pairing far.
        assert_eq!(motion_strength(&p, &two([FAR_X, A], [1, 1])), 0);
        // Both pairings far.
        assert_eq!(motion_strength(&p, &two([FAR_Y, FAR_Y], [1, 1])), 1);
    }

    #[test]
    fn no_vectors_give_0() {
        let side = EdgeSide {
            motion: PbMotion::NONE,
            ref_poc: [0; 2],
        };
        assert_eq!(motion_strength(&side, &side), 0);
    }
}
