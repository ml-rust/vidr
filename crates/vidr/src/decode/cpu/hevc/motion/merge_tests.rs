//! Merge candidates (8.5.3.2.2 to 8.5.3.2.5) on hand-made neighbours.

use super::*;
use crate::decode::cpu::hevc::motion::col::ColMotion;
use crate::decode::cpu::hevc::motion::test_support::{Scene, inter, inter_l1, pb};

const A: PbMotion = inter(Mv::new(1, 1), 0);
const B: PbMotion = inter(Mv::new(2, 2), 0);
const C: PbMotion = inter(Mv::new(3, 3), 0);
const D: PbMotion = inter(Mv::new(4, 4), 0);
const E: PbMotion = inter(Mv::new(5, 5), 0);

/// The current block at ( 16, 16 ), 16x16, with these neighbours inside the CTBs to
/// the left and above. A1 is at ( 15, 31 ), B1 at ( 31, 15 ), B0 at ( 32, 15 ), and
/// B2 at ( 15, 15 ). A0 at ( 15, 32 ) is later in z-scan, so a block lower down in
/// the same CTB row is the only way to reach it.
fn scene_with(a1: PbMotion, b1: PbMotion, b0: PbMotion, b2: PbMotion) -> Scene {
    let mut scene = Scene::new();
    scene.set_inter(0, 16, 16, 16, a1);
    scene.set_inter(16, 0, 16, 16, b1);
    scene.set_inter(32, 0, 16, 16, b0);
    scene.set_inter(0, 0, 16, 16, b2);
    scene
}

fn current() -> PbGeometry {
    pb(16, 16, 16, 16)
}

#[test]
fn spatial_candidates_follow_8_119() {
    let scene = scene_with(A, B, C, D);
    let context = scene.context();
    assert_eq!(context.merge_motion(&current(), 0), A);
    assert_eq!(context.merge_motion(&current(), 1), B);
    assert_eq!(context.merge_motion(&current(), 2), C);
    assert_eq!(context.merge_motion(&current(), 3), D);
}

/// B1 equals A1 and drops. B0 compares with B1, whose available flag stays set, so
/// a B0 equal to B1 drops too.
#[test]
fn equal_neighbours_are_pruned_against_their_pair() {
    let scene = scene_with(A, A, A, D);
    let context = scene.context();
    assert_eq!(context.merge_motion(&current(), 0), A);
    assert_eq!(context.merge_motion(&current(), 1), D);
}

/// A pruned B1 still prunes a B2 equal to it. B2 equals A1 here, the first test
/// in its list.
#[test]
fn b2_is_pruned_against_a1_and_b1() {
    let scene = scene_with(A, B, C, A);
    let context = scene.context();
    assert_eq!(context.merge_motion(&current(), 0), A);
    assert_eq!(context.merge_motion(&current(), 1), B);
    assert_eq!(context.merge_motion(&current(), 2), C);
    // B2 drops, and the list goes on with a combined candidate: A with no L1 gives
    // none, so a zero candidate takes place 3 with zeroIdx 0.
    assert_eq!(
        context.merge_motion(&current(), 3),
        PbMotion {
            mv: [Mv::default(); 2],
            ref_idx: [0, 0],
        }
    );
}

/// Place 4 of a B slice with no temporal candidate: zeroIdx 1, and `numRefIdx` 2.
#[test]
fn zero_candidates_walk_the_reference_indices() {
    let scene = scene_with(A, B, C, A);
    let context = scene.context();
    assert_eq!(
        context.merge_motion(&current(), 4),
        PbMotion {
            mv: [Mv::default(); 2],
            ref_idx: [1, 1],
        }
    );
}

/// (8-152): a P slice takes L0 alone, and past `numRefIdx` the index is 0.
#[test]
fn p_slice_zero_candidates_use_l0_alone() {
    let mut scene = Scene::new();
    scene.params.kind = SliceKind::P;
    scene.params.num_ref_idx_minus1 = [0, 0];
    let context = scene.context();
    let block = pb(0, 0, 16, 16);
    let expected = PbMotion {
        mv: [Mv::default(); 2],
        ref_idx: [0, NO_REF],
    };
    for index in 0..5 {
        assert_eq!(context.merge_motion(&block, index), expected);
    }
}

/// Table 8-7, combIdx 0: L0 of candidate 0 with L1 of candidate 1. Candidate 1 has
/// L1 at POC 12 and candidate 0 has L0 at POC 4, so the pictures differ.
#[test]
fn combined_candidates_pair_l0_with_l1() {
    let l1_motion = inter_l1(Mv::new(9, 9), 0);
    let mut scene = Scene::new();
    scene.set_inter(0, 16, 16, 16, A);
    scene.set_inter(16, 0, 16, 16, l1_motion);
    let context = scene.context();
    assert_eq!(context.merge_motion(&current(), 0), A);
    assert_eq!(context.merge_motion(&current(), 1), l1_motion);
    assert_eq!(
        context.merge_motion(&current(), 2),
        PbMotion {
            mv: [Mv::new(1, 1), Mv::new(9, 9)],
            ref_idx: [0, 0],
        }
    );
}

/// The same picture and the same vector give no combined candidate (8.5.3.2.4).
#[test]
fn a_combination_of_one_picture_and_one_vector_drops() {
    let mut scene = Scene::new();
    scene.refs.poc[1][0] = 4;
    let l1_same = inter_l1(Mv::new(1, 1), 0);
    scene.set_inter(0, 16, 16, 16, A);
    scene.set_inter(16, 0, 16, 16, l1_same);
    let context = scene.context();
    // zeroIdx 0.
    assert_eq!(
        context.merge_motion(&current(), 2),
        PbMotion {
            mv: [Mv::default(); 2],
            ref_idx: [0, 0],
        }
    );
}

/// `numOrigMergeCand` at `MaxNumMergeCand` adds no combined candidate. With 2 of 2
/// the indices never pass the list, so a third request stays in the zero candidates.
#[test]
fn no_combination_at_the_limit() {
    let l1_motion = inter_l1(Mv::new(9, 9), 0);
    let mut scene = Scene::new();
    scene.params.max_merge_cand = 2;
    scene.set_inter(0, 16, 16, 16, A);
    scene.set_inter(16, 0, 16, 16, l1_motion);
    let context = scene.context();
    assert_eq!(
        context.merge_motion(&current(), 2),
        PbMotion {
            mv: [Mv::default(); 2],
            ref_idx: [0, 0],
        }
    );
}

/// Step 10: a bi-predicted candidate on an 8x4 block keeps L0.
#[test]
fn an_8x4_block_drops_l1() {
    let bi = PbMotion {
        mv: [Mv::new(1, 1), Mv::new(2, 2)],
        ref_idx: [0, 1],
    };
    let mut scene = Scene::new();
    scene.set_inter(0, 16, 16, 16, bi);
    let context = scene.context();
    let block = PbGeometry {
        x_cb: 16,
        y_cb: 24,
        n_cb: 8,
        x_pb: 16,
        y_pb: 24,
        width: 8,
        height: 4,
        part_idx: 0,
        part_mode: PartMode::TwoNxN,
    };
    assert_eq!(
        context.merge_motion(&block, 0),
        PbMotion {
            mv: [Mv::new(1, 1), Mv::default()],
            ref_idx: [0, NO_REF],
        }
    );
}

/// 8.5.3.2.3: the second block of a vertical pair skips A1, and the second block of
/// a horizontal pair skips B1.
#[test]
fn the_second_block_of_a_pair_skips_the_first() {
    let mut scene = Scene::new();
    // The whole 16x16 unit at ( 16, 16 ) holds A, in its left half.
    scene.set_inter(16, 16, 8, 16, A);
    scene.set_inter(16, 0, 16, 16, B);
    let context = scene.context();
    let right = PbGeometry {
        x_cb: 16,
        y_cb: 16,
        n_cb: 16,
        x_pb: 24,
        y_pb: 16,
        width: 8,
        height: 16,
        part_idx: 1,
        part_mode: PartMode::NxTwoN,
    };
    // A1 drops, and B1 is the first candidate.
    assert_eq!(context.merge_motion(&right, 0), B);
    // The second block of a 2NxN pair skips B1, and A1 is the first candidate.
    let mut scene = Scene::new();
    scene.set_inter(16, 16, 16, 8, A);
    scene.set_inter(0, 16, 16, 16, B);
    let context = scene.context();
    let lower = PbGeometry {
        x_cb: 16,
        y_cb: 16,
        n_cb: 16,
        x_pb: 16,
        y_pb: 24,
        width: 16,
        height: 8,
        part_idx: 1,
        part_mode: PartMode::TwoNxN,
    };
    assert_eq!(context.merge_motion(&lower, 0), B);
}

/// Log2ParMrgLevel 5: every spatial neighbour inside the 32x32 region drops.
#[test]
fn a_neighbour_in_the_merge_region_drops() {
    let mut scene = scene_with(A, B, C, D);
    scene.params.log2_par_mrg_level = 5;
    let context = scene.context();
    // Every neighbour of ( 16, 16 ) lies in the region at ( 0, 0 ) except B0, at
    // ( 32, 15 ).
    assert_eq!(context.merge_motion(&current(), 0), C);
}

/// (8-110): a level above 2 makes an 8x8 unit share the list of its 2Nx2N block. The
/// second block of the pair still reads A1 of the whole unit.
#[test]
fn an_8x8_unit_shares_one_list() {
    let mut scene = Scene::new();
    scene.params.log2_par_mrg_level = 3;
    scene.set_inter(8, 8, 8, 8, A);
    let context = scene.context();
    let second = PbGeometry {
        x_cb: 16,
        y_cb: 8,
        n_cb: 8,
        x_pb: 20,
        y_pb: 8,
        width: 4,
        height: 8,
        part_idx: 1,
        part_mode: PartMode::NxTwoN,
    };
    // A1 of the unit is ( 15, 15 ). With the block as coded it was ( 19, 15 ) inside
    // the unit, and partIdx 1 of Nx2N skips it there.
    assert_eq!(context.merge_motion(&second, 0), A);
}

/// The temporal candidate takes `refIdxLXCol` 0 in each list, after the spatial ones.
#[test]
fn the_temporal_candidate_follows_the_spatial_ones() {
    let mut scene = Scene::new();
    scene.set_inter(0, 16, 16, 16, A);
    let mut col = ColMotion::new();
    col.prepare(64, 64).unwrap();
    let mut field =
        crate::decode::cpu::hevc::motion::field::MotionField::new(64, 64, &scene.layout).unwrap();
    let mut col_refs = crate::decode::cpu::hevc::motion::types::RefLists::default();
    // The collocated unit points at POC 2 from POC 6. colPocDiff 4.
    col_refs.poc[0][0] = 2;
    field.begin_slice(col_refs).unwrap();
    for rs in 0..16 {
        field.set_ctb(rs);
    }
    field.fill(0, 0, 64, 64, inter(Mv::new(8, 0), 0));
    field.compress(64, 64, &mut col).unwrap();
    scene.col = Some(col);
    scene.col_poc = 6;
    let context = scene.context();
    assert_eq!(context.merge_motion(&current(), 0), A);
    // L0 index 0 is POC 4: currPocDiff 4, equal, so the vector stays. L1 index 0 is
    // POC 12: currPocDiff -4. td 4, tb -4: tx 4096, factor ( -16384 + 32 ) >> 6 is
    // -256. -256 * 8 = -2048, and ( 2048 + 127 ) >> 8 = 8, so -8.
    assert_eq!(
        context.merge_motion(&current(), 1),
        PbMotion {
            mv: [Mv::new(8, 0), Mv::new(-8, 0)],
            ref_idx: [0, 0],
        }
    );
}

#[test]
fn the_list_constant_covers_every_candidate() {
    let scene = scene_with(A, B, C, D);
    let mut list = MergeList::new();
    scene.context().spatial_candidates(&current(), &mut list);
    assert_eq!(list.count, 4);
    for candidate in [A, B, C, D, E, E, E] {
        list.push(candidate);
    }
    assert_eq!(list.count, MAX_LIST);
}
