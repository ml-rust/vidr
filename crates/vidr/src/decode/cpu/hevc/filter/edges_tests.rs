//! Edge marking and `bS` (8.7.2.2 to 8.7.2.4) on hand-made coding units.

use super::*;
use crate::decode::cpu::hevc::layout::LayoutSizes;
use crate::decode::cpu::hevc::motion::types::{Mv, NO_REF, PbMotion, RefLists};

fn cu(x: u32, y: u32, log2_size: u32, first_tu: usize) -> CuRecord {
    CuRecord {
        x,
        y,
        log2_size,
        pred_mode: PredMode::Intra,
        part_mode: PartMode::TwoNxTwoN,
        transquant_bypass: false,
        pcm: false,
        qp_y: 26,
        intra_luma: [1; 4],
        intra_chroma: 1,
        first_pu: 0,
        first_tu,
        first_pcm_sample: 0,
    }
}

fn tu(x: u32, y: u32, log2_size: u32) -> TuRecord {
    TuRecord {
        x,
        y,
        x_base: x,
        y_base: y,
        log2_size,
        depth: 0,
        blk_idx: 0,
        cbf_luma: false,
        cbf_cb: false,
        cbf_cr: false,
        cu: 0,
        first_tb: 0,
        tb_count: 0,
    }
}

struct Fixture {
    params: CodingParams,
    layout: PicLayout,
    maps: ParseMaps,
    field: MotionField,
}

/// 32x32 luma with 16x16 CTBs in `columns` tile columns, every CTB in slice 0.
fn fixture(columns: &[u32]) -> Fixture {
    let params = CodingParams {
        width: 32,
        height: 32,
        deblocking_disabled: false,
        ..CodingParams::for_tests()
    };
    let layout = PicLayout::new(&LayoutSizes {
        width: 32,
        height: 32,
        ctb_log2: 4,
        min_tb_log2: 2,
        column_widths: columns,
        row_heights: &[2],
    })
    .unwrap();
    let mut maps = ParseMaps::new(&params, &layout).unwrap();
    let mut field = MotionField::new(32, 32, &layout).unwrap();
    let mut lists = RefLists::default();
    // Reference index 0 of L0 is POC 1, and index 1 is POC 2.
    lists.poc[0][0] = 1;
    lists.poc[0][1] = 2;
    field.begin_slice(lists).unwrap();
    for rs in 0..4 {
        maps.set_ctb_slice(rs, 0);
        field.set_ctb(rs);
    }
    Fixture {
        params,
        layout,
        maps,
        field,
    }
}

fn input(fixture: &Fixture) -> StrengthInput<'_> {
    StrengthInput {
        params: &fixture.params,
        layout: &fixture.layout,
        maps: &fixture.maps,
        field: &fixture.field,
    }
}

fn mark(fixture: &Fixture, records: &CtuRecords) -> EdgeMap {
    let mut edges = EdgeMap::new(32, 32).unwrap();
    edges.mark_ctu(&input(fixture), records);
    edges
}

fn inter_cu(x: u32, y: u32, log2_size: u32, first_tu: usize) -> CuRecord {
    CuRecord {
        pred_mode: PredMode::Inter,
        ..cu(x, y, log2_size, first_tu)
    }
}

fn motion(x: i16, ref_idx: i8) -> PbMotion {
    PbMotion {
        mv: [Mv::new(x, 0), Mv::default()],
        ref_idx: [ref_idx, NO_REF],
    }
}

fn coded_tu(x: u32, y: u32, log2_size: u32) -> TuRecord {
    TuRecord {
        cbf_luma: true,
        ..tu(x, y, log2_size)
    }
}

/// (8.7.2.4): the left and upper blocks have the same motion, so only a coded
/// transform block raises `bS`. The second run holds a coded block.
#[test]
fn a_coded_transform_block_gives_strength_1() {
    let mut fixture = fixture(&[2]);
    fixture.field.fill(0, 0, 32, 32, motion(0, 0));
    // The left neighbour at ( 0, 0 ) has no coded block.
    let first = CtuRecords {
        cus: vec![inter_cu(0, 0, 4, 0)],
        ..CtuRecords::default()
    };
    let second = CtuRecords {
        cus: vec![inter_cu(16, 0, 4, 0)],
        tus: vec![coded_tu(16, 8, 3), tu(24, 8, 3)],
        ..CtuRecords::default()
    };
    let mut edges = EdgeMap::new(32, 32).unwrap();
    edges.mark_ctu(&input(&fixture), &first);
    edges.mark_ctu(&input(&fixture), &second);
    // The coded block touches the left edge only below y = 8.
    assert_eq!(edges.vertical_at(16, 0), 0);
    assert_eq!(edges.vertical_at(16, 8), 1);
    // The inner edge at x = 24 has the coded block on its p side.
    assert_eq!(edges.vertical_at(24, 8), 1);
    assert_eq!(edges.vertical_at(24, 0), 0);
}

/// The p side counts too: the left unit is coded, and this unit is skipped.
#[test]
fn a_coded_block_on_the_p_side_gives_strength_1() {
    let mut fixture = fixture(&[2]);
    fixture.field.fill(0, 0, 32, 32, motion(0, 0));
    let left = CtuRecords {
        cus: vec![inter_cu(0, 0, 4, 0)],
        tus: vec![coded_tu(0, 0, 4)],
        ..CtuRecords::default()
    };
    let mut skipped = inter_cu(16, 0, 4, 0);
    skipped.pred_mode = PredMode::Skip;
    let right = CtuRecords {
        cus: vec![skipped],
        ..CtuRecords::default()
    };
    let mut edges = EdgeMap::new(32, 32).unwrap();
    edges.mark_ctu(&input(&fixture), &left);
    edges.mark_ctu(&input(&fixture), &right);
    assert_eq!(edges.vertical_at(16, 12), 1);
}

/// A vector four quarter samples away, or another picture, gives 1. A smaller
/// difference gives 0.
#[test]
fn motion_decides_a_section_with_no_coefficients() {
    let mut fixture = fixture(&[2]);
    fixture.field.fill(0, 0, 16, 16, motion(0, 0));
    let records = CtuRecords {
        cus: vec![inter_cu(16, 0, 4, 0)],
        ..CtuRecords::default()
    };
    fixture.field.fill(16, 0, 16, 16, motion(3, 0));
    assert_eq!(mark(&fixture, &records).vertical_at(16, 4), 0);
    fixture.field.fill(16, 0, 16, 16, motion(4, 0));
    assert_eq!(mark(&fixture, &records).vertical_at(16, 4), 1);
    fixture.field.fill(16, 0, 16, 16, motion(0, 1));
    assert_eq!(mark(&fixture, &records).vertical_at(16, 4), 1);
}

/// An inter unit next to an intra unit has `bS` 2 on its own left edge.
#[test]
fn an_intra_neighbour_gives_strength_2() {
    let mut fixture = fixture(&[2]);
    fixture.maps.set_cu_kind(0, 0, 16, true, false);
    fixture.field.fill(16, 0, 16, 16, motion(0, 0));
    let records = CtuRecords {
        cus: vec![inter_cu(16, 0, 4, 0)],
        ..CtuRecords::default()
    };
    assert_eq!(mark(&fixture, &records).vertical_at(16, 0), 2);
}

/// (8.7.2.3): a prediction edge inside the unit is not a transform edge, so a coded
/// block alone does not raise it, and motion decides.
#[test]
fn a_prediction_edge_reads_motion_only() {
    let mut fixture = fixture(&[2]);
    let mut unit = inter_cu(0, 0, 4, 0);
    unit.part_mode = PartMode::NxTwoN;
    fixture.field.fill(0, 0, 8, 16, motion(0, 0));
    fixture.field.fill(8, 0, 8, 16, motion(1, 0));
    let records = CtuRecords {
        cus: vec![unit],
        tus: vec![coded_tu(0, 0, 4)],
        ..CtuRecords::default()
    };
    assert_eq!(mark(&fixture, &records).vertical_at(8, 0), 0);
    fixture.field.fill(8, 0, 8, 16, motion(9, 0));
    assert_eq!(mark(&fixture, &records).vertical_at(8, 0), 1);
}

/// AMP: 2NxnU of a 32x32 unit places the edge at y = 8, on the grid. nLx2N of a
/// 16x16 unit places it at x = 4, off the grid, and it is dropped.
#[test]
fn amp_edges_keep_to_the_grid() {
    let mut fixture = fixture(&[2]);
    fixture.field.fill(0, 0, 32, 8, motion(0, 0));
    fixture.field.fill(0, 8, 32, 24, motion(8, 0));
    let mut upper = inter_cu(0, 0, 5, 0);
    upper.part_mode = PartMode::TwoNxnU;
    let records = CtuRecords {
        cus: vec![upper],
        ..CtuRecords::default()
    };
    assert_eq!(mark(&fixture, &records).horizontal_at(28, 8), 1);
    let mut left = inter_cu(0, 0, 4, 0);
    left.part_mode = PartMode::NLxTwoN;
    fixture.field.fill(0, 0, 4, 16, motion(0, 0));
    fixture.field.fill(4, 0, 12, 16, motion(8, 0));
    let records = CtuRecords {
        cus: vec![left],
        ..CtuRecords::default()
    };
    assert!(!mark(&fixture, &records).has_edges());
}

/// A disabled slice still records its coded blocks for the slice after it.
#[test]
fn a_disabled_slice_records_its_coded_blocks() {
    let mut fixture = fixture(&[2]);
    fixture.field.fill(0, 0, 32, 32, motion(0, 0));
    fixture.params.deblocking_disabled = true;
    let mut edges = EdgeMap::new(32, 32).unwrap();
    let first = CtuRecords {
        cus: vec![inter_cu(0, 0, 4, 0)],
        tus: vec![coded_tu(0, 0, 4)],
        ..CtuRecords::default()
    };
    edges.mark_ctu(&input(&fixture), &first);
    assert!(!edges.has_edges());
    fixture.params.deblocking_disabled = false;
    let second = CtuRecords {
        cus: vec![inter_cu(16, 0, 4, 0)],
        ..CtuRecords::default()
    };
    edges.mark_ctu(&input(&fixture), &second);
    assert_eq!(edges.vertical_at(16, 0), 1);
}

#[test]
fn picture_boundaries_take_no_edge() {
    let fixture = fixture(&[2]);
    let records = CtuRecords {
        cus: vec![cu(0, 0, 4, 0)],
        tus: vec![tu(0, 0, 4)],
        ..CtuRecords::default()
    };
    let edges = mark(&fixture, &records);
    assert_eq!(edges.vertical_at(0, 0), 0);
    assert_eq!(edges.horizontal_at(0, 0), 0);
    assert!(!edges.has_edges());
}

#[test]
fn a_unit_marks_its_left_and_top_edges_with_strength_2() {
    let fixture = fixture(&[2]);
    let records = CtuRecords {
        cus: vec![cu(16, 16, 4, 0)],
        tus: vec![tu(16, 16, 4)],
        ..CtuRecords::default()
    };
    let edges = mark(&fixture, &records);
    for offset in (0..16).step_by(4) {
        assert_eq!(edges.vertical_at(16, 16 + offset), 2);
        assert_eq!(edges.horizontal_at(16 + offset, 16), 2);
    }
    assert_eq!(edges.vertical_at(16, 12), 0);
    assert_eq!(edges.horizontal_at(12, 16), 0);
    assert!(edges.has_edges());
}

/// Inside a unit, a transform edge on the grid is marked, and one off the grid
/// is not (8.7.2.1).
#[test]
fn transform_edges_keep_to_the_grid() {
    let fixture = fixture(&[2]);
    let records = CtuRecords {
        cus: vec![cu(0, 0, 4, 0)],
        tus: vec![
            tu(0, 0, 3),
            tu(8, 0, 2),
            tu(12, 0, 2),
            tu(8, 4, 2),
            tu(12, 4, 2),
            tu(0, 8, 3),
            tu(8, 8, 3),
        ],
        ..CtuRecords::default()
    };
    let edges = mark(&fixture, &records);
    assert_eq!(edges.vertical_at(8, 0), 2);
    assert_eq!(edges.vertical_at(8, 4), 2);
    assert_eq!(edges.vertical_at(8, 12), 2);
    assert_eq!(edges.horizontal_at(0, 8), 2);
    assert_eq!(edges.horizontal_at(12, 8), 2);
    // y = 4 lies off the grid, and no edge holds row 0 of the picture.
    assert_eq!(edges.horizontal_at(8, 4), 0);
}

#[test]
fn an_edge_off_the_grid_is_dropped() {
    let mut edges = EdgeMap::new(16, 16).unwrap();
    let run = EdgeRun {
        x: 12,
        y: 4,
        length: 4,
        strength: 2,
    };
    edges.mark_vertical(run);
    edges.mark_horizontal(run);
    assert!(!edges.has_edges());
}

#[test]
fn a_disabled_slice_marks_nothing() {
    let mut fixture = fixture(&[2]);
    fixture.params.deblocking_disabled = true;
    let records = CtuRecords {
        cus: vec![cu(16, 16, 4, 0)],
        tus: vec![tu(16, 16, 4)],
        ..CtuRecords::default()
    };
    assert!(!mark(&fixture, &records).has_edges());
}

/// The current slice's flag decides its left and top boundaries (8.7.2.1).
#[test]
fn a_slice_boundary_follows_the_current_slice_flag() {
    let mut fixture = fixture(&[2]);
    fixture.maps.set_ctb_slice(1, 1);
    fixture.params.loop_filter_across_slices = false;
    let records = CtuRecords {
        cus: vec![cu(16, 0, 4, 0)],
        tus: vec![tu(16, 0, 4)],
        ..CtuRecords::default()
    };
    assert_eq!(mark(&fixture, &records).vertical_at(16, 0), 0);
    fixture.params.loop_filter_across_slices = true;
    assert_eq!(mark(&fixture, &records).vertical_at(16, 0), 2);
}

#[test]
fn a_tile_boundary_follows_the_pps_flag() {
    let mut fixture = fixture(&[1, 1]);
    fixture.params.loop_filter_across_tiles = false;
    let records = CtuRecords {
        cus: vec![cu(16, 0, 4, 0)],
        tus: vec![tu(16, 0, 4)],
        ..CtuRecords::default()
    };
    assert_eq!(mark(&fixture, &records).vertical_at(16, 0), 0);
    fixture.params.loop_filter_across_tiles = true;
    assert_eq!(mark(&fixture, &records).vertical_at(16, 0), 2);
}

#[test]
fn nxn_marks_the_middle_of_a_large_unit() {
    let fixture = fixture(&[2]);
    let mut unit = cu(0, 0, 4, 0);
    unit.part_mode = PartMode::NxN;
    let records = CtuRecords {
        cus: vec![unit],
        ..CtuRecords::default()
    };
    let edges = mark(&fixture, &records);
    assert_eq!(edges.vertical_at(8, 12), 2);
    assert_eq!(edges.horizontal_at(12, 8), 2);
}

#[test]
fn reset_removes_every_edge() {
    let mut edges = EdgeMap::new(16, 16).unwrap();
    edges.mark_vertical(EdgeRun {
        x: 8,
        y: 0,
        length: 16,
        strength: 2,
    });
    assert_eq!(edges.vertical_at(8, 12), 2);
    edges.reset();
    assert_eq!(edges.vertical_at(8, 12), 0);
    assert!(!edges.has_edges());
}
