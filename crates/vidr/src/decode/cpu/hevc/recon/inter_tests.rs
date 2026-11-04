//! One inter coding unit through motion, prediction, and residual, on hand-worked
//! values.

use std::sync::Arc;

use super::*;
use crate::bitstream::hevc_pred_weight::PredWeightTable;
use crate::bitstream::hevc_sps_full::tests as sps_tests;
use crate::decode::cpu::hevc::dpb::buffer::Frame;
use crate::decode::cpu::hevc::layout::{LayoutSizes, PicLayout};
use crate::decode::cpu::hevc::motion::col::ColMotion;
use crate::decode::cpu::hevc::motion::types::Mv;
use crate::decode::cpu::hevc::params::{CodingParams, SliceKind};
use crate::decode::cpu::hevc::recon::weights::ExplicitWeights;
use crate::decode::cpu::hevc::syntax::maps::ParseMaps;
use crate::decode::cpu::hevc::syntax::records::{
    Coeff, InterPredIdc, PartMode, PredMode, TbRecord, TuRecord,
};

/// A 16x16 picture of one CTB at POC 4, with its maps.
struct Scene {
    params: CodingParams,
    layout: PicLayout,
    maps: ParseMaps,
    field: MotionField,
}

impl Scene {
    fn new() -> Self {
        let params = CodingParams {
            kind: SliceKind::B,
            max_merge_cand: 5,
            ..CodingParams::for_tests()
        };
        let layout = PicLayout::new(&LayoutSizes {
            width: 16,
            height: 16,
            ctb_log2: 4,
            min_tb_log2: 2,
            column_widths: &[1],
            row_heights: &[1],
        })
        .unwrap();
        let mut maps = ParseMaps::new(&params, &layout).unwrap();
        maps.set_ctb_slice(0, 0);
        let field = MotionField::new(16, 16, &layout).unwrap();
        Self {
            params,
            layout,
            maps,
            field,
        }
    }

    fn run(&mut self, refs: &SliceRefs<u8>, records: &CtuRecords, picture: &mut Picture<u8>) {
        self.field.begin_slice(refs.lists).unwrap();
        let context = ReconContext {
            params: &self.params,
            layout: &self.layout,
            maps: &self.maps,
            factors: None,
        };
        let mut inter = InterContext {
            refs,
            field: &mut self.field,
            poc: 4,
        };
        Reconstructor::new().reconstruct_ctu(&context, &mut inter, records, picture);
    }
}

/// A reference frame: luma from `luma`, Cb 60, and Cr 90.
fn frame(poc: i32, luma: impl Fn(usize, usize) -> u8) -> Arc<Frame<u8>> {
    let mut picture = Picture::<u8>::new(16, 16).unwrap();
    for y in 0..16 {
        for (x, sample) in picture
            .luma_mut()
            .row_mut(y)
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            *sample = luma(x, y);
        }
    }
    for y in 0..8 {
        picture.cb_mut().row_mut(y).unwrap().fill(60);
        picture.cr_mut().row_mut(y).unwrap().fill(90);
    }
    Arc::new(Frame {
        picture,
        col: ColMotion::new(),
        poc,
    })
}

/// Lists with these pictures at index 0.
fn refs(l0: Option<Arc<Frame<u8>>>, l1: Option<Arc<Frame<u8>>>) -> SliceRefs<u8> {
    let mut refs = SliceRefs::intra();
    for (list, frame) in [l0, l1].into_iter().enumerate() {
        if let Some(frame) = frame {
            refs.lists.poc[list][0] = frame.poc;
            refs.lists.count[list] = 1;
            refs.frames[list].push(frame);
        }
    }
    refs
}

fn cu() -> CuRecord {
    CuRecord {
        x: 0,
        y: 0,
        log2_size: 4,
        pred_mode: PredMode::Inter,
        part_mode: PartMode::TwoNxTwoN,
        transquant_bypass: false,
        pcm: false,
        qp_y: 26,
        intra_luma: [1; 4],
        intra_chroma: 1,
        first_pu: 0,
        first_tu: 0,
        first_pcm_sample: 0,
    }
}

fn pu(idc: InterPredIdc, mvd: [[i32; 2]; 2]) -> PuRecord {
    PuRecord {
        x: 0,
        y: 0,
        width: 16,
        height: 16,
        merge_flag: false,
        merge_idx: 0,
        inter_pred_idc: idc,
        ref_idx: [0, 0],
        mvd,
        mvp_flag: [false, false],
    }
}

fn records(prediction: PuRecord) -> CtuRecords {
    CtuRecords {
        cus: vec![cu()],
        pus: vec![prediction],
        ..CtuRecords::default()
    }
}

/// With no neighbours the predictor is zero, so mvL0 is the difference. ( 8, 4 ) in
/// quarter samples is two samples right and one down. The sample at ( 3, 2 ) takes
/// the reference at ( 5, 3 ), and the last row repeats the reference's last row.
#[test]
fn a_whole_sample_vector_copies_the_reference() {
    let mut scene = Scene::new();
    let list = refs(Some(frame(0, |x, y| (x + 10 * y) as u8)), None);
    let mut picture = Picture::<u8>::new(16, 16).unwrap();
    let units = records(pu(InterPredIdc::L0, [[8, 4], [0, 0]]));
    scene.run(&list, &units, &mut picture);
    assert_eq!(picture.luma().row(2).unwrap()[3], 5 + 30);
    assert_eq!(picture.luma().row(15).unwrap()[3], 5 + 150);
    // The chroma vector has a fraction, and a flat plane stays flat.
    assert_eq!(picture.cb().row(4).unwrap()[4], 60);
    assert_eq!(picture.cr().row(4).unwrap()[4], 90);
    // (8-80) to (8-85): the block's motion is stored for later blocks.
    assert_eq!(scene.field.at(15, 15).mv[0], Mv::new(8, 4));
    assert_eq!(scene.field.at(15, 15).ref_idx, [0, -1]);
}

/// (8-264): two lists average with rounding up: ( 40 + 81 + 1 ) >> 1 = 61.
#[test]
fn two_lists_average() {
    let mut scene = Scene::new();
    let list = refs(Some(frame(0, |_, _| 40)), Some(frame(8, |_, _| 81)));
    let mut picture = Picture::<u8>::new(16, 16).unwrap();
    let units = records(pu(InterPredIdc::Bi, [[0, 0], [0, 0]]));
    scene.run(&list, &units, &mut picture);
    assert_eq!(picture.luma().row(0).unwrap()[0], 61);
    assert_eq!(picture.cb().row(7).unwrap()[7], 60);
}

/// A skipped unit with no neighbours takes zero candidate 0: both lists at index 0.
#[test]
fn a_skipped_unit_merges_a_zero_candidate() {
    let mut scene = Scene::new();
    let list = refs(Some(frame(0, |_, _| 20)), Some(frame(8, |_, _| 40)));
    let mut picture = Picture::<u8>::new(16, 16).unwrap();
    let mut prediction = pu(InterPredIdc::L0, [[0, 0], [0, 0]]);
    prediction.merge_flag = true;
    let mut units = records(prediction);
    units.cus[0].pred_mode = PredMode::Skip;
    scene.run(&list, &units, &mut picture);
    assert_eq!(picture.luma().row(8).unwrap()[8], 30);
    assert_eq!(scene.field.at(0, 0).ref_idx, [0, 0]);
}

/// (8-275): the table's weights change the samples. Denominator 1 and delta 2 give
/// weight 4 at log2Wd 7. ( 100 << 6 ) * 4 = 25600, ( 25600 + 64 ) >> 7 = 200, and an
/// offset of -10 gives 190.
#[test]
fn a_weight_table_changes_the_samples() {
    let mut scene = Scene::new();
    let mut list = refs(Some(frame(0, |_, _| 100)), None);
    let mut table = PredWeightTable {
        luma_log2_weight_denom: 1,
        ..PredWeightTable::default()
    };
    table.l0.luma_weight_flag[0] = true;
    table.l0.delta_luma_weight[0] = 2;
    table.l0.luma_offset[0] = -10;
    let sps = sps_tests::valid(&sps_tests::Spec::default());
    list.weights = Some(Box::new(ExplicitWeights::new(&table, &sps)));
    let mut picture = Picture::<u8>::new(16, 16).unwrap();
    let units = records(pu(InterPredIdc::L0, [[0, 0], [0, 0]]));
    scene.run(&list, &units, &mut picture);
    assert_eq!(picture.luma().row(0).unwrap()[0], 190);
    // Chroma has no flag in this table: weight 1 at a denominator of 1, which is
    // ( 3840 * 2 + 64 ) >> 7 = 60.
    assert_eq!(picture.cb().row(0).unwrap()[0], 60);
}

/// (8-299): a bypass residual adds to the prediction after it, in luma and chroma, and
/// chroma reads half the location.
#[test]
fn a_residual_adds_to_the_prediction() {
    let mut scene = Scene::new();
    let list = refs(Some(frame(0, |_, _| 100)), None);
    let mut picture = Picture::<u8>::new(16, 16).unwrap();
    let mut units = records(pu(InterPredIdc::L0, [[0, 0], [0, 0]]));
    units.cus[0].transquant_bypass = true;
    units.tus = vec![TuRecord {
        x: 8,
        y: 8,
        x_base: 0,
        y_base: 0,
        log2_size: 3,
        depth: 1,
        blk_idx: 3,
        cbf_luma: true,
        cbf_cb: true,
        cbf_cr: false,
        cu: 0,
        first_tb: 0,
        tb_count: 2,
    }];
    units.tbs = vec![
        TbRecord {
            x: 8,
            y: 8,
            log2_size: 3,
            c_idx: 0,
            transform_skip: false,
            first_coeff: 0,
            coeff_count: 1,
        },
        TbRecord {
            x: 8,
            y: 8,
            log2_size: 2,
            c_idx: 1,
            transform_skip: false,
            first_coeff: 1,
            coeff_count: 1,
        },
    ];
    units.coeffs = vec![
        Coeff {
            x: 1,
            y: 2,
            level: 7,
        },
        Coeff {
            x: 3,
            y: 3,
            level: -9,
        },
    ];
    scene.run(&list, &units, &mut picture);
    assert_eq!(picture.luma().row(10).unwrap()[9], 107);
    assert_eq!(picture.luma().row(10).unwrap()[8], 100);
    assert_eq!(picture.cb().row(7).unwrap()[7], 51);
    assert_eq!(picture.cb().row(4).unwrap()[4], 60);
    assert_eq!(picture.cr().row(7).unwrap()[7], 90);
}

/// A residual clips to the bit depth (8-299).
#[test]
fn a_residual_clips() {
    let mut scene = Scene::new();
    let list = refs(Some(frame(0, |_, _| 250)), None);
    let mut picture = Picture::<u8>::new(16, 16).unwrap();
    let mut units = records(pu(InterPredIdc::L0, [[0, 0], [0, 0]]));
    units.cus[0].transquant_bypass = true;
    units.tus = vec![TuRecord {
        x: 0,
        y: 0,
        x_base: 0,
        y_base: 0,
        log2_size: 4,
        depth: 0,
        blk_idx: 0,
        cbf_luma: true,
        cbf_cb: false,
        cbf_cr: false,
        cu: 0,
        first_tb: 0,
        tb_count: 1,
    }];
    units.tbs = vec![TbRecord {
        x: 0,
        y: 0,
        log2_size: 4,
        c_idx: 0,
        transform_skip: false,
        first_coeff: 0,
        coeff_count: 1,
    }];
    units.coeffs = vec![Coeff {
        x: 0,
        y: 0,
        level: 100,
    }];
    scene.run(&list, &units, &mut picture);
    assert_eq!(picture.luma().row(0).unwrap()[0], 255);
    assert_eq!(picture.luma().row(0).unwrap()[1], 250);
}

/// Two prediction blocks in one unit: the second reads the first as a neighbour.
/// Nx2N with a left block of ( 8, 0 ), and a right block at the same predictor
/// plus a difference of ( 0, 4 ): mvL0 is ( 8, 4 ).
#[test]
fn a_second_block_reads_the_first() {
    let mut scene = Scene::new();
    let list = refs(Some(frame(0, |x, y| (x + 10 * y) as u8)), None);
    let mut picture = Picture::<u8>::new(16, 16).unwrap();
    let mut left = pu(InterPredIdc::L0, [[8, 0], [0, 0]]);
    left.width = 8;
    let mut right = pu(InterPredIdc::L0, [[0, 4], [0, 0]]);
    right.x = 8;
    right.width = 8;
    let mut unit = cu();
    unit.part_mode = PartMode::NxTwoN;
    let units = CtuRecords {
        cus: vec![unit],
        pus: vec![left, right],
        ..CtuRecords::default()
    };
    scene.run(&list, &units, &mut picture);
    assert_eq!(scene.field.at(0, 0).mv[0], Mv::new(8, 0));
    assert_eq!(scene.field.at(8, 0).mv[0], Mv::new(8, 4));
    // The right block at ( 8, 2 ) reads the reference at ( 10, 3 ).
    assert_eq!(picture.luma().row(2).unwrap()[8], 10 + 30);
}

/// An intra unit in the same CTU clears the motion of its area.
#[test]
fn an_intra_unit_stores_no_motion() {
    let mut scene = Scene::new();
    let list = refs(None, None);
    scene.field.fill(
        0,
        0,
        16,
        16,
        PbMotion {
            mv: [Mv::new(1, 1); 2],
            ref_idx: [0, 0],
        },
    );
    let mut picture = Picture::<u8>::new(16, 16).unwrap();
    let mut unit = cu();
    unit.pred_mode = PredMode::Intra;
    let units = CtuRecords {
        cus: vec![unit],
        ..CtuRecords::default()
    };
    scene.run(&list, &units, &mut picture);
    assert_eq!(scene.field.at(4, 4), PbMotion::NONE);
}
