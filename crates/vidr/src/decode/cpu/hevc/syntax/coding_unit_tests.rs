//! Coding quadtree and coding unit syntax, through encoded bins.

use crate::decode::cpu::hevc::cabac::context as ctx;
use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::params::{CodingParams, PcmParams, SliceKind};

use super::records::{PartMode, PredMode, PuRect};
use super::test_residual_writer::write_dc;
use super::test_writer::{Harness, SyntaxWriter};

/// `part_mode` with one bin, a prediction flag, and `mpm_idx` 0. Then chroma mode
/// 4, and all coded block flags 0 at depth 0.
fn intra_leaf(writer: &mut SyntaxWriter, at_min_size: bool) {
    if at_min_size {
        writer.ctx(ctx::PART_MODE, true);
    }
    writer.ctx(ctx::PREV_INTRA_LUMA_PRED, true);
    writer.bypass(false);
    intra_leaf_end(writer);
}

/// Chroma mode 4 and all coded block flags 0 at depth 0.
fn intra_leaf_end(writer: &mut SyntaxWriter) {
    writer.ctx(ctx::INTRA_CHROMA_PRED_MODE, false);
    writer.ctx(ctx::CBF_CHROMA, false);
    writer.ctx(ctx::CBF_CHROMA, false);
    writer.ctx(ctx::CBF_LUMA + 1, false);
}

fn p_params() -> CodingParams {
    CodingParams {
        kind: SliceKind::P,
        init_type: 1,
        max_merge_cand: 1,
        ..CodingParams::for_tests()
    }
}

#[test]
fn a_split_root_gives_four_intra_units() {
    let params = CodingParams::for_tests();
    let mut writer = SyntaxWriter::new(&params);
    writer.ctx(ctx::SPLIT_CU, true);
    for _ in 0..4 {
        intra_leaf(&mut writer, true);
    }
    let records = Harness::new(params).run(writer, &[(0, 0)]);
    let cus = &records[0].cus;
    assert_eq!(cus.len(), 4);
    let locations: Vec<_> = cus.iter().map(|cu| (cu.x, cu.y, cu.log2_size)).collect();
    assert_eq!(locations, [(0, 0, 3), (8, 0, 3), (0, 8, 3), (8, 8, 3)]);
    assert!(cus.iter().all(|cu| cu.pred_mode == PredMode::Intra));
    assert!(cus.iter().all(|cu| cu.part_mode == PartMode::TwoNxTwoN));
    assert!(cus.iter().all(|cu| cu.qp_y == 26));
    // By hand from 8.4.2. The first unit has no neighbours: candModeList is
    // 0, 1, 26. The second has A = 0 and B = DC above the CTB. The third has
    // A = DC and B = 0: candModeList opens with 1. The fourth has A = 1, B = 0.
    let modes: Vec<_> = cus.iter().map(|cu| cu.intra_luma[0]).collect();
    assert_eq!(modes, [0, 0, 1, 1]);
    assert_eq!(records[0].tus.len(), 4);
    assert!(records[0].tus.iter().all(|tu| !tu.cbf_luma));
    assert!(records[0].tbs.is_empty());
}

/// `ctxInc` of `split_cu_flag` counts the deeper neighbours (Table 9-49).
#[test]
fn split_flag_contexts_read_left_and_above_depth() {
    let params = CodingParams {
        width: 32,
        height: 32,
        ..CodingParams::for_tests()
    };
    let mut writer = SyntaxWriter::new(&params);
    // CTB 0: no neighbours, then a split to depth 1.
    writer.ctx(ctx::SPLIT_CU, true);
    for _ in 0..4 {
        intra_leaf(&mut writer, true);
    }
    // CTB 1: the left CTB is deeper.
    writer.ctx(ctx::SPLIT_CU + 1, false);
    intra_leaf(&mut writer, false);
    // CTB 2: the CTB above is deeper.
    writer.ctx(ctx::SPLIT_CU + 1, false);
    intra_leaf(&mut writer, false);
    // CTB 3: neither neighbour is deeper, and this CTB splits.
    writer.ctx(ctx::SPLIT_CU, true);
    for _ in 0..4 {
        intra_leaf(&mut writer, true);
    }
    let ctus = [(0, 0), (16, 0), (0, 16), (16, 16)];
    let records = Harness::new(params).run(writer, &ctus);
    assert_eq!(records[1].cus[0].log2_size, 4);
    assert_eq!(records[2].cus[0].log2_size, 4);
    assert_eq!(records[3].cus.len(), 4);
}

#[test]
fn both_deeper_neighbours_select_context_2() {
    let params = CodingParams {
        width: 32,
        height: 32,
        ..CodingParams::for_tests()
    };
    let mut writer = SyntaxWriter::new(&params);
    let split_ctb = |writer: &mut SyntaxWriter, inc: usize| {
        writer.ctx(ctx::SPLIT_CU + inc, true);
        for _ in 0..4 {
            intra_leaf(writer, true);
        }
    };
    split_ctb(&mut writer, 0);
    split_ctb(&mut writer, 1);
    split_ctb(&mut writer, 1);
    split_ctb(&mut writer, 2);
    let ctus = [(0, 0), (16, 0), (0, 16), (16, 16)];
    let records = Harness::new(params).run(writer, &ctus);
    assert_eq!(records[3].cus.len(), 4);
}

/// At the right edge the split flag is absent and inferred as 1 (7.4.9.4).
#[test]
fn a_ctb_across_the_edge_splits_with_no_flag() {
    let params = CodingParams {
        width: 24,
        height: 16,
        ..CodingParams::for_tests()
    };
    let mut writer = SyntaxWriter::new(&params);
    writer.ctx(ctx::SPLIT_CU, false);
    intra_leaf(&mut writer, false);
    intra_leaf(&mut writer, true);
    intra_leaf(&mut writer, true);
    let records = Harness::new(params).run(writer, &[(0, 0), (16, 0)]);
    assert_eq!(records[0].cus.len(), 1);
    let locations: Vec<_> = records[1].cus.iter().map(|cu| (cu.x, cu.y)).collect();
    assert_eq!(locations, [(16, 0), (16, 8)]);
}

#[derive(Clone, Copy)]
enum Bin {
    /// A bin on `part_mode` context `ctxInc`.
    Part(usize, bool),
    Bypass(bool),
}

/// Table 9-45 for one coding unit at the root of its CTB. Each prediction unit
/// merges. 2Nx2N merge infers `rqt_root_cbf` as 1, and the tree holds one luma
/// level. Other modes write `rqt_root_cbf` 0.
fn check_part_mode(
    ctb_log2: u32,
    min_cb_log2: u32,
    amp_enabled: bool,
    bins: &[Bin],
    part: PartMode,
) {
    let size = 1u32 << ctb_log2;
    let params = CodingParams {
        width: size,
        height: size,
        ctb_log2,
        min_cb_log2,
        max_tb_log2: ctb_log2.min(5),
        amp_enabled,
        log2_min_qg_size: ctb_log2,
        ..p_params()
    };
    let mut writer = SyntaxWriter::new(&params);
    if ctb_log2 > min_cb_log2 {
        writer.ctx(ctx::SPLIT_CU, false);
    }
    writer.ctx(ctx::CU_SKIP, false);
    writer.ctx(ctx::PRED_MODE, false);
    for bin in bins {
        match *bin {
            Bin::Part(inc, value) => writer.ctx(ctx::PART_MODE + inc, value),
            Bin::Bypass(value) => writer.bypass(value),
        }
    }
    let (blocks, count) = part.prediction_blocks(size);
    for _ in 0..count {
        writer.ctx(ctx::MERGE_FLAG, true);
    }
    if part == PartMode::TwoNxTwoN {
        writer.ctx(ctx::CBF_CHROMA, false);
        writer.ctx(ctx::CBF_CHROMA, false);
        write_dc(&mut writer, ctb_log2, 0);
    } else {
        writer.ctx(ctx::RQT_ROOT_CBF, false);
    }
    let records = Harness::new(params).run(writer, &[(0, 0)]);
    let cu = &records[0].cus[0];
    assert_eq!(cu.part_mode, part);
    assert_eq!(cu.pred_mode, PredMode::Inter);
    let found: Vec<_> = records[0]
        .pus
        .iter()
        .map(|pu| PuRect {
            x: pu.x,
            y: pu.y,
            width: pu.width,
            height: pu.height,
        })
        .collect();
    assert_eq!(found, blocks[..count]);
    assert_eq!(
        records[0].tus.len(),
        usize::from(part == PartMode::TwoNxTwoN)
    );
}

#[test]
fn part_mode_above_the_minimum_with_amp_follows_table_9_45() {
    use Bin::{Bypass, Part};
    let cases = [
        (vec![Part(0, true)], PartMode::TwoNxTwoN),
        (
            vec![Part(0, false), Part(1, true), Part(3, true)],
            PartMode::TwoNxN,
        ),
        (
            vec![Part(0, false), Part(1, false), Part(3, true)],
            PartMode::NxTwoN,
        ),
        (
            vec![Part(0, false), Part(1, true), Part(3, false), Bypass(false)],
            PartMode::TwoNxnU,
        ),
        (
            vec![Part(0, false), Part(1, true), Part(3, false), Bypass(true)],
            PartMode::TwoNxnD,
        ),
        (
            vec![
                Part(0, false),
                Part(1, false),
                Part(3, false),
                Bypass(false),
            ],
            PartMode::NLxTwoN,
        ),
        (
            vec![Part(0, false), Part(1, false), Part(3, false), Bypass(true)],
            PartMode::NRxTwoN,
        ),
    ];
    for (bins, part) in cases {
        check_part_mode(4, 3, true, &bins, part);
    }
}

#[test]
fn part_mode_above_the_minimum_with_no_amp_follows_table_9_45() {
    use Bin::Part;
    check_part_mode(4, 3, false, &[Part(0, true)], PartMode::TwoNxTwoN);
    check_part_mode(
        4,
        3,
        false,
        &[Part(0, false), Part(1, true)],
        PartMode::TwoNxN,
    );
    check_part_mode(
        4,
        3,
        false,
        &[Part(0, false), Part(1, false)],
        PartMode::NxTwoN,
    );
}

#[test]
fn part_mode_at_8x8_has_no_nxn() {
    use Bin::Part;
    check_part_mode(3, 3, false, &[Part(0, true)], PartMode::TwoNxTwoN);
    check_part_mode(
        3,
        3,
        false,
        &[Part(0, false), Part(1, true)],
        PartMode::TwoNxN,
    );
    check_part_mode(
        3,
        3,
        false,
        &[Part(0, false), Part(1, false)],
        PartMode::NxTwoN,
    );
}

#[test]
fn part_mode_at_a_larger_minimum_has_nxn() {
    use Bin::Part;
    check_part_mode(4, 4, false, &[Part(0, true)], PartMode::TwoNxTwoN);
    check_part_mode(
        4,
        4,
        false,
        &[Part(0, false), Part(1, true)],
        PartMode::TwoNxN,
    );
    check_part_mode(
        4,
        4,
        false,
        &[Part(0, false), Part(1, false), Part(2, true)],
        PartMode::NxTwoN,
    );
    check_part_mode(
        4,
        4,
        false,
        &[Part(0, false), Part(1, false), Part(2, false)],
        PartMode::NxN,
    );
}

/// `ctxInc` of `cu_skip_flag` counts the skipped neighbours (Table 9-49).
#[test]
fn skip_flag_contexts_read_left_and_above() {
    let params = CodingParams {
        max_merge_cand: 2,
        ..p_params()
    };
    let mut writer = SyntaxWriter::new(&params);
    writer.ctx(ctx::SPLIT_CU, true);
    // Unit 0, no neighbours: skip with merge_idx 1.
    writer.ctx(ctx::CU_SKIP, true);
    writer.ctx(ctx::MERGE_IDX, true);
    // Unit 1, left is skipped: skip with merge_idx 0.
    writer.ctx(ctx::CU_SKIP + 1, true);
    writer.ctx(ctx::MERGE_IDX, false);
    // Unit 2, above is skipped: an intra unit.
    writer.ctx(ctx::CU_SKIP + 1, false);
    writer.ctx(ctx::PRED_MODE, true);
    intra_leaf(&mut writer, true);
    // Unit 3, above is skipped and left is not: skip.
    writer.ctx(ctx::CU_SKIP + 1, true);
    writer.ctx(ctx::MERGE_IDX, false);
    let records = Harness::new(params).run(writer, &[(0, 0)]);
    let modes: Vec<_> = records[0].cus.iter().map(|cu| cu.pred_mode).collect();
    assert_eq!(
        modes,
        [
            PredMode::Skip,
            PredMode::Skip,
            PredMode::Intra,
            PredMode::Skip
        ]
    );
    let merges: Vec<_> = records[0].pus.iter().map(|pu| pu.merge_idx).collect();
    assert_eq!(merges, [1, 0, 0]);
    assert!(records[0].pus.iter().all(|pu| pu.merge_flag));
    // A skipped unit parses no transform tree.
    assert_eq!(records[0].tus.len(), 1);
    assert_eq!(records[0].cus[3].first_pu, 2);
}

#[test]
fn both_skipped_neighbours_select_context_2() {
    let params = p_params();
    let mut writer = SyntaxWriter::new(&params);
    writer.ctx(ctx::SPLIT_CU, true);
    writer.ctx(ctx::CU_SKIP, true);
    writer.ctx(ctx::CU_SKIP + 1, true);
    writer.ctx(ctx::CU_SKIP + 1, true);
    writer.ctx(ctx::CU_SKIP + 2, true);
    let records = Harness::new(params).run(writer, &[(0, 0)]);
    assert_eq!(records[0].cus.len(), 4);
}

/// Worked by hand from 8.4.2 and 8.4.3, in the 8x8 CTB of a test picture.
#[test]
fn nxn_modes_follow_the_candidate_lists() {
    let params = CodingParams {
        width: 8,
        height: 8,
        ctb_log2: 3,
        max_tb_log2: 3,
        log2_min_qg_size: 3,
        ..CodingParams::for_tests()
    };
    let mut writer = SyntaxWriter::new(&params);
    writer.ctx(ctx::PART_MODE, false);
    for prev_flag in [true, false, true, false] {
        writer.ctx(ctx::PREV_INTRA_LUMA_PRED, prev_flag);
    }
    // Block 0: candModeList 0, 1, 26. mpm_idx 2 gives 26.
    writer.bypass(true);
    writer.bypass(true);
    // Block 1: A = 26, B = DC gives 26, 1, 0. rem 5 gives 7.
    writer.bits(5, 5);
    // Block 2: A = DC, B = 26 gives 1, 26, 0. mpm_idx 1 gives 26.
    writer.bypass(true);
    writer.bypass(false);
    // Block 3: A = 26, B = 7 gives 26, 7, 0. rem 0 gives 1.
    writer.bits(0, 5);
    // intra_chroma_pred_mode 2 is mode 10.
    writer.ctx(ctx::INTRA_CHROMA_PRED_MODE, true);
    writer.bits(2, 2);
    // The tree splits with no flag. Chroma flags at depth 0, luma at depth 1.
    writer.ctx(ctx::CBF_CHROMA, false);
    writer.ctx(ctx::CBF_CHROMA, false);
    for _ in 0..4 {
        writer.ctx(ctx::CBF_LUMA, false);
    }
    let records = Harness::new(params).run(writer, &[(0, 0)]);
    let cu = &records[0].cus[0];
    assert_eq!(cu.part_mode, PartMode::NxN);
    assert_eq!(cu.intra_luma, [26, 7, 26, 1]);
    assert_eq!(cu.intra_chroma, 10);
    let tus: Vec<_> = records[0]
        .tus
        .iter()
        .map(|tu| (tu.x, tu.y, tu.log2_size, tu.depth, tu.blk_idx))
        .collect();
    assert_eq!(
        tus,
        [
            (0, 0, 2, 1, 0),
            (4, 0, 2, 1, 1),
            (0, 4, 2, 1, 2),
            (4, 4, 2, 1, 3)
        ]
    );
}

/// The candidate above the CTB is `INTRA_DC`, even when it is available (8.4.2).
#[test]
fn a_mode_above_the_ctb_is_not_a_candidate() {
    let params = CodingParams {
        width: 8,
        height: 16,
        ctb_log2: 3,
        max_tb_log2: 3,
        log2_min_qg_size: 3,
        ..CodingParams::for_tests()
    };
    let mut writer = SyntaxWriter::new(&params);
    // CTB 0: mpm_idx 2 gives 26.
    writer.ctx(ctx::PART_MODE, true);
    writer.ctx(ctx::PREV_INTRA_LUMA_PRED, true);
    writer.bypass(true);
    writer.bypass(true);
    intra_leaf_end(&mut writer);
    // CTB 1: A and B are both DC, so candModeList is 0, 1, 26, and mpm_idx 1 gives
    // 1. Taking B from the unit above gives 1, 26, 0, and mode 26.
    writer.ctx(ctx::PART_MODE, true);
    writer.ctx(ctx::PREV_INTRA_LUMA_PRED, true);
    writer.bypass(true);
    writer.bypass(false);
    intra_leaf_end(&mut writer);
    let records = Harness::new(params).run(writer, &[(0, 0), (0, 8)]);
    assert_eq!(records[0].cus[0].intra_luma, [26; 4]);
    assert_eq!(records[1].cus[0].intra_luma, [1; 4]);
}

/// Table 8-2 replaces a chroma mode equal to the luma mode with 34.
#[test]
fn a_chroma_mode_equal_to_luma_becomes_34() {
    let params = CodingParams::for_tests();
    let mut writer = SyntaxWriter::new(&params);
    writer.ctx(ctx::SPLIT_CU, false);
    // mpm_idx 2 gives 26, and intra_chroma_pred_mode 1 names 26.
    writer.ctx(ctx::PREV_INTRA_LUMA_PRED, true);
    writer.bypass(true);
    writer.bypass(true);
    writer.ctx(ctx::INTRA_CHROMA_PRED_MODE, true);
    writer.bits(1, 2);
    writer.ctx(ctx::CBF_CHROMA, false);
    writer.ctx(ctx::CBF_CHROMA, false);
    writer.ctx(ctx::CBF_LUMA + 1, false);
    let records = Harness::new(params).run(writer, &[(0, 0)]);
    assert_eq!(records[0].cus[0].intra_chroma, 34);
}

fn pcm_params() -> CodingParams {
    CodingParams {
        width: 16,
        height: 8,
        ctb_log2: 3,
        max_tb_log2: 3,
        log2_min_qg_size: 3,
        pcm: Some(PcmParams {
            log2_min_size: 3,
            log2_max_size: 3,
            bit_depth_luma: 5,
            bit_depth_chroma: 3,
        }),
        ..CodingParams::for_tests()
    }
}

/// PCM samples sit between two arithmetic coded runs (9.3.1).
#[test]
fn pcm_samples_read_and_the_engine_restarts() {
    let params = pcm_params();
    let mut writer = SyntaxWriter::new(&params);
    writer.ctx(ctx::PART_MODE, true);
    writer.terminate(true);
    writer.align();
    let mut expected = Vec::new();
    for index in 0..64u32 {
        writer.raw(index % 32, 5);
        expected.push((index % 32) as u16);
    }
    for index in 0..32u32 {
        writer.raw(index % 8, 3);
        expected.push((index % 8) as u16);
    }
    writer.restart();
    // The next unit reads pcm_flag 0 through the restarted engine.
    writer.ctx(ctx::PART_MODE, true);
    writer.terminate(false);
    writer.ctx(ctx::PREV_INTRA_LUMA_PRED, true);
    writer.bypass(false);
    intra_leaf_end(&mut writer);
    let records = Harness::new(params).run(writer, &[(0, 0), (8, 0)]);
    let cu = &records[0].cus[0];
    assert!(cu.pcm);
    assert_eq!(cu.intra_luma, [1; 4]);
    assert!(records[0].tus.is_empty());
    assert_eq!(records[0].pcm_samples, expected);
    // A PCM neighbour gives INTRA_DC: candModeList is 0, 1, 26.
    assert!(!records[1].cus[0].pcm);
    assert_eq!(records[1].cus[0].intra_luma, [0; 4]);
}

/// Sizes outside the PCM range read no `pcm_flag`.
#[test]
fn a_size_outside_the_pcm_range_has_no_flag() {
    let params = CodingParams {
        pcm: Some(PcmParams {
            log2_min_size: 4,
            log2_max_size: 4,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
        }),
        ..CodingParams::for_tests()
    };
    let mut writer = SyntaxWriter::new(&params);
    writer.ctx(ctx::SPLIT_CU, true);
    for _ in 0..4 {
        intra_leaf(&mut writer, true);
    }
    let records = Harness::new(params).run(writer, &[(0, 0)]);
    assert_eq!(records[0].cus.len(), 4);
}

#[test]
fn an_nxn_unit_has_no_pcm_flag() {
    let params = CodingParams {
        width: 8,
        ..pcm_params()
    };
    let mut writer = SyntaxWriter::new(&params);
    writer.ctx(ctx::PART_MODE, false);
    for _ in 0..4 {
        writer.ctx(ctx::PREV_INTRA_LUMA_PRED, true);
    }
    for _ in 0..4 {
        writer.bypass(false);
    }
    writer.ctx(ctx::INTRA_CHROMA_PRED_MODE, false);
    writer.ctx(ctx::CBF_CHROMA, false);
    writer.ctx(ctx::CBF_CHROMA, false);
    for _ in 0..4 {
        writer.ctx(ctx::CBF_LUMA, false);
    }
    let records = Harness::new(params).run(writer, &[(0, 0)]);
    assert_eq!(records[0].cus[0].part_mode, PartMode::NxN);
    assert!(!records[0].cus[0].pcm);
}

#[test]
fn pcm_samples_past_the_end_are_refused() {
    let params = pcm_params();
    let mut writer = SyntaxWriter::new(&params);
    writer.ctx(ctx::PART_MODE, true);
    let bytes = writer.into_bytes();
    let result = Harness::new(params).try_parse(&bytes, &[(0, 0)]);
    assert!(matches!(
        result,
        Err(HevcError::SliceDataTruncated { size, .. }) if size == bytes.len()
    ));
}

#[test]
fn the_transquant_bypass_flag_reads_first() {
    let params = CodingParams {
        transquant_bypass: true,
        ..CodingParams::for_tests()
    };
    let mut writer = SyntaxWriter::new(&params);
    writer.ctx(ctx::SPLIT_CU, false);
    writer.ctx(ctx::CU_TRANSQUANT_BYPASS, true);
    intra_leaf(&mut writer, false);
    let records = Harness::new(params).run(writer, &[(0, 0)]);
    assert!(records[0].cus[0].transquant_bypass);
}

/// Reading past the end of slice data returns the named error.
#[test]
fn truncated_slice_data_is_refused() {
    let params = CodingParams::for_tests();
    let mut harness = Harness::new(params);
    let result = harness.try_parse(&[0], &[(0, 0)]);
    assert!(matches!(
        result,
        Err(HevcError::SliceDataTruncated { size: 1, .. })
    ));
}

/// Arbitrary bytes return records or an error, never a panic. Debug builds also
/// catch any arithmetic overflow here.
#[test]
fn random_bytes_never_panic() {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let base = CodingParams {
        width: 128,
        height: 64,
        ctb_log2: 6,
        min_cb_log2: 3,
        max_tb_log2: 5,
        max_depth_inter: 3,
        max_depth_intra: 3,
        amp_enabled: true,
        pcm: Some(PcmParams {
            log2_min_size: 3,
            log2_max_size: 5,
            bit_depth_luma: 7,
            bit_depth_chroma: 7,
        }),
        bit_depth: 10,
        sign_data_hiding: true,
        transform_skip: true,
        cu_qp_delta: true,
        log2_min_qg_size: 3,
        transquant_bypass: true,
        num_ref_idx_minus1: [15, 15],
        ..CodingParams::for_tests()
    };
    for trial in 0u32..300 {
        let (kind, init_type) = match trial % 3 {
            0 => (SliceKind::I, 0),
            1 => (SliceKind::P, 1),
            _ => (SliceKind::B, 2),
        };
        let params = CodingParams {
            kind,
            init_type,
            mvd_l1_zero: trial & 1 == 1,
            ..base
        };
        let length = (next() % 4096) as usize;
        let bytes: Vec<u8> = (0..length).map(|_| next() as u8).collect();
        let _ = Harness::new(params).try_parse(&bytes, &[(0, 0), (64, 0)]);
    }
}
