//! Transform tree, transform unit, and delta QP syntax, through encoded bins, and
//! the hand-computed QP cases of 8.6.1.

use crate::decode::cpu::hevc::cabac::context as ctx;
use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::params::{CodingParams, SliceKind};

use super::records::CtuRecords;
use super::test_residual_writer::write_dc;
use super::test_writer::{Harness, SyntaxWriter};

fn parse_root(params: CodingParams, writer: SyntaxWriter) -> CtuRecords {
    let mut records = Harness::new(params).run(writer, &[(0, 0)]);
    records.remove(0)
}

/// `prev_intra_luma_pred_flag` 1, `mpm_idx` 0, `intra_chroma_pred_mode` 4.
fn intra_modes(writer: &mut SyntaxWriter) {
    writer.ctx(ctx::PREV_INTRA_LUMA_PRED, true);
    writer.bypass(false);
    writer.ctx(ctx::INTRA_CHROMA_PRED_MODE, false);
}

/// A 16x16 intra unit: the flag and contexts follow `log2TrafoSize` and
/// `trafoDepth` (Table 9-48).
#[test]
fn a_split_tree_reads_chroma_flags_by_depth() {
    let params = CodingParams {
        max_depth_intra: 1,
        ..CodingParams::for_tests()
    };
    let mut writer = SyntaxWriter::new(&params);
    writer.ctx(ctx::SPLIT_CU, false);
    intra_modes(&mut writer);
    // split_transform_flag at log2TrafoSize 4: ctxInc 1.
    writer.ctx(ctx::SPLIT_TRANSFORM + 1, true);
    // Depth 0: Cb coded, Cr not.
    writer.ctx(ctx::CBF_CHROMA, true);
    writer.ctx(ctx::CBF_CHROMA, false);
    // Child 0: Cb coded at depth 1, Cr absent. Luma and Cb residuals.
    writer.ctx(ctx::CBF_CHROMA + 1, true);
    writer.ctx(ctx::CBF_LUMA, true);
    write_dc(&mut writer, 3, 0);
    write_dc(&mut writer, 2, 1);
    // Child 1: no Cb, no luma.
    writer.ctx(ctx::CBF_CHROMA + 1, false);
    writer.ctx(ctx::CBF_LUMA, false);
    // Child 2: no Cb, luma only.
    writer.ctx(ctx::CBF_CHROMA + 1, false);
    writer.ctx(ctx::CBF_LUMA, true);
    write_dc(&mut writer, 3, 0);
    // Child 3: Cb only.
    writer.ctx(ctx::CBF_CHROMA + 1, true);
    writer.ctx(ctx::CBF_LUMA, false);
    write_dc(&mut writer, 2, 1);
    let records = parse_root(params, writer);
    let tus: Vec<_> = records
        .tus
        .iter()
        .map(|tu| (tu.x, tu.y, tu.log2_size, tu.cbf_luma, tu.cbf_cb, tu.cbf_cr))
        .collect();
    assert_eq!(
        tus,
        [
            (0, 0, 3, true, true, false),
            (8, 0, 3, false, false, false),
            (0, 8, 3, true, false, false),
            (8, 8, 3, false, true, false),
        ]
    );
    let tbs: Vec<_> = records
        .tbs
        .iter()
        .map(|tb| (tb.x, tb.y, tb.log2_size, tb.c_idx))
        .collect();
    assert_eq!(
        tbs,
        [(0, 0, 3, 0), (0, 0, 2, 1), (0, 8, 3, 0), (8, 8, 2, 1)]
    );
    let counts: Vec<_> = records.tus.iter().map(|tu| tu.tb_count).collect();
    assert_eq!(counts, [2, 0, 1, 1]);
    assert_eq!(records.tus[3].first_tb, 3);
    assert!(records.tus.iter().all(|tu| tu.cu == 0));
}

/// A 4x4 luma split keeps chroma with the parent: it parses after block 3, at the
/// parent's location, with the flags of depth `trafoDepth - 1` (7.3.8.10). The
/// delta QP reads at block 0, because `cbfChroma` holds for every child.
#[test]
fn chroma_of_a_4x4_split_parses_after_block_3() {
    let params = CodingParams {
        width: 8,
        height: 8,
        ctb_log2: 3,
        max_tb_log2: 3,
        cu_qp_delta: true,
        log2_min_qg_size: 3,
        ..CodingParams::for_tests()
    };
    let mut writer = SyntaxWriter::new(&params);
    // part_mode NxN, four flags, four mpm_idx values of 0, chroma mode 4.
    writer.ctx(ctx::PART_MODE, false);
    for _ in 0..4 {
        writer.ctx(ctx::PREV_INTRA_LUMA_PRED, true);
    }
    writer.bits(0, 4);
    writer.ctx(ctx::INTRA_CHROMA_PRED_MODE, false);
    // Root flags at depth 0, and no split flag.
    writer.ctx(ctx::CBF_CHROMA, true);
    writer.ctx(ctx::CBF_CHROMA, true);
    // Block 0: no luma. cu_qp_delta_abs 1 with a negative sign.
    writer.ctx(ctx::CBF_LUMA, false);
    writer.ctx(ctx::CU_QP_DELTA_ABS, true);
    writer.ctx(ctx::CU_QP_DELTA_ABS + 1, false);
    writer.bypass(true);
    // Block 1: luma.
    writer.ctx(ctx::CBF_LUMA, true);
    write_dc(&mut writer, 2, 0);
    // Block 2: nothing, and the delta stays coded.
    writer.ctx(ctx::CBF_LUMA, false);
    // Block 3: no luma, then Cb and Cr at 4x4.
    writer.ctx(ctx::CBF_LUMA, false);
    write_dc(&mut writer, 2, 1);
    write_dc(&mut writer, 2, 2);
    let records = parse_root(params, writer);
    let tbs: Vec<_> = records
        .tbs
        .iter()
        .map(|tb| (tb.x, tb.y, tb.log2_size, tb.c_idx))
        .collect();
    assert_eq!(tbs, [(4, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2)]);
    assert!(records.tus.iter().all(|tu| tu.cbf_cb && tu.cbf_cr));
    assert_eq!(records.tus[3].tb_count, 2);
    assert_eq!((records.tus[3].x_base, records.tus[3].y_base), (0, 0));
    assert_eq!(records.cus[0].qp_y, 25);
}

fn inter_params() -> CodingParams {
    CodingParams {
        kind: SliceKind::P,
        init_type: 1,
        min_cb_log2: 4,
        max_merge_cand: 1,
        ..CodingParams::for_tests()
    }
}

/// `cu_skip_flag` 0, `pred_mode_flag` 0.
fn start_inter(writer: &mut SyntaxWriter) {
    writer.ctx(ctx::CU_SKIP, false);
    writer.ctx(ctx::PRED_MODE, false);
}

/// An inter tree at depth 0 with no chroma infers `cbf_luma` as 1 (7.4.9.8).
#[test]
fn cbf_luma_is_inferred_with_no_chroma() {
    let params = inter_params();
    let mut writer = SyntaxWriter::new(&params);
    start_inter(&mut writer);
    writer.ctx(ctx::PART_MODE, true);
    writer.ctx(ctx::MERGE_FLAG, true);
    writer.ctx(ctx::CBF_CHROMA, false);
    writer.ctx(ctx::CBF_CHROMA, false);
    write_dc(&mut writer, 4, 0);
    let records = parse_root(params, writer);
    assert!(records.tus[0].cbf_luma);
    assert_eq!(records.tbs.len(), 1);
}

/// `rqt_root_cbf` of 1 on a non-merge unit opens the tree.
#[test]
fn rqt_root_cbf_opens_the_tree() {
    let params = inter_params();
    let mut writer = SyntaxWriter::new(&params);
    start_inter(&mut writer);
    writer.ctx(ctx::PART_MODE, true);
    writer.ctx(ctx::MERGE_FLAG, false);
    for _ in 0..2 {
        writer.ctx(ctx::ABS_MVD_GREATER0, false);
    }
    writer.ctx(ctx::MVP_FLAG, false);
    writer.ctx(ctx::RQT_ROOT_CBF, true);
    writer.ctx(ctx::CBF_CHROMA, true);
    writer.ctx(ctx::CBF_CHROMA, false);
    writer.ctx(ctx::CBF_LUMA + 1, false);
    write_dc(&mut writer, 3, 1);
    let records = parse_root(params, writer);
    assert_eq!(records.tus.len(), 1);
    assert_eq!(records.tbs[0].c_idx, 1);
    assert_eq!(records.tbs[0].log2_size, 3);
}

/// `interSplitFlag`: depth limit 0 and a non-2Nx2N unit split with no flag.
#[test]
fn inter_split_is_inferred() {
    let params = inter_params();
    let mut writer = SyntaxWriter::new(&params);
    start_inter(&mut writer);
    // part_mode 2NxN at a minimum size of 16: 01.
    writer.ctx(ctx::PART_MODE, false);
    writer.ctx(ctx::PART_MODE + 1, true);
    writer.ctx(ctx::MERGE_FLAG, true);
    writer.ctx(ctx::MERGE_FLAG, true);
    writer.ctx(ctx::RQT_ROOT_CBF, true);
    writer.ctx(ctx::CBF_CHROMA, false);
    writer.ctx(ctx::CBF_CHROMA, false);
    for _ in 0..4 {
        writer.ctx(ctx::CBF_LUMA, false);
    }
    let records = parse_root(params, writer);
    assert_eq!(records.tus.len(), 4);
    assert!(
        records
            .tus
            .iter()
            .all(|tu| tu.depth == 1 && tu.log2_size == 3)
    );
}

/// A block above `MaxTbLog2SizeY` splits with no flag, and still reads its chroma
/// flags.
#[test]
fn a_block_above_the_largest_transform_splits_with_no_flag() {
    let params = CodingParams {
        width: 32,
        height: 32,
        ctb_log2: 5,
        min_cb_log2: 5,
        max_tb_log2: 4,
        log2_min_qg_size: 5,
        ..CodingParams::for_tests()
    };
    let mut writer = SyntaxWriter::new(&params);
    writer.ctx(ctx::PART_MODE, true);
    intra_modes(&mut writer);
    writer.ctx(ctx::CBF_CHROMA, false);
    writer.ctx(ctx::CBF_CHROMA, false);
    for _ in 0..4 {
        writer.ctx(ctx::CBF_LUMA, false);
    }
    let records = parse_root(params, writer);
    assert_eq!(records.tus.len(), 4);
    assert!(records.tus.iter().all(|tu| tu.log2_size == 4));
}

/// The bins of `cu_qp_delta_abs` and its sign (9.3.3.10, Table 9-48).
fn write_delta(writer: &mut SyntaxWriter, delta: i32) {
    let abs = delta.unsigned_abs();
    writer.ctx(ctx::CU_QP_DELTA_ABS, abs > 0);
    if abs == 0 {
        return;
    }
    for prefix in 1..5 {
        writer.ctx(ctx::CU_QP_DELTA_ABS + 1, abs > prefix);
        if abs == prefix {
            break;
        }
    }
    if abs >= 5 {
        writer.exp_golomb(abs - 5, 0);
    }
    writer.bypass(delta < 0);
}

/// A unit with one luma level and `CuQpDeltaVal` `delta`.
fn delta_unit(params: &CodingParams, delta: i32) -> SyntaxWriter {
    let mut writer = SyntaxWriter::new(params);
    writer.ctx(ctx::SPLIT_CU, false);
    intra_modes(&mut writer);
    writer.ctx(ctx::CBF_CHROMA, false);
    writer.ctx(ctx::CBF_CHROMA, false);
    writer.ctx(ctx::CBF_LUMA + 1, true);
    write_delta(&mut writer, delta);
    write_dc(&mut writer, 4, 0);
    writer
}

fn delta_params() -> CodingParams {
    CodingParams {
        cu_qp_delta: true,
        ..CodingParams::for_tests()
    }
}

#[test]
fn delta_qp_values_round_trip() {
    for delta in [0, 1, -1, 2, 4, -4, 5, -5, 6, 7, 12, -25, 25, -26] {
        let params = delta_params();
        let writer = delta_unit(&params, delta);
        let records = parse_root(params, writer);
        // (8-283) at 8 bits with qPY_PRED 26.
        assert_eq!(records.cus[0].qp_y, (26 + delta + 52) % 52, "delta {delta}");
    }
}

/// CuQpDeltaVal stops at -( 26 + QpBdOffsetY / 2 ) and 25 + QpBdOffsetY / 2.
#[test]
fn delta_qp_out_of_range_is_refused() {
    for delta in [26, -27] {
        let params = delta_params();
        let bytes = delta_unit(&params, delta).into_bytes();
        let result = Harness::new(params).try_parse(&bytes, &[(0, 0)]);
        assert_eq!(
            result,
            Err(HevcError::InvalidSyntax {
                element: "CuQpDeltaVal",
                value: i64::from(delta),
                allowed: "-( 26 + QpBdOffsetY / 2 ) to 25 + QpBdOffsetY / 2",
            })
        );
    }
}

/// The range widens with bit depth, at 10 bits: -32 to 31.
#[test]
fn delta_qp_range_follows_the_bit_depth() {
    let params = CodingParams {
        bit_depth: 10,
        ..delta_params()
    };
    let writer = delta_unit(&params, 31);
    let records = parse_root(params, writer);
    // ( 26 + 31 + 52 + 24 ) % 64 = 5, minus 12 is -7.
    assert_eq!(records.cus[0].qp_y, -7);
}

/// Four 8x8 units with a quantization group each. By hand from 8.6.1:
/// - Unit 0: no neighbours in the CTB, qPY_PREV 26, delta +2 gives 28.
/// - Unit 1: A = 28 and B takes qPY_PREV 28. Delta -4 gives 24.
/// - Unit 2: A takes qPY_PREV 24 and B = 28: ( 24 + 28 + 1 ) >> 1 = 26, no delta.
/// - Unit 3: A = 26, B = 24: ( 26 + 24 + 1 ) >> 1 = 25, delta +1 gives 26.
#[test]
fn each_quantization_group_predicts_from_its_neighbours() {
    let params = CodingParams {
        cu_qp_delta: true,
        log2_min_qg_size: 3,
        ..CodingParams::for_tests()
    };
    let mut writer = SyntaxWriter::new(&params);
    writer.ctx(ctx::SPLIT_CU, true);
    for delta in [Some(2), Some(-4), None, Some(1)] {
        writer.ctx(ctx::PART_MODE, true);
        intra_modes(&mut writer);
        writer.ctx(ctx::CBF_CHROMA, false);
        writer.ctx(ctx::CBF_CHROMA, false);
        writer.ctx(ctx::CBF_LUMA + 1, delta.is_some());
        if let Some(delta) = delta {
            write_delta(&mut writer, delta);
            write_dc(&mut writer, 3, 0);
        }
    }
    let records = parse_root(params, writer);
    let qps: Vec<_> = records.cus.iter().map(|cu| cu.qp_y).collect();
    assert_eq!(qps, [28, 24, 26, 26]);
}

/// One quantization group over four units. They share qPY_PRED 26. The delta
/// parses once, and it applies from its unit onward (7.3.8.14, 8.6.1).
#[test]
fn one_group_shares_its_prediction_and_delta() {
    let params = delta_params();
    let mut writer = SyntaxWriter::new(&params);
    writer.ctx(ctx::SPLIT_CU, true);
    for (coded, delta) in [(false, None), (true, Some(3)), (false, None), (true, None)] {
        writer.ctx(ctx::PART_MODE, true);
        intra_modes(&mut writer);
        writer.ctx(ctx::CBF_CHROMA, false);
        writer.ctx(ctx::CBF_CHROMA, false);
        writer.ctx(ctx::CBF_LUMA + 1, coded);
        if let Some(delta) = delta {
            write_delta(&mut writer, delta);
        }
        if coded {
            write_dc(&mut writer, 3, 0);
        }
    }
    let records = parse_root(params, writer);
    let qps: Vec<_> = records.cus.iter().map(|cu| cu.qp_y).collect();
    assert_eq!(qps, [26, 29, 29, 29]);
}

/// A neighbour in another CTB gives way to qPY_PREV (8-280).
#[test]
fn a_neighbour_in_another_ctb_is_not_read() {
    let params = CodingParams {
        width: 32,
        height: 16,
        ..delta_params()
    };
    let mut writer = delta_unit(&params, 4);
    // CTB 1 is one unit with no residual: QpY is qPY_PREV, which is 30.
    writer.ctx(ctx::SPLIT_CU, false);
    intra_modes(&mut writer);
    writer.ctx(ctx::CBF_CHROMA, false);
    writer.ctx(ctx::CBF_CHROMA, false);
    writer.ctx(ctx::CBF_LUMA + 1, false);
    let records = Harness::new(params).run(writer, &[(0, 0), (16, 0)]);
    assert_eq!(records[0].cus[0].qp_y, 30);
    assert_eq!(records[1].cus[0].qp_y, 30);
}

/// A bypass unit still reads the delta (7.3.8.10).
#[test]
fn a_transquant_bypass_unit_reads_the_delta() {
    let params = CodingParams {
        transquant_bypass: true,
        ..delta_params()
    };
    let mut writer = SyntaxWriter::new(&params);
    writer.ctx(ctx::SPLIT_CU, false);
    writer.ctx(ctx::CU_TRANSQUANT_BYPASS, true);
    intra_modes(&mut writer);
    writer.ctx(ctx::CBF_CHROMA, false);
    writer.ctx(ctx::CBF_CHROMA, false);
    writer.ctx(ctx::CBF_LUMA + 1, true);
    write_delta(&mut writer, 1);
    write_dc(&mut writer, 4, 0);
    let records = parse_root(params, writer);
    assert!(records.cus[0].transquant_bypass);
    assert_eq!(records.cus[0].qp_y, 27);
}
