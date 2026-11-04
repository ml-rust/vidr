//! Prediction unit and motion vector difference syntax, through encoded bins.

use crate::decode::cpu::hevc::cabac::context as ctx;
use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::params::{CodingParams, SliceKind};

use super::records::{InterPredIdc, PartMode, PuRecord};
use super::test_residual_writer::write_dc;
use super::test_writer::{Harness, SyntaxWriter};

/// One 16x16 coding unit at the root of its CTB.
fn inter_params(kind: SliceKind) -> CodingParams {
    CodingParams {
        kind,
        init_type: if kind == SliceKind::P { 1 } else { 2 },
        min_cb_log2: 4,
        ..CodingParams::for_tests()
    }
}

/// `cu_skip_flag` 0, `pred_mode_flag` 0, and `part_mode` 2Nx2N.
fn start_inter_cu(writer: &mut SyntaxWriter) {
    writer.ctx(ctx::CU_SKIP, false);
    writer.ctx(ctx::PRED_MODE, false);
    writer.ctx(ctx::PART_MODE, true);
}

/// The tree after a 2Nx2N merge unit, which infers `rqt_root_cbf` as 1: no chroma,
/// so `cbf_luma` is inferred as 1, and one luma level.
fn merged_tree(writer: &mut SyntaxWriter) {
    writer.ctx(ctx::CBF_CHROMA, false);
    writer.ctx(ctx::CBF_CHROMA, false);
    write_dc(writer, 4, 0);
}

/// `mvd_coding()` for `[ x, y ]`, from 7.3.8.9 and Table 9-48.
fn write_mvd(writer: &mut SyntaxWriter, mvd: [i32; 2]) {
    let abs = mvd.map(i32::unsigned_abs);
    writer.ctx(ctx::ABS_MVD_GREATER0, abs[0] > 0);
    writer.ctx(ctx::ABS_MVD_GREATER0, abs[1] > 0);
    if abs[0] > 0 {
        writer.ctx(ctx::ABS_MVD_GREATER1, abs[0] > 1);
    }
    if abs[1] > 0 {
        writer.ctx(ctx::ABS_MVD_GREATER1, abs[1] > 1);
    }
    for (component, &magnitude) in mvd.iter().zip(&abs) {
        if magnitude > 0 {
            if magnitude > 1 {
                writer.exp_golomb(magnitude - 2, 1);
            }
            writer.bypass(*component < 0);
        }
    }
}

fn parse_one(params: CodingParams, writer: SyntaxWriter) -> PuRecord {
    let records = Harness::new(params).run(writer, &[(0, 0)]);
    assert_eq!(records[0].pus.len(), 1);
    records[0].pus[0]
}

/// `merge_idx`: the first bin on its context, the others in bypass (Table 9-48).
#[test]
fn merge_idx_follows_its_truncated_rice_code() {
    let cases: [(u32, &[bool]); 5] = [
        (0, &[]),
        (1, &[false]),
        (2, &[true, false]),
        (3, &[true, true, false]),
        (4, &[true, true, true]),
    ];
    for (value, bypass_bins) in cases {
        let params = inter_params(SliceKind::P);
        let mut writer = SyntaxWriter::new(&params);
        start_inter_cu(&mut writer);
        writer.ctx(ctx::MERGE_FLAG, true);
        writer.ctx(ctx::MERGE_IDX, value > 0);
        for &bin in bypass_bins {
            writer.bypass(bin);
        }
        merged_tree(&mut writer);
        let pu = parse_one(params, writer);
        assert!(pu.merge_flag);
        assert_eq!(pu.merge_idx, value);
    }
}

#[test]
fn one_merge_candidate_reads_no_merge_idx() {
    let params = CodingParams {
        max_merge_cand: 1,
        ..inter_params(SliceKind::P)
    };
    let mut writer = SyntaxWriter::new(&params);
    start_inter_cu(&mut writer);
    writer.ctx(ctx::MERGE_FLAG, true);
    merged_tree(&mut writer);
    let pu = parse_one(params, writer);
    assert_eq!(pu.merge_idx, 0);
}

#[test]
fn a_p_slice_reads_list_0_motion() {
    let params = inter_params(SliceKind::P);
    let mut writer = SyntaxWriter::new(&params);
    start_inter_cu(&mut writer);
    writer.ctx(ctx::MERGE_FLAG, false);
    write_mvd(&mut writer, [5, -300]);
    writer.ctx(ctx::MVP_FLAG, true);
    writer.ctx(ctx::RQT_ROOT_CBF, false);
    let pu = parse_one(params, writer);
    assert!(!pu.merge_flag);
    assert_eq!(pu.inter_pred_idc, InterPredIdc::L0);
    assert_eq!(pu.mvd[0], [5, -300]);
    assert_eq!(pu.mvd[1], [0, 0]);
    assert_eq!(pu.mvp_flag, [true, false]);
}

#[test]
fn small_motion_vector_differences_use_only_their_flags() {
    for mvd in [
        [0, 0],
        [1, 0],
        [0, -1],
        [-1, 1],
        [2, -2],
        [3, 0],
        [32_767, -32_768],
    ] {
        let params = inter_params(SliceKind::P);
        let mut writer = SyntaxWriter::new(&params);
        start_inter_cu(&mut writer);
        writer.ctx(ctx::MERGE_FLAG, false);
        write_mvd(&mut writer, mvd);
        writer.ctx(ctx::MVP_FLAG, false);
        writer.ctx(ctx::RQT_ROOT_CBF, false);
        let pu = parse_one(params, writer);
        assert_eq!(pu.mvd[0], mvd);
    }
}

#[test]
fn a_motion_vector_difference_above_the_range_is_refused() {
    let params = inter_params(SliceKind::P);
    let mut writer = SyntaxWriter::new(&params);
    start_inter_cu(&mut writer);
    writer.ctx(ctx::MERGE_FLAG, false);
    write_mvd(&mut writer, [32_768, 0]);
    let bytes = writer.into_bytes();
    let result = Harness::new(params).try_parse(&bytes, &[(0, 0)]);
    assert!(matches!(
        result,
        Err(HevcError::InvalidSyntax {
            element: "lMvd",
            value: 32_768,
            ..
        })
    ));
}

/// `ref_idx`: two context bins, then bypass (Table 9-48).
#[test]
fn ref_idx_follows_its_truncated_rice_code() {
    #[derive(Clone, Copy)]
    enum Bin {
        First(bool),
        Second(bool),
        Bypass(bool),
    }
    use Bin::{Bypass, First, Second};
    let cases: [(u32, &[Bin]); 5] = [
        (0, &[First(false)]),
        (1, &[First(true), Second(false)]),
        (2, &[First(true), Second(true), Bypass(false)]),
        (3, &[First(true), Second(true), Bypass(true), Bypass(false)]),
        (4, &[First(true), Second(true), Bypass(true), Bypass(true)]),
    ];
    for (value, bins) in cases {
        let params = CodingParams {
            num_ref_idx_minus1: [4, 0],
            ..inter_params(SliceKind::P)
        };
        let mut writer = SyntaxWriter::new(&params);
        start_inter_cu(&mut writer);
        writer.ctx(ctx::MERGE_FLAG, false);
        for bin in bins {
            match *bin {
                First(bin) => writer.ctx(ctx::REF_IDX, bin),
                Second(bin) => writer.ctx(ctx::REF_IDX + 1, bin),
                Bypass(bin) => writer.bypass(bin),
            }
        }
        write_mvd(&mut writer, [0, 0]);
        writer.ctx(ctx::MVP_FLAG, false);
        writer.ctx(ctx::RQT_ROOT_CBF, false);
        let pu = parse_one(params, writer);
        assert_eq!(pu.ref_idx, [value, 0]);
    }
}

/// `inter_pred_idc`: the first bin's context is `CtDepth` (Table 9-48).
#[test]
fn inter_pred_idc_reads_the_depth_context() {
    let cases: [(&[(usize, bool)], InterPredIdc); 3] = [
        (&[(1, true)], InterPredIdc::Bi),
        (&[(1, false), (4, false)], InterPredIdc::L0),
        (&[(1, false), (4, true)], InterPredIdc::L1),
    ];
    for (bins, expected) in cases {
        let params = CodingParams {
            min_cb_log2: 3,
            mvd_l1_zero: true,
            ..inter_params(SliceKind::B)
        };
        let mut writer = SyntaxWriter::new(&params);
        // The root splits, so the first unit has CtDepth 1.
        writer.ctx(ctx::SPLIT_CU, true);
        writer.ctx(ctx::CU_SKIP, false);
        writer.ctx(ctx::PRED_MODE, false);
        writer.ctx(ctx::PART_MODE, true);
        writer.ctx(ctx::MERGE_FLAG, false);
        for &(inc, bin) in bins {
            writer.ctx(ctx::INTER_PRED_IDC + inc, bin);
        }
        if expected != InterPredIdc::L1 {
            write_mvd(&mut writer, [4, 0]);
            writer.ctx(ctx::MVP_FLAG, true);
        }
        if expected != InterPredIdc::L0 {
            // mvd_l1_zero_flag leaves a Bi unit with no list 1 difference.
            if expected == InterPredIdc::L1 {
                write_mvd(&mut writer, [0, 6]);
            }
            writer.ctx(ctx::MVP_FLAG, false);
        }
        writer.ctx(ctx::RQT_ROOT_CBF, false);
        // Units 1 to 3 skip, with merge_idx 0.
        writer.ctx(ctx::CU_SKIP, true);
        writer.ctx(ctx::MERGE_IDX, false);
        writer.ctx(ctx::CU_SKIP, true);
        writer.ctx(ctx::MERGE_IDX, false);
        writer.ctx(ctx::CU_SKIP + 2, true);
        writer.ctx(ctx::MERGE_IDX, false);
        let records = Harness::new(params).run(writer, &[(0, 0)]);
        let pu = records[0].pus[0];
        assert_eq!(pu.inter_pred_idc, expected);
        match expected {
            InterPredIdc::L1 => assert_eq!(pu.mvd, [[0, 0], [0, 6]]),
            _ => assert_eq!(pu.mvd, [[4, 0], [0, 0]]),
        }
    }
}

/// `nPbW + nPbH` of 12 leaves one bin, on context 4 (Table 9-47).
#[test]
fn an_8x4_unit_reads_one_inter_pred_idc_bin() {
    let params = CodingParams {
        width: 8,
        height: 8,
        ctb_log2: 3,
        min_cb_log2: 3,
        max_tb_log2: 3,
        log2_min_qg_size: 3,
        ..inter_params(SliceKind::B)
    };
    let mut writer = SyntaxWriter::new(&params);
    writer.ctx(ctx::CU_SKIP, false);
    writer.ctx(ctx::PRED_MODE, false);
    // part_mode 2NxN at 8x8: 01.
    writer.ctx(ctx::PART_MODE, false);
    writer.ctx(ctx::PART_MODE + 1, true);
    writer.ctx(ctx::MERGE_FLAG, false);
    writer.ctx(ctx::INTER_PRED_IDC + 4, true);
    write_mvd(&mut writer, [0, 0]);
    writer.ctx(ctx::MVP_FLAG, false);
    writer.ctx(ctx::MERGE_FLAG, false);
    writer.ctx(ctx::INTER_PRED_IDC + 4, false);
    write_mvd(&mut writer, [0, 0]);
    writer.ctx(ctx::MVP_FLAG, false);
    writer.ctx(ctx::RQT_ROOT_CBF, false);
    let records = Harness::new(params).run(writer, &[(0, 0)]);
    assert_eq!(records[0].cus[0].part_mode, PartMode::TwoNxN);
    assert_eq!(records[0].pus[0].inter_pred_idc, InterPredIdc::L1);
    assert_eq!(records[0].pus[1].inter_pred_idc, InterPredIdc::L0);
    assert_eq!(records[0].pus[1].y, 4);
}

#[test]
fn a_bi_unit_reads_both_lists() {
    let params = CodingParams {
        num_ref_idx_minus1: [1, 1],
        ..inter_params(SliceKind::B)
    };
    let mut writer = SyntaxWriter::new(&params);
    start_inter_cu(&mut writer);
    writer.ctx(ctx::MERGE_FLAG, false);
    writer.ctx(ctx::INTER_PRED_IDC, true);
    writer.ctx(ctx::REF_IDX, true);
    write_mvd(&mut writer, [-9, 12]);
    writer.ctx(ctx::MVP_FLAG, false);
    writer.ctx(ctx::REF_IDX, false);
    write_mvd(&mut writer, [1, -1]);
    writer.ctx(ctx::MVP_FLAG, true);
    writer.ctx(ctx::RQT_ROOT_CBF, false);
    let pu = parse_one(params, writer);
    assert_eq!(pu.inter_pred_idc, InterPredIdc::Bi);
    assert_eq!(pu.ref_idx, [1, 0]);
    assert_eq!(pu.mvd, [[-9, 12], [1, -1]]);
    assert_eq!(pu.mvp_flag, [false, true]);
}
