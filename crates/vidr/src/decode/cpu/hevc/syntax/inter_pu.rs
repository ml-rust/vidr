//! `prediction_unit()` and `mvd_coding()` (7.3.8.6, 7.3.8.9).

use crate::decode::cpu::hevc::cabac::context as ctx;
use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::params::SliceKind;

use super::parser::CtuParser;
use super::records::{InterPredIdc, PuRecord, push_record};

/// The range of `MvdLX` (7.4.9.9).
const MVD_LIMIT: u32 = 1 << 15;

/// The arguments of `prediction_unit()`.
#[derive(Clone, Copy, Debug)]
pub(super) struct PuArgs {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    /// `cu_skip_flag` of the coding unit.
    pub(super) skip: bool,
    /// `CtDepth` of the coding unit.
    pub(super) depth: u32,
}

impl CtuParser<'_, '_> {
    /// Parses one prediction unit, and returns its `merge_flag`.
    pub(super) fn prediction_unit(&mut self, args: &PuArgs) -> Result<bool, HevcError> {
        let mut record = PuRecord {
            x: args.x,
            y: args.y,
            width: args.width,
            height: args.height,
            merge_flag: args.skip,
            merge_idx: 0,
            inter_pred_idc: InterPredIdc::L0,
            ref_idx: [0; 2],
            mvd: [[0; 2]; 2],
            mvp_flag: [false; 2],
        };
        if args.skip {
            record.merge_idx = self.merge_idx();
        } else {
            record.merge_flag = self.cabac.decision(ctx::MERGE_FLAG);
            if record.merge_flag {
                record.merge_idx = self.merge_idx();
            } else {
                self.explicit_motion(args, &mut record)?;
            }
        }
        let merge_flag = record.merge_flag;
        push_record(&mut self.records.pus, record, "prediction unit records")?;
        Ok(merge_flag)
    }

    /// The branch of 7.3.8.6 with `merge_flag` 0.
    fn explicit_motion(&mut self, args: &PuArgs, record: &mut PuRecord) -> Result<(), HevcError> {
        if self.params.kind == SliceKind::B {
            record.inter_pred_idc = self.inter_pred_idc(args);
        }
        if record.inter_pred_idc != InterPredIdc::L1 {
            record.ref_idx[0] = self.ref_idx(self.params.num_ref_idx_minus1[0]);
            record.mvd[0] = self.mvd_coding()?;
            record.mvp_flag[0] = self.cabac.decision(ctx::MVP_FLAG);
        }
        if record.inter_pred_idc != InterPredIdc::L0 {
            record.ref_idx[1] = self.ref_idx(self.params.num_ref_idx_minus1[1]);
            let mvd_is_zero = self.params.mvd_l1_zero && record.inter_pred_idc == InterPredIdc::Bi;
            if !mvd_is_zero {
                record.mvd[1] = self.mvd_coding()?;
            }
            record.mvp_flag[1] = self.cabac.decision(ctx::MVP_FLAG);
        }
        Ok(())
    }

    /// `merge_idx`: TR with `cMax` `MaxNumMergeCand - 1`, the first bin with a
    /// context and the others in bypass (Table 9-48).
    fn merge_idx(&mut self) -> u32 {
        let c_max = self.params.max_merge_cand.saturating_sub(1);
        if c_max == 0 || !self.cabac.decision(ctx::MERGE_IDX) {
            return 0;
        }
        1 + self.bypass_unary(c_max - 1)
    }

    /// `inter_pred_idc` (Table 9-47, Table 9-48).
    fn inter_pred_idc(&mut self, args: &PuArgs) -> InterPredIdc {
        const LAST_CONTEXT: usize = 4;
        if args.width + args.height != 12 {
            let inc = (args.depth as usize).min(LAST_CONTEXT - 1);
            if self.cabac.decision(ctx::INTER_PRED_IDC + inc) {
                return InterPredIdc::Bi;
            }
        }
        if self.cabac.decision(ctx::INTER_PRED_IDC + LAST_CONTEXT) {
            InterPredIdc::L1
        } else {
            InterPredIdc::L0
        }
    }

    /// `ref_idx_l0` or `ref_idx_l1`: TR with the first two bins on contexts and
    /// the others in bypass (Table 9-48).
    fn ref_idx(&mut self, c_max: u32) -> u32 {
        let mut value = 0;
        while value < c_max {
            let bin = if value < 2 {
                self.cabac.decision(ctx::REF_IDX + value as usize)
            } else {
                self.cabac.bypass()
            };
            if !bin {
                break;
            }
            value += 1;
        }
        value
    }

    /// `mvd_coding()`, returning `lMvd` (7-73).
    fn mvd_coding(&mut self) -> Result<[i32; 2], HevcError> {
        let greater0 = [
            self.cabac.decision(ctx::ABS_MVD_GREATER0),
            self.cabac.decision(ctx::ABS_MVD_GREATER0),
        ];
        let greater1 = [
            greater0[0] && self.cabac.decision(ctx::ABS_MVD_GREATER1),
            greater0[1] && self.cabac.decision(ctx::ABS_MVD_GREATER1),
        ];
        Ok([
            self.mvd_component(greater0[0], greater1[0])?,
            self.mvd_component(greater0[1], greater1[1])?,
        ])
    }

    /// `abs_mvd_minus2` and `mvd_sign_flag` of one component.
    fn mvd_component(&mut self, greater0: bool, greater1: bool) -> Result<i32, HevcError> {
        if !greater0 {
            return Ok(0);
        }
        let magnitude = if greater1 {
            self.exp_golomb(1, "abs_mvd_minus2")? + 2
        } else {
            1
        };
        let negative = self.cabac.bypass();
        if magnitude > MVD_LIMIT || (magnitude == MVD_LIMIT && !negative) {
            return Err(HevcError::InvalidSyntax {
                element: "lMvd",
                value: i64::from(magnitude),
                allowed: "-32768 to 32767",
            });
        }
        let magnitude = magnitude as i32;
        Ok(if negative { -magnitude } else { magnitude })
    }
}
