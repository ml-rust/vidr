//! `transform_tree()`, `transform_unit()`, and `delta_qp()` (7.3.8.8, 7.3.8.10,
//! 7.3.8.14) with `ChromaArrayType` 1.

use crate::decode::cpu::hevc::cabac::context as ctx;
use crate::decode::cpu::hevc::error::HevcError;

use super::intra_mode::scan_idx;
use super::parser::CtuParser;
use super::records::{PartMode, TuRecord, push_record};
use super::residual::TbArgs;

/// What a transform tree reads from its coding unit.
#[derive(Clone, Copy, Debug)]
pub(super) struct CuContext {
    /// Index the coding unit takes in `CtuRecords::cus`.
    pub(super) index: usize,
    /// `log2CbSize`.
    pub(super) log2_size: u32,
    /// `CtDepth`.
    pub(super) depth: u32,
    /// True when `CuPredMode` is `MODE_INTRA`.
    pub(super) intra: bool,
    /// `PartMode`.
    pub(super) part_mode: PartMode,
    /// `cu_transquant_bypass_flag`.
    pub(super) transquant_bypass: bool,
    /// `IntraSplitFlag`.
    pub(super) intra_split: bool,
    /// `MaxTrafoDepth`.
    pub(super) max_trafo_depth: u32,
    /// `IntraPredModeC`.
    pub(super) intra_chroma: u8,
}

/// The arguments of `transform_tree()`, and the parent's chroma flags.
#[derive(Clone, Copy, Debug)]
pub(super) struct TreeNode {
    x0: u32,
    y0: u32,
    x_base: u32,
    y_base: u32,
    log2_size: u32,
    depth: u32,
    blk_idx: u32,
    /// `cbf_cb[ xBase ][ yBase ][ trafoDepth - 1 ]`.
    parent_cbf_cb: bool,
    /// `cbf_cr[ xBase ][ yBase ][ trafoDepth - 1 ]`.
    parent_cbf_cr: bool,
}

impl TreeNode {
    /// The root of a coding unit's tree.
    pub(super) const fn root(x0: u32, y0: u32, log2_size: u32) -> Self {
        Self {
            x0,
            y0,
            x_base: x0,
            y_base: y0,
            log2_size,
            depth: 0,
            blk_idx: 0,
            parent_cbf_cb: true,
            parent_cbf_cr: true,
        }
    }
}

impl CtuParser<'_, '_> {
    pub(super) fn transform_tree(
        &mut self,
        cu: &CuContext,
        node: TreeNode,
    ) -> Result<(), HevcError> {
        let log2_size = node.log2_size;
        let split = self.split_transform_flag(cu, &node);
        // At a log2TrafoSize of 2, chroma belongs to the parent (7.3.8.10).
        let (cbf_cb, cbf_cr) = if log2_size > 2 {
            (
                node.parent_cbf_cb && self.cabac.decision(ctx::CBF_CHROMA + node.depth as usize),
                node.parent_cbf_cr && self.cabac.decision(ctx::CBF_CHROMA + node.depth as usize),
            )
        } else {
            (node.parent_cbf_cb, node.parent_cbf_cr)
        };
        if split {
            let half = 1u32 << (log2_size - 1);
            let (x1, y1) = (node.x0 + half, node.y0 + half);
            let child = |x0: u32, y0: u32, blk_idx: u32| TreeNode {
                x0,
                y0,
                x_base: node.x0,
                y_base: node.y0,
                log2_size: log2_size - 1,
                depth: node.depth + 1,
                blk_idx,
                parent_cbf_cb: cbf_cb,
                parent_cbf_cr: cbf_cr,
            };
            self.transform_tree(cu, child(node.x0, node.y0, 0))?;
            self.transform_tree(cu, child(x1, node.y0, 1))?;
            self.transform_tree(cu, child(node.x0, y1, 2))?;
            return self.transform_tree(cu, child(x1, y1, 3));
        }
        let cbf_luma = if cu.intra || node.depth != 0 || cbf_cb || cbf_cr {
            let inc = usize::from(node.depth == 0);
            self.cabac.decision(ctx::CBF_LUMA + inc)
        } else {
            true
        };
        self.transform_unit(cu, &node, [cbf_luma, cbf_cb, cbf_cr])
    }

    /// `split_transform_flag`, parsed or inferred (7.4.9.8).
    fn split_transform_flag(&mut self, cu: &CuContext, node: &TreeNode) -> bool {
        let log2_size = node.log2_size;
        let split_is_forced = cu.intra_split && node.depth == 0;
        if log2_size <= self.params.max_tb_log2
            && log2_size > self.params.min_tb_log2
            && node.depth < cu.max_trafo_depth
            && !split_is_forced
        {
            return self
                .cabac
                .decision(ctx::SPLIT_TRANSFORM + (5 - log2_size) as usize);
        }
        let inter_split = self.params.max_depth_inter == 0
            && !cu.intra
            && cu.part_mode != PartMode::TwoNxTwoN
            && node.depth == 0;
        log2_size > self.params.max_tb_log2 || split_is_forced || inter_split
    }

    /// `transform_unit()`. `cbf` holds `cbf_luma`, then `cbf_cb` and `cbf_cr` of
    /// the unit, or of its parent at a `log2TrafoSize` of 2.
    fn transform_unit(
        &mut self,
        cu: &CuContext,
        node: &TreeNode,
        cbf: [bool; 3],
    ) -> Result<(), HevcError> {
        let [cbf_luma, cbf_cb, cbf_cr] = cbf;
        if cbf_luma || cbf_cb || cbf_cr {
            self.delta_qp()?;
        }
        let first_tb = self.records.tbs.len();
        if cbf_luma {
            let mode = self.maps.intra_mode(node.x0, node.y0);
            self.residual_coding(&TbArgs {
                x0: node.x0,
                y0: node.y0,
                log2_size: node.log2_size,
                c_idx: 0,
                scan_idx: scan_idx(cu.intra, node.log2_size, 0, mode),
                transquant_bypass: cu.transquant_bypass,
            })?;
        }
        if node.log2_size > 2 {
            self.chroma_residuals(cu, (node.x0, node.y0), node.log2_size - 1, [cbf_cb, cbf_cr])?;
        } else if node.blk_idx == 3 {
            self.chroma_residuals(cu, (node.x_base, node.y_base), 2, [cbf_cb, cbf_cr])?;
        }
        let tu = TuRecord {
            x: node.x0,
            y: node.y0,
            x_base: node.x_base,
            y_base: node.y_base,
            log2_size: node.log2_size,
            depth: node.depth,
            blk_idx: node.blk_idx,
            cbf_luma,
            cbf_cb,
            cbf_cr,
            cu: cu.index,
            first_tb,
            tb_count: self.records.tbs.len() - first_tb,
        };
        push_record(&mut self.records.tus, tu, "transform unit records")
    }

    /// The Cb then Cr `residual_coding()` calls of one transform unit.
    fn chroma_residuals(
        &mut self,
        cu: &CuContext,
        location: (u32, u32),
        log2_size: u32,
        cbf: [bool; 2],
    ) -> Result<(), HevcError> {
        for (c_idx, coded) in [(1, cbf[0]), (2, cbf[1])] {
            if coded {
                self.residual_coding(&TbArgs {
                    x0: location.0,
                    y0: location.1,
                    log2_size,
                    c_idx,
                    scan_idx: scan_idx(cu.intra, log2_size, c_idx, cu.intra_chroma),
                    transquant_bypass: cu.transquant_bypass,
                })?;
            }
        }
        Ok(())
    }

    /// `delta_qp()` (7.3.8.14), with the binarization of 9.3.3.10.
    fn delta_qp(&mut self) -> Result<(), HevcError> {
        if !self.params.cu_qp_delta || self.qp.is_delta_coded {
            return Ok(());
        }
        self.qp.is_delta_coded = true;
        let mut abs = 0u32;
        if self.cabac.decision(ctx::CU_QP_DELTA_ABS) {
            abs = 1;
            while abs < 5 && self.cabac.decision(ctx::CU_QP_DELTA_ABS + 1) {
                abs += 1;
            }
            if abs == 5 {
                abs += self.exp_golomb(0, "cu_qp_delta_abs")?;
            }
        }
        if abs == 0 {
            return Ok(());
        }
        let negative = self.cabac.bypass();
        let value = if negative { -(abs as i32) } else { abs as i32 };
        let half_offset = self.params.qp_bd_offset() / 2;
        if !(-(26 + half_offset)..=25 + half_offset).contains(&value) {
            return Err(HevcError::InvalidSyntax {
                element: "CuQpDeltaVal",
                value: i64::from(value),
                allowed: "-( 26 + QpBdOffsetY / 2 ) to 25 + QpBdOffsetY / 2",
            });
        }
        self.qp.delta_val = value;
        Ok(())
    }
}
