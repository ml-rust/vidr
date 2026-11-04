//! `coding_unit()` (7.3.8.5).

use crate::decode::cpu::hevc::cabac::context as ctx;
use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::params::SliceKind;
use crate::decode::cpu::hevc::qp::qp_y;

use super::inter_pu::PuArgs;
use super::intra_mode::{CHROMA_FROM_LUMA, cand_mode_list, chroma_mode, mode_from_rem};
use super::maps::INTRA_DC;
use super::parser::CtuParser;
use super::records::{CuRecord, PartMode, PredMode, push_record};
use super::transform_tree::{CuContext, TreeNode};

impl CtuParser<'_, '_> {
    pub(super) fn coding_unit(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u32,
        depth: u32,
    ) -> Result<(), HevcError> {
        self.begin_quant_group(x0, y0);
        let size = 1u32 << log2_size;
        let transquant_bypass = self.cu_transquant_bypass_flag();
        let inter_slice = self.params.kind != SliceKind::I;
        let skip = inter_slice && {
            let inc = self.skip_ctx_inc(x0, y0);
            self.cabac.decision(ctx::CU_SKIP + inc)
        };
        self.maps.set_cu(x0, y0, size, depth as u8, skip);
        let mut record = CuRecord {
            x: x0,
            y: y0,
            log2_size,
            pred_mode: PredMode::Skip,
            part_mode: PartMode::TwoNxTwoN,
            transquant_bypass,
            pcm: false,
            qp_y: 0,
            intra_luma: [INTRA_DC; 4],
            intra_chroma: INTRA_DC,
            first_pu: self.records.pus.len(),
            first_tu: self.records.tus.len(),
            first_pcm_sample: self.records.pcm_samples.len(),
        };
        if skip {
            self.maps.set_intra_mode(x0, y0, size, INTRA_DC);
            self.prediction_unit(&PuArgs {
                x: x0,
                y: y0,
                width: size,
                height: size,
                skip: true,
                depth,
            })?;
        } else if !inter_slice || self.cabac.decision(ctx::PRED_MODE) {
            record.pred_mode = PredMode::Intra;
            self.intra_coding_unit(&mut record, depth)?;
        } else {
            record.pred_mode = PredMode::Inter;
            self.inter_coding_unit(&mut record, depth)?;
        }
        // (8-283). CuQpDeltaVal is final here, because it parses in the first
        // transform unit with a coded block flag.
        let qp = qp_y(
            self.qp.predicted,
            self.qp.delta_val,
            self.params.qp_bd_offset(),
        );
        record.qp_y = qp;
        self.qp.last_cu_qp_y = qp;
        self.maps.set_qp_y(x0, y0, size, qp);
        // 8.7.2.5.7, 8.7.2.5.8, and 8.7.3.2 share this condition in Main and Main 10.
        let no_filter =
            record.transquant_bypass || (record.pcm && self.params.pcm_loop_filter_disabled);
        let intra = record.pred_mode == PredMode::Intra;
        self.maps.set_cu_kind(x0, y0, size, intra, no_filter);
        push_record(&mut self.records.cus, record, "coding unit records")?;
        self.check_not_exhausted()
    }

    /// The intra branch of 7.3.8.5.
    fn intra_coding_unit(&mut self, record: &mut CuRecord, depth: u32) -> Result<(), HevcError> {
        let (x0, y0, log2_size) = (record.x, record.y, record.log2_size);
        let is_min_size = log2_size == self.params.min_cb_log2;
        // Table 9-45: at the minimum size one bin selects 2Nx2N or NxN.
        if is_min_size && !self.cabac.decision(ctx::PART_MODE) {
            record.part_mode = PartMode::NxN;
        }
        if self.pcm_flag(record.part_mode, log2_size) {
            record.pcm = true;
            self.maps.set_intra_mode(x0, y0, 1 << log2_size, INTRA_DC);
            return self.pcm_sample(log2_size);
        }
        self.intra_modes(record);
        let intra_split = record.part_mode == PartMode::NxN;
        let cu = CuContext {
            index: self.records.cus.len(),
            log2_size,
            depth,
            intra: true,
            part_mode: record.part_mode,
            transquant_bypass: record.transquant_bypass,
            intra_split,
            max_trafo_depth: self.params.max_depth_intra + u32::from(intra_split),
            intra_chroma: record.intra_chroma,
        };
        self.transform_tree(&cu, TreeNode::root(x0, y0, log2_size))
    }

    /// `pcm_flag`, when the syntax holds it.
    fn pcm_flag(&mut self, part_mode: PartMode, log2_size: u32) -> bool {
        let Some(pcm) = self.params.pcm else {
            return false;
        };
        part_mode == PartMode::TwoNxTwoN
            && (pcm.log2_min_size..=pcm.log2_max_size).contains(&log2_size)
            && self.cabac.terminate()
    }

    /// The luma mode syntax and `intra_chroma_pred_mode`, with 8.4.2 and 8.4.3.
    fn intra_modes(&mut self, record: &mut CuRecord) {
        let parts: usize = if record.part_mode == PartMode::NxN {
            4
        } else {
            1
        };
        let block_size = (1u32 << record.log2_size) >> (parts / 4);
        let mut prev_flags = [false; 4];
        for flag in prev_flags.iter_mut().take(parts) {
            *flag = self.cabac.decision(ctx::PREV_INTRA_LUMA_PRED);
        }
        for (part, &prev_flag) in prev_flags.iter().enumerate().take(parts) {
            let x = record.x + ((part as u32) & 1) * block_size;
            let y = record.y + ((part as u32) >> 1) * block_size;
            let candidates = self.intra_candidates(x, y);
            let mode = if prev_flag {
                // mpm_idx: TR with cMax 2, in bypass bins.
                candidates[self.bypass_unary(2) as usize]
            } else {
                mode_from_rem(candidates, self.cabac.bypass_bits(5) as u8)
            };
            self.maps.set_intra_mode(x, y, block_size, mode);
            if parts == 1 {
                record.intra_luma = [mode; 4];
            } else {
                record.intra_luma[part] = mode;
            }
        }
        // Table 9-46: one context bin, then two bypass bins.
        let intra_chroma_pred_mode = if self.cabac.decision(ctx::INTRA_CHROMA_PRED_MODE) {
            self.cabac.bypass_bits(2)
        } else {
            CHROMA_FROM_LUMA
        };
        record.intra_chroma = chroma_mode(intra_chroma_pred_mode, record.intra_luma[0]);
    }

    /// `candModeList` for the prediction block at `( x, y )` (8.4.2, steps 1 to 3).
    fn intra_candidates(&self, x: u32, y: u32) -> [u8; 3] {
        let mode_at = |x_nb: i32, y_nb: i32| {
            if self.available((x, y), x_nb, y_nb) {
                self.maps.intra_mode(x_nb as u32, y_nb as u32)
            } else {
                INTRA_DC
            }
        };
        let a = mode_at(x as i32 - 1, y as i32);
        let ctb_top = (y >> self.params.ctb_log2) << self.params.ctb_log2;
        // A block above the current CTB gives INTRA_DC.
        let b = if y == ctb_top {
            INTRA_DC
        } else {
            mode_at(x as i32, y as i32 - 1)
        };
        cand_mode_list(a, b)
    }

    /// The inter branch of 7.3.8.5.
    fn inter_coding_unit(&mut self, record: &mut CuRecord, depth: u32) -> Result<(), HevcError> {
        let (x0, y0, log2_size) = (record.x, record.y, record.log2_size);
        let size = 1u32 << log2_size;
        self.maps.set_intra_mode(x0, y0, size, INTRA_DC);
        record.part_mode = self.inter_part_mode(log2_size);
        let (blocks, count) = record.part_mode.prediction_blocks(size);
        let mut merge_flag = false;
        for block in blocks.iter().take(count) {
            merge_flag = self.prediction_unit(&PuArgs {
                x: x0 + block.x,
                y: y0 + block.y,
                width: block.width,
                height: block.height,
                skip: false,
                depth,
            })?;
        }
        let root_cbf = (record.part_mode == PartMode::TwoNxTwoN && merge_flag)
            || self.cabac.decision(ctx::RQT_ROOT_CBF);
        if !root_cbf {
            return Ok(());
        }
        let cu = CuContext {
            index: self.records.cus.len(),
            log2_size,
            depth,
            intra: false,
            part_mode: record.part_mode,
            transquant_bypass: record.transquant_bypass,
            intra_split: false,
            max_trafo_depth: self.params.max_depth_inter,
            intra_chroma: INTRA_DC,
        };
        self.transform_tree(&cu, TreeNode::root(x0, y0, log2_size))
    }

    /// `part_mode` of an inter coding unit (Tables 9-45 and 9-48).
    fn inter_part_mode(&mut self, log2_size: u32) -> PartMode {
        if self.cabac.decision(ctx::PART_MODE) {
            return PartMode::TwoNxTwoN;
        }
        let horizontal = self.cabac.decision(ctx::PART_MODE + 1);
        if log2_size == self.params.min_cb_log2 {
            if horizontal {
                return PartMode::TwoNxN;
            }
            // log2CbSize 3 has no NxN, and its bin string ends here.
            if log2_size == 3 || self.cabac.decision(ctx::PART_MODE + 2) {
                return PartMode::NxTwoN;
            }
            return PartMode::NxN;
        }
        if !self.params.amp_enabled || self.cabac.decision(ctx::PART_MODE + 3) {
            return if horizontal {
                PartMode::TwoNxN
            } else {
                PartMode::NxTwoN
            };
        }
        match (horizontal, self.cabac.bypass()) {
            (true, false) => PartMode::TwoNxnU,
            (true, true) => PartMode::TwoNxnD,
            (false, false) => PartMode::NLxTwoN,
            (false, true) => PartMode::NRxTwoN,
        }
    }
}
