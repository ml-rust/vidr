//! The parser state for one slice segment, and the neighbour helpers its syntax
//! functions share.

use crate::decode::cpu::hevc::cabac::context::{self as ctx, ContextTable};
use crate::decode::cpu::hevc::cabac::engine::Cabac;
use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::layout::PicLayout;
use crate::decode::cpu::hevc::params::CodingParams;
use crate::decode::cpu::hevc::qp::{QpState, predict_qp_y};

use super::maps::{CtbFilter, ParseMaps};
use super::records::CtuRecords;

/// Above this Exp-Golomb order, a value exceeds every element's limit: 2 to the
/// 15 for levels and motion vector differences.
const MAX_EXP_GOLOMB_ORDER: u32 = 20;

/// Parses CTUs of one slice segment into [`CtuRecords`].
pub(crate) struct CtuParser<'p, 'd> {
    pub(super) params: &'p CodingParams,
    pub(super) layout: &'p PicLayout,
    pub(super) maps: &'p mut ParseMaps,
    pub(super) cabac: Cabac<'d>,
    pub(super) qp: QpState,
    /// The records of the current CTU.
    pub(crate) records: CtuRecords,
}

impl<'p, 'd> CtuParser<'p, 'd> {
    /// Starts at byte `start` of `data`, with contexts initialized for this slice
    /// (9.3.2.2).
    ///
    /// # Errors
    ///
    /// [`HevcError::InvalidSyntax`] for a first `ivlOffset` of 510 or 511.
    pub(crate) fn new(
        params: &'p CodingParams,
        layout: &'p PicLayout,
        maps: &'p mut ParseMaps,
        data: &'d [u8],
        start: usize,
    ) -> Result<Self, HevcError> {
        let contexts = ContextTable::new(params.init_type, params.slice_qp);
        Ok(Self {
            params,
            layout,
            maps,
            cabac: Cabac::new(data, start, contexts)?,
            qp: QpState::new(params.slice_qp),
            records: CtuRecords::default(),
        })
    }

    /// Prepares the CTU at luma location `( x, y )`: empties the records and marks
    /// the CTB as part of this slice.
    pub(crate) fn begin_ctu(&mut self, x: u32, y: u32) {
        self.records.clear();
        let rs = self.layout.ctb_rs_at(x, y);
        self.maps.set_ctb_slice(rs, self.params.slice_addr_rs);
        self.maps.begin_ctb_filter(
            rs,
            CtbFilter {
                beta_offset_div2: self.params.beta_offset_div2,
                tc_offset_div2: self.params.tc_offset_div2,
                across_slices: self.params.loop_filter_across_slices,
            },
        );
    }

    /// `end_of_slice_segment_flag` (7.3.8.1).
    pub(crate) fn end_of_slice_segment_flag(&mut self) -> bool {
        self.cabac.terminate()
    }

    /// The availability derivation of 6.4.1 for this picture.
    pub(super) fn available(&self, current: (u32, u32), x_nb: i32, y_nb: i32) -> bool {
        self.maps.available(self.layout, current, x_nb, y_nb)
    }

    /// `ctxInc` of `split_cu_flag` (9.3.4.2.2).
    pub(super) fn split_ctx_inc(&self, x0: u32, y0: u32, depth: u32) -> usize {
        let deeper = |x: i32, y: i32| {
            self.available((x0, y0), x, y)
                && u32::from(self.maps.ct_depth(x as u32, y as u32)) > depth
        };
        usize::from(deeper(x0 as i32 - 1, y0 as i32))
            + usize::from(deeper(x0 as i32, y0 as i32 - 1))
    }

    /// `ctxInc` of `cu_skip_flag` (9.3.4.2.2).
    pub(super) fn skip_ctx_inc(&self, x0: u32, y0: u32) -> usize {
        let skipped = |x: i32, y: i32| {
            self.available((x0, y0), x, y) && self.maps.cu_skip(x as u32, y as u32)
        };
        usize::from(skipped(x0 as i32 - 1, y0 as i32))
            + usize::from(skipped(x0 as i32, y0 as i32 - 1))
    }

    /// Runs steps 1 to 4 of 8.6.1 once per quantization group, at its first coding
    /// unit. `qPY_PREV` is the last coding unit before this group.
    pub(super) fn begin_quant_group(&mut self, x_cb: u32, y_cb: u32) {
        let mask = (1u32 << self.params.log2_min_qg_size) - 1;
        let group = (x_cb & !mask, y_cb & !mask);
        if self.qp.group == Some(group) {
            return;
        }
        self.qp.group = Some(group);
        let left = self.group_neighbour_qp((x_cb, y_cb), group.0 as i32 - 1, group.1 as i32);
        let above = self.group_neighbour_qp((x_cb, y_cb), group.0 as i32, group.1 as i32 - 1);
        self.qp.predicted = predict_qp_y(self.qp.last_cu_qp_y, left, above);
    }

    /// `QpY` at a neighbour of the group, or `None` when it is unavailable or in
    /// another CTB (8-280, 8-281).
    fn group_neighbour_qp(&self, current: (u32, u32), x_nb: i32, y_nb: i32) -> Option<i32> {
        if !self.available(current, x_nb, y_nb) {
            return None;
        }
        let (x, y) = (x_nb as u32, y_nb as u32);
        let same_ctb = self.layout.ctb_rs_at(x, y) == self.layout.ctb_rs_at(current.0, current.1);
        same_ctb.then(|| self.maps.qp_y(x, y))
    }

    /// One EGk value in bypass bins (9.3.3.3).
    ///
    /// # Errors
    ///
    /// [`HevcError::InvalidSyntax`] names `element` for a value too large for it.
    pub(super) fn exp_golomb(
        &mut self,
        mut order: u32,
        element: &'static str,
    ) -> Result<u32, HevcError> {
        let mut value = 0u32;
        while self.cabac.bypass() {
            value += 1 << order;
            order += 1;
            if order > MAX_EXP_GOLOMB_ORDER {
                return Err(HevcError::InvalidSyntax {
                    element,
                    value: i64::from(value),
                    allowed: "a value that fits the range of the element",
                });
            }
        }
        Ok(value + self.cabac.bypass_bits(order))
    }

    /// A truncated unary value of up to `max` bypass bins.
    pub(super) fn bypass_unary(&mut self, max: u32) -> u32 {
        let mut value = 0;
        while value < max && self.cabac.bypass() {
            value += 1;
        }
        value
    }

    /// `cu_transquant_bypass_flag`.
    pub(super) fn cu_transquant_bypass_flag(&mut self) -> bool {
        self.params.transquant_bypass && self.cabac.decision(ctx::CU_TRANSQUANT_BYPASS)
    }

    /// Reports a read past the end of the slice data.
    ///
    /// # Errors
    ///
    /// [`HevcError::SliceDataTruncated`] past the end.
    pub(super) fn check_not_exhausted(&self) -> Result<(), HevcError> {
        if self.cabac.exhausted() {
            return Err(self.cabac.engine.truncated());
        }
        Ok(())
    }
}
