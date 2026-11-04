//! `coding_quadtree()` (7.3.8.4).

use crate::decode::cpu::hevc::cabac::context as ctx;
use crate::decode::cpu::hevc::error::HevcError;

use super::parser::CtuParser;

impl CtuParser<'_, '_> {
    /// Parses the tree of the CTU at `( x_ctb, y_ctb )`.
    ///
    /// # Errors
    ///
    /// Any error of the syntax inside the CTU.
    pub(crate) fn parse_coding_tree(&mut self, x_ctb: u32, y_ctb: u32) -> Result<(), HevcError> {
        self.coding_quadtree(x_ctb, y_ctb, self.params.ctb_log2, 0)
    }

    fn coding_quadtree(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u32,
        depth: u32,
    ) -> Result<(), HevcError> {
        let size = 1u32 << log2_size;
        let (width, height) = (self.params.width, self.params.height);
        let can_split = log2_size > self.params.min_cb_log2;
        let split = if x0 + size <= width && y0 + size <= height && can_split {
            let inc = self.split_ctx_inc(x0, y0, depth);
            self.cabac.decision(ctx::SPLIT_CU + inc)
        } else {
            can_split
        };
        if self.params.cu_qp_delta && log2_size >= self.params.log2_min_qg_size {
            self.qp.is_delta_coded = false;
            self.qp.delta_val = 0;
        }
        if !split {
            return self.coding_unit(x0, y0, log2_size, depth);
        }
        let half = size >> 1;
        let (x1, y1) = (x0 + half, y0 + half);
        self.coding_quadtree(x0, y0, log2_size - 1, depth + 1)?;
        if x1 < width {
            self.coding_quadtree(x1, y0, log2_size - 1, depth + 1)?;
        }
        if y1 < height {
            self.coding_quadtree(x0, y1, log2_size - 1, depth + 1)?;
        }
        if x1 < width && y1 < height {
            self.coding_quadtree(x1, y1, log2_size - 1, depth + 1)?;
        }
        Ok(())
    }
}
