//! A small picture state for the motion derivation tests.

use crate::decode::cpu::hevc::layout::{LayoutSizes, PicLayout};
use crate::decode::cpu::hevc::params::{CodingParams, SliceKind};
use crate::decode::cpu::hevc::syntax::maps::ParseMaps;
use crate::decode::cpu::hevc::syntax::records::PartMode;

use super::col::ColMotion;
use super::context::{MotionContext, PbGeometry};
use super::field::MotionField;
use super::temporal::TemporalContext;
use super::types::{Mv, NO_REF, PbMotion, RefLists};

/// A 64x64 picture with 16x16 CTBs, POC 8, in one B slice. L0 holds POC 4 and 2. L1
/// holds POC 12 and 16.
pub(super) struct Scene {
    pub(super) params: CodingParams,
    pub(super) layout: PicLayout,
    pub(super) maps: ParseMaps,
    pub(super) field: MotionField,
    pub(super) refs: RefLists,
    pub(super) col: Option<ColMotion>,
    pub(super) col_poc: i32,
    pub(super) no_backward_pred: bool,
}

impl Scene {
    pub(super) fn new() -> Self {
        let params = CodingParams {
            width: 64,
            height: 64,
            ctb_log2: 4,
            kind: SliceKind::B,
            max_merge_cand: 5,
            num_ref_idx_minus1: [1, 1],
            ..CodingParams::for_tests()
        };
        let layout = PicLayout::new(&LayoutSizes {
            width: 64,
            height: 64,
            ctb_log2: 4,
            min_tb_log2: 2,
            column_widths: &[4],
            row_heights: &[4],
        })
        .unwrap();
        let mut maps = ParseMaps::new(&params, &layout).unwrap();
        // Every block starts intra, so a test names the neighbours it wants.
        maps.set_cu_kind(0, 0, 64, true, false);
        let mut field = MotionField::new(64, 64, &layout).unwrap();
        let mut refs = RefLists::default();
        refs.poc[0][0] = 4;
        refs.poc[0][1] = 2;
        refs.poc[1][0] = 12;
        refs.poc[1][1] = 16;
        refs.count = [2, 2];
        field.begin_slice(refs).unwrap();
        for rs in 0..layout.ctb_count() {
            maps.set_ctb_slice(rs, 0);
            field.set_ctb(rs);
        }
        Self {
            params,
            layout,
            maps,
            field,
            refs,
            col: None,
            col_poc: 0,
            no_backward_pred: false,
        }
    }

    pub(super) fn context(&self) -> MotionContext<'_> {
        MotionContext {
            params: &self.params,
            layout: &self.layout,
            maps: &self.maps,
            field: &self.field,
            refs: &self.refs,
            temporal: TemporalContext {
                col: self.col.as_ref(),
                col_poc: self.col_poc,
                no_backward_pred: self.no_backward_pred,
                collocated_from_l0: true,
                ctb_log2: 4,
                width: 64,
                height: 64,
                poc: 8,
            },
            poc: 8,
        }
    }

    /// Marks an 8-aligned block inter, with this motion.
    pub(super) fn set_inter(&mut self, x: u32, y: u32, width: u32, height: u32, motion: PbMotion) {
        for y_unit in (y..y + height).step_by(8) {
            for x_unit in (x..x + width).step_by(8) {
                self.maps.set_cu_kind(x_unit, y_unit, 8, false, false);
            }
        }
        self.field.fill(x, y, width, height, motion);
    }

    /// Marks an 8-aligned square block intra.
    pub(super) fn set_intra(&mut self, x: u32, y: u32, size: u32) {
        self.maps.set_cu_kind(x, y, size, true, false);
        self.field.fill(x, y, size, size, PbMotion::NONE);
    }
}

/// Motion that uses L0 alone.
pub(super) const fn inter(mv: Mv, ref_idx: i8) -> PbMotion {
    PbMotion {
        mv: [mv, Mv::new(0, 0)],
        ref_idx: [ref_idx, NO_REF],
    }
}

/// Motion that uses L1 alone.
pub(super) const fn inter_l1(mv: Mv, ref_idx: i8) -> PbMotion {
    PbMotion {
        mv: [Mv::new(0, 0), mv],
        ref_idx: [NO_REF, ref_idx],
    }
}

/// A PART_2Nx2N block that fills its coding block.
pub(super) const fn pb(x: u32, y: u32, width: u32, height: u32) -> PbGeometry {
    PbGeometry {
        x_cb: x,
        y_cb: y,
        n_cb: width,
        x_pb: x,
        y_pb: y,
        width,
        height,
        part_idx: 0,
        part_mode: PartMode::TwoNxTwoN,
    }
}
