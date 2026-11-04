//! Block edges and `bS` on the 8x8 luma grid (8.7.2.2 to 8.7.2.4).

use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::layout::PicLayout;
use crate::decode::cpu::hevc::motion::field::MotionField;
use crate::decode::cpu::hevc::motion::types::PbMotion;
use crate::decode::cpu::hevc::params::CodingParams;
use crate::decode::cpu::hevc::syntax::maps::{ParseMaps, filled};
use crate::decode::cpu::hevc::syntax::records::{
    CtuRecords, CuRecord, PartMode, PredMode, TuRecord,
};

use super::strength::{EdgeSide, motion_strength};

/// `bS` when a side of the edge is intra (8.7.2.4).
const BS_INTRA: u8 = 2;
/// `bS` at a transform block edge with a coded luma block on a side (8.7.2.4).
const BS_CODED: u8 = 1;
/// The grid step in luma samples, as a shift.
const GRID_LOG2: u32 = 3;
/// The length of one edge section in luma samples, as a shift.
const SECTION_LOG2: u32 = 2;

/// `bS` of each edge section in a picture. A vertical section is at
/// `( column << 3, row << 2 )`, and a horizontal section at
/// `( column << 2, row << 3 )`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EdgeMap {
    vertical: Vec<u8>,
    horizontal: Vec<u8>,
    /// 1 where the luma transform block holds a level that is not zero, per 4x4 unit.
    /// A coding unit writes its area before any later block reads it.
    coded: Vec<u8>,
    width: u32,
    height: u32,
    any: bool,
}

/// One edge of a block in an inter coding unit.
#[derive(Clone, Copy, Debug)]
struct InterEdge {
    /// True for a vertical edge.
    vertical: bool,
    /// Left end of a horizontal edge, or the column of a vertical one.
    x: u32,
    /// Top end of a vertical edge, or the row of a horizontal one.
    y: u32,
    /// Length in luma samples.
    length: u32,
    /// True when the edge is a transform block edge.
    transform: bool,
}

/// The last motion comparison along one edge. Long edges repeat the same two blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MotionPair {
    p: PbMotion,
    q: PbMotion,
    /// The reference table of each side.
    tables: (Option<u32>, Option<u32>),
}

/// What 8.7.2.4 reads besides the coded map.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StrengthInput<'a> {
    pub(crate) params: &'a CodingParams,
    pub(crate) layout: &'a PicLayout,
    pub(crate) maps: &'a ParseMaps,
    pub(crate) field: &'a MotionField,
}

/// The side of a block to mark.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EdgeRun {
    /// Left or top end, in luma samples.
    pub(crate) x: u32,
    /// Top or left end, in luma samples.
    pub(crate) y: u32,
    /// Length in luma samples.
    pub(crate) length: u32,
    /// `bS`.
    pub(crate) strength: u8,
}

impl EdgeMap {
    /// Allocates a map of no edges for a picture size.
    ///
    /// # Errors
    ///
    /// [`HevcError::Allocation`] for a size that does not fit in memory.
    pub(crate) fn new(width: u32, height: u32) -> Result<Self, HevcError> {
        let vertical = zeroed(
            (width >> GRID_LOG2) as usize,
            (height >> SECTION_LOG2) as usize,
        )?;
        let horizontal = zeroed(
            (width >> SECTION_LOG2) as usize,
            (height >> GRID_LOG2) as usize,
        )?;
        let coded = zeroed(
            (width >> SECTION_LOG2) as usize,
            (height >> SECTION_LOG2) as usize,
        )?;
        Ok(Self {
            vertical,
            horizontal,
            coded,
            width,
            height,
            any: false,
        })
    }

    /// Removes every edge, for a new picture.
    pub(crate) fn reset(&mut self) {
        self.vertical.fill(0);
        self.horizontal.fill(0);
        self.any = false;
    }

    /// True when at least one edge holds a `bS` above 0.
    pub(crate) const fn has_edges(&self) -> bool {
        self.any
    }

    /// Sections per row of vertical edges.
    const fn vertical_stride(&self) -> usize {
        (self.width >> GRID_LOG2) as usize
    }

    /// Sections per row of horizontal edges.
    const fn horizontal_stride(&self) -> usize {
        (self.width >> SECTION_LOG2) as usize
    }

    /// Records a vertical edge. An edge off the grid is not filtered (8.7.2.1).
    pub(crate) fn mark_vertical(&mut self, run: EdgeRun) {
        if run.x & ((1 << GRID_LOG2) - 1) != 0 || run.x >= self.width {
            return;
        }
        let stride = self.vertical_stride();
        let column = (run.x >> GRID_LOG2) as usize;
        let top = (run.y >> SECTION_LOG2) as usize;
        let bottom = ((run.y + run.length) >> SECTION_LOG2) as usize;
        for row in top..bottom {
            if let Some(entry) = self.vertical.get_mut(row * stride + column) {
                *entry = run.strength;
                self.any = true;
            }
        }
    }

    /// Records a horizontal edge. An edge off the grid is not filtered (8.7.2.1).
    pub(crate) fn mark_horizontal(&mut self, run: EdgeRun) {
        if run.y & ((1 << GRID_LOG2) - 1) != 0 || run.y >= self.height {
            return;
        }
        let stride = self.horizontal_stride();
        let row = (run.y >> GRID_LOG2) as usize;
        let left = (run.x >> SECTION_LOG2) as usize;
        let right = (((run.x + run.length) >> SECTION_LOG2) as usize).min(stride);
        for column in left..right {
            if let Some(entry) = self.horizontal.get_mut(row * stride + column) {
                *entry = run.strength;
                self.any = true;
            }
        }
    }

    /// The rows of vertical sections, one entry per column of the 8x8 grid.
    pub(crate) fn vertical_rows(&self) -> impl Iterator<Item = &[u8]> {
        self.vertical.chunks_exact(self.vertical_stride().max(1))
    }

    /// The rows of horizontal sections, one entry per 4 luma samples.
    pub(crate) fn horizontal_rows(&self) -> impl Iterator<Item = &[u8]> {
        self.horizontal
            .chunks_exact(self.horizontal_stride().max(1))
    }

    /// `bS` of the vertical section at `( x, y )`, each in luma samples.
    pub(crate) fn vertical_at(&self, x: u32, y: u32) -> u8 {
        let index =
            (y >> SECTION_LOG2) as usize * self.vertical_stride() + (x >> GRID_LOG2) as usize;
        self.vertical.get(index).copied().unwrap_or(0)
    }

    /// `bS` of the horizontal section at `( x, y )`, each in luma samples.
    pub(crate) fn horizontal_at(&self, x: u32, y: u32) -> u8 {
        let index =
            (y >> GRID_LOG2) as usize * self.horizontal_stride() + (x >> SECTION_LOG2) as usize;
        self.horizontal.get(index).copied().unwrap_or(0)
    }

    /// Marks the edges of the coding units of one CTU, for a slice with the
    /// deblocking filter on. A slice with the filter off records only its coded
    /// blocks, which a later slice reads across its top and left boundaries.
    pub(crate) fn mark_ctu(&mut self, input: &StrengthInput<'_>, records: &CtuRecords) {
        self.record_coded_blocks(records);
        if input.params.deblocking_disabled {
            return;
        }
        for (index, cu) in records.cus.iter().enumerate() {
            let end = records
                .cus
                .get(index + 1)
                .map_or(records.tus.len(), |next| next.first_tu);
            let tus = records.tus.get(cu.first_tu..end).unwrap_or_default();
            if cu.pred_mode == PredMode::Intra {
                self.mark_intra_cu(input, cu, tus);
            } else {
                self.mark_inter_cu(input, cu, tus);
            }
        }
    }

    /// Writes the coded map for the coding units of one CTU.
    fn record_coded_blocks(&mut self, records: &CtuRecords) {
        for cu in &records.cus {
            self.set_coded(cu.x, cu.y, 1 << cu.log2_size, 0);
        }
        for tu in &records.tus {
            if tu.cbf_luma {
                self.set_coded(tu.x, tu.y, 1 << tu.log2_size, 1);
            }
        }
    }

    fn set_coded(&mut self, x: u32, y: u32, size: u32, value: u8) {
        let stride = self.horizontal_stride();
        let count = (size >> SECTION_LOG2) as usize;
        let (left, top) = ((x >> SECTION_LOG2) as usize, (y >> SECTION_LOG2) as usize);
        for row in top..top + count {
            let start = row * stride + left;
            let end = (start + count).min((row + 1) * stride);
            if let Some(cells) = self.coded.get_mut(start..end) {
                cells.fill(value);
            }
        }
    }

    fn is_coded(&self, x: u32, y: u32) -> bool {
        let index =
            (y >> SECTION_LOG2) as usize * self.horizontal_stride() + (x >> SECTION_LOG2) as usize;
        self.coded.get(index).is_some_and(|&value| value != 0)
    }

    /// Every edge of an intra coding unit has `bS` 2.
    fn mark_intra_cu(&mut self, input: &StrengthInput<'_>, cu: &CuRecord, tus: &[TuRecord]) {
        let size = 1u32 << cu.log2_size;
        let run = |x, y, length| EdgeRun {
            x,
            y,
            length,
            strength: BS_INTRA,
        };
        let (filter_left, filter_top) = filter_cb_edges(input, cu);
        if filter_left {
            self.mark_vertical(run(cu.x, cu.y, size));
        }
        if filter_top {
            self.mark_horizontal(run(cu.x, cu.y, size));
        }
        for tu in tus {
            let length = 1u32 << tu.log2_size;
            if tu.x != cu.x {
                self.mark_vertical(run(tu.x, tu.y, length));
            }
            if tu.y != cu.y {
                self.mark_horizontal(run(tu.x, tu.y, length));
            }
        }
        let (vertical, horizontal) = partition_offsets(cu);
        if let Some(offset) = vertical {
            self.mark_vertical(run(cu.x + offset, cu.y, size));
        }
        if let Some(offset) = horizontal {
            self.mark_horizontal(run(cu.x, cu.y + offset, size));
        }
    }

    /// The edges of an inter coding unit: the coding block, each transform block, and
    /// each prediction block (8.7.2.2, 8.7.2.3). A coding block edge is a transform
    /// block edge, also with no transform tree.
    fn mark_inter_cu(&mut self, input: &StrengthInput<'_>, cu: &CuRecord, tus: &[TuRecord]) {
        let size = 1u32 << cu.log2_size;
        let vertical_edge = |x, y, length, transform| InterEdge {
            vertical: true,
            x,
            y,
            length,
            transform,
        };
        let horizontal_edge = |x, y, length, transform| InterEdge {
            vertical: false,
            x,
            y,
            length,
            transform,
        };
        let (filter_left, filter_top) = filter_cb_edges(input, cu);
        if filter_left {
            self.mark_inter_edge(input, vertical_edge(cu.x, cu.y, size, true));
        }
        if filter_top {
            self.mark_inter_edge(input, horizontal_edge(cu.x, cu.y, size, true));
        }
        for tu in tus {
            let length = 1u32 << tu.log2_size;
            if tu.x != cu.x {
                self.mark_inter_edge(input, vertical_edge(tu.x, tu.y, length, true));
            }
            if tu.y != cu.y {
                self.mark_inter_edge(input, horizontal_edge(tu.x, tu.y, length, true));
            }
        }
        let (vertical, horizontal) = partition_offsets(cu);
        if let Some(offset) = vertical {
            self.mark_inter_edge(input, vertical_edge(cu.x + offset, cu.y, size, false));
        }
        if let Some(offset) = horizontal {
            self.mark_inter_edge(input, horizontal_edge(cu.x, cu.y + offset, size, false));
        }
    }

    /// `bS` of each section of one edge on the grid. The largest value of a section
    /// holds, because a prediction edge can lie over a transform edge.
    fn mark_inter_edge(&mut self, input: &StrengthInput<'_>, edge: InterEdge) {
        let across = if edge.vertical { edge.x } else { edge.y };
        if across & ((1 << GRID_LOG2) - 1) != 0 || across == 0 {
            return;
        }
        let mut last_pair = None;
        for offset in (0..edge.length).step_by(1 << SECTION_LOG2) {
            let (q, p) = if edge.vertical {
                ((edge.x, edge.y + offset), (edge.x - 1, edge.y + offset))
            } else {
                ((edge.x + offset, edge.y), (edge.x + offset, edge.y - 1))
            };
            let strength = self.section_strength(input, p, q, edge.transform, &mut last_pair);
            if strength > 0 {
                self.raise_section(edge.vertical, q, strength);
            }
        }
    }

    /// The conditions of 8.7.2.4 for one section with sides `p` and `q`. `last_pair`
    /// holds the last motion comparison of this edge, with its result.
    fn section_strength(
        &self,
        input: &StrengthInput<'_>,
        p: (u32, u32),
        q: (u32, u32),
        transform: bool,
        last_pair: &mut Option<(MotionPair, u8)>,
    ) -> u8 {
        if input.maps.cu_intra(p.0, p.1) || input.maps.cu_intra(q.0, q.1) {
            return BS_INTRA;
        }
        if transform && (self.is_coded(p.0, p.1) || self.is_coded(q.0, q.1)) {
            return BS_CODED;
        }
        let pair = MotionPair {
            p: input.field.at(p.0, p.1),
            q: input.field.at(q.0, q.1),
            tables: (
                input.field.table_at(p.0, p.1),
                input.field.table_at(q.0, q.1),
            ),
        };
        // One motion on one table names the same pictures and vectors on both sides.
        if pair.p == pair.q && pair.tables.0 == pair.tables.1 {
            return 0;
        }
        if let Some((last, strength)) = *last_pair
            && last == pair
        {
            return strength;
        }
        let strength = motion_strength(
            &EdgeSide {
                motion: pair.p,
                ref_poc: input.field.ref_pocs(p.0, p.1, &pair.p),
            },
            &EdgeSide {
                motion: pair.q,
                ref_poc: input.field.ref_pocs(q.0, q.1, &pair.q),
            },
        );
        *last_pair = Some((pair, strength));
        strength
    }

    /// Raises `bS` of the section with `q0` at `q`.
    fn raise_section(&mut self, vertical: bool, q: (u32, u32), strength: u8) {
        let (map, index) = if vertical {
            let stride = self.vertical_stride();
            let index = (q.1 >> SECTION_LOG2) as usize * stride + (q.0 >> GRID_LOG2) as usize;
            (&mut self.vertical, index)
        } else {
            let stride = self.horizontal_stride();
            let index = (q.1 >> GRID_LOG2) as usize * stride + (q.0 >> SECTION_LOG2) as usize;
            (&mut self.horizontal, index)
        };
        if let Some(entry) = map.get_mut(index)
            && *entry < strength
        {
            *entry = strength;
            self.any = true;
        }
    }
}

/// The prediction block edges of 8.7.2.3 inside a coding block: the vertical offset,
/// then the horizontal offset.
fn partition_offsets(cu: &CuRecord) -> (Option<u32>, Option<u32>) {
    let size = 1u32 << cu.log2_size;
    let (half, quarter) = (size / 2, size / 4);
    let vertical = match cu.part_mode {
        PartMode::NxTwoN | PartMode::NxN => Some(half),
        PartMode::NLxTwoN => Some(quarter),
        PartMode::NRxTwoN => Some(3 * quarter),
        _ => None,
    };
    let horizontal = match cu.part_mode {
        PartMode::TwoNxN | PartMode::NxN => Some(half),
        PartMode::TwoNxnU => Some(quarter),
        PartMode::TwoNxnD => Some(3 * quarter),
        _ => None,
    };
    (vertical, horizontal)
}

/// `filterLeftCbEdgeFlag` and `filterTopCbEdgeFlag` (8.7.2.1) of a coding block.
fn filter_cb_edges(input: &StrengthInput<'_>, cu: &CuRecord) -> (bool, bool) {
    let at = |x_nb, y_nb| {
        filter_across(
            input.params,
            input.layout,
            input.maps,
            (cu.x, cu.y),
            x_nb,
            y_nb,
        )
    };
    (
        at(cu.x.checked_sub(1), Some(cu.y)),
        at(Some(cu.x), cu.y.checked_sub(1)),
    )
}

/// `filterLeftCbEdgeFlag` or `filterTopCbEdgeFlag` (8.7.2.1) for the neighbouring
/// location. A coordinate of `None` lies outside the picture.
fn filter_across(
    params: &CodingParams,
    layout: &PicLayout,
    maps: &ParseMaps,
    current: (u32, u32),
    x_nb: Option<u32>,
    y_nb: Option<u32>,
) -> bool {
    let (Some(x_nb), Some(y_nb)) = (x_nb, y_nb) else {
        return false;
    };
    let here = layout.ctb_rs_at(current.0, current.1);
    let there = layout.ctb_rs_at(x_nb, y_nb);
    if here == there {
        return true;
    }
    let tiles_allow =
        params.loop_filter_across_tiles || layout.tile_id(here) == layout.tile_id(there);
    let slices_allow =
        params.loop_filter_across_slices || maps.ctb_slice(here) == maps.ctb_slice(there);
    tiles_allow && slices_allow
}

fn zeroed(columns: usize, rows: usize) -> Result<Vec<u8>, HevcError> {
    let Some(count) = columns.checked_mul(rows) else {
        return Err(HevcError::Allocation {
            what: "deblocking edges",
            count: usize::MAX,
        });
    };
    filled(count, 0, "deblocking edges")
}

#[cfg(test)]
#[path = "edges_tests.rs"]
mod tests;
