//! Block edges and `bS` on the 8x8 luma grid (8.7.2.2 to 8.7.2.4).

use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::layout::PicLayout;
use crate::decode::cpu::hevc::params::CodingParams;
use crate::decode::cpu::hevc::syntax::maps::ParseMaps;
use crate::decode::cpu::hevc::syntax::records::{CtuRecords, CuRecord, PartMode, PredMode};

/// `bS` when a side of the edge is intra (8.7.2.4).
const BS_INTRA: u8 = 2;
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
    width: u32,
    height: u32,
    any: bool,
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
        Ok(Self {
            vertical,
            horizontal,
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

    /// Marks the edges of the intra coding units of one CTU, for a slice with the
    /// deblocking filter on. Inter units add the rest of 8.7.2.4 here.
    pub(crate) fn mark_ctu(
        &mut self,
        params: &CodingParams,
        layout: &PicLayout,
        maps: &ParseMaps,
        records: &CtuRecords,
    ) {
        if params.deblocking_disabled {
            return;
        }
        for (index, cu) in records.cus.iter().enumerate() {
            if cu.pred_mode != PredMode::Intra {
                continue;
            }
            let size = 1u32 << cu.log2_size;
            let run = |x, y, length| EdgeRun {
                x,
                y,
                length,
                strength: BS_INTRA,
            };
            if filter_across(
                params,
                layout,
                maps,
                (cu.x, cu.y),
                cu.x.checked_sub(1),
                Some(cu.y),
            ) {
                self.mark_vertical(run(cu.x, cu.y, size));
            }
            if filter_across(
                params,
                layout,
                maps,
                (cu.x, cu.y),
                Some(cu.x),
                cu.y.checked_sub(1),
            ) {
                self.mark_horizontal(run(cu.x, cu.y, size));
            }
            let end = records
                .cus
                .get(index + 1)
                .map_or(records.tus.len(), |next| next.first_tu);
            for tu in records.tus.get(cu.first_tu..end).unwrap_or_default() {
                let length = 1u32 << tu.log2_size;
                if tu.x != cu.x {
                    self.mark_vertical(run(tu.x, tu.y, length));
                }
                if tu.y != cu.y {
                    self.mark_horizontal(run(tu.x, tu.y, length));
                }
            }
            self.mark_partitions(cu);
        }
    }

    /// The prediction block boundary of 8.7.2.3.
    fn mark_partitions(&mut self, cu: &CuRecord) {
        let size = 1u32 << cu.log2_size;
        let run = |x, y| EdgeRun {
            x,
            y,
            length: size,
            strength: BS_INTRA,
        };
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
        if let Some(offset) = vertical {
            self.mark_vertical(run(cu.x + offset, cu.y));
        }
        if let Some(offset) = horizontal {
            self.mark_horizontal(run(cu.x, cu.y + offset));
        }
    }
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
    let mut map = Vec::new();
    map.try_reserve_exact(count)
        .map_err(|_| HevcError::Allocation {
            what: "deblocking edges",
            count,
        })?;
    map.resize(count, 0);
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::cpu::hevc::layout::LayoutSizes;
    use crate::decode::cpu::hevc::syntax::records::TuRecord;

    fn cu(x: u32, y: u32, log2_size: u32, first_tu: usize) -> CuRecord {
        CuRecord {
            x,
            y,
            log2_size,
            pred_mode: PredMode::Intra,
            part_mode: PartMode::TwoNxTwoN,
            transquant_bypass: false,
            pcm: false,
            qp_y: 26,
            intra_luma: [1; 4],
            intra_chroma: 1,
            first_pu: 0,
            first_tu,
            first_pcm_sample: 0,
        }
    }

    fn tu(x: u32, y: u32, log2_size: u32) -> TuRecord {
        TuRecord {
            x,
            y,
            x_base: x,
            y_base: y,
            log2_size,
            depth: 0,
            blk_idx: 0,
            cbf_luma: false,
            cbf_cb: false,
            cbf_cr: false,
            cu: 0,
            first_tb: 0,
            tb_count: 0,
        }
    }

    struct Fixture {
        params: CodingParams,
        layout: PicLayout,
        maps: ParseMaps,
    }

    /// 32x32 luma with 16x16 CTBs in `columns` tile columns, every CTB in slice 0.
    fn fixture(columns: &[u32]) -> Fixture {
        let params = CodingParams {
            width: 32,
            height: 32,
            deblocking_disabled: false,
            ..CodingParams::for_tests()
        };
        let layout = PicLayout::new(&LayoutSizes {
            width: 32,
            height: 32,
            ctb_log2: 4,
            min_tb_log2: 2,
            column_widths: columns,
            row_heights: &[2],
        })
        .unwrap();
        let mut maps = ParseMaps::new(&params, &layout).unwrap();
        for rs in 0..4 {
            maps.set_ctb_slice(rs, 0);
        }
        Fixture {
            params,
            layout,
            maps,
        }
    }

    fn mark(fixture: &Fixture, records: &CtuRecords) -> EdgeMap {
        let mut edges = EdgeMap::new(32, 32).unwrap();
        edges.mark_ctu(&fixture.params, &fixture.layout, &fixture.maps, records);
        edges
    }

    #[test]
    fn picture_boundaries_take_no_edge() {
        let fixture = fixture(&[2]);
        let records = CtuRecords {
            cus: vec![cu(0, 0, 4, 0)],
            tus: vec![tu(0, 0, 4)],
            ..CtuRecords::default()
        };
        let edges = mark(&fixture, &records);
        assert_eq!(edges.vertical_at(0, 0), 0);
        assert_eq!(edges.horizontal_at(0, 0), 0);
        assert!(!edges.has_edges());
    }

    #[test]
    fn a_unit_marks_its_left_and_top_edges_with_strength_2() {
        let fixture = fixture(&[2]);
        let records = CtuRecords {
            cus: vec![cu(16, 16, 4, 0)],
            tus: vec![tu(16, 16, 4)],
            ..CtuRecords::default()
        };
        let edges = mark(&fixture, &records);
        for offset in (0..16).step_by(4) {
            assert_eq!(edges.vertical_at(16, 16 + offset), 2);
            assert_eq!(edges.horizontal_at(16 + offset, 16), 2);
        }
        assert_eq!(edges.vertical_at(16, 12), 0);
        assert_eq!(edges.horizontal_at(12, 16), 0);
        assert!(edges.has_edges());
    }

    /// Inside a unit, a transform edge on the grid is marked, and one off the grid
    /// is not (8.7.2.1).
    #[test]
    fn transform_edges_keep_to_the_grid() {
        let fixture = fixture(&[2]);
        let records = CtuRecords {
            cus: vec![cu(0, 0, 4, 0)],
            tus: vec![
                tu(0, 0, 3),
                tu(8, 0, 2),
                tu(12, 0, 2),
                tu(8, 4, 2),
                tu(12, 4, 2),
                tu(0, 8, 3),
                tu(8, 8, 3),
            ],
            ..CtuRecords::default()
        };
        let edges = mark(&fixture, &records);
        assert_eq!(edges.vertical_at(8, 0), 2);
        assert_eq!(edges.vertical_at(8, 4), 2);
        assert_eq!(edges.vertical_at(8, 12), 2);
        assert_eq!(edges.horizontal_at(0, 8), 2);
        assert_eq!(edges.horizontal_at(12, 8), 2);
        // y = 4 lies off the grid, and no edge holds row 0 of the picture.
        assert_eq!(edges.horizontal_at(8, 4), 0);
    }

    #[test]
    fn an_edge_off_the_grid_is_dropped() {
        let mut edges = EdgeMap::new(16, 16).unwrap();
        let run = EdgeRun {
            x: 12,
            y: 4,
            length: 4,
            strength: 2,
        };
        edges.mark_vertical(run);
        edges.mark_horizontal(run);
        assert!(!edges.has_edges());
    }

    #[test]
    fn a_disabled_slice_marks_nothing() {
        let mut fixture = fixture(&[2]);
        fixture.params.deblocking_disabled = true;
        let records = CtuRecords {
            cus: vec![cu(16, 16, 4, 0)],
            tus: vec![tu(16, 16, 4)],
            ..CtuRecords::default()
        };
        assert!(!mark(&fixture, &records).has_edges());
    }

    /// The current slice's flag decides its left and top boundaries (8.7.2.1).
    #[test]
    fn a_slice_boundary_follows_the_current_slice_flag() {
        let mut fixture = fixture(&[2]);
        fixture.maps.set_ctb_slice(1, 1);
        fixture.params.loop_filter_across_slices = false;
        let records = CtuRecords {
            cus: vec![cu(16, 0, 4, 0)],
            tus: vec![tu(16, 0, 4)],
            ..CtuRecords::default()
        };
        assert_eq!(mark(&fixture, &records).vertical_at(16, 0), 0);
        fixture.params.loop_filter_across_slices = true;
        assert_eq!(mark(&fixture, &records).vertical_at(16, 0), 2);
    }

    #[test]
    fn a_tile_boundary_follows_the_pps_flag() {
        let mut fixture = fixture(&[1, 1]);
        fixture.params.loop_filter_across_tiles = false;
        let records = CtuRecords {
            cus: vec![cu(16, 0, 4, 0)],
            tus: vec![tu(16, 0, 4)],
            ..CtuRecords::default()
        };
        assert_eq!(mark(&fixture, &records).vertical_at(16, 0), 0);
        fixture.params.loop_filter_across_tiles = true;
        assert_eq!(mark(&fixture, &records).vertical_at(16, 0), 2);
    }

    #[test]
    fn nxn_marks_the_middle_of_a_large_unit() {
        let fixture = fixture(&[2]);
        let mut unit = cu(0, 0, 4, 0);
        unit.part_mode = PartMode::NxN;
        let records = CtuRecords {
            cus: vec![unit],
            ..CtuRecords::default()
        };
        let edges = mark(&fixture, &records);
        assert_eq!(edges.vertical_at(8, 12), 2);
        assert_eq!(edges.horizontal_at(12, 8), 2);
    }

    #[test]
    fn reset_removes_every_edge() {
        let mut edges = EdgeMap::new(16, 16).unwrap();
        edges.mark_vertical(EdgeRun {
            x: 8,
            y: 0,
            length: 16,
            strength: 2,
        });
        assert_eq!(edges.vertical_at(8, 12), 2);
        edges.reset();
        assert_eq!(edges.vertical_at(8, 12), 0);
        assert!(!edges.has_edges());
    }
}
