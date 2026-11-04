//! Per-picture parse state that neighbouring blocks read, and the z-scan
//! availability process (6.4.1).

use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::layout::PicLayout;
use crate::decode::cpu::hevc::params::CodingParams;

use super::sao::SaoCtb;

/// `SliceAddrRs` of a CTB no slice has reached in this picture.
const NO_SLICE: u32 = u32::MAX;
/// `IntraPredModeY` units are 4x4 luma samples: the smallest prediction block.
const INTRA_UNIT_LOG2: u32 = 2;
/// `INTRA_DC` (Table 8-1).
pub(crate) const INTRA_DC: u8 = 1;

/// What the in-loop filters read from the slice that holds one CTB.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CtbFilter {
    /// `slice_beta_offset_div2`.
    pub(crate) beta_offset_div2: i32,
    /// `slice_tc_offset_div2`.
    pub(crate) tc_offset_div2: i32,
    /// `slice_loop_filter_across_slices_enabled_flag`.
    pub(crate) across_slices: bool,
}

/// A square region to fill, in luma samples.
#[derive(Clone, Copy)]
struct Region {
    x: u32,
    y: u32,
    size: u32,
}

/// Neighbour data of one picture. The check in `check_stream` bounds its size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ParseMaps {
    min_cb_log2: u32,
    /// Map width in minimum coding blocks.
    cb_stride: usize,
    /// Map width in 4x4 units.
    intra_stride: usize,
    /// `CtDepth`, per minimum coding block.
    ct_depth: Vec<u8>,
    /// `cu_skip_flag`, per minimum coding block.
    cu_skip: Vec<bool>,
    /// `QpY`, per minimum coding block.
    qp_y: Vec<i8>,
    /// `IntraPredModeY`, per 4x4 unit. A block with no mode holds `INTRA_DC`,
    /// the candidate 8.4.2 assigns to it.
    intra_mode: Vec<u8>,
    /// True when `CuPredMode` is `MODE_INTRA`, per minimum coding block.
    cu_intra: Vec<bool>,
    /// True when neither in-loop filter can change the samples, per minimum coding
    /// block: `cu_transquant_bypass_flag` 1, or PCM with
    /// `pcm_loop_filter_disabled_flag` 1.
    no_filter: Vec<bool>,
    /// `SliceAddrRs`, per CTB in raster order.
    ctb_slice: Vec<u32>,
    /// The slice values of each CTB, in raster order.
    ctb_filter: Vec<CtbFilter>,
    /// The SAO values of each CTB, in raster order.
    sao: Vec<SaoCtb>,
}

impl ParseMaps {
    /// Allocates the maps of one picture size.
    ///
    /// # Errors
    ///
    /// [`HevcError::Allocation`] for a size that does not fit in memory.
    pub(crate) fn new(params: &CodingParams, layout: &PicLayout) -> Result<Self, HevcError> {
        let cb_stride = (params.width >> params.min_cb_log2) as usize;
        let cb_rows = (params.height >> params.min_cb_log2) as usize;
        let intra_stride = (params.width >> INTRA_UNIT_LOG2) as usize;
        let intra_rows = (params.height >> INTRA_UNIT_LOG2) as usize;
        let cb_count = checked_area(cb_stride, cb_rows)?;
        let intra_count = checked_area(intra_stride, intra_rows)?;
        Ok(Self {
            min_cb_log2: params.min_cb_log2,
            cb_stride,
            intra_stride,
            ct_depth: filled(cb_count, 0, "CtDepth")?,
            cu_skip: filled(cb_count, false, "cu_skip_flag")?,
            qp_y: filled(cb_count, 0, "QpY")?,
            intra_mode: filled(intra_count, INTRA_DC, "IntraPredModeY")?,
            cu_intra: filled(cb_count, false, "CuPredMode")?,
            no_filter: filled(cb_count, false, "in-loop filter flags")?,
            ctb_slice: filled(layout.ctb_count(), NO_SLICE, "SliceAddrRs")?,
            ctb_filter: filled(
                layout.ctb_count(),
                CtbFilter::default(),
                "CTB filter values",
            )?,
            sao: filled(layout.ctb_count(), SaoCtb::default(), "SAO values")?,
        })
    }

    /// Prepares the maps for a new picture of the same size.
    pub(crate) fn reset(&mut self) {
        self.ctb_slice.fill(NO_SLICE);
    }

    /// Records the slice that holds CTB `rs`.
    pub(crate) fn set_ctb_slice(&mut self, rs: usize, slice_addr_rs: u32) {
        if let Some(entry) = self.ctb_slice.get_mut(rs) {
            *entry = slice_addr_rs;
        }
    }

    /// True when a slice has reached CTB `rs` in this picture.
    pub(crate) fn ctb_reached(&self, rs: usize) -> bool {
        self.ctb_slice
            .get(rs)
            .is_some_and(|&slice| slice != NO_SLICE)
    }

    /// `SliceAddrRs` of CTB `rs`. A CTB outside the picture returns no slice.
    pub(crate) fn ctb_slice(&self, rs: usize) -> u32 {
        self.ctb_slice.get(rs).copied().unwrap_or(NO_SLICE)
    }

    /// Records the slice values of CTB `rs`, and clears its SAO values.
    pub(crate) fn begin_ctb_filter(&mut self, rs: usize, filter: CtbFilter) {
        if let Some(entry) = self.ctb_filter.get_mut(rs) {
            *entry = filter;
        }
        self.set_sao(rs, SaoCtb::default());
    }

    /// The slice values of CTB `rs`.
    pub(crate) fn ctb_filter(&self, rs: usize) -> CtbFilter {
        self.ctb_filter.get(rs).copied().unwrap_or_default()
    }

    /// The SAO values of CTB `rs`.
    pub(crate) fn sao(&self, rs: usize) -> SaoCtb {
        self.sao.get(rs).copied().unwrap_or_default()
    }

    /// Records the SAO values of CTB `rs`.
    pub(crate) fn set_sao(&mut self, rs: usize, values: SaoCtb) {
        if let Some(entry) = self.sao.get_mut(rs) {
            *entry = values;
        }
    }

    /// `MinCbLog2SizeY`.
    pub(crate) const fn min_cb_log2(&self) -> u32 {
        self.min_cb_log2
    }

    /// The availability derivation of 6.4.1.
    pub(crate) fn available(
        &self,
        layout: &PicLayout,
        current: (u32, u32),
        x_nb: i32,
        y_nb: i32,
    ) -> bool {
        let (Ok(x), Ok(y)) = (u32::try_from(x_nb), u32::try_from(y_nb)) else {
            return false;
        };
        if x >= layout.width() || y >= layout.height() {
            return false;
        }
        let ctb_nb = layout.ctb_rs_at(x, y);
        let ctb_current = layout.ctb_rs_at(current.0, current.1);
        if ctb_nb == ctb_current {
            return layout.min_tb_addr_in_ctb(x, y)
                <= layout.min_tb_addr_in_ctb(current.0, current.1);
        }
        if layout.min_tb_addr_zs(x, y) > layout.min_tb_addr_zs(current.0, current.1) {
            return false;
        }
        self.ctb_slice.get(ctb_nb) == self.ctb_slice.get(ctb_current)
            && layout.tile_id(ctb_nb) == layout.tile_id(ctb_current)
    }

    fn cb_index(&self, x: u32, y: u32) -> usize {
        (y >> self.min_cb_log2) as usize * self.cb_stride + (x >> self.min_cb_log2) as usize
    }

    fn intra_index(&self, x: u32, y: u32) -> usize {
        (y >> INTRA_UNIT_LOG2) as usize * self.intra_stride + (x >> INTRA_UNIT_LOG2) as usize
    }

    /// `CtDepth` at a luma location.
    pub(crate) fn ct_depth(&self, x: u32, y: u32) -> u8 {
        self.ct_depth.get(self.cb_index(x, y)).copied().unwrap_or(0)
    }

    /// `cu_skip_flag` at a luma location.
    pub(crate) fn cu_skip(&self, x: u32, y: u32) -> bool {
        self.cu_skip
            .get(self.cb_index(x, y))
            .copied()
            .unwrap_or(false)
    }

    /// `QpY` at a luma location.
    pub(crate) fn qp_y(&self, x: u32, y: u32) -> i32 {
        i32::from(self.qp_y.get(self.cb_index(x, y)).copied().unwrap_or(0))
    }

    /// `IntraPredModeY` at a luma location.
    pub(crate) fn intra_mode(&self, x: u32, y: u32) -> u8 {
        self.intra_mode
            .get(self.intra_index(x, y))
            .copied()
            .unwrap_or(INTRA_DC)
    }

    /// True when `CuPredMode` at a luma location is `MODE_INTRA`.
    pub(crate) fn cu_intra(&self, x: u32, y: u32) -> bool {
        self.cu_intra
            .get(self.cb_index(x, y))
            .copied()
            .unwrap_or(false)
    }

    /// True when neither in-loop filter can change the sample at a luma location.
    pub(crate) fn no_filter(&self, x: u32, y: u32) -> bool {
        self.no_filter
            .get(self.cb_index(x, y))
            .copied()
            .unwrap_or(false)
    }

    /// True when any block holds the flag of [`Self::no_filter`].
    pub(crate) fn any_no_filter(&self) -> bool {
        self.no_filter.iter().any(|&flag| flag)
    }

    /// Records the prediction mode and in-loop filter flag of a coding unit.
    pub(crate) fn set_cu_kind(&mut self, x: u32, y: u32, size: u32, intra: bool, no_filter: bool) {
        let region = Region { x, y, size };
        let (stride, log2) = (self.cb_stride, self.min_cb_log2);
        fill_region(&mut self.cu_intra, stride, log2, region, intra);
        fill_region(&mut self.no_filter, stride, log2, region, no_filter);
    }

    /// Records `CtDepth` and `cu_skip_flag` of a coding unit.
    pub(crate) fn set_cu(&mut self, x: u32, y: u32, size: u32, depth: u8, skip: bool) {
        let region = Region { x, y, size };
        let (stride, log2) = (self.cb_stride, self.min_cb_log2);
        fill_region(&mut self.ct_depth, stride, log2, region, depth);
        fill_region(&mut self.cu_skip, stride, log2, region, skip);
    }

    /// Records `QpY` of a coding unit. (8-283) holds it within -12 to 51.
    pub(crate) fn set_qp_y(&mut self, x: u32, y: u32, size: u32, qp_y: i32) {
        let region = Region { x, y, size };
        let (stride, log2) = (self.cb_stride, self.min_cb_log2);
        fill_region(&mut self.qp_y, stride, log2, region, qp_y as i8);
    }

    /// Records `IntraPredModeY` of a prediction block.
    pub(crate) fn set_intra_mode(&mut self, x: u32, y: u32, size: u32, mode: u8) {
        let region = Region { x, y, size };
        let stride = self.intra_stride;
        fill_region(&mut self.intra_mode, stride, INTRA_UNIT_LOG2, region, mode);
    }
}

fn checked_area(width: usize, height: usize) -> Result<usize, HevcError> {
    width.checked_mul(height).ok_or(HevcError::Allocation {
        what: "parse maps",
        count: usize::MAX,
    })
}

/// A vector of `count` copies of `value`, or an allocation error that names `what`.
pub(crate) fn filled<T: Clone>(
    count: usize,
    value: T,
    what: &'static str,
) -> Result<Vec<T>, HevcError> {
    let mut map = Vec::new();
    map.try_reserve_exact(count)
        .map_err(|_| HevcError::Allocation { what, count })?;
    map.resize(count, value);
    Ok(map)
}

/// Writes `value` over `region`. A block past the picture edge writes only the
/// entries inside the picture.
fn fill_region<T: Copy>(map: &mut [T], stride: usize, log2_unit: u32, region: Region, value: T) {
    if stride == 0 {
        return;
    }
    let left = (region.x >> log2_unit) as usize;
    let right = ((region.x + region.size) >> log2_unit) as usize;
    let top = (region.y >> log2_unit) as usize;
    let rows = (region.size >> log2_unit) as usize;
    for row in map.chunks_exact_mut(stride).skip(top).take(rows) {
        if let Some(cells) = row.get_mut(left..right.min(stride)) {
            cells.fill(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::cpu::hevc::layout::LayoutSizes;

    fn setup(width: u32, height: u32, columns: &[u32]) -> (ParseMaps, PicLayout) {
        let layout = PicLayout::new(&LayoutSizes {
            width,
            height,
            ctb_log2: 4,
            min_tb_log2: 2,
            column_widths: columns,
            row_heights: &[height.div_ceil(16)],
        })
        .unwrap();
        let params = CodingParams {
            width,
            height,
            ..CodingParams::for_tests()
        };
        let mut maps = ParseMaps::new(&params, &layout).unwrap();
        for rs in 0..layout.ctb_count() {
            maps.set_ctb_slice(rs, 0);
        }
        (maps, layout)
    }

    #[test]
    fn a_location_outside_the_picture_is_unavailable() {
        let (maps, layout) = setup(32, 32, &[2]);
        assert!(!maps.available(&layout, (0, 0), -1, 0));
        assert!(!maps.available(&layout, (0, 0), 0, -1));
        assert!(!maps.available(&layout, (28, 28), 32, 28));
        assert!(!maps.available(&layout, (28, 28), 28, 32));
    }

    #[test]
    fn a_later_block_in_z_scan_is_unavailable() {
        let (maps, layout) = setup(32, 32, &[2]);
        assert!(maps.available(&layout, (8, 0), 7, 0));
        assert!(!maps.available(&layout, (7, 0), 8, 0));
        // Inside a CTB, the top right block of a split precedes the bottom left.
        assert!(maps.available(&layout, (0, 8), 8, 7));
        assert!(!maps.available(&layout, (8, 7), 0, 8));
        // The next CTB is later, and the one above is earlier.
        assert!(!maps.available(&layout, (15, 0), 16, 0));
        assert!(maps.available(&layout, (16, 16), 16, 15));
    }

    #[test]
    fn another_slice_is_unavailable() {
        let (mut maps, layout) = setup(32, 32, &[2]);
        maps.set_ctb_slice(1, 1);
        assert!(!maps.available(&layout, (16, 0), 15, 0));
        // CTB 1 and CTB 3 differ too, and CTB 0 and CTB 2 still match.
        assert!(!maps.available(&layout, (16, 16), 16, 15));
        assert!(maps.available(&layout, (0, 16), 0, 15));
    }

    #[test]
    fn a_ctb_no_slice_reached_is_unavailable() {
        let (mut maps, layout) = setup(32, 32, &[2]);
        maps.reset();
        maps.set_ctb_slice(1, 0);
        assert!(!maps.available(&layout, (16, 0), 15, 0));
    }

    #[test]
    fn another_tile_is_unavailable() {
        let (maps, layout) = setup(32, 16, &[1, 1]);
        assert!(!maps.available(&layout, (16, 0), 15, 0));
        assert!(maps.available(&layout, (8, 0), 7, 0));
    }

    #[test]
    fn regions_fill_exactly_and_clamp_at_the_edge() {
        let (mut maps, _layout) = setup(24, 16, &[2]);
        maps.set_cu(8, 0, 8, 2, true);
        assert_eq!(maps.ct_depth(8, 7), 2);
        assert_eq!(maps.ct_depth(15, 0), 2);
        assert_eq!(maps.ct_depth(16, 0), 0);
        assert_eq!(maps.ct_depth(8, 8), 0);
        assert!(maps.cu_skip(12, 4));
        assert!(!maps.cu_skip(4, 4));
        // A block past the right edge does not wrap to the next row.
        maps.set_qp_y(16, 0, 16, 40);
        assert_eq!(maps.qp_y(23, 15), 40);
        assert_eq!(maps.qp_y(0, 8), 0);
        maps.set_qp_y(0, 8, 8, -12);
        assert_eq!(maps.qp_y(7, 15), -12);
        maps.set_intra_mode(4, 4, 4, 26);
        assert_eq!(maps.intra_mode(7, 7), 26);
        assert_eq!(maps.intra_mode(8, 4), INTRA_DC);
        assert_eq!(maps.intra_mode(4, 8), INTRA_DC);
    }
}
