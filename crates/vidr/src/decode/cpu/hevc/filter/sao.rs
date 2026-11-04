//! The sample adaptive offset process on a whole picture (8.7.3).
//!
//! Every read comes from a copy of the deblocked plane, and a CTB writes only its own
//! rows. So CTB rows run in any order, one writer per row.

use fearless_simd::Simd;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

use crate::decode::cpu::dsp::sample::Sample;
use crate::decode::cpu::dsp::sao::{band_run, edge_run};
use crate::decode::cpu::dsp::simd::{Cpu, with_simd};
use crate::decode::cpu::hevc::layout::PicLayout;
use crate::decode::cpu::hevc::params::CodingParams;
use crate::decode::cpu::hevc::picture::{Picture, Plane, PlaneRows};
use crate::decode::cpu::hevc::syntax::maps::ParseMaps;
use crate::decode::cpu::hevc::syntax::sao::{SaoKind, SaoPlane};

/// `hPos` and `vPos` of each neighbour (Table 8-13), by `SaoEoClass`.
const EDGE_NEIGHBOURS: [[(isize, isize); 2]; 4] = [
    [(-1, 0), (1, 0)],
    [(0, -1), (0, 1)],
    [(-1, -1), (1, 1)],
    [(1, -1), (-1, 1)],
];

/// What the process reads besides the samples.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SaoContext<'a> {
    /// Any slice of the picture: the fields read here hold for the whole picture.
    pub(crate) params: &'a CodingParams,
    pub(crate) layout: &'a PicLayout,
    pub(crate) maps: &'a ParseMaps,
}

/// One CTB's samples in one plane.
#[derive(Clone, Copy, Debug)]
struct CtbArea {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    /// 0 for luma, and 1 for chroma.
    chroma_shift: u32,
}

/// True when SAO changes at least one plane of `picture`, so a copy is needed.
pub(crate) fn sao_is_active(context: &SaoContext<'_>) -> bool {
    (0..context.layout.ctb_count()).any(|rs| {
        context
            .maps
            .sao(rs)
            .planes
            .iter()
            .any(|plane| plane.kind != SaoKind::Off)
    })
}

/// Runs 8.7.3 on `picture`. `deblocked` holds the working copy of `recPicture`, and
/// has the size of `picture`.
pub(crate) fn sao_picture<S: Sample>(
    picture: &mut Picture<S>,
    deblocked: &mut Picture<S>,
    context: &SaoContext<'_>,
) {
    for c_idx in 0..3 {
        let plane_active = (0..context.layout.ctb_count())
            .any(|rs| context.maps.sao(rs).planes[c_idx].kind != SaoKind::Off);
        if !plane_active {
            continue;
        }
        deblocked
            .plane_mut(c_idx)
            .copy_samples_from(picture.plane(c_idx));
        let source = deblocked.plane(c_idx);
        let mut target = picture.plane_mut(c_idx).rows_mut();
        for rs in 0..context.layout.ctb_count() {
            let values = context.maps.sao(rs).planes[c_idx];
            if values.kind != SaoKind::Off {
                sao_ctb(source, &mut target, context, rs, c_idx, &values);
            }
        }
    }
}

/// [`sao_picture`] with the copy and each CTB row on the current rayon pool. The output
/// equals it.
pub(crate) fn sao_picture_in_rows<S: Sample>(
    picture: &mut Picture<S>,
    deblocked: &mut Picture<S>,
    context: &SaoContext<'_>,
) {
    let width_in_ctbs = (context.layout.width_in_ctbs() as usize).max(1);
    let ctb_rows = context.layout.ctb_count() / width_in_ctbs;
    for c_idx in 0..3 {
        let plane_active = (0..context.layout.ctb_count())
            .any(|rs| context.maps.sao(rs).planes[c_idx].kind != SaoKind::Off);
        if !plane_active {
            continue;
        }
        let ctb_height = (1usize << context.layout.ctb_log2()) >> usize::from(c_idx > 0);
        let chunk = picture
            .plane(c_idx)
            .stride()
            .saturating_mul(ctb_height)
            .max(1);
        let source = picture.plane(c_idx).samples();
        let copy = deblocked.plane_mut(c_idx).samples_mut();
        if copy.len() == source.len() {
            copy.par_chunks_mut(chunk)
                .zip(source.par_chunks(chunk))
                .for_each(|(to, from)| to.copy_from_slice(from));
        }
        let source = deblocked.plane(c_idx);
        let ends = (1..=ctb_rows).map(|row| {
            if row == ctb_rows {
                usize::MAX
            } else {
                row * ctb_height
            }
        });
        let bands = picture.plane_mut(c_idx).split_rows(ends);
        bands
            .into_par_iter()
            .enumerate()
            .for_each(|(row, mut target)| {
                for rs in row * width_in_ctbs..(row + 1) * width_in_ctbs {
                    let values = context.maps.sao(rs).planes[c_idx];
                    if values.kind != SaoKind::Off {
                        sao_ctb(source, &mut target, context, rs, c_idx, &values);
                    }
                }
            });
    }
}

/// The CTB modification process (8.7.3.2).
fn sao_ctb<S: Sample>(
    source: &Plane<S>,
    target: &mut PlaneRows<'_, S>,
    context: &SaoContext<'_>,
    rs: usize,
    c_idx: usize,
    values: &SaoPlane,
) {
    let width_in_ctbs = context.layout.width_in_ctbs() as usize;
    let chroma_shift = u32::from(c_idx > 0);
    let size = (1usize << context.layout.ctb_log2()) >> chroma_shift;
    let (x0, y0) = ((rs % width_in_ctbs) * size, (rs / width_in_ctbs) * size);
    let area = CtbArea {
        x0,
        y0,
        x1: (x0 + size).min(source.width()),
        y1: (y0 + size).min(source.height()),
        chroma_shift,
    };
    let bit_depth = context.params.bit_depth;
    let max_value = (1i32 << bit_depth) - 1;
    match values.kind {
        SaoKind::Off => {}
        SaoKind::Band(position) => {
            with_simd!(Cpu::detected(), simd => {
                for y in area.y0..area.y1 {
                    if let (Some(from), Some(to)) = (
                        source.row(y).and_then(|row| row.get(area.x0..area.x1)),
                        target
                            .row_mut(y)
                            .and_then(|row| row.get_mut(area.x0..area.x1)),
                    ) {
                        band_run(
                            simd,
                            from,
                            to,
                            position,
                            &values.offsets,
                            bit_depth - 5,
                            max_value,
                        );
                    }
                }
            });
        }
        SaoKind::Edge(class) => {
            let allowed = allowed_neighbours(context, rs);
            let edge = EdgeRow {
                area: &area,
                neighbours: EDGE_NEIGHBOURS[usize::from(class) & 3],
                allowed: &allowed,
                offsets: &values.offsets,
                max_value,
            };
            with_simd!(Cpu::detected(), simd => {
                for y in area.y0..area.y1 {
                    edge_row(simd, source, target, y, &edge);
                }
            });
        }
    }
    restore_unfiltered(source, target, context, &area);
}

/// Which of the 3x3 CTBs around CTB `rs` a neighbour sample can lie in, indexed by
/// `( dy + 1 ) * 3 + dx + 1`. These are the conditions on `edgeIdx` 0 in 8.7.3.2. CTBs
/// bound slices and tiles, so one flag serves all samples in a neighbouring CTB.
fn allowed_neighbours(context: &SaoContext<'_>, rs: usize) -> [bool; 9] {
    let (layout, maps) = (context.layout, context.maps);
    let width = layout.width_in_ctbs() as isize;
    let height = (layout.ctb_count() as isize) / width;
    let (column, row) = (rs as isize % width, rs as isize / width);
    let here_ts = layout.rs_to_ts(rs);
    let here_flag = maps.ctb_filter(rs).across_slices;
    let mut allowed = [false; 9];
    for (index, flag) in allowed.iter_mut().enumerate() {
        let (dx, dy) = ((index % 3) as isize - 1, (index / 3) as isize - 1);
        let (nb_column, nb_row) = (column + dx, row + dy);
        if nb_column < 0 || nb_row < 0 || nb_column >= width || nb_row >= height {
            continue;
        }
        let there = (nb_row * width + nb_column) as usize;
        let slice_ok = maps.ctb_slice(there) == maps.ctb_slice(rs)
            || if layout.rs_to_ts(there) < here_ts {
                here_flag
            } else {
                maps.ctb_filter(there).across_slices
            };
        let tile_ok =
            context.params.loop_filter_across_tiles || layout.tile_id(there) == layout.tile_id(rs);
        *flag = slice_ok && tile_ok;
    }
    allowed
}

/// `-1`, 0, or 1 for a coordinate before, inside, or after `start..end`.
const fn side_of(position: isize, start: usize, end: usize) -> isize {
    if position < start as isize {
        -1
    } else if position >= end as isize {
        1
    } else {
        0
    }
}

/// The values one row of edge offsets reads.
#[derive(Clone, Copy, Debug)]
struct EdgeRow<'a> {
    area: &'a CtbArea,
    /// `hPos` and `vPos` of each neighbour.
    neighbours: [(isize, isize); 2],
    allowed: &'a [bool; 9],
    /// `SaoOffsetVal`.
    offsets: &'a [i16; 5],
    max_value: i32,
}

/// One row of edge offsets. A sample's neighbours leave the CTB's columns only in the
/// first and last column, and the flags hold for a whole row, so the row splits only where
/// a flag differs. A sample with a neighbour that is not allowed keeps its deblocked
/// value: `target` holds that value already.
#[inline(always)]
fn edge_row<V: Simd, S: Sample>(
    simd: V,
    source: &Plane<S>,
    target: &mut PlaneRows<'_, S>,
    y: usize,
    edge: &EdgeRow<'_>,
) {
    let area = edge.area;
    if area.x1 <= area.x0 {
        return;
    }
    let [(h0, v0), (h1, v1)] = edge.neighbours;
    let rows = (y as isize + v0, y as isize + v1);
    let cell_a = (side_of(rows.0, area.y0, area.y1) + 1) * 3;
    let cell_b = (side_of(rows.1, area.y0, area.y1) + 1) * 3;
    let span = EdgeSpan { y, rows, h0, h1 };
    let inner_ok = edge.allowed[(cell_a + 1) as usize] && edge.allowed[(cell_b + 1) as usize];
    if h0 == 0 {
        if inner_ok {
            edge_span(simd, source, target, &span, (area.x0, area.x1), edge);
        }
        return;
    }
    let cell = |x: isize, row_cell: isize| {
        edge.allowed[(row_cell + side_of(x, area.x0, area.x1) + 1) as usize]
    };
    let column_ok = |x: usize| cell(x as isize + h0, cell_a) && cell(x as isize + h1, cell_b);
    let (first, last) = (area.x0, area.x1 - 1);
    let (first_ok, last_ok) = (column_ok(first), column_ok(last));
    // Columns between the first and last read neighbours inside the CTB's columns.
    let has_inner = last > first + 1;
    if first_ok && last_ok && (inner_ok || !has_inner) {
        edge_span(simd, source, target, &span, (first, area.x1), edge);
        return;
    }
    if first_ok {
        edge_span(simd, source, target, &span, (first, first + 1), edge);
    }
    if inner_ok && has_inner {
        edge_span(simd, source, target, &span, (first + 1, last), edge);
    }
    if last_ok && last > first {
        edge_span(simd, source, target, &span, (last, last + 1), edge);
    }
}

/// The row and neighbours of one row of edge offsets.
#[derive(Clone, Copy, Debug)]
struct EdgeSpan {
    y: usize,
    /// The rows of the two neighbours.
    rows: (isize, isize),
    /// `hPos` of the first neighbour.
    h0: isize,
    /// `hPos` of the second neighbour.
    h1: isize,
}

/// Edge offsets on the columns `columns.0` to `columns.1 - 1` of one row.
#[inline(always)]
fn edge_span<V: Simd, S: Sample>(
    simd: V,
    source: &Plane<S>,
    target: &mut PlaneRows<'_, S>,
    span: &EdgeSpan,
    columns: (usize, usize),
    edge: &EdgeRow<'_>,
) {
    let (start, end) = columns;
    let slices = (
        source.row(span.y).and_then(|row| row.get(start..end)),
        neighbour_run(source, span.rows.0, start, end, span.h0),
        neighbour_run(source, span.rows.1, start, end, span.h1),
        target
            .row_mut(span.y)
            .and_then(|row| row.get_mut(start..end)),
    );
    if let (Some(from), Some(first), Some(second), Some(to)) = slices {
        edge_run(
            simd,
            from,
            (first, second),
            to,
            edge.offsets,
            edge.max_value,
        );
    }
}

/// The samples of row `y` at columns `start + h` to `end + h`, or `None` outside the
/// plane.
#[inline(always)]
fn neighbour_run<S: Sample>(
    plane: &Plane<S>,
    y: isize,
    start: usize,
    end: usize,
    h: isize,
) -> Option<&[S]> {
    let row = plane.row(usize::try_from(y).ok()?)?;
    let first = start.checked_add_signed(h)?;
    let last = end.checked_add_signed(h)?;
    row.get(first..last)
}

/// Puts back the samples of the coding units that 8.7.3.2 does not modify:
/// `cu_transquant_bypass_flag` 1, or PCM with `pcm_loop_filter_disabled_flag` 1.
fn restore_unfiltered<S: Sample>(
    source: &Plane<S>,
    target: &mut PlaneRows<'_, S>,
    context: &SaoContext<'_>,
    area: &CtbArea,
) {
    let params = context.params;
    if !params.transquant_bypass && !params.pcm_loop_filter_disabled {
        return;
    }
    let unit = (1usize << context.maps.min_cb_log2()) >> area.chroma_shift;
    for y in (area.y0..area.y1).step_by(unit) {
        for x in (area.x0..area.x1).step_by(unit) {
            let luma_x = (x << area.chroma_shift) as u32;
            let luma_y = (y << area.chroma_shift) as u32;
            if !context.maps.no_filter(luma_x, luma_y) {
                continue;
            }
            for line in y..y + unit {
                if let (Some(from), Some(to)) = (
                    source.row(line).and_then(|row| row.get(x..x + unit)),
                    target
                        .row_mut(line)
                        .and_then(|row| row.get_mut(x..x + unit)),
                ) {
                    to.copy_from_slice(from);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::cpu::hevc::layout::LayoutSizes;
    use crate::decode::cpu::hevc::syntax::maps::CtbFilter;
    use crate::decode::cpu::hevc::syntax::sao::SaoCtb;

    struct Fixture {
        params: CodingParams,
        layout: PicLayout,
        maps: ParseMaps,
    }

    /// 32x16 luma with 16x16 CTBs in `tiles` columns, all in slice 0.
    fn fixture(tiles: &[u32]) -> Fixture {
        let params = CodingParams {
            width: 32,
            height: 16,
            ..CodingParams::for_tests()
        };
        let layout = PicLayout::new(&LayoutSizes {
            width: 32,
            height: 16,
            ctb_log2: 4,
            min_tb_log2: 2,
            column_widths: tiles,
            row_heights: &[1],
        })
        .unwrap();
        let mut maps = ParseMaps::new(&params, &layout).unwrap();
        maps.set_ctb_slice(0, 0);
        maps.set_ctb_slice(1, 0);
        maps.begin_ctb_filter(0, filter(true));
        maps.begin_ctb_filter(1, filter(true));
        Fixture {
            params,
            layout,
            maps,
        }
    }

    const fn filter(across_slices: bool) -> CtbFilter {
        CtbFilter {
            beta_offset_div2: 0,
            tc_offset_div2: 0,
            across_slices,
        }
    }

    fn luma_only(kind: SaoKind, offsets: [i16; 5]) -> SaoCtb {
        let mut values = SaoCtb::default();
        values.planes[0] = SaoPlane { kind, offsets };
        values
    }

    /// Luma holds 100, with a dip of 90 at `dip`.
    fn dipped(dip: (usize, usize)) -> (Picture<u8>, Picture<u8>) {
        let mut picture = Picture::<u8>::new(32, 16).unwrap();
        for y in 0..16 {
            picture.luma_mut().row_mut(y).unwrap().fill(100);
        }
        picture.luma_mut().row_mut(dip.1).unwrap()[dip.0] = 90;
        let scratch = Picture::<u8>::new(32, 16).unwrap();
        (picture, scratch)
    }

    fn run(fixture: &Fixture, picture: &mut Picture<u8>, scratch: &mut Picture<u8>) {
        let context = SaoContext {
            params: &fixture.params,
            layout: &fixture.layout,
            maps: &fixture.maps,
        };
        sao_picture(picture, scratch, &context);
    }

    const VALLEY: [i16; 5] = [0, 5, 0, 0, 0];

    /// Class 0 at a valley inside the CTB: edgeIdx 1 adds its offset.
    #[test]
    fn an_edge_offset_changes_a_valley() {
        let mut fixture = fixture(&[2]);
        fixture.maps.set_sao(0, luma_only(SaoKind::Edge(0), VALLEY));
        let (mut picture, mut scratch) = dipped((5, 5));
        run(&fixture, &mut picture, &mut scratch);
        assert_eq!(picture.luma().row(5).unwrap()[5], 95);
        assert_eq!(picture.luma().row(5).unwrap()[6], 100);
    }

    /// The first column has no left neighbour, and the top row no upper neighbour:
    /// edgeIdx 0 at the picture boundary (8.7.3.2).
    #[test]
    fn the_picture_boundary_gets_no_offset() {
        let mut fixture = fixture(&[2]);
        fixture.maps.set_sao(0, luma_only(SaoKind::Edge(0), VALLEY));
        let (mut picture, mut scratch) = dipped((0, 5));
        run(&fixture, &mut picture, &mut scratch);
        assert_eq!(picture.luma().row(5).unwrap()[0], 90);
        fixture.maps.set_sao(0, luma_only(SaoKind::Edge(1), VALLEY));
        let (mut picture, mut scratch) = dipped((5, 0));
        run(&fixture, &mut picture, &mut scratch);
        assert_eq!(picture.luma().row(0).unwrap()[5], 90);
    }

    /// A dip at the CTB's right column reads the next CTB's first column.
    #[test]
    fn a_neighbour_in_the_next_ctb_counts() {
        let mut fixture = fixture(&[2]);
        fixture.maps.set_sao(0, luma_only(SaoKind::Edge(0), VALLEY));
        let (mut picture, mut scratch) = dipped((15, 3));
        run(&fixture, &mut picture, &mut scratch);
        assert_eq!(picture.luma().row(3).unwrap()[15], 95);
    }

    /// The source stays the deblocked picture: a changed sample does not feed the
    /// next one. Two dips side by side are each one step from a valley.
    #[test]
    fn offsets_read_the_deblocked_copy() {
        let mut fixture = fixture(&[2]);
        fixture
            .maps
            .set_sao(0, luma_only(SaoKind::Edge(0), [0, 5, 3, 0, 0]));
        let (mut picture, mut scratch) = dipped((5, 5));
        picture.luma_mut().row_mut(5).unwrap()[6] = 90;
        run(&fixture, &mut picture, &mut scratch);
        // Each dip has one equal and one higher neighbour: edgeIdx 2.
        assert_eq!(picture.luma().row(5).unwrap()[5], 93);
        assert_eq!(picture.luma().row(5).unwrap()[6], 93);
    }

    /// A neighbour in a later slice follows that slice's flag, and a neighbour in an
    /// earlier slice follows the current slice's flag (8.7.3.2).
    #[test]
    fn slice_flags_follow_decoding_order() {
        let mut fixture = fixture(&[2]);
        fixture.maps.set_ctb_slice(1, 1);
        fixture.maps.begin_ctb_filter(1, filter(false));
        fixture.maps.set_sao(0, luma_only(SaoKind::Edge(0), VALLEY));
        let (mut picture, mut scratch) = dipped((15, 3));
        run(&fixture, &mut picture, &mut scratch);
        assert_eq!(picture.luma().row(3).unwrap()[15], 90);
        fixture.maps.begin_ctb_filter(0, filter(false));
        fixture.maps.begin_ctb_filter(1, filter(true));
        fixture.maps.set_sao(0, luma_only(SaoKind::Edge(0), VALLEY));
        let (mut picture, mut scratch) = dipped((15, 3));
        run(&fixture, &mut picture, &mut scratch);
        assert_eq!(picture.luma().row(3).unwrap()[15], 95);
        // From the later CTB, the earlier slice's neighbour follows the later flag.
        fixture.maps.begin_ctb_filter(0, filter(true));
        fixture.maps.begin_ctb_filter(1, filter(false));
        fixture.maps.set_sao(1, luma_only(SaoKind::Edge(0), VALLEY));
        let (mut picture, mut scratch) = dipped((16, 3));
        run(&fixture, &mut picture, &mut scratch);
        assert_eq!(picture.luma().row(3).unwrap()[16], 90);
    }

    #[test]
    fn a_tile_boundary_follows_the_pps_flag() {
        let mut fixture = fixture(&[1, 1]);
        fixture.params.loop_filter_across_tiles = false;
        fixture.maps.set_sao(0, luma_only(SaoKind::Edge(0), VALLEY));
        let (mut picture, mut scratch) = dipped((15, 3));
        run(&fixture, &mut picture, &mut scratch);
        assert_eq!(picture.luma().row(3).unwrap()[15], 90);
        fixture.params.loop_filter_across_tiles = true;
        let (mut picture, mut scratch) = dipped((15, 3));
        run(&fixture, &mut picture, &mut scratch);
        assert_eq!(picture.luma().row(3).unwrap()[15], 95);
    }

    /// Band offsets stop at the CTB's own samples.
    #[test]
    fn band_offsets_cover_one_ctb() {
        let mut fixture = fixture(&[2]);
        // 100 >> 3 = band 12.
        fixture
            .maps
            .set_sao(1, luma_only(SaoKind::Band(12), [0, 7, 0, 0, 0]));
        let (mut picture, mut scratch) = dipped((0, 0));
        run(&fixture, &mut picture, &mut scratch);
        assert_eq!(picture.luma().row(8).unwrap()[15], 100);
        assert_eq!(picture.luma().row(8).unwrap()[16], 107);
        assert_eq!(picture.luma().row(8).unwrap()[31], 107);
    }

    /// Chroma CTBs are half the size. The Cb values read only the Cb plane.
    #[test]
    fn chroma_uses_its_own_size_and_values() {
        let mut fixture = fixture(&[2]);
        let mut values = SaoCtb::default();
        values.planes[1] = SaoPlane {
            kind: SaoKind::Band(0),
            offsets: [0, 3, 0, 0, 0],
        };
        fixture.maps.set_sao(1, values);
        let mut picture = Picture::<u8>::new(32, 16).unwrap();
        let mut scratch = Picture::<u8>::new(32, 16).unwrap();
        run(&fixture, &mut picture, &mut scratch);
        assert_eq!(picture.cb().row(7).unwrap()[7], 0);
        assert_eq!(picture.cb().row(7).unwrap()[8], 3);
        assert_eq!(picture.cr().row(7).unwrap()[8], 0);
        assert_eq!(picture.luma().row(7).unwrap()[20], 0);
    }

    /// A bypass unit keeps its deblocked samples, and the unit beside it changes.
    #[test]
    fn a_no_filter_unit_keeps_its_samples() {
        let mut fixture = fixture(&[2]);
        fixture.params.transquant_bypass = true;
        fixture.maps.set_cu_kind(0, 0, 8, true, true);
        fixture
            .maps
            .set_sao(0, luma_only(SaoKind::Band(12), [0, 7, 0, 0, 0]));
        let (mut picture, mut scratch) = dipped((31, 15));
        run(&fixture, &mut picture, &mut scratch);
        assert_eq!(picture.luma().row(7).unwrap()[7], 100);
        assert_eq!(picture.luma().row(7).unwrap()[8], 107);
        assert_eq!(picture.luma().row(8).unwrap()[7], 107);
    }

    #[test]
    fn nothing_is_active_with_every_ctb_off() {
        let mut fixture = fixture(&[2]);
        let context = |fixture: &Fixture| -> bool {
            sao_is_active(&SaoContext {
                params: &fixture.params,
                layout: &fixture.layout,
                maps: &fixture.maps,
            })
        };
        assert!(!context(&fixture));
        fixture.maps.set_sao(1, luma_only(SaoKind::Edge(1), VALLEY));
        assert!(context(&fixture));
    }
}
