//! The deblocking filter process on a whole picture (8.7.2). Sections of one edge
//! direction touch no common samples, so their order does not matter. Vertical edges
//! run first, because horizontal edges read their result (8.7.2.1).
//!
//! The same passes run over bands of rows, one writer per band, because each section's
//! samples lie in one band. A vertical section stays in its four rows, so each CTB row is
//! one band. A horizontal luma section at row `y` reads rows `y - 4` to `y + 3`, and those
//! ranges of the 8x8 grid never overlap. So a band of that pass starts 4 luma rows above its
//! CTB row, and 2 chroma rows above in chroma.

use std::ops::Range;

use fearless_simd::Simd;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::decode::cpu::dsp::deblock::{
    EdgeFilter, EdgeSection, beta, filter_chroma_section, luma_section, tc,
};
use crate::decode::cpu::dsp::sample::Sample;
use crate::decode::cpu::dsp::simd::{Cpu, with_simd};
use crate::decode::cpu::hevc::layout::PicLayout;
use crate::decode::cpu::hevc::params::CodingParams;
use crate::decode::cpu::hevc::picture::{Picture, PlaneRows};
use crate::decode::cpu::hevc::qp::qp_c_from_index;
use crate::decode::cpu::hevc::syntax::maps::ParseMaps;

use super::edges::EdgeMap;

/// Luma samples between edges on the grid.
const GRID: u32 = 8;
/// Luma samples in one edge section.
const SECTION: u32 = 4;
/// Luma samples in one chroma edge section: four chroma samples.
const CHROMA_SECTION: u32 = 8;
/// Luma samples between chroma edges: the 8x8 grid of the chroma plane.
const CHROMA_GRID: u32 = 16;
/// `bS` that filters chroma (8.7.2.5.1, 8.7.2.5.2).
const BS_CHROMA: u8 = 2;
/// Luma rows that a band of horizontal edges starts above its CTB row.
const HORIZONTAL_BAND_LEAD: usize = 4;
/// The smallest CTB side, as a shift, for which a band holds whole sections and grid rows.
const MIN_BAND_CTB_LOG2: u32 = 4;

/// What the filter reads besides the samples.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DeblockContext<'a> {
    /// Any slice of the picture: the fields read here hold for the whole picture.
    pub(crate) params: &'a CodingParams,
    pub(crate) layout: &'a PicLayout,
    pub(crate) maps: &'a ParseMaps,
    pub(crate) edges: &'a EdgeMap,
}

/// One section's two sides, as luma locations of `q0` and `p0`.
#[derive(Clone, Copy, Debug)]
struct Sides {
    q: (u32, u32),
    p: (u32, u32),
}

impl DeblockContext<'_> {
    /// `qPL` (8-347), or the first term of (8-382).
    fn average_qp(&self, sides: Sides) -> i32 {
        let qp_q = self.maps.qp_y(sides.q.0, sides.q.1);
        let qp_p = self.maps.qp_y(sides.p.0, sides.p.1);
        (qp_q + qp_p + 1) >> 1
    }

    /// `slice_beta_offset_div2` and `slice_tc_offset_div2` of the slice holding `q0`.
    fn slice_offsets(&self, sides: Sides) -> (i32, i32) {
        let filter = self
            .maps
            .ctb_filter(self.layout.ctb_rs_at(sides.q.0, sides.q.1));
        (filter.beta_offset_div2, filter.tc_offset_div2)
    }

    /// (8-348) to (8-351), with the side flags of 8.7.2.5.7. `protected` is false when no
    /// block of the picture holds the side flag.
    fn luma_filter(&self, sides: Sides, strength: u8, protected: bool) -> EdgeFilter {
        let bit_depth = self.params.bit_depth;
        let qp_l = self.average_qp(sides);
        let (beta_offset, tc_offset) = self.slice_offsets(sides);
        EdgeFilter {
            beta: beta(qp_l, beta_offset, bit_depth),
            tc: tc(qp_l + 2 * (i32::from(strength) - 1), tc_offset, bit_depth),
            modify_p: !protected || !self.maps.no_filter(sides.p.0, sides.p.1),
            modify_q: !protected || !self.maps.no_filter(sides.q.0, sides.q.1),
            max_value: (1 << bit_depth) - 1,
        }
    }

    /// (8-382) to (8-384), with the side flags of 8.7.2.5.8.
    fn chroma_filter(&self, sides: Sides, pic_offset: i32) -> EdgeFilter {
        let bit_depth = self.params.bit_depth;
        let qp_c = qp_c_from_index(self.average_qp(sides) + pic_offset);
        let (_, tc_offset) = self.slice_offsets(sides);
        EdgeFilter {
            beta: 0,
            tc: tc(qp_c + 2, tc_offset, bit_depth),
            modify_p: !self.maps.no_filter(sides.p.0, sides.p.1),
            modify_q: !self.maps.no_filter(sides.q.0, sides.q.1),
            max_value: (1 << bit_depth) - 1,
        }
    }
}

/// The rows of a picture one writer filters, and the luma rows of the sections it takes.
struct EdgeBand<'a, S> {
    luma: PlaneRows<'a, S>,
    cb: PlaneRows<'a, S>,
    cr: PlaneRows<'a, S>,
    /// The luma rows of `q0` of the sections of this band.
    rows: Range<u32>,
}

/// Runs the whole process of 8.7.2 on `picture`.
pub(crate) fn deblock_picture<S: Sample>(picture: &mut Picture<S>, context: &DeblockContext<'_>) {
    if !context.edges.has_edges() {
        return;
    }
    deblock_on(Cpu::detected(), picture, context);
}

/// [`deblock_picture`] compiled for `cpu`.
fn deblock_on<S: Sample>(cpu: Cpu, picture: &mut Picture<S>, context: &DeblockContext<'_>) {
    let protected = context.maps.any_no_filter();
    let (luma, cb, cr) = picture.planes_mut();
    let mut band = EdgeBand {
        luma: luma.rows_mut(),
        cb: cb.rows_mut(),
        cr: cr.rows_mut(),
        rows: 0..context.params.height,
    };
    with_simd!(cpu, simd => {
        vertical_edges(simd, &mut band, context, protected);
        horizontal_edges(simd, &mut band, context, protected);
    });
}

/// [`deblock_picture`] with each pass over bands of rows on the current rayon pool. The
/// output equals it, because 8.7.2 orders only the two passes.
pub(crate) fn deblock_picture_in_rows<S: Sample>(
    picture: &mut Picture<S>,
    context: &DeblockContext<'_>,
) {
    if !context.edges.has_edges() {
        return;
    }
    if context.layout.ctb_log2() < MIN_BAND_CTB_LOG2 {
        deblock_on(Cpu::detected(), picture, context);
        return;
    }
    let protected = context.maps.any_no_filter();
    edge_bands(picture, context, 0)
        .into_par_iter()
        .for_each(|mut band| {
            with_simd!(Cpu::detected(), simd => {
                vertical_edges(simd, &mut band, context, protected);
            });
        });
    edge_bands(picture, context, HORIZONTAL_BAND_LEAD)
        .into_par_iter()
        .for_each(|mut band| {
            with_simd!(Cpu::detected(), simd => {
                horizontal_edges(simd, &mut band, context, protected);
            });
        });
}

/// One band per CTB row. Each band but the first starts `lead` luma rows above its CTB row,
/// and half that in chroma.
fn edge_bands<'a, S: Sample>(
    picture: &'a mut Picture<S>,
    context: &DeblockContext<'_>,
    lead: usize,
) -> Vec<EdgeBand<'a, S>> {
    let ctb = 1usize << context.layout.ctb_log2();
    let height = context.params.height as usize;
    let count = height.div_ceil(ctb);
    let ends = |size: usize, lead: usize| {
        (1..=count).map(move |row| {
            if row == count {
                usize::MAX
            } else {
                row * size - lead
            }
        })
    };
    let (luma, cb, cr) = picture.planes_mut();
    let luma = luma.split_rows(ends(ctb, lead));
    let cb = cb.split_rows(ends(ctb / 2, lead / 2));
    let cr = cr.split_rows(ends(ctb / 2, lead / 2));
    luma.into_iter()
        .zip(cb)
        .zip(cr)
        .enumerate()
        .map(|(row, ((luma, cb), cr))| EdgeBand {
            luma,
            cb,
            cr,
            rows: (row * ctb) as u32..(((row + 1) * ctb).min(height)) as u32,
        })
        .collect()
}

#[inline(always)]
fn vertical_edges<V: Simd, S: Sample>(
    simd: V,
    band: &mut EdgeBand<'_, S>,
    context: &DeblockContext<'_>,
    protected: bool,
) {
    let width = context.params.width;
    let (first, end) = (band.rows.start, band.rows.end);
    for (y, row) in (first..end).step_by(SECTION as usize).zip(
        context
            .edges
            .vertical_rows()
            .skip((first / SECTION) as usize),
    ) {
        // Column 0 is the picture edge, and no edge lies there.
        for (x, &strength) in (0..width)
            .step_by(GRID as usize)
            .zip(row)
            .skip(1)
            .filter(|(_, strength)| **strength > 0)
        {
            let sides = Sides {
                q: (x, y),
                p: (x - 1, y),
            };
            let filter = context.luma_filter(sides, strength, protected);
            let section = vertical_section(&band.luma, x as usize, y as usize);
            luma_section(simd, band.luma.samples_mut(), section, &filter);
        }
    }
    for y in (first..end).step_by(CHROMA_SECTION as usize) {
        for x in (CHROMA_GRID..width).step_by(CHROMA_GRID as usize) {
            if context.edges.vertical_at(x, y) == BS_CHROMA {
                let sides = Sides {
                    q: (x, y),
                    p: (x - 1, y),
                };
                let section = vertical_section(&band.cb, x as usize / 2, y as usize / 2);
                chroma_section(band, context, sides, section);
            }
        }
    }
}

#[inline(always)]
fn horizontal_edges<V: Simd, S: Sample>(
    simd: V,
    band: &mut EdgeBand<'_, S>,
    context: &DeblockContext<'_>,
    protected: bool,
) {
    let width = context.params.width;
    let end = band.rows.end;
    // Row 0 is the picture edge, and no edge lies there.
    let first = band.rows.start.max(GRID);
    for (y, row) in (first..end).step_by(GRID as usize).zip(
        context
            .edges
            .horizontal_rows()
            .skip((first / GRID) as usize),
    ) {
        for (x, &strength) in (0..width)
            .step_by(SECTION as usize)
            .zip(row)
            .filter(|(_, strength)| **strength > 0)
        {
            let sides = Sides {
                q: (x, y),
                p: (x, y - 1),
            };
            let filter = context.luma_filter(sides, strength, protected);
            let section = horizontal_section(&band.luma, x as usize, y as usize);
            luma_section(simd, band.luma.samples_mut(), section, &filter);
        }
    }
    let first_chroma = band.rows.start.max(CHROMA_GRID);
    for y in (first_chroma..end).step_by(CHROMA_GRID as usize) {
        for x in (0..width).step_by(CHROMA_SECTION as usize) {
            if context.edges.horizontal_at(x, y) == BS_CHROMA {
                let sides = Sides {
                    q: (x, y),
                    p: (x, y - 1),
                };
                let section = horizontal_section(&band.cb, x as usize / 2, y as usize / 2);
                chroma_section(band, context, sides, section);
            }
        }
    }
}

/// Filters Cb then Cr at one section. Both planes have one layout, so `section` serves
/// each. `cQpPicOffset` differs per plane (8.7.2.5.1).
fn chroma_section<S: Sample>(
    band: &mut EdgeBand<'_, S>,
    context: &DeblockContext<'_>,
    sides: Sides,
    section: EdgeSection,
) {
    let filter = context.chroma_filter(sides, context.params.pps_cb_qp_offset);
    filter_chroma_section(band.cb.samples_mut(), section, &filter);
    let filter = context.chroma_filter(sides, context.params.pps_cr_qp_offset);
    filter_chroma_section(band.cr.samples_mut(), section, &filter);
}

const fn vertical_section<S>(rows: &PlaneRows<'_, S>, x: usize, y: usize) -> EdgeSection {
    EdgeSection {
        q0: rows.index_of(x, y),
        across: 1,
        along: rows.stride(),
    }
}

const fn horizontal_section<S>(rows: &PlaneRows<'_, S>, x: usize, y: usize) -> EdgeSection {
    EdgeSection {
        q0: rows.index_of(x, y),
        across: rows.stride(),
        along: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::super::edges::EdgeRun;
    use super::*;
    use crate::decode::cpu::hevc::layout::LayoutSizes;

    struct Fixture {
        params: CodingParams,
        layout: PicLayout,
        maps: ParseMaps,
        edges: EdgeMap,
    }

    /// 32x32 luma with 16x16 CTBs in one slice, `QpY` 37 everywhere.
    fn fixture() -> Fixture {
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
            column_widths: &[2],
            row_heights: &[2],
        })
        .unwrap();
        let mut maps = ParseMaps::new(&params, &layout).unwrap();
        maps.set_qp_y(0, 0, 16, 37);
        maps.set_qp_y(16, 0, 16, 37);
        maps.set_qp_y(0, 16, 16, 37);
        maps.set_qp_y(16, 16, 16, 37);
        let edges = EdgeMap::new(32, 32).unwrap();
        Fixture {
            params,
            layout,
            maps,
            edges,
        }
    }

    /// Columns left of `x` hold `left`, and the others hold `right`. Chroma columns
    /// split at `x / 2`.
    fn split_columns(x: usize, left: u8, right: u8) -> Picture<u8> {
        let mut picture = Picture::<u8>::new(32, 32).unwrap();
        for c_idx in 0..3 {
            let split = if c_idx == 0 { x } else { x / 2 };
            let plane = picture.plane_mut(c_idx);
            for y in 0..plane.height() {
                for (column, sample) in plane.row_mut(y).unwrap().iter_mut().enumerate() {
                    *sample = if column < split { left } else { right };
                }
            }
        }
        picture
    }

    fn run(fixture: &Fixture, picture: &mut Picture<u8>) {
        let context = DeblockContext {
            params: &fixture.params,
            layout: &fixture.layout,
            maps: &fixture.maps,
            edges: &fixture.edges,
        };
        deblock_picture(picture, &context);
    }

    /// `QpY` 37, bS 2: β′ 36, tC′ at Q 39 is 5. A step of 4 passes dSam, and
    /// the strong filter runs. Chroma: qPi 37 gives QpC 34, Q 36, tC′ 4.
    /// Chroma Δ = ( 16 + 100 - 104 + 4 ) >> 3 = 2.
    #[test]
    fn a_marked_edge_filters_luma_and_chroma() {
        let mut fixture = fixture();
        fixture.edges.mark_vertical(EdgeRun {
            x: 16,
            y: 0,
            length: 16,
            strength: 2,
        });
        let mut picture = split_columns(16, 100, 104);
        run(&fixture, &mut picture);
        let row = picture.luma().row(0).unwrap();
        assert_eq!(&row[12..20], [100, 101, 101, 102, 103, 103, 104, 104]);
        // The section below the marked edge stays.
        assert_eq!(&picture.luma().row(16).unwrap()[15..17], [100, 104]);
        let cb = picture.cb().row(0).unwrap();
        assert_eq!(&cb[6..10], [100, 102, 102, 104]);
        assert_eq!(&picture.cr().row(7).unwrap()[6..10], [100, 102, 102, 104]);
        assert_eq!(&picture.cb().row(8).unwrap()[7..9], [100, 104]);
    }

    /// Chroma needs an edge on its own 8x8 grid: luma x 8 is chroma x 4.
    #[test]
    fn a_luma_only_edge_leaves_chroma() {
        let mut fixture = fixture();
        fixture.edges.mark_vertical(EdgeRun {
            x: 8,
            y: 0,
            length: 16,
            strength: 2,
        });
        let mut picture = split_columns(8, 100, 104);
        run(&fixture, &mut picture);
        assert_eq!(picture.luma().row(0).unwrap()[7], 102);
        assert_eq!(&picture.cb().row(0).unwrap()[3..5], [100, 104]);
    }

    /// `pcm_loop_filter_disabled_flag` or bypass on one side keeps that side.
    #[test]
    fn a_no_filter_side_keeps_its_samples() {
        let mut fixture = fixture();
        fixture.edges.mark_vertical(EdgeRun {
            x: 16,
            y: 0,
            length: 16,
            strength: 2,
        });
        fixture.maps.set_cu_kind(16, 0, 16, true, true);
        let mut picture = split_columns(16, 100, 104);
        run(&fixture, &mut picture);
        let row = picture.luma().row(0).unwrap();
        assert_eq!(&row[12..20], [100, 101, 101, 102, 104, 104, 104, 104]);
        assert_eq!(&picture.cb().row(0).unwrap()[6..10], [100, 102, 104, 104]);
    }

    /// The offset of the slice holding `q0` selects tC: offset -6 lowers Q by 12.
    /// `QpY` 37 at bS 2 gives Q 27, tC′ 2. A step of 6 fails dSam at
    /// ( 5 * 2 + 1 ) >> 1 = 5, so the weak filter clips Δ to 2.
    #[test]
    fn slice_offsets_change_the_filter() {
        let mut fixture = fixture();
        fixture.edges.mark_vertical(EdgeRun {
            x: 16,
            y: 0,
            length: 16,
            strength: 2,
        });
        fixture.maps.begin_ctb_filter(
            1,
            crate::decode::cpu::hevc::syntax::maps::CtbFilter {
                beta_offset_div2: 0,
                tc_offset_div2: -6,
                across_slices: true,
            },
        );
        let mut picture = split_columns(16, 100, 106);
        run(&fixture, &mut picture);
        let row = picture.luma().row(0).unwrap();
        // Δ = ( 9 * 6 - 3 * 6 + 8 ) >> 4 = 2. dp is 0, so p1 and q1 move by
        // ( 0 + 2 ) >> 1 = 1, within tC >> 1.
        assert_eq!(&row[13..19], [100, 101, 102, 104, 105, 106]);
    }

    /// Horizontal edges read the output of vertical edges.
    #[test]
    fn horizontal_edges_run_second() {
        let mut fixture = fixture();
        fixture.edges.mark_vertical(EdgeRun {
            x: 16,
            y: 0,
            length: 32,
            strength: 2,
        });
        fixture.edges.mark_horizontal(EdgeRun {
            x: 0,
            y: 16,
            length: 32,
            strength: 2,
        });
        let mut picture = Picture::<u8>::new(32, 32).unwrap();
        for y in 0..32 {
            let row = picture.luma_mut().row_mut(y).unwrap();
            for (x, sample) in row.iter_mut().enumerate() {
                *sample = 100 + u8::from(x >= 16) * 4 + u8::from(y >= 16) * 4;
            }
        }
        run(&fixture, &mut picture);
        // At x = 16 the vertical filter leaves 103 above the horizontal edge and 107
        // below it. The horizontal strong filter reads those values.
        // p0: ( 103 + 206 + 206 + 214 + 107 + 4 ) >> 3 = 105.
        assert_eq!(picture.luma().row(15).unwrap()[16], 105);
    }
}
