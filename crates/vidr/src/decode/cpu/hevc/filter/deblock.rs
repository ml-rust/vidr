//! The deblocking filter process on a whole picture (8.7.2). Sections of one edge
//! direction touch no common samples, so their order does not matter. Vertical edges
//! run first, because horizontal edges read their result (8.7.2.1).

use crate::decode::cpu::dsp::deblock::{
    EdgeFilter, EdgeSection, beta, filter_chroma_section, filter_luma_section, tc,
};
use crate::decode::cpu::dsp::sample::Sample;
use crate::decode::cpu::hevc::layout::PicLayout;
use crate::decode::cpu::hevc::params::CodingParams;
use crate::decode::cpu::hevc::picture::{Picture, Plane};
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

    /// (8-348) to (8-351), with the side flags of 8.7.2.5.7.
    fn luma_filter(&self, sides: Sides, strength: u8) -> EdgeFilter {
        let bit_depth = self.params.bit_depth;
        let qp_l = self.average_qp(sides);
        let (beta_offset, tc_offset) = self.slice_offsets(sides);
        EdgeFilter {
            beta: beta(qp_l, beta_offset, bit_depth),
            tc: tc(qp_l + 2 * (i32::from(strength) - 1), tc_offset, bit_depth),
            modify_p: !self.maps.no_filter(sides.p.0, sides.p.1),
            modify_q: !self.maps.no_filter(sides.q.0, sides.q.1),
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

/// Runs the whole process of 8.7.2 on `picture`.
pub(crate) fn deblock_picture<S: Sample>(picture: &mut Picture<S>, context: &DeblockContext<'_>) {
    if !context.edges.has_edges() {
        return;
    }
    vertical_edges(picture, context);
    horizontal_edges(picture, context);
}

fn vertical_edges<S: Sample>(picture: &mut Picture<S>, context: &DeblockContext<'_>) {
    let (width, height) = (context.params.width, context.params.height);
    for y in (0..height).step_by(SECTION as usize) {
        for x in (GRID..width).step_by(GRID as usize) {
            let strength = context.edges.vertical_at(x, y);
            if strength > 0 {
                let sides = Sides {
                    q: (x, y),
                    p: (x - 1, y),
                };
                let filter = context.luma_filter(sides, strength);
                let plane = picture.luma_mut();
                let section = vertical_section(plane, x as usize, y as usize);
                filter_luma_section(plane.samples_mut(), section, &filter);
            }
        }
    }
    for y in (0..height).step_by(CHROMA_SECTION as usize) {
        for x in (CHROMA_GRID..width).step_by(CHROMA_GRID as usize) {
            if context.edges.vertical_at(x, y) == BS_CHROMA {
                let sides = Sides {
                    q: (x, y),
                    p: (x - 1, y),
                };
                chroma_section(picture, context, sides, |plane| {
                    vertical_section(plane, x as usize / 2, y as usize / 2)
                });
            }
        }
    }
}

fn horizontal_edges<S: Sample>(picture: &mut Picture<S>, context: &DeblockContext<'_>) {
    let (width, height) = (context.params.width, context.params.height);
    for y in (GRID..height).step_by(GRID as usize) {
        for x in (0..width).step_by(SECTION as usize) {
            let strength = context.edges.horizontal_at(x, y);
            if strength > 0 {
                let sides = Sides {
                    q: (x, y),
                    p: (x, y - 1),
                };
                let filter = context.luma_filter(sides, strength);
                let plane = picture.luma_mut();
                let section = horizontal_section(plane, x as usize, y as usize);
                filter_luma_section(plane.samples_mut(), section, &filter);
            }
        }
    }
    for y in (CHROMA_GRID..height).step_by(CHROMA_GRID as usize) {
        for x in (0..width).step_by(CHROMA_SECTION as usize) {
            if context.edges.horizontal_at(x, y) == BS_CHROMA {
                let sides = Sides {
                    q: (x, y),
                    p: (x, y - 1),
                };
                chroma_section(picture, context, sides, |plane| {
                    horizontal_section(plane, x as usize / 2, y as usize / 2)
                });
            }
        }
    }
}

/// Filters Cb then Cr at one section. `cQpPicOffset` differs per plane (8.7.2.5.1).
fn chroma_section<S: Sample>(
    picture: &mut Picture<S>,
    context: &DeblockContext<'_>,
    sides: Sides,
    locate: impl Fn(&Plane<S>) -> EdgeSection,
) {
    let offsets = [
        context.params.pps_cb_qp_offset,
        context.params.pps_cr_qp_offset,
    ];
    for (index, offset) in offsets.into_iter().enumerate() {
        let filter = context.chroma_filter(sides, offset);
        let plane = picture.plane_mut(index + 1);
        let section = locate(plane);
        filter_chroma_section(plane.samples_mut(), section, &filter);
    }
}

const fn vertical_section<S: Sample>(plane: &Plane<S>, x: usize, y: usize) -> EdgeSection {
    EdgeSection {
        q0: y * plane.stride() + x,
        across: 1,
        along: plane.stride(),
    }
}

const fn horizontal_section<S: Sample>(plane: &Plane<S>, x: usize, y: usize) -> EdgeSection {
    EdgeSection {
        q0: y * plane.stride() + x,
        across: plane.stride(),
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
