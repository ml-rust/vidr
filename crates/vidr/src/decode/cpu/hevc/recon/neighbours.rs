//! The neighbouring samples of one transform block (8.4.4.2.1).

use crate::decode::cpu::dsp::intra::refs::RefSamples;
use crate::decode::cpu::dsp::sample::Sample;
use crate::decode::cpu::hevc::layout::PicLayout;
use crate::decode::cpu::hevc::params::CodingParams;
use crate::decode::cpu::hevc::picture::Plane;
use crate::decode::cpu::hevc::syntax::maps::ParseMaps;

/// Luma samples that share one availability result. No block edge falls inside.
const UNIT_LUMA: usize = 4;

/// The block whose neighbours are read.
#[derive(Clone, Copy, Debug)]
pub(super) struct BlockAt {
    /// `xTbCmp`.
    pub(super) x: usize,
    /// `yTbCmp`.
    pub(super) y: usize,
    /// `Log2( nTbS )`.
    pub(super) log2_size: u32,
    /// `SubWidthC` and `SubHeightC` of this plane: 1 for luma, 2 for chroma.
    pub(super) sub: usize,
}

/// The maps that availability reads.
#[derive(Clone, Copy, Debug)]
pub(super) struct Availability<'a> {
    pub(super) params: &'a CodingParams,
    pub(super) layout: &'a PicLayout,
    pub(super) maps: &'a ParseMaps,
}

impl Availability<'_> {
    /// "Available for intra prediction" for the sample at plane location
    /// `( x_nb, y_nb )`, seen from the block at luma location `current`.
    fn usable(&self, current: (u32, u32), sub: usize, x_nb: i64, y_nb: i64) -> bool {
        let (x_luma, y_luma) = (x_nb * sub as i64, y_nb * sub as i64);
        let (Ok(x), Ok(y)) = (i32::try_from(x_luma), i32::try_from(y_luma)) else {
            return false;
        };
        self.maps.available(self.layout, current, x, y)
            && (!self.params.constrained_intra_pred || self.maps.cu_intra(x as u32, y as u32))
    }
}

/// Fills `refs` from `plane`, then runs the substitution of 8.4.4.2.2.
pub(super) fn read_neighbours<S: Sample>(
    plane: &Plane<S>,
    context: &Availability<'_>,
    block: BlockAt,
    refs: &mut RefSamples,
) {
    refs.begin(block.log2_size);
    let size = refs.size();
    let unit = (UNIT_LUMA / block.sub).max(1);
    let current = ((block.x * block.sub) as u32, (block.y * block.sub) as u32);
    let (x0, y0) = (block.x as i64, block.y as i64);
    for offset in (0..2 * size).step_by(unit) {
        if !context.usable(current, block.sub, x0 - 1, y0 + offset as i64) {
            continue;
        }
        for line in offset..(offset + unit).min(2 * size) {
            let sample = plane
                .row(block.y + line)
                .and_then(|row| row.get(block.x.wrapping_sub(1)));
            if let Some(sample) = sample {
                refs.set_left(line, sample.to_i32() as u16);
            }
        }
    }
    if context.usable(current, block.sub, x0 - 1, y0 - 1) {
        let sample = plane
            .row(block.y.wrapping_sub(1))
            .and_then(|row| row.get(block.x.wrapping_sub(1)));
        if let Some(sample) = sample {
            refs.set_corner(sample.to_i32() as u16);
        }
    }
    let row_above = plane.row(block.y.wrapping_sub(1));
    for offset in (0..2 * size).step_by(unit) {
        if !context.usable(current, block.sub, x0 + offset as i64, y0 - 1) {
            continue;
        }
        for column in offset..(offset + unit).min(2 * size) {
            if let Some(sample) = row_above.and_then(|row| row.get(block.x + column)) {
                refs.set_top(column, sample.to_i32() as u16);
            }
        }
    }
    refs.substitute(context.params.bit_depth);
}
