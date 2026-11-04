//! Fractional sample interpolation (8.5.3.3.3).
//!
//! Each call copies its reference area into a dense scratch buffer through clamped
//! coordinates, (8-222), (8-223), (8-239), and (8-240). The filters then run on one
//! path for every location and sample type.

use crate::decode::cpu::dsp::sample::Sample;

/// The largest prediction block side.
const MAX_BLOCK: usize = 64;
/// The most taps of a filter.
const MAX_TAPS: usize = 8;
/// Samples per row and rows in the reference scratch.
const FETCH_SIZE: usize = MAX_BLOCK + MAX_TAPS - 1;
/// `shift2`: 6 at every bit depth.
const SECOND_STAGE_SHIFT: u32 = 6;

/// The 8-tap luma filters by fraction, from `-3` to `+4` (8-224) to (8-229).
const LUMA_FILTERS: [[i32; 8]; 4] = [
    [0, 0, 0, 64, 0, 0, 0, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 40, 40, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
];

/// The 4-tap chroma filters by fraction, from `-1` to `+2` (8-241) to (8-247).
const CHROMA_FILTERS: [[i32; 4]; 8] = [
    [0, 64, 0, 0],
    [-2, 58, 10, -2],
    [-4, 54, 16, -2],
    [-6, 46, 28, -4],
    [-4, 36, 36, -4],
    [-4, 28, 46, -6],
    [-2, 16, 54, -4],
    [-2, 10, 58, -2],
];

/// A reference plane as the kernels read it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RefPlane<'a, S> {
    /// Every sample, stride gaps included. Row `y` starts at `y * stride`.
    pub(crate) samples: &'a [S],
    /// Samples per row.
    pub(crate) width: usize,
    /// Rows.
    pub(crate) height: usize,
    /// Samples from one row start to the next.
    pub(crate) stride: usize,
}

impl<S> RefPlane<'_, S> {
    /// Row `y`, or `None` below the last row.
    fn row(&self, y: usize) -> Option<&[S]> {
        if y >= self.height {
            return None;
        }
        let start = y * self.stride;
        self.samples.get(start..start + self.width)
    }
}

/// `predSamplesLX` of one block: 14-bit intermediate values.
#[derive(Clone, Debug)]
pub(crate) struct PredSamples {
    samples: [i16; MAX_BLOCK * MAX_BLOCK],
}

impl PredSamples {
    /// An array of zeros.
    pub(crate) const fn new() -> Self {
        Self {
            samples: [0; MAX_BLOCK * MAX_BLOCK],
        }
    }

    /// `width` values of row `y`.
    #[inline]
    pub(crate) fn row(&self, y: usize, width: usize) -> &[i16] {
        let start = y * MAX_BLOCK;
        self.samples.get(start..start + width).unwrap_or_default()
    }

    fn row_mut(&mut self, y: usize, width: usize) -> &mut [i16] {
        let start = y * MAX_BLOCK;
        self.samples
            .get_mut(start..start + width)
            .unwrap_or_default()
    }
}

/// The working memory of the interpolation process.
#[derive(Clone, Debug)]
pub(crate) struct InterpScratch {
    reference: [u16; FETCH_SIZE * FETCH_SIZE],
    /// The first-stage values of a block with two fractions: `a`, `b`, `c`, `aY`.
    first_stage: [i16; MAX_BLOCK * FETCH_SIZE],
}

impl InterpScratch {
    /// Working memory for any block size.
    pub(crate) const fn new() -> Self {
        Self {
            reference: [0; FETCH_SIZE * FETCH_SIZE],
            first_stage: [0; MAX_BLOCK * FETCH_SIZE],
        }
    }
}

/// A block in a reference plane, at full-sample units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RefBlock {
    /// `xInt` of the first sample. It can lie outside the plane.
    pub(crate) x: i32,
    /// `yInt` of the first sample. It can lie outside the plane.
    pub(crate) y: i32,
    /// Width in samples, at most 64.
    pub(crate) width: usize,
    /// Height in samples, at most 64.
    pub(crate) height: usize,
}

/// `predSampleLXL` of a block (8.5.3.3.3.2). `frac` holds `xFracL` and `yFracL`, each
/// taken modulo 4.
pub(crate) fn interpolate_luma<S: Sample>(
    plane: &RefPlane<'_, S>,
    block: RefBlock,
    frac: (usize, usize),
    bit_depth: u32,
    scratch: &mut InterpScratch,
    out: &mut PredSamples,
) {
    let filters = (LUMA_FILTERS[frac.0 & 3], LUMA_FILTERS[frac.1 & 3]);
    interpolate(
        plane,
        block,
        frac.0 & 3 != 0,
        frac.1 & 3 != 0,
        &filters,
        bit_depth,
        scratch,
        out,
    );
}

/// `predSampleLXC` of a block (8.5.3.3.3.3). `frac` holds `xFracC` and `yFracC`, each
/// taken modulo 8.
pub(crate) fn interpolate_chroma<S: Sample>(
    plane: &RefPlane<'_, S>,
    block: RefBlock,
    frac: (usize, usize),
    bit_depth: u32,
    scratch: &mut InterpScratch,
    out: &mut PredSamples,
) {
    let filters = (CHROMA_FILTERS[frac.0 & 7], CHROMA_FILTERS[frac.1 & 7]);
    interpolate(
        plane,
        block,
        frac.0 & 7 != 0,
        frac.1 & 7 != 0,
        &filters,
        bit_depth,
        scratch,
        out,
    );
}

/// The stages of Table 8-8 and Table 8-9. `TAPS` is even, and position `TAPS / 2 - 1`
/// of a filter is the sample at the integer location.
#[allow(
    clippy::too_many_arguments,
    reason = "the plane, block, and kernel state stay plain arguments for the two callers"
)]
fn interpolate<S: Sample, const TAPS: usize>(
    plane: &RefPlane<'_, S>,
    block: RefBlock,
    x_fraction: bool,
    y_fraction: bool,
    filters: &([i32; TAPS], [i32; TAPS]),
    bit_depth: u32,
    scratch: &mut InterpScratch,
    out: &mut PredSamples,
) {
    let width = block.width.min(MAX_BLOCK);
    let height = block.height.min(MAX_BLOCK);
    let before = TAPS as i32 / 2 - 1;
    let shift1 = bit_depth.saturating_sub(8).min(4);
    let shift3 = 14u32.saturating_sub(bit_depth).max(2);
    let fetch = RefBlock {
        x: if x_fraction {
            block.x.saturating_sub(before)
        } else {
            block.x
        },
        y: if y_fraction {
            block.y.saturating_sub(before)
        } else {
            block.y
        },
        width: if x_fraction { width + TAPS - 1 } else { width },
        height: if y_fraction {
            height + TAPS - 1
        } else {
            height
        },
    };
    fetch_reference(plane, fetch, &mut scratch.reference);
    let InterpScratch {
        reference,
        first_stage,
    } = scratch;
    let reference = reference.as_slice();
    match (x_fraction, y_fraction) {
        (false, false) => {
            for y in 0..height {
                let source = fetched_row(reference, y, width);
                for (target, &sample) in out.row_mut(y, width).iter_mut().zip(source) {
                    *target = (i32::from(sample) << shift3) as i16;
                }
            }
        }
        (true, false) => {
            for y in 0..height {
                let source = fetched_row(reference, y, width + TAPS - 1);
                horizontal(source, &filters.0, shift1, out.row_mut(y, width));
            }
        }
        (false, true) => {
            let rows = |index: usize| fetched_row(reference, index, width);
            vertical(rows, &filters.1, shift1, width, height, out);
        }
        (true, true) => {
            for y in 0..height + TAPS - 1 {
                let source = fetched_row(reference, y, width + TAPS - 1);
                let start = y * MAX_BLOCK;
                let target = first_stage
                    .get_mut(start..start + width)
                    .unwrap_or_default();
                horizontal(source, &filters.0, shift1, target);
            }
            let stage: &[i16] = first_stage;
            let rows = |index: usize| {
                let start = index * MAX_BLOCK;
                stage.get(start..start + width).unwrap_or_default()
            };
            vertical(rows, &filters.1, SECOND_STAGE_SHIFT, width, height, out);
        }
    }
}

/// Row `y` of the reference scratch, `width` samples long.
#[inline]
fn fetched_row(reference: &[u16], y: usize, width: usize) -> &[u16] {
    let start = y * FETCH_SIZE;
    reference.get(start..start + width).unwrap_or_default()
}

/// Copies an area of `plane` into `reference`, each coordinate clamped to the plane.
fn fetch_reference<S: Sample>(plane: &RefPlane<'_, S>, area: RefBlock, reference: &mut [u16]) {
    let max_x = plane.width as i32 - 1;
    let max_y = plane.height as i32 - 1;
    let width = area.width as i32;
    let inside = area.x >= 0 && area.x.saturating_add(width) - 1 <= max_x;
    for (line, target) in reference
        .as_chunks_mut::<FETCH_SIZE>()
        .0
        .iter_mut()
        .enumerate()
        .take(area.height)
    {
        let row_index = area.y.saturating_add(line as i32).min(max_y).max(0);
        let Some(row) = plane.row(row_index as usize) else {
            continue;
        };
        if inside {
            let start = area.x as usize;
            let source = row.get(start..start + area.width).unwrap_or_default();
            for (out, &sample) in target.iter_mut().zip(source) {
                *out = sample.to_i32() as u16;
            }
        } else {
            for (offset, out) in target.iter_mut().enumerate().take(area.width) {
                let column = area.x.saturating_add(offset as i32).min(max_x).max(0);
                let sample = row.get(column as usize).copied().unwrap_or_default();
                *out = sample.to_i32() as u16;
            }
        }
    }
}

/// One row of `a`, `b`, `c`, `ab` to `ah`: `source` starts at the first tap of the
/// first sample.
#[inline]
fn horizontal<const TAPS: usize>(
    source: &[u16],
    filter: &[i32; TAPS],
    shift: u32,
    target: &mut [i16],
) {
    for (out, window) in target.iter_mut().zip(source.windows(TAPS)) {
        let sum: i32 = window
            .iter()
            .zip(filter)
            .map(|(&sample, &tap)| i32::from(sample) * tap)
            .sum();
        *out = (sum >> shift) as i16;
    }
}

/// The input of the vertical stage: samples or first-stage values.
trait StageValue: Copy {
    fn widen(self) -> i32;
}

impl StageValue for u16 {
    #[inline]
    fn widen(self) -> i32 {
        i32::from(self)
    }
}

impl StageValue for i16 {
    #[inline]
    fn widen(self) -> i32 {
        i32::from(self)
    }
}

/// The vertical stage: `rows( i )` returns input row `i`, and output row `y` reads
/// rows `y` to `y + TAPS - 1`.
fn vertical<'a, T: StageValue + 'a, const TAPS: usize>(
    rows: impl Fn(usize) -> &'a [T],
    filter: &[i32; TAPS],
    shift: u32,
    width: usize,
    height: usize,
    out: &mut PredSamples,
) {
    let mut sums = [0i32; MAX_BLOCK];
    for y in 0..height {
        sums.fill(0);
        for (index, &tap) in filter.iter().enumerate() {
            for (sum, &value) in sums.iter_mut().zip(rows(y + index)) {
                *sum += value.widen() * tap;
            }
        }
        for (target, &sum) in out.row_mut(y, width).iter_mut().zip(&sums) {
            *target = (sum >> shift) as i16;
        }
    }
}

#[cfg(test)]
#[path = "interp_tests.rs"]
mod tests;
