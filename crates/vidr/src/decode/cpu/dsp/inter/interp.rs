//! Fractional sample interpolation (8.5.3.3.3).
//!
//! A block inside the picture reads the reference plane directly. A block across an edge
//! first copies its reference area into a dense scratch buffer through clamped
//! coordinates, (8-222), (8-223), (8-239), and (8-240). The same stages run on both: each
//! input row widens to 16-bit lanes, then the filters run in vector blocks of 16 and 8.
//!
//! A block computes whole vectors. Lanes past its width land in working memory that
//! nothing reads, so a row has no scalar tail. Every plane sample holds at most
//! `( 1 << bit_depth ) - 1`.

use fearless_simd::{Simd, i16x8, i16x16};

use crate::decode::cpu::dsp::sample::Sample;
use crate::decode::cpu::dsp::simd::{
    Cpu, load_i16x8, load_i16x16, store_i16x8, store_i16x16, with_simd,
};

use super::fir::{sum16_x8, sum16_x16, sum32_x8, sum32_x16};

/// The largest prediction block side.
pub(crate) const MAX_BLOCK: usize = 64;
/// The most taps of a filter.
const MAX_TAPS: usize = 8;
/// Samples per row and rows in the reference scratch.
const FETCH_SIZE: usize = MAX_BLOCK + MAX_TAPS - 1;
/// Lanes per row of widened input: a whole number of vectors covering `FETCH_SIZE`.
const WIDE_ROW: usize = FETCH_SIZE + 1;
/// `shift2`: 6 at every bit depth.
const SECOND_STAGE_SHIFT: u32 = 6;

/// The 8-tap luma filters by fraction, from `-3` to `+4` (8-224) to (8-229).
const LUMA_FILTERS: [[i16; 8]; 4] = [
    [0, 0, 0, 64, 0, 0, 0, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 40, 40, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
];

/// The 4-tap chroma filters by fraction, from `-1` to `+2` (8-241) to (8-247).
static CHROMA_FILTERS: [[i16; 4]; 8] = [
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

    /// Sets every value to zero.
    pub(crate) fn clear(&mut self) {
        self.samples.fill(0);
    }

    /// `width` values of row `y`.
    #[cfg(test)]
    pub(crate) fn row(&self, y: usize, width: usize) -> &[i16] {
        let start = y * MAX_BLOCK;
        self.samples.get(start..start + width).unwrap_or_default()
    }

    /// All 64 values of row `y`, or `None` past the last row.
    #[inline(always)]
    pub(crate) fn full_row(&self, y: usize) -> Option<&[i16; MAX_BLOCK]> {
        self.samples.as_chunks::<MAX_BLOCK>().0.get(y)
    }
}

/// The working memory of the interpolation process.
#[derive(Clone, Debug)]
pub(crate) struct InterpScratch {
    reference: [u16; FETCH_SIZE * FETCH_SIZE],
    /// The input rows as 16-bit lanes.
    widened: [i16; WIDE_ROW * FETCH_SIZE],
    /// The first-stage values of a block with two fractions: `a`, `b`, `c`, `aY`.
    first_stage: [i16; MAX_BLOCK * FETCH_SIZE],
}

impl InterpScratch {
    /// Working memory for any block size.
    pub(crate) const fn new() -> Self {
        Self {
            reference: [0; FETCH_SIZE * FETCH_SIZE],
            widened: [0; WIDE_ROW * FETCH_SIZE],
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

/// Rows of a dense area of samples. A row runs on past its width into whatever follows
/// it, because a whole-vector load can read further than a block needs.
#[derive(Clone, Copy, Debug)]
struct Rows<'a, T> {
    /// The first sample of the first row, and all that follows it.
    samples: &'a [T],
    /// Samples from one row start to the next.
    stride: usize,
}

impl<'a, T> Rows<'a, T> {
    /// Row `y` and all that follows it, or an empty row past the end of `samples`.
    #[inline(always)]
    fn row(&self, y: usize) -> &'a [T] {
        self.samples.get(y * self.stride..).unwrap_or_default()
    }
}

/// The values every stage of one block reads.
#[derive(Clone, Copy, Debug)]
struct Job {
    /// The reference area the filters read, with the extra samples of their taps.
    area: RefBlock,
    /// Output samples per row.
    width: usize,
    /// Output rows.
    height: usize,
    /// True at 8 bits, where a first-stage sum fits 16 bits and `shift1` is 0.
    eight_bit: bool,
    /// `shift1`: `Min( 4, BitDepth - 8 )`.
    shift1: u32,
    /// `shift3`: `Max( 2, 14 - BitDepth )`.
    shift3: u32,
}

impl Job {
    /// `TAPS` is even, and position `TAPS / 2 - 1` of a filter is the sample at the
    /// integer location.
    fn new<const TAPS: usize>(block: RefBlock, fractions: (bool, bool), bit_depth: u32) -> Self {
        let width = block.width.min(MAX_BLOCK);
        let height = block.height.min(MAX_BLOCK);
        let before = TAPS as i32 / 2 - 1;
        Self {
            area: RefBlock {
                x: if fractions.0 {
                    block.x.saturating_sub(before)
                } else {
                    block.x
                },
                y: if fractions.1 {
                    block.y.saturating_sub(before)
                } else {
                    block.y
                },
                width: if fractions.0 { width + TAPS - 1 } else { width },
                height: if fractions.1 {
                    height + TAPS - 1
                } else {
                    height
                },
            },
            width,
            height,
            eight_bit: bit_depth == 8,
            shift1: bit_depth.saturating_sub(8).min(4),
            shift3: 14u32.saturating_sub(bit_depth).max(2),
        }
    }
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
    luma_on(Cpu::detected(), plane, block, frac, bit_depth, scratch, out);
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
    chroma_on(Cpu::detected(), plane, block, frac, bit_depth, scratch, out);
}

/// [`interpolate_luma`] compiled for `cpu`.
fn luma_on<S: Sample>(
    cpu: Cpu,
    plane: &RefPlane<'_, S>,
    block: RefBlock,
    frac: (usize, usize),
    bit_depth: u32,
    scratch: &mut InterpScratch,
    out: &mut PredSamples,
) {
    let frac = (frac.0 & 3, frac.1 & 3);
    let job = Job::new::<8>(block, (frac.0 != 0, frac.1 != 0), bit_depth);
    let InterpScratch {
        reference,
        widened,
        first_stage,
    } = scratch;
    if let Some(rows) = direct_rows(plane, job.area) {
        with_simd!(cpu, simd => luma_stages(simd, rows, frac, &job, widened, first_stage, out));
    } else {
        fetch_reference(plane, job.area, reference);
        let rows = copied_rows(reference);
        with_simd!(cpu, simd => luma_stages(simd, rows, frac, &job, widened, first_stage, out));
    }
}

/// [`interpolate_chroma`] compiled for `cpu`.
fn chroma_on<S: Sample>(
    cpu: Cpu,
    plane: &RefPlane<'_, S>,
    block: RefBlock,
    frac: (usize, usize),
    bit_depth: u32,
    scratch: &mut InterpScratch,
    out: &mut PredSamples,
) {
    let frac = (frac.0 & 7, frac.1 & 7);
    let job = Job::new::<4>(block, (frac.0 != 0, frac.1 != 0), bit_depth);
    let InterpScratch {
        reference,
        widened,
        first_stage,
    } = scratch;
    if let Some(rows) = direct_rows(plane, job.area) {
        with_simd!(cpu, simd => chroma_stages(simd, rows, frac, &job, widened, first_stage, out));
    } else {
        fetch_reference(plane, job.area, reference);
        let rows = copied_rows(reference);
        with_simd!(cpu, simd => chroma_stages(simd, rows, frac, &job, widened, first_stage, out));
    }
}

/// The rows of `area` in the plane, or `None` when `area` crosses a plane edge.
fn direct_rows<'a, S: Sample>(plane: &RefPlane<'a, S>, area: RefBlock) -> Option<Rows<'a, S>> {
    let x = usize::try_from(area.x).ok()?;
    let y = usize::try_from(area.y).ok()?;
    if x.checked_add(area.width)? > plane.width || y.checked_add(area.height)? > plane.height {
        return None;
    }
    let start = y.checked_mul(plane.stride)?.checked_add(x)?;
    Some(Rows {
        samples: plane.samples.get(start..)?,
        stride: plane.stride,
    })
}

/// The rows of the reference scratch.
fn copied_rows(reference: &[u16]) -> Rows<'_, u16> {
    Rows {
        samples: reference,
        stride: FETCH_SIZE,
    }
}

/// Runs `body` with one luma filter of `frac` 1, 2, or 3, as a constant.
#[inline(always)]
fn with_luma_taps(frac: usize, mut body: impl FnMut(&[i16; 8])) {
    match frac {
        1 => body(&LUMA_FILTERS[1]),
        2 => body(&LUMA_FILTERS[2]),
        _ => body(&LUMA_FILTERS[3]),
    }
}

/// The stages of Table 8-8.
#[inline(always)]
fn luma_stages<V: Simd, T: Sample>(
    simd: V,
    rows: Rows<'_, T>,
    frac: (usize, usize),
    job: &Job,
    widened: &mut [i16],
    first_stage: &mut [i16],
    out: &mut PredSamples,
) {
    let (width, height) = (job.width, job.height);
    let target = out.samples.as_chunks_mut::<MAX_BLOCK>().0;
    if frac == (0, 0) {
        full_sample_stage(simd, rows, job, target);
        return;
    }
    widen_rows(simd, rows, job, widened);
    let wide = widened.as_chunks::<WIDE_ROW>().0;
    match (frac.0 != 0, frac.1 != 0) {
        (true, false) => with_luma_taps(
            frac.0,
            #[inline(always)]
            |taps| horizontal_stage(simd, wide, taps, job, (width, height), target),
        ),
        (false, _) => with_luma_taps(
            frac.1,
            #[inline(always)]
            |taps| vertical_first_stage(simd, wide, taps, job, target),
        ),
        (true, true) => {
            let stage_target = first_stage.as_chunks_mut::<MAX_BLOCK>().0;
            with_luma_taps(
                frac.0,
                #[inline(always)]
                |taps| horizontal_stage(simd, wide, taps, job, (width, height + 7), stage_target),
            );
            let stage = first_stage.as_chunks::<MAX_BLOCK>().0;
            with_luma_taps(
                frac.1,
                #[inline(always)]
                |taps| vertical_second_stage(simd, stage, taps, job, target),
            );
        }
    }
}

/// The stages of Table 8-9.
#[inline(always)]
fn chroma_stages<V: Simd, T: Sample>(
    simd: V,
    rows: Rows<'_, T>,
    frac: (usize, usize),
    job: &Job,
    widened: &mut [i16],
    first_stage: &mut [i16],
    out: &mut PredSamples,
) {
    let (width, height) = (job.width, job.height);
    let target = out.samples.as_chunks_mut::<MAX_BLOCK>().0;
    if frac == (0, 0) {
        full_sample_stage(simd, rows, job, target);
        return;
    }
    widen_rows(simd, rows, job, widened);
    let wide = widened.as_chunks::<WIDE_ROW>().0;
    let (x_taps, y_taps) = (&CHROMA_FILTERS[frac.0], &CHROMA_FILTERS[frac.1]);
    match (frac.0 != 0, frac.1 != 0) {
        (true, false) => horizontal_stage(simd, wide, x_taps, job, (width, height), target),
        (false, _) => vertical_first_stage(simd, wide, y_taps, job, target),
        (true, true) => {
            let stage_target = first_stage.as_chunks_mut::<MAX_BLOCK>().0;
            horizontal_stage(simd, wide, x_taps, job, (width, height + 3), stage_target);
            let stage = first_stage.as_chunks::<MAX_BLOCK>().0;
            vertical_second_stage(simd, stage, y_taps, job, target);
        }
    }
}

/// Eight samples from the start of `src` as lanes. Near the end of the samples, the
/// lanes past `src` read as zero.
#[inline(always)]
fn load_samples_x8<V: Simd, T: Sample>(simd: V, src: &[T]) -> i16x8<V> {
    if let Some(lanes) = src.first_chunk::<8>() {
        T::load_x8(simd, lanes)
    } else {
        let mut lanes = [T::default(); 8];
        for (lane, &sample) in lanes.iter_mut().zip(src) {
            *lane = sample;
        }
        T::load_x8(simd, &lanes)
    }
}

/// A full-sample location: `sample << shift3`, with the bits past 16 dropped.
#[inline(always)]
fn full_sample_stage<V: Simd, T: Sample>(
    simd: V,
    rows: Rows<'_, T>,
    job: &Job,
    target: &mut [[i16; MAX_BLOCK]],
) {
    let (width, shift) = (job.width, job.shift3);
    for (y, line) in target.iter_mut().enumerate().take(job.height) {
        let source = rows.row(y);
        write_line(
            line,
            width,
            #[inline(always)]
            |x| {
                let lanes = load_samples_x8(simd, source.get(x..).unwrap_or_default());
                simd.shl_i16x8(lanes, shift)
            },
            #[inline(always)]
            |x| {
                let lanes = match source.get(x..).and_then(<[T]>::first_chunk::<16>) {
                    Some(lanes) => T::load_x16(simd, lanes),
                    None => simd.splat_i16x16(0),
                };
                simd.shl_i16x16(lanes, shift)
            },
        );
    }
}

/// Widens each row of the reference area to 16-bit lanes.
#[inline(always)]
fn widen_rows<V: Simd, T: Sample>(simd: V, rows: Rows<'_, T>, job: &Job, widened: &mut [i16]) {
    let targets = widened.as_chunks_mut::<WIDE_ROW>().0;
    let width = job.area.width;
    for (y, line) in targets.iter_mut().enumerate().take(job.area.height) {
        let source = rows.row(y);
        let mut x = 0;
        while x < width {
            let lanes = load_samples_x8(simd, source.get(x..).unwrap_or_default());
            store_i16x8(lanes, line, x);
            x += 8;
        }
    }
}

/// Writes `width` lanes of one output row, in whole vectors. `x8` and `x16` return the
/// vector at a lane index. Each index is a multiple of 8 below 64, so the bounds of every
/// load and store of a `MAX_BLOCK` row fold away through the masks below.
#[inline(always)]
fn write_line<V: Simd>(
    line: &mut [i16; MAX_BLOCK],
    width: usize,
    x8: impl Fn(usize) -> i16x8<V>,
    x16: impl Fn(usize) -> i16x16<V>,
) {
    let mut x = 0;
    while x + 16 <= width {
        let at = x & 48;
        store_i16x16(x16(at), line, at);
        x += 16;
    }
    while x < width {
        let at = x & 56;
        store_i16x8(x8(at), line, at);
        x += 8;
    }
}

/// `a`, `b`, `c`, `ab` to `ah`: output lane `x` reads lanes `x` to `x + TAPS - 1` of its
/// input row. `size` holds the output width and height.
#[inline(always)]
fn horizontal_stage<V: Simd, const TAPS: usize>(
    simd: V,
    wide: &[[i16; WIDE_ROW]],
    taps: &[i16; TAPS],
    job: &Job,
    size: (usize, usize),
    target: &mut [[i16; MAX_BLOCK]],
) {
    let shift = job.shift1;
    for (line, row) in target.iter_mut().zip(wide).take(size.1) {
        if job.eight_bit {
            write_line(
                line,
                size.0,
                #[inline(always)]
                |x| {
                    sum16_x8(
                        simd,
                        taps,
                        #[inline(always)]
                        |k| load_i16x8(simd, row, x + k),
                    )
                },
                #[inline(always)]
                |x| {
                    sum16_x16(
                        simd,
                        taps,
                        #[inline(always)]
                        |k| load_i16x16(simd, row, x + k),
                    )
                },
            );
        } else {
            write_line(
                line,
                size.0,
                #[inline(always)]
                |x| {
                    sum32_x8(
                        simd,
                        taps,
                        shift,
                        #[inline(always)]
                        |k| load_i16x8(simd, row, x + k),
                    )
                },
                #[inline(always)]
                |x| {
                    sum32_x16(
                        simd,
                        taps,
                        shift,
                        #[inline(always)]
                        |k| load_i16x16(simd, row, x + k),
                    )
                },
            );
        }
    }
}

/// The vertical stage on widened samples: output row `y` reads rows `y` to
/// `y + TAPS - 1`.
#[inline(always)]
fn vertical_first_stage<V: Simd, const TAPS: usize>(
    simd: V,
    wide: &[[i16; WIDE_ROW]],
    taps: &[i16; TAPS],
    job: &Job,
    target: &mut [[i16; MAX_BLOCK]],
) {
    if job.eight_bit {
        for (y, line) in target.iter_mut().enumerate().take(job.height) {
            let Some(rows) = wide.get(y..).and_then(<[_]>::first_chunk::<TAPS>) else {
                return;
            };
            write_line(
                line,
                job.width,
                #[inline(always)]
                |x| {
                    sum16_x8(
                        simd,
                        taps,
                        #[inline(always)]
                        |k| load_i16x8(simd, &rows[k], x),
                    )
                },
                #[inline(always)]
                |x| {
                    sum16_x16(
                        simd,
                        taps,
                        #[inline(always)]
                        |k| load_i16x16(simd, &rows[k], x),
                    )
                },
            );
        }
    } else {
        vertical_sum32(simd, wide, taps, job.shift1, job, target);
    }
}

/// The vertical stage on first-stage values.
#[inline(always)]
fn vertical_second_stage<V: Simd, const TAPS: usize>(
    simd: V,
    stage: &[[i16; MAX_BLOCK]],
    taps: &[i16; TAPS],
    job: &Job,
    target: &mut [[i16; MAX_BLOCK]],
) {
    vertical_sum32(simd, stage, taps, SECOND_STAGE_SHIFT, job, target);
}

/// The vertical stage in 32-bit sums.
#[inline(always)]
fn vertical_sum32<V: Simd, const TAPS: usize, const ROW: usize>(
    simd: V,
    input: &[[i16; ROW]],
    taps: &[i16; TAPS],
    shift: u32,
    job: &Job,
    target: &mut [[i16; MAX_BLOCK]],
) {
    for (y, line) in target.iter_mut().enumerate().take(job.height) {
        let Some(rows) = input.get(y..).and_then(<[_]>::first_chunk::<TAPS>) else {
            return;
        };
        write_line(
            line,
            job.width,
            #[inline(always)]
            |x| {
                sum32_x8(
                    simd,
                    taps,
                    shift,
                    #[inline(always)]
                    |k| load_i16x8(simd, &rows[k], x),
                )
            },
            #[inline(always)]
            |x| {
                sum32_x16(
                    simd,
                    taps,
                    shift,
                    #[inline(always)]
                    |k| load_i16x16(simd, &rows[k], x),
                )
            },
        );
    }
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

#[cfg(test)]
#[path = "interp_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "interp_simd_tests.rs"]
mod simd_tests;
