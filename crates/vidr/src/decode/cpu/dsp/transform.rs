//! Inverse transforms and the transform bypass paths (8.6.2, 8.6.4, 8.6.5).
//!
//! A pass runs all rows of a square array at once: a row holds one frequency, and
//! its columns are lanes. The lanes run on explicit 32-bit vectors. The result equals
//! the matrix product of (8-317) exactly. Each matrix row is exactly even or odd about its centre.
//!
//! The extent of the coded levels bounds the DCT: a row past `max_y` and a column past
//! `max_x` hold only zeros, and zeros transform to zeros.

use fearless_simd::{Simd, SimdFrom, i32x4, i32x8};

use super::scale::{COEFF_MAX, COEFF_MIN};
use super::simd::{Cpu, with_simd};
use super::transform_matrix::{DCT_COL0TO15, DST};

/// Samples in the largest transform block.
const BLOCK_SAMPLES: usize = 32 * 32;

/// The location of the last level that is not zero, along each axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CoeffExtent {
    /// The largest `xC` of a level.
    pub(crate) max_x: usize,
    /// The largest `yC` of a level.
    pub(crate) max_y: usize,
}

/// One transform block of residual data, row by row with no gaps. A block of size
/// `n` takes the first `n * n` entries. The DCT reads each working array only where it
/// wrote the same call, so a new block size clears none of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TbBlock {
    samples: [i32; BLOCK_SAMPLES],
    /// The output of the column pass, by position and column.
    columns: [i32; BLOCK_SAMPLES],
    /// The input of the row pass, by frequency and row.
    lanes: [i32; BLOCK_SAMPLES],
    /// The output of the row pass, by position and row.
    rows: [i32; BLOCK_SAMPLES],
    /// The odd parts of one pass.
    odd: [i32; BLOCK_SAMPLES],
}

impl TbBlock {
    /// A block of zeros.
    pub(crate) const fn new() -> Self {
        Self {
            samples: [0; BLOCK_SAMPLES],
            columns: [0; BLOCK_SAMPLES],
            lanes: [0; BLOCK_SAMPLES],
            rows: [0; BLOCK_SAMPLES],
            odd: [0; BLOCK_SAMPLES],
        }
    }

    /// Sets every sample of a block of `Log2( nTbS )` `log2_size` to zero.
    pub(crate) fn clear(&mut self, log2_size: u32) {
        let count = 1usize << (2 * log2_size.min(5));
        self.samples[..count].fill(0);
    }

    /// The sample at row `y`, column `x` of a block of `Log2( nTbS )` `log2_size`.
    #[cfg(test)]
    pub(crate) fn get(&self, log2_size: u32, x: usize, y: usize) -> i32 {
        self.samples.get((y << log2_size) + x).copied().unwrap_or(0)
    }

    /// Writes one sample. A location outside the array is ignored.
    pub(crate) fn set(&mut self, log2_size: u32, x: usize, y: usize, value: i32) {
        if let Some(sample) = self.samples.get_mut((y << log2_size) + x) {
            *sample = value;
        }
    }

    /// Row `y` of a block of `Log2( nTbS )` `log2_size`, or an empty row outside it.
    pub(crate) fn row(&self, log2_size: u32, y: usize) -> &[i32] {
        let start = y << log2_size;
        self.samples
            .get(start..start + (1usize << log2_size))
            .unwrap_or_default()
    }

    /// The first `n` rows as `n` arrays. `None` when they do not fit, and no
    /// size this module instantiates reaches that.
    fn square<const N: usize>(&mut self) -> Option<&mut [[i32; N]; N]> {
        square_of(&mut self.samples)
    }
}

/// The first `n` rows of `samples` as `n` arrays, or `None` when they do not fit.
#[inline(always)]
fn square_of<const N: usize>(samples: &mut [i32]) -> Option<&mut [[i32; N]; N]> {
    let (rows, _) = samples.as_chunks_mut::<N>();
    rows.first_chunk_mut::<N>()
}

/// `sums[ lane ] += coefficient * src[ lane ]` over the first `blocks` lane blocks. `BLOCK`
/// is 4 or 8.
#[inline(always)]
fn add_scaled<V: Simd, const N: usize, const BLOCK: usize>(
    simd: V,
    sums: &mut [i32; N],
    src: &[i32; N],
    blocks: usize,
    coefficient: i32,
) {
    if BLOCK == 8 {
        let factor = simd.splat_i32x8(coefficient);
        let targets = sums.as_chunks_mut::<8>().0;
        let inputs = src.as_chunks::<8>().0;
        for (target, input) in targets.iter_mut().zip(inputs).take(blocks) {
            let product = simd.mul_i32x8(i32x8::simd_from(simd, *input), factor);
            *target = simd
                .add_i32x8(i32x8::simd_from(simd, *target), product)
                .into();
        }
    } else {
        let factor = simd.splat_i32x4(coefficient);
        let targets = sums.as_chunks_mut::<4>().0;
        let inputs = src.as_chunks::<4>().0;
        for (target, input) in targets.iter_mut().zip(inputs).take(blocks) {
            let product = simd.mul_i32x4(i32x4::simd_from(simd, *input), factor);
            *target = simd
                .add_i32x4(i32x4::simd_from(simd, *target), product)
                .into();
        }
    }
}

/// `mirror = even - odd`, then `even = even + odd`, over the first `blocks` lane blocks.
/// `BLOCK` is 4 or 8.
#[inline(always)]
fn butterfly<V: Simd, const N: usize, const BLOCK: usize>(
    simd: V,
    even: &mut [i32; N],
    mirror: &mut [i32; N],
    odd: &[i32; N],
    blocks: usize,
) {
    if BLOCK == 8 {
        let evens = even.as_chunks_mut::<8>().0;
        let mirrors = mirror.as_chunks_mut::<8>().0;
        let odds = odd.as_chunks::<8>().0;
        for ((high, low), part) in evens.iter_mut().zip(mirrors).zip(odds).take(blocks) {
            let (value, part) = (i32x8::simd_from(simd, *high), i32x8::simd_from(simd, *part));
            *low = simd.sub_i32x8(value, part).into();
            *high = simd.add_i32x8(value, part).into();
        }
    } else {
        let evens = even.as_chunks_mut::<4>().0;
        let mirrors = mirror.as_chunks_mut::<4>().0;
        let odds = odd.as_chunks::<4>().0;
        for ((high, low), part) in evens.iter_mut().zip(mirrors).zip(odds).take(blocks) {
            let (value, part) = (i32x4::simd_from(simd, *high), i32x4::simd_from(simd, *part));
            *low = simd.sub_i32x4(value, part).into();
            *high = simd.add_i32x4(value, part).into();
        }
    }
}

/// (8-314) on the first `blocks` lane blocks of one row. `BLOCK` is 4 or 8.
#[inline(always)]
fn clip_row<V: Simd, const N: usize, const BLOCK: usize>(
    simd: V,
    row: &mut [i32; N],
    blocks: usize,
) {
    if BLOCK == 8 {
        let round = simd.splat_i32x8(64);
        let (low, high) = (simd.splat_i32x8(COEFF_MIN), simd.splat_i32x8(COEFF_MAX));
        for lanes in row.as_chunks_mut::<8>().0.iter_mut().take(blocks) {
            let rounded = simd.add_i32x8(i32x8::simd_from(simd, *lanes), round);
            *lanes = simd
                .min_i32x8(simd.max_i32x8(simd.shr_i32x8(rounded, 7), low), high)
                .into();
        }
    } else {
        let round = simd.splat_i32x4(64);
        let (low, high) = (simd.splat_i32x4(COEFF_MIN), simd.splat_i32x4(COEFF_MAX));
        for lanes in row.as_chunks_mut::<4>().0.iter_mut().take(blocks) {
            let rounded = simd.add_i32x4(i32x4::simd_from(simd, *lanes), round);
            *lanes = simd
                .min_i32x4(simd.max_i32x4(simd.shr_i32x4(rounded, 7), low), high)
                .into();
        }
    }
}

/// One DCT pass (8-317): `out[ i ][ lane ]` from `src[ j ][ lane ]`, on the first
/// `blocks` lane blocks of `BLOCK` lanes. `live` rows of `src` hold levels, and the
/// others hold zeros and stay unread.
///
/// The output of size `n` builds on the output of size `n / 2`. The even rows of
/// the input make the even part, and the odd rows make the odd part. The even
/// part stays in the first half of `out` until this step overwrites it.
#[inline(always)]
fn dct_pass<V: Simd, const N: usize, const BLOCK: usize>(
    simd: V,
    src: &[[i32; N]; N],
    out: &mut [[i32; N]; N],
    odd: &mut [[i32; N]; N],
    live: usize,
    blocks: usize,
) {
    let lanes = (blocks * BLOCK).min(N);
    for (target, &value) in out[0].iter_mut().zip(&src[0]).take(lanes) {
        *target = 64 * value;
    }
    let mut size = 2;
    while size <= N {
        let half = size / 2;
        let input_step = N / size;
        let matrix_step = 32 / size;
        for (position, sums) in odd.iter_mut().take(half).enumerate() {
            sums[..lanes].fill(0);
            for frequency in (1..size).step_by(2) {
                let row = frequency * input_step;
                if row >= live {
                    break;
                }
                let coefficient = i32::from(DCT_COL0TO15[frequency * matrix_step][position]);
                add_scaled::<V, N, BLOCK>(simd, sums, &src[row], blocks, coefficient);
            }
        }
        let (low, high) = out.split_at_mut(half);
        for (position, sums) in odd.iter().take(half).enumerate() {
            butterfly::<V, N, BLOCK>(
                simd,
                &mut low[position],
                &mut high[half - 1 - position],
                sums,
                blocks,
            );
        }
        size *= 2;
    }
}

/// One DST pass (8-315).
fn dst_pass(src: &[[i32; 4]; 4], out: &mut [[i32; 4]; 4]) {
    for (position, target) in out.iter_mut().enumerate() {
        *target = [0; 4];
        for (basis, row) in DST.iter().zip(src) {
            let coefficient = basis[position];
            for (sum, &value) in target.iter_mut().zip(row) {
                *sum += coefficient * value;
            }
        }
    }
}

/// The two passes of 8.6.4.1, with the clip of (8-314) between them, for the DST.
fn transform_2d<const N: usize>(
    block: &mut [[i32; N]; N],
    pass: impl Fn(&[[i32; N]; N], &mut [[i32; N]; N]),
) {
    let mut columns_done = [[0i32; N]; N];
    let coefficients: &[[i32; N]; N] = block;
    pass(coefficients, &mut columns_done);
    let mut rows_as_lanes = [[0i32; N]; N];
    for (y, row) in columns_done.iter().enumerate() {
        for (x, &value) in row.iter().enumerate() {
            rows_as_lanes[x][y] = ((value + 64) >> 7).clamp(COEFF_MIN, COEFF_MAX);
        }
    }
    let mut rows_done = [[0i32; N]; N];
    pass(&rows_as_lanes, &mut rows_done);
    for (x, row) in rows_done.iter().enumerate() {
        for (y, &value) in row.iter().enumerate() {
            block[y][x] = value;
        }
    }
}

/// The DCT case of 8.6.4 on a block of `Log2( nTbS )` `log2_size`, 2 to 5, every level
/// inside `extent`. Other sizes leave the block unchanged. Scaling (8.6.3) clips each
/// level to `COEFF_MIN` to `COEFF_MAX`, so the input fits 16 bits.
pub(crate) fn inverse_dct(log2_size: u32, block: &mut TbBlock, extent: CoeffExtent) {
    inverse_dct_on(Cpu::detected(), log2_size, block, extent);
}

/// [`inverse_dct`] compiled for `cpu`.
fn inverse_dct_on(cpu: Cpu, log2_size: u32, block: &mut TbBlock, extent: CoeffExtent) {
    with_simd!(cpu, simd => match log2_size {
        2 => dct_square::<_, 4, 4>(simd, block, extent),
        3 => dct_square::<_, 8, 8>(simd, block, extent),
        4 => dct_square::<_, 16, 8>(simd, block, extent),
        5 => dct_square::<_, 32, 8>(simd, block, extent),
        _ => {}
    });
}

/// The two passes of 8.6.4.1 inside `extent`. Lanes run in blocks of `BLOCK`, so a column
/// count rounds up to a whole block.
#[inline(always)]
fn dct_square<V: Simd, const N: usize, const BLOCK: usize>(
    simd: V,
    block: &mut TbBlock,
    extent: CoeffExtent,
) {
    let TbBlock {
        samples,
        columns,
        lanes,
        rows,
        odd,
    } = block;
    let squares = (
        square_of::<N>(samples),
        square_of::<N>(columns),
        square_of::<N>(lanes),
        square_of::<N>(rows),
        square_of::<N>(odd),
    );
    let (Some(samples), Some(columns), Some(lanes), Some(rows), Some(odd)) = squares else {
        return;
    };
    let max_x = extent.max_x.min(N - 1);
    let max_y = extent.max_y.min(N - 1);
    if max_x == 0 && max_y == 0 {
        // A lone DC level: 64 at every position of each pass.
        let clipped = ((64 * samples[0][0] + 64) >> 7).clamp(COEFF_MIN, COEFF_MAX);
        for row in samples.iter_mut() {
            row.fill(64 * clipped);
        }
        return;
    }
    let column_blocks = max_x / BLOCK + 1;
    let column_count = column_blocks * BLOCK;
    dct_pass::<V, N, BLOCK>(simd, samples, columns, odd, max_y + 1, column_blocks);
    for row in columns.iter_mut() {
        clip_row::<V, N, BLOCK>(simd, row, column_blocks);
    }
    for (y, row) in columns.iter().enumerate() {
        for (x, &value) in row.iter().enumerate().take(column_count) {
            lanes[x][y] = value;
        }
    }
    dct_pass::<V, N, BLOCK>(simd, lanes, rows, odd, max_x + 1, N / BLOCK);
    for (x, row) in rows.iter().enumerate() {
        for (y, &value) in row.iter().enumerate() {
            samples[y][x] = value;
        }
    }
}

/// The DST case of 8.6.4 on a 4x4 block.
pub(crate) fn inverse_dst(block: &mut TbBlock) {
    if let Some(square) = block.square::<4>() {
        transform_2d(square, dst_pass);
    }
}

/// The transform skip case of 8.6.2: the left shift by `tsShift` (8-296, 8-298).
pub(crate) fn transform_skip(log2_size: u32, block: &mut TbBlock) {
    let log2_size = log2_size.min(5);
    let ts_shift = 5 + log2_size;
    for sample in block.samples.iter_mut().take(1 << (2 * log2_size)) {
        *sample <<= ts_shift;
    }
}

/// The rounding of (8-299), with `bdShift` from (8-295).
pub(crate) fn round_residual(log2_size: u32, bit_depth: u32, block: &mut TbBlock) {
    round_residual_on(Cpu::detected(), log2_size, bit_depth, block);
}

/// [`round_residual`] compiled for `cpu`.
fn round_residual_on(cpu: Cpu, log2_size: u32, bit_depth: u32, block: &mut TbBlock) {
    let bd_shift = 20 - bit_depth;
    let round = 1i32 << (bd_shift - 1);
    let count = 1usize << (2 * log2_size.min(5));
    let samples = block.samples.get_mut(..count).unwrap_or_default();
    with_simd!(cpu, simd => {
        let offset = simd.splat_i32x8(round);
        let (blocks, rest) = samples.as_chunks_mut::<8>();
        for lanes in blocks {
            let sum = simd.add_i32x8(i32x8::simd_from(simd, *lanes), offset);
            *lanes = simd.shr_i32x8(sum, bd_shift).into();
        }
        for sample in rest {
            *sample = (*sample + round) >> bd_shift;
        }
    });
}

#[cfg(test)]
#[path = "transform_tests.rs"]
mod tests;
