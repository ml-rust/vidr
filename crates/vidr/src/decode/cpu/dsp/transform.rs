//! Inverse transforms and the transform bypass paths (8.6.2, 8.6.4, 8.6.5).
//!
//! A pass runs all rows of a square array at once: a row holds one frequency, and
//! its columns are lanes. Each loop over lanes is plain integer arithmetic, so
//! the compiler vectorises it. The result equals the matrix product of (8-317)
//! exactly. Each matrix row is exactly even or odd about its centre.

use super::scale::{COEFF_MAX, COEFF_MIN};
use super::transform_matrix::{DCT_COL0TO15, DST};

/// Samples in the largest transform block.
const BLOCK_SAMPLES: usize = 32 * 32;

/// One transform block of residual data, row by row with no gaps. A block of size
/// `n` takes the first `n * n` entries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TbBlock {
    samples: [i32; BLOCK_SAMPLES],
}

impl TbBlock {
    /// A block of zeros.
    pub(crate) const fn new() -> Self {
        Self {
            samples: [0; BLOCK_SAMPLES],
        }
    }

    /// Sets every sample of a block of `Log2( nTbS )` `log2_size` to zero.
    pub(crate) fn clear(&mut self, log2_size: u32) {
        let count = 1usize << (2 * log2_size.min(5));
        self.samples[..count].fill(0);
    }

    /// The sample at row `y`, column `x` of a block of `Log2( nTbS )` `log2_size`.
    pub(crate) fn get(&self, log2_size: u32, x: usize, y: usize) -> i32 {
        self.samples.get((y << log2_size) + x).copied().unwrap_or(0)
    }

    /// Writes one sample. A location outside the array is ignored.
    pub(crate) fn set(&mut self, log2_size: u32, x: usize, y: usize, value: i32) {
        if let Some(sample) = self.samples.get_mut((y << log2_size) + x) {
            *sample = value;
        }
    }

    /// The first `n` rows as `n` arrays. `None` when they do not fit, and no
    /// size this module instantiates reaches that.
    fn square<const N: usize>(&mut self) -> Option<&mut [[i32; N]; N]> {
        let (rows, _) = self.samples.as_chunks_mut::<N>();
        rows.first_chunk_mut::<N>()
    }
}

/// One DCT pass (8-317): `out[ i ][ lane ]` from `src[ j ][ lane ]`.
///
/// The output of size `n` builds on the output of size `n / 2`. The even rows of
/// the input make the even part, and the odd rows make the odd part. The even
/// part stays in the first half of `out` until this step overwrites it.
fn dct_pass<const N: usize>(src: &[[i32; N]; N], out: &mut [[i32; N]; N]) {
    let mut live = [false; N];
    for (flag, row) in live.iter_mut().zip(src) {
        *flag = row.iter().any(|&value| value != 0);
    }
    for (target, &value) in out[0].iter_mut().zip(&src[0]) {
        *target = 64 * value;
    }
    let mut odd = [[0i32; N]; N];
    let mut size = 2;
    while size <= N {
        let half = size / 2;
        let input_step = N / size;
        let matrix_step = 32 / size;
        for (position, sums) in odd.iter_mut().take(half).enumerate() {
            *sums = [0; N];
            for frequency in (1..size).step_by(2) {
                let row = frequency * input_step;
                if !live[row] {
                    continue;
                }
                let coefficient = i32::from(DCT_COL0TO15[frequency * matrix_step][position]);
                for (sum, &value) in sums.iter_mut().zip(&src[row]) {
                    *sum += coefficient * value;
                }
            }
        }
        let (low, high) = out.split_at_mut(half);
        for position in 0..half {
            let mirror = &mut high[half - 1 - position];
            let even = &mut low[position];
            let sums = &odd[position];
            for lane in 0..N {
                mirror[lane] = even[lane] - sums[lane];
                even[lane] += sums[lane];
            }
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

/// The two passes of 8.6.4.1, with the clip of (8-314) between them.
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

/// The DCT case of 8.6.4 on a block of `Log2( nTbS )` `log2_size`, 2 to 5. Other
/// sizes leave the block unchanged. Coefficients stay within 16 bits.
pub(crate) fn inverse_dct(log2_size: u32, block: &mut TbBlock) {
    match log2_size {
        2 => dct_square::<4>(block),
        3 => dct_square::<8>(block),
        4 => dct_square::<16>(block),
        5 => dct_square::<32>(block),
        _ => {}
    }
}

fn dct_square<const N: usize>(block: &mut TbBlock) {
    if let Some(square) = block.square::<N>() {
        transform_2d(square, dct_pass::<N>);
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
    let bd_shift = 20 - bit_depth;
    let round = 1 << (bd_shift - 1);
    for sample in block.samples.iter_mut().take(1 << (2 * log2_size.min(5))) {
        *sample = (*sample + round) >> bd_shift;
    }
}

/// The residual modification of 8.6.5: each sample adds the sample before it, to
/// the left when `vertical` is false and above when it is true.
///
/// Main and Main 10 streams never reach this, because the range extension tools
/// that invoke it are refused.
pub(crate) fn accumulate_residual(log2_size: u32, vertical: bool, block: &mut TbBlock) {
    let log2_size = log2_size.min(5);
    let size = 1usize << log2_size;
    if vertical {
        for y in 1..size {
            for x in 0..size {
                let previous = block.get(log2_size, x, y - 1);
                let current = block.get(log2_size, x, y);
                block.set(log2_size, x, y, current + previous);
            }
        }
    } else {
        for y in 0..size {
            for x in 1..size {
                let previous = block.get(log2_size, x - 1, y);
                let current = block.get(log2_size, x, y);
                block.set(log2_size, x, y, current + previous);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `transMatrixCol16to31` (8-321), as printed.
    const DCT_COL16TO31: [[i8; 16]; 32] = [
        [
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        ],
        [
            -4, -13, -22, -31, -38, -46, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90,
        ],
        [
            -90, -87, -80, -70, -57, -43, -25, -9, 9, 25, 43, 57, 70, 80, 87, 90,
        ],
        [
            13, 38, 61, 78, 88, 90, 85, 73, 54, 31, 4, -22, -46, -67, -82, -90,
        ],
        [
            89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89,
        ],
        [
            -22, -61, -85, -90, -73, -38, 4, 46, 78, 90, 82, 54, 13, -31, -67, -88,
        ],
        [
            -87, -57, -9, 43, 80, 90, 70, 25, -25, -70, -90, -80, -43, 9, 57, 87,
        ],
        [
            31, 78, 90, 61, 4, -54, -88, -82, -38, 22, 73, 90, 67, 13, -46, -85,
        ],
        [
            83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83,
        ],
        [
            -38, -88, -73, -4, 67, 90, 46, -31, -85, -78, -13, 61, 90, 54, -22, -82,
        ],
        [
            -80, -9, 70, 87, 25, -57, -90, -43, 43, 90, 57, -25, -87, -70, 9, 80,
        ],
        [
            46, 90, 38, -54, -90, -31, 61, 88, 22, -67, -85, -13, 73, 82, 4, -78,
        ],
        [
            75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75,
        ],
        [
            -54, -85, 4, 88, 46, -61, -82, 13, 90, 38, -67, -78, 22, 90, 31, -73,
        ],
        [
            -70, 43, 87, -9, -90, -25, 80, 57, -57, -80, 25, 90, 9, -87, -43, 70,
        ],
        [
            61, 73, -46, -82, 31, 88, -13, -90, -4, 90, 22, -85, -38, 78, 54, -67,
        ],
        [
            64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64,
        ],
        [
            -67, -54, 78, 38, -85, -22, 90, 4, -90, 13, 88, -31, -82, 46, 73, -61,
        ],
        [
            -57, 80, 25, -90, 9, 87, -43, -70, 70, 43, -87, -9, 90, -25, -80, 57,
        ],
        [
            73, 31, -90, 22, 78, -67, -38, 90, -13, -82, 61, 46, -88, 4, 85, -54,
        ],
        [
            50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50,
        ],
        [
            -78, -4, 82, -73, -13, 85, -67, -22, 88, -61, -31, 90, -54, -38, 90, -46,
        ],
        [
            -43, 90, -57, -25, 87, -70, -9, 80, -80, 9, 70, -87, 25, 57, -90, 43,
        ],
        [
            82, -22, -54, 90, -61, -13, 78, -85, 31, 46, -90, 67, 4, -73, 88, -38,
        ],
        [
            36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36,
        ],
        [
            -85, 46, 13, -67, 90, -73, 22, 38, -82, 88, -54, -4, 61, -90, 78, -31,
        ],
        [
            -25, 70, -90, 80, -43, -9, 57, -87, 87, -57, 9, 43, -80, 90, -70, 25,
        ],
        [
            88, -67, 31, 13, -54, 82, -90, 78, -46, 4, 38, -73, 90, -85, 61, -22,
        ],
        [
            18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18,
        ],
        [
            -90, 82, -67, 46, -22, -4, 31, -54, 73, -85, 90, -88, 78, -61, 38, -13,
        ],
        [
            -9, 25, -43, 57, -70, 80, -87, 90, -90, 87, -80, 70, -57, 43, -25, 9,
        ],
        [
            90, -90, 88, -85, 82, -78, 73, -67, 61, -54, 46, -38, 31, -22, 13, -4,
        ],
    ];

    /// `transMatrix[ position ][ frequency ]` of (8-318) and (8-320).
    fn matrix(position: usize, frequency: usize) -> i32 {
        if position < 16 {
            i32::from(DCT_COL0TO15[frequency][position])
        } else {
            i32::from(DCT_COL16TO31[frequency][position - 16])
        }
    }

    /// The matrix product of (8-317), written as the specification prints it.
    fn direct_dct(size: usize, input: &[i32]) -> Vec<i32> {
        let step = 32 / size;
        (0..size)
            .map(|position| {
                input
                    .iter()
                    .enumerate()
                    .map(|(frequency, &value)| matrix(position, frequency * step) * value)
                    .sum::<i32>()
            })
            .collect()
    }

    /// The matrix product of (8-315).
    fn direct_dst(input: &[i32]) -> Vec<i32> {
        (0..4)
            .map(|position| {
                input
                    .iter()
                    .enumerate()
                    .map(|(frequency, &value)| DST[frequency][position] * value)
                    .sum::<i32>()
            })
            .collect()
    }

    /// The two passes of 8.6.4.1 in direct form. Input and output are row by row.
    fn direct_2d(
        size: usize,
        coefficients: &[i32],
        one_dimension: impl Fn(&[i32]) -> Vec<i32>,
    ) -> Vec<i32> {
        let mut intermediate = vec![0; size * size];
        for x in 0..size {
            let column: Vec<i32> = (0..size).map(|y| coefficients[y * size + x]).collect();
            for (y, value) in one_dimension(&column).into_iter().enumerate() {
                intermediate[y * size + x] = ((value + 64) >> 7).clamp(COEFF_MIN, COEFF_MAX);
            }
        }
        let mut output = Vec::with_capacity(size * size);
        for row in intermediate.chunks(size) {
            output.extend(one_dimension(row));
        }
        output
    }

    struct Xorshift(u64);

    impl Xorshift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        /// A coefficient: often zero, sometimes small, sometimes at full range.
        fn coefficient(&mut self) -> i32 {
            let value = self.next();
            match value % 4 {
                0 | 1 => 0,
                2 => ((value >> 8) % 201) as i32 - 100,
                _ => ((value >> 8) % 65_536) as i32 - 32_768,
            }
        }
    }

    fn block_from(log2_size: u32, values: &[i32]) -> TbBlock {
        let mut block = TbBlock::new();
        let size = 1usize << log2_size;
        for (index, &value) in values.iter().enumerate() {
            block.set(log2_size, index % size, index / size, value);
        }
        block
    }

    fn block_rows(log2_size: u32, block: &TbBlock) -> Vec<i32> {
        let size = 1usize << log2_size;
        (0..size * size)
            .map(|index| block.get(log2_size, index % size, index / size))
            .collect()
    }

    /// Every printed value of (8-319) and (8-321) satisfies the symmetry the
    /// passes rely on.
    #[test]
    fn matrix_rows_are_even_or_odd_about_their_centre() {
        for frequency in 0..32 {
            let sign = if frequency & 1 == 0 { 1 } else { -1 };
            for position in 0..16 {
                assert_eq!(
                    matrix(31 - position, frequency),
                    sign * matrix(position, frequency),
                    "frequency {frequency}, position {position}"
                );
            }
        }
    }

    /// A row of a smaller transform is a row of the 32-point matrix.
    #[test]
    fn matrix_holds_the_known_values() {
        assert_eq!(matrix(0, 0), 64);
        assert_eq!(matrix(31, 0), 64);
        assert_eq!(matrix(0, 1), 90);
        assert_eq!(matrix(31, 1), -90);
        assert_eq!(matrix(0, 8), 83);
        assert_eq!(matrix(1, 8), 36);
        assert_eq!(matrix(15, 31), -90);
        assert_eq!(matrix(16, 31), 90);
    }

    #[test]
    fn dct_equals_the_direct_form_on_random_blocks() {
        let mut rng = Xorshift(0x2545_f491_4f6c_dd1d);
        for log2_size in 2..=5u32 {
            let size = 1usize << log2_size;
            for _ in 0..40 {
                let values: Vec<i32> = (0..size * size).map(|_| rng.coefficient()).collect();
                let expected = direct_2d(size, &values, |input| direct_dct(size, input));
                let mut block = block_from(log2_size, &values);
                inverse_dct(log2_size, &mut block);
                assert_eq!(block_rows(log2_size, &block), expected, "size {size}");
            }
        }
    }

    #[test]
    fn dct_equals_the_direct_form_on_every_single_coefficient() {
        for log2_size in 2..=5u32 {
            let size = 1usize << log2_size;
            for index in 0..size * size {
                let mut values = vec![0; size * size];
                values[index] = -32_768;
                let expected = direct_2d(size, &values, |input| direct_dct(size, input));
                let mut block = block_from(log2_size, &values);
                inverse_dct(log2_size, &mut block);
                assert_eq!(block_rows(log2_size, &block), expected);
            }
        }
    }

    #[test]
    fn dct_at_full_range_stays_in_32_bits() {
        for log2_size in 2..=5u32 {
            let size = 1usize << log2_size;
            let values: Vec<i32> = (0..size * size)
                .map(|index| if index & 1 == 0 { 32_767 } else { -32_768 })
                .collect();
            let expected = direct_2d(size, &values, |input| direct_dct(size, input));
            let mut block = block_from(log2_size, &values);
            inverse_dct(log2_size, &mut block);
            assert_eq!(block_rows(log2_size, &block), expected);
        }
    }

    #[test]
    fn dst_equals_the_direct_form_on_random_blocks() {
        let mut rng = Xorshift(0x9e37_79b9_7f4a_7c15);
        for _ in 0..200 {
            let values: Vec<i32> = (0..16).map(|_| rng.coefficient()).collect();
            let expected = direct_2d(4, &values, direct_dst);
            let mut block = block_from(2, &values);
            inverse_dst(&mut block);
            assert_eq!(block_rows(2, &block), expected);
        }
    }

    /// Worked by hand. Column pass: 64 * 64 = 4096 in column 0, and
    /// ( 4096 + 64 ) >> 7 = 32. Row pass: 32 * 64 = 2048 everywhere.
    #[test]
    fn a_dc_coefficient_gives_a_flat_block() {
        for log2_size in 2..=5u32 {
            let size = 1usize << log2_size;
            let mut block = TbBlock::new();
            block.set(log2_size, 0, 0, 64);
            inverse_dct(log2_size, &mut block);
            assert_eq!(block_rows(log2_size, &block), vec![2048; size * size]);
            round_residual(log2_size, 8, &mut block);
            assert_eq!(block_rows(log2_size, &block), vec![1; size * size]);
        }
    }

    /// Worked by hand. A lone first coefficient `c` follows the first DST basis
    /// in both directions: 29, 55, 74, 84. Take `c` = 128. The column pass gives
    /// 29 * 128 = 3712, and ( 3712 + 64 ) >> 7 = 29. The others give 55, 74, 84.
    /// The row pass multiplies by the basis again.
    #[test]
    fn the_first_dst_coefficient_follows_the_first_basis() {
        let mut block = TbBlock::new();
        block.set(2, 0, 0, 128);
        inverse_dst(&mut block);
        let basis = [29, 55, 74, 84];
        for (y, &vertical) in basis.iter().enumerate() {
            for (x, &horizontal) in basis.iter().enumerate() {
                assert_eq!(block.get(2, x, y), vertical * horizontal, "({x}, {y})");
            }
        }
    }

    #[test]
    fn transform_skip_shifts_by_ts_shift() {
        let mut block = TbBlock::new();
        block.set(2, 1, 2, -3);
        block.set(2, 3, 3, 5);
        transform_skip(2, &mut block);
        assert_eq!(block.get(2, 1, 2), -3 << 7);
        assert_eq!(block.get(2, 3, 3), 5 << 7);
        // At 8 bits: ( 640 + 2048 ) >> 12 = 0. At 10 bits, bdShift 10:
        // ( 640 + 512 ) >> 10 = 1, and ( -384 + 512 ) >> 10 = 0.
        let mut ten_bit = block.clone();
        round_residual(2, 10, &mut ten_bit);
        assert_eq!(ten_bit.get(2, 3, 3), 1);
        assert_eq!(ten_bit.get(2, 1, 2), 0);
        round_residual(2, 8, &mut block);
        assert_eq!(block.get(2, 3, 3), 0);
    }

    #[test]
    fn accumulate_adds_along_each_direction() {
        let values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut block = block_from(2, &values);
        accumulate_residual(2, false, &mut block);
        assert_eq!(
            block_rows(2, &block),
            [1, 3, 6, 10, 5, 11, 18, 26, 9, 19, 30, 42, 13, 27, 42, 58]
        );
        let mut block = block_from(2, &values);
        accumulate_residual(2, true, &mut block);
        assert_eq!(
            block_rows(2, &block),
            [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24, 28, 32, 36, 40]
        );
    }

    #[test]
    fn clear_covers_only_the_block() {
        let mut block = TbBlock::new();
        block.set(5, 31, 31, 7);
        block.set(2, 3, 3, 9);
        block.clear(2);
        assert_eq!(block.get(2, 3, 3), 0);
        assert_eq!(block.get(5, 31, 31), 7);
        block.clear(5);
        assert_eq!(block.get(5, 31, 31), 0);
    }
}
