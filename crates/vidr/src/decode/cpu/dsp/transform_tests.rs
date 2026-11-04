//! Every form of the DCT, the DST, and the rounding against the formulas of 8.6.2 to 8.6.5.

use super::*;
use crate::decode::cpu::dsp::test_rng::Rng;

/// The DCT with no knowledge of where the levels lie.
fn full_dct(log2_size: u32, block: &mut TbBlock) {
    let last = (1usize << log2_size) - 1;
    let extent = CoeffExtent {
        max_x: last,
        max_y: last,
    };
    inverse_dct(log2_size, block, extent);
}

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

/// A coefficient: often zero, sometimes small, sometimes at full range.
fn coefficient(rng: &mut Rng) -> i32 {
    let value = rng.next();
    match value % 4 {
        0 | 1 => 0,
        2 => ((value >> 8) % 201) as i32 - 100,
        _ => ((value >> 8) % 65_536) as i32 - 32_768,
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
    let mut rng = Rng::new(0x2545_f491_4f6c_dd1d);
    for log2_size in 2..=5u32 {
        let size = 1usize << log2_size;
        for _ in 0..40 {
            let values: Vec<i32> = (0..size * size).map(|_| coefficient(&mut rng)).collect();
            let expected = direct_2d(size, &values, |input| direct_dct(size, input));
            let mut block = block_from(log2_size, &values);
            full_dct(log2_size, &mut block);
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
            full_dct(log2_size, &mut block);
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
        full_dct(log2_size, &mut block);
        assert_eq!(block_rows(log2_size, &block), expected);
    }
}

#[test]
fn dst_equals_the_direct_form_on_random_blocks() {
    let mut rng = Rng::new(0x9e37_79b9_7f4a_7c15);
    for _ in 0..200 {
        let values: Vec<i32> = (0..16).map(|_| coefficient(&mut rng)).collect();
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
        full_dct(log2_size, &mut block);
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

/// Every form, at each extent, equals the direct form. A stale block between calls
/// leaves no trace.
#[test]
fn dct_inside_an_extent_equals_the_direct_form_in_every_form() {
    let mut rng = Rng::new(0xd1b5_4a32_d192_ed03);
    let mut block = TbBlock::new();
    for log2_size in 2..=5u32 {
        let size = 1usize << log2_size;
        for _ in 0..120 {
            let max_x = rng.below(size as u64) as usize;
            let max_y = rng.below(size as u64) as usize;
            let mut values = vec![0; size * size];
            for y in 0..=max_y {
                for x in 0..=max_x {
                    values[y * size + x] = coefficient(&mut rng);
                }
            }
            let expected = direct_2d(size, &values, |input| direct_dct(size, input));
            let extent = CoeffExtent { max_x, max_y };
            for cpu in [Cpu::scalar(), Cpu::detected()] {
                block.clear(log2_size);
                for (index, &value) in values.iter().enumerate() {
                    block.set(log2_size, index % size, index / size, value);
                }
                inverse_dct_on(cpu, log2_size, &mut block, extent);
                assert_eq!(
                    block_rows(log2_size, &block),
                    expected,
                    "{cpu:?} size {size} extent {extent:?}"
                );
            }
        }
    }
}

/// The lone level path equals the direct form across the clip of (8-314).
#[test]
fn a_lone_dc_level_equals_the_direct_form() {
    let mut block = TbBlock::new();
    for log2_size in 2..=5u32 {
        let size = 1usize << log2_size;
        for level in [0, 1, -1, 63, 64, -65, 32_767, -32_768] {
            let mut values = vec![0; size * size];
            values[0] = level;
            let expected = direct_2d(size, &values, |input| direct_dct(size, input));
            block.clear(log2_size);
            block.set(log2_size, 0, 0, level);
            let extent = CoeffExtent { max_x: 0, max_y: 0 };
            inverse_dct(log2_size, &mut block, extent);
            assert_eq!(block_rows(log2_size, &block), expected, "level {level}");
        }
    }
}

/// Every form of the rounding equals (8-299), at 8 and 10 bits.
#[test]
fn rounding_equals_the_formula_in_every_form() {
    let mut rng = Rng::new(0xa076_1d64_78bd_642f);
    let mut block = TbBlock::new();
    for log2_size in 2..=5u32 {
        let size = 1usize << log2_size;
        for bit_depth in [8u32, 10] {
            let values: Vec<i32> = (0..size * size)
                .map(|_| rng.below(1 << 24) as i32 - (1 << 23))
                .collect();
            let bd_shift = 20 - bit_depth;
            let expected: Vec<i32> = values
                .iter()
                .map(|value| (value + (1 << (bd_shift - 1))) >> bd_shift)
                .collect();
            for cpu in [Cpu::scalar(), Cpu::detected()] {
                for (index, &value) in values.iter().enumerate() {
                    block.set(log2_size, index % size, index / size, value);
                }
                round_residual_on(cpu, log2_size, bit_depth, &mut block);
                assert_eq!(block_rows(log2_size, &block), expected, "{cpu:?}");
            }
        }
    }
}
