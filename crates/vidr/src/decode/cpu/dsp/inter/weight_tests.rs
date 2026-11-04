//! Hand-worked values from (8-262) to (8-277), and every form against the formulas.

use super::*;
use crate::decode::cpu::dsp::inter::interp::{InterpScratch, RefBlock, interpolate_luma};
use crate::decode::cpu::dsp::test_rng::Rng;
use crate::decode::cpu::hevc::picture::Plane;

const BLOCK: TargetBlock = TargetBlock {
    x: 1,
    y: 1,
    width: 2,
    height: 1,
};

/// `predSamples` of a block filled with `value`, built by a full-sample read.
fn filled(value: u16, bit_depth: u32) -> PredSamples {
    let mut source = Plane::<u16>::new(8, 8).unwrap();
    for y in 0..8 {
        source.row_mut(y).unwrap().fill(value);
    }
    let mut out = PredSamples::new();
    let mut scratch = Box::new(InterpScratch::new());
    interpolate_luma(
        &source.view(),
        RefBlock {
            x: 0,
            y: 0,
            width: 4,
            height: 4,
        },
        (0, 0),
        bit_depth,
        &mut scratch,
        &mut out,
    );
    out
}

fn first_sample<S: Sample>(plane: &Plane<S>) -> i32 {
    plane.row(1).unwrap()[1].to_i32()
}

/// A value through shift3 and (8-262) returns unchanged, and the neighbour stays.
#[test]
fn default_uni_returns_the_value() {
    let mut plane = Plane::<u8>::new(4, 4).unwrap();
    put_uni(&mut plane.target(), BLOCK, &filled(100, 8), 8);
    assert_eq!(first_sample(&plane), 100);
    assert_eq!(plane.row(1).unwrap()[3], 0);
    assert_eq!(plane.row(0).unwrap()[1], 0);
}

/// (8-264): ( 100 << 6 + 201 << 6 + 64 ) >> 7 = 151.
#[test]
fn default_bi_rounds_the_mean_up() {
    let mut plane = Plane::<u8>::new(4, 4).unwrap();
    let (first, second) = (filled(100, 8), filled(201, 8));
    put_bi(&mut plane.target(), BLOCK, [&first, &second], 8);
    assert_eq!(first_sample(&plane), 151);
}

/// 10 bits: shift1 4. ( 512 << 4 + 8 ) >> 4 = 512.
#[test]
fn default_uni_at_10_bits() {
    let mut plane = Plane::<u16>::new(4, 4).unwrap();
    put_uni(&mut plane.target(), BLOCK, &filled(512, 10), 10);
    assert_eq!(first_sample(&plane), 512);
}

/// (8-275): log2Wd 7 with a denominator of 1 at 8 bits. w0 3, o0 5.
/// ( 6400 * 3 + 64 ) >> 7 = 150, plus 5 is 155.
#[test]
fn explicit_uni_follows_8_275() {
    let mut plane = Plane::<u8>::new(4, 4).unwrap();
    let weights = BlockWeights {
        log2_wd: 7,
        weight: [3, 0],
        offset: [5, 0],
    };
    put_weighted_uni(&mut plane.target(), BLOCK, &filled(100, 8), &weights, 0, 8);
    assert_eq!(first_sample(&plane), 155);
}

/// The second list takes `w1` and `o1`. ( 6400 * -1 + 64 ) >> 7 = -50 at log2Wd
/// 7, plus 60 is 10.
#[test]
fn explicit_uni_reads_the_second_list() {
    let mut plane = Plane::<u8>::new(4, 4).unwrap();
    let weights = BlockWeights {
        log2_wd: 7,
        weight: [0, -1],
        offset: [0, 60],
    };
    put_weighted_uni(&mut plane.target(), BLOCK, &filled(100, 8), &weights, 1, 8);
    assert_eq!(first_sample(&plane), 10);
}

/// (8-277): log2Wd 6. a = 6400, b = 3200, w0 1, w1 3, o0 2, o1 3.
/// ( 6400 + 9600 + ( 6 << 6 ) ) >> 7 = 16384 >> 7 = 128.
#[test]
fn explicit_bi_follows_8_277() {
    let mut plane = Plane::<u8>::new(4, 4).unwrap();
    let weights = BlockWeights {
        log2_wd: 6,
        weight: [1, 3],
        offset: [2, 3],
    };
    let (first, second) = (filled(100, 8), filled(50, 8));
    put_weighted_bi(&mut plane.target(), BLOCK, [&first, &second], &weights, 8);
    assert_eq!(first_sample(&plane), 128);
}

/// Weights of 1 << denominator with no offset equal the default process.
#[test]
fn unit_weights_equal_the_default_process() {
    let weights = BlockWeights {
        log2_wd: 6,
        weight: [1, 1],
        offset: [0, 0],
    };
    for (a, b) in [(100, 201), (3, 4), (255, 254)] {
        let (first, second) = (filled(a, 8), filled(b, 8));
        let mut with_default = Plane::<u8>::new(4, 4).unwrap();
        let mut with_explicit = Plane::<u8>::new(4, 4).unwrap();
        put_bi(&mut with_default.target(), BLOCK, [&first, &second], 8);
        put_weighted_bi(
            &mut with_explicit.target(),
            BLOCK,
            [&first, &second],
            &weights,
            8,
        );
        assert_eq!(first_sample(&with_default), first_sample(&with_explicit));
    }
}

/// The result clips to the bit depth at both ends.
#[test]
fn results_clip() {
    let mut plane = Plane::<u8>::new(4, 4).unwrap();
    let high = BlockWeights {
        log2_wd: 6,
        weight: [4, 0],
        offset: [0, 0],
    };
    put_weighted_uni(&mut plane.target(), BLOCK, &filled(100, 8), &high, 0, 8);
    assert_eq!(first_sample(&plane), 255);
    let low = BlockWeights {
        log2_wd: 6,
        weight: [-4, 0],
        offset: [0, 0],
    };
    put_weighted_uni(&mut plane.target(), BLOCK, &filled(100, 8), &low, 0, 8);
    assert_eq!(first_sample(&plane), 0);
}

/// `predSamples` with values across the whole 16-bit range, the extremes included.
fn random_pred(rng: &mut Rng) -> Box<PredSamples> {
    let mut source = Plane::<u16>::new(64, 64).unwrap();
    for y in 0..64 {
        for sample in source.row_mut(y).unwrap().iter_mut() {
            *sample = match rng.below(6) {
                0 => 0x7fff >> 2,
                1 => 0x8000 >> 2,
                _ => rng.below(0x4000) as u16,
            };
        }
    }
    // Bit depth 12 shifts by 2, so a full 14-bit sample reaches the sign bit.
    let mut out = Box::new(PredSamples::new());
    let mut scratch = Box::new(InterpScratch::new());
    let area = RefBlock {
        x: 0,
        y: 0,
        width: 64,
        height: 64,
    };
    interpolate_luma(&source.view(), area, (0, 0), 12, &mut scratch, &mut out);
    out
}

/// Runs `put` in each form on a fresh plane, and compares the block to `formula`
/// after the clip. `formula` takes the column and row.
fn check_put<S: Sample>(
    name: &str,
    block: TargetBlock,
    bit_depth: u32,
    put: impl Fn(Cpu, &mut TargetPlane<'_, S>),
    formula: impl Fn(usize, usize) -> i32,
) {
    let max_value = (1i32 << bit_depth) - 1;
    for cpu in [Cpu::scalar(), Cpu::detected()] {
        let mut plane = Plane::<S>::new(80, 80).unwrap();
        put(cpu, &mut plane.target());
        for y in 0..block.height {
            let row = &plane.row(block.y + y).unwrap()[block.x..block.x + block.width];
            let found: Vec<i32> = row.iter().map(|sample| sample.to_i32()).collect();
            let expected: Vec<i32> = (0..block.width)
                .map(|x| formula(x, y).clamp(0, max_value))
                .collect();
            assert_eq!(found, expected, "{name} {cpu:?} {block:?} row {y}");
        }
    }
}

/// Every put in every form against its formula in 32 bits, across the block widths.
fn compare_puts<S: Sample>(bit_depth: u32, seed: u64) {
    let mut rng = Rng::new(seed);
    let sizes = [
        (2, 4),
        (4, 8),
        (6, 8),
        (8, 4),
        (12, 16),
        (16, 8),
        (24, 32),
        (32, 8),
        (48, 16),
        (64, 64),
    ];
    for (width, height) in sizes {
        let (first_box, second_box) = (random_pred(&mut rng), random_pred(&mut rng));
        let (first, second): (&PredSamples, &PredSamples) = (&*first_box, &*second_box);
        let block = TargetBlock {
            x: 3,
            y: 2,
            width,
            height,
        };
        let weights = BlockWeights {
            log2_wd: shift1(bit_depth) + rng.below(4) as u32,
            weight: [rng.below(256) as i32 - 128, rng.below(256) as i32 - 128],
            offset: [rng.below(256) as i32 - 128, rng.below(256) as i32 - 128],
        };
        let at = |source: &PredSamples, x: usize, y: usize| i32::from(source.row(y, width)[x]);
        let single_shift = shift1(bit_depth);
        let single_offset = 1i32 << (single_shift - 1);
        let both_shift = single_shift + 1;
        let both_offset = 1i32 << (both_shift - 1);
        let log2_wd = weights.log2_wd;
        let single_round = 1i32 << (log2_wd - 1);
        let both_round = (weights.offset[0] + weights.offset[1] + 1) << log2_wd;
        check_put::<S>(
            "uni",
            block,
            bit_depth,
            |cpu, plane| put_uni_on(cpu, plane, block, first, bit_depth),
            |x, y| (at(first, x, y) + single_offset) >> single_shift,
        );
        check_put::<S>(
            "bi",
            block,
            bit_depth,
            |cpu, plane| put_bi_on(cpu, plane, block, [first, second], bit_depth),
            |x, y| (at(first, x, y) + at(second, x, y) + both_offset) >> both_shift,
        );
        check_put::<S>(
            "weighted uni",
            block,
            bit_depth,
            |cpu, plane| put_weighted_uni_on(cpu, plane, block, second, &weights, 1, bit_depth),
            |x, y| {
                let scaled = at(second, x, y) * weights.weight[1] + single_round;
                (scaled >> log2_wd) + weights.offset[1]
            },
        );
        check_put::<S>(
            "weighted bi",
            block,
            bit_depth,
            |cpu, plane| {
                put_weighted_bi_on(cpu, plane, block, [first, second], &weights, bit_depth);
            },
            |x, y| {
                let sum = at(first, x, y) * weights.weight[0]
                    + at(second, x, y) * weights.weight[1]
                    + both_round;
                sum >> (log2_wd + 1)
            },
        );
    }
}

#[test]
fn every_put_equals_its_formula_at_8_bits() {
    compare_puts::<u8>(8, 0x2bd5_6d21_98a4_11f3);
}

#[test]
fn every_put_equals_its_formula_at_10_bits() {
    compare_puts::<u16>(10, 0x3c6e_f372_fe94_f82b);
}

/// The saturating forms hold at the extreme 16-bit inputs.
#[test]
fn extreme_inputs_clip_exactly() {
    for bit_depth in [8u32, 9, 10, 12] {
        let single_shift = shift1(bit_depth);
        let single_offset = 1i16 << (single_shift - 1);
        let both_shift = single_shift + 1;
        let both_offset = 1i16 << (both_shift - 1);
        let max = (1i32 << bit_depth) - 1;
        let values = [i16::MIN, -1, 0, 1, i16::MAX, i16::MAX - 40, i16::MIN + 40];
        for &a in &values {
            let exact = ((i32::from(a) + i32::from(single_offset)) >> single_shift).clamp(0, max);
            let fast = (a.saturating_add(single_offset) >> single_shift)
                .max(0)
                .min(max as i16);
            assert_eq!(i32::from(fast), exact, "uni {a} at {bit_depth}");
            for &b in &values {
                let exact = ((i32::from(a) + i32::from(b) + i32::from(both_offset)) >> both_shift)
                    .clamp(0, max);
                let sum = a.saturating_add(b).saturating_add(both_offset);
                let fast = (sum >> both_shift).max(0).min(max as i16);
                assert_eq!(i32::from(fast), exact, "bi {a} {b} at {bit_depth}");
            }
        }
    }
}
