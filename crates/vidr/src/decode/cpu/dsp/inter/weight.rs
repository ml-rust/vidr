//! Weighted sample prediction (8.5.3.3.4): 14-bit intermediate values to samples.

use crate::decode::cpu::dsp::sample::Sample;

use super::interp::PredSamples;

/// A plane of the current picture as the kernels write it.
#[derive(Debug)]
pub(crate) struct TargetPlane<'a, S> {
    /// Every sample, stride gaps included. Row `y` starts at `y * stride`.
    pub(crate) samples: &'a mut [S],
    /// Samples from one row start to the next.
    pub(crate) stride: usize,
}

/// Where a block lands in the current plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TargetBlock {
    /// Left edge in plane samples.
    pub(crate) x: usize,
    /// Top edge in plane samples.
    pub(crate) y: usize,
    /// Width in plane samples.
    pub(crate) width: usize,
    /// Height in plane samples.
    pub(crate) height: usize,
}

/// `log2Wd`, `w0`, `w1`, `o0`, and `o1` of one block and component (8-265) to (8-274).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlockWeights {
    /// `log2Wd`.
    pub(crate) log2_wd: u32,
    /// `w0` and `w1`.
    pub(crate) weight: [i32; 2],
    /// `o0` and `o1`.
    pub(crate) offset: [i32; 2],
}

/// `shift1` of 8.5.3.3.4.2 and 8.5.3.3.4.3.
pub(crate) const fn shift1(bit_depth: u32) -> u32 {
    let shift = 14u32.saturating_sub(bit_depth);
    if shift > 2 { shift } else { 2 }
}

/// Runs `write_row` on each target row, with its index in the block.
fn write_rows<S: Sample>(
    plane: &mut TargetPlane<'_, S>,
    block: TargetBlock,
    mut write_row: impl FnMut(usize, &mut [S]),
) {
    for line in 0..block.height {
        let start = (block.y + line) * plane.stride + block.x;
        if let Some(target) = plane.samples.get_mut(start..start + block.width) {
            write_row(line, target);
        }
    }
}

/// (8-262) and (8-263): one list.
pub(crate) fn put_uni<S: Sample>(
    plane: &mut TargetPlane<'_, S>,
    block: TargetBlock,
    source: &PredSamples,
    bit_depth: u32,
) {
    let shift = shift1(bit_depth);
    let offset = 1i32 << (shift - 1);
    let max_value = (1i32 << bit_depth) - 1;
    write_rows(plane, block, |line, target| {
        for (out, &value) in target.iter_mut().zip(source.row(line, block.width)) {
            *out = S::clipped((i32::from(value) + offset) >> shift, max_value);
        }
    });
}

/// (8-264): two lists.
pub(crate) fn put_bi<S: Sample>(
    plane: &mut TargetPlane<'_, S>,
    block: TargetBlock,
    sources: [&PredSamples; 2],
    bit_depth: u32,
) {
    let shift = shift1(bit_depth) + 1;
    let offset = 1i32 << (shift - 1);
    let max_value = (1i32 << bit_depth) - 1;
    write_rows(plane, block, |line, target| {
        let first = sources[0].row(line, block.width);
        let second = sources[1].row(line, block.width);
        for ((out, &a), &b) in target.iter_mut().zip(first).zip(second) {
            *out = S::clipped((i32::from(a) + i32::from(b) + offset) >> shift, max_value);
        }
    });
}

/// (8-275) and (8-276): one list. `list` selects `w0` and `o0`, or `w1` and `o1`.
pub(crate) fn put_weighted_uni<S: Sample>(
    plane: &mut TargetPlane<'_, S>,
    block: TargetBlock,
    source: &PredSamples,
    weights: &BlockWeights,
    list: usize,
    bit_depth: u32,
) {
    let (weight, offset) = (weights.weight[list], weights.offset[list]);
    let log2_wd = weights.log2_wd;
    let round = 1i32 << (log2_wd - 1);
    let max_value = (1i32 << bit_depth) - 1;
    write_rows(plane, block, |line, target| {
        for (out, &value) in target.iter_mut().zip(source.row(line, block.width)) {
            let scaled = (i32::from(value) * weight + round) >> log2_wd;
            *out = S::clipped(scaled + offset, max_value);
        }
    });
}

/// (8-277): two lists.
pub(crate) fn put_weighted_bi<S: Sample>(
    plane: &mut TargetPlane<'_, S>,
    block: TargetBlock,
    sources: [&PredSamples; 2],
    weights: &BlockWeights,
    bit_depth: u32,
) {
    let [w0, w1] = weights.weight;
    let log2_wd = weights.log2_wd;
    let round = (weights.offset[0] + weights.offset[1] + 1) << log2_wd;
    let max_value = (1i32 << bit_depth) - 1;
    write_rows(plane, block, |line, target| {
        let first = sources[0].row(line, block.width);
        let second = sources[1].row(line, block.width);
        for ((out, &a), &b) in target.iter_mut().zip(first).zip(second) {
            let sum = i32::from(a) * w0 + i32::from(b) * w1 + round;
            *out = S::clipped(sum >> (log2_wd + 1), max_value);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::cpu::dsp::inter::interp::{InterpScratch, RefBlock, interpolate_luma};
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
}
