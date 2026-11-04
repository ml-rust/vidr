//! Weighted sample prediction (8.5.3.3.4): 14-bit intermediate values to samples.

use fearless_simd::{Bytes, Simd, i16x8, i16x16};

use crate::decode::cpu::dsp::sample::Sample;
use crate::decode::cpu::dsp::simd::{
    Cpu, clip_x8, clip_x16, load_i16x8, load_i16x16, tap_pair, with_simd,
};

use super::interp::{MAX_BLOCK, PredSamples};

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

/// Writes one row of samples in vectors of 16, then 8, then the rest from one vector of
/// 8. `x8` and `x16` return the lanes at a column, a multiple of 8 below 64.
#[inline(always)]
fn write_row<V: Simd, S: Sample>(
    simd: V,
    target: &mut [S],
    x8: impl Fn(usize) -> i16x8<V>,
    x16: impl Fn(usize) -> i16x16<V>,
) {
    let (blocks, rest) = target.as_chunks_mut::<16>();
    let mut x = 0;
    for block in blocks {
        *block = S::pack_x16(simd, x16(x & 48));
        x += 16;
    }
    write_row_x8(simd, rest, x, x8);
}

/// [`write_row`] from column `x`, in vectors of 8 only.
#[inline(always)]
fn write_row_x8<V: Simd, S: Sample>(
    simd: V,
    target: &mut [S],
    start: usize,
    x8: impl Fn(usize) -> i16x8<V>,
) {
    let (blocks, rest) = target.as_chunks_mut::<8>();
    let mut x = start;
    for block in blocks {
        *block = S::pack_x8(simd, x8(x & 56));
        x += 8;
    }
    if !rest.is_empty() {
        let lanes = S::pack_x8(simd, x8(x & 56));
        for (sample, &lane) in rest.iter_mut().zip(&lanes) {
            *sample = lane;
        }
    }
}

/// Runs `write` on each target row, with the source row of the same index. A block wider
/// than [`MAX_BLOCK`] writes nothing.
#[inline(always)]
fn for_each_row<S: Sample>(
    plane: &mut TargetPlane<'_, S>,
    block: TargetBlock,
    mut write: impl FnMut(usize, &mut [S]),
) {
    if block.width > MAX_BLOCK {
        return;
    }
    for line in 0..block.height {
        let start = (block.y + line) * plane.stride + block.x;
        if let Some(target) = plane.samples.get_mut(start..start + block.width) {
            write(line, target);
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
    put_uni_on(Cpu::detected(), plane, block, source, bit_depth);
}

/// [`put_uni`] compiled for `cpu`. The rounding addition saturates at 16 bits, and the
/// result stays exact: a saturated sum shifts to a value at or above the clip limit.
fn put_uni_on<S: Sample>(
    cpu: Cpu,
    plane: &mut TargetPlane<'_, S>,
    block: TargetBlock,
    source: &PredSamples,
    bit_depth: u32,
) {
    let shift = shift1(bit_depth);
    let offset = 1i16 << (shift - 1);
    let max_value = ((1i32 << bit_depth) - 1) as i16;
    with_simd!(cpu, simd => {
        for_each_row(
            plane,
            block,
            #[inline(always)]
            |line, target: &mut [S]| {
                let Some(row) = source.full_row(line) else {
                    return;
                };
                write_row(
                    simd,
                    target,
                    #[inline(always)]
                    |x| {
                        let sum = simd.saturating_add_i16x8(
                            load_i16x8(simd, row, x),
                            simd.splat_i16x8(offset),
                        );
                        clip_x8(simd, simd.shr_i16x8(sum, shift), simd.splat_i16x8(max_value))
                    },
                    #[inline(always)]
                    |x| {
                        let sum = simd.saturating_add_i16x16(
                            load_i16x16(simd, row, x),
                            simd.splat_i16x16(offset),
                        );
                        clip_x16(simd, simd.shr_i16x16(sum, shift), simd.splat_i16x16(max_value))
                    },
                );
            },
        );
    });
}

/// (8-264): two lists.
pub(crate) fn put_bi<S: Sample>(
    plane: &mut TargetPlane<'_, S>,
    block: TargetBlock,
    sources: [&PredSamples; 2],
    bit_depth: u32,
) {
    put_bi_on(Cpu::detected(), plane, block, sources, bit_depth);
}

/// [`put_bi`] compiled for `cpu`. Each addition saturates at 16 bits, and the result
/// stays exact: a sum saturated high shifts to a value at or above the clip limit, and a
/// sum saturated low stays negative after the rounding offset.
fn put_bi_on<S: Sample>(
    cpu: Cpu,
    plane: &mut TargetPlane<'_, S>,
    block: TargetBlock,
    sources: [&PredSamples; 2],
    bit_depth: u32,
) {
    let shift = shift1(bit_depth) + 1;
    let offset = 1i16 << (shift - 1);
    let max_value = ((1i32 << bit_depth) - 1) as i16;
    with_simd!(cpu, simd => {
        for_each_row(
            plane,
            block,
            #[inline(always)]
            |line, target: &mut [S]| {
                let (Some(first), Some(second)) =
                    (sources[0].full_row(line), sources[1].full_row(line))
                else {
                    return;
                };
                write_row(
                    simd,
                    target,
                    #[inline(always)]
                    |x| {
                        let pair = simd.saturating_add_i16x8(
                            load_i16x8(simd, first, x),
                            load_i16x8(simd, second, x),
                        );
                        let sum = simd.saturating_add_i16x8(pair, simd.splat_i16x8(offset));
                        clip_x8(simd, simd.shr_i16x8(sum, shift), simd.splat_i16x8(max_value))
                    },
                    #[inline(always)]
                    |x| {
                        let pair = simd.saturating_add_i16x16(
                            load_i16x16(simd, first, x),
                            load_i16x16(simd, second, x),
                        );
                        let sum = simd.saturating_add_i16x16(pair, simd.splat_i16x16(offset));
                        clip_x16(simd, simd.shr_i16x16(sum, shift), simd.splat_i16x16(max_value))
                    },
                );
            },
        );
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
    put_weighted_uni_on(
        Cpu::detected(),
        plane,
        block,
        source,
        weights,
        list,
        bit_depth,
    );
}

/// [`put_weighted_uni`] compiled for `cpu`. A dot product on the pair of a lane and 1
/// gives `value * weight + round`. `round` is at most `1 << 12`, and a weight is at most
/// 255 (7.4.7.3), so both fit 16 bits. Saturating to 16 bits before the clip keeps the
/// result.
fn put_weighted_uni_on<S: Sample>(
    cpu: Cpu,
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
    let pair = tap_pair(weight as i16, round as i16);
    let max_value = ((1i32 << bit_depth) - 1) as i16;
    with_simd!(cpu, simd => {
        for_each_row(
            plane,
            block,
            #[inline(always)]
            |line, target: &mut [S]| {
                let Some(row) = source.full_row(line) else {
                    return;
                };
                write_row_x8(
                    simd,
                    target,
                    0,
                    #[inline(always)]
                    |x| {
                        let pairs = weight_pairs(simd, pair);
                        let (low, high) =
                            simd.interleave_i16x8(load_i16x8(simd, row, x), simd.splat_i16x8(1));
                        let low = simd.pairwise_dot_product_i16x8(low, pairs);
                        let high = simd.pairwise_dot_product_i16x8(high, pairs);
                        let offsets = simd.splat_i32x4(offset);
                        let low = simd.add_i32x4(simd.shr_i32x4(low, log2_wd), offsets);
                        let high = simd.add_i32x4(simd.shr_i32x4(high, log2_wd), offsets);
                        clip_x8(
                            simd,
                            simd.saturating_narrow_i32x4(low, high),
                            simd.splat_i16x8(max_value),
                        )
                    },
                );
            },
        );
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
    put_weighted_bi_on(Cpu::detected(), plane, block, sources, weights, bit_depth);
}

/// [`put_weighted_bi`] compiled for `cpu`. A dot product on the two lists gives the
/// sum of products. A weight is at most 255 (7.4.7.3), so it fits 16 bits.
fn put_weighted_bi_on<S: Sample>(
    cpu: Cpu,
    plane: &mut TargetPlane<'_, S>,
    block: TargetBlock,
    sources: [&PredSamples; 2],
    weights: &BlockWeights,
    bit_depth: u32,
) {
    let [w0, w1] = weights.weight;
    let pair = tap_pair(w0 as i16, w1 as i16);
    let shift = weights.log2_wd + 1;
    let round = (weights.offset[0] + weights.offset[1] + 1) << weights.log2_wd;
    let max_value = ((1i32 << bit_depth) - 1) as i16;
    with_simd!(cpu, simd => {
        for_each_row(
            plane,
            block,
            #[inline(always)]
            |line, target: &mut [S]| {
                let (Some(first), Some(second)) =
                    (sources[0].full_row(line), sources[1].full_row(line))
                else {
                    return;
                };
                write_row_x8(
                    simd,
                    target,
                    0,
                    #[inline(always)]
                    |x| {
                        let pairs = weight_pairs(simd, pair);
                        let (low, high) = simd.interleave_i16x8(
                            load_i16x8(simd, first, x),
                            load_i16x8(simd, second, x),
                        );
                        let rounds = simd.splat_i32x4(round);
                        let low = simd.add_i32x4(simd.pairwise_dot_product_i16x8(low, pairs), rounds);
                        let high = simd.add_i32x4(simd.pairwise_dot_product_i16x8(high, pairs), rounds);
                        clip_x8(
                            simd,
                            simd.saturating_narrow_i32x4(
                                simd.shr_i32x4(low, shift),
                                simd.shr_i32x4(high, shift),
                            ),
                            simd.splat_i16x8(max_value),
                        )
                    },
                );
            },
        );
    });
}

/// A splat of two 16-bit weights, as the second operand of a dot product on pairs.
#[inline(always)]
fn weight_pairs<V: Simd>(simd: V, pair: i32) -> i16x8<V> {
    simd.splat_i32x4(pair).bitcast()
}

#[cfg(test)]
#[path = "weight_tests.rs"]
mod tests;
