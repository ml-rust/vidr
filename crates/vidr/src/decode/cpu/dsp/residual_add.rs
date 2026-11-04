//! Picture construction (8.6.7): the residual added to the prediction.

use fearless_simd::Simd;

use super::sample::Sample;
use super::simd::{clip_x8, clip_x16, load_i32x4, load_i32x8};

/// (8-299) on one row: each sample takes `Clip3( 0, max_value, sample + value )`.
///
/// Residual values saturate to 16 bits, and the addition saturates. The result stays
/// exact, because a saturated value lies past the clip limit on the same side. A row
/// holds 4, 8, 16, or 32 samples: any other tail runs one sample at a time.
#[inline(always)]
pub(crate) fn add_residual_row<V: Simd, S: Sample>(
    simd: V,
    target: &mut [S],
    values: &[i32],
    max_value: i32,
) {
    let max = max_value as i16;
    let (blocks, rest) = target.as_chunks_mut::<16>();
    let (value_blocks, value_rest) = values.as_chunks::<16>();
    for (block, lanes) in blocks.iter_mut().zip(value_blocks) {
        let (low, high) = lanes.split_at(8);
        let residual = simd.saturating_narrow_i32x8(load_i32x8(simd, low), load_i32x8(simd, high));
        let sum = simd.saturating_add_i16x16(S::load_x16(simd, block), residual);
        *block = S::pack_x16(simd, clip_x16(simd, sum, simd.splat_i16x16(max)));
    }
    let (blocks, rest) = rest.as_chunks_mut::<8>();
    let (value_blocks, value_rest) = value_rest.as_chunks::<8>();
    for (block, lanes) in blocks.iter_mut().zip(value_blocks) {
        let (low, high) = lanes.split_at(4);
        let residual = simd.saturating_narrow_i32x4(load_i32x4(simd, low), load_i32x4(simd, high));
        let sum = simd.saturating_add_i16x8(S::load_x8(simd, block), residual);
        *block = S::pack_x8(simd, clip_x8(simd, sum, simd.splat_i16x8(max)));
    }
    if rest.len() == 4 && value_rest.len() == 4 {
        let lanes = load_i32x4(simd, value_rest);
        let residual = simd.saturating_narrow_i32x4(lanes, lanes);
        let mut samples = [S::default(); 8];
        for (sample, &current) in samples.iter_mut().zip(rest.iter()) {
            *sample = current;
        }
        let sum = simd.saturating_add_i16x8(S::load_x8(simd, &samples), residual);
        let packed = S::pack_x8(simd, clip_x8(simd, sum, simd.splat_i16x8(max)));
        for (sample, &lane) in rest.iter_mut().zip(&packed) {
            *sample = lane;
        }
    } else {
        for (sample, &value) in rest.iter_mut().zip(value_rest) {
            *sample = S::clipped(sample.to_i32() + value, max_value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::cpu::dsp::simd::{Cpu, with_simd};
    use crate::decode::cpu::dsp::test_rng::Rng;

    fn compare<S: Sample>(bit_depth: u32, make: fn(i32) -> S, seed: u64) {
        let mut rng = Rng::new(seed);
        let max_value = (1i32 << bit_depth) - 1;
        for size in [1usize, 2, 3, 4, 5, 8, 12, 16, 20, 24, 32] {
            for _ in 0..50 {
                let start: Vec<S> = (0..size)
                    .map(|_| make(rng.sample(128, bit_depth)))
                    .collect();
                let values: Vec<i32> = (0..size)
                    .map(|_| match rng.below(4) {
                        0 => rng.below(1 << 20) as i32 - (1 << 19),
                        _ => rng.below(4096) as i32 - 2048,
                    })
                    .collect();
                let expected: Vec<S> = start
                    .iter()
                    .zip(&values)
                    .map(|(&sample, &value)| make((sample.to_i32() + value).clamp(0, max_value)))
                    .collect();
                for cpu in [Cpu::scalar(), Cpu::detected()] {
                    let mut out = start.clone();
                    with_simd!(cpu, simd => add_residual_row(simd, &mut out, &values, max_value));
                    assert_eq!(out, expected, "{cpu:?} size {size}");
                }
            }
        }
    }

    #[test]
    fn every_form_equals_the_formula_at_8_bits() {
        compare::<u8>(8, |value| value as u8, 0x1234_5678_9abc_def1);
    }

    #[test]
    fn every_form_equals_the_formula_at_10_bits() {
        compare::<u16>(10, |value| value as u16, 0x0fed_cba9_8765_4321);
    }
}
