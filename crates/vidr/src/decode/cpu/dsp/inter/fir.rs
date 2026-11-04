//! Sums of tap products over 16-bit lanes, on explicit vectors.
//!
//! At 8 bits the taps of a filter add to at most 88 and at least -24, so a sum fits 16
//! bits and a multiply takes one instruction per vector. Other inputs need 32 bits: a dot
//! product on interleaved pairs of lanes gives each sum, and the result keeps its low 16
//! bits.

use fearless_simd::{Bytes, Simd, i16x8, i16x16};

use crate::decode::cpu::dsp::simd::tap_pair;

/// One output vector of eight lanes, in 16-bit arithmetic. `load( k )` returns the input
/// of tap `k`.
#[inline(always)]
pub(crate) fn sum16_x8<V: Simd, const TAPS: usize>(
    simd: V,
    taps: &[i16; TAPS],
    load: impl Fn(usize) -> i16x8<V>,
) -> i16x8<V> {
    let mut sum = simd.splat_i16x8(0);
    for (k, &tap) in taps.iter().enumerate() {
        sum = simd.add_i16x8(sum, simd.mul_i16x8(load(k), simd.splat_i16x8(tap)));
    }
    sum
}

/// One output vector of sixteen lanes, in 16-bit arithmetic.
#[inline(always)]
pub(crate) fn sum16_x16<V: Simd, const TAPS: usize>(
    simd: V,
    taps: &[i16; TAPS],
    load: impl Fn(usize) -> i16x16<V>,
) -> i16x16<V> {
    let mut sum = simd.splat_i16x16(0);
    for (k, &tap) in taps.iter().enumerate() {
        sum = simd.add_i16x16(sum, simd.mul_i16x16(load(k), simd.splat_i16x16(tap)));
    }
    sum
}

/// One output vector of eight lanes: the low 16 bits of each 32-bit sum, shifted.
#[inline(always)]
pub(crate) fn sum32_x8<V: Simd, const TAPS: usize>(
    simd: V,
    taps: &[i16; TAPS],
    shift: u32,
    load: impl Fn(usize) -> i16x8<V>,
) -> i16x8<V> {
    let mut low = simd.splat_i32x4(0);
    let mut high = simd.splat_i32x4(0);
    for (pair, &[first, second]) in taps.as_chunks::<2>().0.iter().enumerate() {
        let weights: i16x8<V> = simd.splat_i32x4(tap_pair(first, second)).bitcast();
        let (lanes_low, lanes_high) = simd.interleave_i16x8(load(2 * pair), load(2 * pair + 1));
        low = simd.add_i32x4(low, simd.pairwise_dot_product_i16x8(lanes_low, weights));
        high = simd.add_i32x4(high, simd.pairwise_dot_product_i16x8(lanes_high, weights));
    }
    simd.narrow_i32x4(simd.shr_i32x4(low, shift), simd.shr_i32x4(high, shift))
}

/// One output vector of sixteen lanes: the low 16 bits of each 32-bit sum, shifted.
#[inline(always)]
pub(crate) fn sum32_x16<V: Simd, const TAPS: usize>(
    simd: V,
    taps: &[i16; TAPS],
    shift: u32,
    load: impl Fn(usize) -> i16x16<V>,
) -> i16x16<V> {
    let mut low = simd.splat_i32x8(0);
    let mut high = simd.splat_i32x8(0);
    for (pair, &[first, second]) in taps.as_chunks::<2>().0.iter().enumerate() {
        let weights: i16x16<V> = simd.splat_i32x8(tap_pair(first, second)).bitcast();
        let (lanes_low, lanes_high) = simd.interleave_i16x16(load(2 * pair), load(2 * pair + 1));
        low = simd.add_i32x8(low, simd.pairwise_dot_product_i16x16(lanes_low, weights));
        high = simd.add_i32x8(high, simd.pairwise_dot_product_i16x16(lanes_high, weights));
    }
    simd.narrow_i32x8(simd.shr_i32x8(low, shift), simd.shr_i32x8(high, shift))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::cpu::dsp::simd::{
        Cpu, load_i16x8, load_i16x16, store_i16x8, store_i16x16, with_simd,
    };

    const TAPS: [i16; 8] = [-1, 4, -11, 40, 40, -11, 4, -1];

    /// `shift` on the exact sum, and its low 16 bits.
    fn expected(input: &[i16], at: usize, shift: u32) -> i16 {
        let sum: i32 = (0..8)
            .map(|k| i32::from(input[at + k]) * i32::from(TAPS[k]))
            .sum();
        (sum >> shift) as i16
    }

    /// Every sum form equals the formula, in every CPU form.
    #[test]
    fn every_sum_follows_the_formula() {
        let wide: Vec<i16> = (0..64).map(|i| ((i * 37) % 251) as i16).collect();
        let large: Vec<i16> = (0..64)
            .map(|i| ((i * 4099) % 65_536 - 32_768) as i16)
            .collect();
        for cpu in [Cpu::scalar(), Cpu::detected()] {
            let mut out8 = [0i16; 8];
            let mut out16 = [0i16; 16];
            let run16 = |input: &[i16], out8: &mut [i16; 8], out16: &mut [i16; 16]| {
                with_simd!(cpu, simd => {
                    store_i16x8(sum16_x8(simd, &TAPS, #[inline(always)] |k| load_i16x8(simd, input, k)), out8, 0);
                    store_i16x16(sum16_x16(simd, &TAPS, #[inline(always)] |k| load_i16x16(simd, input, k)), out16, 0);
                });
            };
            run16(&wide, &mut out8, &mut out16);
            for (at, &value) in out16.iter().enumerate() {
                assert_eq!(value, expected(&wide, at, 0), "sum16 {cpu:?} lane {at}");
            }
            assert_eq!(&out8[..], &out16[..8]);
            for shift in [0u32, 2, 6] {
                with_simd!(cpu, simd => {
                    store_i16x8(sum32_x8(simd, &TAPS, shift, #[inline(always)] |k| load_i16x8(simd, &large, k)), &mut out8, 0);
                    store_i16x16(sum32_x16(simd, &TAPS, shift, #[inline(always)] |k| load_i16x16(simd, &large, k)), &mut out16, 0);
                });
                for (at, &value) in out16.iter().enumerate() {
                    assert_eq!(
                        value,
                        expected(&large, at, shift),
                        "sum32 {cpu:?} lane {at}"
                    );
                }
                assert_eq!(&out8[..], &out16[..8]);
            }
        }
    }
}
