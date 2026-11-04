//! Sample adaptive offset kernels (8.7.3.2) over runs of samples, on 16-bit lanes. A byte
//! shuffle selects each offset from a table of 16-bit values. A run ends with a scalar
//! tail. The caller runs the kernels through `with_simd`.

use fearless_simd::{Bytes, Simd, SimdFrom, i16x8, i16x16};

use super::sample::Sample;
use super::simd::{clip_x8, clip_x16};

/// `edgeIdx` after (8-412), by `2 + Sign( ) + Sign( )` of (8-411).
const EDGE_INDEX: [usize; 5] = [1, 2, 0, 3, 4];
/// The byte pair of table entry 0, as one little-endian lane.
const FIRST_ENTRY_BYTES: i16 = 0x0100;
/// The step from the byte pair of one table entry to the next.
const NEXT_ENTRY_BYTES: i16 = 0x0202;

/// `table[ index ]` in each lane. `index` runs from 0 to 7.
#[inline(always)]
fn lookup_x8<V: Simd>(simd: V, table: i16x8<V>, index: i16x8<V>) -> i16x8<V> {
    let pairs = simd.add_i16x8(
        simd.mul_i16x8(index, simd.splat_i16x8(NEXT_ENTRY_BYTES)),
        simd.splat_i16x8(FIRST_ENTRY_BYTES),
    );
    simd.swizzle_dyn_within_blocks_i16x8(table, pairs.bitcast())
}

/// `table[ index ]` in each lane, with the same table in both 128-bit blocks.
#[inline(always)]
fn lookup_x16<V: Simd>(simd: V, table: i16x16<V>, index: i16x16<V>) -> i16x16<V> {
    let pairs = simd.add_i16x16(
        simd.mul_i16x16(index, simd.splat_i16x16(NEXT_ENTRY_BYTES)),
        simd.splat_i16x16(FIRST_ENTRY_BYTES),
    );
    simd.swizzle_dyn_within_blocks_i16x16(table, pairs.bitcast())
}

/// `Sign( a - b )` (5.8) in each lane. Samples fit in 15 bits, so the difference fits.
#[inline(always)]
fn sign_x8<V: Simd>(simd: V, a: i16x8<V>, b: i16x8<V>) -> i16x8<V> {
    let low = simd.max_i16x8(simd.sub_i16x8(a, b), simd.splat_i16x8(-1));
    simd.min_i16x8(low, simd.splat_i16x8(1))
}

/// `Sign( a - b )` (5.8) in each lane.
#[inline(always)]
fn sign_x16<V: Simd>(simd: V, a: i16x16<V>, b: i16x16<V>) -> i16x16<V> {
    let low = simd.max_i16x16(simd.sub_i16x16(a, b), simd.splat_i16x16(-1));
    simd.min_i16x16(low, simd.splat_i16x16(1))
}

/// (8-414) and (8-415) on one run. `position` is `sao_band_position`, and `offsets` holds
/// `SaoOffsetVal`. Four bands from `position` take an offset, wrapping at 32. `band_shift`
/// is `bitDepth - 5`.
#[inline(always)]
pub(crate) fn band_run<V: Simd, S: Sample>(
    simd: V,
    source: &[S],
    target: &mut [S],
    position: u8,
    offsets: &[i16; 5],
    band_shift: u32,
    max_value: i32,
) {
    let [_, a, b, c, d] = *offsets;
    let position = i16::from(position);
    let max = max_value as i16;
    let table = i16x8::simd_from(simd, [a, b, c, d, 0, 0, 0, 0]);
    let wide_table = simd.combine_i16x8(table, table);
    let count = source.len().min(target.len());
    let (blocks, rest) = target[..count].as_chunks_mut::<16>();
    let (inputs, input_rest) = source[..count].as_chunks::<16>();
    for (out, input) in blocks.iter_mut().zip(inputs) {
        let value = S::load_x16(simd, input);
        let band = simd.shr_i16x16(value, band_shift);
        let slot = simd.and_i16x16(
            simd.sub_i16x16(band, simd.splat_i16x16(position)),
            simd.splat_i16x16(31),
        );
        let index = simd.min_i16x16(slot, simd.splat_i16x16(4));
        let sum = simd.add_i16x16(value, lookup_x16(simd, wide_table, index));
        *out = S::pack_x16(simd, clip_x16(simd, sum, simd.splat_i16x16(max)));
    }
    let (blocks, rest) = rest.as_chunks_mut::<8>();
    let (inputs, input_rest) = input_rest.as_chunks::<8>();
    for (out, input) in blocks.iter_mut().zip(inputs) {
        let value = S::load_x8(simd, input);
        let band = simd.shr_i16x8(value, band_shift);
        let slot = simd.and_i16x8(
            simd.sub_i16x8(band, simd.splat_i16x8(position)),
            simd.splat_i16x8(31),
        );
        let index = simd.min_i16x8(slot, simd.splat_i16x8(4));
        let sum = simd.add_i16x8(value, lookup_x8(simd, table, index));
        *out = S::pack_x8(simd, clip_x8(simd, sum, simd.splat_i16x8(max)));
    }
    let scalar = [a, b, c, d, 0];
    for (out, &sample) in rest.iter_mut().zip(input_rest) {
        let value = sample.to_i32();
        let slot = ((value >> band_shift) - i32::from(position)) & 31;
        let offset = scalar[slot.min(4) as usize];
        *out = S::clipped(value + i32::from(offset), max_value);
    }
}

/// (8-411) to (8-413) on one run. `neighbours` start at the two neighbours of the first
/// sample.
#[inline(always)]
pub(crate) fn edge_run<V: Simd, S: Sample>(
    simd: V,
    source: &[S],
    neighbours: (&[S], &[S]),
    target: &mut [S],
    offsets: &[i16; 5],
    max_value: i32,
) {
    let (first, second) = neighbours;
    let scalar = EDGE_INDEX.map(|index| offsets[index]);
    let max = max_value as i16;
    let [t0, t1, t2, t3, t4] = scalar;
    let table = i16x8::simd_from(simd, [t0, t1, t2, t3, t4, 0, 0, 0]);
    let wide_table = simd.combine_i16x8(table, table);
    let count = source
        .len()
        .min(first.len())
        .min(second.len())
        .min(target.len());
    let (blocks, rest) = target[..count].as_chunks_mut::<16>();
    let (sources, source_rest) = source[..count].as_chunks::<16>();
    let (firsts, first_rest) = first[..count].as_chunks::<16>();
    let (seconds, second_rest) = second[..count].as_chunks::<16>();
    let inputs = sources.iter().zip(firsts).zip(seconds);
    for (out, ((input, a), b)) in blocks.iter_mut().zip(inputs) {
        let value = S::load_x16(simd, input);
        let shape = simd.add_i16x16(
            sign_x16(simd, value, S::load_x16(simd, a)),
            sign_x16(simd, value, S::load_x16(simd, b)),
        );
        let index = simd.add_i16x16(shape, simd.splat_i16x16(2));
        let sum = simd.add_i16x16(value, lookup_x16(simd, wide_table, index));
        *out = S::pack_x16(simd, clip_x16(simd, sum, simd.splat_i16x16(max)));
    }
    let (blocks, rest) = rest.as_chunks_mut::<8>();
    let (sources, source_rest) = source_rest.as_chunks::<8>();
    let (firsts, first_rest) = first_rest.as_chunks::<8>();
    let (seconds, second_rest) = second_rest.as_chunks::<8>();
    let inputs = sources.iter().zip(firsts).zip(seconds);
    for (out, ((input, a), b)) in blocks.iter_mut().zip(inputs) {
        let value = S::load_x8(simd, input);
        let shape = simd.add_i16x8(
            sign_x8(simd, value, S::load_x8(simd, a)),
            sign_x8(simd, value, S::load_x8(simd, b)),
        );
        let index = simd.add_i16x8(shape, simd.splat_i16x8(2));
        let sum = simd.add_i16x8(value, lookup_x8(simd, table, index));
        *out = S::pack_x8(simd, clip_x8(simd, sum, simd.splat_i16x8(max)));
    }
    let tail = source_rest.iter().zip(first_rest).zip(second_rest);
    for (out, ((&sample, &a), &b)) in rest.iter_mut().zip(tail) {
        let value = sample.to_i32();
        let index = 2 + (value - a.to_i32()).signum() + (value - b.to_i32()).signum();
        *out = S::clipped(value + i32::from(scalar[index as usize]), max_value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::cpu::dsp::simd::{Cpu, with_simd};
    use crate::decode::cpu::dsp::test_rng::Rng;

    /// [`band_run`] on the CPU's best form.
    fn apply_band<S: Sample>(
        source: &[S],
        target: &mut [S],
        position: u8,
        offsets: &[i16; 5],
        band_shift: u32,
        max_value: i32,
    ) {
        with_simd!(Cpu::detected(), simd => {
            band_run(simd, source, target, position, offsets, band_shift, max_value);
        });
    }

    /// [`edge_run`] on the CPU's best form.
    fn apply_edge<S: Sample>(
        source: &[S],
        first: &[S],
        second: &[S],
        target: &mut [S],
        offsets: &[i16; 5],
        max_value: i32,
    ) {
        with_simd!(Cpu::detected(), simd => {
            edge_run(simd, source, (first, second), target, offsets, max_value);
        });
    }

    /// `SaoOffsetVal` by band, from `bandTable` of (8-414).
    fn band_table(position: u8, offsets: &[i16; 5]) -> [i16; 32] {
        let mut table = [0i16; 32];
        for k in 0..4usize {
            table[(k + usize::from(position)) & 31] = offsets[k + 1];
        }
        table
    }

    /// (8-415) with the table of (8-414), one sample at a time in 32 bits.
    fn reference_band<S: Sample>(
        source: &[S],
        target: &mut [S],
        table: &[i16; 32],
        band_shift: u32,
        max_value: i32,
    ) {
        for (out, &sample) in target.iter_mut().zip(source) {
            let value = sample.to_i32();
            let band = (value >> band_shift) as usize & 31;
            *out = S::clipped(value + i32::from(table[band]), max_value);
        }
    }

    /// (8-411) to (8-413), one sample at a time in 32 bits.
    fn reference_edge<S: Sample>(
        source: &[S],
        first: &[S],
        second: &[S],
        target: &mut [S],
        offsets: &[i16; 5],
        max_value: i32,
    ) {
        let table = EDGE_INDEX.map(|index| i32::from(offsets[index]));
        let samples = source.iter().zip(first).zip(second);
        for (out, ((&sample, &a), &b)) in target.iter_mut().zip(samples) {
            let value = sample.to_i32();
            let index = 2 + (value - a.to_i32()).signum() + (value - b.to_i32()).signum();
            *out = S::clipped(value + table[index as usize], max_value);
        }
    }

    fn compare_kernels<S: Sample>(bit_depth: u32, make: fn(i32) -> S, seed: u64) {
        let mut rng = Rng::new(seed);
        let max_value = (1i32 << bit_depth) - 1;
        let limit = (31i64 << (bit_depth - 8).min(2)) as u64;
        for length in [1usize, 2, 3, 4, 7, 8, 14, 15, 16, 17, 30, 31, 32, 62, 64] {
            for _ in 0..40 {
                let around = rng.below(max_value as u64 + 1) as i32;
                let mut run = || -> Vec<S> {
                    (0..length)
                        .map(|_| make(rng.sample(around, bit_depth)))
                        .collect()
                };
                let (source, first, second) = (run(), run(), run());
                let mut offset = || (rng.below(2 * limit + 1) as i64 - limit as i64) as i16;
                let offsets = [0, offset(), offset(), offset(), offset()];
                let position = rng.below(32) as u8;
                let table = band_table(position, &offsets);
                let band_shift = bit_depth - 5;
                let mut band = vec![S::default(); length];
                reference_band(&source, &mut band, &table, band_shift, max_value);
                let mut edge = vec![S::default(); length];
                reference_edge(&source, &first, &second, &mut edge, &offsets, max_value);
                for cpu in [Cpu::scalar(), Cpu::detected()] {
                    let mut out = vec![S::default(); length];
                    with_simd!(cpu, simd => {
                        band_run(simd, &source, &mut out, position, &offsets, band_shift, max_value);
                    });
                    assert_eq!(out, band, "band {cpu:?} length {length}");
                    let mut out = vec![S::default(); length];
                    with_simd!(cpu, simd => {
                        edge_run(simd, &source, (&first[..], &second[..]), &mut out, &offsets, max_value);
                    });
                    assert_eq!(out, edge, "edge {cpu:?} length {length}");
                }
            }
        }
    }

    #[test]
    fn every_form_equals_the_formulas_at_8_bits() {
        compare_kernels::<u8>(8, |value| value as u8, 0x9e37_79b1_85eb_ca87);
    }

    #[test]
    fn every_form_equals_the_formulas_at_10_bits() {
        compare_kernels::<u16>(10, |value| value as u16, 0xc2b2_ae3d_27d4_eb4f);
    }

    /// (8-414): four bands from the position, wrapping at 32.
    #[test]
    fn band_tables_wrap() {
        let offsets = [0, 1, 2, 3, 4];
        let table = band_table(30, &offsets);
        assert_eq!(table[30], 1);
        assert_eq!(table[31], 2);
        assert_eq!(table[0], 3);
        assert_eq!(table[1], 4);
        assert_eq!(table[2], 0);
        assert_eq!(table[29], 0);
    }

    #[test]
    fn band_offsets_select_by_the_top_five_bits() {
        // At 8 bits, a band spans 8 values: band 2 starts at 16.
        let source = [15u8, 16, 24, 32, 40, 48, 2];
        let mut target = [0u8; 7];
        apply_band(&source, &mut target, 2, &[0, 5, -5, 7, -7], 3, 255);
        assert_eq!(target, [15, 21, 19, 39, 33, 48, 2]);
    }

    #[test]
    fn band_offsets_clip() {
        let source = [250u8];
        let mut target = [0u8];
        apply_band(&source, &mut target, 31, &[0, 9, 0, 0, 0], 3, 255);
        assert_eq!(target, [255]);
        let source = [4u16];
        let mut wide = [0u16];
        apply_band(&source, &mut wide, 0, &[0, -9, 0, 0, 0], 5, 1023);
        assert_eq!(wide, [0]);
    }

    /// Each shape of (8-411): a valley, a concave corner, flat, a convex corner, and
    /// a peak.
    #[test]
    fn edge_shapes_select_their_offsets() {
        let offsets = [0, 4, 2, -2, -4];
        let source = [10u8, 10, 10, 10, 10, 10];
        let first = [20u8, 20, 10, 10, 0, 0];
        let second = [20u8, 10, 10, 0, 0, 20];
        let mut target = [0u8; 6];
        apply_edge(&source, &first, &second, &mut target, &offsets, 255);
        // Lower than both: edgeIdx 1. One equal and one higher: edgeIdx 2.
        // Equal: edgeIdx 0. One equal and one lower: edgeIdx 3. Higher than both:
        // edgeIdx 4. One higher and one lower sums to flat: edgeIdx 0.
        assert_eq!(target, [14, 12, 10, 8, 6, 10]);
    }
}
