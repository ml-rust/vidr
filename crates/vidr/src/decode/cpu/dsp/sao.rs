//! Sample adaptive offset kernels (8.7.3.2) over runs of samples.

use super::sample::Sample;

/// `edgeIdx` after (8-412), by `2 + Sign( ) + Sign( )` of (8-411).
const EDGE_INDEX: [usize; 5] = [1, 2, 0, 3, 4];

/// `SaoOffsetVal` by band, from `bandTable` of (8-414).
pub(crate) fn band_table(position: u8, offsets: &[i16; 5]) -> [i16; 32] {
    let mut table = [0i16; 32];
    for k in 0..4usize {
        table[(k + usize::from(position)) & 31] = offsets[k + 1];
    }
    table
}

/// (8-415) on one run. `band_shift` is `bitDepth - 5`.
pub(crate) fn apply_band<S: Sample>(
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

/// (8-411) to (8-413) on one run. `first` and `second` start at the two neighbours
/// of the first sample.
pub(crate) fn apply_edge<S: Sample>(
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let table = band_table(2, &[0, 5, -5, 7, -7]);
        // At 8 bits, a band spans 8 values: band 2 starts at 16.
        let source = [15u8, 16, 24, 32, 40, 48, 2];
        let mut target = [0u8; 7];
        apply_band(&source, &mut target, &table, 3, 255);
        assert_eq!(target, [15, 21, 19, 39, 33, 48, 2]);
    }

    #[test]
    fn band_offsets_clip() {
        let table = band_table(31, &[0, 9, 0, 0, 0]);
        let source = [250u8];
        let mut target = [0u8];
        apply_band(&source, &mut target, &table, 3, 255);
        assert_eq!(target, [255]);
        let table = band_table(0, &[0, -9, 0, 0, 0]);
        let source = [4u16];
        let mut wide = [0u16];
        apply_band(&source, &mut wide, &table, 5, 1023);
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
