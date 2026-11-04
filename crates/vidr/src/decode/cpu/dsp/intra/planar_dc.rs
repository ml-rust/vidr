//! `INTRA_PLANAR` (8.4.4.2.4) and `INTRA_DC` (8.4.4.2.5).

use super::refs::{PredBlock, RefSamples};

/// (8-46).
pub(crate) fn predict_planar(refs: &RefSamples, log2_size: u32, out: &mut PredBlock) {
    let size = refs.size();
    let top_right = u32::from(refs.top(size));
    let bottom_left = u32::from(refs.left(size));
    let shift = log2_size + 1;
    for (y, row) in out.rows_mut(size).enumerate() {
        let left = u32::from(refs.left(y));
        let row_weight = (size - 1 - y) as u32;
        let bottom = (y as u32 + 1) * bottom_left;
        for (x, sample) in row.iter_mut().enumerate() {
            let x_weight = x as u32 + 1;
            let horizontal = (size as u32 - x_weight) * left + x_weight * top_right;
            let vertical = row_weight * u32::from(refs.top(x)) + bottom;
            *sample = ((horizontal + vertical + size as u32) >> shift) as u16;
        }
    }
}

/// (8-47) to (8-52). `luma` is true for `cIdx` 0, which takes the edge filter below
/// an `nTbS` of 32.
pub(crate) fn predict_dc(refs: &RefSamples, log2_size: u32, luma: bool, out: &mut PredBlock) {
    let size = refs.size();
    let sum: u32 = (0..size)
        .map(|index| u32::from(refs.top(index)) + u32::from(refs.left(index)))
        .sum();
    let dc = (sum + size as u32) >> (log2_size + 1);
    for row in out.rows_mut(size) {
        row.fill(dc as u16);
    }
    if !luma || size >= 32 {
        return;
    }
    let edge = |neighbour: u16| ((u32::from(neighbour) + 3 * dc + 2) >> 2) as u16;
    for index in 1..size {
        out.set(size, index, 0, edge(refs.top(index)));
        out.set(size, 0, index, edge(refs.left(index)));
    }
    let first = (u32::from(refs.left(0)) + 2 * dc + u32::from(refs.top(0)) + 2) >> 2;
    out.set(size, 0, 0, first as u16);
}

#[cfg(test)]
mod tests {
    use super::super::refs::tests::filled;
    use super::*;

    #[test]
    fn planar_on_equal_references_is_flat() {
        let refs = filled(3, 77, 77, 77);
        let mut out = PredBlock::new();
        predict_planar(&refs, 3, &mut out);
        for row in out.rows(8) {
            assert!(row.iter().all(|&sample| sample == 77));
        }
    }

    /// Worked from (8-46) for a 4x4 block with left 10 and top 90.
    #[test]
    fn planar_blends_both_directions() {
        let refs = filled(2, 10, 0, 90);
        let mut out = PredBlock::new();
        predict_planar(&refs, 2, &mut out);
        // ( 3 * 10 + 1 * 90 + 3 * 90 + 1 * 10 + 4 ) >> 3 = 50.
        assert_eq!(out.get(4, 0, 0), 50);
        // ( 0 * 10 + 4 * 90 + 3 * 90 + 1 * 10 + 4 ) >> 3 = 80.
        assert_eq!(out.get(4, 3, 0), 80);
        // ( 3 * 10 + 1 * 90 + 0 * 90 + 4 * 10 + 4 ) >> 3 = 20.
        assert_eq!(out.get(4, 0, 3), 20);
        // ( 0 * 10 + 4 * 90 + 0 * 90 + 4 * 10 + 4 ) >> 3 = 50.
        assert_eq!(out.get(4, 3, 3), 50);
    }

    /// dcVal = ( 4 * 90 + 4 * 10 + 4 ) >> 3 = 50, and (8-48) to (8-50) at the edges.
    #[test]
    fn dc_filters_the_first_row_and_column_of_luma() {
        let refs = filled(2, 10, 0, 90);
        let mut out = PredBlock::new();
        predict_dc(&refs, 2, true, &mut out);
        assert_eq!(out.get(4, 0, 0), (10 + 100 + 90 + 2) >> 2);
        assert_eq!(out.get(4, 1, 0), (90 + 150 + 2) >> 2);
        assert_eq!(out.get(4, 3, 0), (90 + 150 + 2) >> 2);
        assert_eq!(out.get(4, 0, 1), (10 + 150 + 2) >> 2);
        assert_eq!(out.get(4, 0, 3), (10 + 150 + 2) >> 2);
        assert_eq!(out.get(4, 1, 1), 50);
        assert_eq!(out.get(4, 3, 3), 50);
    }

    #[test]
    fn chroma_dc_takes_no_edge_filter() {
        let refs = filled(2, 10, 0, 90);
        let mut out = PredBlock::new();
        predict_dc(&refs, 2, false, &mut out);
        for row in out.rows(4) {
            assert!(row.iter().all(|&sample| sample == 50));
        }
    }

    #[test]
    fn dc_at_32x32_takes_no_edge_filter() {
        let refs = filled(5, 10, 0, 90);
        let mut out = PredBlock::new();
        predict_dc(&refs, 5, true, &mut out);
        assert_eq!(out.get(32, 0, 0), 50);
        assert_eq!(out.get(32, 5, 0), 50);
    }
}
