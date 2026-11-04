//! `ScalingFactor` (7.4.5) and the choice of list (7.4.3.2.1, 7.4.3.3.1).

use super::scan::{SCAN_DIAGONAL, scan_order};
use crate::bitstream::hevc_pps::PicParameterSet;
use crate::bitstream::hevc_scaling::ScalingListData;
use crate::bitstream::hevc_sps_full::SeqParameterSet;

/// The factor when no list applies (8.6.3).
pub(crate) const FLAT_FACTOR: i32 = 16;

/// `ScalingFactor`, each array row by row. Size 32 holds `matrixId` 0 and 3 only:
/// the other matrices need `ChromaArrayType` 3 (7-50).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ScalingFactors {
    size_4: [[u8; 16]; 6],
    size_8: [[u8; 64]; 6],
    size_16: [[u8; 256]; 6],
    size_32: [[u8; 1024]; 2],
}

impl ScalingFactors {
    /// Expands `ScalingList` with (7-44) to (7-49).
    pub(crate) fn new(data: &ScalingListData) -> Self {
        let mut factors = Self {
            size_4: [[0; 16]; 6],
            size_8: [[0; 64]; 6],
            size_16: [[0; 256]; 6],
            size_32: [[0; 1024]; 2],
        };
        let scan_4 = scan_order(2, SCAN_DIAGONAL);
        let scan_8 = scan_order(3, SCAN_DIAGONAL);
        for matrix_id in 0..6 {
            for (&[x, y], &value) in scan_4.iter().zip(&data.scaling_list[0][matrix_id]) {
                factors.size_4[matrix_id][usize::from(y) * 4 + usize::from(x)] = value;
            }
            for (&[x, y], &value) in scan_8.iter().zip(&data.scaling_list[1][matrix_id]) {
                factors.size_8[matrix_id][usize::from(y) * 8 + usize::from(x)] = value;
            }
            let expanded = &mut factors.size_16[matrix_id];
            expand(expanded, 16, scan_8, &data.scaling_list[2][matrix_id]);
            expanded[0] = dc_factor(data.scaling_list_dc_coef_minus8[0][matrix_id]);
        }
        for (index, matrix_id) in [0usize, 3].into_iter().enumerate() {
            let expanded = &mut factors.size_32[index];
            expand(expanded, 32, scan_8, &data.scaling_list[3][matrix_id]);
            expanded[0] = dc_factor(data.scaling_list_dc_coef_minus8[1][matrix_id]);
        }
        factors
    }

    /// The factors of the active parameter sets, or `None` with
    /// `scaling_list_enabled_flag` 0. A PPS list wins over an SPS list, and the
    /// default lists apply when neither set holds one.
    pub(crate) fn select(sps: &SeqParameterSet, pps: &PicParameterSet) -> Option<Self> {
        if !sps.scaling_list_enabled_flag {
            return None;
        }
        Some(match (&pps.scaling_list_data, &sps.scaling_list_data) {
            (Some(list), _) | (None, Some(list)) => Self::new(list),
            (None, None) => Self::new(&ScalingListData::default_lists()),
        })
    }

    /// The array for `Log2( nTbS )` `log2_size` and `matrixId` (Table 7-4), row by
    /// row. A size 32 chroma matrix returns its luma matrix, and 4:2:0 has no such
    /// block.
    pub(crate) fn matrix(&self, log2_size: u32, matrix_id: usize) -> &[u8] {
        let matrix_id = matrix_id.min(5);
        match log2_size {
            2 => &self.size_4[matrix_id],
            3 => &self.size_8[matrix_id],
            4 => &self.size_16[matrix_id],
            _ => &self.size_32[matrix_id / 3],
        }
    }
}

/// `matrixId` for a block (Table 7-4).
pub(crate) const fn matrix_id(intra: bool, c_idx: usize) -> usize {
    if intra { c_idx } else { 3 + c_idx }
}

/// Fills a `size` by `size` array from an 8x8 list. Each list entry covers a square
/// of `size / 8` samples (7-46, 7-48).
fn expand(target: &mut [u8], size: usize, scan_8: &[[u8; 2]], list: &[u8; 64]) {
    let ratio = size / 8;
    for (&[x, y], &value) in scan_8.iter().zip(list) {
        let (x, y) = (usize::from(x) * ratio, usize::from(y) * ratio);
        for row in target.chunks_exact_mut(size).skip(y).take(ratio) {
            row[x..x + ratio].fill(value);
        }
    }
}

/// `scaling_list_dc_coef_minus8 + 8` (7-47, 7-49). The parser holds the coefficient
/// to -7 to 247, so the sum fits.
const fn dc_factor(dc_minus8: i16) -> u8 {
    (dc_minus8 + 8) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::hevc_pps::tests as pps_tests;
    use crate::bitstream::hevc_sps_full::tests as sps_tests;

    fn at(factors: &ScalingFactors, log2_size: u32, matrix: usize, x: usize, y: usize) -> u8 {
        factors.matrix(log2_size, matrix)[(y << log2_size) + x]
    }

    #[test]
    fn default_lists_expand_as_tables_7_5_and_7_6_state() {
        let factors = ScalingFactors::new(&ScalingListData::default_lists());
        // Table 7-5 is flat.
        assert!(factors.matrix(2, 0).iter().all(|&value| value == 16));
        assert!(factors.matrix(2, 5).iter().all(|&value| value == 16));
        // Table 7-6, i = 63, sits at the far corner of each size.
        assert_eq!(at(&factors, 3, 0, 7, 7), 115);
        assert_eq!(at(&factors, 3, 3, 7, 7), 91);
        assert_eq!(at(&factors, 4, 1, 15, 15), 115);
        assert_eq!(at(&factors, 4, 4, 14, 14), 91);
        assert_eq!(at(&factors, 5, 0, 31, 31), 115);
        assert_eq!(at(&factors, 5, 3, 28, 28), 91);
        // The inferred DC value is 16.
        assert_eq!(at(&factors, 4, 0, 0, 0), 16);
        assert_eq!(at(&factors, 5, 3, 0, 0), 16);
        // i = 59 of the intra list is 70, at ( 6, 6 ) of the 8x8 scan.
        assert_eq!(at(&factors, 3, 0, 6, 6), 70);
    }

    /// A list with `value = i + 1` shows each entry's position from the scan.
    #[test]
    fn a_counting_list_follows_the_scan_and_the_expansion() {
        let mut data = ScalingListData::default_lists();
        let mut counting = [0u8; 64];
        for (index, value) in counting.iter_mut().enumerate() {
            *value = index as u8 + 1;
        }
        data.scaling_list[0][2] = counting;
        data.scaling_list[1][2] = counting;
        data.scaling_list[2][2] = counting;
        data.scaling_list[3][3] = counting;
        data.scaling_list_dc_coef_minus8[0][2] = 92;
        data.scaling_list_dc_coef_minus8[1][1] = 0;
        data.scaling_list_dc_coef_minus8[1][3] = -7;
        let factors = ScalingFactors::new(&data);
        // Diagonal scan positions: i = 1 is ( 0, 1 ), i = 2 is ( 1, 0 ).
        assert_eq!(at(&factors, 2, 2, 0, 1), 2);
        assert_eq!(at(&factors, 2, 2, 1, 0), 3);
        assert_eq!(at(&factors, 2, 2, 3, 3), 16);
        assert_eq!(at(&factors, 3, 2, 0, 2), 4);
        assert_eq!(at(&factors, 3, 2, 7, 7), 64);
        // 16x16: each entry covers two samples per side, and the DC is separate.
        assert_eq!(at(&factors, 4, 2, 0, 0), 100);
        assert_eq!(at(&factors, 4, 2, 1, 1), 1);
        assert_eq!(at(&factors, 4, 2, 0, 2), 2);
        assert_eq!(at(&factors, 4, 2, 1, 3), 2);
        assert_eq!(at(&factors, 4, 2, 2, 0), 3);
        // 32x32: each entry covers four samples per side.
        assert_eq!(at(&factors, 5, 3, 0, 0), 1);
        assert_eq!(at(&factors, 5, 3, 3, 3), 1);
        assert_eq!(at(&factors, 5, 3, 0, 4), 2);
        assert_eq!(at(&factors, 5, 3, 4, 3), 3);
    }

    #[test]
    fn matrix_ids_follow_table_7_4() {
        assert_eq!(matrix_id(true, 0), 0);
        assert_eq!(matrix_id(true, 2), 2);
        assert_eq!(matrix_id(false, 0), 3);
        assert_eq!(matrix_id(false, 2), 5);
    }

    #[test]
    fn the_flag_off_selects_nothing() {
        let sps = sps_tests::valid(&sps_tests::Spec::default());
        let pps = pps_tests::valid(&pps_tests::Spec::default());
        assert_eq!(ScalingFactors::select(&sps, &pps), None);
    }

    #[test]
    fn the_flag_on_alone_selects_the_defaults() {
        let sps = sps_tests::valid(&sps_tests::Spec {
            scaling: true,
            ..sps_tests::Spec::default()
        });
        let pps = pps_tests::valid(&pps_tests::Spec::default());
        assert_eq!(
            ScalingFactors::select(&sps, &pps),
            Some(ScalingFactors::new(&ScalingListData::default_lists()))
        );
    }

    #[test]
    fn a_pps_list_wins_over_an_sps_list() {
        let mut sps = sps_tests::valid(&sps_tests::Spec {
            scaling: true,
            ..sps_tests::Spec::default()
        });
        let mut pps = pps_tests::valid(&pps_tests::Spec::default());
        let mut sps_list = ScalingListData::default_lists();
        sps_list.scaling_list[0][0] = [1; 64];
        let mut pps_list = ScalingListData::default_lists();
        pps_list.scaling_list[0][0] = [2; 64];
        sps.scaling_list_data = Some(Box::new(sps_list.clone()));
        assert_eq!(
            ScalingFactors::select(&sps, &pps),
            Some(ScalingFactors::new(&sps_list))
        );
        pps.scaling_list_data = Some(Box::new(pps_list.clone()));
        assert_eq!(
            ScalingFactors::select(&sps, &pps),
            Some(ScalingFactors::new(&pps_list))
        );
    }
}
