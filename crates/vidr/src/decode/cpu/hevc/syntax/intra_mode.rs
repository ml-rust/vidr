//! Intra mode derivation (8.4.2, 8.4.3) and `scanIdx` (7.4.9.11). The parse stage
//! needs them, because `scanIdx` selects how `residual_coding` reads.

use crate::decode::cpu::hevc::scan::{SCAN_DIAGONAL, SCAN_HORIZONTAL, SCAN_VERTICAL};

const INTRA_PLANAR: u8 = 0;
const INTRA_DC: u8 = 1;
const INTRA_ANGULAR26: u8 = 26;
/// Replaces a chroma mode equal to the luma mode (Table 8-2).
const INTRA_ANGULAR34: u8 = 34;
/// `intra_chroma_pred_mode` that copies the luma mode (Table 8-2).
pub(crate) const CHROMA_FROM_LUMA: u32 = 4;
/// Chroma modes for `intra_chroma_pred_mode` 0 to 3 (Table 8-2).
const CHROMA_MODES: [u8; 4] = [0, 26, 10, 1];

/// `candModeList` from `candIntraPredModeA` and `candIntraPredModeB` (8-21 to
/// 8-28).
pub(crate) const fn cand_mode_list(a: u8, b: u8) -> [u8; 3] {
    if a == b {
        if a < 2 {
            [INTRA_PLANAR, INTRA_DC, INTRA_ANGULAR26]
        } else {
            [a, 2 + ((a + 29) % 32), 2 + ((a - 2 + 1) % 32)]
        }
    } else {
        let third = if a != INTRA_PLANAR && b != INTRA_PLANAR {
            INTRA_PLANAR
        } else if a != INTRA_DC && b != INTRA_DC {
            INTRA_DC
        } else {
            INTRA_ANGULAR26
        };
        [a, b, third]
    }
}

/// `IntraPredModeY` from `rem_intra_luma_pred_mode` (8-29 to 8-31 and the steps
/// after them).
pub(crate) fn mode_from_rem(mut candidates: [u8; 3], rem: u8) -> u8 {
    candidates.sort_unstable();
    candidates.iter().fold(
        rem,
        |mode, &candidate| {
            if mode >= candidate { mode + 1 } else { mode }
        },
    )
}

/// `IntraPredModeC` for `ChromaArrayType` 1 (Table 8-2). A value above 4 acts as
/// 4, and the binarization of 9.3.3.8 cannot produce one.
pub(crate) const fn chroma_mode(intra_chroma_pred_mode: u32, luma_mode: u8) -> u8 {
    if intra_chroma_pred_mode >= CHROMA_FROM_LUMA {
        return luma_mode;
    }
    let mode = CHROMA_MODES[intra_chroma_pred_mode as usize];
    if mode == luma_mode {
        INTRA_ANGULAR34
    } else {
        mode
    }
}

/// `scanIdx` (7.4.9.11) with `ChromaArrayType` 1. `log2_size` is the
/// `log2TrafoSize` of this `residual_coding`.
pub(crate) const fn scan_idx(intra: bool, log2_size: u32, c_idx: usize, pred_mode: u8) -> usize {
    let mode_dependent = log2_size == 2 || (log2_size == 3 && c_idx == 0);
    if !intra || !mode_dependent {
        return SCAN_DIAGONAL;
    }
    match pred_mode {
        6..=14 => SCAN_VERTICAL,
        22..=30 => SCAN_HORIZONTAL,
        _ => SCAN_DIAGONAL,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_candidates_below_2_give_the_default_list() {
        assert_eq!(cand_mode_list(0, 0), [0, 1, 26]);
        assert_eq!(cand_mode_list(1, 1), [0, 1, 26]);
    }

    /// (8-25) and (8-26) step around the angular modes and wrap.
    #[test]
    fn equal_angular_candidates_give_their_neighbours() {
        assert_eq!(cand_mode_list(10, 10), [10, 9, 11]);
        assert_eq!(cand_mode_list(2, 2), [2, 33, 3]);
        assert_eq!(cand_mode_list(34, 34), [34, 33, 3]);
    }

    #[test]
    fn different_candidates_take_the_first_free_default() {
        assert_eq!(cand_mode_list(10, 20), [10, 20, 0]);
        assert_eq!(cand_mode_list(0, 20), [0, 20, 1]);
        assert_eq!(cand_mode_list(1, 20), [1, 20, 0]);
        assert_eq!(cand_mode_list(0, 1), [0, 1, 26]);
        assert_eq!(cand_mode_list(1, 0), [1, 0, 26]);
    }

    /// The remaining modes skip the three candidates, in order.
    #[test]
    fn rem_counts_the_modes_outside_the_list() {
        let list = [26, 0, 1];
        let found: Vec<u8> = (0..32).map(|rem| mode_from_rem(list, rem)).collect();
        let expected: Vec<u8> = (0..35).filter(|mode| !list.contains(mode)).collect();
        assert_eq!(found, expected);
        let list = [10, 9, 11];
        let found: Vec<u8> = (0..32).map(|rem| mode_from_rem(list, rem)).collect();
        let expected: Vec<u8> = (0..35).filter(|mode| !list.contains(mode)).collect();
        assert_eq!(found, expected);
    }

    #[test]
    fn chroma_mode_follows_table_8_2() {
        assert_eq!(chroma_mode(0, 0), 34);
        assert_eq!(chroma_mode(0, 26), 0);
        assert_eq!(chroma_mode(1, 26), 34);
        assert_eq!(chroma_mode(1, 0), 26);
        assert_eq!(chroma_mode(2, 10), 34);
        assert_eq!(chroma_mode(2, 7), 10);
        assert_eq!(chroma_mode(3, 1), 34);
        assert_eq!(chroma_mode(3, 7), 1);
        assert_eq!(chroma_mode(4, 7), 7);
        assert_eq!(chroma_mode(4, 34), 34);
    }

    #[test]
    fn scan_idx_follows_7_4_9_11() {
        // Luma 4x4 and 8x8 follow the mode.
        assert_eq!(scan_idx(true, 2, 0, 6), SCAN_VERTICAL);
        assert_eq!(scan_idx(true, 2, 0, 14), SCAN_VERTICAL);
        assert_eq!(scan_idx(true, 3, 0, 22), SCAN_HORIZONTAL);
        assert_eq!(scan_idx(true, 3, 0, 30), SCAN_HORIZONTAL);
        assert_eq!(scan_idx(true, 2, 0, 5), SCAN_DIAGONAL);
        assert_eq!(scan_idx(true, 2, 0, 15), SCAN_DIAGONAL);
        assert_eq!(scan_idx(true, 2, 0, 21), SCAN_DIAGONAL);
        assert_eq!(scan_idx(true, 2, 0, 31), SCAN_DIAGONAL);
        // Luma above 8x8 stays diagonal.
        assert_eq!(scan_idx(true, 4, 0, 10), SCAN_DIAGONAL);
        // Chroma follows the mode at 4x4 only with ChromaArrayType 1.
        assert_eq!(scan_idx(true, 2, 1, 10), SCAN_VERTICAL);
        assert_eq!(scan_idx(true, 2, 2, 26), SCAN_HORIZONTAL);
        assert_eq!(scan_idx(true, 3, 1, 10), SCAN_DIAGONAL);
        // Inter stays diagonal.
        assert_eq!(scan_idx(false, 2, 0, 10), SCAN_DIAGONAL);
    }
}
