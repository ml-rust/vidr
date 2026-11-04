//! Arithmetic decoder tables: rangeTabLps (Table 9-52) and the state
//! transitions (Table 9-53).
//!
//! A context state packs `pStateIdx << 1 | valMps` in one byte. The transition
//! tables here take that packed form.

/// `rangeTabLps[ pStateIdx ][ qRangeIdx ]` (Table 9-52).
pub(crate) const RANGE_TAB_LPS: [[u8; 4]; 64] = [
    [128, 176, 208, 240],
    [128, 167, 197, 227],
    [128, 158, 187, 216],
    [123, 150, 178, 205],
    [116, 142, 169, 195],
    [111, 135, 160, 185],
    [105, 128, 152, 175],
    [100, 122, 144, 166],
    [95, 116, 137, 158],
    [90, 110, 130, 150],
    [85, 104, 123, 142],
    [81, 99, 117, 135],
    [77, 94, 111, 128],
    [73, 89, 105, 122],
    [69, 85, 100, 116],
    [66, 80, 95, 110],
    [62, 76, 90, 104],
    [59, 72, 86, 99],
    [56, 69, 81, 94],
    [53, 65, 77, 89],
    [51, 62, 73, 85],
    [48, 59, 69, 80],
    [46, 56, 66, 76],
    [43, 53, 63, 72],
    [41, 50, 59, 69],
    [39, 48, 56, 65],
    [37, 45, 54, 62],
    [35, 43, 51, 59],
    [33, 41, 48, 56],
    [32, 39, 46, 53],
    [30, 37, 43, 50],
    [29, 35, 41, 48],
    [27, 33, 39, 45],
    [26, 31, 37, 43],
    [24, 30, 35, 41],
    [23, 28, 33, 39],
    [22, 27, 32, 37],
    [21, 26, 30, 35],
    [20, 24, 29, 33],
    [19, 23, 27, 31],
    [18, 22, 26, 30],
    [17, 21, 25, 28],
    [16, 20, 23, 27],
    [15, 19, 22, 25],
    [14, 18, 21, 24],
    [14, 17, 20, 23],
    [13, 16, 19, 22],
    [12, 15, 18, 21],
    [12, 14, 17, 20],
    [11, 14, 16, 19],
    [11, 13, 15, 18],
    [10, 12, 15, 17],
    [10, 12, 14, 16],
    [9, 11, 13, 15],
    [9, 11, 12, 14],
    [8, 10, 12, 14],
    [8, 9, 11, 13],
    [7, 9, 11, 12],
    [7, 9, 10, 12],
    [7, 8, 10, 11],
    [6, 8, 9, 11],
    [6, 7, 9, 10],
    [6, 7, 8, 9],
    [2, 2, 2, 2],
];

/// `transIdxLps` (Table 9-53).
const TRANS_IDX_LPS: [u8; 64] = [
    0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12, 13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21,
    21, 22, 22, 23, 24, 24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33, 33, 33, 34,
    34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
];

/// `transIdxMps` (Table 9-53).
const TRANS_IDX_MPS: [u8; 64] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63,
];

/// The packed state after a bin equal to `valMps`.
pub(crate) const NEXT_STATE_MPS: [u8; 128] = next_state_mps();

/// The packed state after a bin equal to `1 - valMps`. State 0 swaps `valMps`
/// (9-66).
pub(crate) const NEXT_STATE_LPS: [u8; 128] = next_state_lps();

const fn next_state_mps() -> [u8; 128] {
    let mut table = [0u8; 128];
    let mut packed = 0;
    while packed < 128 {
        let mps = (packed & 1) as u8;
        table[packed] = (TRANS_IDX_MPS[packed >> 1] << 1) | mps;
        packed += 1;
    }
    table
}

const fn next_state_lps() -> [u8; 128] {
    let mut table = [0u8; 128];
    let mut packed = 0;
    while packed < 128 {
        let state = packed >> 1;
        let mps = (packed & 1) as u8;
        let next_mps = if state == 0 { 1 - mps } else { mps };
        table[packed] = (TRANS_IDX_LPS[state] << 1) | next_mps;
        packed += 1;
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_transitions_follow_table_9_53() {
        // pStateIdx 0 with valMps 1: an LPS keeps state 0 and swaps valMps.
        assert_eq!(NEXT_STATE_LPS[1], 0);
        // pStateIdx 0 with valMps 0: an MPS moves to state 1.
        assert_eq!(NEXT_STATE_MPS[0], 2);
        // pStateIdx 62 stays at 62 on an MPS.
        assert_eq!(NEXT_STATE_MPS[62 << 1], 62 << 1);
        // pStateIdx 63 is the non-adapting state.
        assert_eq!(NEXT_STATE_MPS[(63 << 1) | 1], (63 << 1) | 1);
        assert_eq!(NEXT_STATE_LPS[63 << 1], 63 << 1);
        // pStateIdx 5 with valMps 1 moves to state 4 on an LPS.
        assert_eq!(NEXT_STATE_LPS[(5 << 1) | 1], (4 << 1) | 1);
    }

    #[test]
    fn range_table_corners_match_table_9_52() {
        assert_eq!(RANGE_TAB_LPS[0], [128, 176, 208, 240]);
        assert_eq!(RANGE_TAB_LPS[31], [29, 35, 41, 48]);
        assert_eq!(RANGE_TAB_LPS[62], [6, 7, 8, 9]);
        assert_eq!(RANGE_TAB_LPS[63], [2, 2, 2, 2]);
    }
}
