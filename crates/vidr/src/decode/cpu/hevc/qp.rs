//! Quantization parameter derivation (8.6.1).

/// `QpC` at `qPi` 30 to 43 (Table 8-10).
const QP_C_TABLE: [i32; 14] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37];

/// `qPY_PRED` (8-282). `left` and `above` hold `QpY` of the neighbours when both
/// are available and inside the current CTB. Otherwise each takes `previous`,
/// which is `qPY_PREV`.
pub(crate) fn predict_qp_y(previous: i32, left: Option<i32>, above: Option<i32>) -> i32 {
    (left.unwrap_or(previous) + above.unwrap_or(previous) + 1) >> 1
}

/// `QpY` (8-283). The sum stays positive for a `CuQpDeltaVal` in range
/// (7.4.9.14), and `rem_euclid` keeps any other value in range too.
pub(crate) const fn qp_y(predicted: i32, cu_qp_delta: i32, qp_bd_offset: i32) -> i32 {
    (predicted + cu_qp_delta + 52 + 2 * qp_bd_offset).rem_euclid(52 + qp_bd_offset) - qp_bd_offset
}

/// `Qp′Y` (8-284), the luma `qP` of (8-291) in Main and Main 10. `QpY` is at least
/// `-qp_bd_offset`, so the result is not negative.
pub(crate) const fn qp_prime_y(qp_y: i32, qp_bd_offset: i32) -> u32 {
    (qp_y + qp_bd_offset) as u32
}

/// `Qp′Cb` or `Qp′Cr` (8-285, 8-286, 8-289, 8-290) for `ChromaArrayType` 1.
/// `offset` is the PPS offset plus the slice offset.
pub(crate) fn qp_prime_c(qp_y: i32, offset: i32, qp_bd_offset: i32) -> u32 {
    let qp_i = (qp_y + offset).max(-qp_bd_offset).min(57);
    (qp_c_from_index(qp_i) + qp_bd_offset) as u32
}

/// `QpC` from `qPi` for `ChromaArrayType` 1 (Table 8-10).
pub(crate) fn qp_c_from_index(qp_i: i32) -> i32 {
    match qp_i {
        ..=29 => qp_i,
        30..=43 => QP_C_TABLE[(qp_i - 30) as usize],
        _ => qp_i - 6,
    }
}

/// The running state of 8.6.1 for one slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QpState {
    /// `IsCuQpDeltaCoded`.
    pub(crate) is_delta_coded: bool,
    /// `CuQpDeltaVal`.
    pub(crate) delta_val: i32,
    /// `QpY` of the last coding unit, for `qPY_PREV`.
    pub(crate) last_cu_qp_y: i32,
    /// `( xQg, yQg )` of the current quantization group.
    pub(crate) group: Option<(u32, u32)>,
    /// `qPY_PRED` of the current quantization group.
    pub(crate) predicted: i32,
}

impl QpState {
    /// The state at the first quantization group of a slice, a tile, or a CTB row
    /// with `entropy_coding_sync_enabled_flag`. `qPY_PREV` is `SliceQpY`.
    pub(crate) const fn new(slice_qp: i32) -> Self {
        Self {
            is_delta_coded: false,
            delta_val: 0,
            last_cu_qp_y: slice_qp,
            group: None,
            predicted: slice_qp,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prediction_averages_with_rounding_up() {
        assert_eq!(predict_qp_y(30, None, None), 30);
        assert_eq!(predict_qp_y(30, Some(33), None), 32);
        assert_eq!(predict_qp_y(30, None, Some(31)), 31);
        assert_eq!(predict_qp_y(30, Some(20), Some(23)), 22);
        // Negative at 10 bits: ( -11 + -12 + 1 ) >> 1 = -11.
        assert_eq!(predict_qp_y(0, Some(-11), Some(-12)), -11);
    }

    /// Worked by hand from (8-283).
    #[test]
    fn qp_y_wraps_through_the_full_range() {
        assert_eq!(qp_y(30, 0, 0), 30);
        assert_eq!(qp_y(30, -5, 0), 25);
        // 8 bits: ( 50 + 5 + 52 ) % 52 = 3.
        assert_eq!(qp_y(50, 5, 0), 3);
        // 8 bits: ( 2 - 5 + 52 ) % 52 = 49.
        assert_eq!(qp_y(2, -5, 0), 49);
        // 10 bits, offset 12: ( 50 + 5 + 52 + 24 ) % 64 = 3, minus 12 is -9.
        assert_eq!(qp_y(50, 5, 12), -9);
        // 10 bits: ( -10 - 5 + 52 + 24 ) % 64 = 61, minus 12 is 49.
        assert_eq!(qp_y(-10, -5, 12), 49);
    }

    #[test]
    fn luma_qp_prime_adds_the_offset() {
        assert_eq!(qp_prime_y(-12, 12), 0);
        assert_eq!(qp_prime_y(51, 12), 63);
    }

    /// Table 8-10 at every step.
    #[test]
    fn chroma_qp_follows_table_8_10() {
        let expected = [
            (29, 29),
            (30, 29),
            (31, 30),
            (32, 31),
            (33, 32),
            (34, 33),
            (35, 33),
            (36, 34),
            (37, 34),
            (38, 35),
            (39, 35),
            (40, 36),
            (41, 36),
            (42, 37),
            (43, 37),
            (44, 38),
            (51, 45),
            (57, 51),
        ];
        for (qp_i, qp_c) in expected {
            assert_eq!(qp_prime_c(qp_i, 0, 0), qp_c, "qPi {qp_i}");
        }
    }

    /// The index of Table 8-10 has no clip in the deblocking use (8-382).
    #[test]
    fn the_table_alone_takes_any_index() {
        assert_eq!(qp_c_from_index(-20), -20);
        assert_eq!(qp_c_from_index(29), 29);
        assert_eq!(qp_c_from_index(43), 37);
        assert_eq!(qp_c_from_index(70), 64);
    }

    #[test]
    fn chroma_qp_clips_and_applies_offsets() {
        // qPi clips to 57, and QpC is 51.
        assert_eq!(qp_prime_c(51, 12, 0), 51);
        // qPi clips to -QpBdOffsetC, and Qp′C is 0.
        assert_eq!(qp_prime_c(-12, -12, 12), 0);
        // 10 bits: qPi 31 gives QpC 30, plus 12.
        assert_eq!(qp_prime_c(29, 2, 12), 42);
    }

    #[test]
    fn a_new_state_predicts_the_slice_qp() {
        let state = QpState::new(33);
        assert_eq!(state.last_cu_qp_y, 33);
        assert_eq!(state.delta_val, 0);
        assert!(!state.is_delta_coded);
        assert_eq!(state.group, None);
    }
}
