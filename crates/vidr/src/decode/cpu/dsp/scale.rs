//! The scaling step for one transform coefficient (8.6.3).

/// `CoeffMinY` and `CoeffMinC` without extended precision (7-27, 7-28).
pub(crate) const COEFF_MIN: i32 = -32_768;
/// `CoeffMaxY` and `CoeffMaxC` without extended precision (7-29, 7-30).
pub(crate) const COEFF_MAX: i32 = 32_767;

/// `levelScale` (8.6.3).
const LEVEL_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];

/// The parts of (8-309) that one transform block shares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScaleStep {
    factor: i64,
    round: i64,
    shift: u32,
}

impl ScaleStep {
    /// The step for quantization parameter `qp`, `BitDepth` `bit_depth`, and
    /// `Log2( nTbS )` `log2_size`. `bdShift` is (8-301) with a
    /// `log2TransformRange` of 15.
    pub(crate) const fn new(qp: u32, bit_depth: u32, log2_size: u32) -> Self {
        let shift = bit_depth + log2_size + 10 - 15;
        Self {
            factor: LEVEL_SCALE[(qp % 6) as usize] << (qp / 6),
            round: 1 << (shift - 1),
            shift,
        }
    }

    /// `d[ x ][ y ]` of (8-309) for one level and scaling factor.
    #[inline]
    pub(crate) fn scale(&self, level: i32, m: i32) -> i32 {
        let scaled = (i64::from(level) * i64::from(m) * self.factor + self.round) >> self.shift;
        scaled.clamp(i64::from(COEFF_MIN), i64::from(COEFF_MAX)) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Worked by hand from (8-301) and (8-309).
    #[test]
    fn scale_follows_8_309() {
        // qP 28: levelScale[ 4 ] = 64, shift 4. bdShift 5.
        // ( 3 * 16 * 64 << 4 ) = 49152. ( 49152 + 16 ) >> 5 = 1536.
        assert_eq!(ScaleStep::new(28, 8, 2).scale(3, 16), 1536);
        // qP 0: levelScale[ 0 ] = 40. ( -640 + 16 ) >> 5 = -20, rounding down.
        assert_eq!(ScaleStep::new(0, 8, 2).scale(-1, 16), -20);
        // qP 51: levelScale[ 3 ] = 57, shift 8. bdShift 6 at 8 bits and 8x8.
        // 2 * 16 * 57 << 8 = 466944. ( 466944 + 32 ) >> 6 = 7296.
        assert_eq!(ScaleStep::new(51, 8, 3).scale(2, 16), 7296);
        // 10 bits, 32x32: bdShift 10. qP 7: levelScale[ 1 ] = 45, shift 1.
        // 5 * 20 * 90 = 9000. ( 9000 + 512 ) >> 10 = 9.
        assert_eq!(ScaleStep::new(7, 10, 5).scale(5, 20), 9);
    }

    #[test]
    fn scale_clips_to_16_bits() {
        let step = ScaleStep::new(63, 10, 2);
        assert_eq!(step.scale(32_767, 255), COEFF_MAX);
        assert_eq!(step.scale(-32_768, 255), COEFF_MIN);
    }
}
