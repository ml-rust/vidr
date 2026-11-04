//! The frame rate an encoder takes as a hint.
//!
//! The pump passes every frame's pts and duration, so the rate never sets the
//! output's timing. A variable-rate source has no exact rate, and its mean can
//! need more than 32 bits a part, which `EncodeConfig::frame_rate_parts` refuses.

use vidr::Rational;

/// The largest part of a rate an encoder takes.
const PART_LIMIT: u128 = u32::MAX as u128;

/// Returns the mean frame rate as a rational with 32-bit parts.
///
/// The rate is `(frames - 1) / span`, over the pts span from the first frame to
/// the last. With one frame or no span, it is `1 / last_duration`. With neither,
/// it is 1. The result is above zero in every case.
pub fn frame_rate_hint(frames: u64, span: Rational, last_duration: Rational) -> Rational {
    if frames > 1 && span.is_positive() {
        let intervals = u128::from(frames - 1) * u128::from(span.den());
        return within_32_bits(intervals, u128::from(span.num().unsigned_abs()));
    }
    if last_duration.is_positive() {
        return within_32_bits(
            u128::from(last_duration.den()),
            u128::from(last_duration.num().unsigned_abs()),
        );
    }
    Rational::ONE
}

/// The last continued-fraction convergent of `numerator / denominator` with both
/// parts in 32 bits. A value that fits exactly returns in lowest terms.
///
/// A value above the largest part gives that part over 1. A value below its
/// reciprocal gives that reciprocal.
fn within_32_bits(numerator: u128, denominator: u128) -> Rational {
    let (mut previous_num, mut num) = (0_u128, 1_u128);
    let (mut previous_den, mut den) = (1_u128, 0_u128);
    let (mut dividend, mut divisor) = (numerator, denominator);
    while divisor != 0 {
        let term = dividend / divisor;
        let next_num = term
            .checked_mul(num)
            .and_then(|n| n.checked_add(previous_num));
        let next_den = term
            .checked_mul(den)
            .and_then(|d| d.checked_add(previous_den));
        let (Some(next_num), Some(next_den)) = (next_num, next_den) else {
            break;
        };
        if next_num > PART_LIMIT || next_den > PART_LIMIT {
            break;
        }
        (previous_num, num) = (num, next_num);
        (previous_den, den) = (den, next_den);
        (dividend, divisor) = (divisor, dividend % divisor);
    }
    let (num, den) = match (num, den) {
        // The first term alone exceeds the limit.
        (_, 0) => (PART_LIMIT, 1),
        // The value is below the reciprocal of the limit.
        (0, _) => (1, PART_LIMIT),
        parts => parts,
    };
    // Both parts fit 32 bits and `den` is above zero, so neither conversion fails.
    match (i64::try_from(num), u64::try_from(den)) {
        (Ok(num), Ok(den)) => Rational::new(num, den).unwrap_or(Rational::ONE),
        _ => Rational::ONE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(num: i64, den: u64) -> Rational {
        Rational::new(num, den).expect("rational")
    }

    fn fits(rate: Rational) -> bool {
        u32::try_from(rate.num()).is_ok_and(|num| num > 0) && u32::try_from(rate.den()).is_ok()
    }

    #[test]
    fn a_constant_rate_stays_exact() {
        assert_eq!(frame_rate_hint(26, r(1, 1), r(1, 25)), r(25, 1));
        assert_eq!(
            frame_rate_hint(240, r(239 * 1001, 30000), r(1001, 30000)),
            r(30000, 1001)
        );
    }

    #[test]
    fn one_frame_uses_its_duration() {
        assert_eq!(frame_rate_hint(1, Rational::ZERO, r(1, 25)), r(25, 1));
    }

    #[test]
    fn no_span_and_no_duration_gives_one() {
        assert_eq!(
            frame_rate_hint(1, Rational::ZERO, Rational::ZERO),
            Rational::ONE
        );
    }

    #[test]
    fn a_mean_with_wide_parts_reduces_to_32_bits() {
        // Parts near 2^61 that share no factor stay wide until the reduction.
        let span = r(2_305_843_009_213_693_951, 2_305_843_009_213_693_921);
        let rate = frame_rate_hint(31, span, Rational::ZERO);
        assert!(fits(rate), "{rate}");
        let error = rate.checked_sub(r(30, 1)).expect("difference");
        let bound = r(1, 1_000_000);
        assert!(error.checked_abs().expect("abs") < bound, "{rate}");
    }

    #[test]
    fn extreme_rates_stay_above_zero_and_within_32_bits() {
        let huge = frame_rate_hint(u64::MAX, r(1, u64::MAX), Rational::ZERO);
        assert_eq!(huge, r(i64::from(u32::MAX), 1));
        let tiny = frame_rate_hint(2, r(i64::MAX, 1), Rational::ZERO);
        assert_eq!(tiny, r(1, u64::from(u32::MAX)));
    }

    #[test]
    fn every_result_passes_the_encoder_rule() {
        let cases = [
            (2, r(1, 1000)),
            (1000, r(39_999, 1000)),
            (100_000, r(3_336_667, 1000)),
            (3, r(7, 90000)),
        ];
        for (frames, span) in cases {
            let rate = frame_rate_hint(frames, span, Rational::ZERO);
            assert!(fits(rate), "{frames} over {span}: {rate}");
        }
    }
}
