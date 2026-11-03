//! Rounding of exact quotients to whole ticks, and timestamp rescaling.
//!
//! Every conversion names its [`Rounding`] mode. The PoC writes each pts and
//! duration as Python `round(value / clock)` on a `Fraction`, which rounds
//! half to even. [`Rounding::HalfEven`] reproduces it.

use std::cmp::Ordering;

use super::rational::{Rational, gcd};
use crate::error::{Operation, TimeError};

/// How a quotient between two integers becomes one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rounding {
    /// Nearest integer, ties to the even one. Python's `round()` on a
    /// `Fraction`, as the PoC uses for every pts and duration it writes.
    HalfEven,
    /// Nearest integer, ties away from zero. FFmpeg's `av_rescale_q` default
    /// (`AV_ROUND_NEAR_INF`).
    HalfAwayFromZero,
    /// Toward negative infinity.
    Floor,
    /// Toward positive infinity.
    Ceil,
    /// Toward zero.
    TowardZero,
    /// No rounding. A quotient that is not an integer is an error.
    Exact,
}

/// Why a quotient did not become an `i64`.
enum Fault {
    Inexact,
    Overflow,
}

impl Rational {
    /// Rounds to an integer.
    ///
    /// # Errors
    ///
    /// [`TimeError::NotAnInteger`] under [`Rounding::Exact`] for a fraction.
    pub fn round(self, rounding: Rounding) -> Result<i64, TimeError> {
        let quotient = div_round(i128::from(self.num()), i128::from(self.den()), rounding);
        narrow(quotient).map_err(|fault| match fault {
            Fault::Inexact => TimeError::NotAnInteger { value: self },
            Fault::Overflow => TimeError::Overflow {
                operation: Operation::Round,
                lhs: self,
                rhs: None,
            },
        })
    }

    /// Converts seconds to ticks of `time_base`: `round(self / time_base)`.
    ///
    /// # Errors
    ///
    /// [`TimeError::NonPositiveTimeBase`] for a time base of zero or less.
    /// [`TimeError::TicksInexact`] under [`Rounding::Exact`] for a fraction of a tick.
    /// [`TimeError::TicksOverflow`] when the tick count does not fit `i64`.
    pub fn to_ticks(self, time_base: Rational, rounding: Rounding) -> Result<i64, TimeError> {
        require_time_base(time_base)?;
        let overflow = TimeError::TicksOverflow {
            value: self,
            time_base,
        };
        let num = i128::from(self.num())
            .checked_mul(i128::from(time_base.den()))
            .ok_or(overflow)?;
        let den = i128::from(self.den())
            .checked_mul(i128::from(time_base.num()))
            .ok_or(overflow)?;
        narrow(div_round(num, den, rounding)).map_err(|fault| match fault {
            Fault::Inexact => TimeError::TicksInexact {
                value: self,
                time_base,
            },
            Fault::Overflow => overflow,
        })
    }
}

/// Re-expresses `ticks` of time base `from` in time base `to`:
/// `round(ticks * from / to)`.
///
/// The product never passes through floating point or an intermediate
/// [`Rational`]. Common factors cancel first, so only a result that itself
/// overflows `i64` fails.
///
/// # Errors
///
/// [`TimeError::NonPositiveTimeBase`] when either time base is zero or less.
/// [`TimeError::RescaleInexact`] under [`Rounding::Exact`] for a fraction of a tick.
/// [`TimeError::RescaleOverflow`] when the result does not fit `i64`.
pub fn rescale(
    ticks: i64,
    from: Rational,
    to: Rational,
    rounding: Rounding,
) -> Result<i64, TimeError> {
    require_time_base(from)?;
    require_time_base(to)?;
    let overflow = TimeError::RescaleOverflow { ticks, from, to };
    // ticks * from / to = ticks * (from.num * to.den) / (from.den * to.num).
    // Both time bases are positive, so each factor is positive.
    let scale_num = u128::from(from.num().unsigned_abs())
        .checked_mul(u128::from(to.den()))
        .ok_or(overflow)?;
    let scale_den = u128::from(from.den())
        .checked_mul(u128::from(to.num().unsigned_abs()))
        .ok_or(overflow)?;
    let common = gcd(scale_num, scale_den);
    let (scale_num, scale_den) = (scale_num / common, scale_den / common);
    let magnitude = u128::from(ticks.unsigned_abs());
    let common = gcd(magnitude, scale_den);
    let (magnitude, scale_den) = (magnitude / common, scale_den / common);
    let product = magnitude.checked_mul(scale_num).ok_or(overflow)?;
    let product = i128::try_from(product).map_err(|_| overflow)?;
    let num = if ticks < 0 { -product } else { product };
    let den = i128::try_from(scale_den).map_err(|_| overflow)?;
    narrow(div_round(num, den, rounding)).map_err(|fault| match fault {
        Fault::Inexact => TimeError::RescaleInexact { ticks, from, to },
        Fault::Overflow => overflow,
    })
}

fn require_time_base(time_base: Rational) -> Result<(), TimeError> {
    if time_base.is_positive() {
        Ok(())
    } else {
        Err(TimeError::NonPositiveTimeBase { time_base })
    }
}

fn narrow(quotient: Result<i128, Fault>) -> Result<i64, Fault> {
    quotient.and_then(|value| i64::try_from(value).map_err(|_| Fault::Overflow))
}

/// Divides `num` by `den` and rounds. `den` must be positive.
fn div_round(num: i128, den: i128, rounding: Rounding) -> Result<i128, Fault> {
    let quotient = num.checked_div_euclid(den).ok_or(Fault::Overflow)?;
    // 0 <= remainder < den, and num = quotient * den + remainder.
    let remainder = num.checked_rem_euclid(den).ok_or(Fault::Overflow)?;
    if remainder == 0 {
        return Ok(quotient);
    }
    let up = match rounding {
        Rounding::Exact => return Err(Fault::Inexact),
        Rounding::Floor => false,
        Rounding::Ceil => true,
        Rounding::TowardZero => num < 0,
        Rounding::HalfEven | Rounding::HalfAwayFromZero => {
            match remainder.cmp(&(den - remainder)) {
                Ordering::Less => false,
                Ordering::Greater => true,
                Ordering::Equal if rounding == Rounding::HalfEven => quotient % 2 != 0,
                Ordering::Equal => num > 0,
            }
        }
    };
    if up {
        quotient.checked_add(1).ok_or(Fault::Overflow)
    } else {
        Ok(quotient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(num: i64, den: u64) -> Rational {
        Rational::new(num, den).unwrap()
    }

    fn round(num: i64, den: u64, rounding: Rounding) -> i64 {
        r(num, den).round(rounding).unwrap()
    }

    #[test]
    fn half_even_matches_python_round_on_ties() {
        // Python: round(Fraction(n, 2)) for n in -5..=5.
        let expected = [-2, -2, -2, -1, 0, 0, 0, 1, 2, 2, 2];
        for (n, want) in (-5..=5).zip(expected) {
            assert_eq!(round(n, 2, Rounding::HalfEven), want, "{n}/2");
        }
    }

    #[test]
    fn half_away_from_zero_on_ties() {
        assert_eq!(round(5, 2, Rounding::HalfAwayFromZero), 3);
        assert_eq!(round(-5, 2, Rounding::HalfAwayFromZero), -3);
        assert_eq!(round(1, 2, Rounding::HalfAwayFromZero), 1);
    }

    #[test]
    fn directed_modes_on_negative_values() {
        assert_eq!(round(-7, 3, Rounding::Floor), -3);
        assert_eq!(round(-7, 3, Rounding::Ceil), -2);
        assert_eq!(round(-7, 3, Rounding::TowardZero), -2);
        assert_eq!(round(7, 3, Rounding::TowardZero), 2);
    }

    #[test]
    fn nearest_modes_away_from_ties() {
        assert_eq!(round(7, 3, Rounding::HalfEven), 2);
        assert_eq!(round(8, 3, Rounding::HalfEven), 3);
        assert_eq!(round(-8, 3, Rounding::HalfAwayFromZero), -3);
    }

    #[test]
    fn exact_refuses_fractions() {
        assert_eq!(round(6, 3, Rounding::Exact), 2);
        assert_eq!(
            r(7, 3).round(Rounding::Exact),
            Err(TimeError::NotAnInteger { value: r(7, 3) })
        );
    }

    #[test]
    fn to_ticks_of_ntsc_frame() {
        let frame = r(1001, 30000);
        assert_eq!(frame.to_ticks(r(1, 90000), Rounding::Exact).unwrap(), 3003);
        // 1001/30000 s in milliseconds is 33.3666..., which rounds to 33.
        assert_eq!(frame.to_ticks(r(1, 1000), Rounding::HalfEven).unwrap(), 33);
        assert!(frame.to_ticks(r(1, 1000), Rounding::Exact).is_err());
    }

    #[test]
    fn to_ticks_refuses_non_positive_time_base() {
        assert_eq!(
            Rational::ONE.to_ticks(r(-1, 1000), Rounding::HalfEven),
            Err(TimeError::NonPositiveTimeBase {
                time_base: r(-1, 1000)
            })
        );
    }

    #[test]
    fn rescale_is_exact_between_compatible_clocks() {
        assert_eq!(
            rescale(1001, r(1, 30000), r(1, 90000), Rounding::Exact).unwrap(),
            3003
        );
        assert_eq!(
            rescale(-1001, r(1, 30000), r(1, 90000), Rounding::Exact).unwrap(),
            -3003
        );
    }

    #[test]
    fn rescale_ties_follow_the_mode() {
        let (from, to) = (r(1, 2000), r(1, 1000));
        assert_eq!(rescale(1, from, to, Rounding::HalfEven).unwrap(), 0);
        assert_eq!(rescale(3, from, to, Rounding::HalfEven).unwrap(), 2);
        assert_eq!(rescale(1, from, to, Rounding::HalfAwayFromZero).unwrap(), 1);
        assert_eq!(rescale(-1, from, to, Rounding::HalfEven).unwrap(), 0);
        assert_eq!(
            rescale(-1, from, to, Rounding::HalfAwayFromZero).unwrap(),
            -1
        );
        assert_eq!(rescale(-3, from, to, Rounding::HalfEven).unwrap(), -2);
    }

    #[test]
    fn rescale_cancels_before_multiplying() {
        // i64::MAX ticks of 1/i64::MAX s is 1 s, which is exact despite the operands.
        let tiny = r(1, i64::MAX as u64);
        assert_eq!(
            rescale(i64::MAX, tiny, Rational::ONE, Rounding::Exact).unwrap(),
            1
        );
    }

    #[test]
    fn rescale_overflow_names_inputs() {
        assert_eq!(
            rescale(i64::MAX, Rational::ONE, r(1, 1000), Rounding::HalfEven),
            Err(TimeError::RescaleOverflow {
                ticks: i64::MAX,
                from: Rational::ONE,
                to: r(1, 1000),
            })
        );
    }

    #[test]
    fn rescale_handles_i64_min() {
        assert_eq!(
            rescale(i64::MIN, r(1, 2), r(1, 1), Rounding::Exact).unwrap(),
            i64::MIN / 2
        );
    }
}
