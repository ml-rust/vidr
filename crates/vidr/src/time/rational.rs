//! Exact rational numbers for timestamps, time bases, and aspect ratios.

use std::cmp::Ordering;
use std::fmt;

use crate::error::{Operation, TimeError};

/// An exact rational number in lowest terms with a positive denominator.
///
/// Arithmetic is checked. A result that does not fit an `i64` numerator and a
/// `u64` denominator returns [`TimeError::Overflow`]. No value passes through
/// floating point.
///
/// Equality compares the reduced form. Ordering cross-multiplies in `i128`,
/// which holds every product of an `i64` and a `u64` exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rational {
    num: i64,
    den: u64,
}

impl Rational {
    /// Zero, `0/1`.
    pub const ZERO: Self = Self { num: 0, den: 1 };
    /// One, `1/1`.
    pub const ONE: Self = Self { num: 1, den: 1 };

    /// Creates `num / den` in lowest terms.
    ///
    /// # Errors
    ///
    /// [`TimeError::ZeroDenominator`] when `den` is zero.
    pub const fn new(num: i64, den: u64) -> Result<Self, TimeError> {
        if den == 0 {
            return Err(TimeError::ZeroDenominator { numerator: num });
        }
        match Self::from_parts(num < 0, num.unsigned_abs() as u128, den as u128) {
            Some(value) => Ok(value),
            // Not reached: reducing an `i64 / u64` pair never grows either part.
            None => Err(TimeError::ZeroDenominator { numerator: num }),
        }
    }

    /// Creates `value / 1`.
    pub const fn from_integer(value: i64) -> Self {
        Self { num: value, den: 1 }
    }

    /// Creates the time of `ticks` ticks of `time_base`, in seconds.
    ///
    /// # Errors
    ///
    /// [`TimeError::Overflow`] when the product does not fit.
    pub fn from_ticks(ticks: i64, time_base: Self) -> Result<Self, TimeError> {
        Self::from_integer(ticks).checked_mul(time_base)
    }

    /// The numerator, sign included.
    pub const fn num(self) -> i64 {
        self.num
    }

    /// The denominator, always positive.
    pub const fn den(self) -> u64 {
        self.den
    }

    /// Returns `true` for zero.
    pub const fn is_zero(self) -> bool {
        self.num == 0
    }

    /// Returns `true` above zero.
    pub const fn is_positive(self) -> bool {
        self.num > 0
    }

    /// Returns `true` below zero.
    pub const fn is_negative(self) -> bool {
        self.num < 0
    }

    /// Returns `self + rhs`.
    ///
    /// # Errors
    ///
    /// [`TimeError::Overflow`] when the sum does not fit.
    pub fn checked_add(self, rhs: Self) -> Result<Self, TimeError> {
        self.combine(rhs, Operation::Add, i128::checked_add)
    }

    /// Returns `self - rhs`.
    ///
    /// # Errors
    ///
    /// [`TimeError::Overflow`] when the difference does not fit.
    pub fn checked_sub(self, rhs: Self) -> Result<Self, TimeError> {
        self.combine(rhs, Operation::Subtract, i128::checked_sub)
    }

    /// Returns `self * rhs`.
    ///
    /// # Errors
    ///
    /// [`TimeError::Overflow`] when the product does not fit.
    pub fn checked_mul(self, rhs: Self) -> Result<Self, TimeError> {
        let overflow = overflow(Operation::Multiply, self, Some(rhs));
        let num = i128::from(self.num)
            .checked_mul(i128::from(rhs.num))
            .ok_or(overflow)?;
        let den = u128::from(self.den)
            .checked_mul(u128::from(rhs.den))
            .ok_or(overflow)?;
        Self::from_parts(num < 0, num.unsigned_abs(), den).ok_or(overflow)
    }

    /// Returns `self / rhs`.
    ///
    /// # Errors
    ///
    /// [`TimeError::DivisionByZero`] when `rhs` is zero.
    /// [`TimeError::Overflow`] when the quotient does not fit.
    pub fn checked_div(self, rhs: Self) -> Result<Self, TimeError> {
        if rhs.is_zero() {
            return Err(TimeError::DivisionByZero { dividend: self });
        }
        let overflow = overflow(Operation::Divide, self, Some(rhs));
        let num = u128::from(self.num.unsigned_abs())
            .checked_mul(u128::from(rhs.den))
            .ok_or(overflow)?;
        let den = u128::from(self.den)
            .checked_mul(u128::from(rhs.num.unsigned_abs()))
            .ok_or(overflow)?;
        Self::from_parts(self.is_negative() != rhs.is_negative(), num, den).ok_or(overflow)
    }

    /// Returns `1 / self`.
    ///
    /// # Errors
    ///
    /// [`TimeError::DivisionByZero`] when `self` is zero.
    /// [`TimeError::Overflow`] when the reciprocal does not fit.
    pub fn recip(self) -> Result<Self, TimeError> {
        Self::ONE.checked_div(self)
    }

    /// Returns `-self`.
    ///
    /// # Errors
    ///
    /// [`TimeError::Overflow`] when `self` has numerator `i64::MIN`.
    pub fn checked_neg(self) -> Result<Self, TimeError> {
        Self::from_parts(
            self.is_positive(),
            u128::from(self.num.unsigned_abs()),
            u128::from(self.den),
        )
        .ok_or(overflow(Operation::Negate, self, None))
    }

    /// Returns `|self|`.
    ///
    /// # Errors
    ///
    /// [`TimeError::Overflow`] when `self` has numerator `i64::MIN`.
    pub fn checked_abs(self) -> Result<Self, TimeError> {
        Self::from_parts(
            false,
            u128::from(self.num.unsigned_abs()),
            u128::from(self.den),
        )
        .ok_or(overflow(Operation::Absolute, self, None))
    }

    /// Adds or subtracts over the least common denominator.
    fn combine(
        self,
        rhs: Self,
        operation: Operation,
        apply: fn(i128, i128) -> Option<i128>,
    ) -> Result<Self, TimeError> {
        let overflow = overflow(operation, self, Some(rhs));
        let lhs_den = u128::from(self.den);
        let rhs_den = u128::from(rhs.den);
        // Both denominators are positive, so `common` is at least 1.
        let common = gcd(lhs_den, rhs_den);
        let lhs_scale = i128::try_from(rhs_den / common).map_err(|_| overflow)?;
        let rhs_scale = i128::try_from(lhs_den / common).map_err(|_| overflow)?;
        let lhs_num = i128::from(self.num)
            .checked_mul(lhs_scale)
            .ok_or(overflow)?;
        let rhs_num = i128::from(rhs.num).checked_mul(rhs_scale).ok_or(overflow)?;
        let num = apply(lhs_num, rhs_num).ok_or(overflow)?;
        let den = lhs_den.checked_mul(rhs_den / common).ok_or(overflow)?;
        Self::from_parts(num < 0, num.unsigned_abs(), den).ok_or(overflow)
    }

    /// Reduces `±num / den` and narrows it to `i64 / u64`.
    ///
    /// Returns `None` when `den` is zero or a reduced part does not fit.
    const fn from_parts(negative: bool, num: u128, den: u128) -> Option<Self> {
        if den == 0 {
            return None;
        }
        let common = gcd(num, den);
        let num = num / common;
        let den = den / common;
        if den > u64::MAX as u128 {
            return None;
        }
        let limit = if negative {
            i64::MIN.unsigned_abs() as u128
        } else {
            i64::MAX as u128
        };
        if num > limit {
            return None;
        }
        // `num` is at most 2^63 here, so both casts are exact.
        let num = if negative {
            (0 - num as i128) as i64
        } else {
            num as i64
        };
        Some(Self {
            num,
            den: den as u64,
        })
    }
}

impl Default for Rational {
    fn default() -> Self {
        Self::ZERO
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> Ordering {
        // |i64| * u64 stays below 2^127, so neither product overflows i128.
        let lhs = i128::from(self.num) * i128::from(other.den);
        let rhs = i128::from(other.num) * i128::from(self.den);
        lhs.cmp(&rhs)
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.num, self.den)
    }
}

/// Greatest common divisor. `gcd(0, b)` is `b`.
pub(crate) const fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let rest = a % b;
        a = b;
        b = rest;
    }
    a
}

const fn overflow(operation: Operation, lhs: Rational, rhs: Option<Rational>) -> TimeError {
    TimeError::Overflow {
        operation,
        lhs,
        rhs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(num: i64, den: u64) -> Rational {
        Rational::new(num, den).unwrap()
    }

    #[test]
    fn new_reduces_and_keeps_sign_on_numerator() {
        let value = r(-6, 4);
        assert_eq!((value.num(), value.den()), (-3, 2));
        assert_eq!(r(0, 7), Rational::ZERO);
    }

    #[test]
    fn new_reduces_i64_min() {
        let value = r(i64::MIN, 1 << 63);
        assert_eq!((value.num(), value.den()), (-1, 1));
        let value = r(i64::MIN, 2);
        assert_eq!((value.num(), value.den()), (i64::MIN / 2, 1));
    }

    #[test]
    fn new_refuses_zero_denominator() {
        assert_eq!(
            Rational::new(5, 0),
            Err(TimeError::ZeroDenominator { numerator: 5 })
        );
    }

    #[test]
    fn add_overflow_names_operands() {
        let max = Rational::from_integer(i64::MAX);
        assert_eq!(
            max.checked_add(Rational::ONE),
            Err(TimeError::Overflow {
                operation: Operation::Add,
                lhs: max,
                rhs: Some(Rational::ONE),
            })
        );
    }

    #[test]
    fn add_reduces_through_common_denominator() {
        assert_eq!(r(1, 6).checked_add(r(1, 3)).unwrap(), r(1, 2));
        assert_eq!(r(1, 6).checked_sub(r(1, 3)).unwrap(), r(-1, 6));
    }

    #[test]
    fn sub_reaches_i64_min_without_overflow() {
        let value = Rational::from_integer(-1)
            .checked_sub(Rational::from_integer(i64::MAX))
            .unwrap();
        assert_eq!(value.num(), i64::MIN);
    }

    #[test]
    fn mul_and_div_cancel_large_factors() {
        let big = r(1, u64::MAX);
        assert_eq!(
            big.checked_mul(Rational::from_integer(3)).unwrap(),
            r(1, u64::MAX / 3)
        );
        assert_eq!(big.checked_div(big).unwrap(), Rational::ONE);
        assert!(big.checked_mul(big).is_err());
    }

    #[test]
    fn div_by_zero_is_an_error() {
        assert_eq!(
            Rational::ONE.checked_div(Rational::ZERO),
            Err(TimeError::DivisionByZero {
                dividend: Rational::ONE
            })
        );
        assert!(Rational::ZERO.recip().is_err());
    }

    #[test]
    fn negating_i64_min_overflows() {
        let min = Rational::from_integer(i64::MIN);
        assert!(min.checked_neg().is_err());
        assert!(min.checked_abs().is_err());
        assert_eq!(r(-3, 2).checked_neg().unwrap(), r(3, 2));
        assert_eq!(r(-3, 2).checked_abs().unwrap(), r(3, 2));
    }

    #[test]
    fn recip_moves_sign_to_numerator() {
        let value = r(-2, 3).recip().unwrap();
        assert_eq!((value.num(), value.den()), (-3, 2));
    }

    #[test]
    fn ordering_cross_multiplies_without_overflow() {
        let near_one = r(i64::MAX, u64::MAX >> 1);
        assert_eq!(near_one, Rational::ONE);
        assert!(r(i64::MAX - 1, i64::MAX as u64) < Rational::ONE);
        assert!(r(i64::MIN, 1) < r(-1, u64::MAX));
        assert!(r(1, 3) > r(1, 4));
    }

    #[test]
    fn displays_as_fraction() {
        assert_eq!(r(1001, 30000).to_string(), "1001/30000");
        assert_eq!(Rational::from_integer(-4).to_string(), "-4/1");
    }
}
