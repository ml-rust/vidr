//! Tick arithmetic on `i64` timestamps.

use super::rational::Rational;
use crate::error::{Operation, TimeError};

/// Returns `lhs - rhs` in ticks.
///
/// # Errors
///
/// [`TimeError::Overflow`] when the difference does not fit `i64`.
pub(crate) fn checked_tick_sub(lhs: i64, rhs: i64) -> Result<i64, TimeError> {
    lhs.checked_sub(rhs)
        .ok_or(tick_overflow(Operation::Subtract, lhs, rhs))
}

/// Returns `lhs + rhs` in ticks.
///
/// # Errors
///
/// [`TimeError::Overflow`] when the sum does not fit `i64`.
pub(crate) fn checked_tick_add(lhs: i64, rhs: i64) -> Result<i64, TimeError> {
    lhs.checked_add(rhs)
        .ok_or(tick_overflow(Operation::Add, lhs, rhs))
}

fn tick_overflow(operation: Operation, lhs: i64, rhs: i64) -> TimeError {
    TimeError::Overflow {
        operation,
        lhs: Rational::from_integer(lhs),
        rhs: Some(Rational::from_integer(rhs)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_helpers_report_overflow() {
        assert_eq!(checked_tick_sub(5, 7).unwrap(), -2);
        assert!(checked_tick_sub(i64::MIN, 1).is_err());
        assert!(checked_tick_add(i64::MAX, 1).is_err());
    }
}
