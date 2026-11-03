//! Errors in exact arithmetic and tick conversion.

use std::error::Error as StdError;
use std::fmt;

use crate::time::Rational;

/// The arithmetic step that overflowed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operation {
    /// `lhs + rhs`.
    Add,
    /// `lhs - rhs`.
    Subtract,
    /// `lhs * rhs`.
    Multiply,
    /// `lhs / rhs`.
    Divide,
    /// `-lhs`.
    Negate,
    /// `|lhs|`.
    Absolute,
    /// `lhs` to the nearest integer.
    Round,
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Add => "addition",
            Self::Subtract => "subtraction",
            Self::Multiply => "multiplication",
            Self::Divide => "division",
            Self::Negate => "negation",
            Self::Absolute => "absolute value",
            Self::Round => "rounding",
        })
    }
}

/// An error in exact rational arithmetic or in a tick conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeError {
    /// A rational with a zero denominator.
    ZeroDenominator {
        /// The numerator passed with the zero denominator.
        numerator: i64,
    },
    /// A result that does not fit an `i64` numerator and a `u64` denominator.
    Overflow {
        /// The step that overflowed.
        operation: Operation,
        /// The left operand.
        lhs: Rational,
        /// The right operand. `None` for a unary step.
        rhs: Option<Rational>,
    },
    /// A division by zero.
    DivisionByZero {
        /// The value divided.
        dividend: Rational,
    },
    /// A time base of zero or less.
    NonPositiveTimeBase {
        /// The time base passed.
        time_base: Rational,
    },
    /// A value that is not an integer under [`Rounding::Exact`](crate::time::Rounding::Exact).
    NotAnInteger {
        /// The value passed.
        value: Rational,
    },
    /// A duration in seconds whose tick count overflows `i64`.
    TicksOverflow {
        /// The value in seconds.
        value: Rational,
        /// The target time base.
        time_base: Rational,
    },
    /// A duration in seconds that is not a whole number of ticks under exact rounding.
    TicksInexact {
        /// The value in seconds.
        value: Rational,
        /// The target time base.
        time_base: Rational,
    },
    /// A rescaled timestamp that overflows `i64`.
    RescaleOverflow {
        /// The timestamp in ticks of `from`.
        ticks: i64,
        /// The source time base.
        from: Rational,
        /// The target time base.
        to: Rational,
    },
    /// A rescaled timestamp that is not a whole number of ticks under exact rounding.
    RescaleInexact {
        /// The timestamp in ticks of `from`.
        ticks: i64,
        /// The source time base.
        from: Rational,
        /// The target time base.
        to: Rational,
    },
}

impl fmt::Display for TimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDenominator { numerator } => write!(
                f,
                "rational {numerator}/0 has a zero denominator. Pass a nonzero denominator."
            ),
            Self::Overflow {
                operation,
                lhs,
                rhs: Some(rhs),
            } => write!(
                f,
                "{operation} of {lhs} and {rhs} overflows an i64 numerator or u64 denominator. \
                 Use smaller timestamps or a coarser time base."
            ),
            Self::Overflow {
                operation,
                lhs,
                rhs: None,
            } => write!(
                f,
                "{operation} of {lhs} overflows an i64 numerator or u64 denominator. \
                 Use smaller timestamps or a coarser time base."
            ),
            Self::DivisionByZero { dividend } => {
                write!(f, "division of {dividend} by zero. Pass a nonzero divisor.")
            }
            Self::NonPositiveTimeBase { time_base } => write!(
                f,
                "time base {time_base} is not positive. Read the time base from the stream header."
            ),
            Self::NotAnInteger { value } => write!(
                f,
                "{value} is not an integer under exact rounding. Pass a rounding mode that rounds."
            ),
            Self::TicksOverflow { value, time_base } => write!(
                f,
                "{value} s in time base {time_base} overflows i64 ticks. Use a coarser time base."
            ),
            Self::TicksInexact { value, time_base } => write!(
                f,
                "{value} s is not a whole number of ticks in time base {time_base}. \
                 Pass a rounding mode that rounds, or a time base that holds the value exactly."
            ),
            Self::RescaleOverflow { ticks, from, to } => write!(
                f,
                "rescaling {ticks} ticks from time base {from} to {to} overflows i64. \
                 Use a coarser target time base."
            ),
            Self::RescaleInexact { ticks, from, to } => write!(
                f,
                "{ticks} ticks in time base {from} are not a whole number of ticks in {to}. \
                 Pass a rounding mode that rounds, or a target time base that holds the value exactly."
            ),
        }
    }
}

impl StdError for TimeError {}
