//! Helpers shared by the timing tests.

use vidr::Rational;

/// Builds a rational the test knows is valid.
pub(crate) fn r(num: i64, den: u64) -> Rational {
    Rational::new(num, den).unwrap()
}
