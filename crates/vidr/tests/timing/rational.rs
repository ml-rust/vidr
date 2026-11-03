use vidr::{Operation, Rational, Rounding, TimeError, rescale};

use super::r;

#[test]
fn overflow_is_an_error_never_a_wrap() {
    let max = Rational::from_integer(i64::MAX);
    assert_eq!(
        max.checked_add(Rational::ONE),
        Err(TimeError::Overflow {
            operation: Operation::Add,
            lhs: max,
            rhs: Some(Rational::ONE),
        })
    );
    assert!(Rational::from_integer(i64::MIN).checked_neg().is_err());
    assert!(r(1, u64::MAX).checked_mul(r(1, u64::MAX)).is_err());
    assert!(rescale(i64::MAX, Rational::ONE, r(1, 90000), Rounding::Floor).is_err());
}

#[test]
fn negative_pts_rescale_symmetrically_under_half_even() {
    let (from, to) = (r(1, 90000), r(1, 1000));
    // 45 ticks of 1/90000 s are exactly half a millisecond.
    assert_eq!(rescale(45, from, to, Rounding::HalfEven), Ok(0));
    assert_eq!(rescale(-45, from, to, Rounding::HalfEven), Ok(0));
    assert_eq!(rescale(135, from, to, Rounding::HalfEven), Ok(2));
    assert_eq!(rescale(-135, from, to, Rounding::HalfEven), Ok(-2));
    assert_eq!(rescale(-45, from, to, Rounding::HalfAwayFromZero), Ok(-1));
    assert_eq!(rescale(-46, from, to, Rounding::Floor), Ok(-1));
    assert_eq!(rescale(-46, from, to, Rounding::TowardZero), Ok(0));
}

#[test]
fn half_even_ties_round_to_even_like_python() {
    // Python: [round(Fraction(n, 2)) for n in (1, 3, 5, 7, -7)] == [0, 2, 2, 4, -4].
    let cases = [(1, 0), (3, 2), (5, 2), (7, 4), (-7, -4)];
    for (num, want) in cases {
        assert_eq!(r(num, 2).round(Rounding::HalfEven), Ok(want), "{num}/2");
    }
}

#[test]
fn exact_rounding_refuses_fractions() {
    assert_eq!(
        rescale(1, r(1, 30), r(1, 1000), Rounding::Exact),
        Err(TimeError::RescaleInexact {
            ticks: 1,
            from: r(1, 30),
            to: r(1, 1000),
        })
    );
    assert_eq!(rescale(3, r(1, 30), r(1, 1000), Rounding::Exact), Ok(100));
}

#[test]
fn time_base_must_be_positive() {
    assert_eq!(
        rescale(1, Rational::ZERO, r(1, 1000), Rounding::HalfEven),
        Err(TimeError::NonPositiveTimeBase {
            time_base: Rational::ZERO
        })
    );
}

#[test]
fn comparison_is_exact_for_close_values() {
    let a = r(i64::MAX - 1, i64::MAX as u64);
    let b = r(i64::MAX - 2, i64::MAX as u64 - 1);
    // (n - 1) / n > (n - 2) / (n - 1), which floating point cannot tell apart.
    assert!(a > b);
    assert_ne!(a, b);
}
