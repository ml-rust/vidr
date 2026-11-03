//! Timing checks: time bases, pts order, pts values, and total duration.

use super::DURATION_TOLERANCE;
use super::StreamFacts;
use super::deviation::{Deviation, Side};
use crate::error::{TimeError, TimingError};
use crate::time::{
    Durations, Rational, Rounding, checked_tick_add, checked_tick_sub, durations,
    presentation_order_violations, rescale,
};

/// Pushes one deviation per timing mismatch.
pub(super) fn check(source: &StreamFacts, output: &StreamFacts, deviations: &mut Vec<Deviation>) {
    // `&` runs both sides, so each one reports.
    let bases =
        time_base(Side::Source, source, deviations) & time_base(Side::Output, output, deviations);
    let ordered = order(Side::Source, source, deviations) & order(Side::Output, output, deviations);
    if !bases {
        return;
    }
    pts(source, output, deviations);
    if ordered {
        duration(source, output, deviations);
    }
}

fn time_base(side: Side, facts: &StreamFacts, deviations: &mut Vec<Deviation>) -> bool {
    if facts.time_base.is_positive() {
        return true;
    }
    let error = TimeError::NonPositiveTimeBase {
        time_base: facts.time_base,
    };
    deviations.push(Deviation::Timing {
        side,
        error: error.into(),
    });
    false
}

fn order(side: Side, facts: &StreamFacts, deviations: &mut Vec<Deviation>) -> bool {
    let mut violations = presentation_order_violations(&facts.frames);
    let Some(first) = violations.next() else {
        return true;
    };
    deviations.push(Deviation::PtsOrder {
        side,
        index: first.index,
        previous: first.previous,
        pts: first.pts,
        violations: 1 + violations.count(),
    });
    false
}

/// Compares pts relative to each stream's first frame, over the frames both
/// streams hold.
fn pts(source: &StreamFacts, output: &StreamFacts, deviations: &mut Vec<Deviation>) {
    let (Some(source_origin), Some(output_origin)) = (source.frames.first(), output.frames.first())
    else {
        return;
    };
    let mut first = None;
    let mut mismatches = 0_usize;
    for (index, (from, to)) in source.frames.iter().zip(&output.frames).enumerate() {
        // Half-even: the PoC writes `round(timestamp / clock)` (`ai_upscale.py`).
        let expected = checked_tick_sub(from.pts, source_origin.pts).and_then(|ticks| {
            rescale(
                ticks,
                source.time_base,
                output.time_base,
                Rounding::HalfEven,
            )
        });
        let expected = match expected {
            Ok(expected) => expected,
            Err(error) => return push_timing(Side::Source, error.into(), deviations),
        };
        let actual = match checked_tick_sub(to.pts, output_origin.pts) {
            Ok(actual) => actual,
            Err(error) => return push_timing(Side::Output, error.into(), deviations),
        };
        if expected != actual {
            mismatches += 1;
            if first.is_none() {
                first = Some((index, expected, actual));
            }
        }
    }
    if let Some((index, expected, actual)) = first {
        deviations.push(Deviation::Pts {
            index,
            expected,
            output: actual,
            mismatches,
            time_base: output.time_base,
        });
    }
}

fn duration(source: &StreamFacts, output: &StreamFacts, deviations: &mut Vec<Deviation>) {
    let source_durations = durations(source.time_base, &source.frames, &source.hints);
    let output_durations = durations(output.time_base, &output.frames, &output.hints);
    let (source_durations, output_durations) = match (source_durations, output_durations) {
        (Ok(source_durations), Ok(output_durations)) => (source_durations, output_durations),
        (source_result, output_result) => {
            for (side, result) in [(Side::Source, source_result), (Side::Output, output_result)] {
                if let Err(error) = result {
                    push_timing(side, error, deviations);
                }
            }
            return;
        }
    };
    let expected = if source.time_base == output.time_base {
        exact_total(source.time_base, &source_durations).map(|total| (total, None))
    } else {
        Ok((source_durations.total, Some(DURATION_TOLERANCE)))
    };
    let (expected, tolerance) = match expected {
        Ok(expected) => expected,
        Err(error) => return push_timing(Side::Source, error.into(), deviations),
    };
    let actual = output_durations.total;
    let within = match tolerance {
        None => actual == expected,
        Some(tolerance) => match actual.checked_sub(expected).and_then(Rational::checked_abs) {
            Ok(difference) => difference <= tolerance,
            Err(error) => return push_timing(Side::Output, error.into(), deviations),
        },
    };
    if !within {
        deviations.push(Deviation::Duration {
            expected,
            output: actual,
            tolerance,
        });
    }
}

/// The total an output in the source time base must have. The pipeline
/// copies every tick and writes the last-frame duration as
/// `round(duration / clock)` (`ai_upscale.py`), so half-even.
fn exact_total(time_base: Rational, source: &Durations) -> Result<Rational, TimeError> {
    let last = source
        .last_duration
        .to_ticks(time_base, Rounding::HalfEven)?;
    let span = checked_tick_sub(source.last_pts, source.first_pts)?;
    Rational::from_ticks(checked_tick_add(span, last)?, time_base)
}

fn push_timing(side: Side, error: TimingError, deviations: &mut Vec<Deviation>) {
    deviations.push(Deviation::Timing { side, error });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Dimensions, Geometry, Rotation, SampleAspectRatio};
    use crate::time::{DurationHints, FrameTiming};

    fn facts(time_base: Rational, pts: &[i64], duration: i64, hints: DurationHints) -> StreamFacts {
        StreamFacts::new(
            Geometry::new(
                Dimensions::new(64, 64).unwrap(),
                Rotation::Upright,
                SampleAspectRatio::SQUARE,
            ),
            time_base,
            pts.iter()
                .map(|&pts| FrameTiming::new(pts, None, duration))
                .collect(),
            hints,
        )
    }

    fn run(source: &StreamFacts, output: &StreamFacts) -> Vec<Deviation> {
        let mut deviations = Vec::new();
        check(source, output, &mut deviations);
        deviations
    }

    #[test]
    fn rate_derived_last_duration_rounds_half_even_in_one_time_base() {
        let base = Rational::new(1, 1000).unwrap();
        let hints = DurationHints {
            average_frame_rate: Some(Rational::from_integer(24)),
            ..DurationHints::default()
        };
        // Source last duration is 1/24 s, 41.67 ticks. The writer stores 42.
        let source = facts(base, &[0, 42, 83], 0, hints);
        let output = facts(base, &[0, 42, 83], 42, DurationHints::default());
        assert!(run(&source, &output).is_empty());
        let short = facts(base, &[0, 42, 83], 41, DurationHints::default());
        assert!(matches!(
            run(&source, &short).as_slice(),
            [Deviation::Duration {
                tolerance: None,
                ..
            }]
        ));
    }

    #[test]
    fn non_positive_time_base_skips_dependent_checks() {
        let source = facts(Rational::ZERO, &[0, 1], 1, DurationHints::default());
        let output = facts(
            Rational::new(1, 30).unwrap(),
            &[0, 1],
            1,
            DurationHints::default(),
        );
        let deviations = run(&source, &output);
        assert_eq!(deviations.len(), 1, "{deviations:?}");
        assert!(matches!(
            deviations[0],
            Deviation::Timing {
                side: Side::Source,
                ..
            }
        ));
    }

    #[test]
    fn empty_output_reports_no_frames() {
        let base = Rational::new(1, 30).unwrap();
        let source = facts(base, &[0, 1], 1, DurationHints::default());
        let output = facts(base, &[], 1, DurationHints::default());
        assert_eq!(
            run(&source, &output),
            [Deviation::Timing {
                side: Side::Output,
                error: TimingError::NoFrames,
            }]
        );
    }
}
