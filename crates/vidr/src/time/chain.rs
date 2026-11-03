//! The checked timeline facts of a frame list and the last-frame duration
//! chain.

use super::rational::Rational;
use super::ticks::checked_tick_sub;
use super::timing::{
    DurationHints, FrameTiming, LastDurationSource, presentation_order_violations,
};
use crate::error::{TimeError, TimingError};

/// The checked timeline facts of a frame list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Durations {
    pub(crate) first_pts: i64,
    pub(crate) last_pts: i64,
    pub(crate) last_duration: Rational,
    pub(crate) source: LastDurationSource,
    pub(crate) total: Rational,
}

/// Checks a frame list and derives its durations. [`TrackTiming::new`]
/// documents the rules and errors.
pub(crate) fn durations(
    time_base: Rational,
    frames: &[FrameTiming],
    hints: &DurationHints,
) -> Result<Durations, TimingError> {
    if !time_base.is_positive() {
        return Err(TimeError::NonPositiveTimeBase { time_base }.into());
    }
    let (Some(first), Some(last)) = (frames.first(), frames.last()) else {
        return Err(TimingError::NoFrames);
    };
    if let Some(violation) = presentation_order_violations(frames).next() {
        return Err(violation.into());
    }
    let previous = frames
        .len()
        .checked_sub(2)
        .and_then(|index| frames.get(index));
    let (last_duration, source) =
        last_frame_duration(time_base, last, previous, hints, frames.len())?;
    let span = Rational::from_ticks(checked_tick_sub(last.pts, first.pts)?, time_base)?;
    Ok(Durations {
        first_pts: first.pts,
        last_pts: last.pts,
        last_duration,
        source,
        total: span.checked_add(last_duration)?,
    })
}

/// Runs the last-frame duration chain.
fn last_frame_duration(
    time_base: Rational,
    last: &FrameTiming,
    previous: Option<&FrameTiming>,
    hints: &DurationHints,
    frame_count: usize,
) -> Result<(Rational, LastDurationSource), TimingError> {
    if last.duration > 0 {
        let duration = Rational::from_ticks(last.duration, time_base)?;
        return Ok((duration, LastDurationSource::Frame));
    }
    if let Some(ticks) = hints.packet_duration.filter(|ticks| *ticks > 0) {
        let duration = Rational::from_ticks(ticks, time_base)?;
        return Ok((duration, LastDurationSource::Packet));
    }
    let rate = hints
        .average_frame_rate
        .filter(|rate| rate.is_positive())
        .or_else(|| hints.base_frame_rate.filter(|rate| rate.is_positive()));
    if let Some(rate) = rate {
        return Ok((rate.recip()?, LastDurationSource::FrameRate(rate)));
    }
    if let Some(previous) = previous {
        // Strictly increasing pts make this gap positive.
        let gap = checked_tick_sub(last.pts, previous.pts)?;
        let duration = Rational::from_ticks(gap, time_base)?;
        return Ok((duration, LastDurationSource::PreviousGap));
    }
    Err(TimingError::NoLastFrameDuration {
        frame_duration: last.duration,
        packet_duration: hints.packet_duration,
        average_frame_rate: hints.average_frame_rate,
        base_frame_rate: hints.base_frame_rate,
        frame_count,
    })
}
