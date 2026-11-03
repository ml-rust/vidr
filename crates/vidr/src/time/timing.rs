//! Frame timestamps and the track timeline they form.
//!
//! The rules follow the PoC's `full_video_io.py`: pts strictly increase in
//! presentation order, the last frame's duration comes from a fixed chain of
//! sources, and the total duration runs from the first pts to the end of the
//! last frame.

use super::chain::{Durations, durations};
use super::rational::Rational;
use super::rounding::Rounding;
use crate::error::{TimeError, TimingError};

/// Timestamps of one frame, in ticks of its stream's time base.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameTiming {
    /// Presentation timestamp.
    pub pts: i64,
    /// Decode timestamp. `None` when the container carries none. vidr carries
    /// it and does not check it.
    pub dts: Option<i64>,
    /// Display duration. Zero or less means the source carries no duration.
    pub duration: i64,
}

impl FrameTiming {
    /// Creates a frame's timing.
    pub const fn new(pts: i64, dts: Option<i64>, duration: i64) -> Self {
        Self { pts, dts, duration }
    }
}

/// Stream-level inputs to the last-frame duration chain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DurationHints {
    /// Duration of the last frame's packet, in ticks. Zero or less counts as missing.
    pub packet_duration: Option<i64>,
    /// Average frame rate (ffprobe `avg_frame_rate`). Zero or less counts as missing.
    pub average_frame_rate: Option<Rational>,
    /// Base frame rate (ffprobe `r_frame_rate`). The chain reads it only when
    /// the average rate is missing. Zero or less counts as missing.
    pub base_frame_rate: Option<Rational>,
}

/// Where a track starts, in seconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct StartOffset {
    /// Time before the first frame from leading empty edits.
    /// [`EditTimeline::start_offset`](super::EditTimeline::start_offset) gives it.
    pub edit: Rational,
    /// The container's start time (ffprobe `format.start_time`).
    pub container_start: Rational,
}

/// Which step of the chain gave the last frame's duration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LastDurationSource {
    /// The last frame's own duration.
    Frame,
    /// The last packet's duration.
    Packet,
    /// One frame period, `1 / rate`, at the rate named.
    FrameRate(Rational),
    /// The gap between the last two pts.
    PreviousGap,
}

/// A pts that does not exceed the pts before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OrderViolation {
    /// Index of the frame whose pts does not increase.
    pub index: usize,
    /// Pts of the frame before it.
    pub previous: i64,
    /// Pts of the frame at `index`.
    pub pts: i64,
}

/// Yields every frame whose pts does not exceed the pts before it.
pub fn presentation_order_violations(
    frames: &[FrameTiming],
) -> impl Iterator<Item = OrderViolation> {
    frames
        .windows(2)
        .enumerate()
        .filter_map(|(index, pair)| match pair {
            [previous, current] if current.pts <= previous.pts => Some(OrderViolation {
                index: index + 1,
                previous: previous.pts,
                pts: current.pts,
            }),
            _ => None,
        })
}

/// A track's frames in presentation order, with its time base, start, and
/// durations.
///
/// Construction checks every invariant, so a `TrackTiming` always holds at
/// least one frame, strictly increasing pts, and a positive last-frame duration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrackTiming {
    time_base: Rational,
    start: StartOffset,
    frames: Vec<FrameTiming>,
    durations: Durations,
}

impl TrackTiming {
    /// Builds a track from frames in presentation order.
    ///
    /// The last frame's duration comes from the first source present, in the
    /// PoC's order (`full_video_io.py`):
    ///
    /// 1. the last frame's own duration
    /// 2. `hints.packet_duration`
    /// 3. `1 / rate`, at `hints.average_frame_rate`, else `hints.base_frame_rate`
    /// 4. the gap between the last two pts
    ///
    /// # Errors
    ///
    /// - [`TimingError::Time`] with [`TimeError::NonPositiveTimeBase`] for a
    ///   time base of zero or less.
    /// - [`TimingError::NoFrames`] for an empty frame list.
    /// - [`TimingError::NonIncreasingPts`] at the first pts that does not
    ///   increase.
    /// - [`TimingError::NoLastFrameDuration`] when every step of the chain is
    ///   missing.
    pub fn new(
        time_base: Rational,
        start: StartOffset,
        frames: Vec<FrameTiming>,
        hints: DurationHints,
    ) -> Result<Self, TimingError> {
        let durations = durations(time_base, &frames, &hints)?;
        Ok(Self {
            time_base,
            start,
            frames,
            durations,
        })
    }

    /// The stream time base, in seconds per tick.
    pub const fn time_base(&self) -> Rational {
        self.time_base
    }

    /// Where the track starts.
    pub const fn start(&self) -> StartOffset {
        self.start
    }

    /// The frames, in presentation order.
    pub fn frames(&self) -> &[FrameTiming] {
        &self.frames
    }

    /// The number of frames, always at least one.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// The first frame's pts.
    pub const fn first_pts(&self) -> i64 {
        self.durations.first_pts
    }

    /// The last frame's pts.
    pub const fn last_pts(&self) -> i64 {
        self.durations.last_pts
    }

    /// The last frame's duration, in seconds, exact.
    pub const fn last_duration(&self) -> Rational {
        self.durations.last_duration
    }

    /// Which step of the chain gave the last frame's duration.
    pub const fn last_duration_source(&self) -> LastDurationSource {
        self.durations.source
    }

    /// The last frame's duration in ticks of the time base.
    ///
    /// A `1 / rate` duration can fall between ticks. The PoC writes it as
    /// `round(duration / clock)` (`ai_upscale.py`), which
    /// [`Rounding::HalfEven`] reproduces.
    ///
    /// # Errors
    ///
    /// [`TimeError::TicksInexact`] under [`Rounding::Exact`] for a fraction of a
    /// tick. [`TimeError::TicksOverflow`] when the tick count does not fit.
    pub fn last_duration_ticks(&self, rounding: Rounding) -> Result<i64, TimeError> {
        self.durations
            .last_duration
            .to_ticks(self.time_base, rounding)
    }

    /// Total duration in seconds: `(last pts - first pts) * time base + last
    /// duration`. Exact.
    pub const fn total_duration(&self) -> Rational {
        self.durations.total
    }

    /// When the first frame shows, in seconds: the edit offset plus the first
    /// pts.
    ///
    /// # Errors
    ///
    /// [`TimeError::Overflow`] when the sum does not fit.
    pub fn presentation_start(&self) -> Result<Rational, TimeError> {
        Rational::from_ticks(self.durations.first_pts, self.time_base)?.checked_add(self.start.edit)
    }

    /// The first frame's start relative to the container start, in seconds.
    /// The PoC's video offset (`full_video_io.py`, `mux_audio`).
    ///
    /// # Errors
    ///
    /// [`TimeError::Overflow`] when the difference does not fit.
    pub fn video_offset(&self) -> Result<Rational, TimeError> {
        self.presentation_start()?
            .checked_sub(self.start.container_start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(num: i64, den: u64) -> Rational {
        Rational::new(num, den).unwrap()
    }

    fn frames(pts: &[i64], duration: i64) -> Vec<FrameTiming> {
        pts.iter()
            .map(|&pts| FrameTiming::new(pts, None, duration))
            .collect()
    }

    fn track(pts: &[i64], duration: i64, hints: DurationHints) -> Result<TrackTiming, TimingError> {
        TrackTiming::new(
            r(1, 1000),
            StartOffset::default(),
            frames(pts, duration),
            hints,
        )
    }

    #[test]
    fn total_duration_spans_first_pts_to_last_frame_end() {
        let track = track(&[100, 140, 180], 40, DurationHints::default()).unwrap();
        assert_eq!(track.last_duration_source(), LastDurationSource::Frame);
        assert_eq!(track.total_duration(), r(120, 1000));
        assert_eq!(track.frame_count(), 3);
    }

    #[test]
    fn average_rate_precedes_base_rate() {
        let hints = DurationHints {
            packet_duration: Some(0),
            average_frame_rate: Some(r(24, 1)),
            base_frame_rate: Some(r(25, 1)),
        };
        let track = track(&[0, 40], -1, hints).unwrap();
        assert_eq!(
            track.last_duration_source(),
            LastDurationSource::FrameRate(r(24, 1))
        );
        assert_eq!(track.last_duration(), r(1, 24));
        // 1/24 s is 41.666... ms, which rounds half-even to 42 ticks.
        assert_eq!(track.last_duration_ticks(Rounding::HalfEven).unwrap(), 42);
    }

    #[test]
    fn previous_gap_is_the_last_resort() {
        let hints = DurationHints {
            average_frame_rate: Some(Rational::ZERO),
            ..DurationHints::default()
        };
        let track = track(&[0, 33, 70], 0, hints).unwrap();
        assert_eq!(
            track.last_duration_source(),
            LastDurationSource::PreviousGap
        );
        assert_eq!(track.last_duration(), r(37, 1000));
    }

    #[test]
    fn single_frame_without_hints_names_missing_inputs() {
        assert_eq!(
            track(&[7], 0, DurationHints::default()),
            Err(TimingError::NoLastFrameDuration {
                frame_duration: 0,
                packet_duration: None,
                average_frame_rate: None,
                base_frame_rate: None,
                frame_count: 1,
            })
        );
    }

    #[test]
    fn equal_pts_is_refused() {
        assert_eq!(
            track(&[0, 40, 40], 40, DurationHints::default()),
            Err(TimingError::NonIncreasingPts {
                index: 2,
                previous: 40,
                pts: 40,
            })
        );
    }

    #[test]
    fn empty_track_is_refused() {
        assert_eq!(
            track(&[], 40, DurationHints::default()),
            Err(TimingError::NoFrames)
        );
    }

    #[test]
    fn negative_pts_are_valid() {
        let track = track(&[-80, -40, 0], 40, DurationHints::default()).unwrap();
        assert_eq!(track.total_duration(), r(120, 1000));
        assert_eq!(track.first_pts(), -80);
    }

    #[test]
    fn violations_report_every_break() {
        let list = frames(&[0, 2, 1, 1, 5], 1);
        let found: Vec<_> = presentation_order_violations(&list).collect();
        assert_eq!(found.len(), 2);
        assert_eq!(
            found[0],
            OrderViolation {
                index: 2,
                previous: 2,
                pts: 1
            }
        );
        assert_eq!(found[1].index, 3);
    }
}
