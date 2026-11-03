//! Errors in frame timing and edit lists.

use std::error::Error as StdError;
use std::fmt;

use super::TimeError;
use super::support::Shown;
use crate::time::{OrderViolation, Rational};

/// An error in a track's frame timing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimingError {
    /// A track with no frames.
    NoFrames,
    /// A pts that does not exceed the pts before it in presentation order.
    NonIncreasingPts {
        /// Index of the frame whose pts does not increase.
        index: usize,
        /// Pts of the frame before it.
        previous: i64,
        /// Pts of the frame at `index`.
        pts: i64,
    },
    /// Every step of the last-frame duration chain is missing.
    NoLastFrameDuration {
        /// The last frame's own duration, in ticks. Zero or less counts as missing.
        frame_duration: i64,
        /// The last packet's duration, in ticks.
        packet_duration: Option<i64>,
        /// The stream's average frame rate.
        average_frame_rate: Option<Rational>,
        /// The stream's base frame rate.
        base_frame_rate: Option<Rational>,
        /// The track's frame count. One frame gives no previous gap.
        frame_count: usize,
    },
    /// An arithmetic error.
    Time(TimeError),
}

impl fmt::Display for TimingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoFrames => {
                f.write_str("track has no frames. Pass a stream with at least one decoded frame.")
            }
            Self::NonIncreasingPts {
                index,
                previous,
                pts,
            } => write!(
                f,
                "pts {pts} at frame {index} does not exceed pts {previous} of the frame before it. \
                 Pass frames in presentation order, or repair the source timing."
            ),
            Self::NoLastFrameDuration {
                frame_duration,
                packet_duration,
                average_frame_rate,
                base_frame_rate,
                frame_count,
            } => write!(
                f,
                "no last-frame duration: frame duration is {frame_duration}, packet duration is {}, \
                 average frame rate is {}, base frame rate is {}, and {frame_count} frame(s) give \
                 no previous gap. Supply a frame rate or repair the source timing.",
                Shown(*packet_duration),
                Shown(*average_frame_rate),
                Shown(*base_frame_rate),
            ),
            Self::Time(error) => fmt::Display::fmt(error, f),
        }
    }
}

impl StdError for TimingError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Time(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TimeError> for TimingError {
    fn from(error: TimeError) -> Self {
        Self::Time(error)
    }
}

impl From<OrderViolation> for TimingError {
    fn from(violation: OrderViolation) -> Self {
        Self::NonIncreasingPts {
            index: violation.index,
            previous: violation.previous,
            pts: violation.pts,
        }
    }
}

/// An edit list that vidr cannot apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditListError {
    /// An edit with a rate other than 1.
    UnsupportedRate {
        /// Index of the edit.
        index: usize,
        /// The edit's rate.
        rate: Rational,
    },
    /// An edit with a negative segment duration.
    NegativeSegmentDuration {
        /// Index of the edit.
        index: usize,
        /// The segment duration, in movie ticks.
        segment_duration: i64,
    },
    /// An edit with a media time below -1.
    InvalidMediaTime {
        /// Index of the edit.
        index: usize,
        /// The media time, in media ticks.
        media_time: i64,
    },
    /// An edit after the media edit.
    SegmentAfterMedia {
        /// Index of the edit.
        index: usize,
        /// Index of the media edit before it.
        media_index: usize,
    },
    /// An edit list of empty edits only.
    NoMediaSegment {
        /// The number of edits.
        entries: usize,
    },
    /// No frame falls inside the media edit's range.
    NoPresentedFrames {
        /// First media tick the edit presents.
        media_start: i64,
        /// Media tick where the edit ends, exclusive. `None` runs to the end of the media.
        media_end: Option<i64>,
        /// The track's frame count.
        frames: usize,
    },
    /// The frames are not in presentation order.
    Timing(TimingError),
    /// An arithmetic error.
    Time(TimeError),
}

impl fmt::Display for EditListError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedRate { index, rate } => write!(
                f,
                "edit {index} has rate {rate}, and only rate 1/1 is supported. \
                 Remux the source without slow-motion or dwell edits."
            ),
            Self::NegativeSegmentDuration {
                index,
                segment_duration,
            } => write!(
                f,
                "edit {index} has negative segment duration {segment_duration}. \
                 Repair the source edit list."
            ),
            Self::InvalidMediaTime { index, media_time } => write!(
                f,
                "edit {index} has media time {media_time}, and only -1 (an empty edit) or a value \
                 of 0 or more is valid. Repair the source edit list."
            ),
            Self::SegmentAfterMedia { index, media_index } => write!(
                f,
                "edit {index} follows media edit {media_index}, and only leading empty edits and \
                 one media edit are supported. Remux the source to a single media edit."
            ),
            Self::NoMediaSegment { entries } => write!(
                f,
                "edit list of {entries} edit(s) holds only empty edits, so no media plays. \
                 Repair the source edit list."
            ),
            Self::NoPresentedFrames {
                media_start,
                media_end,
                frames,
            } => {
                write!(
                    f,
                    "no frame of {frames} falls inside the media range from {media_start} to ",
                )?;
                match media_end {
                    Some(end) => write!(f, "{end}")?,
                    None => f.write_str("the end of the media")?,
                }
                f.write_str(". Check the edit list against the track's timestamps.")
            }
            Self::Timing(error) => fmt::Display::fmt(error, f),
            Self::Time(error) => fmt::Display::fmt(error, f),
        }
    }
}

impl StdError for EditListError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Timing(error) => Some(error),
            Self::Time(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TimeError> for EditListError {
    fn from(error: TimeError) -> Self {
        Self::Time(error)
    }
}

impl From<TimingError> for EditListError {
    fn from(error: TimingError) -> Self {
        Self::Timing(error)
    }
}
