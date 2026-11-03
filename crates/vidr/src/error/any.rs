//! The crate-wide error enum.

use std::error::Error as StdError;
use std::fmt;

#[cfg(any(feature = "h264", feature = "hevc", feature = "av1"))]
use super::BitstreamError;
use super::{EditListError, GeometryError, TimeError, TimingError};
use crate::check::CheckError;

/// Any vidr error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Exact arithmetic or tick conversion.
    Time(TimeError),
    /// Track frame timing.
    Timing(TimingError),
    /// Edit list.
    EditList(EditListError),
    /// Frame geometry.
    Geometry(GeometryError),
    /// Output checks.
    Check(CheckError),
    /// Bitstream filters and codec config records.
    #[cfg(any(feature = "h264", feature = "hevc", feature = "av1"))]
    Bitstream(BitstreamError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Time(error) => fmt::Display::fmt(error, f),
            Self::Timing(error) => fmt::Display::fmt(error, f),
            Self::EditList(error) => fmt::Display::fmt(error, f),
            Self::Geometry(error) => fmt::Display::fmt(error, f),
            Self::Check(error) => fmt::Display::fmt(error, f),
            #[cfg(any(feature = "h264", feature = "hevc", feature = "av1"))]
            Self::Bitstream(error) => fmt::Display::fmt(error, f),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Time(error) => Some(error),
            Self::Timing(error) => Some(error),
            Self::EditList(error) => Some(error),
            Self::Geometry(error) => Some(error),
            Self::Check(error) => Some(error),
            #[cfg(any(feature = "h264", feature = "hevc", feature = "av1"))]
            Self::Bitstream(error) => Some(error),
        }
    }
}

impl From<TimeError> for Error {
    fn from(error: TimeError) -> Self {
        Self::Time(error)
    }
}

impl From<TimingError> for Error {
    fn from(error: TimingError) -> Self {
        Self::Timing(error)
    }
}

impl From<EditListError> for Error {
    fn from(error: EditListError) -> Self {
        Self::EditList(error)
    }
}

impl From<GeometryError> for Error {
    fn from(error: GeometryError) -> Self {
        Self::Geometry(error)
    }
}

impl From<CheckError> for Error {
    fn from(error: CheckError) -> Self {
        Self::Check(error)
    }
}

#[cfg(any(feature = "h264", feature = "hevc", feature = "av1"))]
impl From<BitstreamError> for Error {
    fn from(error: BitstreamError) -> Self {
        Self::Bitstream(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Operation;
    use crate::time::Rational;

    #[test]
    fn last_duration_message_names_every_missing_input() {
        let error = TimingError::NoLastFrameDuration {
            frame_duration: 0,
            packet_duration: None,
            average_frame_rate: None,
            base_frame_rate: None,
            frame_count: 1,
        };
        let message = error.to_string();
        assert!(message.contains("frame duration is 0"));
        assert!(message.contains("packet duration is missing"));
        assert!(message.contains("average frame rate is missing"));
        assert!(message.contains("base frame rate is missing"));
        assert!(message.contains("1 frame(s)"));
        assert!(message.contains("Supply a frame rate"));
    }

    #[test]
    fn overflow_message_names_operands() {
        let error = TimeError::Overflow {
            operation: Operation::Add,
            lhs: Rational::from_integer(i64::MAX),
            rhs: Some(Rational::ONE),
        };
        let message = error.to_string();
        assert!(message.contains("addition of 9223372036854775807/1 and 1/1"));
    }

    #[test]
    fn wrapped_errors_expose_their_source() {
        let inner = TimeError::NonPositiveTimeBase {
            time_base: Rational::ZERO,
        };
        let error = Error::from(TimingError::from(inner));
        assert!(error.source().is_some());
        assert_eq!(error.to_string(), inner.to_string());
    }
}
