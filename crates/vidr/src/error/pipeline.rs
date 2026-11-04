//! Errors of the threaded pump.

use std::error::Error as StdError;
use std::fmt;

#[cfg(any(feature = "mp4", feature = "mkv"))]
use super::MediaError;
use super::{DecodeError, EditListError, EncodeError, FrameError, TimeError, TimingError};
use crate::time::Rational;

/// One thread of the pump.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PumpStage {
    /// T1: demux and decode.
    Decode,
    /// T2: transform.
    Transform,
    /// T3: encode and mux.
    Encode,
}

impl PumpStage {
    /// The OS name of the stage's thread.
    pub(crate) const fn thread_name(self) -> &'static str {
        match self {
            Self::Decode => "vidr-decode",
            Self::Transform => "vidr-transform",
            Self::Encode => "vidr-encode",
        }
    }
}

impl fmt::Display for PumpStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Decode => "the demux and decode stage",
            Self::Transform => "the transform stage",
            Self::Encode => "the encode and mux stage",
        })
    }
}

/// An error of a pump run, or the named result of a stop request.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PipelineError {
    /// A caller's source, transform, or sink fails.
    Failed {
        /// What fails, with its input.
        what: String,
        /// Why it fails, with the next action.
        reason: String,
    },
    /// The decoder fails.
    Decode(DecodeError),
    /// The encoder fails.
    Encode(EncodeError),
    /// A frame conversion fails.
    Frame(FrameError),
    /// Media input or output fails.
    #[cfg(any(feature = "mp4", feature = "mkv"))]
    Media(MediaError),
    /// Frame timing breaks a rule, or the last-frame duration chain finds no
    /// source.
    Timing(TimingError),
    /// The edit window presents no frame.
    EditList(EditListError),
    /// A last-frame duration does not convert to ticks.
    Time(TimeError),
    /// The last frame's duration rounds to no whole tick above zero.
    LastDurationBelowTick {
        /// The last frame's pts, in ticks.
        pts: i64,
        /// The duration, in seconds.
        duration: Rational,
        /// The time base, in seconds per tick.
        time_base: Rational,
    },
    /// A thread of the pump panics.
    Panic {
        /// The stage that panics.
        stage: PumpStage,
        /// The panic message.
        message: String,
    },
    /// The OS refuses to start a thread of the pump.
    Spawn {
        /// The stage that does not start.
        stage: PumpStage,
        /// The OS message.
        reason: String,
    },
    /// The caller asks the run to stop. The sink never finishes, so the output
    /// is not final. A command-line caller maps this to exit code 130.
    Stopped {
        /// Frames the decoder returned before the stop.
        frames_in: u64,
        /// Packets the sink took before the stop.
        packets: u64,
    },
    /// The source holds no frame the decoder returns.
    NoFrames {
        /// Packets the source supplied.
        packets_read: u64,
    },
    /// The encoder's frame or packet count differs from the frames the decoder
    /// returned and the edit window kept.
    CountMismatch {
        /// Frames the decoder returned.
        frames_in: u64,
        /// Frames the edit window dropped.
        frames_dropped: u64,
        /// Frames the encoder took.
        frames_out: u64,
        /// Packets the encoder returned.
        packets: u64,
    },
    /// An encoder returns a packet and no config record.
    NoConfigRecord {
        /// The packet's pts, in ticks.
        pts: i64,
    },
    /// Every stage ends without an error, a stop request, or a finished sink.
    Incomplete,
}

impl PipelineError {
    /// An error of a caller's source, transform, or sink. `what` names the step
    /// and its input. `reason` states why, and the next action.
    pub fn failed(what: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Failed {
            what: what.into(),
            reason: reason.into(),
        }
    }

    /// Returns `true` for the result of a stop request.
    pub const fn is_stop(&self) -> bool {
        matches!(self, Self::Stopped { .. })
    }
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Failed { what, reason } => write!(f, "{what} fails: {reason}"),
            Self::Decode(error) => fmt::Display::fmt(error, f),
            Self::Encode(error) => fmt::Display::fmt(error, f),
            Self::Frame(error) => fmt::Display::fmt(error, f),
            #[cfg(any(feature = "mp4", feature = "mkv"))]
            Self::Media(error) => fmt::Display::fmt(error, f),
            Self::Timing(error) => fmt::Display::fmt(error, f),
            Self::EditList(error) => fmt::Display::fmt(error, f),
            Self::Time(error) => fmt::Display::fmt(error, f),
            Self::LastDurationBelowTick {
                pts,
                duration,
                time_base,
            } => write!(
                f,
                "the last frame at pts {pts} lasts {duration} s, which rounds to no tick of \
                 {time_base} s. Pass a finer time base, or repair the source timing."
            ),
            Self::Panic { stage, message } => write!(
                f,
                "{stage} panics: {message}. Report this panic with the input that causes it."
            ),
            Self::Spawn { stage, reason } => write!(
                f,
                "{stage} does not start: {reason}. Free OS thread or memory limits, then retry."
            ),
            Self::Stopped { frames_in, packets } => write!(
                f,
                "the run stops on request after {frames_in} decoded frame(s) and {packets} \
                 packet(s), and the output is not final. Remove the output file, or rerun to \
                 write it."
            ),
            Self::NoFrames { packets_read } => write!(
                f,
                "the decoder returns no frame from {packets_read} packet(s). Pass a source with \
                 at least one decodable frame."
            ),
            Self::CountMismatch {
                frames_in,
                frames_dropped,
                frames_out,
                packets,
            } => write!(
                f,
                "the decoder returns {frames_in} frame(s) and the edit window drops \
                 {frames_dropped}, but the encoder takes {frames_out} and returns {packets} \
                 packet(s). The output is not final. Check the transform and the encoder."
            ),
            Self::NoConfigRecord { pts } => write!(
                f,
                "the encoder returns the packet at pts {pts} with no config record, so no output \
                 can start. Return the record from `Encoder::config_record` with the first \
                 packet."
            ),
            Self::Incomplete => f.write_str(
                "the pump ends with no error, no stop request, and no finished sink. Report this \
                 run, because every other result is named.",
            ),
        }
    }
}

impl StdError for PipelineError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Decode(error) => Some(error),
            Self::Encode(error) => Some(error),
            Self::Frame(error) => Some(error),
            #[cfg(any(feature = "mp4", feature = "mkv"))]
            Self::Media(error) => Some(error),
            Self::Timing(error) => Some(error),
            Self::EditList(error) => Some(error),
            Self::Time(error) => Some(error),
            _ => None,
        }
    }
}

impl From<DecodeError> for PipelineError {
    fn from(error: DecodeError) -> Self {
        Self::Decode(error)
    }
}

impl From<EncodeError> for PipelineError {
    fn from(error: EncodeError) -> Self {
        Self::Encode(error)
    }
}

impl From<FrameError> for PipelineError {
    fn from(error: FrameError) -> Self {
        Self::Frame(error)
    }
}

#[cfg(any(feature = "mp4", feature = "mkv"))]
impl From<MediaError> for PipelineError {
    fn from(error: MediaError) -> Self {
        Self::Media(error)
    }
}

impl From<TimingError> for PipelineError {
    fn from(error: TimingError) -> Self {
        Self::Timing(error)
    }
}

impl From<EditListError> for PipelineError {
    fn from(error: EditListError) -> Self {
        Self::EditList(error)
    }
}

impl From<TimeError> for PipelineError {
    fn from(error: TimeError) -> Self {
        Self::Time(error)
    }
}
