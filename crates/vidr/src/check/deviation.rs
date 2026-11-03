//! Mismatches between a source stream and an output stream.

use std::error::Error as StdError;
use std::fmt;

use crate::error::{GeometryError, TimingError};
use crate::geometry::{Dimensions, Rotation};
use crate::time::Rational;

/// Which stream a deviation concerns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    /// The stream the pipeline read.
    Source,
    /// The stream the pipeline wrote.
    Output,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Source => "source",
            Self::Output => "output",
        })
    }
}

/// One mismatch found by [`check_output`](super::check_output).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Deviation {
    /// The streams hold different frame counts.
    FrameCount {
        /// Source frame count.
        source: usize,
        /// Output frame count.
        output: usize,
    },
    /// The output's display dimensions differ from the target.
    Dimensions {
        /// Requested display dimensions.
        target: Dimensions,
        /// Output display dimensions.
        output: Dimensions,
    },
    /// The output rotation differs from the source rotation.
    Rotation {
        /// Source rotation.
        source: Rotation,
        /// Output rotation.
        output: Rotation,
    },
    /// The display aspect ratios differ by more than the tolerance.
    AspectRatio {
        /// Source display aspect ratio.
        source: Rational,
        /// Output display aspect ratio.
        output: Rational,
        /// Largest difference allowed.
        tolerance: Rational,
    },
    /// A stream's geometry cannot be evaluated.
    Geometry {
        /// The stream.
        side: Side,
        /// The error.
        error: GeometryError,
    },
    /// A stream's pts do not strictly increase.
    PtsOrder {
        /// The stream.
        side: Side,
        /// Index of the first frame whose pts does not increase.
        index: usize,
        /// Pts of the frame before it.
        previous: i64,
        /// Pts of the frame at `index`.
        pts: i64,
        /// Number of frames whose pts do not increase.
        violations: usize,
    },
    /// Output pts differ from the source pts rescaled to the output time base.
    Pts {
        /// Index of the first frame that differs.
        index: usize,
        /// Source pts relative to its first frame, rescaled half-even.
        expected: i64,
        /// Output pts relative to its first frame.
        output: i64,
        /// Number of frames that differ, over the frames both streams hold.
        mismatches: usize,
        /// Output time base.
        time_base: Rational,
    },
    /// The total durations differ.
    Duration {
        /// Duration the output must have, in seconds.
        expected: Rational,
        /// Output duration, in seconds.
        output: Rational,
        /// Largest difference allowed. `None` means exact equality.
        tolerance: Option<Rational>,
    },
    /// A stream's timing cannot be evaluated.
    Timing {
        /// The stream.
        side: Side,
        /// The error.
        error: TimingError,
    },
}

impl fmt::Display for Deviation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FrameCount { source, output } => write!(
                f,
                "output has {output} frames and source has {source}. \
                 Check the decoder and encoder for dropped or duplicated frames."
            ),
            Self::Dimensions { target, output } => write!(
                f,
                "output display size is {output} and the target is {target}. \
                 Check the scale step and the output rotation."
            ),
            Self::Rotation { source, output } => write!(
                f,
                "output rotation is {output} and source rotation is {source}. \
                 Copy the source rotation to the output."
            ),
            Self::AspectRatio {
                source,
                output,
                tolerance,
            } => write!(
                f,
                "output display aspect ratio {output} differs from source {source} by more than \
                 {tolerance}. Set the output sample aspect ratio from the source display aspect ratio."
            ),
            Self::Geometry { side, error } => write!(f, "{side} geometry: {error}"),
            Self::PtsOrder {
                side,
                index,
                previous,
                pts,
                violations,
            } => write!(
                f,
                "{side} pts do not strictly increase: frame {index} has pts {pts} after {previous}, \
                 and {violations} frame(s) break the order. Repair the {side} timing."
            ),
            Self::Pts {
                index,
                expected,
                output,
                mismatches,
                time_base,
            } => write!(
                f,
                "output pts {output} at frame {index} differs from the source pts rescaled \
                 half-even to time base {time_base}, {expected}, and {mismatches} frame(s) differ. \
                 Write each output pts as the source pts relative to its first frame, rescaled \
                 half-even."
            ),
            Self::Duration {
                expected,
                output,
                tolerance: None,
            } => write!(
                f,
                "output duration {output} s differs from the expected {expected} s, and one time \
                 base requires exact equality. Copy the source last-frame duration to the output."
            ),
            Self::Duration {
                expected,
                output,
                tolerance: Some(tolerance),
            } => write!(
                f,
                "output duration {output} s differs from source duration {expected} s by more \
                 than {tolerance} s. Check the last-frame duration the output container stores."
            ),
            Self::Timing { side, error } => write!(f, "{side} timing: {error}"),
        }
    }
}

/// Every mismatch [`check_output`](super::check_output) found. Never empty.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckError {
    deviations: Vec<Deviation>,
}

impl CheckError {
    pub(super) const fn new(deviations: Vec<Deviation>) -> Self {
        Self { deviations }
    }

    /// The mismatches, in check order.
    pub fn deviations(&self) -> &[Deviation] {
        &self.deviations
    }

    /// Takes the mismatches.
    pub fn into_deviations(self) -> Vec<Deviation> {
        self.deviations
    }
}

impl fmt::Display for CheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "output check found {} mismatch(es):",
            self.deviations.len()
        )?;
        for deviation in &self.deviations {
            write!(f, "\n- {deviation}")?;
        }
        Ok(())
    }
}

impl StdError for CheckError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_lists_every_deviation() {
        let error = CheckError::new(vec![
            Deviation::FrameCount {
                source: 3,
                output: 2,
            },
            Deviation::Rotation {
                source: Rotation::Upright,
                output: Rotation::Clockwise90,
            },
        ]);
        let message = error.to_string();
        assert!(message.starts_with("output check found 2 mismatch(es):"));
        assert!(message.contains("\n- output has 2 frames and source has 3."));
        assert!(message.contains("output rotation is 90 degrees clockwise"));
    }
}
