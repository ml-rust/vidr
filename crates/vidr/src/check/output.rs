//! The output check and the stream facts it reads.

use super::deviation::{CheckError, Deviation};
use super::{shape, timing};
use crate::geometry::{Dimensions, Geometry};
use crate::time::{DurationHints, FrameTiming, Rational};

/// Largest total-duration difference allowed between streams in different
/// time bases: 0.002 s, the PoC's bound (`upscale.py`, `video_io.py`).
///
/// It covers the rounding of a millisecond clock, such as Matroska's: 1 ms
/// at most. [`check_output`] uses it only when the source and output time
/// bases differ. In one time base the comparison is exact.
pub const DURATION_TOLERANCE: Rational = match Rational::new(1, 500) {
    Ok(tolerance) => tolerance,
    // Not reached: 500 is nonzero. Zero fails closed to an exact comparison.
    Err(_) => Rational::ZERO,
};

/// What a probe reads from one video stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamFacts {
    /// Coded size, rotation, and sample aspect ratio.
    pub geometry: Geometry,
    /// Stream time base, in seconds per tick.
    pub time_base: Rational,
    /// Every frame, in presentation order. The frame count is its length.
    pub frames: Vec<FrameTiming>,
    /// Inputs to the last-frame duration chain.
    pub hints: DurationHints,
}

impl StreamFacts {
    /// Creates stream facts.
    pub const fn new(
        geometry: Geometry,
        time_base: Rational,
        frames: Vec<FrameTiming>,
        hints: DurationHints,
    ) -> Self {
        Self {
            geometry,
            time_base,
            frames,
            hints,
        }
    }

    /// The number of frames.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

/// Checks `output` against `source` and the `target` display dimensions.
///
/// Pts compare relative to each stream's first frame. The edit offset and
/// container start do not enter the comparison.
///
/// # Errors
///
/// [`CheckError`] holding one [`Deviation`] per mismatch, in this order:
/// frame count, display dimensions, rotation, display aspect ratio, pts
/// order, pts values, total duration. A check that depends on an earlier
/// failed one is skipped: the duration check needs strictly increasing pts on
/// both sides.
pub fn check_output(
    source: &StreamFacts,
    output: &StreamFacts,
    target: Dimensions,
) -> Result<(), CheckError> {
    let mut deviations = Vec::new();
    if source.frame_count() != output.frame_count() {
        deviations.push(Deviation::FrameCount {
            source: source.frame_count(),
            output: output.frame_count(),
        });
    }
    shape::check(&source.geometry, &output.geometry, target, &mut deviations);
    timing::check(source, output, &mut deviations);
    if deviations.is_empty() {
        Ok(())
    } else {
        Err(CheckError::new(deviations))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Rotation, SampleAspectRatio};

    fn facts(width: u32, height: u32, pts: &[i64]) -> StreamFacts {
        StreamFacts::new(
            Geometry::new(
                Dimensions::new(width, height).unwrap(),
                Rotation::Upright,
                SampleAspectRatio::SQUARE,
            ),
            Rational::new(1, 30).unwrap(),
            pts.iter()
                .map(|&pts| FrameTiming::new(pts, None, 1))
                .collect(),
            DurationHints::default(),
        )
    }

    #[test]
    fn tolerance_is_two_milliseconds() {
        assert_eq!(DURATION_TOLERANCE, Rational::new(2, 1000).unwrap());
    }

    #[test]
    fn matching_output_passes() {
        let source = facts(640, 480, &[0, 1, 2]);
        let output = facts(1280, 960, &[0, 1, 2]);
        let target = Dimensions::new(1280, 960).unwrap();
        assert_eq!(check_output(&source, &output, target), Ok(()));
    }

    #[test]
    fn frame_count_mismatch_is_reported() {
        let source = facts(640, 480, &[0, 1, 2]);
        let output = facts(640, 480, &[0, 1]);
        let target = Dimensions::new(640, 480).unwrap();
        let error = check_output(&source, &output, target).unwrap_err();
        assert!(error.deviations().contains(&Deviation::FrameCount {
            source: 3,
            output: 2
        }));
    }
}
