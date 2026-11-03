//! Geometry checks: display dimensions, rotation, and display aspect ratio.

use super::deviation::{Deviation, Side};
use crate::error::{GeometryError, TimeError};
use crate::geometry::{Dimensions, Geometry};
use crate::time::Rational;

/// Pushes one deviation per geometry mismatch.
pub(super) fn check(
    source: &Geometry,
    output: &Geometry,
    target: Dimensions,
    deviations: &mut Vec<Deviation>,
) {
    let displayed = output.display_dimensions();
    if displayed != target {
        deviations.push(Deviation::Dimensions {
            target,
            output: displayed,
        });
    }
    if source.rotation != output.rotation {
        deviations.push(Deviation::Rotation {
            source: source.rotation,
            output: output.rotation,
        });
    }
    aspect(source, output, deviations);
}

fn aspect(source: &Geometry, output: &Geometry, deviations: &mut Vec<Deviation>) {
    let ratios = (source.display_aspect_ratio(), output.display_aspect_ratio());
    let (source_ratio, output_ratio) = match ratios {
        (Ok(source_ratio), Ok(output_ratio)) => (source_ratio, output_ratio),
        (source_result, output_result) => {
            for (side, result) in [(Side::Source, source_result), (Side::Output, output_result)] {
                if let Err(error) = result {
                    deviations.push(Deviation::Geometry { side, error });
                }
            }
            return;
        }
    };
    match aspect_deviation(source, output, source_ratio, output_ratio) {
        Ok(Some(deviation)) => deviations.push(deviation),
        Ok(None) => {}
        Err(error) => deviations.push(Deviation::Geometry {
            side: Side::Output,
            error: GeometryError::Time(error),
        }),
    }
}

/// Compares display aspect ratios within `max(a, b) * 2 / shortest side`,
/// the PoC's bound (`reuse_upscaled.py`).
fn aspect_deviation(
    source: &Geometry,
    output: &Geometry,
    source_ratio: Rational,
    output_ratio: Rational,
) -> Result<Option<Deviation>, TimeError> {
    let shortest = [
        source.coded.width(),
        source.coded.height(),
        output.coded.width(),
        output.coded.height(),
    ]
    .into_iter()
    .min()
    .unwrap_or(1);
    let tolerance = source_ratio
        .max(output_ratio)
        .checked_mul(Rational::new(2, u64::from(shortest))?)?;
    let difference = source_ratio.checked_sub(output_ratio)?.checked_abs()?;
    Ok((difference > tolerance).then_some(Deviation::AspectRatio {
        source: source_ratio,
        output: output_ratio,
        tolerance,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Rotation, SampleAspectRatio};

    fn geometry(width: u32, height: u32, rotation: Rotation) -> Geometry {
        Geometry::new(
            Dimensions::new(width, height).unwrap(),
            rotation,
            SampleAspectRatio::SQUARE,
        )
    }

    #[test]
    fn rounding_within_two_pixels_passes() {
        // 853x480 doubled is 1706x960. An even-width 1708x960 output is within bound.
        let source = geometry(853, 480, Rotation::Upright);
        let output = geometry(1708, 960, Rotation::Upright);
        let mut deviations = Vec::new();
        check(&source, &output, output.coded, &mut deviations);
        assert!(deviations.is_empty(), "{deviations:?}");
    }

    #[test]
    fn rotated_output_reports_size_rotation_and_aspect() {
        let source = geometry(1920, 1080, Rotation::Upright);
        let output = geometry(1920, 1080, Rotation::Clockwise90);
        let mut deviations = Vec::new();
        check(&source, &output, source.coded, &mut deviations);
        assert_eq!(deviations.len(), 3, "{deviations:?}");
        assert!(matches!(deviations[0], Deviation::Dimensions { .. }));
        assert!(matches!(deviations[1], Deviation::Rotation { .. }));
        assert!(matches!(deviations[2], Deviation::AspectRatio { .. }));
    }
}
