use vidr::{Dimensions, Geometry, GeometryError, Rotation, SampleAspectRatio};

use super::r;

const ONE: i32 = 0x0001_0000;
const W: i32 = 0x4000_0000;

/// A `tkhd` matrix `a b u c d v x y w` with no translation.
const fn matrix(a: i32, b: i32, c: i32, d: i32) -> [i32; 9] {
    [a, b, 0, c, d, 0, 0, 0, W]
}

#[test]
fn matrix_parses_each_quarter_turn() {
    assert_eq!(
        Rotation::from_display_matrix(matrix(ONE, 0, 0, ONE)),
        Ok(Rotation::Upright)
    );
    // The matrix a phone writes for a portrait clip: `rotate` tag 90.
    assert_eq!(
        Rotation::from_display_matrix(matrix(0, ONE, -ONE, 0)),
        Ok(Rotation::Clockwise90)
    );
    assert_eq!(
        Rotation::from_display_matrix(matrix(-ONE, 0, 0, -ONE)),
        Ok(Rotation::Clockwise180)
    );
    assert_eq!(
        Rotation::from_display_matrix(matrix(0, -ONE, ONE, 0)),
        Ok(Rotation::Clockwise270)
    );
}

#[test]
fn matrix_translation_does_not_change_the_angle() {
    let mut rotated = matrix(0, ONE, -ONE, 0);
    rotated[6] = 1080 * ONE;
    assert_eq!(
        Rotation::from_display_matrix(rotated),
        Ok(Rotation::Clockwise90)
    );
}

#[test]
fn matrix_at_45_degrees_is_refused() {
    // cos 45 = sin 45 = 0.70710678 in 16.16 fixed point.
    let half_root = 46341;
    let tilted = matrix(half_root, half_root, -half_root, half_root);
    assert_eq!(
        Rotation::from_display_matrix(tilted),
        Err(GeometryError::UnsupportedDisplayMatrix { matrix: tilted })
    );
}

#[test]
fn matrix_with_scale_is_refused() {
    assert!(Rotation::from_display_matrix(matrix(2 * ONE, 0, 0, 2 * ONE)).is_err());
    let mut wrong_w = matrix(ONE, 0, 0, ONE);
    wrong_w[8] = 0;
    assert!(Rotation::from_display_matrix(wrong_w).is_err());
}

#[test]
fn degrees_refuse_45_and_normalise_negatives() {
    assert_eq!(
        Rotation::from_degrees(45),
        Err(GeometryError::UnsupportedRotation { degrees: 45 })
    );
    assert_eq!(Rotation::from_degrees(-90), Ok(Rotation::Clockwise270));
    assert_eq!(Rotation::from_degrees(-270), Ok(Rotation::Clockwise90));
    assert_eq!(Rotation::from_degrees(-180), Ok(Rotation::Clockwise180));
    assert_eq!(Rotation::from_degrees(720), Ok(Rotation::Upright));
}

#[test]
fn rotation_swaps_display_dimensions_and_inverts_dar() {
    let coded = Dimensions::new(1920, 1080).unwrap();
    let upright = Geometry::new(coded, Rotation::Upright, SampleAspectRatio::SQUARE);
    let portrait = Geometry::new(coded, Rotation::Clockwise90, SampleAspectRatio::SQUARE);
    assert_eq!(upright.display_aspect_ratio(), Ok(r(16, 9)));
    assert_eq!(portrait.display_aspect_ratio(), Ok(r(9, 16)));
    assert_eq!(portrait.display_dimensions(), coded.transposed());
    let flipped = Geometry::new(coded, Rotation::Clockwise180, SampleAspectRatio::SQUARE);
    assert_eq!(flipped.display_dimensions(), coded);
}

#[test]
fn rotation_inverts_an_anamorphic_sar() {
    let coded = Dimensions::new(1440, 1080).unwrap();
    let sar = SampleAspectRatio::new(4, 3);
    let upright = Geometry::new(coded, Rotation::Upright, sar);
    let rotated = Geometry::new(coded, Rotation::Clockwise270, sar);
    assert_eq!(upright.display_aspect_ratio(), Ok(r(16, 9)));
    assert_eq!(
        rotated.display_sample_aspect_ratio(),
        Ok(SampleAspectRatio::new(3, 4))
    );
    assert_eq!(rotated.display_aspect_ratio(), Ok(r(9, 16)));
}
