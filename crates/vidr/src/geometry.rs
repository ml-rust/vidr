//! Frame geometry: coded size, rotation, and aspect ratios, in exact rationals.
//!
//! The PoC (`full_video_io.py`) accepts rotations in multiples of 90 degrees
//! only. For 90 and 270 degrees it swaps width and height and inverts the
//! sample aspect ratio, so display values describe the upright picture.

use std::fmt;

use crate::error::GeometryError;
use crate::time::Rational;

/// 1.0 in the 16.16 fixed point of display matrix entries `a b c d x y`.
const FIXED_ONE: i32 = 1 << 16;
/// -1.0 in 16.16 fixed point.
const NEGATIVE_ONE: i32 = -FIXED_ONE;
/// 1.0 in the 2.30 fixed point of display matrix entries `u v w`.
const PROJECTIVE_ONE: i32 = 1 << 30;

/// Frame width and height in pixels, both nonzero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dimensions {
    width: u32,
    height: u32,
}

impl Dimensions {
    /// Creates dimensions.
    ///
    /// # Errors
    ///
    /// [`GeometryError::ZeroDimension`] when either side is zero.
    pub const fn new(width: u32, height: u32) -> Result<Self, GeometryError> {
        if width == 0 || height == 0 {
            Err(GeometryError::ZeroDimension { width, height })
        } else {
            Ok(Self { width, height })
        }
    }

    /// Width in pixels.
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Height in pixels.
    pub const fn height(self) -> u32 {
        self.height
    }

    /// The dimensions with width and height swapped.
    pub const fn transposed(self) -> Self {
        Self {
            width: self.height,
            height: self.width,
        }
    }

    /// `width / height`, exact.
    ///
    /// # Errors
    ///
    /// None in practice. The signature carries [`GeometryError::Time`] for
    /// the checked constructor.
    pub fn aspect_ratio(self) -> Result<Rational, GeometryError> {
        Ok(Rational::new(
            i64::from(self.width),
            u64::from(self.height),
        )?)
    }
}

impl fmt::Display for Dimensions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

/// Clockwise rotation a player applies to show the frame upright.
///
/// vidr uses the clockwise sense of the MP4 `rotate` tag. ffprobe's
/// side-data `rotation` field is counterclockwise: a phone portrait clip
/// reports `-90` there and `90` here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Rotation {
    /// No rotation.
    #[default]
    Upright,
    /// 90 degrees clockwise.
    Clockwise90,
    /// 180 degrees.
    Clockwise180,
    /// 270 degrees clockwise, which is 90 degrees counterclockwise.
    Clockwise270,
}

impl Rotation {
    /// Parses clockwise degrees. Negative and large values normalise modulo
    /// 360, so `-90` is [`Rotation::Clockwise270`].
    ///
    /// # Errors
    ///
    /// [`GeometryError::UnsupportedRotation`] for a value that is not a
    /// multiple of 90.
    pub const fn from_degrees(degrees: i64) -> Result<Self, GeometryError> {
        match degrees.rem_euclid(360) {
            0 => Ok(Self::Upright),
            90 => Ok(Self::Clockwise90),
            180 => Ok(Self::Clockwise180),
            270 => Ok(Self::Clockwise270),
            _ => Err(GeometryError::UnsupportedRotation { degrees }),
        }
    }

    /// Parses a 3x3 display matrix in `tkhd` order `a b u c d v x y w`.
    ///
    /// `a b c d x y` are 16.16 fixed point and `u v w` are 2.30 fixed point.
    /// The matrix must be a pure rotation: `a b c d` each 0 or ±1.0, `u` and
    /// `v` zero, and `w` 1.0. The translation `x y` places the rotated frame
    /// and does not change the angle, so any value passes. The mapping matches
    /// FFmpeg's `av_display_rotation_get`, negated to the clockwise sense.
    ///
    /// # Errors
    ///
    /// [`GeometryError::UnsupportedDisplayMatrix`] for scaling, mirroring,
    /// shearing, a projective part, or another angle.
    pub const fn from_display_matrix(matrix: [i32; 9]) -> Result<Self, GeometryError> {
        let [a, b, u, c, d, v, _, _, w] = matrix;
        if u != 0 || v != 0 || w != PROJECTIVE_ONE {
            return Err(GeometryError::UnsupportedDisplayMatrix { matrix });
        }
        match (a, b, c, d) {
            (FIXED_ONE, 0, 0, FIXED_ONE) => Ok(Self::Upright),
            (0, FIXED_ONE, NEGATIVE_ONE, 0) => Ok(Self::Clockwise90),
            (NEGATIVE_ONE, 0, 0, NEGATIVE_ONE) => Ok(Self::Clockwise180),
            (0, NEGATIVE_ONE, FIXED_ONE, 0) => Ok(Self::Clockwise270),
            _ => Err(GeometryError::UnsupportedDisplayMatrix { matrix }),
        }
    }

    /// The 3x3 display matrix in `tkhd` order `a b u c d v x y w`, with no
    /// translation. [`Rotation::from_display_matrix`] reads it back.
    pub const fn display_matrix(self) -> [i32; 9] {
        let (a, b, c, d) = match self {
            Self::Upright => (FIXED_ONE, 0, 0, FIXED_ONE),
            Self::Clockwise90 => (0, FIXED_ONE, NEGATIVE_ONE, 0),
            Self::Clockwise180 => (NEGATIVE_ONE, 0, 0, NEGATIVE_ONE),
            Self::Clockwise270 => (0, NEGATIVE_ONE, FIXED_ONE, 0),
        };
        [a, b, 0, c, d, 0, 0, 0, PROJECTIVE_ONE]
    }

    /// Clockwise degrees: 0, 90, 180, or 270.
    pub const fn degrees(self) -> u16 {
        match self {
            Self::Upright => 0,
            Self::Clockwise90 => 90,
            Self::Clockwise180 => 180,
            Self::Clockwise270 => 270,
        }
    }

    /// Counterclockwise degrees, as ffprobe's side-data `rotation` reports
    /// them modulo 360.
    pub const fn counterclockwise_degrees(self) -> u16 {
        (360 - self.degrees()) % 360
    }

    /// Returns `true` for 90 and 270 degrees, which swap width and height.
    pub const fn swaps_axes(self) -> bool {
        matches!(self, Self::Clockwise90 | Self::Clockwise270)
    }
}

impl fmt::Display for Rotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} degrees clockwise", self.degrees())
    }
}

/// Width of one pixel over its height, exact and positive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SampleAspectRatio(Rational);

impl SampleAspectRatio {
    /// Square pixels, `1:1`.
    pub const SQUARE: Self = Self(Rational::ONE);

    /// Creates `num:den`. A zero part means the stream leaves the ratio
    /// unspecified, and gives square pixels, as the PoC reads it.
    pub const fn new(num: u32, den: u32) -> Self {
        if num == 0 || den == 0 {
            return Self::SQUARE;
        }
        match Rational::new(num as i64, den as u64) {
            Ok(ratio) => Self(ratio),
            // Not reached: `den` is nonzero here.
            Err(_) => Self::SQUARE,
        }
    }

    /// Creates the ratio `num:den` from wide parts, in lowest terms.
    ///
    /// Returns `None` for a zero part, which leaves the ratio unspecified, and
    /// for a numerator above the signed 64-bit range.
    pub fn from_fraction(num: u64, den: u64) -> Option<Self> {
        if num == 0 {
            return None;
        }
        let ratio = Rational::new(i64::try_from(num).ok()?, den).ok()?;
        Some(Self(ratio))
    }

    /// The ratio as a rational.
    pub const fn ratio(self) -> Rational {
        self.0
    }

    /// The ratio of the upright picture: inverted for 90 and 270 degrees.
    ///
    /// # Errors
    ///
    /// None in practice. The signature carries [`GeometryError::Time`] for
    /// the checked reciprocal.
    pub fn for_rotation(self, rotation: Rotation) -> Result<Self, GeometryError> {
        if rotation.swaps_axes() {
            Ok(Self(self.0.recip()?))
        } else {
            Ok(self)
        }
    }
}

impl Default for SampleAspectRatio {
    fn default() -> Self {
        Self::SQUARE
    }
}

impl fmt::Display for SampleAspectRatio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.0.num(), self.0.den())
    }
}

/// A stream's coded size, rotation, and sample aspect ratio.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Geometry {
    /// Size of the decoded frame, before rotation.
    pub coded: Dimensions,
    /// Rotation to show the frame upright.
    pub rotation: Rotation,
    /// Sample aspect ratio of the coded frame.
    pub sample_aspect_ratio: SampleAspectRatio,
}

impl Geometry {
    /// Creates a geometry.
    pub const fn new(
        coded: Dimensions,
        rotation: Rotation,
        sample_aspect_ratio: SampleAspectRatio,
    ) -> Self {
        Self {
            coded,
            rotation,
            sample_aspect_ratio,
        }
    }

    /// Size of the upright picture: coded size, swapped for 90 and 270 degrees.
    pub const fn display_dimensions(&self) -> Dimensions {
        if self.rotation.swaps_axes() {
            self.coded.transposed()
        } else {
            self.coded
        }
    }

    /// Sample aspect ratio of the upright picture.
    ///
    /// # Errors
    ///
    /// None in practice. See [`SampleAspectRatio::for_rotation`].
    pub fn display_sample_aspect_ratio(&self) -> Result<SampleAspectRatio, GeometryError> {
        self.sample_aspect_ratio.for_rotation(self.rotation)
    }

    /// Display aspect ratio of the upright picture: display width times sample
    /// aspect ratio over display height. Exact.
    ///
    /// # Errors
    ///
    /// [`GeometryError::Time`] when the product overflows.
    pub fn display_aspect_ratio(&self) -> Result<Rational, GeometryError> {
        let shape = self.display_dimensions().aspect_ratio()?;
        let pixel = self.display_sample_aspect_ratio()?.ratio();
        Ok(shape.checked_mul(pixel)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(num: i64, den: u64) -> Rational {
        Rational::new(num, den).unwrap()
    }

    #[test]
    fn display_matrix_reads_back_as_the_same_rotation() {
        for rotation in [
            Rotation::Upright,
            Rotation::Clockwise90,
            Rotation::Clockwise180,
            Rotation::Clockwise270,
        ] {
            assert_eq!(
                Rotation::from_display_matrix(rotation.display_matrix()),
                Ok(rotation)
            );
        }
    }

    #[test]
    fn degrees_normalise_negative_and_large_values() {
        assert_eq!(Rotation::from_degrees(-90), Ok(Rotation::Clockwise270));
        assert_eq!(Rotation::from_degrees(450), Ok(Rotation::Clockwise90));
        assert_eq!(Rotation::from_degrees(-360), Ok(Rotation::Upright));
        assert_eq!(
            Rotation::from_degrees(45),
            Err(GeometryError::UnsupportedRotation { degrees: 45 })
        );
    }

    #[test]
    fn counterclockwise_matches_ffprobe_side_data() {
        assert_eq!(Rotation::Clockwise90.counterclockwise_degrees(), 270);
        assert_eq!(Rotation::Upright.counterclockwise_degrees(), 0);
    }

    #[test]
    fn matrix_with_projective_part_is_refused() {
        let matrix = [FIXED_ONE, 0, 1, 0, FIXED_ONE, 0, 0, 0, PROJECTIVE_ONE];
        assert_eq!(
            Rotation::from_display_matrix(matrix),
            Err(GeometryError::UnsupportedDisplayMatrix { matrix })
        );
    }

    #[test]
    fn mirror_matrix_is_refused() {
        let matrix = [-FIXED_ONE, 0, 0, 0, FIXED_ONE, 0, 0, 0, PROJECTIVE_ONE];
        assert!(Rotation::from_display_matrix(matrix).is_err());
    }

    #[test]
    fn zero_sar_reads_as_square() {
        assert_eq!(SampleAspectRatio::new(0, 1), SampleAspectRatio::SQUARE);
        assert_eq!(SampleAspectRatio::new(8, 0), SampleAspectRatio::SQUARE);
        assert_eq!(SampleAspectRatio::new(16, 18).ratio(), r(8, 9));
    }

    #[test]
    fn from_fraction_reduces_wide_parts_and_refuses_unspecified_ones() {
        assert_eq!(
            SampleAspectRatio::from_fraction(16 << 32, 18 << 32),
            Some(SampleAspectRatio::new(8, 9))
        );
        assert_eq!(SampleAspectRatio::from_fraction(0, 9), None);
        assert_eq!(SampleAspectRatio::from_fraction(8, 0), None);
        assert_eq!(SampleAspectRatio::from_fraction(u64::MAX, 1), None);
    }

    #[test]
    fn anamorphic_dar_without_rotation() {
        let geometry = Geometry::new(
            Dimensions::new(720, 480).unwrap(),
            Rotation::Upright,
            SampleAspectRatio::new(8, 9),
        );
        assert_eq!(geometry.display_aspect_ratio().unwrap(), r(4, 3));
    }

    #[test]
    fn zero_dimension_is_refused() {
        assert_eq!(
            Dimensions::new(0, 480),
            Err(GeometryError::ZeroDimension {
                width: 0,
                height: 480
            })
        );
    }
}
