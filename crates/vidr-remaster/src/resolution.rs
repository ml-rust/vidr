//! Aspect-preserving output sizes with a 480-pixel minimum shorter edge.

use std::fmt;
use std::str::FromStr;

use vidr::{Dimensions, GeometryError, Rotation};

/// Smallest shorter edge any mode produces.
const MINIMUM_SHORT_EDGE: u64 = 480;

/// A standard shorter edge, in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StandardResolution {
    /// 480 pixels.
    P480,
    /// 720 pixels.
    P720,
    /// 1080 pixels.
    P1080,
    /// 1440 pixels.
    P1440,
    /// 2160 pixels.
    P2160,
    /// 4320 pixels.
    P4320,
}

impl StandardResolution {
    /// Every standard shorter edge, smallest first.
    pub const ALL: [Self; 6] = [
        Self::P480,
        Self::P720,
        Self::P1080,
        Self::P1440,
        Self::P2160,
        Self::P4320,
    ];

    /// The shorter edge in pixels.
    pub const fn short_edge(self) -> u32 {
        match self {
            Self::P480 => 480,
            Self::P720 => 720,
            Self::P1080 => 1080,
            Self::P1440 => 1440,
            Self::P2160 => 2160,
            Self::P4320 => 4320,
        }
    }
}

/// How to choose the output size from the source size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ResolutionMode {
    /// The largest standard shorter edge at or below the source's, at least 480.
    #[default]
    Auto,
    /// The source's shorter edge rounded to even, at least 480.
    Original,
    /// Twice the source's shorter edge, at least 480.
    Double,
    /// A fixed standard shorter edge.
    Fixed(StandardResolution),
}

/// Every mode name, in the order the help text lists them.
const MODE_NAMES: &str = "auto, original, 2x, 480, 720, 1080, 1440, 2160, 4320";

impl fmt::Display for ResolutionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Original => f.write_str("original"),
            Self::Double => f.write_str("2x"),
            Self::Fixed(size) => write!(f, "{}", size.short_edge()),
        }
    }
}

impl FromStr for ResolutionMode {
    type Err = ResolutionError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "auto" => Ok(Self::Auto),
            "original" => Ok(Self::Original),
            "2x" => Ok(Self::Double),
            _ => StandardResolution::ALL
                .into_iter()
                .find(|size| size.short_edge().to_string() == text)
                .map(Self::Fixed)
                .ok_or_else(|| ResolutionError::InvalidMode {
                    mode: text.to_owned(),
                }),
        }
    }
}

/// An error choosing an output size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolutionError {
    /// A mode name that is not one of the listed modes.
    InvalidMode {
        /// The text passed.
        mode: String,
    },
    /// An output side that does not fit 32 bits.
    TooLarge {
        /// The source size.
        input: Dimensions,
        /// The mode requested.
        mode: ResolutionMode,
    },
    /// An output size the geometry type refuses.
    Geometry(GeometryError),
}

impl fmt::Display for ResolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMode { mode } => {
                write!(f, "Invalid resolution mode '{mode}'. Use {MODE_NAMES}.")
            }
            Self::TooLarge { input, mode } => write!(
                f,
                "Output size for {input} in mode {mode} exceeds 4294967295 pixels on one side. \
                 Choose a smaller resolution mode."
            ),
            Self::Geometry(error) => fmt::Display::fmt(error, f),
        }
    }
}

impl std::error::Error for ResolutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Geometry(error) => Some(error),
            _ => None,
        }
    }
}

impl From<GeometryError> for ResolutionError {
    fn from(error: GeometryError) -> Self {
        Self::Geometry(error)
    }
}

/// `numerator / denominator` rounded to the nearest even integer, ties up.
const fn nearest_even(numerator: u128, denominator: u128) -> u128 {
    2 * ((numerator + denominator) / (2 * denominator))
}

/// Chooses even output dimensions, keeping portrait or landscape orientation.
///
/// # Errors
///
/// [`ResolutionError::TooLarge`] when an output side exceeds 32 bits.
pub fn choose_resolution(
    input: Dimensions,
    mode: ResolutionMode,
) -> Result<Dimensions, ResolutionError> {
    let width = u128::from(input.width());
    let height = u128::from(input.height());
    let short = width.min(height);
    let long = width.max(height);
    let minimum = u128::from(MINIMUM_SHORT_EDGE);
    let target = match mode {
        ResolutionMode::Auto => StandardResolution::ALL
            .into_iter()
            .map(|size| u128::from(size.short_edge()))
            .filter(|edge| *edge <= short)
            .max()
            .unwrap_or(minimum),
        ResolutionMode::Original => minimum.max(nearest_even(short, 1)),
        ResolutionMode::Double => minimum.max(short * 2),
        ResolutionMode::Fixed(size) => u128::from(size.short_edge()),
    };
    let other = target.max(nearest_even(long * target, short));
    let too_large = || ResolutionError::TooLarge { input, mode };
    let target = u32::try_from(target).map_err(|_| too_large())?;
    let other = u32::try_from(other).map_err(|_| too_large())?;
    let output = if input.width() <= input.height() {
        Dimensions::new(target, other)?
    } else {
        Dimensions::new(other, target)?
    };
    Ok(output)
}

/// Returns the size the encoder takes, from the display size
/// [`choose_resolution`] returns.
///
/// The model runs on coded pixels, and the output keeps the source's rotation.
/// So a source rotated by 90 or 270 degrees encodes its output with width and
/// height swapped. A 180-degree rotation keeps both.
pub const fn coded_output_size(display_output: Dimensions, rotation: Rotation) -> Dimensions {
    if rotation.swaps_axes() {
        display_output.transposed()
    } else {
        display_output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(width: u32, height: u32) -> Dimensions {
        Dimensions::new(width, height).expect("nonzero dimensions")
    }

    fn pair(dimensions: Dimensions) -> (u32, u32) {
        (dimensions.width(), dimensions.height())
    }

    fn choose(width: u32, height: u32, mode: &str) -> (u32, u32) {
        let mode = mode.parse().expect("valid mode");
        pair(choose_resolution(dims(width, height), mode).expect("output size"))
    }

    #[test]
    fn auto_uses_lower_standard_bound_in_both_orientations() {
        let cases = [
            ((406, 720), (480, 852)),
            ((720, 1280), (720, 1280)),
            ((900, 1600), (720, 1280)),
            ((1920, 1080), (1920, 1080)),
            ((3840, 2160), (3840, 2160)),
            ((320, 180), (854, 480)),
            ((1280, 719), (854, 480)),
            ((1280, 720), (1280, 720)),
            ((15360, 8640), (7680, 4320)),
        ];
        for ((width, height), expected) in cases {
            assert_eq!(choose(width, height, "auto"), expected, "{width}x{height}");
        }
    }

    #[test]
    fn explicit_modes_keep_minimum_and_even_dimensions() {
        let cases = [
            ((406, 720), "original", (480, 852)),
            ((720, 1280), "original", (720, 1280)),
            ((641, 721), "original", (642, 722)),
            ((406, 720), "2x", (812, 1440)),
            ((320, 180), "2x", (854, 480)),
            ((406, 720), "720", (720, 1276)),
            ((1920, 1080), "480", (854, 480)),
            ((1, 1), "auto", (480, 480)),
        ];
        for ((width, height), mode, expected) in cases {
            assert_eq!(
                choose(width, height, mode),
                expected,
                "{width}x{height} {mode}"
            );
        }
    }

    #[test]
    fn chosen_aspect_ratio_stays_within_one_pixel_rounding() {
        for (width, height) in [(481, 999), (1001, 563), (17, 19), (3840, 2161)] {
            for mode in ["auto", "original", "2x", "480", "1080"] {
                let (out_width, out_height) = choose(width, height, mode);
                let out_short = u64::from(out_width.min(out_height));
                let out_long = u64::from(out_width.max(out_height));
                assert!(out_short >= 480, "{width}x{height} {mode}");
                assert_eq!(out_width % 2, 0, "{width}x{height} {mode}");
                assert_eq!(out_height % 2, 0, "{width}x{height} {mode}");
                // |out_long - long * out_short / short| <= 1, scaled by short.
                let short = u64::from(width.min(height));
                let long = u64::from(width.max(height));
                let scaled = (out_long * short).abs_diff(long * out_short);
                assert!(scaled <= short, "{width}x{height} {mode}");
            }
        }
    }

    #[test]
    fn invalid_dimensions_and_modes_are_rejected() {
        // The type makes negative, boolean, float, and string sizes
        // unrepresentable. A zero side is the remaining invalid size.
        for (width, height) in [(0, 720), (406, 0)] {
            assert_eq!(
                Dimensions::new(width, height),
                Err(GeometryError::ZeroDimension { width, height })
            );
        }
        for mode in ["360", "2161", "AUTO", "", "480p"] {
            let error = mode.parse::<ResolutionMode>().expect_err("invalid mode");
            assert!(
                error.to_string().starts_with("Invalid resolution mode"),
                "{mode}: {error}"
            );
        }
    }

    #[test]
    fn mode_names_round_trip() {
        for name in [
            "auto", "original", "2x", "480", "720", "1080", "1440", "2160", "4320",
        ] {
            let mode: ResolutionMode = name.parse().expect("valid mode");
            assert_eq!(mode.to_string(), name);
        }
    }

    #[test]
    fn the_coded_output_size_swaps_sides_for_90_and_270_degrees() {
        let display = dims(720, 1280);
        let cases = [
            (Rotation::Upright, (720, 1280)),
            (Rotation::Clockwise90, (1280, 720)),
            (Rotation::Clockwise180, (720, 1280)),
            (Rotation::Clockwise270, (1280, 720)),
        ];
        for (rotation, expected) in cases {
            assert_eq!(pair(coded_output_size(display, rotation)), expected);
        }
    }

    #[test]
    fn a_rotated_phone_clip_encodes_landscape_and_displays_portrait() {
        // Coded 1920x1080, rotated 90 degrees, displays as 1080x1920.
        let coded = dims(1920, 1080);
        let rotation = Rotation::Clockwise90;
        let display = coded_output_size(coded, rotation);
        let output = choose_resolution(display, ResolutionMode::Fixed(StandardResolution::P720))
            .expect("output size");
        assert_eq!(pair(output), (720, 1280));
        let encoder = coded_output_size(output, rotation);
        assert_eq!(pair(encoder), (1280, 720));
        assert_eq!(encoder.width() % 2, 0);
        assert_eq!(encoder.height() % 2, 0);
    }

    #[test]
    fn an_output_side_beyond_32_bits_is_an_error() {
        let result = choose_resolution(dims(u32::MAX, 1), ResolutionMode::Double);
        assert!(matches!(result, Err(ResolutionError::TooLarge { .. })));
    }
}
