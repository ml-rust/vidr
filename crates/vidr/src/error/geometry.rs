//! Errors in frame geometry.

use std::error::Error as StdError;
use std::fmt;

use super::TimeError;

/// An error in frame geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeometryError {
    /// Frame dimensions with a zero side.
    ZeroDimension {
        /// The width passed.
        width: u32,
        /// The height passed.
        height: u32,
    },
    /// A rotation that is not a multiple of 90 degrees.
    UnsupportedRotation {
        /// The rotation passed, in degrees.
        degrees: i64,
    },
    /// A display matrix that is not a pure 0, 90, 180, or 270 degree rotation.
    UnsupportedDisplayMatrix {
        /// The matrix, in `tkhd` order `a b u c d v x y w`.
        matrix: [i32; 9],
    },
    /// An arithmetic error.
    Time(TimeError),
}

impl fmt::Display for GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension { width, height } => write!(
                f,
                "frame dimensions {width}x{height} have a zero side. \
                 Read the dimensions from the stream header."
            ),
            Self::UnsupportedRotation { degrees } => write!(
                f,
                "rotation of {degrees} degrees is not a multiple of 90. \
                 Normalize the source rotation to 0, 90, 180, or 270 degrees."
            ),
            Self::UnsupportedDisplayMatrix { matrix } => write!(
                f,
                "display matrix {matrix:?} is not a pure 0, 90, 180, or 270 degree rotation. \
                 Remux the source without scaling, mirroring, or shearing in its display matrix."
            ),
            Self::Time(error) => fmt::Display::fmt(error, f),
        }
    }
}

impl StdError for GeometryError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Time(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TimeError> for GeometryError {
    fn from(error: TimeError) -> Self {
        Self::Time(error)
    }
}
