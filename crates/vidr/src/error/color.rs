//! Errors in colour policy.

use std::error::Error as StdError;
use std::fmt;

/// A stream's colour facts vidr does not take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ColorError {
    /// An H.273 code that marks HDR video.
    Hdr {
        /// The H.273 field: `TransferCharacteristics`, `MatrixCoefficients`
        /// or `ColourPrimaries`.
        field: &'static str,
        /// The code the stream carries.
        code: u8,
    },
    /// An H.273 matrix with no conversion in vidr.
    UnsupportedMatrix {
        /// The `MatrixCoefficients` code the stream carries.
        code: u8,
    },
}

impl fmt::Display for ColorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hdr { field, code } => write!(
                f,
                "H.273 {field} {code} marks HDR video, and HDR is unsupported. \
                 Use an SDR source, or tone-map the source first."
            ),
            Self::UnsupportedMatrix { code } => write!(
                f,
                "H.273 MatrixCoefficients {code} is unsupported. vidr converts 1 (BT.709), \
                 4 (FCC), 5 and 6 (BT.601), 7 (SMPTE 240M), and 2 (unspecified, as BT.601). \
                 Convert the source to one of these matrices."
            ),
        }
    }
}

impl StdError for ColorError {}
