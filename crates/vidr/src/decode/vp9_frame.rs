//! What a VP9 frame header says of the frame vidr can make from it.

use super::{
    ColorFacts, MATRIX_UNSPECIFIED, PRIMARIES_UNSPECIFIED, PixelFormat, TRANSFER_UNSPECIFIED,
    vp9_matrix,
};
use crate::bitstream::{ColorConfig, UncompressedHeader};
use crate::error::DecodeError;

/// Colour facts before any frame states them. A VP9 frame header states no transfer or
/// primaries.
pub(crate) const VP9_UNSTATED: ColorFacts = ColorFacts {
    matrix_coefficients: MATRIX_UNSPECIFIED,
    transfer_characteristics: TRANSFER_UNSPECIFIED,
    colour_primaries: PRIMARIES_UNSPECIFIED,
    full_range: false,
    chroma_location: None,
};

/// Refuses a frame header with no vidr frame. A `show_existing_frame` header states no
/// layout, and the frame it repeats passed this check.
pub(crate) fn check_layout(
    decoder: &'static str,
    header: &UncompressedHeader,
) -> Result<(), DecodeError> {
    if header.show_existing_frame {
        return Ok(());
    }
    let color = &header.color_config;
    let takes = matches!(header.profile, 0 | 2)
        && color.subsampling_x
        && color.subsampling_y
        && PixelFormat::for_bit_depth(color.bit_depth).is_some();
    if takes {
        return Ok(());
    }
    Err(DecodeError::Vp9Layout {
        decoder,
        profile: header.profile,
        bit_depth: color.bit_depth,
        subsampling_x: color.subsampling_x,
        subsampling_y: color.subsampling_y,
    })
}

/// Colour from the frame header.
pub(crate) fn vp9_color_facts(color: &ColorConfig) -> ColorFacts {
    ColorFacts {
        matrix_coefficients: vp9_matrix(color.color_space),
        full_range: color.color_range,
        ..VP9_UNSTATED
    }
}
