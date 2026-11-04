//! VP9 colour for the sequence callback.
//!
//! NVDEC reports an all-zero video signal description for VP9, which reads
//! as the identity matrix. vidr reads `color_space` and `color_range` from
//! the frame header instead, and maps them to H.273 code points here. A VP9
//! header carries no primaries or transfer, so those stay unspecified and
//! the container fills them later.

use crate::bitstream::Vp9Color;
use crate::decode::{
    ColorFacts, MATRIX_UNSPECIFIED, PRIMARIES_UNSPECIFIED, TRANSFER_UNSPECIFIED, vp9_matrix,
};

/// Replaces the signal fields of `color` with the VP9 header's values.
///
/// `None` means no key frame header has been read yet: the matrix is
/// unspecified and the range is limited. The chroma location stays.
pub(super) fn apply(color: &mut ColorFacts, vp9: Option<Vp9Color>) {
    color.transfer_characteristics = TRANSFER_UNSPECIFIED;
    color.colour_primaries = PRIMARIES_UNSPECIFIED;
    (color.matrix_coefficients, color.full_range) = match vp9 {
        Some(header) => (vp9_matrix(header.color_space), header.full_range),
        None => (MATRIX_UNSPECIFIED, false),
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{ChromaLocation, MATRIX_BT470BG};

    #[test]
    fn each_color_space_maps_to_its_matrix() {
        let expected = [2, 5, 1, 6, 7, 9, 2, 0];
        for (space, matrix) in expected.into_iter().enumerate() {
            assert_eq!(vp9_matrix(space as u8), matrix, "color_space {space}");
        }
        // A value past the 3-bit field cannot come from the parser.
        assert_eq!(vp9_matrix(8), MATRIX_UNSPECIFIED);
    }

    fn facts() -> ColorFacts {
        ColorFacts {
            matrix_coefficients: 0,
            transfer_characteristics: 0,
            colour_primaries: 0,
            full_range: false,
            chroma_location: Some(ChromaLocation::Center),
        }
    }

    #[test]
    fn a_parsed_header_sets_matrix_and_range() {
        let mut color = facts();
        let header = Vp9Color {
            color_space: 1,
            full_range: true,
            bit_depth: 8,
        };
        apply(&mut color, Some(header));
        assert_eq!(color.matrix_coefficients, MATRIX_BT470BG);
        assert!(color.full_range);
        assert_eq!(color.transfer_characteristics, TRANSFER_UNSPECIFIED);
        assert_eq!(color.colour_primaries, PRIMARIES_UNSPECIFIED);
        assert_eq!(color.chroma_location, Some(ChromaLocation::Center));
    }

    #[test]
    fn no_header_is_unspecified_never_identity() {
        let mut color = facts();
        color.full_range = true;
        apply(&mut color, None);
        assert_eq!(color.matrix_coefficients, MATRIX_UNSPECIFIED);
        assert!(!color.full_range);
    }
}
