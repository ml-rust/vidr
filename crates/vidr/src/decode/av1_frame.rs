//! What an AV1 sequence header says of the frames vidr can make from it.

use super::{ChromaFormat, ChromaLocation, ColorFacts, DecodeCodec, PixelFormat};
use crate::bitstream::av1::Av1SequenceFacts;
use crate::error::DecodeError;

/// Refuses a layout that has no vidr frame.
pub(crate) fn check_layout(
    bit_depth: u8,
    monochrome: bool,
    subsampling_x: bool,
    subsampling_y: bool,
) -> Result<(), DecodeError> {
    let chroma = match (monochrome, subsampling_x, subsampling_y) {
        (true, _, _) => ChromaFormat::Monochrome,
        (false, true, true) => ChromaFormat::Yuv420,
        (false, true, false) => ChromaFormat::Yuv422,
        (false, false, _) => ChromaFormat::Yuv444,
    };
    if chroma == ChromaFormat::Yuv420 && PixelFormat::for_bit_depth(bit_depth).is_some() {
        return Ok(());
    }
    Err(DecodeError::UnsupportedLayout {
        codec: DecodeCodec::Av1,
        bit_depth_luma: bit_depth,
        bit_depth_chroma: bit_depth,
        chroma,
    })
}

/// Refuses a sequence header whose layout has no vidr frame.
pub(crate) fn check_facts(facts: &Av1SequenceFacts) -> Result<(), DecodeError> {
    check_layout(
        facts.bit_depth,
        facts.monochrome,
        facts.subsampling_x,
        facts.subsampling_y,
    )
}

/// Colour from the sequence header. The wrapper's own colour accessors panic on
/// reserved code points in the stream, so this module never calls them.
pub(crate) fn color_facts(facts: &Av1SequenceFacts) -> ColorFacts {
    ColorFacts {
        matrix_coefficients: facts.matrix_coefficients,
        transfer_characteristics: facts.transfer_characteristics,
        colour_primaries: facts.color_primaries,
        full_range: facts.full_range,
        chroma_location: ChromaLocation::from_av1(facts.chroma_sample_position),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_420_at_8_or_10_bits_passes() {
        assert_eq!(check_layout(8, false, true, true), Ok(()));
        assert_eq!(check_layout(10, false, true, true), Ok(()));
        let refused = |bit_depth, chroma| {
            Err(DecodeError::UnsupportedLayout {
                codec: DecodeCodec::Av1,
                bit_depth_luma: bit_depth,
                bit_depth_chroma: bit_depth,
                chroma,
            })
        };
        assert_eq!(
            check_layout(12, false, true, true),
            refused(12, ChromaFormat::Yuv420)
        );
        assert_eq!(
            check_layout(8, true, true, true),
            refused(8, ChromaFormat::Monochrome)
        );
        assert_eq!(
            check_layout(10, false, true, false),
            refused(10, ChromaFormat::Yuv422)
        );
        assert_eq!(
            check_layout(8, false, false, false),
            refused(8, ChromaFormat::Yuv444)
        );
    }

    #[test]
    fn colour_comes_from_the_sequence_fields() {
        let facts = Av1SequenceFacts {
            bit_depth: 8,
            monochrome: false,
            subsampling_x: true,
            subsampling_y: true,
            chroma_sample_position: 2,
            color_primaries: 9,
            transfer_characteristics: 16,
            matrix_coefficients: 9,
            full_range: true,
        };
        assert_eq!(
            color_facts(&facts),
            ColorFacts {
                matrix_coefficients: 9,
                transfer_characteristics: 16,
                colour_primaries: 9,
                full_range: true,
                chroma_location: Some(ChromaLocation::TopLeft),
            }
        );
    }
}
