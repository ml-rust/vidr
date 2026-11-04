//! What `av1C` and the software decoders read from a sequence header. The parser itself is
//! `seq_header`.

use super::seq_header::{SequenceHeader, parse_sequence_header};
use crate::error::BitstreamError;

/// Sequence header fields that `av1C` copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SequenceFields {
    pub(crate) seq_profile: u8,
    pub(crate) seq_level_idx_0: u8,
    pub(crate) seq_tier_0: bool,
    pub(crate) high_bitdepth: bool,
    pub(crate) twelve_bit: bool,
    pub(crate) mono_chrome: bool,
    pub(crate) subsampling_x: bool,
    pub(crate) subsampling_y: bool,
    pub(crate) chroma_sample_position: u8,
    pub(crate) color_primaries: u8,
    pub(crate) transfer_characteristics: u8,
    pub(crate) matrix_coefficients: u8,
    pub(crate) full_range: bool,
}

/// What a decoder reads from a sequence header: sample layout and colour.
///
/// Colour code points follow ITU-T H.273. A header without
/// `color_description_present_flag` gives 2, unspecified, for all three.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Av1SequenceFacts {
    /// Bits per sample: 8, 10, or 12.
    pub bit_depth: u8,
    /// `mono_chrome`.
    pub monochrome: bool,
    /// `subsampling_x`.
    pub subsampling_x: bool,
    /// `subsampling_y`.
    pub subsampling_y: bool,
    /// `chroma_sample_position`, 0 to 3.
    pub chroma_sample_position: u8,
    /// `color_primaries`.
    pub color_primaries: u8,
    /// `transfer_characteristics`.
    pub transfer_characteristics: u8,
    /// `matrix_coefficients`.
    pub matrix_coefficients: u8,
    /// `color_range`: true for full range.
    pub full_range: bool,
}

/// Reads the [`Av1SequenceFacts`] of a sequence header OBU payload.
///
/// # Errors
///
/// [`BitstreamError`] for a payload that ends early or a field out of range.
pub fn av1_sequence_facts(payload: &[u8]) -> Result<Av1SequenceFacts, BitstreamError> {
    Ok(Av1SequenceFacts::from(&parse_sequence_header(payload)?))
}

impl From<&SequenceHeader> for Av1SequenceFacts {
    fn from(header: &SequenceHeader) -> Self {
        let color = &header.color;
        Self {
            bit_depth: color.bit_depth,
            monochrome: color.mono_chrome,
            subsampling_x: color.subsampling_x,
            subsampling_y: color.subsampling_y,
            chroma_sample_position: color.chroma_sample_position,
            color_primaries: color.color_primaries,
            transfer_characteristics: color.transfer_characteristics,
            matrix_coefficients: color.matrix_coefficients,
            full_range: color.color_range,
        }
    }
}

/// Parses a sequence header OBU payload.
pub(crate) fn parse(payload: &[u8]) -> Result<SequenceFields, BitstreamError> {
    let header = parse_sequence_header(payload)?;
    let first = header.operating_points.first().copied().unwrap_or_default();
    let color = header.color;
    Ok(SequenceFields {
        seq_profile: header.seq_profile,
        seq_level_idx_0: first.seq_level_idx,
        seq_tier_0: first.seq_tier,
        high_bitdepth: color.high_bitdepth,
        twelve_bit: color.twelve_bit,
        mono_chrome: color.mono_chrome,
        subsampling_x: color.subsampling_x,
        subsampling_y: color.subsampling_y,
        chroma_sample_position: color.chroma_sample_position,
        color_primaries: color.color_primaries,
        transfer_characteristics: color.transfer_characteristics,
        matrix_coefficients: color.matrix_coefficients,
        full_range: color.color_range,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The sequence header payload of a libaom 128x96 4:2:0 8-bit stream.
    pub(crate) const LIBAOM_PAYLOAD: [u8; 10] =
        [0x00, 0x00, 0x00, 0x03, 0x37, 0xfb, 0xe6, 0xd7, 0xcc, 0x02];

    #[test]
    fn reads_a_libaom_header() {
        assert_eq!(
            parse(&LIBAOM_PAYLOAD),
            Ok(SequenceFields {
                seq_profile: 0,
                seq_level_idx_0: 0,
                seq_tier_0: false,
                high_bitdepth: false,
                twelve_bit: false,
                mono_chrome: false,
                subsampling_x: true,
                subsampling_y: true,
                chroma_sample_position: 0,
                color_primaries: 2,
                transfer_characteristics: 2,
                matrix_coefficients: 2,
                full_range: false,
            })
        );
    }

    #[test]
    fn facts_name_the_depth_and_the_colour() {
        assert_eq!(
            av1_sequence_facts(&LIBAOM_PAYLOAD),
            Ok(Av1SequenceFacts {
                bit_depth: 8,
                monochrome: false,
                subsampling_x: true,
                subsampling_y: true,
                chroma_sample_position: 0,
                color_primaries: 2,
                transfer_characteristics: 2,
                matrix_coefficients: 2,
                full_range: false,
            })
        );
    }

    #[test]
    fn reduced_still_picture_header_reads_one_level() {
        // Profile 1, still 1, reduced 1, level 31, 4-bit widths, 4:4:4 8-bit.
        // 001 1 1 11111 0000 0000 0000 0000 000 0 0 0 0
        let bits = "00111111110000000000000000000000000";
        let mut bytes = vec![0u8; bits.len().div_ceil(8)];
        for (i, bit) in bits.bytes().enumerate() {
            if bit == b'1' {
                bytes[i / 8] |= 0x80 >> (i % 8);
            }
        }
        let fields = parse(&bytes);
        assert_eq!(
            fields.map(|f| (f.seq_profile, f.seq_level_idx_0, f.subsampling_x)),
            Ok((1, 31, false))
        );
    }

    #[test]
    fn truncations_never_panic() {
        for end in 0..LIBAOM_PAYLOAD.len() {
            let _ = parse(&LIBAOM_PAYLOAD[..end]);
        }
        assert!(parse(&LIBAOM_PAYLOAD[..3]).is_err());
        assert!(parse(&[0xe0]).is_err());
    }
}
