//! The VUI fields H.264 (ITU-T H.264 E.1.1) and HEVC (ITU-T H.265 E.2.1)
//! share, from `aspect_ratio_info_present_flag` to
//! `chroma_sample_loc_type_bottom_field`.
//!
//! Both standards lay these fields out the same way. Everything before
//! `chroma_loc_info_present_flag` is read and kept only when it describes
//! the colour of the picture.

use super::bits::BitReader;
use crate::error::BitstreamError;

/// `aspect_ratio_idc` value that sends `sar_width` and `sar_height`.
const EXTENDED_SAR: u8 = 255;

/// The `colour_description_present_flag` fields, as ITU-T H.273 code
/// points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColourDescription {
    /// `colour_primaries`.
    pub colour_primaries: u8,
    /// `transfer_characteristics`.
    pub transfer_characteristics: u8,
    /// `matrix_coeffs`.
    pub matrix_coefficients: u8,
}

/// The fields `video_signal_type_present_flag` guards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoSignalType {
    /// `video_format`, 0 to 7.
    pub video_format: u8,
    /// `video_full_range_flag`.
    pub full_range: bool,
    /// `None` when `colour_description_present_flag` is 0.
    pub colour: Option<ColourDescription>,
}

/// `chroma_sample_loc_type_top_field` and `chroma_sample_loc_type_bottom_field`:
/// ITU-T H.273 `ChromaSampleLocType`, 0 to 5 each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChromaSampleLoc {
    /// Location of chroma samples in top fields and in frames.
    pub top_field: u8,
    /// Location of chroma samples in bottom fields.
    pub bottom_field: u8,
}

/// The colour fields of a VUI. A field is `None` when its presence flag is
/// 0, or when the SPS has no VUI.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VuiSignal {
    /// `video_signal_type_present_flag` fields.
    pub video_signal_type: Option<VideoSignalType>,
    /// `chroma_loc_info_present_flag` fields.
    pub chroma_sample_loc: Option<ChromaSampleLoc>,
}

/// Reads the VUI from `aspect_ratio_info_present_flag` through the chroma
/// sample location, and leaves `reader` after the last of them.
///
/// # Errors
///
/// [`BitstreamError::Truncated`] when the VUI ends early, and
/// [`BitstreamError::InvalidField`] for a chroma sample location type above
/// 5.
pub(crate) fn read_signal(
    reader: &mut BitReader<'_>,
    structure: &'static str,
) -> Result<VuiSignal, BitstreamError> {
    if reader.bit("aspect_ratio_info_present_flag")?
        && reader.u8(8, "aspect_ratio_idc")? == EXTENDED_SAR
    {
        reader.skip(32, "sar_width")?;
    }
    if reader.bit("overscan_info_present_flag")? {
        reader.skip(1, "overscan_appropriate_flag")?;
    }
    let video_signal_type = if reader.bit("video_signal_type_present_flag")? {
        let video_format = reader.u8(3, "video_format")?;
        let full_range = reader.bit("video_full_range_flag")?;
        let colour = if reader.bit("colour_description_present_flag")? {
            Some(ColourDescription {
                colour_primaries: reader.u8(8, "colour_primaries")?,
                transfer_characteristics: reader.u8(8, "transfer_characteristics")?,
                matrix_coefficients: reader.u8(8, "matrix_coeffs")?,
            })
        } else {
            None
        };
        Some(VideoSignalType {
            video_format,
            full_range,
            colour,
        })
    } else {
        None
    };
    let chroma_sample_loc = if reader.bit("chroma_loc_info_present_flag")? {
        Some(ChromaSampleLoc {
            top_field: location_type(reader, structure, "chroma_sample_loc_type_top_field")?,
            bottom_field: location_type(reader, structure, "chroma_sample_loc_type_bottom_field")?,
        })
    } else {
        None
    };
    Ok(VuiSignal {
        video_signal_type,
        chroma_sample_loc,
    })
}

fn location_type(
    reader: &mut BitReader<'_>,
    structure: &'static str,
    field: &'static str,
) -> Result<u8, BitstreamError> {
    let value = reader.ue(field)?;
    match u8::try_from(value) {
        Ok(kind) if kind <= 5 => Ok(kind),
        _ => Err(BitstreamError::InvalidField {
            structure,
            field,
            value: u64::from(value),
            allowed: "0 to 5",
        }),
    }
}

/// Test helper: packs a string of `0` and `1` characters into bytes, padded
/// with zero bits.
#[cfg(test)]
pub(crate) fn pack_bits(bits: &str) -> Vec<u8> {
    let mut bytes = vec![0u8; bits.len().div_ceil(8)];
    for (index, bit) in bits.bytes().enumerate() {
        if bit == b'1' {
            bytes[index / 8] |= 0x80 >> (index % 8);
        }
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(bits: &str) -> Result<VuiSignal, BitstreamError> {
        let bytes = pack_bits(bits);
        read_signal(&mut BitReader::new(&bytes, "test"), "test")
    }

    #[test]
    fn no_flags_gives_an_empty_signal() {
        assert_eq!(read("000"), Ok(VuiSignal::default()));
    }

    #[test]
    fn extended_sar_and_overscan_are_skipped() {
        // aspect 1, idc 255, 32 SAR bits, overscan 1 + 1 bit, no signal type,
        // chroma loc 1, top `010` (1), bottom `011` (2).
        let bits = "1".to_owned() + "11111111" + &"0".repeat(32) + "11" + "0" + "1" + "010" + "011";
        assert_eq!(
            read(&bits),
            Ok(VuiSignal {
                video_signal_type: None,
                chroma_sample_loc: Some(ChromaSampleLoc {
                    top_field: 1,
                    bottom_field: 2,
                }),
            })
        );
    }

    #[test]
    fn signal_type_without_a_colour_description() {
        // aspect 0, overscan 0, signal 1, format 5 `101`, full range 1,
        // description 0, chroma loc 0.
        assert_eq!(
            read(concat!("00", "1", "101", "1", "0", "0")),
            Ok(VuiSignal {
                video_signal_type: Some(VideoSignalType {
                    video_format: 5,
                    full_range: true,
                    colour: None,
                }),
                chroma_sample_loc: None,
            })
        );
    }

    #[test]
    fn colour_description_and_location_type_zero() {
        // signal 1, format 5, limited, description 1: 1, 1, 1; chroma loc 1,
        // top `1` (0), bottom `1` (0).
        let bits = "00".to_owned()
            + "1"
            + "101"
            + "0"
            + "1"
            + "00000001"
            + "00000001"
            + "00000001"
            + "1"
            + "1"
            + "1";
        let signal = read(&bits).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(
            signal.video_signal_type.and_then(|s| s.colour),
            Some(ColourDescription {
                colour_primaries: 1,
                transfer_characteristics: 1,
                matrix_coefficients: 1,
            })
        );
        assert_eq!(
            signal.chroma_sample_loc,
            Some(ChromaSampleLoc {
                top_field: 0,
                bottom_field: 0,
            })
        );
    }

    #[test]
    fn location_type_above_five_is_an_error() {
        // chroma loc 1, top `00111` (6).
        assert!(matches!(
            read(concat!("00", "0", "1", "00111", "1")),
            Err(BitstreamError::InvalidField {
                field: "chroma_sample_loc_type_top_field",
                value: 6,
                ..
            })
        ));
    }

    #[test]
    fn every_truncation_is_an_error() {
        let bits = "1".to_owned()
            + "11111111"
            + &"0".repeat(32)
            + "11"
            + "1"
            + "101"
            + "1"
            + "1"
            + "00000001"
            + "00000001"
            + "00000001"
            + "1"
            + "010"
            + "011";
        let bytes = pack_bits(&bits);
        for end in 0..bytes.len() {
            let result = read_signal(&mut BitReader::new(&bytes[..end], "test"), "test");
            assert!(
                matches!(result, Err(BitstreamError::Truncated { .. })),
                "a cut at byte {end} must be a truncation error"
            );
        }
        assert!(read(&bits).is_ok());
    }
}
