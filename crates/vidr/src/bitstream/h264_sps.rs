//! The leading fields of an H.264 sequence parameter set (ITU-T H.264
//! 7.3.2.1.1), as far as `avcC` needs them.

use super::bits::BitReader;
use super::nal::H264NalType;
use super::rbsp::unescape;
use crate::error::BitstreamError;

pub(super) const STRUCTURE: &str = "H.264 SPS";

/// Profiles whose SPS carries `chroma_format_idc` and the bit depths.
const CHROMA_PROFILES: [u8; 13] = [100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135];

/// SPS fields that `avcC` copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SpsSummary {
    pub(crate) profile_idc: u8,
    /// `constraint_set0_flag` to `constraint_set5_flag` and the two
    /// reserved bits: the byte between `profile_idc` and `level_idc`.
    pub(crate) constraint_flags: u8,
    pub(crate) level_idc: u8,
    pub(crate) chroma_format_idc: u8,
    pub(crate) bit_depth_luma_minus8: u8,
    pub(crate) bit_depth_chroma_minus8: u8,
}

/// Parses the summary from a whole SPS NAL unit, header byte included.
///
/// Profiles without chroma fields take the defaults ITU-T H.264 infers:
/// 4:2:0 and 8 bits.
pub(crate) fn parse(nal: &[u8]) -> Result<SpsSummary, BitstreamError> {
    let payload = payload(nal)?;
    read_leading(&mut BitReader::new(&payload, STRUCTURE))
}

/// True when an SPS of this profile carries `chroma_format_idc` and the
/// bit depths.
pub(super) fn has_chroma_fields(profile_idc: u8) -> bool {
    CHROMA_PROFILES.contains(&profile_idc)
}

/// Checks the NAL unit header and returns the RBSP after it.
pub(super) fn payload(nal: &[u8]) -> Result<Vec<u8>, BitstreamError> {
    let header = nal.first().copied().ok_or(BitstreamError::Truncated {
        structure: STRUCTURE,
        field: "nal_unit_type",
        offset: 0,
        size: 0,
    })?;
    if H264NalType::from_header(header) != H264NalType::SPS {
        return Err(BitstreamError::InvalidField {
            structure: STRUCTURE,
            field: "nal_unit_type",
            value: u64::from(header & 0x1f),
            allowed: "7 (SPS)",
        });
    }
    Ok(unescape(&nal[1..]))
}

/// Reads the fields up to and including the bit depths, and leaves `reader`
/// after them.
pub(super) fn read_leading(reader: &mut BitReader<'_>) -> Result<SpsSummary, BitstreamError> {
    let profile_idc = reader.u8(8, "profile_idc")?;
    let constraint_flags = reader.u8(8, "constraint_set_flags")?;
    let level_idc = reader.u8(8, "level_idc")?;
    reader.ue("seq_parameter_set_id")?;

    let mut summary = SpsSummary {
        profile_idc,
        constraint_flags,
        level_idc,
        chroma_format_idc: 1,
        bit_depth_luma_minus8: 0,
        bit_depth_chroma_minus8: 0,
    };
    if has_chroma_fields(profile_idc) {
        summary.chroma_format_idc = bounded(
            reader.ue("chroma_format_idc")?,
            3,
            "chroma_format_idc",
            "0 to 3",
        )?;
        if summary.chroma_format_idc == 3 {
            reader.bit("separate_colour_plane_flag")?;
        }
        summary.bit_depth_luma_minus8 = bounded(
            reader.ue("bit_depth_luma_minus8")?,
            6,
            "bit_depth_luma_minus8",
            "0 to 6",
        )?;
        summary.bit_depth_chroma_minus8 = bounded(
            reader.ue("bit_depth_chroma_minus8")?,
            6,
            "bit_depth_chroma_minus8",
            "0 to 6",
        )?;
    }
    Ok(summary)
}

fn bounded(
    value: u32,
    max: u32,
    field: &'static str,
    allowed: &'static str,
) -> Result<u8, BitstreamError> {
    if value > max {
        return Err(BitstreamError::InvalidField {
            structure: STRUCTURE,
            field,
            value: u64::from(value),
            allowed,
        });
    }
    Ok(value as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_takes_the_inferred_chroma_fields() {
        // profile 66, constraints 0xc0, level 30, sps id 0.
        let sps = [0x67, 66, 0xc0, 30, 0x80];
        assert_eq!(
            parse(&sps),
            Ok(SpsSummary {
                profile_idc: 66,
                constraint_flags: 0xc0,
                level_idc: 30,
                chroma_format_idc: 1,
                bit_depth_luma_minus8: 0,
                bit_depth_chroma_minus8: 0,
            })
        );
    }

    #[test]
    fn high_10_reads_chroma_and_bit_depth() {
        // profile 110, sps id 0 `1`, chroma 1 `010`, luma 2 `011`, chroma 2 `011`.
        let sps = [0x67, 110, 0, 40, 0b1010_0110, 0b1100_0000];
        let summary = parse(&sps);
        assert_eq!(
            summary.map(|s| (
                s.chroma_format_idc,
                s.bit_depth_luma_minus8,
                s.bit_depth_chroma_minus8
            )),
            Ok((1, 2, 2))
        );
    }

    #[test]
    fn emulation_prevention_is_removed_before_parsing() {
        // level_idc 0 and constraints 0 form `00 00`, so the stream holds `00 00 03`.
        let sps = [0x67, 66, 0, 0, 3, 0x80];
        assert_eq!(parse(&sps).map(|s| s.level_idc), Ok(0));
    }

    #[test]
    fn wrong_type_and_short_input_are_errors() {
        assert!(matches!(
            parse(&[0x68, 66, 0, 30, 0x80]),
            Err(BitstreamError::InvalidField {
                field: "nal_unit_type",
                ..
            })
        ));
        assert!(matches!(
            parse(&[0x67, 100, 0]),
            Err(BitstreamError::Truncated {
                field: "level_idc",
                ..
            })
        ));
        assert!(parse(&[]).is_err());
        // High profile cut before chroma_format_idc.
        assert!(parse(&[0x67, 100, 0, 30, 0x80]).is_err());
    }
}
