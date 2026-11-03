//! `AVCDecoderConfigurationRecord`, the `avcC` box payload (ISO/IEC
//! 14496-15 5.3.3.1).
//!
//! Readers ignore the reserved bits, as ffmpeg does, and writers set them to
//! ones. The record carries the chroma and bit depth extension when
//! `AVCProfileIndication` is not 66, 77, or 88, the condition of ISO/IEC
//! 14496-15 since its 2014 edition. A reader also accepts a record without
//! the extension: muxers that follow the 2004 edition omit it.

use super::bytes::ByteReader;
use super::h264_sps;
use super::length_prefixed::LengthSize;
use super::nal::H264NalType;
use crate::error::BitstreamError;

const STRUCTURE: &str = "avcC";

/// The chroma and bit depth extension of `avcC`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AvcHighProfile {
    /// `chroma_format_idc` of the SPS, 0 to 3.
    pub chroma_format: u8,
    /// `bit_depth_luma_minus8` of the SPS, 0 to 6.
    pub bit_depth_luma_minus8: u8,
    /// `bit_depth_chroma_minus8` of the SPS, 0 to 6.
    pub bit_depth_chroma_minus8: u8,
    /// SPS extension NAL units, type 13.
    pub sps_ext: Vec<Vec<u8>>,
}

/// An `AVCDecoderConfigurationRecord`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AvcConfig {
    /// `AVCProfileIndication`: `profile_idc` of the SPS.
    pub profile_indication: u8,
    /// `profile_compatibility`: the SPS byte between `profile_idc` and
    /// `level_idc`.
    pub profile_compatibility: u8,
    /// `AVCLevelIndication`: `level_idc` of the SPS.
    pub level_indication: u8,
    /// Size of the NAL unit length field in samples.
    pub length_size: LengthSize,
    /// SPS NAL units, type 7, at most 31.
    pub sps: Vec<Vec<u8>>,
    /// PPS NAL units, type 8, at most 255.
    pub pps: Vec<Vec<u8>>,
    /// The extension. `None` writes a record without it.
    pub high_profile: Option<AvcHighProfile>,
}

/// True when ISO/IEC 14496-15 puts the chroma and bit depth extension in a
/// record of this profile.
pub const fn avcc_has_extension(profile_indication: u8) -> bool {
    !matches!(profile_indication, 66 | 77 | 88)
}

impl AvcConfig {
    /// Parses an `avcC` box payload.
    ///
    /// # Errors
    ///
    /// [`BitstreamError`] naming the field that is short, out of range, or of
    /// the wrong NAL unit type, or [`BitstreamError::TrailingBytes`].
    pub fn parse(bytes: &[u8]) -> Result<Self, BitstreamError> {
        let mut reader = ByteReader::new(bytes, STRUCTURE);
        let version = reader.u8("configurationVersion")?;
        if version != 1 {
            return Err(invalid("configurationVersion", version.into(), "1"));
        }
        let profile_indication = reader.u8("AVCProfileIndication")?;
        let profile_compatibility = reader.u8("profile_compatibility")?;
        let level_indication = reader.u8("AVCLevelIndication")?;
        let minus_one = reader.u8("lengthSizeMinusOne")? & 0x03;
        let length_size = LengthSize::from_minus_one(minus_one)
            .ok_or_else(|| invalid("lengthSizeMinusOne", minus_one.into(), "0, 1, or 3"))?;
        let sps_count = reader.u8("numOfSequenceParameterSets")? & 0x1f;
        let sps = read_units(
            &mut reader,
            sps_count.into(),
            H264NalType::SPS,
            "sequenceParameterSetLength",
        )?;
        let pps_count = reader.u8("numOfPictureParameterSets")?;
        let pps = read_units(
            &mut reader,
            pps_count.into(),
            H264NalType::PPS,
            "pictureParameterSetLength",
        )?;

        let high_profile = if avcc_has_extension(profile_indication) && reader.remaining() > 0 {
            let chroma_format = reader.u8("chroma_format")? & 0x03;
            let bit_depth_luma_minus8 = depth(
                reader.u8("bit_depth_luma_minus8")? & 0x07,
                "bit_depth_luma_minus8",
            )?;
            let bit_depth_chroma_minus8 = depth(
                reader.u8("bit_depth_chroma_minus8")? & 0x07,
                "bit_depth_chroma_minus8",
            )?;
            let count = reader.u8("numOfSequenceParameterSetExt")?;
            let sps_ext = read_units(
                &mut reader,
                count.into(),
                H264NalType::SPS_EXTENSION,
                "sequenceParameterSetExtLength",
            )?;
            Some(AvcHighProfile {
                chroma_format,
                bit_depth_luma_minus8,
                bit_depth_chroma_minus8,
                sps_ext,
            })
        } else {
            None
        };
        reader.finish()?;
        Ok(Self {
            profile_indication,
            profile_compatibility,
            level_indication,
            length_size,
            sps,
            pps,
            high_profile,
        })
    }

    /// Builds a record from parameter sets, as ffmpeg's `ff_isom_write_avcc`
    /// does.
    ///
    /// The profile, compatibility, and level bytes come from the first SPS.
    /// When the profile needs the extension, its chroma format and bit
    /// depths come from the first SPS too.
    ///
    /// # Errors
    ///
    /// [`BitstreamError::InvalidField`] when `sps` or `pps` is empty, when a
    /// unit has the wrong type, or when `sps_ext` is not empty for profile
    /// 66, 77, or 88. Errors from the SPS parse.
    pub fn from_parameter_sets(
        sps: &[&[u8]],
        pps: &[&[u8]],
        sps_ext: &[&[u8]],
        length_size: LengthSize,
    ) -> Result<Self, BitstreamError> {
        let first = sps
            .first()
            .ok_or_else(|| invalid("numOfSequenceParameterSets", 0, "at least 1"))?;
        if pps.is_empty() {
            return Err(invalid("numOfPictureParameterSets", 0, "at least 1"));
        }
        let summary = h264_sps::parse(first)?;
        let high_profile = if avcc_has_extension(summary.profile_idc) {
            Some(AvcHighProfile {
                chroma_format: summary.chroma_format_idc,
                bit_depth_luma_minus8: summary.bit_depth_luma_minus8,
                bit_depth_chroma_minus8: summary.bit_depth_chroma_minus8,
                sps_ext: sps_ext.iter().map(|unit| unit.to_vec()).collect(),
            })
        } else if !sps_ext.is_empty() {
            return Err(invalid(
                "numOfSequenceParameterSetExt",
                sps_ext.len() as u64,
                "0 for AVCProfileIndication 66, 77, and 88",
            ));
        } else {
            None
        };
        let config = Self {
            profile_indication: summary.profile_idc,
            profile_compatibility: summary.constraint_flags,
            level_indication: summary.level_idc,
            length_size,
            sps: sps.iter().map(|unit| unit.to_vec()).collect(),
            pps: pps.iter().map(|unit| unit.to_vec()).collect(),
            high_profile,
        };
        config.check()?;
        Ok(config)
    }

    /// Writes the record as an `avcC` box payload.
    ///
    /// # Errors
    ///
    /// [`BitstreamError::InvalidField`] for a count, length, NAL unit type,
    /// or extension field the record cannot hold.
    pub fn to_bytes(&self) -> Result<Vec<u8>, BitstreamError> {
        self.check()?;
        let mut out = vec![
            1,
            self.profile_indication,
            self.profile_compatibility,
            self.level_indication,
            0xfc | self.length_size.minus_one(),
            0xe0 | self.sps.len() as u8,
        ];
        write_units(&mut out, &self.sps);
        out.push(self.pps.len() as u8);
        write_units(&mut out, &self.pps);
        if let Some(high) = &self.high_profile {
            out.push(0xfc | high.chroma_format);
            out.push(0xf8 | high.bit_depth_luma_minus8);
            out.push(0xf8 | high.bit_depth_chroma_minus8);
            out.push(high.sps_ext.len() as u8);
            write_units(&mut out, &high.sps_ext);
        }
        Ok(out)
    }

    fn check(&self) -> Result<(), BitstreamError> {
        check_units(
            &self.sps,
            31,
            H264NalType::SPS,
            "numOfSequenceParameterSets",
            "sequenceParameterSetLength",
        )?;
        check_units(
            &self.pps,
            255,
            H264NalType::PPS,
            "numOfPictureParameterSets",
            "pictureParameterSetLength",
        )?;
        let Some(high) = &self.high_profile else {
            return Ok(());
        };
        if !avcc_has_extension(self.profile_indication) {
            return Err(invalid(
                "AVCProfileIndication",
                self.profile_indication.into(),
                "other than 66, 77, and 88 in a record with the extension",
            ));
        }
        if high.chroma_format > 3 {
            return Err(invalid(
                "chroma_format",
                high.chroma_format.into(),
                "0 to 3",
            ));
        }
        depth(high.bit_depth_luma_minus8, "bit_depth_luma_minus8")?;
        depth(high.bit_depth_chroma_minus8, "bit_depth_chroma_minus8")?;
        check_units(
            &high.sps_ext,
            255,
            H264NalType::SPS_EXTENSION,
            "numOfSequenceParameterSetExt",
            "sequenceParameterSetExtLength",
        )
    }
}

fn invalid(field: &'static str, value: u64, allowed: &'static str) -> BitstreamError {
    BitstreamError::InvalidField {
        structure: STRUCTURE,
        field,
        value,
        allowed,
    }
}

fn depth(value: u8, field: &'static str) -> Result<u8, BitstreamError> {
    if value > 6 {
        return Err(invalid(field, value.into(), "0 to 6"));
    }
    Ok(value)
}

fn nal_type_allowed(kind: H264NalType) -> &'static str {
    match kind.value() {
        7 => "a NAL unit of type 7 (SPS)",
        8 => "a NAL unit of type 8 (PPS)",
        _ => "a NAL unit of type 13 (SPS extension)",
    }
}

fn read_units(
    reader: &mut ByteReader<'_>,
    count: usize,
    kind: H264NalType,
    length_field: &'static str,
) -> Result<Vec<Vec<u8>>, BitstreamError> {
    let mut units = Vec::with_capacity(count);
    for _ in 0..count {
        let length = reader.u16(length_field)?;
        if length == 0 {
            return Err(invalid(length_field, 0, "at least 1"));
        }
        let unit = reader.take(length.into(), length_field)?;
        let found = H264NalType::from_header(unit[0]);
        if found != kind {
            return Err(invalid(
                "nal_unit_type",
                found.value().into(),
                nal_type_allowed(kind),
            ));
        }
        units.push(unit.to_vec());
    }
    Ok(units)
}

fn check_units(
    units: &[Vec<u8>],
    max_count: usize,
    kind: H264NalType,
    count_field: &'static str,
    length_field: &'static str,
) -> Result<(), BitstreamError> {
    if units.len() > max_count {
        let allowed = if max_count == 31 {
            "at most 31"
        } else {
            "at most 255"
        };
        return Err(invalid(count_field, units.len() as u64, allowed));
    }
    for unit in units {
        if unit.is_empty() || unit.len() > usize::from(u16::MAX) {
            return Err(invalid(length_field, unit.len() as u64, "1 to 65535"));
        }
        let found = H264NalType::from_header(unit[0]);
        if found != kind {
            return Err(invalid(
                "nal_unit_type",
                found.value().into(),
                nal_type_allowed(kind),
            ));
        }
    }
    Ok(())
}

fn write_units(out: &mut Vec<u8>, units: &[Vec<u8>]) {
    for unit in units {
        out.extend_from_slice(&(unit.len() as u16).to_be_bytes());
        out.extend_from_slice(unit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Main profile SPS: profile 77, constraints 0x40, level 30, id 0.
    const MAIN_SPS: [u8; 5] = [0x67, 77, 0x40, 30, 0x80];
    /// A High profile SPS: profile 100, level 31, id 0, 4:2:0, 8 bits.
    const HIGH_SPS: [u8; 6] = [0x67, 100, 0, 31, 0b1010_1100, 0x80];
    const PPS: [u8; 3] = [0x68, 0xce, 0x38];

    fn main_record() -> Vec<u8> {
        let mut bytes = vec![1, 77, 0x40, 30, 0xff, 0xe1, 0, 5];
        bytes.extend_from_slice(&MAIN_SPS);
        bytes.extend_from_slice(&[1, 0, 3]);
        bytes.extend_from_slice(&PPS);
        bytes
    }

    #[test]
    fn parses_and_writes_a_main_profile_record() {
        let bytes = main_record();
        let config = AvcConfig::parse(&bytes);
        assert_eq!(
            config,
            Ok(AvcConfig {
                profile_indication: 77,
                profile_compatibility: 0x40,
                level_indication: 30,
                length_size: LengthSize::Four,
                sps: vec![MAIN_SPS.to_vec()],
                pps: vec![PPS.to_vec()],
                high_profile: None,
            })
        );
        assert_eq!(config.and_then(|c| c.to_bytes()), Ok(bytes));
    }

    #[test]
    fn high_profile_record_carries_the_extension() {
        let built = AvcConfig::from_parameter_sets(&[&HIGH_SPS], &[&PPS], &[], LengthSize::Four);
        let bytes = built.as_ref().map(|c| c.to_bytes());
        let Ok(Ok(bytes)) = bytes else {
            panic!("build failed: {built:?}");
        };
        assert_eq!(&bytes[..6], &[1, 100, 0, 31, 0xff, 0xe1]);
        assert_eq!(&bytes[bytes.len() - 4..], &[0xfd, 0xf8, 0xf8, 0]);
        assert_eq!(AvcConfig::parse(&bytes).as_ref(), built.as_ref());
    }
}
