//! `HEVCDecoderConfigurationRecord`, the `hvcC` box payload (ISO/IEC
//! 14496-15 8.3.3.1).
//!
//! [`HevcConfig::from_parameter_sets`] fills the profile, tier, and level
//! fields from the VPS and SPS `profile_tier_level()`, and the other header
//! fields from the SPS and PPS. vidr never copies them from a source
//! record. The record describes the parameter sets it carries, and an
//! encoder can write parameter sets that differ from the source's, so a
//! copied field can describe a stream that no longer exists. A caller that
//! keeps the source's parameter sets keeps the source record unchanged.
//!
//! Readers ignore the reserved bits, as ffmpeg does, and writers set them to
//! ones.

use super::bytes::ByteReader;
use super::length_prefixed::LengthSize;
use super::nal::HevcNalType;
use crate::error::BitstreamError;

const STRUCTURE: &str = "hvcC";
/// The 48-bit `general_constraint_indicator_flags` mask.
const CONSTRAINT_MASK: u64 = (1 << 48) - 1;

/// One `hvcC` array: the NAL units of one type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HvccArray {
    /// True when every NAL unit of this type is in the array, none in band.
    pub array_completeness: bool,
    /// VPS, SPS, PPS, prefix SEI, or suffix SEI.
    pub nal_unit_type: HevcNalType,
    /// The NAL units, headers included.
    pub nal_units: Vec<Vec<u8>>,
}

/// An `HEVCDecoderConfigurationRecord`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HevcConfig {
    /// `general_profile_space`, 0 to 3.
    pub general_profile_space: u8,
    /// `general_tier_flag`.
    pub general_tier_flag: bool,
    /// `general_profile_idc`, 0 to 31.
    pub general_profile_idc: u8,
    /// `general_profile_compatibility_flags`.
    pub general_profile_compatibility_flags: u32,
    /// `general_constraint_indicator_flags`, 48 bits.
    pub general_constraint_indicator_flags: u64,
    /// `general_level_idc`.
    pub general_level_idc: u8,
    /// `min_spatial_segmentation_idc`, 0 to 4095.
    pub min_spatial_segmentation_idc: u16,
    /// `parallelismType`, 0 to 3.
    pub parallelism_type: u8,
    /// `chromaFormat`: `chroma_format_idc`, 0 to 3.
    pub chroma_format: u8,
    /// `bitDepthLumaMinus8`, 0 to 7.
    pub bit_depth_luma_minus8: u8,
    /// `bitDepthChromaMinus8`, 0 to 7.
    pub bit_depth_chroma_minus8: u8,
    /// `avgFrameRate` in frames per 256 seconds, 0 for unspecified.
    pub avg_frame_rate: u16,
    /// `constantFrameRate`, 0 to 3.
    pub constant_frame_rate: u8,
    /// `numTemporalLayers`, 0 to 7.
    pub num_temporal_layers: u8,
    /// `temporalIdNested`.
    pub temporal_id_nested: bool,
    /// Size of the NAL unit length field in samples.
    pub length_size: LengthSize,
    /// The arrays, in record order.
    pub arrays: Vec<HvccArray>,
}

fn invalid(field: &'static str, value: u64, allowed: &'static str) -> BitstreamError {
    BitstreamError::InvalidField {
        structure: STRUCTURE,
        field,
        value,
        allowed,
    }
}

fn array_type_allowed(kind: HevcNalType) -> bool {
    kind.parameter_set().is_some() || kind.is_sei()
}

impl HevcConfig {
    /// Parses an `hvcC` box payload.
    ///
    /// # Errors
    ///
    /// [`BitstreamError`] naming the field that is short or out of range, an
    /// array of a type other than VPS, SPS, PPS, or SEI, a NAL unit whose
    /// header type differs from its array's, or
    /// [`BitstreamError::TrailingBytes`].
    pub fn parse(bytes: &[u8]) -> Result<Self, BitstreamError> {
        let mut reader = ByteReader::new(bytes, STRUCTURE);
        let version = reader.u8("configurationVersion")?;
        if version != 1 {
            return Err(invalid("configurationVersion", version.into(), "1"));
        }
        let profile = reader.u8("general_profile_idc")?;
        let compatibility = reader.take(4, "general_profile_compatibility_flags")?;
        let constraint = reader.take(6, "general_constraint_indicator_flags")?;
        let general_level_idc = reader.u8("general_level_idc")?;
        let min_spatial_segmentation_idc = reader.u16("min_spatial_segmentation_idc")? & 0x0fff;
        let parallelism_type = reader.u8("parallelismType")? & 0x03;
        let chroma_format = reader.u8("chromaFormat")? & 0x03;
        let bit_depth_luma_minus8 = reader.u8("bitDepthLumaMinus8")? & 0x07;
        let bit_depth_chroma_minus8 = reader.u8("bitDepthChromaMinus8")? & 0x07;
        let avg_frame_rate = reader.u16("avgFrameRate")?;
        let packed = reader.u8("lengthSizeMinusOne")?;
        let minus_one = packed & 0x03;
        let length_size = LengthSize::from_minus_one(minus_one)
            .ok_or_else(|| invalid("lengthSizeMinusOne", minus_one.into(), "0, 1, or 3"))?;
        let array_count = reader.u8("numOfArrays")?;
        let mut arrays = Vec::with_capacity(array_count.into());
        for _ in 0..array_count {
            arrays.push(read_array(&mut reader)?);
        }
        reader.finish()?;
        Ok(Self {
            general_profile_space: profile >> 6,
            general_tier_flag: profile & 0x20 != 0,
            general_profile_idc: profile & 0x1f,
            general_profile_compatibility_flags: u32::from_be_bytes([
                compatibility[0],
                compatibility[1],
                compatibility[2],
                compatibility[3],
            ]),
            general_constraint_indicator_flags: constraint
                .iter()
                .fold(0, |acc, &byte| (acc << 8) | u64::from(byte)),
            general_level_idc,
            min_spatial_segmentation_idc,
            parallelism_type,
            chroma_format,
            bit_depth_luma_minus8,
            bit_depth_chroma_minus8,
            avg_frame_rate,
            constant_frame_rate: packed >> 6,
            num_temporal_layers: (packed >> 3) & 0x07,
            temporal_id_nested: packed & 0x04 != 0,
            length_size,
            arrays,
        })
    }

    /// Every NAL unit of every array, in record order.
    pub fn nal_units(&self) -> impl Iterator<Item = &[u8]> {
        self.arrays
            .iter()
            .flat_map(|array| array.nal_units.iter().map(Vec::as_slice))
    }

    /// Writes the record as an `hvcC` box payload.
    ///
    /// # Errors
    ///
    /// [`BitstreamError::InvalidField`] for a field, count, length, or NAL
    /// unit type the record cannot hold.
    pub fn to_bytes(&self) -> Result<Vec<u8>, BitstreamError> {
        self.check()?;
        let mut out =
            Vec::with_capacity(23 + self.nal_units().map(|unit| unit.len() + 2).sum::<usize>());
        out.push(1);
        out.push(
            self.general_profile_space << 6
                | u8::from(self.general_tier_flag) << 5
                | self.general_profile_idc,
        );
        out.extend_from_slice(&self.general_profile_compatibility_flags.to_be_bytes());
        out.extend_from_slice(&self.general_constraint_indicator_flags.to_be_bytes()[2..]);
        out.push(self.general_level_idc);
        out.extend_from_slice(&(0xf000 | self.min_spatial_segmentation_idc).to_be_bytes());
        out.push(0xfc | self.parallelism_type);
        out.push(0xfc | self.chroma_format);
        out.push(0xf8 | self.bit_depth_luma_minus8);
        out.push(0xf8 | self.bit_depth_chroma_minus8);
        out.extend_from_slice(&self.avg_frame_rate.to_be_bytes());
        out.push(
            self.constant_frame_rate << 6
                | self.num_temporal_layers << 3
                | u8::from(self.temporal_id_nested) << 2
                | self.length_size.minus_one(),
        );
        out.push(self.arrays.len() as u8);
        for array in &self.arrays {
            out.push(u8::from(array.array_completeness) << 7 | array.nal_unit_type.value());
            out.extend_from_slice(&(array.nal_units.len() as u16).to_be_bytes());
            for unit in &array.nal_units {
                out.extend_from_slice(&(unit.len() as u16).to_be_bytes());
                out.extend_from_slice(unit);
            }
        }
        Ok(out)
    }

    /// Checks every field, count, length, and NAL unit type.
    pub(crate) fn check(&self) -> Result<(), BitstreamError> {
        let ranges: [(&'static str, u64, u64, &'static str); 8] = [
            (
                "general_profile_space",
                self.general_profile_space.into(),
                3,
                "0 to 3",
            ),
            (
                "general_profile_idc",
                self.general_profile_idc.into(),
                31,
                "0 to 31",
            ),
            (
                "general_constraint_indicator_flags",
                self.general_constraint_indicator_flags,
                CONSTRAINT_MASK,
                "below 2^48",
            ),
            (
                "min_spatial_segmentation_idc",
                self.min_spatial_segmentation_idc.into(),
                4095,
                "0 to 4095",
            ),
            ("parallelismType", self.parallelism_type.into(), 3, "0 to 3"),
            ("chromaFormat", self.chroma_format.into(), 3, "0 to 3"),
            (
                "constantFrameRate",
                self.constant_frame_rate.into(),
                3,
                "0 to 3",
            ),
            (
                "numTemporalLayers",
                self.num_temporal_layers.into(),
                7,
                "0 to 7",
            ),
        ];
        for (field, value, max, allowed) in ranges {
            if value > max {
                return Err(invalid(field, value, allowed));
            }
        }
        for (field, value) in [
            ("bitDepthLumaMinus8", self.bit_depth_luma_minus8),
            ("bitDepthChromaMinus8", self.bit_depth_chroma_minus8),
        ] {
            if value > 7 {
                return Err(invalid(field, value.into(), "0 to 7"));
            }
        }
        if self.arrays.len() > 255 {
            return Err(invalid(
                "numOfArrays",
                self.arrays.len() as u64,
                "at most 255",
            ));
        }
        for array in &self.arrays {
            if !array_type_allowed(array.nal_unit_type) {
                return Err(invalid(
                    "NAL_unit_type",
                    array.nal_unit_type.value().into(),
                    "32 to 34 or 39 to 40",
                ));
            }
            if array.nal_units.len() > usize::from(u16::MAX) {
                return Err(invalid(
                    "numNalus",
                    array.nal_units.len() as u64,
                    "at most 65535",
                ));
            }
            for unit in &array.nal_units {
                check_unit(unit, array.nal_unit_type)?;
            }
        }
        Ok(())
    }
}

fn check_unit(unit: &[u8], kind: HevcNalType) -> Result<(), BitstreamError> {
    if unit.len() < 2 || unit.len() > usize::from(u16::MAX) {
        return Err(invalid("nalUnitLength", unit.len() as u64, "2 to 65535"));
    }
    let found = HevcNalType::from_header(unit[0]);
    if found != kind {
        return Err(invalid(
            "nal_unit_type",
            found.value().into(),
            "the NAL_unit_type of its array",
        ));
    }
    Ok(())
}

fn read_array(reader: &mut ByteReader<'_>) -> Result<HvccArray, BitstreamError> {
    let header = reader.u8("NAL_unit_type")?;
    let nal_unit_type = HevcNalType::from_header(header << 1);
    if !array_type_allowed(nal_unit_type) {
        return Err(invalid(
            "NAL_unit_type",
            nal_unit_type.value().into(),
            "32 to 34 or 39 to 40",
        ));
    }
    let count = reader.u16("numNalus")?;
    let mut nal_units = Vec::with_capacity(count.into());
    for _ in 0..count {
        let length = reader.u16("nalUnitLength")?;
        let unit = reader.take(length.into(), "nalUnit")?;
        check_unit(unit, nal_unit_type)?;
        nal_units.push(unit.to_vec());
    }
    Ok(HvccArray {
        array_completeness: header & 0x80 != 0,
        nal_unit_type,
        nal_units,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::bitstream::hevc_params::tests::{X265_PPS, X265_VPS};
    use crate::bitstream::hevc_sps::tests::X265_SPS;

    /// The `hvcC` that ffmpeg writes for the x265 stream of the constants.
    pub(crate) fn x265_record() -> Vec<u8> {
        let mut bytes = vec![
            0x01, 0x01, 0x60, 0x00, 0x00, 0x00, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1e, 0xf0,
            0x00, 0xfc, 0xfd, 0xf8, 0xf8, 0x00, 0x00, 0x0f, 0x03,
        ];
        for (kind, unit) in [
            (0xa0, &X265_VPS[..]),
            (0xa1, &X265_SPS[..]),
            (0xa2, &X265_PPS[..]),
        ] {
            bytes.extend_from_slice(&[kind, 0, 1, 0, unit.len() as u8]);
            bytes.extend_from_slice(unit);
        }
        bytes
    }

    #[test]
    fn parses_the_ffmpeg_record() {
        let config = HevcConfig::parse(&x265_record()).unwrap();
        assert_eq!(config.general_profile_idc, 1);
        assert_eq!(config.general_level_idc, 30);
        assert_eq!(config.general_constraint_indicator_flags, 0x9000_0000_0000);
        assert_eq!(
            (config.num_temporal_layers, config.temporal_id_nested),
            (1, true)
        );
        assert_eq!(config.nal_units().count(), 3);
        assert_eq!(config.to_bytes(), Ok(x265_record()));
    }
}
