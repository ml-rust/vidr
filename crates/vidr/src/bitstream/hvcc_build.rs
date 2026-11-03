//! `HEVCDecoderConfigurationRecord` built from VPS, SPS, and PPS NAL units.

use super::hevc_params;
use super::hvcc::{HevcConfig, HvccArray};
use super::length_prefixed::LengthSize;
use super::nal::HevcNalType;
use crate::error::BitstreamError;

impl HevcConfig {
    /// Builds a record from parameter sets, as ffmpeg's `ff_isom_write_hvcc`
    /// does for an `hvc1` sample entry.
    ///
    /// The arrays hold VPS, SPS, and PPS in that order, each with
    /// `array_completeness` 1: the samples then carry no parameter sets.
    /// Set it to 0 for an `hev1` sample entry.
    ///
    /// The header fields merge across parameter sets by ISO/IEC 14496-15
    /// 8.3.3.1.2, as ffmpeg applies it:
    ///
    /// - `general_profile_space`: from the last parameter set.
    /// - `general_tier_flag`: the highest tier.
    /// - `general_level_idc`: the highest level of the highest tier.
    /// - `general_profile_idc`: the highest profile.
    /// - Compatibility and constraint flags: the bits every VPS and SPS sets.
    /// - `min_spatial_segmentation_idc`: the lowest SPS value, 0 when no SPS
    ///   has `bitstream_restriction_flag`.
    /// - `parallelismType`: from the last PPS, 0 when
    ///   `min_spatial_segmentation_idc` is 0.
    /// - `numTemporalLayers`: the most sub-layers of any VPS or SPS.
    /// - `temporalIdNested`, chroma format, and bit depths: from the last SPS.
    /// - `avgFrameRate` and `constantFrameRate`: 0, unspecified.
    ///
    /// # Errors
    ///
    /// [`BitstreamError::InvalidField`] when `vps`, `sps`, or `pps` is empty,
    /// and errors from the parameter set parses.
    pub fn from_parameter_sets(
        vps: &[&[u8]],
        sps: &[&[u8]],
        pps: &[&[u8]],
        length_size: LengthSize,
    ) -> Result<Self, BitstreamError> {
        for (units, field) in [
            (vps, "VPS numNalus"),
            (sps, "SPS numNalus"),
            (pps, "PPS numNalus"),
        ] {
            if units.is_empty() {
                return Err(BitstreamError::InvalidField {
                    structure: "hvcC",
                    field,
                    value: 0,
                    allowed: "at least 1",
                });
            }
        }
        let fields = hevc_params::derive(vps, sps, pps)?;
        let array = |kind, units: &[&[u8]]| HvccArray {
            array_completeness: true,
            nal_unit_type: kind,
            nal_units: units.iter().map(|unit| unit.to_vec()).collect(),
        };
        let config = Self {
            general_profile_space: fields.profile_space,
            general_tier_flag: fields.tier_flag,
            general_profile_idc: fields.profile_idc,
            general_profile_compatibility_flags: fields.compatibility_flags,
            general_constraint_indicator_flags: fields.constraint_flags,
            general_level_idc: fields.level_idc,
            min_spatial_segmentation_idc: fields.min_spatial_segmentation_idc,
            parallelism_type: fields.parallelism_type,
            chroma_format: fields.chroma_format,
            bit_depth_luma_minus8: fields.bit_depth_luma_minus8,
            bit_depth_chroma_minus8: fields.bit_depth_chroma_minus8,
            avg_frame_rate: 0,
            constant_frame_rate: 0,
            num_temporal_layers: fields.num_temporal_layers,
            temporal_id_nested: fields.temporal_id_nested,
            length_size,
            arrays: vec![
                array(HevcNalType::VPS, vps),
                array(HevcNalType::SPS, sps),
                array(HevcNalType::PPS, pps),
            ],
        };
        config.check()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::hevc_params::tests::{X265_PPS, X265_VPS};
    use crate::bitstream::hevc_sps::tests::X265_SPS;
    use crate::bitstream::hvcc::tests::x265_record;

    #[test]
    fn built_record_matches_ffmpeg() {
        let built = HevcConfig::from_parameter_sets(
            &[&X265_VPS],
            &[&X265_SPS],
            &[&X265_PPS],
            LengthSize::Four,
        );
        assert_eq!(
            built.as_ref().map(HevcConfig::to_bytes),
            Ok(Ok(x265_record()))
        );
        assert_eq!(HevcConfig::parse(&x265_record()).as_ref(), built.as_ref());
    }
}
