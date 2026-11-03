//! ffmpeg's `hevc_mp4toannexb` rules for one sample, without its duplicate
//! parameter sets. The `to_annexb` module documentation lists them.

use super::annexb::StartCode;
use super::nal::{HevcNalType, ParameterSetKind};
use crate::error::BitstreamError;

/// Converts the NAL units of one sample, with `config` as Annex-B.
pub(super) fn filter(
    config: &[u8],
    units: &[&[u8]],
    out: &mut Vec<u8>,
) -> Result<(), BitstreamError> {
    if let Some(short) = units.iter().find(|unit| unit.len() < 2) {
        return Err(BitstreamError::InvalidField {
            structure: "length-prefixed sample",
            field: "NAL unit length",
            value: short.len() as u64,
            allowed: "at least 2, the HEVC NAL unit header size",
        });
    }
    let type_of = |unit: &[u8]| HevcNalType::from_header(unit[0]);

    // Parameter sets before the first IRAP.
    let mut leading = [false; 3];
    let mut has_irap = false;
    for &unit in units {
        let kind = type_of(unit);
        if kind.is_irap() {
            has_irap = true;
            break;
        }
        match kind.parameter_set() {
            Some(ParameterSetKind::Vps) => leading[0] = true,
            Some(ParameterSetKind::Sps) => leading[1] = true,
            Some(ParameterSetKind::Pps) => leading[2] = true,
            _ => {}
        }
    }
    let ps_before_irap = has_irap && leading.contains(&true);
    let complete = has_irap && !leading.contains(&false);

    let mut inserted = complete;
    for &unit in units {
        let kind = type_of(unit);
        let anchor = kind.is_irap() || (ps_before_irap && kind.parameter_set().is_some());
        if anchor && !inserted {
            out.extend_from_slice(config);
            inserted = true;
        }
        out.extend_from_slice(StartCode::Four.bytes());
        out.extend_from_slice(unit);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::annexb;
    use crate::bitstream::{HevcConfig, HvccArray, LengthSize, ToAnnexB};

    fn sample(units: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for unit in units {
            out.extend_from_slice(&(unit.len() as u32).to_be_bytes());
            out.extend_from_slice(unit);
        }
        out
    }

    const VPS: &[u8] = &[0x40, 0x01, 0x0c];
    const SPS: &[u8] = &[0x42, 0x01, 0x01];
    const PPS: &[u8] = &[0x44, 0x01, 0xc1];
    const SEI: &[u8] = &[0x4e, 0x01, 0x05];
    const IDR: &[u8] = &[0x28, 0x01, 0xaf];
    const CRA: &[u8] = &[0x2a, 0x01, 0xaf];
    const TRAIL: &[u8] = &[0x02, 0x01, 0xd0];

    fn filter() -> ToAnnexB {
        let array = |kind, unit: &[u8]| HvccArray {
            array_completeness: true,
            nal_unit_type: kind,
            nal_units: vec![unit.to_vec()],
        };
        ToAnnexB::hevc(&HevcConfig {
            general_profile_space: 0,
            general_tier_flag: false,
            general_profile_idc: 1,
            general_profile_compatibility_flags: 0x6000_0000,
            general_constraint_indicator_flags: 0x9000_0000_0000,
            general_level_idc: 30,
            min_spatial_segmentation_idc: 0,
            parallelism_type: 0,
            chroma_format: 1,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            avg_frame_rate: 0,
            constant_frame_rate: 0,
            num_temporal_layers: 1,
            temporal_id_nested: true,
            length_size: LengthSize::Four,
            arrays: vec![
                array(HevcNalType::VPS, VPS),
                array(HevcNalType::SPS, SPS),
                array(HevcNalType::PPS, PPS),
                array(HevcNalType::SEI_PREFIX, SEI),
            ],
        })
    }

    fn annexb(units: &[&[u8]]) -> Vec<u8> {
        annexb::framed(units.iter().copied())
    }

    #[test]
    fn config_goes_before_every_irap_sample() {
        let mut f = filter();
        assert_eq!(
            f.filter(&sample(&[IDR])),
            Ok(annexb(&[VPS, SPS, PPS, SEI, IDR]))
        );
        assert_eq!(f.filter(&sample(&[TRAIL])), Ok(annexb(&[TRAIL])));
        assert_eq!(
            f.filter(&sample(&[SEI, CRA, CRA])),
            Ok(annexb(&[SEI, VPS, SPS, PPS, SEI, CRA, CRA]))
        );
    }

    #[test]
    fn complete_in_band_parameter_sets_are_not_duplicated() {
        let mut f = filter();
        assert_eq!(
            f.filter(&sample(&[VPS, SPS, PPS, IDR])),
            Ok(annexb(&[VPS, SPS, PPS, IDR]))
        );
    }

    #[test]
    fn partial_in_band_parameter_sets_follow_the_config() {
        let mut f = filter();
        assert_eq!(
            f.filter(&sample(&[SPS, PPS, IDR])),
            Ok(annexb(&[VPS, SPS, PPS, SEI, SPS, PPS, IDR]))
        );
    }

    #[test]
    fn short_units_are_errors() {
        let mut f = filter();
        assert!(matches!(
            f.filter(&sample(&[IDR, &[0x02]])),
            Err(BitstreamError::InvalidField { value: 1, .. })
        ));
        assert!(f.filter(&[0, 0, 0, 0]).is_err());
        assert!(f.filter(&[0, 0, 1]).is_err());
    }
}
