//! `hvcC` header fields derived from VPS, SPS, and PPS NAL units. The merge
//! rules are listed on `HevcConfig::from_parameter_sets`.

use super::bits::BitReader;
use super::hevc_ptl::{self, GeneralPtl};
use super::hevc_sps;
use super::nal::HevcNalType;
use super::rbsp::unescape;
use crate::error::BitstreamError;

/// The largest `min_spatial_segmentation_idc` the 12-bit field allows.
const MAX_SPATIAL_SEGMENTATION: u32 = 4095;

/// The `hvcC` header fields, before the arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeaderFields {
    pub(crate) profile_space: u8,
    pub(crate) tier_flag: bool,
    pub(crate) profile_idc: u8,
    pub(crate) compatibility_flags: u32,
    pub(crate) constraint_flags: u64,
    pub(crate) level_idc: u8,
    pub(crate) min_spatial_segmentation_idc: u16,
    pub(crate) parallelism_type: u8,
    pub(crate) chroma_format: u8,
    pub(crate) bit_depth_luma_minus8: u8,
    pub(crate) bit_depth_chroma_minus8: u8,
    pub(crate) num_temporal_layers: u8,
    pub(crate) temporal_id_nested: bool,
}

impl HeaderFields {
    const fn new() -> Self {
        Self {
            profile_space: 0,
            tier_flag: false,
            profile_idc: 0,
            compatibility_flags: u32::MAX,
            constraint_flags: (1 << 48) - 1,
            level_idc: 0,
            min_spatial_segmentation_idc: 0,
            parallelism_type: 0,
            chroma_format: 0,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            num_temporal_layers: 0,
            temporal_id_nested: false,
        }
    }

    fn merge_ptl(&mut self, ptl: &GeneralPtl) {
        self.profile_space = ptl.profile_space;
        if !self.tier_flag && ptl.tier_flag {
            self.level_idc = ptl.level_idc;
        } else {
            self.level_idc = self.level_idc.max(ptl.level_idc);
        }
        self.tier_flag |= ptl.tier_flag;
        self.profile_idc = self.profile_idc.max(ptl.profile_idc);
        self.compatibility_flags &= ptl.compatibility_flags;
        self.constraint_flags &= ptl.constraint_flags;
    }
}

fn invalid(
    structure: &'static str,
    field: &'static str,
    value: u64,
    allowed: &'static str,
) -> BitstreamError {
    BitstreamError::InvalidField {
        structure,
        field,
        value,
        allowed,
    }
}

/// Checks the 2-byte NAL unit header and returns the RBSP after it.
fn payload(
    nal: &[u8],
    kind: HevcNalType,
    structure: &'static str,
) -> Result<Vec<u8>, BitstreamError> {
    let mut rbsp = unescape(nal);
    let mut reader = BitReader::new(&rbsp, structure);
    if reader.bit("forbidden_zero_bit")? {
        return Err(invalid(structure, "forbidden_zero_bit", 1, "0"));
    }
    let found = reader.u8(6, "nal_unit_type")?;
    if found != kind.value() {
        return Err(invalid(
            structure,
            "nal_unit_type",
            found.into(),
            "the type of the parameter set",
        ));
    }
    let layer = reader.u8(6, "nuh_layer_id")?;
    if layer != 0 {
        return Err(invalid(
            structure,
            "nuh_layer_id",
            layer.into(),
            "0, as multi-layer HEVC needs an lhvC record, which vidr does not write",
        ));
    }
    reader.skip(3, "nuh_temporal_id_plus1")?;
    rbsp.drain(..2);
    Ok(rbsp)
}

/// Reads a VPS up to its `profile_tier_level()`.
fn parse_vps(nal: &[u8]) -> Result<(u8, GeneralPtl), BitstreamError> {
    const STRUCTURE: &str = "HEVC VPS";
    let rbsp = payload(nal, HevcNalType::VPS, STRUCTURE)?;
    let mut reader = BitReader::new(&rbsp, STRUCTURE);
    reader.skip(12, "vps_max_layers_minus1")?;
    let max_sub_layers_minus1 = reader.u8(3, "vps_max_sub_layers_minus1")?;
    if max_sub_layers_minus1 > 6 {
        return Err(invalid(
            STRUCTURE,
            "vps_max_sub_layers_minus1",
            max_sub_layers_minus1.into(),
            "0 to 6",
        ));
    }
    reader.skip(17, "vps_reserved_0xffff_16bits")?;
    let ptl = hevc_ptl::parse(&mut reader, max_sub_layers_minus1)?;
    Ok((max_sub_layers_minus1, ptl))
}

/// Reads a PPS up to `entropy_coding_sync_enabled_flag` and returns the
/// `parallelismType` it implies.
fn parse_pps(nal: &[u8]) -> Result<u8, BitstreamError> {
    const STRUCTURE: &str = "HEVC PPS";
    let rbsp = payload(nal, HevcNalType::PPS, STRUCTURE)?;
    let mut reader = BitReader::new(&rbsp, STRUCTURE);
    reader.ue("pps_pic_parameter_set_id")?;
    reader.ue("pps_seq_parameter_set_id")?;
    reader.skip(7, "cabac_init_present_flag")?;
    reader.ue("num_ref_idx_l0_default_active_minus1")?;
    reader.ue("num_ref_idx_l1_default_active_minus1")?;
    reader.se("init_qp_minus26")?;
    reader.skip(2, "transform_skip_enabled_flag")?;
    if reader.bit("cu_qp_delta_enabled_flag")? {
        reader.ue("diff_cu_qp_delta_depth")?;
    }
    reader.se("pps_cb_qp_offset")?;
    reader.se("pps_cr_qp_offset")?;
    reader.skip(4, "transquant_bypass_enabled_flag")?;
    let tiles = reader.bit("tiles_enabled_flag")?;
    let wavefront = reader.bit("entropy_coding_sync_enabled_flag")?;
    Ok(match (tiles, wavefront) {
        (true, true) => 0,
        (false, true) => 3,
        (true, false) => 2,
        (false, false) => 1,
    })
}

fn narrow(
    value: u32,
    max: u32,
    field: &'static str,
    allowed: &'static str,
) -> Result<u8, BitstreamError> {
    if value > max {
        return Err(invalid(hevc_sps::STRUCTURE, field, value.into(), allowed));
    }
    Ok(value as u8)
}

/// Derives the header fields from parameter sets in stream order: every
/// VPS, then every SPS, then every PPS.
///
/// # Errors
///
/// [`BitstreamError`] from a parameter set that does not parse, that sets
/// `nuh_layer_id`, or whose bit depth the 3-bit `hvcC` field cannot hold.
pub(crate) fn derive(
    vps: &[&[u8]],
    sps: &[&[u8]],
    pps: &[&[u8]],
) -> Result<HeaderFields, BitstreamError> {
    let mut fields = HeaderFields::new();
    for nal in vps {
        let (max_sub_layers_minus1, ptl) = parse_vps(nal)?;
        fields.num_temporal_layers = fields.num_temporal_layers.max(max_sub_layers_minus1 + 1);
        fields.merge_ptl(&ptl);
    }
    let mut min_spatial = MAX_SPATIAL_SEGMENTATION + 1;
    for nal in sps {
        let rbsp = payload(nal, HevcNalType::SPS, hevc_sps::STRUCTURE)?;
        let parsed = hevc_sps::parse(&rbsp)?;
        fields.num_temporal_layers = fields
            .num_temporal_layers
            .max(parsed.max_sub_layers_minus1 + 1);
        fields.temporal_id_nested = parsed.temporal_id_nesting;
        fields.merge_ptl(&parsed.ptl);
        fields.chroma_format = narrow(parsed.chroma_format_idc, 3, "chroma_format_idc", "0 to 3")?;
        fields.bit_depth_luma_minus8 = narrow(
            parsed.bit_depth_luma_minus8,
            7,
            "bit_depth_luma_minus8",
            "0 to 7",
        )?;
        fields.bit_depth_chroma_minus8 = narrow(
            parsed.bit_depth_chroma_minus8,
            7,
            "bit_depth_chroma_minus8",
            "0 to 7",
        )?;
        if let Some(value) = parsed.min_spatial_segmentation_idc {
            min_spatial = min_spatial.min(value);
        }
    }
    for nal in pps {
        fields.parallelism_type = parse_pps(nal)?;
    }
    fields.min_spatial_segmentation_idc = if min_spatial > MAX_SPATIAL_SEGMENTATION {
        0
    } else {
        min_spatial as u16
    };
    if fields.min_spatial_segmentation_idc == 0 {
        fields.parallelism_type = 0;
    }
    Ok(fields)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::bitstream::hevc_sps::tests::X265_SPS;

    /// The VPS of the x265 stream that [`X265_SPS`] comes from.
    pub(crate) const X265_VPS: [u8; 24] = [
        0x40, 0x01, 0x0c, 0x01, 0xff, 0xff, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00,
        0x03, 0x00, 0x00, 0x03, 0x00, 0x1e, 0x95, 0x90, 0x09,
    ];
    /// The PPS of the same stream.
    pub(crate) const X265_PPS: [u8; 7] = [0x44, 0x01, 0xc1, 0x72, 0xb4, 0x22, 0x40];

    #[test]
    fn derives_the_x265_header() {
        let fields = derive(&[&X265_VPS], &[&X265_SPS], &[&X265_PPS]);
        assert_eq!(
            fields,
            Ok(HeaderFields {
                profile_space: 0,
                tier_flag: false,
                profile_idc: 1,
                compatibility_flags: 0x6000_0000,
                constraint_flags: 0x9000_0000_0000,
                level_idc: 30,
                min_spatial_segmentation_idc: 0,
                parallelism_type: 0,
                chroma_format: 1,
                bit_depth_luma_minus8: 0,
                bit_depth_chroma_minus8: 0,
                num_temporal_layers: 1,
                temporal_id_nested: true,
            })
        );
    }

    #[test]
    fn pps_flags_give_the_parallelism_type() {
        assert_eq!(parse_pps(&X265_PPS), Ok(1));
    }

    #[test]
    fn ptl_merge_keeps_the_highest_tier_level() {
        let mut fields = HeaderFields::new();
        let main = GeneralPtl {
            profile_space: 0,
            tier_flag: false,
            profile_idc: 1,
            compatibility_flags: 0x6000_0000,
            constraint_flags: 0x9000_0000_0000,
            level_idc: 120,
        };
        let high = GeneralPtl {
            tier_flag: true,
            profile_idc: 2,
            compatibility_flags: 0x2000_0000,
            level_idc: 90,
            ..main
        };
        fields.merge_ptl(&main);
        fields.merge_ptl(&high);
        assert!(fields.tier_flag);
        assert_eq!(fields.level_idc, 90);
        assert_eq!(fields.profile_idc, 2);
        assert_eq!(fields.compatibility_flags, 0x2000_0000);
    }

    #[test]
    fn wrong_type_and_layer_are_errors() {
        assert!(matches!(
            derive(&[&X265_SPS], &[], &[]),
            Err(BitstreamError::InvalidField {
                field: "nal_unit_type",
                ..
            })
        ));
        let mut layered = X265_SPS;
        layered[1] |= 0x08;
        assert!(matches!(
            derive(&[], &[&layered], &[]),
            Err(BitstreamError::InvalidField {
                field: "nuh_layer_id",
                ..
            })
        ));
    }

    #[test]
    fn truncated_parameter_sets_never_panic() {
        for end in 0..X265_VPS.len() {
            let _ = derive(&[&X265_VPS[..end]], &[], &[]);
        }
        for end in 0..X265_SPS.len() {
            let _ = derive(&[], &[&X265_SPS[..end]], &[]);
        }
        for end in 0..X265_PPS.len() {
            let _ = derive(&[], &[], &[&X265_PPS[..end]]);
        }
        assert!(derive(&[], &[], &[&X265_PPS[..3]]).is_err());
    }
}
