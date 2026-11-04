//! HEVC `profile_tier_level()` (ITU-T H.265 7.3.3) with
//! `profilePresentFlag` 1, as the VPS and SPS carry it.

use super::bits::BitReader;
use crate::error::BitstreamError;

/// The general profile, tier, and level fields.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GeneralPtl {
    pub(crate) profile_space: u8,
    pub(crate) tier_flag: bool,
    pub(crate) profile_idc: u8,
    pub(crate) compatibility_flags: u32,
    /// The 48 bits from `general_progressive_source_flag` to
    /// `general_inbld_flag` or its reserved bit.
    pub(crate) constraint_flags: u64,
    pub(crate) level_idc: u8,
}

/// The fields of one sub-layer. The profile fields are 0 without
/// `sub_layer_profile_present_flag`, and the level is 0 without
/// `sub_layer_level_present_flag`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(dead_code, reason = "read by the Vulkan parameter set unit")]
pub(crate) struct SubLayerPtl {
    pub(crate) sub_layer_profile_present_flag: bool,
    pub(crate) sub_layer_level_present_flag: bool,
    pub(crate) sub_layer_profile_space: u8,
    pub(crate) sub_layer_tier_flag: bool,
    pub(crate) sub_layer_profile_idc: u8,
    /// `sub_layer_profile_compatibility_flag[ j ]` at bit `31 - j`.
    pub(crate) compatibility_flags: u32,
    /// The 48 bits from `sub_layer_progressive_source_flag` to
    /// `sub_layer_inbld_flag` or its reserved bit.
    pub(crate) constraint_flags: u64,
    pub(crate) sub_layer_level_idc: u8,
}

/// `profile_tier_level()` with `profilePresentFlag` 1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(dead_code, reason = "read by the Vulkan parameter set unit")]
pub(crate) struct ProfileTierLevel {
    pub(crate) general: GeneralPtl,
    /// The first `maxNumSubLayersMinus1` entries hold sub-layers 0 and up.
    pub(crate) sub_layers: [SubLayerPtl; 6],
}

/// Reads `profile_tier_level(1, max_sub_layers_minus1)` and returns its
/// general fields.
pub(crate) fn parse(
    reader: &mut BitReader<'_>,
    max_sub_layers_minus1: u8,
) -> Result<GeneralPtl, BitstreamError> {
    Ok(parse_full(reader, max_sub_layers_minus1)?.general)
}

/// Reads `profile_tier_level(1, max_sub_layers_minus1)`. The caller refuses a
/// `max_sub_layers_minus1` above 6.
pub(crate) fn parse_full(
    reader: &mut BitReader<'_>,
    max_sub_layers_minus1: u8,
) -> Result<ProfileTierLevel, BitstreamError> {
    let general = GeneralPtl {
        profile_space: reader.u8(2, "general_profile_space")?,
        tier_flag: reader.bit("general_tier_flag")?,
        profile_idc: reader.u8(5, "general_profile_idc")?,
        compatibility_flags: reader.bits(32, "general_profile_compatibility_flag")? as u32,
        constraint_flags: reader.bits(48, "general_constraint_indicator_flags")?,
        level_idc: reader.u8(8, "general_level_idc")?,
    };
    let count = usize::from(max_sub_layers_minus1.min(6));
    let mut sub_layers = [SubLayerPtl::default(); 6];
    for layer in sub_layers.iter_mut().take(count) {
        layer.sub_layer_profile_present_flag = reader.bit("sub_layer_profile_present_flag")?;
        layer.sub_layer_level_present_flag = reader.bit("sub_layer_level_present_flag")?;
    }
    if count > 0 {
        reader.skip(2 * (8 - count as u64), "reserved_zero_2bits")?;
    }
    for layer in sub_layers.iter_mut().take(count) {
        if layer.sub_layer_profile_present_flag {
            layer.sub_layer_profile_space = reader.u8(2, "sub_layer_profile_space")?;
            layer.sub_layer_tier_flag = reader.bit("sub_layer_tier_flag")?;
            layer.sub_layer_profile_idc = reader.u8(5, "sub_layer_profile_idc")?;
            layer.compatibility_flags =
                reader.bits(32, "sub_layer_profile_compatibility_flag")? as u32;
            layer.constraint_flags = reader.bits(48, "sub_layer_constraint_flags")?;
        }
        if layer.sub_layer_level_present_flag {
            layer.sub_layer_level_idc = reader.u8(8, "sub_layer_level_idc")?;
        }
    }
    Ok(ProfileTierLevel {
        general,
        sub_layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Main profile, main tier, level 3.0 (90): the 12 general bytes.
    const MAIN_L30: [u8; 12] = [0x01, 0x60, 0, 0, 0, 0x90, 0, 0, 0, 0, 0, 90];

    #[test]
    fn reads_the_general_fields() {
        let mut reader = BitReader::new(&MAIN_L30, "test");
        assert_eq!(
            parse(&mut reader, 0),
            Ok(GeneralPtl {
                profile_space: 0,
                tier_flag: false,
                profile_idc: 1,
                compatibility_flags: 0x6000_0000,
                constraint_flags: 0x9000_0000_0000,
                level_idc: 90,
            })
        );
        assert_eq!(reader.bits_left(), 0);
    }

    #[test]
    fn skips_sub_layer_fields() {
        // Two sub-layers: first has profile and level, second has level only.
        let mut bytes = MAIN_L30.to_vec();
        // Flags `11 01`, then 6 reserved pairs: 12 zero bits. Two bytes.
        bytes.extend_from_slice(&[0b1101_0000, 0]);
        bytes.extend_from_slice(&[0xaa; 11]);
        bytes.push(0x33);
        bytes.push(0x44);
        let mut reader = BitReader::new(&bytes, "test");
        assert_eq!(parse(&mut reader, 2).map(|p| p.level_idc), Ok(90));
        assert_eq!(reader.bits_left(), 0);
    }

    #[test]
    fn keeps_sub_layer_fields() {
        let mut bytes = MAIN_L30.to_vec();
        // First sub-layer: profile and level. Second: level only.
        bytes.extend_from_slice(&[0b1101_0000, 0]);
        // Space 1, tier 1, idc 2, compatibility, and constraint bits.
        bytes.push(0b0110_0010);
        bytes.extend_from_slice(&[0x20, 0, 0, 0]);
        bytes.extend_from_slice(&[0x90, 0, 0, 0, 0, 0x01]);
        bytes.push(93);
        bytes.push(60);
        let mut reader = BitReader::new(&bytes, "test");
        let ptl = parse_full(&mut reader, 2).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(reader.bits_left(), 0);
        assert_eq!(
            ptl.sub_layers[0],
            SubLayerPtl {
                sub_layer_profile_present_flag: true,
                sub_layer_level_present_flag: true,
                sub_layer_profile_space: 1,
                sub_layer_tier_flag: true,
                sub_layer_profile_idc: 2,
                compatibility_flags: 0x2000_0000,
                constraint_flags: 0x9000_0000_0001,
                sub_layer_level_idc: 93,
            }
        );
        assert_eq!(
            ptl.sub_layers[1],
            SubLayerPtl {
                sub_layer_level_present_flag: true,
                sub_layer_level_idc: 60,
                ..SubLayerPtl::default()
            }
        );
    }

    #[test]
    fn short_input_names_the_field() {
        let mut reader = BitReader::new(&MAIN_L30[..11], "test");
        assert!(matches!(
            parse(&mut reader, 0),
            Err(BitstreamError::Truncated {
                field: "general_level_idc",
                ..
            })
        ));
    }
}
