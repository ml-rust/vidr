//! HEVC `profile_tier_level()` (ITU-T H.265 7.3.3) with
//! `profilePresentFlag` 1, as the VPS and SPS carry it.

use super::bits::BitReader;
use crate::error::BitstreamError;

/// The general profile, tier, and level fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// Reads `profile_tier_level(1, max_sub_layers_minus1)` and returns its
/// general fields. Sub-layer fields are read and dropped.
pub(crate) fn parse(
    reader: &mut BitReader<'_>,
    max_sub_layers_minus1: u8,
) -> Result<GeneralPtl, BitstreamError> {
    let ptl = GeneralPtl {
        profile_space: reader.u8(2, "general_profile_space")?,
        tier_flag: reader.bit("general_tier_flag")?,
        profile_idc: reader.u8(5, "general_profile_idc")?,
        compatibility_flags: reader.bits(32, "general_profile_compatibility_flag")? as u32,
        constraint_flags: reader.bits(48, "general_constraint_indicator_flags")?,
        level_idc: reader.u8(8, "general_level_idc")?,
    };
    let sub_layers = usize::from(max_sub_layers_minus1);
    let mut present = [(false, false); 7];
    for flags in present.iter_mut().take(sub_layers) {
        *flags = (
            reader.bit("sub_layer_profile_present_flag")?,
            reader.bit("sub_layer_level_present_flag")?,
        );
    }
    if sub_layers > 0 {
        reader.skip(2 * (8 - sub_layers as u64), "reserved_zero_2bits")?;
    }
    for &(profile, level) in present.iter().take(sub_layers) {
        if profile {
            // Space, tier, idc, 32 compatibility flags, 48 constraint bits.
            reader.skip(88, "sub_layer_profile_idc")?;
        }
        if level {
            reader.skip(8, "sub_layer_level_idc")?;
        }
    }
    Ok(ptl)
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
