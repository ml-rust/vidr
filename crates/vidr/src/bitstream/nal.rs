//! NAL unit types for H.264 (ITU-T H.264 table 7-1) and HEVC (ITU-T H.265
//! table 7-1).

use std::fmt;

/// A parameter set kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParameterSetKind {
    /// HEVC video parameter set.
    Vps,
    /// Sequence parameter set.
    Sps,
    /// H.264 sequence parameter set extension.
    SpsExtension,
    /// Picture parameter set.
    Pps,
}

impl ParameterSetKind {
    /// The short name: `VPS`, `SPS`, `SPS extension`, or `PPS`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Vps => "VPS",
            Self::Sps => "SPS",
            Self::SpsExtension => "SPS extension",
            Self::Pps => "PPS",
        }
    }
}

impl fmt::Display for ParameterSetKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// An H.264 `nal_unit_type`, 0 to 31.
#[cfg(feature = "h264")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct H264NalType(u8);

#[cfg(feature = "h264")]
impl H264NalType {
    /// Coded slice of a non-IDR picture.
    pub const SLICE: Self = Self(1);
    /// Coded slice data partition A.
    pub const SLICE_DATA_A: Self = Self(2);
    /// Coded slice of an IDR picture.
    pub const IDR: Self = Self(5);
    /// Supplemental enhancement information.
    pub const SEI: Self = Self(6);
    /// Sequence parameter set.
    pub const SPS: Self = Self(7);
    /// Picture parameter set.
    pub const PPS: Self = Self(8);
    /// Access unit delimiter.
    pub const AUD: Self = Self(9);
    /// End of sequence.
    pub const END_OF_SEQUENCE: Self = Self(10);
    /// End of stream.
    pub const END_OF_STREAM: Self = Self(11);
    /// Filler data.
    pub const FILLER: Self = Self(12);
    /// Sequence parameter set extension.
    pub const SPS_EXTENSION: Self = Self(13);
    /// Subset sequence parameter set.
    pub const SUBSET_SPS: Self = Self(15);

    /// The type in the low 5 bits of a NAL unit header byte.
    pub const fn from_header(byte: u8) -> Self {
        Self(byte & 0x1f)
    }

    /// The type of a NAL unit, from its first byte. `None` for an empty unit.
    pub fn of(nal: &[u8]) -> Option<Self> {
        nal.first().map(|&byte| Self::from_header(byte))
    }

    /// The numeric `nal_unit_type`.
    pub const fn value(self) -> u8 {
        self.0
    }

    /// True for a coded slice of an IDR picture.
    pub const fn is_idr(self) -> bool {
        self.0 == Self::IDR.0
    }

    /// True for VCL NAL unit types 1 to 5.
    pub const fn is_vcl(self) -> bool {
        matches!(self.0, 1..=5)
    }

    /// True for an access unit delimiter.
    pub const fn is_aud(self) -> bool {
        self.0 == Self::AUD.0
    }

    /// True for SEI.
    pub const fn is_sei(self) -> bool {
        self.0 == Self::SEI.0
    }

    /// The parameter set kind, for SPS, SPS extension, and PPS.
    pub const fn parameter_set(self) -> Option<ParameterSetKind> {
        match self.0 {
            7 => Some(ParameterSetKind::Sps),
            8 => Some(ParameterSetKind::Pps),
            13 => Some(ParameterSetKind::SpsExtension),
            _ => None,
        }
    }
}

/// An HEVC `nal_unit_type`, 0 to 63.
#[cfg(feature = "hevc")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HevcNalType(u8);

#[cfg(feature = "hevc")]
impl HevcNalType {
    /// Trailing picture, sub-layer non-reference.
    pub const TRAIL_N: Self = Self(0);
    /// Trailing picture, sub-layer reference.
    pub const TRAIL_R: Self = Self(1);
    /// Broken link access, with leading pictures.
    pub const BLA_W_LP: Self = Self(16);
    /// Broken link access, with RADL pictures.
    pub const BLA_W_RADL: Self = Self(17);
    /// Broken link access, without leading pictures.
    pub const BLA_N_LP: Self = Self(18);
    /// Instantaneous decoding refresh, with RADL pictures.
    pub const IDR_W_RADL: Self = Self(19);
    /// Instantaneous decoding refresh, without leading pictures.
    pub const IDR_N_LP: Self = Self(20);
    /// Clean random access.
    pub const CRA: Self = Self(21);
    /// Video parameter set.
    pub const VPS: Self = Self(32);
    /// Sequence parameter set.
    pub const SPS: Self = Self(33);
    /// Picture parameter set.
    pub const PPS: Self = Self(34);
    /// Access unit delimiter.
    pub const AUD: Self = Self(35);
    /// End of sequence.
    pub const END_OF_SEQUENCE: Self = Self(36);
    /// End of bitstream.
    pub const END_OF_BITSTREAM: Self = Self(37);
    /// Filler data.
    pub const FILLER: Self = Self(38);
    /// Prefix SEI.
    pub const SEI_PREFIX: Self = Self(39);
    /// Suffix SEI.
    pub const SEI_SUFFIX: Self = Self(40);

    /// The type in bits 1 to 6 of the first NAL unit header byte.
    pub const fn from_header(byte: u8) -> Self {
        Self((byte >> 1) & 0x3f)
    }

    /// The type of a NAL unit, from its first byte. `None` for an empty unit.
    pub fn of(nal: &[u8]) -> Option<Self> {
        nal.first().map(|&byte| Self::from_header(byte))
    }

    /// The type for a numeric `nal_unit_type`. `None` above 63.
    pub const fn new(value: u8) -> Option<Self> {
        if value <= 63 { Some(Self(value)) } else { None }
    }

    /// The numeric `nal_unit_type`.
    pub const fn value(self) -> u8 {
        self.0
    }

    /// True for VCL NAL unit types 0 to 31.
    pub const fn is_vcl(self) -> bool {
        self.0 <= 31
    }

    /// True for intra random access point types 16 to 23, reserved 22 and
    /// 23 included, as ffmpeg's `hevc_mp4toannexb` counts them.
    pub const fn is_irap(self) -> bool {
        matches!(self.0, 16..=23)
    }

    /// True for `IDR_W_RADL` and `IDR_N_LP`.
    pub const fn is_idr(self) -> bool {
        self.0 == 19 || self.0 == 20
    }

    /// True for an access unit delimiter.
    pub const fn is_aud(self) -> bool {
        self.0 == Self::AUD.0
    }

    /// True for prefix and suffix SEI.
    pub const fn is_sei(self) -> bool {
        self.0 == Self::SEI_PREFIX.0 || self.0 == Self::SEI_SUFFIX.0
    }

    /// The parameter set kind, for VPS, SPS, and PPS.
    pub const fn parameter_set(self) -> Option<ParameterSetKind> {
        match self.0 {
            32 => Some(ParameterSetKind::Vps),
            33 => Some(ParameterSetKind::Sps),
            34 => Some(ParameterSetKind::Pps),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "h264")]
    #[test]
    fn h264_types_come_from_the_low_bits() {
        assert_eq!(H264NalType::from_header(0x67), H264NalType::SPS);
        assert_eq!(H264NalType::from_header(0x68), H264NalType::PPS);
        assert_eq!(H264NalType::from_header(0x65), H264NalType::IDR);
        assert!(H264NalType::IDR.is_idr() && H264NalType::IDR.is_vcl());
        assert!(!H264NalType::SEI.is_vcl() && H264NalType::SEI.is_sei());
        assert!(H264NalType::from_header(0x09).is_aud());
        assert_eq!(
            H264NalType::SPS_EXTENSION.parameter_set(),
            Some(ParameterSetKind::SpsExtension)
        );
        assert_eq!(H264NalType::SLICE.parameter_set(), None);
        assert_eq!(H264NalType::of(&[]), None);
    }

    #[cfg(feature = "hevc")]
    #[test]
    fn hevc_types_come_from_bits_one_to_six() {
        assert_eq!(HevcNalType::from_header(0x40), HevcNalType::VPS);
        assert_eq!(HevcNalType::from_header(0x42), HevcNalType::SPS);
        assert_eq!(HevcNalType::from_header(0x44), HevcNalType::PPS);
        assert_eq!(HevcNalType::from_header(0x28), HevcNalType::IDR_N_LP);
        assert_eq!(HevcNalType::from_header(0x4e), HevcNalType::SEI_PREFIX);
        assert!(HevcNalType::from_header(0x46).is_aud());
        assert_eq!(HevcNalType::new(64), None);
    }

    #[cfg(feature = "hevc")]
    #[test]
    fn hevc_irap_covers_16_to_23() {
        for value in 0..=63 {
            let kind = HevcNalType(value);
            assert_eq!(kind.is_irap(), (16..=23).contains(&value), "type {value}");
        }
        assert!(HevcNalType::CRA.is_irap() && !HevcNalType::CRA.is_idr());
        assert!(HevcNalType::IDR_W_RADL.is_idr());
        assert_eq!(
            HevcNalType::VPS.parameter_set(),
            Some(ParameterSetKind::Vps)
        );
    }
}
