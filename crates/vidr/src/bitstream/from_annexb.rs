//! Annex-B access units, as NVENC writes them, to MP4 and Matroska samples
//! and a config record.
//!
//! - Parameter sets and access unit delimiters leave the samples. Every
//!   other NAL unit, SEI included, stays.
//! - The config record comes from the parameter sets of the first access
//!   unit. Its HEVC arrays have `array_completeness` 1, for an `hvc1`
//!   sample entry.
//! - A later parameter set must equal one in the record byte for byte. A
//!   different one returns [`BitstreamError::ParameterSetChanged`]: the mux
//!   needs a new sample description, and vidr does not write one.

use super::annexb;
#[cfg(feature = "h264")]
use super::avcc::AvcConfig;
#[cfg(feature = "hevc")]
use super::hvcc::HevcConfig;
use super::length_prefixed::{self, LengthSize};
#[cfg(feature = "h264")]
use super::nal::H264NalType;
#[cfg(feature = "hevc")]
use super::nal::HevcNalType;
use super::nal::ParameterSetKind;
use crate::error::BitstreamError;

/// A codec whose samples are NAL units.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NalCodec {
    /// H.264, ITU-T H.264.
    #[cfg(feature = "h264")]
    H264,
    /// HEVC, ITU-T H.265.
    #[cfg(feature = "hevc")]
    Hevc,
}

impl NalCodec {
    /// The codec name in error messages: `H.264` or `HEVC`.
    pub const fn name(self) -> &'static str {
        match self {
            #[cfg(feature = "h264")]
            Self::H264 => "H.264",
            #[cfg(feature = "hevc")]
            Self::Hevc => "HEVC",
        }
    }

    /// The parameter set kind of a NAL unit, and whether it is an access
    /// unit delimiter.
    fn classify(self, unit: &[u8]) -> (Option<ParameterSetKind>, bool) {
        match self {
            #[cfg(feature = "h264")]
            Self::H264 => {
                let kind = H264NalType::from_header(unit[0]);
                (kind.parameter_set(), kind.is_aud())
            }
            #[cfg(feature = "hevc")]
            Self::Hevc => {
                let kind = HevcNalType::from_header(unit[0]);
                (kind.parameter_set(), kind.is_aud())
            }
        }
    }
}

/// A config record for a NAL unit codec.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NalConfig {
    /// `avcC`.
    #[cfg(feature = "h264")]
    Avc(AvcConfig),
    /// `hvcC`.
    #[cfg(feature = "hevc")]
    Hevc(HevcConfig),
}

impl NalConfig {
    /// Writes the box payload.
    ///
    /// # Errors
    ///
    /// As the `to_bytes` of the record type.
    pub fn to_bytes(&self) -> Result<Vec<u8>, BitstreamError> {
        match self {
            #[cfg(feature = "h264")]
            Self::Avc(config) => config.to_bytes(),
            #[cfg(feature = "hevc")]
            Self::Hevc(config) => config.to_bytes(),
        }
    }
}

/// A per-track filter from Annex-B access units to length-prefixed samples.
///
/// Pass one access unit per call, in decode order.
#[derive(Clone, Debug)]
pub struct FromAnnexB {
    codec: NalCodec,
    length_size: LengthSize,
    config: Option<NalConfig>,
    /// The record's parameter sets, for the change check.
    parameter_sets: Vec<(ParameterSetKind, Vec<u8>)>,
    access_unit: u64,
}

impl FromAnnexB {
    /// Creates a filter that writes NAL unit lengths of `length_size`.
    pub const fn new(codec: NalCodec, length_size: LengthSize) -> Self {
        Self {
            codec,
            length_size,
            config: None,
            parameter_sets: Vec::new(),
            access_unit: 0,
        }
    }

    /// The config record, after the first access unit.
    pub const fn config(&self) -> Option<&NalConfig> {
        self.config.as_ref()
    }

    /// Converts one access unit to a sample.
    ///
    /// # Errors
    ///
    /// As [`FromAnnexB::filter_into`].
    pub fn filter(&mut self, access_unit: &[u8]) -> Result<Vec<u8>, BitstreamError> {
        let mut out = Vec::with_capacity(access_unit.len());
        self.filter_into(access_unit, &mut out)?;
        Ok(out)
    }

    /// Converts one access unit and appends the sample to `out`.
    ///
    /// # Errors
    ///
    /// - [`BitstreamError::MissingStartCode`] for data before the first start
    ///   code.
    /// - [`BitstreamError::MissingParameterSet`] when the first access unit
    ///   lacks a parameter set the record needs.
    /// - [`BitstreamError::ParameterSetChanged`] for a parameter set the
    ///   record does not hold.
    /// - Errors from the record build and from the length field write.
    ///
    /// On an error `out` and the filter state are unchanged, and the next
    /// call counts the next access unit.
    pub fn filter_into(
        &mut self,
        access_unit: &[u8],
        out: &mut Vec<u8>,
    ) -> Result<(), BitstreamError> {
        let index = self.access_unit;
        self.access_unit += 1;
        let mut kept = Vec::new();
        let mut found: Vec<(ParameterSetKind, &[u8])> = Vec::new();
        for unit in annexb::split(access_unit)? {
            match self.codec.classify(unit) {
                (Some(kind), _) => {
                    if !found.iter().any(|(k, u)| *k == kind && *u == unit) {
                        found.push((kind, unit));
                    }
                }
                (None, true) => {}
                (None, false) => kept.push(unit),
            }
        }

        let built = match &self.config {
            Some(_) => {
                self.check_unchanged(&found, index)?;
                None
            }
            None => Some(self.build(&found, index)?),
        };
        length_prefixed::join(kept, self.length_size, out)?;
        if let Some(config) = built {
            self.config = Some(config);
            self.parameter_sets = found
                .into_iter()
                .map(|(kind, unit)| (kind, unit.to_vec()))
                .collect();
        }
        Ok(())
    }

    fn check_unchanged(
        &self,
        found: &[(ParameterSetKind, &[u8])],
        index: u64,
    ) -> Result<(), BitstreamError> {
        for (kind, unit) in found {
            let known = self
                .parameter_sets
                .iter()
                .any(|(k, u)| k == kind && u.as_slice() == *unit);
            if !known {
                return Err(BitstreamError::ParameterSetChanged {
                    codec: self.codec.name(),
                    kind: kind.name(),
                    access_unit: index,
                });
            }
        }
        Ok(())
    }

    fn build(
        &self,
        found: &[(ParameterSetKind, &[u8])],
        index: u64,
    ) -> Result<NalConfig, BitstreamError> {
        let of = |wanted: ParameterSetKind| -> Vec<&[u8]> {
            found
                .iter()
                .filter(|(kind, _)| *kind == wanted)
                .map(|(_, unit)| *unit)
                .collect()
        };
        let missing = |kind: ParameterSetKind| BitstreamError::MissingParameterSet {
            codec: self.codec.name(),
            kind: kind.name(),
            access_unit: index,
        };
        let required = match self.codec {
            #[cfg(feature = "h264")]
            NalCodec::H264 => &[ParameterSetKind::Sps, ParameterSetKind::Pps][..],
            #[cfg(feature = "hevc")]
            NalCodec::Hevc => &[
                ParameterSetKind::Vps,
                ParameterSetKind::Sps,
                ParameterSetKind::Pps,
            ][..],
        };
        if let Some(&kind) = required.iter().find(|&&kind| of(kind).is_empty()) {
            return Err(missing(kind));
        }
        match self.codec {
            #[cfg(feature = "h264")]
            NalCodec::H264 => Ok(NalConfig::Avc(AvcConfig::from_parameter_sets(
                &of(ParameterSetKind::Sps),
                &of(ParameterSetKind::Pps),
                &of(ParameterSetKind::SpsExtension),
                self.length_size,
            )?)),
            #[cfg(feature = "hevc")]
            NalCodec::Hevc => Ok(NalConfig::Hevc(HevcConfig::from_parameter_sets(
                &of(ParameterSetKind::Vps),
                &of(ParameterSetKind::Sps),
                &of(ParameterSetKind::Pps),
                self.length_size,
            )?)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn annexb(units: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for unit in units {
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(unit);
        }
        out
    }

    fn sample(units: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for unit in units {
            out.extend_from_slice(&(unit.len() as u32).to_be_bytes());
            out.extend_from_slice(unit);
        }
        out
    }

    #[cfg(feature = "h264")]
    mod h264 {
        use super::*;

        const SPS: &[u8] = &[0x67, 77, 0x40, 30, 0x80];
        const OTHER_SPS: &[u8] = &[0x67, 77, 0x40, 31, 0x80];
        const PPS: &[u8] = &[0x68, 0xce, 0x38];
        const AUD: &[u8] = &[0x09, 0xf0];
        const SEI: &[u8] = &[0x06, 0x05, 0x10];
        const IDR: &[u8] = &[0x65, 0x88, 0x84];
        const SLICE: &[u8] = &[0x41, 0x9a];

        #[test]
        fn removes_parameter_sets_and_builds_avcc() {
            let mut f = FromAnnexB::new(NalCodec::H264, LengthSize::Four);
            assert!(f.config().is_none());
            assert_eq!(
                f.filter(&annexb(&[AUD, SPS, PPS, SEI, IDR])),
                Ok(sample(&[SEI, IDR]))
            );
            let Some(NalConfig::Avc(config)) = f.config() else {
                panic!("no avcC after the first access unit");
            };
            assert_eq!(config.sps, vec![SPS.to_vec()]);
            assert_eq!(config.pps, vec![PPS.to_vec()]);
            assert_eq!(config.level_indication, 30);
            // Repeated identical parameter sets are removed without error.
            assert_eq!(f.filter(&annexb(&[SPS, PPS, IDR])), Ok(sample(&[IDR])));
            assert_eq!(f.filter(&annexb(&[SLICE])), Ok(sample(&[SLICE])));
        }

        #[test]
        fn changed_sps_needs_a_new_sample_description() {
            let mut f = FromAnnexB::new(NalCodec::H264, LengthSize::Two);
            assert!(f.filter(&annexb(&[SPS, PPS, IDR])).is_ok());
            assert!(f.filter(&annexb(&[SLICE])).is_ok());
            let mut out = vec![1];
            assert_eq!(
                f.filter_into(&annexb(&[OTHER_SPS, PPS, IDR]), &mut out),
                Err(BitstreamError::ParameterSetChanged {
                    codec: "H.264",
                    kind: "SPS",
                    access_unit: 2,
                })
            );
            assert_eq!(out, vec![1]);
        }

        #[test]
        fn first_access_unit_needs_sps_and_pps() {
            let mut f = FromAnnexB::new(NalCodec::H264, LengthSize::Four);
            assert_eq!(
                f.filter(&annexb(&[SPS, IDR])),
                Err(BitstreamError::MissingParameterSet {
                    codec: "H.264",
                    kind: "PPS",
                    access_unit: 0,
                })
            );
            assert!(f.config().is_none());
            assert!(f.filter(&annexb(&[SPS, PPS, IDR])).is_ok());
            assert!(f.config().is_some());
        }

        #[test]
        fn corrupt_input_is_an_error() {
            let mut f = FromAnnexB::new(NalCodec::H264, LengthSize::Four);
            assert!(f.filter(&[0x65, 0, 0, 1, 0x65]).is_err());
            assert!(f.filter(&annexb(&[&[0x67], PPS, IDR])).is_err());
        }
    }

    #[cfg(feature = "hevc")]
    mod hevc {
        use super::*;
        use crate::bitstream::hevc_params::tests::{X265_PPS, X265_VPS};
        use crate::bitstream::hevc_sps::tests::X265_SPS;

        const AUD: &[u8] = &[0x46, 0x01, 0x50];
        const IDR: &[u8] = &[0x28, 0x01, 0xaf];

        #[test]
        fn removes_parameter_sets_and_builds_hvcc() {
            let mut f = FromAnnexB::new(NalCodec::Hevc, LengthSize::Four);
            assert_eq!(
                f.filter(&annexb(&[AUD, &X265_VPS, &X265_SPS, &X265_PPS, IDR])),
                Ok(sample(&[IDR]))
            );
            let Some(NalConfig::Hevc(config)) = f.config() else {
                panic!("no hvcC after the first access unit");
            };
            assert_eq!(config.general_level_idc, 30);
            assert!(config.arrays.iter().all(|array| array.array_completeness));
            assert_eq!(config.nal_units().count(), 3);
        }

        #[test]
        fn first_access_unit_needs_a_vps() {
            let mut f = FromAnnexB::new(NalCodec::Hevc, LengthSize::Four);
            assert_eq!(
                f.filter(&annexb(&[&X265_SPS, &X265_PPS, IDR])),
                Err(BitstreamError::MissingParameterSet {
                    codec: "HEVC",
                    kind: "VPS",
                    access_unit: 0,
                })
            );
        }
    }
}
