//! `AV1CodecConfigurationRecord`, the `av1C` box payload (AV1 Codec ISO
//! Media File Format Binding 2.3).
//!
//! The 4-byte header copies fields of the sequence header. `configOBUs`
//! holds at most one sequence header OBU, first, and metadata OBUs. Readers
//! ignore the reserved bits. Writers set them to zero, as the binding
//! requires.

use super::obu::{self, ObuType};
use super::sequence::{self, SequenceFields};
use crate::bitstream::bytes::ByteReader;
use crate::error::BitstreamError;

const STRUCTURE: &str = "av1C";

/// An `AV1CodecConfigurationRecord`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Av1Config {
    /// `seq_profile`, 0 to 2.
    pub seq_profile: u8,
    /// `seq_level_idx_0`, 0 to 31.
    pub seq_level_idx_0: u8,
    /// `seq_tier_0`.
    pub seq_tier_0: bool,
    /// `high_bitdepth`.
    pub high_bitdepth: bool,
    /// `twelve_bit`.
    pub twelve_bit: bool,
    /// `monochrome`.
    pub monochrome: bool,
    /// `chroma_subsampling_x`.
    pub chroma_subsampling_x: bool,
    /// `chroma_subsampling_y`.
    pub chroma_subsampling_y: bool,
    /// `chroma_sample_position`, 0 to 3.
    pub chroma_sample_position: u8,
    /// `initial_presentation_delay_minus_one`, 0 to 15, when present.
    pub initial_presentation_delay_minus_one: Option<u8>,
    /// `configOBUs`: low-overhead OBUs.
    pub config_obus: Vec<u8>,
}

fn invalid(field: &'static str, value: u64, allowed: &'static str) -> BitstreamError {
    BitstreamError::InvalidField {
        structure: STRUCTURE,
        field,
        value,
        allowed,
    }
}

impl Av1Config {
    /// Parses an `av1C` box payload.
    ///
    /// # Errors
    ///
    /// [`BitstreamError`] for a short record, a `marker` or `version` other
    /// than 1, `configOBUs` that do not parse or hold OBUs other than one
    /// leading sequence header and metadata, or a header field that differs
    /// from the sequence header's.
    pub fn parse(bytes: &[u8]) -> Result<Self, BitstreamError> {
        let mut reader = ByteReader::new(bytes, STRUCTURE);
        let first = reader.u8("version")?;
        if first & 0x80 == 0 {
            return Err(invalid("marker", 0, "1"));
        }
        if first & 0x7f != 1 {
            return Err(invalid("version", (first & 0x7f).into(), "1"));
        }
        let profile = reader.u8("seq_level_idx_0")?;
        let flags = reader.u8("chroma_sample_position")?;
        let delay = reader.u8("initial_presentation_delay_minus_one")?;
        let config = Self {
            seq_profile: profile >> 5,
            seq_level_idx_0: profile & 0x1f,
            seq_tier_0: flags & 0x80 != 0,
            high_bitdepth: flags & 0x40 != 0,
            twelve_bit: flags & 0x20 != 0,
            monochrome: flags & 0x10 != 0,
            chroma_subsampling_x: flags & 0x08 != 0,
            chroma_subsampling_y: flags & 0x04 != 0,
            chroma_sample_position: flags & 0x03,
            initial_presentation_delay_minus_one: (delay & 0x10 != 0).then_some(delay & 0x0f),
            config_obus: reader.rest().to_vec(),
        };
        config.check()?;
        Ok(config)
    }

    /// Builds a record from a sequence header OBU, as ffmpeg's
    /// `ff_isom_write_av1c` does.
    ///
    /// `configOBUs` holds the OBU with its size field set.
    /// `initial_presentation_delay_present` is 0.
    ///
    /// # Errors
    ///
    /// [`BitstreamError::InvalidField`] when `obu` is not exactly one
    /// sequence header OBU, and errors from the sequence header parse.
    pub fn from_sequence_header(obu: &[u8]) -> Result<Self, BitstreamError> {
        let mut obus = obu::split(obu);
        let header = match (obus.next(), obus.next()) {
            (Some(header), None) => header?,
            (None, _) => return Err(invalid("configOBUs", 0, "one sequence header OBU")),
            (Some(_), Some(_)) => return Err(invalid("configOBUs", 2, "one sequence header OBU")),
        };
        if header.header.obu_type != ObuType::SEQUENCE_HEADER {
            return Err(invalid(
                "obu_type",
                header.header.obu_type.value().into(),
                "1 (sequence header)",
            ));
        }
        let fields = sequence::parse(header.payload)?;
        let mut config_obus = Vec::with_capacity(obu.len() + 1);
        header.write_sized(&mut config_obus);
        Ok(Self::from_fields(&fields, config_obus))
    }

    fn from_fields(fields: &SequenceFields, config_obus: Vec<u8>) -> Self {
        Self {
            seq_profile: fields.seq_profile,
            seq_level_idx_0: fields.seq_level_idx_0,
            seq_tier_0: fields.seq_tier_0,
            high_bitdepth: fields.high_bitdepth,
            twelve_bit: fields.twelve_bit,
            monochrome: fields.mono_chrome,
            chroma_subsampling_x: fields.subsampling_x,
            chroma_subsampling_y: fields.subsampling_y,
            chroma_sample_position: fields.chroma_sample_position,
            initial_presentation_delay_minus_one: None,
            config_obus,
        }
    }

    /// The sequence header OBU of `configOBUs`, as stored.
    ///
    /// # Errors
    ///
    /// Errors from the OBU parse.
    pub fn sequence_header(&self) -> Result<Option<obu::Obu<'_>>, BitstreamError> {
        match obu::split(&self.config_obus).next() {
            Some(Ok(first)) if first.header.obu_type == ObuType::SEQUENCE_HEADER => Ok(Some(first)),
            Some(Err(error)) => Err(error),
            _ => Ok(None),
        }
    }

    /// Writes the record as an `av1C` box payload.
    ///
    /// # Errors
    ///
    /// [`BitstreamError::InvalidField`] for a field out of range, or the
    /// errors of [`Av1Config::parse`] for `configOBUs`.
    pub fn to_bytes(&self) -> Result<Vec<u8>, BitstreamError> {
        self.check()?;
        let mut out = Vec::with_capacity(4 + self.config_obus.len());
        out.push(0x81);
        out.push(self.seq_profile << 5 | self.seq_level_idx_0);
        out.push(
            u8::from(self.seq_tier_0) << 7
                | u8::from(self.high_bitdepth) << 6
                | u8::from(self.twelve_bit) << 5
                | u8::from(self.monochrome) << 4
                | u8::from(self.chroma_subsampling_x) << 3
                | u8::from(self.chroma_subsampling_y) << 2
                | self.chroma_sample_position,
        );
        out.push(match self.initial_presentation_delay_minus_one {
            Some(delay) => 0x10 | delay,
            None => 0,
        });
        out.extend_from_slice(&self.config_obus);
        Ok(out)
    }

    fn check(&self) -> Result<(), BitstreamError> {
        if self.seq_profile > 2 {
            return Err(invalid("seq_profile", self.seq_profile.into(), "0 to 2"));
        }
        if self.seq_level_idx_0 > 31 {
            return Err(invalid(
                "seq_level_idx_0",
                self.seq_level_idx_0.into(),
                "0 to 31",
            ));
        }
        if self.chroma_sample_position > 3 {
            return Err(invalid(
                "chroma_sample_position",
                self.chroma_sample_position.into(),
                "0 to 3",
            ));
        }
        if let Some(delay) = self.initial_presentation_delay_minus_one
            && delay > 15
        {
            return Err(invalid(
                "initial_presentation_delay_minus_one",
                delay.into(),
                "0 to 15",
            ));
        }
        for (index, parsed) in obu::split(&self.config_obus).enumerate() {
            let parsed = parsed?;
            match parsed.header.obu_type {
                ObuType::SEQUENCE_HEADER if index == 0 => {
                    self.check_matches(&sequence::parse(parsed.payload)?)?
                }
                ObuType::METADATA => {}
                other => {
                    return Err(invalid(
                        "configOBUs",
                        other.value().into(),
                        "one leading sequence header OBU and metadata OBUs",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Checks the header fields against the sequence header's.
    fn check_matches(&self, fields: &SequenceFields) -> Result<(), BitstreamError> {
        let pairs: [(&'static str, u8, u8); 9] = [
            ("seq_profile", self.seq_profile, fields.seq_profile),
            (
                "seq_level_idx_0",
                self.seq_level_idx_0,
                fields.seq_level_idx_0,
            ),
            (
                "seq_tier_0",
                self.seq_tier_0.into(),
                fields.seq_tier_0.into(),
            ),
            (
                "high_bitdepth",
                self.high_bitdepth.into(),
                fields.high_bitdepth.into(),
            ),
            (
                "twelve_bit",
                self.twelve_bit.into(),
                fields.twelve_bit.into(),
            ),
            (
                "monochrome",
                self.monochrome.into(),
                fields.mono_chrome.into(),
            ),
            (
                "chroma_subsampling_x",
                self.chroma_subsampling_x.into(),
                fields.subsampling_x.into(),
            ),
            (
                "chroma_subsampling_y",
                self.chroma_subsampling_y.into(),
                fields.subsampling_y.into(),
            ),
            (
                "chroma_sample_position",
                self.chroma_sample_position,
                fields.chroma_sample_position,
            ),
        ];
        for (field, record, header) in pairs {
            if record != header {
                return Err(invalid(
                    field,
                    record.into(),
                    "the value in the sequence header OBU",
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::av1::sequence::tests::LIBAOM_PAYLOAD;

    fn libaom_obu() -> Vec<u8> {
        let mut obu = vec![0x0a, LIBAOM_PAYLOAD.len() as u8];
        obu.extend_from_slice(&LIBAOM_PAYLOAD);
        obu
    }

    fn libaom_record() -> Vec<u8> {
        let mut bytes = vec![0x81, 0x00, 0x0c, 0x00];
        bytes.extend_from_slice(&libaom_obu());
        bytes
    }

    #[test]
    fn parses_and_writes_the_libaom_record() {
        let config = Av1Config::parse(&libaom_record());
        assert_eq!(
            config
                .as_ref()
                .map(|c| (c.chroma_subsampling_x, c.chroma_subsampling_y)),
            Ok((true, true))
        );
        assert_eq!(
            config.as_ref().map(Av1Config::to_bytes),
            Ok(Ok(libaom_record()))
        );
        let header = config
            .as_ref()
            .map(|c| c.sequence_header().map(|h| h.map(|h| h.bytes.to_vec())));
        assert_eq!(header, Ok(Ok(Some(libaom_obu()))));
    }

    #[test]
    fn built_record_matches_ffmpeg() {
        assert_eq!(
            Av1Config::from_sequence_header(&libaom_obu()).and_then(|c| c.to_bytes()),
            Ok(libaom_record())
        );
        // A header without a size field gains one.
        let mut bare = vec![0x08];
        bare.extend_from_slice(&LIBAOM_PAYLOAD);
        assert_eq!(
            Av1Config::from_sequence_header(&bare).and_then(|c| c.to_bytes()),
            Ok(libaom_record())
        );
    }

    #[test]
    fn mismatched_header_field_is_an_error() {
        let mut bytes = libaom_record();
        bytes[2] = 0x08;
        assert!(matches!(
            Av1Config::parse(&bytes),
            Err(BitstreamError::InvalidField {
                field: "chroma_subsampling_y",
                ..
            })
        ));
    }

    #[test]
    fn marker_version_and_obu_types_are_checked() {
        let mut bytes = libaom_record();
        bytes[0] = 0x01;
        assert!(matches!(
            Av1Config::parse(&bytes),
            Err(BitstreamError::InvalidField {
                field: "marker",
                ..
            })
        ));
        bytes[0] = 0x82;
        assert!(matches!(
            Av1Config::parse(&bytes),
            Err(BitstreamError::InvalidField {
                field: "version",
                ..
            })
        ));
        let mut bytes = libaom_record();
        bytes.extend_from_slice(&[0x12, 0x00]);
        assert!(matches!(
            Av1Config::parse(&bytes),
            Err(BitstreamError::InvalidField {
                field: "configOBUs",
                ..
            })
        ));
        assert!(Av1Config::from_sequence_header(&[0x12, 0x00]).is_err());
        assert!(Av1Config::from_sequence_header(&[]).is_err());
    }

    #[test]
    fn record_without_obus_is_valid() {
        let config = Av1Config::parse(&[0x81, 0x00, 0x0c, 0x00]);
        assert_eq!(config.as_ref().map(|c| c.config_obus.is_empty()), Ok(true));
        assert_eq!(
            config.and_then(|c| c.sequence_header().map(|h| h.is_none())),
            Ok(true)
        );
    }

    #[test]
    fn every_truncation_is_an_error_or_shorter_record() {
        let bytes = libaom_record();
        for end in 0..bytes.len() {
            let parsed = Av1Config::parse(&bytes[..end]);
            // 4 bytes alone is a valid record without configOBUs.
            assert_eq!(parsed.is_ok(), end == 4, "prefix of {end} bytes");
        }
    }
}
