//! AV1 encoder temporal units to MP4 and Matroska samples and `av1C`.
//!
//! - Temporal delimiter, redundant frame header, tile list, and padding OBUs
//!   leave the samples. The AV1 ISO-BMFF binding forbids tile lists and
//!   advises against the others. ffmpeg's MP4 muxer drops the same four.
//! - Every other OBU stays, sequence headers included, with
//!   `obu_has_size_field` 1.
//! - `av1C` comes from the first temporal unit's sequence header.
//! - A later sequence header must carry the same payload. A different one
//!   returns [`BitstreamError::ParameterSetChanged`]: the mux needs a new
//!   sample description, and vidr does not write one.

use super::av1c::Av1Config;
use super::obu::{self, ObuType};
use crate::error::BitstreamError;

const CODEC: &str = "AV1";
const SEQUENCE_HEADER: &str = "sequence header";

/// A per-track filter from encoder temporal units to AV1 samples.
///
/// Pass one temporal unit per call, in decode order.
#[derive(Clone, Debug)]
pub struct Av1FromEncoder {
    config: Option<Av1Config>,
    /// The payload of the `av1C` sequence header, for the change check.
    sequence_payload: Vec<u8>,
    temporal_unit: u64,
}

impl Default for Av1FromEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Av1FromEncoder {
    /// Creates a filter.
    pub const fn new() -> Self {
        Self {
            config: None,
            sequence_payload: Vec::new(),
            temporal_unit: 0,
        }
    }

    /// The config record, after the first temporal unit.
    pub const fn config(&self) -> Option<&Av1Config> {
        self.config.as_ref()
    }

    /// Converts one temporal unit to a sample.
    ///
    /// # Errors
    ///
    /// As [`Av1FromEncoder::filter_into`].
    pub fn filter(&mut self, temporal_unit: &[u8]) -> Result<Vec<u8>, BitstreamError> {
        let mut out = Vec::with_capacity(temporal_unit.len());
        self.filter_into(temporal_unit, &mut out)?;
        Ok(out)
    }

    /// Converts one temporal unit and appends the sample to `out`.
    ///
    /// # Errors
    ///
    /// - [`BitstreamError`] for an OBU or sequence header that does not
    ///   parse.
    /// - [`BitstreamError::MissingParameterSet`] when the first temporal unit
    ///   has no sequence header.
    /// - [`BitstreamError::ParameterSetChanged`] for a sequence header that
    ///   differs from the one in `av1C`.
    ///
    /// On an error `out` and the filter state are unchanged, and the next
    /// call counts the next temporal unit.
    pub fn filter_into(
        &mut self,
        temporal_unit: &[u8],
        out: &mut Vec<u8>,
    ) -> Result<(), BitstreamError> {
        let index = self.temporal_unit;
        self.temporal_unit += 1;
        let obus = obu::split(temporal_unit).collect::<Result<Vec<_>, _>>()?;

        let mut built = None;
        for parsed in obus
            .iter()
            .filter(|parsed| parsed.header.obu_type == ObuType::SEQUENCE_HEADER)
        {
            if self.config.is_none() && built.is_none() {
                built = Some((
                    Av1Config::from_sequence_header(parsed.bytes)?,
                    parsed.payload,
                ));
                continue;
            }
            let reference: &[u8] = match &built {
                Some((_, payload)) => payload,
                None => self.sequence_payload.as_slice(),
            };
            if parsed.payload != reference {
                return Err(BitstreamError::ParameterSetChanged {
                    codec: CODEC,
                    kind: SEQUENCE_HEADER,
                    access_unit: index,
                });
            }
        }
        if self.config.is_none() && built.is_none() {
            return Err(BitstreamError::MissingParameterSet {
                codec: CODEC,
                kind: SEQUENCE_HEADER,
                access_unit: index,
            });
        }

        for parsed in &obus {
            let dropped = matches!(
                parsed.header.obu_type,
                ObuType::TEMPORAL_DELIMITER
                    | ObuType::REDUNDANT_FRAME_HEADER
                    | ObuType::TILE_LIST
                    | ObuType::PADDING
            );
            if !dropped {
                parsed.write_sized(out);
            }
        }
        if let Some((config, payload)) = built {
            self.sequence_payload = payload.to_vec();
            self.config = Some(config);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::av1::sequence::tests::LIBAOM_PAYLOAD;

    const DELIMITER: [u8; 2] = [0x12, 0x00];
    const FRAME: [u8; 4] = [0x32, 0x02, 0xaa, 0xbb];
    const PADDING: [u8; 3] = [0x7a, 0x01, 0x00];

    fn header_obu() -> Vec<u8> {
        let mut obu = vec![0x0a, LIBAOM_PAYLOAD.len() as u8];
        obu.extend_from_slice(&LIBAOM_PAYLOAD);
        obu
    }

    fn unit(parts: &[&[u8]]) -> Vec<u8> {
        parts.concat()
    }

    #[test]
    fn drops_delimiters_and_builds_av1c() {
        let mut f = Av1FromEncoder::new();
        let header = header_obu();
        assert_eq!(
            f.filter(&unit(&[&DELIMITER, &header, &FRAME, &PADDING])),
            Ok(unit(&[&header, &FRAME]))
        );
        let config = f.config().map(Av1Config::to_bytes);
        assert_eq!(
            config,
            Some(Ok(unit(&[&[0x81, 0x00, 0x0c, 0x00], &header])))
        );
        assert_eq!(f.filter(&unit(&[&DELIMITER, &FRAME])), Ok(FRAME.to_vec()));
        // A repeated identical header stays in the sample.
        assert_eq!(
            f.filter(&unit(&[&DELIMITER, &header, &FRAME])),
            Ok(unit(&[&header, &FRAME]))
        );
    }

    #[test]
    fn first_unit_needs_a_sequence_header() {
        let mut f = Av1FromEncoder::new();
        assert_eq!(
            f.filter(&unit(&[&DELIMITER, &FRAME])),
            Err(BitstreamError::MissingParameterSet {
                codec: "AV1",
                kind: "sequence header",
                access_unit: 0,
            })
        );
        assert!(f.config().is_none());
    }

    #[test]
    fn changed_sequence_header_needs_a_new_sample_description() {
        let mut f = Av1FromEncoder::new();
        assert!(f.filter(&unit(&[&header_obu(), &FRAME])).is_ok());
        let mut changed = header_obu();
        changed[3] ^= 0x01;
        let mut out = vec![5];
        assert_eq!(
            f.filter_into(&unit(&[&changed, &FRAME]), &mut out),
            Err(BitstreamError::ParameterSetChanged {
                codec: "AV1",
                kind: "sequence header",
                access_unit: 1,
            })
        );
        assert_eq!(out, vec![5]);
    }

    #[test]
    fn unsized_obus_gain_a_size() {
        let mut f = Av1FromEncoder::new();
        assert!(f.filter(&unit(&[&header_obu(), &FRAME])).is_ok());
        assert_eq!(
            f.filter(&[0x12, 0x00, 0x30, 0xaa, 0xbb]),
            Ok(FRAME.to_vec())
        );
        assert!(f.filter(&[0x32, 0x09]).is_err());
    }
}
