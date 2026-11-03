//! MP4 and Matroska AV1 samples to temporal units for NVDEC's AV1 parser.
//!
//! Each output temporal unit starts with a temporal delimiter OBU, and every
//! OBU has `obu_has_size_field` 1: the low-overhead bitstream format of AV1
//! specification section 5.2. The first output temporal unit carries the
//! sequence header from `av1C` when its sample has none. The output matches
//! ffmpeg's `obu` muxer, which runs `av1_metadata=td=insert`, byte for byte.

use super::av1c::Av1Config;
use super::obu::{self, ObuType};
use crate::error::BitstreamError;

/// A temporal delimiter OBU with an empty payload.
pub(crate) const TEMPORAL_DELIMITER: [u8; 2] = [0x12, 0x00];

/// A per-track filter from AV1 samples to decoder temporal units.
#[derive(Clone, Debug)]
pub struct Av1ToDecoder {
    /// The `av1C` sequence header OBU with its size field set.
    sequence_header: Option<Vec<u8>>,
    first: bool,
}

impl Av1ToDecoder {
    /// Creates a filter from the track's `av1C`.
    ///
    /// # Errors
    ///
    /// Errors from the `configOBUs` parse.
    pub fn new(config: &Av1Config) -> Result<Self, BitstreamError> {
        let sequence_header = config.sequence_header()?.map(|header| {
            let mut bytes = Vec::with_capacity(header.bytes.len() + 1);
            header.write_sized(&mut bytes);
            bytes
        });
        Ok(Self {
            sequence_header,
            first: true,
        })
    }

    /// Converts one sample to a temporal unit.
    ///
    /// # Errors
    ///
    /// As [`Av1ToDecoder::filter_into`].
    pub fn filter(&mut self, sample: &[u8]) -> Result<Vec<u8>, BitstreamError> {
        let mut out = Vec::with_capacity(sample.len() + 16);
        self.filter_into(sample, &mut out)?;
        Ok(out)
    }

    /// Converts one sample and appends the temporal unit to `out`.
    ///
    /// # Errors
    ///
    /// [`BitstreamError`] for an OBU that does not parse. On an error `out`
    /// and the filter state are unchanged.
    pub fn filter_into(&mut self, sample: &[u8], out: &mut Vec<u8>) -> Result<(), BitstreamError> {
        let obus = obu::split(sample).collect::<Result<Vec<_>, _>>()?;
        let rest = match obus.split_first() {
            Some((delimiter, tail)) if delimiter.header.obu_type == ObuType::TEMPORAL_DELIMITER => {
                delimiter.write_sized(out);
                tail
            }
            _ => {
                out.extend_from_slice(&TEMPORAL_DELIMITER);
                &obus[..]
            }
        };
        let has_header = obus
            .iter()
            .any(|parsed| parsed.header.obu_type == ObuType::SEQUENCE_HEADER);
        if self.first
            && !has_header
            && let Some(header) = &self.sequence_header
        {
            out.extend_from_slice(header);
        }
        for parsed in rest {
            parsed.write_sized(out);
        }
        self.first = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::av1::sequence::tests::LIBAOM_PAYLOAD;

    fn config() -> Av1Config {
        match Av1Config::from_sequence_header(&header_obu()) {
            Ok(config) => config,
            Err(error) => panic!("config failed: {error}"),
        }
    }

    fn header_obu() -> Vec<u8> {
        let mut obu = vec![0x0a, LIBAOM_PAYLOAD.len() as u8];
        obu.extend_from_slice(&LIBAOM_PAYLOAD);
        obu
    }

    const FRAME: [u8; 4] = [0x32, 0x02, 0xaa, 0xbb];

    fn filter() -> Av1ToDecoder {
        match Av1ToDecoder::new(&config()) {
            Ok(filter) => filter,
            Err(error) => panic!("filter failed: {error}"),
        }
    }

    #[test]
    fn first_sample_without_header_gets_the_config_header() {
        let mut f = filter();
        let mut expected = TEMPORAL_DELIMITER.to_vec();
        expected.extend_from_slice(&header_obu());
        expected.extend_from_slice(&FRAME);
        assert_eq!(f.filter(&FRAME), Ok(expected));
        let mut second = TEMPORAL_DELIMITER.to_vec();
        second.extend_from_slice(&FRAME);
        assert_eq!(f.filter(&FRAME), Ok(second));
    }

    #[test]
    fn first_sample_with_header_is_kept() {
        let mut f = filter();
        let mut sample = header_obu();
        sample.extend_from_slice(&FRAME);
        let mut expected = TEMPORAL_DELIMITER.to_vec();
        expected.extend_from_slice(&sample);
        assert_eq!(f.filter(&sample), Ok(expected));
    }

    #[test]
    fn unsized_last_obu_gains_a_size_and_delimiters_are_not_doubled() {
        let mut f = filter();
        assert!(f.filter(&FRAME).is_ok());
        let sample = [0x12, 0x00, 0x30, 0xaa, 0xbb];
        let mut expected = TEMPORAL_DELIMITER.to_vec();
        expected.extend_from_slice(&FRAME);
        assert_eq!(f.filter(&sample), Ok(expected));
    }

    #[test]
    fn corrupt_sample_keeps_state() {
        let mut f = filter();
        let mut out = vec![1];
        assert!(f.filter_into(&[0x32, 0x09, 0xaa], &mut out).is_err());
        assert_eq!(out, vec![1]);
        // Still the first sample: the header goes in.
        assert_eq!(f.filter(&FRAME).map(|tu| tu.len()), Ok(2 + 12 + 4));
    }
}
