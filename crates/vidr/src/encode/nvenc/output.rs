//! Encoder output to MP4 samples and a config record, through the bitstream
//! filters.

use crate::bitstream::av1::Av1FromEncoder;
use crate::bitstream::{FromAnnexB, LengthSize, NalCodec};
use crate::encode::EncodeCodec;
use crate::error::EncodeError;

/// One track's filter from NVENC output to samples.
#[derive(Clone, Debug)]
pub(super) enum SampleFilter {
    /// H.264 and HEVC: Annex-B to length-prefixed NAL units.
    Nal(FromAnnexB),
    /// AV1: encoder temporal units to samples.
    Av1(Av1FromEncoder),
}

impl SampleFilter {
    pub(super) const fn new(codec: EncodeCodec) -> Self {
        match codec {
            EncodeCodec::H264 => Self::Nal(FromAnnexB::new(NalCodec::H264, LengthSize::Four)),
            EncodeCodec::Hevc => Self::Nal(FromAnnexB::new(NalCodec::Hevc, LengthSize::Four)),
            EncodeCodec::Av1 => Self::Av1(Av1FromEncoder::new()),
        }
    }

    /// Converts one picture's output to a sample.
    pub(super) fn sample(&mut self, output: &[u8]) -> Result<Vec<u8>, EncodeError> {
        match self {
            Self::Nal(filter) => filter.filter(output),
            Self::Av1(filter) => filter.filter(output),
        }
        .map_err(EncodeError::from)
    }

    /// The config record's bytes, once the first picture has passed.
    pub(super) fn config_record(&self) -> Result<Option<Vec<u8>>, EncodeError> {
        let bytes = match self {
            Self::Nal(filter) => filter.config().map(|config| config.to_bytes()),
            Self::Av1(filter) => filter.config().map(|config| config.to_bytes()),
        };
        bytes.transpose().map_err(EncodeError::from)
    }
}
