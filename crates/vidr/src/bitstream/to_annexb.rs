//! MP4 and Matroska samples to Annex-B access units, the input NVDEC's
//! H.264 and HEVC parsers take.
//!
//! The output matches ffmpeg's `h264_mp4toannexb` and `hevc_mp4toannexb`
//! byte for byte, with one documented difference for HEVC.
//!
//! H.264, per sample:
//!
//! - The config's SPS and PPS go before the first IDR slice of an IDR
//!   picture that follows a non-IDR slice, when the sample carries no SPS or
//!   PPS before it. The config's PPS alone goes when the sample carries an
//!   SPS but no PPS.
//! - The config's SPS goes before an in-band PPS that no SPS precedes.
//! - The config's SPS and PPS go before a buffering period SEI that no
//!   parameter set precedes.
//! - In-band SPS and PPS replace the config's for later insertions.
//! - The first NAL unit of a sample and every in-band SPS or PPS get a
//!   4-byte start code. Other units get a 3-byte start code. Inserted
//!   parameter sets get 4-byte start codes.
//!
//! HEVC, per sample:
//!
//! - Every NAL unit gets a 4-byte start code.
//! - Every NAL unit of the config, SEI arrays included, goes once before
//!   the first IRAP NAL unit. When parameter sets precede that IRAP in the
//!   sample, the config goes before the first of them instead.
//! - Difference from ffmpeg: when a VPS, an SPS, and a PPS all precede the
//!   first IRAP in the sample, vidr inserts nothing. ffmpeg inserts the
//!   config before them, so the stream carries each parameter set twice.

#[cfg(feature = "hevc")]
use super::annexb;
#[cfg(feature = "h264")]
use super::avcc::AvcConfig;
#[cfg(feature = "hevc")]
use super::hvcc::HevcConfig;
use super::length_prefixed::{self, LengthSize};
#[cfg(feature = "h264")]
use super::to_annexb_h264::H264State;
#[cfg(feature = "hevc")]
use super::to_annexb_hevc;
use crate::error::BitstreamError;

/// A per-track filter from length-prefixed samples to Annex-B access units.
///
/// Create one filter per track and sample description, and pass the samples
/// in decode order.
#[derive(Clone, Debug)]
pub struct ToAnnexB {
    length_size: LengthSize,
    codec: Codec,
}

#[derive(Clone, Debug)]
enum Codec {
    #[cfg(feature = "h264")]
    H264(H264State),
    #[cfg(feature = "hevc")]
    Hevc {
        /// Every config NAL unit after a 4-byte start code.
        config: Vec<u8>,
    },
}

impl ToAnnexB {
    /// Creates an H.264 filter from the track's `avcC`.
    #[cfg(feature = "h264")]
    pub fn h264(config: &AvcConfig) -> Self {
        Self {
            length_size: config.length_size,
            codec: Codec::H264(H264State::new(config)),
        }
    }

    /// Creates an HEVC filter from the track's `hvcC`.
    #[cfg(feature = "hevc")]
    pub fn hevc(config: &HevcConfig) -> Self {
        Self {
            length_size: config.length_size,
            codec: Codec::Hevc {
                config: annexb::framed(config.nal_units()),
            },
        }
    }

    /// Converts one sample to an Annex-B access unit.
    ///
    /// # Errors
    ///
    /// As [`ToAnnexB::filter_into`].
    pub fn filter(&mut self, sample: &[u8]) -> Result<Vec<u8>, BitstreamError> {
        let mut out = Vec::with_capacity(sample.len() + 64);
        self.filter_into(sample, &mut out)?;
        Ok(out)
    }

    /// Converts one sample and appends the access unit to `out`.
    ///
    /// # Errors
    ///
    /// [`BitstreamError::Truncated`] or [`BitstreamError::NalUnitOverrun`]
    /// for a length field that does not fit the sample. For HEVC,
    /// [`BitstreamError::InvalidField`] for a NAL unit shorter than its
    /// 2-byte header. On an error `out` and the filter state are unchanged.
    pub fn filter_into(&mut self, sample: &[u8], out: &mut Vec<u8>) -> Result<(), BitstreamError> {
        let units =
            length_prefixed::split(sample, self.length_size).collect::<Result<Vec<_>, _>>()?;
        match &mut self.codec {
            #[cfg(feature = "h264")]
            Codec::H264(state) => {
                state.filter(&units, out);
                Ok(())
            }
            #[cfg(feature = "hevc")]
            Codec::Hevc { config } => to_annexb_hevc::filter(config, &units, out),
        }
    }
}
