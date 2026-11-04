//! Container samples to the elementary stream NVDEC's parser takes.

use super::signal;
use crate::bitstream::av1::{Av1Config, Av1ToDecoder};
use crate::bitstream::{AvcConfig, HevcConfig, HevcNalType, LengthSize, ToAnnexB};
use crate::bitstream::{Vp9Color, vp9_packet_color};
use crate::bitstream::{annexb, length_prefixed};
use crate::decode::{ChromaLocation, DecodeCodec};
use crate::error::DecodeError;

/// The per-track filter in front of the parser.
#[derive(Debug)]
pub(super) enum Input {
    /// Packets are already Annex-B access units, AV1 temporal units, or VP9
    /// frames.
    Raw,
    /// H.264 or HEVC length-prefixed samples.
    AnnexB {
        filter: ToAnnexB,
        length_size: LengthSize,
    },
    /// AV1 samples.
    Av1(Av1ToDecoder),
}

impl Input {
    /// The filter for `codec` and its config record.
    ///
    /// An empty `config` selects raw input. VP9 always takes raw input:
    /// `vpcC` carries nothing the parser needs, and the colour comes from
    /// the frame headers (see `Input::vp9_color`). For AV1 the result also
    /// holds the `av1C` chroma sample position, and for H.264 and HEVC the
    /// chroma sample location of the config's first SPS.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Bitstream`] for a config record that does not parse.
    pub(super) fn new(
        codec: DecodeCodec,
        config: &[u8],
    ) -> Result<(Self, Option<ChromaLocation>), DecodeError> {
        if config.is_empty() {
            return Ok((Self::Raw, None));
        }
        Ok(match codec {
            DecodeCodec::H264 => {
                let parsed = AvcConfig::parse(config)?;
                let location =
                    signal::first_sps_location(codec, parsed.sps.iter().map(Vec::as_slice))
                        .flatten();
                let input = Self::AnnexB {
                    filter: ToAnnexB::h264(&parsed),
                    length_size: parsed.length_size,
                };
                (input, location)
            }
            DecodeCodec::Hevc => {
                let parsed = HevcConfig::parse(config)?;
                let sps = parsed
                    .arrays
                    .iter()
                    .filter(|array| array.nal_unit_type == HevcNalType::SPS)
                    .flat_map(|array| array.nal_units.iter().map(Vec::as_slice));
                let location = signal::first_sps_location(codec, sps).flatten();
                let input = Self::AnnexB {
                    filter: ToAnnexB::hevc(&parsed),
                    length_size: parsed.length_size,
                };
                (input, location)
            }
            DecodeCodec::Av1 => {
                let parsed = Av1Config::parse(config)?;
                let location = ChromaLocation::from_av1(parsed.chroma_sample_position);
                (Self::Av1(Av1ToDecoder::new(&parsed)?), location)
            }
            DecodeCodec::Vp9 => (Self::Raw, None),
        })
    }

    /// The chroma sample location of the last SPS in `packet`, read before
    /// the filter runs. The outer `None` means the packet holds no SPS, so
    /// the location in effect stays. The inner `None` means the SPS signals
    /// no location. AV1 and VP9 packets give the outer `None`.
    pub(super) fn sps_location(
        &self,
        codec: DecodeCodec,
        packet: &[u8],
    ) -> Option<Option<ChromaLocation>> {
        if !matches!(codec, DecodeCodec::H264 | DecodeCodec::Hevc) {
            return None;
        }
        match self {
            Self::Raw => {
                let units = annexb::split(packet).ok()?;
                signal::last_sps_location(codec, units)
            }
            Self::AnnexB { length_size, .. } => signal::last_sps_location(
                codec,
                length_prefixed::split(packet, *length_size).map_while(Result::ok),
            ),
            Self::Av1(_) => None,
        }
    }

    /// The colour a VP9 packet states, read from the header of every frame
    /// in it before the parser runs. `None` means the packet holds no key
    /// frame or intra-only header with colour, so the colour in effect
    /// stays. Other codecs, and a header that does not parse, give `None`.
    pub(super) fn vp9_color(&self, codec: DecodeCodec, packet: &[u8]) -> Option<Vp9Color> {
        if codec != DecodeCodec::Vp9 || !matches!(self, Self::Raw) {
            return None;
        }
        vp9_packet_color(packet).ok().flatten()
    }

    /// Converts one packet into `out`, which is cleared first. Raw input
    /// copies nothing and returns the packet itself.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Bitstream`] for a sample the filter rejects.
    pub(super) fn filter<'a>(
        &mut self,
        packet: &'a [u8],
        out: &'a mut Vec<u8>,
    ) -> Result<&'a [u8], DecodeError> {
        out.clear();
        match self {
            Self::Raw => return Ok(packet),
            Self::AnnexB { filter, .. } => filter.filter_into(packet, out)?,
            Self::Av1(filter) => filter.filter_into(packet, out)?,
        }
        Ok(out.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_config_and_vp9_take_raw_packets() {
        let mut out = Vec::new();
        for codec in [DecodeCodec::H264, DecodeCodec::Vp9] {
            let config: &[u8] = if codec == DecodeCodec::Vp9 {
                &[1, 0, 0]
            } else {
                &[]
            };
            let Ok((mut input, location)) = Input::new(codec, config) else {
                panic!("{codec} input failed");
            };
            assert!(matches!(input, Input::Raw));
            assert_eq!(location, None);
            assert_eq!(input.filter(&[0, 0, 1, 9], &mut out), Ok(&[0, 0, 1, 9][..]));
        }
    }

    #[test]
    fn vp9_packets_report_the_key_frame_colour() {
        // Key frame, profile 0, sync code, color_space 2 (BT.709), limited.
        let key = [0x82, 0x49, 0x83, 0x42, 0x40, 0, 0, 0];
        let Ok((input, _)) = Input::new(DecodeCodec::Vp9, &[1, 0, 0]) else {
            panic!("VP9 input failed");
        };
        let expected = Vp9Color {
            color_space: 2,
            full_range: false,
            bit_depth: 8,
        };
        assert_eq!(input.vp9_color(DecodeCodec::Vp9, &key), Some(expected));
        // An inter frame, a header that does not parse, and other codecs
        // give no colour.
        assert_eq!(input.vp9_color(DecodeCodec::Vp9, &[0x86, 0, 0]), None);
        assert_eq!(input.vp9_color(DecodeCodec::Vp9, &key[..3]), None);
        assert_eq!(input.vp9_color(DecodeCodec::Av1, &key), None);
    }

    /// An H.264 Main profile SPS with chroma sample location type 1 in both
    /// fields and a stop bit. ffmpeg's `trace_headers` reads the same
    /// fields from it.
    const CENTER_SPS: [u8; 9] = [103, 77, 0, 30, 218, 5, 58, 41, 64];
    const PPS: [u8; 3] = [0x68, 0xce, 0x38];

    #[test]
    fn annexb_packets_report_the_in_band_sps_location() {
        let mut packet = vec![0, 0, 0, 1];
        packet.extend(CENTER_SPS);
        packet.extend([0, 0, 0, 1]);
        packet.extend(PPS);
        let Ok((input, _)) = Input::new(DecodeCodec::H264, &[]) else {
            panic!("raw input failed");
        };
        assert_eq!(
            input.sps_location(DecodeCodec::H264, &packet),
            Some(Some(ChromaLocation::Center))
        );
        // No SPS: the location in effect stays.
        assert_eq!(
            input.sps_location(DecodeCodec::H264, &[0, 0, 0, 1, 0x68, 0xce, 0x38]),
            None
        );
        // VP9 and AV1 packets are not scanned.
        assert_eq!(input.sps_location(DecodeCodec::Vp9, &packet), None);
        assert_eq!(input.sps_location(DecodeCodec::Av1, &packet), None);
    }

    #[test]
    fn length_prefixed_samples_report_the_in_band_sps_location() {
        let mut sample = vec![0, 0, 0, CENTER_SPS.len() as u8];
        sample.extend(CENTER_SPS);
        sample.extend([0, 0, 0, PPS.len() as u8]);
        sample.extend(PPS);
        let input = Input::AnnexB {
            filter: ToAnnexB::h264(&AvcConfig {
                profile_indication: 77,
                profile_compatibility: 0,
                level_indication: 30,
                length_size: LengthSize::Four,
                sps: vec![CENTER_SPS.to_vec()],
                pps: vec![PPS.to_vec()],
                high_profile: None,
            }),
            length_size: LengthSize::Four,
        };
        assert_eq!(
            input.sps_location(DecodeCodec::H264, &sample),
            Some(Some(ChromaLocation::Center))
        );
        // A length past the end of the sample ends the scan without a panic.
        assert_eq!(
            input.sps_location(DecodeCodec::H264, &[0, 0, 9, 9, 1]),
            None
        );
    }

    #[test]
    fn config_gives_the_first_sps_location() {
        let config = AvcConfig {
            profile_indication: 77,
            profile_compatibility: 0,
            level_indication: 30,
            length_size: LengthSize::Four,
            sps: vec![CENTER_SPS.to_vec()],
            pps: vec![PPS.to_vec()],
            high_profile: None,
        };
        let Ok(bytes) = config.to_bytes() else {
            panic!("avcC did not serialize");
        };
        let Ok((input, location)) = Input::new(DecodeCodec::H264, &bytes) else {
            panic!("avcC input failed");
        };
        assert!(matches!(input, Input::AnnexB { .. }));
        assert_eq!(location, Some(ChromaLocation::Center));
    }

    #[test]
    fn corrupt_config_is_a_bitstream_error() {
        assert!(matches!(
            Input::new(DecodeCodec::H264, &[1, 2]),
            Err(DecodeError::Bitstream(_))
        ));
        assert!(matches!(
            Input::new(DecodeCodec::Av1, &[0x81]),
            Err(DecodeError::Bitstream(_))
        ));
    }
}
