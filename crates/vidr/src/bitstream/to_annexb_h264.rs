//! ffmpeg's `h264_mp4toannexb` rules for one sample. The `to_annexb`
//! module documentation lists them.

use super::annexb::{self, StartCode};
use super::avcc::AvcConfig;
use super::nal::H264NalType;

/// ffmpeg's `H264BSFContext` state.
#[derive(Clone, Debug)]
pub(super) struct H264State {
    /// SPS NAL units after 4-byte start codes.
    sps: Vec<u8>,
    /// PPS NAL units after 4-byte start codes.
    pps: Vec<u8>,
    new_idr: bool,
    sps_seen: bool,
    pps_seen: bool,
}

impl H264State {
    /// The state for a new track: the next IDR gets the config.
    pub(super) fn new(config: &AvcConfig) -> Self {
        Self {
            sps: annexb::framed(config.sps.iter().map(Vec::as_slice)),
            pps: annexb::framed(config.pps.iter().map(Vec::as_slice)),
            new_idr: true,
            sps_seen: false,
            pps_seen: false,
        }
    }

    /// Converts the NAL units of one sample and updates the state.
    pub(super) fn filter(&mut self, units: &[&[u8]], out: &mut Vec<u8>) {
        let units = units.iter().copied().filter(|unit| !unit.is_empty());
        let type_of = |unit: &[u8]| H264NalType::from_header(unit[0]);

        // In-band parameter sets replace the stored ones, as ffmpeg's
        // `h264_mp4toannexb_filter_ps` does.
        let in_band = |wanted: H264NalType| {
            annexb::framed(units.clone().filter(|unit| type_of(unit) == wanted))
        };
        let sps = in_band(H264NalType::SPS);
        if !sps.is_empty() {
            self.sps = sps;
        }
        let pps = in_band(H264NalType::PPS);
        if !pps.is_empty() {
            self.pps = pps;
        }

        let start = out.len();
        let (mut new_idr, mut sps_seen, mut pps_seen) =
            (self.new_idr, self.sps_seen, self.pps_seen);
        for unit in units {
            let kind = type_of(unit);
            if kind == H264NalType::SPS {
                sps_seen = true;
                new_idr = true;
            } else if kind == H264NalType::PPS {
                pps_seen = true;
                new_idr = true;
                if !sps_seen && !self.sps.is_empty() {
                    out.extend_from_slice(&self.sps);
                    sps_seen = true;
                }
            }
            // A slice with first_mb_in_slice 0 starts a new IDR picture.
            let first_slice = unit.get(1).is_some_and(|byte| byte & 0x80 != 0);
            if !new_idr && kind.is_idr() && first_slice {
                new_idr = true;
            }
            // SEI payload type 0 is a buffering period.
            if kind.is_sei() && unit.get(1) == Some(&0) && !sps_seen && !pps_seen {
                if !self.sps.is_empty() {
                    out.extend_from_slice(&self.sps);
                    sps_seen = true;
                }
                if !self.pps.is_empty() {
                    out.extend_from_slice(&self.pps);
                    pps_seen = true;
                }
            }
            if new_idr && kind.is_idr() && !sps_seen && !pps_seen {
                out.extend_from_slice(&self.sps);
                out.extend_from_slice(&self.pps);
                new_idr = false;
            } else if new_idr && kind.is_idr() && sps_seen && !pps_seen {
                out.extend_from_slice(&self.pps);
            }
            let in_band_ps = kind == H264NalType::SPS || kind == H264NalType::PPS;
            let code = if in_band_ps || out.len() == start {
                StartCode::Four
            } else {
                StartCode::Three
            };
            out.extend_from_slice(code.bytes());
            out.extend_from_slice(unit);
            if kind == H264NalType::SLICE {
                new_idr = true;
                sps_seen = false;
                pps_seen = false;
            }
        }
        (self.new_idr, self.sps_seen, self.pps_seen) = (new_idr, sps_seen, pps_seen);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::{LengthSize, ToAnnexB};

    fn sample(units: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for unit in units {
            out.extend_from_slice(&(unit.len() as u32).to_be_bytes());
            out.extend_from_slice(unit);
        }
        out
    }

    const SPS: &[u8] = &[0x67, 0x64, 0, 0x1f, 0xac];
    const PPS: &[u8] = &[0x68, 0xee, 0x3c];
    const IDR: &[u8] = &[0x65, 0x88, 0x84];
    const IDR_NEXT_SLICE: &[u8] = &[0x65, 0x40, 0x11];
    const SLICE: &[u8] = &[0x41, 0x9a, 0x02];
    const SEI: &[u8] = &[0x06, 0x05, 0x10];

    fn filter() -> ToAnnexB {
        ToAnnexB::h264(&AvcConfig {
            profile_indication: 0x64,
            profile_compatibility: 0,
            level_indication: 0x1f,
            length_size: LengthSize::Four,
            sps: vec![SPS.to_vec()],
            pps: vec![PPS.to_vec()],
            high_profile: None,
        })
    }

    fn annexb(parts: &[(StartCode, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        for (code, unit) in parts {
            out.extend_from_slice(code.bytes());
            out.extend_from_slice(unit);
        }
        out
    }

    use StartCode::{Four, Three};

    #[test]
    fn config_goes_before_the_first_idr() {
        let mut f = filter();
        assert_eq!(
            f.filter(&sample(&[SEI, IDR, IDR_NEXT_SLICE])),
            Ok(annexb(&[
                (Four, SEI),
                (Four, SPS),
                (Four, PPS),
                (Three, IDR),
                (Three, IDR_NEXT_SLICE)
            ]))
        );
        assert_eq!(f.filter(&sample(&[SLICE])), Ok(annexb(&[(Four, SLICE)])));
        // An IDR after a non-IDR slice gets the config again.
        assert_eq!(
            f.filter(&sample(&[IDR])),
            Ok(annexb(&[(Four, SPS), (Four, PPS), (Three, IDR)]))
        );
    }

    #[test]
    fn in_band_parameter_sets_are_not_duplicated() {
        let mut f = filter();
        assert_eq!(
            f.filter(&sample(&[SPS, PPS, IDR])),
            Ok(annexb(&[(Four, SPS), (Four, PPS), (Three, IDR)]))
        );
    }

    #[test]
    fn sps_alone_gets_the_config_pps() {
        let mut f = filter();
        assert_eq!(
            f.filter(&sample(&[SPS, IDR])),
            Ok(annexb(&[(Four, SPS), (Four, PPS), (Three, IDR)]))
        );
    }

    #[test]
    fn pps_alone_gets_the_config_sps() {
        let mut f = filter();
        assert_eq!(
            f.filter(&sample(&[PPS, IDR])),
            Ok(annexb(&[(Four, SPS), (Four, PPS), (Three, IDR)]))
        );
    }

    #[test]
    fn in_band_sps_replaces_the_config_sps() {
        let other_sps: &[u8] = &[0x67, 0x64, 0, 0x28, 0xac];
        let mut f = filter();
        assert!(f.filter(&sample(&[other_sps, PPS, IDR])).is_ok());
        assert!(f.filter(&sample(&[SLICE])).is_ok());
        assert_eq!(
            f.filter(&sample(&[IDR])),
            Ok(annexb(&[(Four, other_sps), (Four, PPS), (Three, IDR)]))
        );
    }

    #[test]
    fn buffering_period_sei_gets_the_config() {
        let buffering: &[u8] = &[0x06, 0x00, 0x01, 0x80];
        let mut f = filter();
        assert_eq!(
            f.filter(&sample(&[buffering, IDR])),
            Ok(annexb(&[
                (Four, SPS),
                (Four, PPS),
                (Three, buffering),
                (Three, IDR)
            ]))
        );
    }

    #[test]
    fn zero_length_units_are_dropped_and_errors_keep_state() {
        let mut f = filter();
        let mut bytes = sample(&[SLICE]);
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        assert_eq!(f.filter(&bytes), Ok(annexb(&[(Four, SLICE)])));
        let mut out = vec![7];
        assert!(f.filter_into(&[0, 0, 0, 9, 1], &mut out).is_err());
        assert_eq!(out, vec![7]);
        // The slice set new_idr, so the next IDR still gets the config.
        assert_eq!(
            f.filter(&sample(&[IDR])),
            Ok(annexb(&[(Four, SPS), (Four, PPS), (Three, IDR)]))
        );
    }
}
