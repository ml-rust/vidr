//! The chroma sample location of H.264 and HEVC streams.
//!
//! NVDEC reports no chroma location, so vidr reads it from the SPS VUI. The
//! value comes from `chroma_sample_loc_type_top_field`, which covers frames
//! and top fields. A bottom field value that differs is not kept.
//!
//! The location belongs to the SPS in effect. When several SPS NAL units
//! come in one place, the config uses the first and a packet uses the last.
//! An SPS that does not parse gives `None`, the same as an SPS without the
//! chroma location.

use crate::bitstream::{H264NalType, HevcNalType, VuiSignal, h264_sps_vui, hevc_sps_vui};
use crate::decode::{ChromaLocation, DecodeCodec};

/// The location a VUI signals, or `None` without `chroma_loc_info_present_flag`.
fn location_of(signal: &VuiSignal) -> Option<ChromaLocation> {
    signal
        .chroma_sample_loc
        .and_then(|loc| ChromaLocation::from_h273(loc.top_field))
}

/// For an SPS NAL unit of `codec`, the location it signals. `None` when
/// `nal` is not an SPS of `codec`.
fn sps_location(codec: DecodeCodec, nal: &[u8]) -> Option<Option<ChromaLocation>> {
    let signal = match codec {
        DecodeCodec::H264 if H264NalType::of(nal) == Some(H264NalType::SPS) => h264_sps_vui(nal),
        DecodeCodec::Hevc if HevcNalType::of(nal) == Some(HevcNalType::SPS) => hevc_sps_vui(nal),
        _ => return None,
    };
    Some(signal.ok().and_then(|signal| location_of(&signal)))
}

/// The location of the first SPS among `units`. The outer `None` means no
/// unit is an SPS.
pub(super) fn first_sps_location<'a>(
    codec: DecodeCodec,
    units: impl IntoIterator<Item = &'a [u8]>,
) -> Option<Option<ChromaLocation>> {
    units.into_iter().find_map(|nal| sps_location(codec, nal))
}

/// The location of the last SPS among `units`. The outer `None` means no
/// unit is an SPS.
pub(super) fn last_sps_location<'a>(
    codec: DecodeCodec,
    units: impl IntoIterator<Item = &'a [u8]>,
) -> Option<Option<ChromaLocation>> {
    units
        .into_iter()
        .filter_map(|nal| sps_location(codec, nal))
        .last()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An H.264 Main profile SPS whose VUI bits are `vui`: sps id 0, frame
    /// num, poc type 2, one reference frame, 320 pixels wide, 16 high,
    /// progressive, no cropping.
    fn h264_sps(vui: &str) -> Vec<u8> {
        let bits = [
            "1",
            "1",
            "011",
            "010",
            "0",
            "000010100",
            "1",
            "1",
            "1",
            "0",
            vui,
        ]
        .concat();
        let mut nal = vec![0x67, 77, 0, 30];
        let mut bytes = vec![0u8; bits.len().div_ceil(8)];
        for (index, bit) in bits.bytes().enumerate() {
            if bit == b'1' {
                bytes[index / 8] |= 0x80 >> (index % 8);
            }
        }
        nal.extend(bytes);
        nal
    }

    /// A VUI with chroma location types `top` and `bottom` as `ue` codes.
    fn with_location(top: &str, bottom: &str) -> Vec<u8> {
        h264_sps(&["1", "0", "0", "0", "1", top, bottom].concat())
    }

    const PPS: [u8; 3] = [0x68, 0xce, 0x38];

    #[test]
    fn each_flag_state_maps_to_a_location() {
        // No VUI, and a VUI without the chroma location.
        let none = [h264_sps("0"), h264_sps("10000")];
        for sps in &none {
            assert_eq!(sps_location(DecodeCodec::H264, sps), Some(None));
        }
        // Types 0 `1`, 1 `010`, 2 `011`, 3 `00100`, 4 `00101`, 5 `00110`.
        let codes = ["1", "010", "011", "00100", "00101", "00110"];
        for (kind, code) in codes.iter().enumerate() {
            let sps = with_location(code, "1");
            assert_eq!(
                sps_location(DecodeCodec::H264, &sps),
                Some(ChromaLocation::from_h273(kind as u8)),
                "type {kind}"
            );
        }
    }

    #[test]
    fn the_top_field_value_wins_over_the_bottom() {
        let sps = with_location("010", "011");
        assert_eq!(
            sps_location(DecodeCodec::H264, &sps),
            Some(Some(ChromaLocation::Center))
        );
    }

    #[test]
    fn a_truncated_sps_gives_none_and_a_pps_is_not_an_sps() {
        let sps = with_location("010", "011");
        assert_eq!(
            sps_location(DecodeCodec::H264, &sps[..sps.len() - 2]),
            Some(None)
        );
        assert_eq!(sps_location(DecodeCodec::H264, &PPS), None);
        assert_eq!(sps_location(DecodeCodec::Hevc, &sps), None);
        assert_eq!(sps_location(DecodeCodec::Vp9, &sps), None);
    }

    #[test]
    fn first_and_last_pick_their_sps() {
        let left = with_location("1", "1");
        let center = with_location("010", "010");
        let units = [&PPS[..], &left[..], &center[..]];
        assert_eq!(
            first_sps_location(DecodeCodec::H264, units),
            Some(Some(ChromaLocation::Left))
        );
        assert_eq!(
            last_sps_location(DecodeCodec::H264, units),
            Some(Some(ChromaLocation::Center))
        );
        assert_eq!(first_sps_location(DecodeCodec::H264, [&PPS[..]]), None);
        assert_eq!(last_sps_location(DecodeCodec::H264, []), None);
    }
}
