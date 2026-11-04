//! HEVC `seq_parameter_set_rbsp()` (ITU-T H.265 7.3.2.2) for
//! `nuh_layer_id` 0, read up to the end of `vui_parameters()`.

use super::bits::BitReader;
use super::hevc_params;
use super::hevc_ptl::GeneralPtl;
use super::hevc_sps_full;
use super::nal::HevcNalType;
use super::vui::VuiSignal;
use crate::error::BitstreamError;

pub(crate) const STRUCTURE: &str = "HEVC SPS";

/// SPS fields that `hvcC` copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SpsFields {
    pub(crate) max_sub_layers_minus1: u8,
    pub(crate) temporal_id_nesting: bool,
    pub(crate) ptl: GeneralPtl,
    pub(crate) chroma_format_idc: u32,
    pub(crate) bit_depth_luma_minus8: u32,
    pub(crate) bit_depth_chroma_minus8: u32,
    /// `None` without `bitstream_restriction_flag`.
    pub(crate) min_spatial_segmentation_idc: Option<u32>,
    /// The VUI colour fields. Empty without `vui_parameters_present_flag`.
    pub(crate) signal: VuiSignal,
}

/// Reads an SPS payload: the RBSP after the 2-byte NAL unit header. The syntax
/// walk is [`hevc_sps_full::read_through_vui`].
pub(crate) fn parse(rbsp: &[u8]) -> Result<SpsFields, BitstreamError> {
    let mut reader = BitReader::new(rbsp, STRUCTURE);
    let sps = hevc_sps_full::read_through_vui(&mut reader)?;
    let (min_spatial_segmentation_idc, signal) = match sps.vui {
        Some(vui) => (vui.min_spatial_segmentation_idc, vui.signal),
        None => (None, VuiSignal::default()),
    };
    Ok(SpsFields {
        max_sub_layers_minus1: sps.sps_max_sub_layers_minus1,
        temporal_id_nesting: sps.sps_temporal_id_nesting_flag,
        ptl: sps.profile_tier_level.general,
        chroma_format_idc: sps.chroma_format_idc,
        bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
        bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
        min_spatial_segmentation_idc,
        signal,
    })
}

/// Reads the VUI colour fields of an SPS NAL unit, 2-byte header included.
///
/// An SPS without `vui_parameters_present_flag` gives an empty
/// [`VuiSignal`]. The whole SPS up to the end of the VUI must parse.
///
/// # Errors
///
/// [`BitstreamError`] for a NAL unit that is not an SPS of `nuh_layer_id` 0,
/// an SPS that ends early, or a field out of range.
pub fn hevc_sps_vui(nal: &[u8]) -> Result<VuiSignal, BitstreamError> {
    let rbsp = hevc_params::payload(nal, HevcNalType::SPS, STRUCTURE)?;
    Ok(parse(&rbsp)?.signal)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::vui::{ChromaSampleLoc, pack_bits};
    use super::*;
    use crate::bitstream::rbsp::unescape;

    /// The SPS of an x265 128x96 Main profile stream at level 1 (30), from
    /// `hvcC`. It carries a VUI without `bitstream_restriction_flag`.
    pub(crate) const X265_SPS: [u8; 41] = [
        0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03, 0x00, 0x00,
        0x03, 0x00, 0x1e, 0xa0, 0x10, 0x20, 0x61, 0x65, 0x95, 0x92, 0x49, 0x32, 0xbc, 0x05, 0xa0,
        0x20, 0x00, 0x00, 0x03, 0x00, 0x20, 0x00, 0x00, 0x03, 0x03, 0x21,
    ];

    #[test]
    fn reads_the_fields_of_a_real_sps() {
        let rbsp = unescape(&X265_SPS[2..]);
        let fields = parse(&rbsp).unwrap();
        assert_eq!(fields.max_sub_layers_minus1, 0);
        assert!(fields.temporal_id_nesting);
        assert_eq!(fields.ptl.profile_idc, 1);
        assert_eq!(fields.ptl.level_idc, 30);
        assert_eq!(fields.ptl.compatibility_flags, 0x6000_0000);
        assert_eq!(fields.ptl.constraint_flags, 0x9000_0000_0000);
        assert_eq!(fields.chroma_format_idc, 1);
        assert_eq!(fields.bit_depth_luma_minus8, 0);
        assert_eq!(fields.min_spatial_segmentation_idc, None);
    }

    /// A 1x1 Main profile SPS NAL unit whose bits after
    /// `vui_parameters_present_flag` are `vui`.
    fn sps_with(vui: &str) -> Vec<u8> {
        let zeros = "0".repeat(80);
        let ptl = ["00000001", zeros.as_str(), "00011110"].concat();
        let bits = [
            // vps id, max_sub_layers_minus1, temporal_id_nesting.
            "0000",
            "000",
            "1",
            ptl.as_str(),
            // sps id, chroma_format_idc 1, width, height, conformance window.
            "1",
            "010",
            "1",
            "1",
            "0",
            // bit depths, poc lsb, sub_layer_ordering_info_present_flag and
            // its three fields, six coding block fields.
            "1",
            "1",
            "1",
            "1",
            "111",
            "111111",
            // scaling list, amp, sao, pcm, one empty short-term set, no
            // long-term pictures, temporal mvp, strong intra smoothing.
            "0",
            "0",
            "0",
            "0",
            "1",
            "0",
            "0",
            "0",
            vui,
        ]
        .concat();
        let mut nal = vec![0x42, 0x01];
        nal.extend(pack_bits(&bits));
        nal
    }

    /// The VUI tail after the chroma location: three flags, display window,
    /// timing, and bitstream restriction.
    const VUI_TAIL: &str = "000000";

    #[test]
    fn each_vui_flag_state_gives_its_signal() {
        let loc = |top_field, bottom_field| {
            Some(ChromaSampleLoc {
                top_field,
                bottom_field,
            })
        };
        // No VUI.
        assert_eq!(hevc_sps_vui(&sps_with("0")), Ok(VuiSignal::default()));
        // VUI with every presence flag 0.
        let none = ["1", "0", "0", "0", "0", VUI_TAIL].concat();
        assert_eq!(hevc_sps_vui(&sps_with(&none)), Ok(VuiSignal::default()));
        // Chroma location only: top `010` (1), bottom `011` (2).
        let chroma = ["1", "0", "0", "0", "1", "010", "011", VUI_TAIL].concat();
        assert_eq!(
            hevc_sps_vui(&sps_with(&chroma)).map(|v| v.chroma_sample_loc),
            Ok(loc(1, 2))
        );
        // Signal type and chroma location type 0: top `1`, bottom `1`.
        let both = [
            "1", "0", "0", "1", "101", "0", "1", "00000001", "00000001", "00000001", "1", "1", "1",
            VUI_TAIL,
        ]
        .concat();
        let signal = hevc_sps_vui(&sps_with(&both)).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(signal.chroma_sample_loc, loc(0, 0));
        assert_eq!(
            signal
                .video_signal_type
                .map(|t| (t.full_range, t.colour.is_some())),
            Some((false, true))
        );
    }

    #[test]
    fn truncated_vui_is_the_bitstream_error() {
        // Chroma location flag set, bottom field missing.
        let nal = sps_with("10001010");
        assert!(matches!(
            hevc_sps_vui(&nal),
            Err(BitstreamError::Truncated { .. })
        ));
        for end in 0..nal.len() {
            // It must never panic.
            let _ = hevc_sps_vui(&nal[..end]);
        }
    }

    #[test]
    fn x265_sps_parses_to_a_signal() {
        assert!(hevc_sps_vui(&X265_SPS).is_ok());
    }

    #[test]
    fn wrong_nal_type_is_an_error() {
        let mut pps = X265_SPS;
        pps[0] = 0x44;
        assert!(hevc_sps_vui(&pps).is_err());
        assert!(hevc_sps_vui(&[]).is_err());
    }

    #[test]
    fn every_truncation_is_an_error_or_a_prefix_read() {
        let rbsp = unescape(&X265_SPS[2..]);
        for end in 0..rbsp.len() {
            // A cut inside trailing VUI fields can still parse. It must never panic.
            let _ = parse(&rbsp[..end]);
        }
        assert!(parse(&rbsp[..10]).is_err());
    }
}
