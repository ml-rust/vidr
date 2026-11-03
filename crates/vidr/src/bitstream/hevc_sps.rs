//! HEVC `seq_parameter_set_rbsp()` (ITU-T H.265 7.3.2.2) for
//! `nuh_layer_id` 0, read up to the end of `vui_parameters()`.

use super::bits::BitReader;
use super::hevc_params;
use super::hevc_ptl::{self, GeneralPtl};
use super::hevc_vui;
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

fn invalid(field: &'static str, value: u64, allowed: &'static str) -> BitstreamError {
    BitstreamError::InvalidField {
        structure: STRUCTURE,
        field,
        value,
        allowed,
    }
}

/// Reads an SPS payload: the RBSP after the 2-byte NAL unit header.
pub(crate) fn parse(rbsp: &[u8]) -> Result<SpsFields, BitstreamError> {
    let mut reader = BitReader::new(rbsp, STRUCTURE);
    reader.skip(4, "sps_video_parameter_set_id")?;
    let max_sub_layers_minus1 = reader.u8(3, "sps_max_sub_layers_minus1")?;
    if max_sub_layers_minus1 > 6 {
        return Err(invalid(
            "sps_max_sub_layers_minus1",
            max_sub_layers_minus1.into(),
            "0 to 6",
        ));
    }
    let temporal_id_nesting = reader.bit("sps_temporal_id_nesting_flag")?;
    let ptl = hevc_ptl::parse(&mut reader, max_sub_layers_minus1)?;
    reader.ue("sps_seq_parameter_set_id")?;
    let chroma_format_idc = reader.ue("chroma_format_idc")?;
    if chroma_format_idc > 3 {
        return Err(invalid(
            "chroma_format_idc",
            chroma_format_idc.into(),
            "0 to 3",
        ));
    }
    if chroma_format_idc == 3 {
        reader.skip(1, "separate_colour_plane_flag")?;
    }
    reader.ue("pic_width_in_luma_samples")?;
    reader.ue("pic_height_in_luma_samples")?;
    if reader.bit("conformance_window_flag")? {
        for field in [
            "conf_win_left_offset",
            "conf_win_right_offset",
            "conf_win_top_offset",
            "conf_win_bottom_offset",
        ] {
            reader.ue(field)?;
        }
    }
    let bit_depth_luma_minus8 = reader.ue("bit_depth_luma_minus8")?;
    let bit_depth_chroma_minus8 = reader.ue("bit_depth_chroma_minus8")?;
    let poc_lsb_minus4 = reader.ue("log2_max_pic_order_cnt_lsb_minus4")?;
    if poc_lsb_minus4 > 12 {
        return Err(invalid(
            "log2_max_pic_order_cnt_lsb_minus4",
            poc_lsb_minus4.into(),
            "0 to 12",
        ));
    }
    let first = if reader.bit("sps_sub_layer_ordering_info_present_flag")? {
        0
    } else {
        max_sub_layers_minus1
    };
    for _ in first..=max_sub_layers_minus1 {
        reader.ue("sps_max_dec_pic_buffering_minus1")?;
        reader.ue("sps_max_num_reorder_pics")?;
        reader.ue("sps_max_latency_increase_plus1")?;
    }
    for field in [
        "log2_min_luma_coding_block_size_minus3",
        "log2_diff_max_min_luma_coding_block_size",
        "log2_min_luma_transform_block_size_minus2",
        "log2_diff_max_min_luma_transform_block_size",
        "max_transform_hierarchy_depth_inter",
        "max_transform_hierarchy_depth_intra",
    ] {
        reader.ue(field)?;
    }
    if reader.bit("scaling_list_enabled_flag")?
        && reader.bit("sps_scaling_list_data_present_flag")?
    {
        skip_scaling_list_data(&mut reader)?;
    }
    reader.skip(2, "sample_adaptive_offset_enabled_flag")?;
    if reader.bit("pcm_enabled_flag")? {
        reader.skip(8, "pcm_sample_bit_depth_chroma_minus1")?;
        reader.ue("log2_min_pcm_luma_coding_block_size_minus3")?;
        reader.ue("log2_diff_max_min_pcm_luma_coding_block_size")?;
        reader.skip(1, "pcm_loop_filter_disabled_flag")?;
    }
    skip_short_term_ref_pic_sets(&mut reader)?;
    if reader.bit("long_term_ref_pics_present_flag")? {
        let count = reader.ue("num_long_term_ref_pics_sps")?;
        if count > 32 {
            return Err(invalid(
                "num_long_term_ref_pics_sps",
                count.into(),
                "0 to 32",
            ));
        }
        let width = u64::from(poc_lsb_minus4) + 4;
        for _ in 0..count {
            reader.skip(width + 1, "used_by_curr_pic_lt_sps_flag")?;
        }
    }
    reader.skip(2, "strong_intra_smoothing_enabled_flag")?;
    let (min_spatial_segmentation_idc, signal) = if reader.bit("vui_parameters_present_flag")? {
        let vui = hevc_vui::parse(&mut reader, max_sub_layers_minus1)?;
        (vui.min_spatial_segmentation_idc, vui.signal)
    } else {
        (None, VuiSignal::default())
    };
    Ok(SpsFields {
        max_sub_layers_minus1,
        temporal_id_nesting,
        ptl,
        chroma_format_idc,
        bit_depth_luma_minus8,
        bit_depth_chroma_minus8,
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

/// Reads `scaling_list_data()` (7.3.4).
fn skip_scaling_list_data(reader: &mut BitReader<'_>) -> Result<(), BitstreamError> {
    for size_id in 0..4u32 {
        let matrices = if size_id == 3 { 2 } else { 6 };
        for _ in 0..matrices {
            if !reader.bit("scaling_list_pred_mode_flag")? {
                reader.ue("scaling_list_pred_matrix_id_delta")?;
                continue;
            }
            let coefficients = (1u32 << (4 + (size_id << 1))).min(64);
            if size_id > 1 {
                reader.se("scaling_list_dc_coef_minus8")?;
            }
            for _ in 0..coefficients {
                reader.se("scaling_list_delta_coef")?;
            }
        }
    }
    Ok(())
}

/// Reads `num_short_term_ref_pic_sets` and each `st_ref_pic_set()` (7.3.7).
fn skip_short_term_ref_pic_sets(reader: &mut BitReader<'_>) -> Result<(), BitstreamError> {
    let count = reader.ue("num_short_term_ref_pic_sets")?;
    if count > 64 {
        return Err(invalid(
            "num_short_term_ref_pic_sets",
            count.into(),
            "0 to 64",
        ));
    }
    // NumDeltaPocs of each set read so far.
    let mut delta_pocs = [0u32; 64];
    for index in 0..count as usize {
        let predicted = index > 0 && reader.bit("inter_ref_pic_set_prediction_flag")?;
        delta_pocs[index] = if predicted {
            // In an SPS the reference set is the previous one.
            reader.skip(1, "delta_rps_sign")?;
            reader.ue("abs_delta_rps_minus1")?;
            let mut used = 0;
            for _ in 0..=delta_pocs[index - 1] {
                let used_by_curr = reader.bit("used_by_curr_pic_flag")?;
                if used_by_curr || reader.bit("use_delta_flag")? {
                    used += 1;
                }
            }
            used
        } else {
            let negative = reader.ue("num_negative_pics")?;
            let positive = reader.ue("num_positive_pics")?;
            // Each picture takes at least 2 bits. This bounds the loops.
            if (u64::from(negative) + u64::from(positive)) * 2 > reader.bits_left() {
                return Err(reader.truncated("delta_poc_s0_minus1"));
            }
            for _ in 0..u64::from(negative) + u64::from(positive) {
                reader.ue("delta_poc_s0_minus1")?;
                reader.skip(1, "used_by_curr_pic_s0_flag")?;
            }
            negative.saturating_add(positive)
        };
    }
    Ok(())
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
