//! HEVC `seq_parameter_set_rbsp()` (ITU-T H.265 7.3.2.2) for
//! `nuh_layer_id` 0, read up to the end of `vui_parameters()`.

use super::bits::BitReader;
use super::hevc_ptl::{self, GeneralPtl};
use super::hevc_vui;
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
    let min_spatial_segmentation_idc = if reader.bit("vui_parameters_present_flag")? {
        hevc_vui::parse(&mut reader, max_sub_layers_minus1)?
    } else {
        None
    };
    Ok(SpsFields {
        max_sub_layers_minus1,
        temporal_id_nesting,
        ptl,
        chroma_format_idc,
        bit_depth_luma_minus8,
        bit_depth_chroma_minus8,
        min_spatial_segmentation_idc,
    })
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
