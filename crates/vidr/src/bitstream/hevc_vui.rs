//! HEVC `vui_parameters()` (ITU-T H.265 E.2.1) and `hrd_parameters()`
//! (E.2.2), read up to `min_spatial_segmentation_idc`.

use super::bits::BitReader;
use super::hevc_sps::STRUCTURE;
use crate::error::BitstreamError;

/// Reads `vui_parameters()` and returns `min_spatial_segmentation_idc`, or
/// `None` when `bitstream_restriction_flag` is 0.
pub(crate) fn parse(
    reader: &mut BitReader<'_>,
    max_sub_layers_minus1: u8,
) -> Result<Option<u32>, BitstreamError> {
    if reader.bit("aspect_ratio_info_present_flag")? && reader.u8(8, "aspect_ratio_idc")? == 255 {
        reader.skip(32, "sar_width")?;
    }
    if reader.bit("overscan_info_present_flag")? {
        reader.skip(1, "overscan_appropriate_flag")?;
    }
    if reader.bit("video_signal_type_present_flag")? {
        reader.skip(4, "video_format")?;
        if reader.bit("colour_description_present_flag")? {
            reader.skip(24, "colour_primaries")?;
        }
    }
    if reader.bit("chroma_loc_info_present_flag")? {
        reader.ue("chroma_sample_loc_type_top_field")?;
        reader.ue("chroma_sample_loc_type_bottom_field")?;
    }
    reader.skip(3, "frame_field_info_present_flag")?;
    if reader.bit("default_display_window_flag")? {
        for field in [
            "def_disp_win_left_offset",
            "def_disp_win_right_offset",
            "def_disp_win_top_offset",
            "def_disp_win_bottom_offset",
        ] {
            reader.ue(field)?;
        }
    }
    if reader.bit("vui_timing_info_present_flag")? {
        reader.skip(64, "vui_time_scale")?;
        if reader.bit("vui_poc_proportional_to_timing_flag")? {
            reader.ue("vui_num_ticks_poc_diff_one_minus1")?;
        }
        if reader.bit("vui_hrd_parameters_present_flag")? {
            skip_hrd_parameters(reader, max_sub_layers_minus1)?;
        }
    }
    if !reader.bit("bitstream_restriction_flag")? {
        return Ok(None);
    }
    reader.skip(3, "restricted_ref_pic_lists_flag")?;
    Ok(Some(reader.ue("min_spatial_segmentation_idc")?))
}

/// Reads `hrd_parameters(1, max_sub_layers_minus1)`.
fn skip_hrd_parameters(
    reader: &mut BitReader<'_>,
    max_sub_layers_minus1: u8,
) -> Result<(), BitstreamError> {
    let nal = reader.bit("nal_hrd_parameters_present_flag")?;
    let vcl = reader.bit("vcl_hrd_parameters_present_flag")?;
    let mut sub_pic = false;
    if nal || vcl {
        sub_pic = reader.bit("sub_pic_hrd_params_present_flag")?;
        if sub_pic {
            reader.skip(19, "dpb_output_delay_du_length_minus1")?;
        }
        reader.skip(8, "cpb_size_scale")?;
        if sub_pic {
            reader.skip(4, "cpb_size_du_scale")?;
        }
        reader.skip(15, "dpb_output_delay_length_minus1")?;
    }
    for _ in 0..=max_sub_layers_minus1 {
        let fixed_general = reader.bit("fixed_pic_rate_general_flag")?;
        let fixed_within_cvs = fixed_general || reader.bit("fixed_pic_rate_within_cvs_flag")?;
        let low_delay = if fixed_within_cvs {
            reader.ue("elemental_duration_in_tc_minus1")?;
            false
        } else {
            reader.bit("low_delay_hrd_flag")?
        };
        let mut cpb_cnt_minus1 = 0;
        if !low_delay {
            cpb_cnt_minus1 = reader.ue("cpb_cnt_minus1")?;
            if cpb_cnt_minus1 > 31 {
                return Err(BitstreamError::InvalidField {
                    structure: STRUCTURE,
                    field: "cpb_cnt_minus1",
                    value: u64::from(cpb_cnt_minus1),
                    allowed: "0 to 31",
                });
            }
        }
        for present in [nal, vcl] {
            if present {
                skip_sub_layer_hrd(reader, cpb_cnt_minus1, sub_pic)?;
            }
        }
    }
    Ok(())
}

/// Reads `sub_layer_hrd_parameters()`.
fn skip_sub_layer_hrd(
    reader: &mut BitReader<'_>,
    cpb_cnt_minus1: u32,
    sub_pic: bool,
) -> Result<(), BitstreamError> {
    for _ in 0..=cpb_cnt_minus1 {
        reader.ue("bit_rate_value_minus1")?;
        reader.ue("cpb_size_value_minus1")?;
        if sub_pic {
            reader.ue("cpb_size_du_value_minus1")?;
            reader.ue("bit_rate_du_value_minus1")?;
        }
        reader.skip(1, "cbr_flag")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_vui_has_no_restriction() {
        // Eleven zero flags.
        let mut reader = BitReader::new(&[0, 0], "test");
        assert_eq!(parse(&mut reader, 0), Ok(None));
    }

    #[test]
    fn restriction_after_timing_and_hrd() {
        // aspect 0, overscan 0, signal 0, chroma loc 0, 3 flags 000,
        // display window 0, timing 1, 64 timing bits, poc 0, hrd 1:
        // nal 1, vcl 0, sub_pic 0, 8 + 15 bits, fixed_general 1, ue `1`,
        // cpb_cnt `1`, bit rate `1`, cpb size `1`, cbr 0,
        // restriction 1, 3 flags, min_spatial_segmentation_idc 3 `00100`.
        let bits = "00000000".to_owned()
            + "1"
            + &"0".repeat(64)
            + "01"
            + "100"
            + &"0".repeat(23)
            + "1"
            + "1"
            + "1"
            + "110"
            + "1000"
            + "00100";
        let mut bytes = vec![0u8; bits.len().div_ceil(8)];
        for (i, bit) in bits.bytes().enumerate() {
            if bit == b'1' {
                bytes[i / 8] |= 0x80 >> (i % 8);
            }
        }
        let mut reader = BitReader::new(&bytes, "test");
        assert_eq!(parse(&mut reader, 0), Ok(Some(3)));
    }

    #[test]
    fn truncated_vui_is_an_error() {
        // bitstream_restriction_flag 1, then the fields cut short.
        let mut reader = BitReader::new(&[0b0000_0000, 0b0100_0000], "test");
        assert!(parse(&mut reader, 0).is_err());
    }
}
