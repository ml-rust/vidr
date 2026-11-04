//! HEVC `vui_parameters()` (ITU-T H.265 E.2.1) and `hrd_parameters()`
//! (E.2.2). [`parse`] reads up to `min_spatial_segmentation_idc`, and
//! [`read_restriction_tail`] reads the rest. The colour fields come from
//! [`vui::read_signal`].

use super::bits::BitReader;
use super::hevc_sps::STRUCTURE;
use super::vui::{self, VuiSignal};
use crate::error::BitstreamError;

/// The fields of `vui_parameters()` that vidr keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Vui {
    /// The colour fields.
    pub(crate) signal: VuiSignal,
    /// `None` without `bitstream_restriction_flag`.
    pub(crate) min_spatial_segmentation_idc: Option<u32>,
}

/// Reads `vui_parameters()`.
pub(crate) fn parse(
    reader: &mut BitReader<'_>,
    max_sub_layers_minus1: u8,
) -> Result<Vui, BitstreamError> {
    let signal = vui::read_signal(reader, STRUCTURE)?;
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
            skip_hrd_parameters(reader, None, max_sub_layers_minus1)?;
        }
    }
    let min_spatial_segmentation_idc = if reader.bit("bitstream_restriction_flag")? {
        reader.skip(3, "restricted_ref_pic_lists_flag")?;
        Some(reader.ue("min_spatial_segmentation_idc")?)
    } else {
        None
    };
    Ok(Vui {
        signal,
        min_spatial_segmentation_idc,
    })
}

/// Reads the bitstream restriction fields after
/// `min_spatial_segmentation_idc`, which [`parse`] leaves unread. It reads
/// nothing without `bitstream_restriction_flag`.
pub(crate) fn read_restriction_tail(
    reader: &mut BitReader<'_>,
    vui: &Vui,
) -> Result<(), BitstreamError> {
    if vui.min_spatial_segmentation_idc.is_some() {
        for field in [
            "max_bytes_per_pic_denom",
            "max_bits_per_min_cu_denom",
            "log2_max_mv_length_horizontal",
            "log2_max_mv_length_vertical",
        ] {
            reader.ue(field)?;
        }
    }
    Ok(())
}

/// The `hrd_parameters()` flags common to all sub-layers that later syntax
/// reads. A structure with `commonInfPresentFlag` 0 takes them from the
/// structure before it (7.4.3.1, `cprms_present_flag`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HrdCommon {
    nal: bool,
    vcl: bool,
    sub_pic: bool,
}

/// Reads `hrd_parameters()`. `inherited` is `None` for `commonInfPresentFlag`
/// 1. It holds the flags of the previous structure for 0.
pub(crate) fn skip_hrd_parameters(
    reader: &mut BitReader<'_>,
    inherited: Option<HrdCommon>,
    max_sub_layers_minus1: u8,
) -> Result<HrdCommon, BitstreamError> {
    let common = match inherited {
        Some(common) => common,
        None => read_hrd_common(reader)?,
    };
    let HrdCommon { nal, vcl, sub_pic } = common;
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
                return Err(reader.invalid("cpb_cnt_minus1", u64::from(cpb_cnt_minus1), "0 to 31"));
            }
        }
        for present in [nal, vcl] {
            if present {
                skip_sub_layer_hrd(reader, cpb_cnt_minus1, sub_pic)?;
            }
        }
    }
    Ok(common)
}

/// Reads the fields of `hrd_parameters()` that `commonInfPresentFlag` guards.
fn read_hrd_common(reader: &mut BitReader<'_>) -> Result<HrdCommon, BitstreamError> {
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
    Ok(HrdCommon { nal, vcl, sub_pic })
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
    use super::super::vui::{ChromaSampleLoc, pack_bits};
    use super::*;

    #[test]
    fn empty_vui_has_no_restriction() {
        // Eleven zero flags.
        let mut reader = BitReader::new(&[0, 0], "test");
        assert_eq!(
            parse(&mut reader, 0),
            Ok(Vui {
                signal: VuiSignal::default(),
                min_spatial_segmentation_idc: None,
            })
        );
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
        let bytes = pack_bits(&bits);
        let mut reader = BitReader::new(&bytes, "test");
        assert_eq!(
            parse(&mut reader, 0).map(|vui| vui.min_spatial_segmentation_idc),
            Ok(Some(3))
        );
    }

    #[test]
    fn chroma_location_precedes_the_rest() {
        // aspect 0, overscan 0, signal 0, chroma loc 1, top `010` (1),
        // bottom `1` (0), 3 flags 000, window 0, timing 0, restriction 0.
        let bytes = pack_bits(concat!("000", "1", "010", "1", "000", "0", "0", "0"));
        let mut reader = BitReader::new(&bytes, "test");
        let vui = parse(&mut reader, 0).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(
            vui.signal.chroma_sample_loc,
            Some(ChromaSampleLoc {
                top_field: 1,
                bottom_field: 0,
            })
        );
        assert_eq!(vui.min_spatial_segmentation_idc, None);
    }

    #[test]
    fn restriction_tail_reads_four_codes_only_with_the_flag() {
        let vui = Vui {
            signal: VuiSignal::default(),
            min_spatial_segmentation_idc: Some(0),
        };
        // Four `1` codes, then one spare bit.
        let bytes = [0b1111_1000];
        let mut reader = BitReader::new(&bytes, "test");
        assert_eq!(read_restriction_tail(&mut reader, &vui), Ok(()));
        assert_eq!(reader.position(), 4);
        // A cut inside the codes is a truncation.
        let mut reader = BitReader::new(&[], "test");
        assert!(read_restriction_tail(&mut reader, &vui).is_err());
        // No restriction flag: no bits.
        let none = Vui {
            signal: VuiSignal::default(),
            min_spatial_segmentation_idc: None,
        };
        let mut reader = BitReader::new(&[], "test");
        assert_eq!(read_restriction_tail(&mut reader, &none), Ok(()));
    }

    #[test]
    fn inherited_hrd_flags_select_the_sub_layer_syntax() {
        // nal 1, vcl 0, sub_pic 0, 8 + 15 bits.
        let bits = "100".to_owned()
            + &"0".repeat(23)
            // fixed_general 1, elemental `1`, cpb_cnt `1`, one NAL entry:
            // bit rate `1`, cpb size `1`, cbr 0.
            + "1"
            + "1"
            + "1"
            + "110";
        let bytes = pack_bits(&bits);
        let mut reader = BitReader::new(&bytes, "test");
        let common =
            skip_hrd_parameters(&mut reader, None, 0).unwrap_or_else(|error| panic!("{error}"));
        assert!(common.nal && !common.vcl && !common.sub_pic);
        // The same flags with no common fields: only the sub-layer syntax.
        let bytes = pack_bits("111110");
        let mut reader = BitReader::new(&bytes, "test");
        assert_eq!(
            skip_hrd_parameters(&mut reader, Some(common), 0),
            Ok(common)
        );
        assert_eq!(reader.position(), 6);
    }

    #[test]
    fn truncated_vui_is_an_error() {
        // bitstream_restriction_flag 1, then the fields cut short.
        let mut reader = BitReader::new(&[0b0000_0000, 0b0100_0000], "test");
        assert!(parse(&mut reader, 0).is_err());
    }
}
