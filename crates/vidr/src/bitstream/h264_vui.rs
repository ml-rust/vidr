//! The colour fields of an H.264 SPS (ITU-T H.264 7.3.2.1.1 and E.1.1).
//!
//! The reader walks the SPS past the scaling lists, the picture order count
//! fields, the picture size, and the frame cropping to `vui_parameters()`,
//! and stops after `chroma_sample_loc_type_bottom_field`.

use super::bits::BitReader;
use super::h264_sps::{self, STRUCTURE};
use super::vui::{self, VuiSignal};
use crate::error::BitstreamError;

/// `num_ref_frames_in_pic_order_cnt_cycle` is at most 255.
const MAX_REF_FRAME_CYCLE: u32 = 255;

/// Reads the VUI colour fields of an SPS NAL unit, header byte included.
///
/// An SPS without `vui_parameters_present_flag` gives an empty
/// [`VuiSignal`].
///
/// # Errors
///
/// [`BitstreamError`] for a NAL unit that is not an SPS, an SPS that ends
/// before the chroma sample location, or a field out of range.
pub fn h264_sps_vui(nal: &[u8]) -> Result<VuiSignal, BitstreamError> {
    let payload = h264_sps::payload(nal)?;
    let mut reader = BitReader::new(&payload, STRUCTURE);
    let leading = h264_sps::read_leading(&mut reader)?;
    if h264_sps::has_chroma_fields(leading.profile_idc) {
        skip_scaling_matrix(&mut reader, leading.chroma_format_idc)?;
    }
    reader.ue("log2_max_frame_num_minus4")?;
    skip_pic_order_count(&mut reader)?;
    reader.ue("max_num_ref_frames")?;
    reader.skip(1, "gaps_in_frame_num_value_allowed_flag")?;
    reader.ue("pic_width_in_mbs_minus1")?;
    reader.ue("pic_height_in_map_units_minus1")?;
    if !reader.bit("frame_mbs_only_flag")? {
        reader.skip(1, "mb_adaptive_frame_field_flag")?;
    }
    reader.skip(1, "direct_8x8_inference_flag")?;
    if reader.bit("frame_cropping_flag")? {
        for field in [
            "frame_crop_left_offset",
            "frame_crop_right_offset",
            "frame_crop_top_offset",
            "frame_crop_bottom_offset",
        ] {
            reader.ue(field)?;
        }
    }
    if !reader.bit("vui_parameters_present_flag")? {
        return Ok(VuiSignal::default());
    }
    vui::read_signal(&mut reader, STRUCTURE)
}

/// Reads `qpprime_y_zero_transform_bypass_flag`,
/// `seq_scaling_matrix_present_flag`, and the scaling lists.
fn skip_scaling_matrix(
    reader: &mut BitReader<'_>,
    chroma_format_idc: u8,
) -> Result<(), BitstreamError> {
    reader.skip(1, "qpprime_y_zero_transform_bypass_flag")?;
    if !reader.bit("seq_scaling_matrix_present_flag")? {
        return Ok(());
    }
    let lists = if chroma_format_idc == 3 { 12 } else { 8 };
    for index in 0..lists {
        if reader.bit("seq_scaling_list_present_flag")? {
            skip_scaling_list(reader, if index < 6 { 16 } else { 64 })?;
        }
    }
    Ok(())
}

/// Reads `scaling_list()` (7.3.2.1.1.1) of `size` entries.
fn skip_scaling_list(reader: &mut BitReader<'_>, size: u32) -> Result<(), BitstreamError> {
    let mut last_scale = 8i64;
    let mut next_scale = 8i64;
    for _ in 0..size {
        if next_scale != 0 {
            let delta = reader.se("delta_scale")?;
            next_scale = (last_scale + delta + 256).rem_euclid(256);
        }
        if next_scale != 0 {
            last_scale = next_scale;
        }
    }
    Ok(())
}

/// Reads `pic_order_cnt_type` and the fields it selects.
fn skip_pic_order_count(reader: &mut BitReader<'_>) -> Result<(), BitstreamError> {
    match reader.ue("pic_order_cnt_type")? {
        0 => {
            reader.ue("log2_max_pic_order_cnt_lsb_minus4")?;
        }
        1 => {
            reader.skip(1, "delta_pic_order_always_zero_flag")?;
            reader.se("offset_for_non_ref_pic")?;
            reader.se("offset_for_top_to_bottom_field")?;
            let cycle = reader.ue("num_ref_frames_in_pic_order_cnt_cycle")?;
            if cycle > MAX_REF_FRAME_CYCLE {
                return Err(BitstreamError::InvalidField {
                    structure: STRUCTURE,
                    field: "num_ref_frames_in_pic_order_cnt_cycle",
                    value: u64::from(cycle),
                    allowed: "0 to 255",
                });
            }
            for _ in 0..cycle {
                reader.se("offset_for_ref_frame")?;
            }
        }
        2 => {}
        other => {
            return Err(BitstreamError::InvalidField {
                structure: STRUCTURE,
                field: "pic_order_cnt_type",
                value: u64::from(other),
                allowed: "0 to 2",
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::vui::{ChromaSampleLoc, pack_bits};
    use super::*;

    /// NAL header, then profile `profile`, constraints 0, level 30, and the
    /// bits of `body`.
    fn sps(profile: u8, body: &str) -> Vec<u8> {
        let mut nal = vec![0x67, profile, 0, 30];
        nal.extend(pack_bits(body));
        nal
    }

    /// A Main profile SPS with `vui` after `frame_cropping_flag`:
    /// sps id 0 `1`, frame num `1`, poc type 2 `011`, refs 1 `010`, gaps 0,
    /// width 19 `000010100`, height 0 `1`, frame_mbs_only 1, direct 8x8 1,
    /// cropping 0.
    fn main_sps(vui: &str) -> Vec<u8> {
        sps(
            77,
            &[
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
            .concat(),
        )
    }

    fn loc(top_field: u8, bottom_field: u8) -> Option<ChromaSampleLoc> {
        Some(ChromaSampleLoc {
            top_field,
            bottom_field,
        })
    }

    #[test]
    fn no_vui_gives_an_empty_signal() {
        assert_eq!(h264_sps_vui(&main_sps("0")), Ok(VuiSignal::default()));
    }

    #[test]
    fn vui_without_chroma_loc_has_none() {
        // vui 1, aspect 0, overscan 0, signal 0, chroma loc 0.
        assert_eq!(h264_sps_vui(&main_sps("10000")), Ok(VuiSignal::default()));
    }

    #[test]
    fn chroma_loc_types_come_from_the_vui() {
        // vui 1, aspect 0, overscan 0, signal 0, chroma loc 1, top `010` (1),
        // bottom `00101` (4).
        assert_eq!(
            h264_sps_vui(&main_sps("1000101000101")).map(|v| v.chroma_sample_loc),
            Ok(loc(1, 4))
        );
    }

    #[test]
    fn signal_type_and_colour_description_are_read() {
        // vui 1, aspect 0, overscan 0, signal 1, format 5, full range 1,
        // description 1: primaries 1, transfer 1, matrix 6; chroma loc 1,
        // top `1` (0), bottom `1` (0).
        let vui = [
            "1", "0", "0", "1", "101", "1", "1", "00000001", "00000001", "00000110", "1", "1", "1",
        ]
        .concat();
        let signal = h264_sps_vui(&main_sps(&vui)).unwrap_or_else(|error| panic!("{error}"));
        let kind = signal
            .video_signal_type
            .unwrap_or_else(|| panic!("no signal type"));
        assert_eq!(kind.video_format, 5);
        assert!(kind.full_range);
        assert_eq!(kind.colour.map(|c| c.matrix_coefficients), Some(6));
        assert_eq!(signal.chroma_sample_loc, loc(0, 0));
    }

    #[test]
    fn high_profile_scaling_lists_and_poc_type_one_are_skipped() {
        // sps id, chroma_format_idc 1 `010`, both depths 0, bypass 0, scaling
        // matrix 1. Each present flag comes right before its list. List 0 is
        // present with 16 entries: delta_scale +2 `00100`, then 15 zeros `1`.
        // The other seven lists are absent.
        let ones = "1".repeat(15);
        let high = [
            "1",
            "010",
            "1",
            "1",
            "0",
            "1",
            "1",
            "00100",
            ones.as_str(),
            "0000000",
        ]
        .concat();
        // frame num `1`, poc type 1 `010`, always zero 0, two offsets `1`
        // `1`, cycle 1 `010` and its offset `1`, refs `1`, gaps 0, width `1`,
        // height `1`, frame_mbs_only 0 and mbaff 0, direct 1, cropping 1 and
        // four offsets, vui 1, no aspect, overscan or signal, chroma loc 1,
        // top `1`, bottom `1`.
        let rest = [
            "1", "010", "0", "1", "1", "010", "1", "1", "0", "1", "1", "0", "0", "1", "1", "1111",
            "1", "0", "0", "0", "1", "1", "1",
        ]
        .concat();
        let nal = sps(100, &[high, rest].concat());
        assert_eq!(
            h264_sps_vui(&nal).map(|v| v.chroma_sample_loc),
            Ok(loc(0, 0))
        );
    }

    #[test]
    fn truncation_inside_the_vui_is_an_error() {
        // The bottom field is missing.
        assert!(h264_sps_vui(&main_sps("10001010")).is_err());
    }

    #[test]
    fn every_truncation_is_an_error_or_a_prefix_read() {
        let nal = main_sps("1000101000101");
        for end in 0..nal.len() {
            // It must never panic.
            let _ = h264_sps_vui(&nal[..end]);
        }
        assert!(h264_sps_vui(&nal[..6]).is_err());
    }

    #[test]
    fn wrong_nal_type_is_an_error() {
        assert!(matches!(
            h264_sps_vui(&[0x68, 77, 0, 30, 0x80]),
            Err(BitstreamError::InvalidField {
                field: "nal_unit_type",
                ..
            })
        ));
    }
}
