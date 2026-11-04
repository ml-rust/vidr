//! `quantization_params()`, `segmentation_params()`, `delta_q_params()`, `delta_lf_params()`, and
//! the lossless values (AV1 specification 5.9.2, 5.9.12 to 5.9.14, 5.9.17, and 5.9.18).

use super::types::{FrameHeader, MAX_SEGMENTS, PRIMARY_REF_NONE, SEG_LVL_ALT_Q, SEG_LVL_MAX};
use crate::bitstream::av1::seq_header::SequenceHeader;
use crate::bitstream::bits::BitReader;
use crate::error::BitstreamError;

/// `MAX_LOOP_FILTER`.
const MAX_LOOP_FILTER: i32 = 63;
/// `Segmentation_Feature_Bits`.
const FEATURE_BITS: [u32; SEG_LVL_MAX] = [8, 6, 6, 6, 6, 3, 0, 0];
/// `Segmentation_Feature_Signed`.
const FEATURE_SIGNED: [bool; SEG_LVL_MAX] = [true, true, true, true, true, false, false, false];
/// `Segmentation_Feature_Max`.
const FEATURE_MAX: [i32; SEG_LVL_MAX] = [
    255,
    MAX_LOOP_FILTER,
    MAX_LOOP_FILTER,
    MAX_LOOP_FILTER,
    MAX_LOOP_FILTER,
    7,
    0,
    0,
];

/// Reads an `su(n)` value.
pub(super) fn read_su(
    reader: &mut BitReader<'_>,
    n: u32,
    field: &'static str,
) -> Result<i32, BitstreamError> {
    let value = reader.bits(n, field)? as i32;
    let sign_mask = 1_i32 << (n - 1);
    Ok(if value & sign_mask != 0 {
        value - 2 * sign_mask
    } else {
        value
    })
}

fn read_delta_q(
    reader: &mut BitReader<'_>,
    coded: &mut bool,
    value: &'static str,
) -> Result<i8, BitstreamError> {
    *coded = reader.bit("delta_coded")?;
    if *coded {
        Ok(read_su(reader, 7, value)? as i8)
    } else {
        Ok(0)
    }
}

/// Reads `quantization_params()`.
pub(super) fn quantization_params(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    header: &mut FrameHeader,
) -> Result<(), BitstreamError> {
    let separate = sequence.color.separate_uv_delta_q;
    let q = &mut header.quantization;
    q.base_q_idx = reader.u8(8, "base_q_idx")?;
    q.delta_q_y_dc = read_delta_q(reader, &mut q.delta_coded[0], "delta_q_y_dc")?;
    if !sequence.color.mono_chrome {
        q.diff_uv_delta = separate && reader.bit("diff_uv_delta")?;
        q.delta_q_u_dc = read_delta_q(reader, &mut q.delta_coded[1], "delta_q_u_dc")?;
        q.delta_q_u_ac = read_delta_q(reader, &mut q.delta_coded[2], "delta_q_u_ac")?;
        if q.diff_uv_delta {
            q.delta_q_v_dc = read_delta_q(reader, &mut q.delta_coded[3], "delta_q_v_dc")?;
            q.delta_q_v_ac = read_delta_q(reader, &mut q.delta_coded[4], "delta_q_v_ac")?;
        } else {
            q.delta_q_v_dc = q.delta_q_u_dc;
            q.delta_q_v_ac = q.delta_q_u_ac;
        }
    }
    q.using_qmatrix = reader.bit("using_qmatrix")?;
    if q.using_qmatrix {
        q.qm_y = reader.u8(4, "qm_y")?;
        q.qm_u = reader.u8(4, "qm_u")?;
        q.qm_v = if separate {
            reader.u8(4, "qm_v")?
        } else {
            q.qm_u
        };
    }
    Ok(())
}

/// Reads `segmentation_params()`. `header` already holds what `load_previous()` loads.
pub(super) fn segmentation_params(
    reader: &mut BitReader<'_>,
    header: &mut FrameHeader,
) -> Result<(), BitstreamError> {
    let primary_none = header.primary_ref_frame == PRIMARY_REF_NONE;
    let seg = &mut header.segmentation;
    seg.enabled = reader.bit("segmentation_enabled")?;
    if !seg.enabled {
        seg.feature_enabled = [[false; SEG_LVL_MAX]; MAX_SEGMENTS];
        seg.feature_data = [[0; SEG_LVL_MAX]; MAX_SEGMENTS];
        return Ok(());
    }
    if primary_none {
        seg.update_map = true;
        seg.temporal_update = false;
        seg.update_data = true;
    } else {
        seg.update_map = reader.bit("segmentation_update_map")?;
        seg.temporal_update = seg.update_map && reader.bit("segmentation_temporal_update")?;
        seg.update_data = reader.bit("segmentation_update_data")?;
    }
    if !seg.update_data {
        return Ok(());
    }
    for segment in 0..MAX_SEGMENTS {
        for feature in 0..SEG_LVL_MAX {
            let enabled = reader.bit("feature_enabled")?;
            seg.feature_enabled[segment][feature] = enabled;
            let mut clipped = 0;
            if enabled {
                let (bits, limit) = (FEATURE_BITS[feature], FEATURE_MAX[feature]);
                clipped = if FEATURE_SIGNED[feature] {
                    read_su(reader, 1 + bits, "feature_value")?.clamp(-limit, limit)
                } else {
                    (reader.bits(bits, "feature_value")? as i32).clamp(0, limit)
                };
            }
            seg.feature_data[segment][feature] = clipped as i16;
        }
    }
    Ok(())
}

/// Reads `delta_q_params()`.
pub(super) fn delta_q_params(
    reader: &mut BitReader<'_>,
    header: &mut FrameHeader,
) -> Result<(), BitstreamError> {
    header.delta_q_present = header.quantization.base_q_idx > 0 && reader.bit("delta_q_present")?;
    if header.delta_q_present {
        header.delta_q_res = reader.u8(2, "delta_q_res")?;
    }
    Ok(())
}

/// Reads `delta_lf_params()`.
pub(super) fn delta_lf_params(
    reader: &mut BitReader<'_>,
    header: &mut FrameHeader,
) -> Result<(), BitstreamError> {
    if !header.delta_q_present {
        return Ok(());
    }
    header.delta_lf_present = !header.allow_intrabc && reader.bit("delta_lf_present")?;
    if header.delta_lf_present {
        header.delta_lf_res = reader.u8(2, "delta_lf_res")?;
        header.delta_lf_multi = reader.bit("delta_lf_multi")?;
    }
    Ok(())
}

/// `get_qindex( 1, segmentId )`.
fn qindex(header: &FrameHeader, segment: usize) -> i32 {
    let base = i32::from(header.quantization.base_q_idx);
    let seg = &header.segmentation;
    if seg.enabled && seg.feature_enabled[segment][SEG_LVL_ALT_Q] {
        (base + i32::from(seg.feature_data[segment][SEG_LVL_ALT_Q])).clamp(0, 255)
    } else {
        base
    }
}

/// Sets `CodedLossless` and `AllLossless`.
pub(super) fn coded_lossless(header: &mut FrameHeader) {
    let q = header.quantization;
    let no_delta = q.delta_q_y_dc == 0
        && q.delta_q_u_ac == 0
        && q.delta_q_u_dc == 0
        && q.delta_q_v_ac == 0
        && q.delta_q_v_dc == 0;
    header.coded_lossless =
        no_delta && (0..MAX_SEGMENTS).all(|segment| qindex(header, segment) == 0);
    header.all_lossless = header.coded_lossless && header.frame_width == header.upscaled_width;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::test_bits::BitWriter;

    #[test]
    fn su_reads_the_sign_from_the_top_bit() {
        let mut w = BitWriter::new();
        w.u(7, 0b000_0011).u(7, 0b111_1111).u(7, 0b100_0000);
        let bytes = w.bytes();
        let mut reader = BitReader::new(&bytes, "test");
        assert_eq!(read_su(&mut reader, 7, "v"), Ok(3));
        assert_eq!(read_su(&mut reader, 7, "v"), Ok(-1));
        assert_eq!(read_su(&mut reader, 7, "v"), Ok(-64));
    }

    /// `segmentation_update_data` 0 keeps what `load_previous()` loaded, and it stays through
    /// `update_map`.
    #[test]
    fn segmentation_keeps_the_loaded_features_with_no_update() {
        let mut header = FrameHeader {
            primary_ref_frame: 0,
            ..FrameHeader::default()
        };
        header.segmentation.feature_enabled[3][1] = true;
        header.segmentation.feature_data[3][1] = -9;
        let mut w = BitWriter::new();
        // Enabled, update map, temporal update, no data.
        w.flag(true).flag(true).flag(true).flag(false);
        let bytes = w.bytes();
        let mut reader = BitReader::new(&bytes, "test");
        segmentation_params(&mut reader, &mut header).expect("parses");
        assert!(header.segmentation.update_map);
        assert!(header.segmentation.temporal_update);
        assert!(header.segmentation.feature_enabled[3][1]);
        assert_eq!(header.segmentation.feature_data[3][1], -9);
    }

    #[test]
    fn segmentation_off_clears_the_loaded_features() {
        let mut header = FrameHeader {
            primary_ref_frame: 0,
            ..FrameHeader::default()
        };
        header.segmentation.feature_enabled[3][1] = true;
        header.segmentation.feature_data[3][1] = -9;
        let bytes = [0];
        let mut reader = BitReader::new(&bytes, "test");
        segmentation_params(&mut reader, &mut header).expect("parses");
        assert!(!header.segmentation.feature_enabled[3][1]);
        assert_eq!(header.segmentation.feature_data[3][1], 0);
    }

    /// A value past its limit clips: feature 1 holds at most 63 and 7 bits hold up to -64.
    #[test]
    fn a_feature_past_its_limit_clips() {
        let mut header = FrameHeader {
            primary_ref_frame: PRIMARY_REF_NONE,
            ..FrameHeader::default()
        };
        let mut w = BitWriter::new();
        w.flag(true);
        for segment in 0..MAX_SEGMENTS {
            for feature in 0..SEG_LVL_MAX {
                let on = segment == 2 && (feature == 1 || feature == 5);
                w.flag(on);
                if on && feature == 1 {
                    w.u(7, 0b100_0000);
                }
                if on && feature == 5 {
                    w.u(3, 7);
                }
            }
        }
        let bytes = w.bytes();
        let mut reader = BitReader::new(&bytes, "test");
        segmentation_params(&mut reader, &mut header).expect("parses");
        assert_eq!(header.segmentation.feature_data[2][1], -63);
        assert_eq!(header.segmentation.feature_data[2][5], 7);
        assert!(header.segmentation.update_data);
    }

    #[test]
    fn lossless_needs_a_zero_index_in_every_segment() {
        let mut header = FrameHeader {
            frame_width: 64,
            upscaled_width: 64,
            ..FrameHeader::default()
        };
        coded_lossless(&mut header);
        assert!(header.coded_lossless);
        assert!(header.all_lossless);
        header.frame_width = 32;
        coded_lossless(&mut header);
        assert!(header.coded_lossless);
        assert!(!header.all_lossless);
        header.segmentation.enabled = true;
        header.segmentation.feature_enabled[5][SEG_LVL_ALT_Q] = true;
        header.segmentation.feature_data[5][SEG_LVL_ALT_Q] = 4;
        coded_lossless(&mut header);
        assert!(!header.coded_lossless);
    }
}
