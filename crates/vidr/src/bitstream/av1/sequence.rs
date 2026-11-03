//! AV1 `sequence_header_obu()` (AV1 specification 5.5), read up to
//! `color_config()`, as far as `av1C` needs it.

use crate::bitstream::bits::BitReader;
use crate::error::BitstreamError;

const STRUCTURE: &str = "AV1 sequence header";

/// Sequence header fields that `av1C` copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SequenceFields {
    pub(crate) seq_profile: u8,
    pub(crate) seq_level_idx_0: u8,
    pub(crate) seq_tier_0: bool,
    pub(crate) high_bitdepth: bool,
    pub(crate) twelve_bit: bool,
    pub(crate) mono_chrome: bool,
    pub(crate) subsampling_x: bool,
    pub(crate) subsampling_y: bool,
    pub(crate) chroma_sample_position: u8,
}

/// `CP_BT_709`, `TC_SRGB`, and `MC_IDENTITY`.
const BT709_SRGB_IDENTITY: (u8, u8, u8) = (1, 13, 0);

/// Parses a sequence header OBU payload.
pub(crate) fn parse(payload: &[u8]) -> Result<SequenceFields, BitstreamError> {
    let mut reader = BitReader::new(payload, STRUCTURE);
    let seq_profile = reader.u8(3, "seq_profile")?;
    if seq_profile > 2 {
        return Err(BitstreamError::InvalidField {
            structure: STRUCTURE,
            field: "seq_profile",
            value: seq_profile.into(),
            allowed: "0 to 2",
        });
    }
    reader.skip(1, "still_picture")?;
    let reduced = reader.bit("reduced_still_picture_header")?;
    let (seq_level_idx_0, seq_tier_0) = if reduced {
        (reader.u8(5, "seq_level_idx")?, false)
    } else {
        operating_points(&mut reader)?
    };
    let width_bits = reader.bits(4, "frame_width_bits_minus_1")? + 1;
    let height_bits = reader.bits(4, "frame_height_bits_minus_1")? + 1;
    reader.skip(width_bits + height_bits, "max_frame_height_minus_1")?;
    if !reduced && reader.bit("frame_id_numbers_present_flag")? {
        reader.skip(7, "additional_frame_id_length_minus_1")?;
    }
    reader.skip(3, "enable_intra_edge_filter")?;
    if !reduced {
        reader.skip(4, "enable_dual_filter")?;
        let order_hint = reader.bit("enable_order_hint")?;
        if order_hint {
            reader.skip(2, "enable_ref_frame_mvs")?;
        }
        let force_screen_content_tools = if reader.bit("seq_choose_screen_content_tools")? {
            true
        } else {
            reader.bit("seq_force_screen_content_tools")?
        };
        if force_screen_content_tools && !reader.bit("seq_choose_integer_mv")? {
            reader.skip(1, "seq_force_integer_mv")?;
        }
        if order_hint {
            reader.skip(3, "order_hint_bits_minus_1")?;
        }
    }
    reader.skip(3, "enable_restoration")?;
    color_config(&mut reader, seq_profile, seq_level_idx_0, seq_tier_0)
}

/// Reads the timing, decoder model, and operating point fields, and returns
/// `seq_level_idx[0]` and `seq_tier[0]`.
fn operating_points(reader: &mut BitReader<'_>) -> Result<(u8, bool), BitstreamError> {
    let mut decoder_model = false;
    let mut buffer_delay_bits = 0;
    if reader.bit("timing_info_present_flag")? {
        reader.skip(64, "time_scale")?;
        if reader.bit("equal_picture_interval")? {
            reader.uvlc("num_ticks_per_picture_minus_1")?;
        }
        decoder_model = reader.bit("decoder_model_info_present_flag")?;
        if decoder_model {
            buffer_delay_bits = reader.bits(5, "buffer_delay_length_minus_1")? + 1;
            reader.skip(42, "frame_presentation_time_length_minus_1")?;
        }
    }
    let initial_display_delay = reader.bit("initial_display_delay_present_flag")?;
    let count = reader.bits(5, "operating_points_cnt_minus_1")? + 1;
    let mut first = (0, false);
    for index in 0..count {
        reader.skip(12, "operating_point_idc")?;
        let level = reader.u8(5, "seq_level_idx")?;
        let tier = level > 7 && reader.bit("seq_tier")?;
        if decoder_model && reader.bit("decoder_model_present_for_this_op")? {
            reader.skip(2 * buffer_delay_bits + 1, "low_delay_mode_flag")?;
        }
        if initial_display_delay && reader.bit("initial_display_delay_present_for_this_op")? {
            reader.skip(4, "initial_display_delay_minus_1")?;
        }
        if index == 0 {
            first = (level, tier);
        }
    }
    Ok(first)
}

/// Reads `color_config()` (5.5.2) and completes the fields.
fn color_config(
    reader: &mut BitReader<'_>,
    seq_profile: u8,
    seq_level_idx_0: u8,
    seq_tier_0: bool,
) -> Result<SequenceFields, BitstreamError> {
    let high_bitdepth = reader.bit("high_bitdepth")?;
    let twelve_bit = seq_profile == 2 && high_bitdepth && reader.bit("twelve_bit")?;
    let mono_chrome = seq_profile != 1 && reader.bit("mono_chrome")?;
    let colors = if reader.bit("color_description_present_flag")? {
        (
            reader.u8(8, "color_primaries")?,
            reader.u8(8, "transfer_characteristics")?,
            reader.u8(8, "matrix_coefficients")?,
        )
    } else {
        (2, 2, 2)
    };
    let mut fields = SequenceFields {
        seq_profile,
        seq_level_idx_0,
        seq_tier_0,
        high_bitdepth,
        twelve_bit,
        mono_chrome,
        subsampling_x: true,
        subsampling_y: true,
        chroma_sample_position: 0,
    };
    if mono_chrome {
        reader.skip(1, "color_range")?;
        return Ok(fields);
    }
    if colors == BT709_SRGB_IDENTITY {
        fields.subsampling_x = false;
        fields.subsampling_y = false;
    } else {
        reader.skip(1, "color_range")?;
        (fields.subsampling_x, fields.subsampling_y) = match seq_profile {
            0 => (true, true),
            1 => (false, false),
            _ if twelve_bit => {
                let x = reader.bit("subsampling_x")?;
                (x, x && reader.bit("subsampling_y")?)
            }
            _ => (true, false),
        };
        if fields.subsampling_x && fields.subsampling_y {
            fields.chroma_sample_position = reader.u8(2, "chroma_sample_position")?;
        }
    }
    reader.skip(1, "separate_uv_delta_q")?;
    Ok(fields)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The sequence header payload of a libaom 128x96 4:2:0 8-bit stream.
    pub(crate) const LIBAOM_PAYLOAD: [u8; 10] =
        [0x00, 0x00, 0x00, 0x03, 0x37, 0xfb, 0xe6, 0xd7, 0xcc, 0x02];

    #[test]
    fn reads_a_libaom_header() {
        assert_eq!(
            parse(&LIBAOM_PAYLOAD),
            Ok(SequenceFields {
                seq_profile: 0,
                seq_level_idx_0: 0,
                seq_tier_0: false,
                high_bitdepth: false,
                twelve_bit: false,
                mono_chrome: false,
                subsampling_x: true,
                subsampling_y: true,
                chroma_sample_position: 0,
            })
        );
    }

    #[test]
    fn reduced_still_picture_header_reads_one_level() {
        // Profile 1, still 1, reduced 1, level 31, 4-bit widths, 4:4:4 8-bit.
        // 001 1 1 11111 0000 0000 0000 0000 000 0 0 0 0
        let bits = "00111111110000000000000000000000000";
        let mut bytes = vec![0u8; bits.len().div_ceil(8)];
        for (i, bit) in bits.bytes().enumerate() {
            if bit == b'1' {
                bytes[i / 8] |= 0x80 >> (i % 8);
            }
        }
        let fields = parse(&bytes);
        assert_eq!(
            fields.map(|f| (f.seq_profile, f.seq_level_idx_0, f.subsampling_x)),
            Ok((1, 31, false))
        );
    }

    #[test]
    fn truncations_never_panic() {
        for end in 0..LIBAOM_PAYLOAD.len() {
            let _ = parse(&LIBAOM_PAYLOAD[..end]);
        }
        assert!(parse(&LIBAOM_PAYLOAD[..3]).is_err());
        assert!(parse(&[0xe0]).is_err());
    }
}
