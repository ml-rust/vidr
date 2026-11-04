//! The whole AV1 `sequence_header_obu()` (AV1 specification 5.5).

use crate::bitstream::bits::BitReader;
use crate::error::BitstreamError;

const STRUCTURE: &str = "AV1 sequence header";

/// `SELECT_SCREEN_CONTENT_TOOLS`.
pub(crate) const SELECT_SCREEN_CONTENT_TOOLS: u8 = 2;
/// `SELECT_INTEGER_MV`.
pub(crate) const SELECT_INTEGER_MV: u8 = 2;

/// `CP_BT_709`, `TC_SRGB`, and `MC_IDENTITY`.
const BT709_SRGB_IDENTITY: (u8, u8, u8) = (1, 13, 0);

/// `BitDepth` of `high_bitdepth` and `twelve_bit`.
pub(crate) const fn bit_depth(high_bitdepth: bool, twelve_bit: bool) -> u8 {
    match (high_bitdepth, twelve_bit) {
        (false, _) => 8,
        (true, false) => 10,
        (true, true) => 12,
    }
}

/// `color_config()` (5.5.2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ColorConfig {
    pub(crate) high_bitdepth: bool,
    pub(crate) twelve_bit: bool,
    /// `BitDepth`.
    pub(crate) bit_depth: u8,
    pub(crate) mono_chrome: bool,
    pub(crate) color_description_present_flag: bool,
    pub(crate) color_primaries: u8,
    pub(crate) transfer_characteristics: u8,
    pub(crate) matrix_coefficients: u8,
    pub(crate) color_range: bool,
    pub(crate) subsampling_x: bool,
    pub(crate) subsampling_y: bool,
    pub(crate) chroma_sample_position: u8,
    pub(crate) separate_uv_delta_q: bool,
}

impl ColorConfig {
    /// `NumPlanes`.
    pub(crate) const fn num_planes(&self) -> usize {
        if self.mono_chrome { 1 } else { 3 }
    }
}

/// `timing_info()` (5.5.3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TimingInfo {
    pub(crate) num_units_in_display_tick: u32,
    pub(crate) time_scale: u32,
    pub(crate) equal_picture_interval: bool,
    pub(crate) num_ticks_per_picture_minus_1: u32,
}

/// `decoder_model_info()` (5.5.4).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DecoderModelInfo {
    pub(crate) buffer_delay_length_minus_1: u8,
    pub(crate) num_units_in_decoding_tick: u32,
    pub(crate) buffer_removal_time_length_minus_1: u8,
    pub(crate) frame_presentation_time_length_minus_1: u8,
}

/// One entry of the operating point loop.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct OperatingPoint {
    pub(crate) idc: u16,
    pub(crate) seq_level_idx: u8,
    pub(crate) seq_tier: bool,
    pub(crate) decoder_model_present: bool,
    pub(crate) decoder_buffer_delay: u32,
    pub(crate) encoder_buffer_delay: u32,
    pub(crate) low_delay_mode_flag: bool,
    pub(crate) initial_display_delay_present: bool,
    pub(crate) initial_display_delay_minus_1: u8,
}

/// `sequence_header_obu()`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SequenceHeader {
    pub(crate) seq_profile: u8,
    pub(crate) still_picture: bool,
    pub(crate) reduced_still_picture_header: bool,
    pub(crate) timing_info: Option<TimingInfo>,
    pub(crate) decoder_model_info: Option<DecoderModelInfo>,
    pub(crate) initial_display_delay_present_flag: bool,
    /// One entry per operating point. It always holds at least one.
    pub(crate) operating_points: Vec<OperatingPoint>,
    pub(crate) frame_width_bits_minus_1: u8,
    pub(crate) frame_height_bits_minus_1: u8,
    pub(crate) max_frame_width_minus_1: u16,
    pub(crate) max_frame_height_minus_1: u16,
    pub(crate) frame_id_numbers_present_flag: bool,
    pub(crate) delta_frame_id_length_minus_2: u8,
    pub(crate) additional_frame_id_length_minus_1: u8,
    pub(crate) use_128x128_superblock: bool,
    pub(crate) enable_filter_intra: bool,
    pub(crate) enable_intra_edge_filter: bool,
    pub(crate) enable_interintra_compound: bool,
    pub(crate) enable_masked_compound: bool,
    pub(crate) enable_warped_motion: bool,
    pub(crate) enable_dual_filter: bool,
    pub(crate) enable_order_hint: bool,
    pub(crate) enable_jnt_comp: bool,
    pub(crate) enable_ref_frame_mvs: bool,
    pub(crate) seq_choose_screen_content_tools: bool,
    pub(crate) seq_force_screen_content_tools: u8,
    pub(crate) seq_choose_integer_mv: bool,
    pub(crate) seq_force_integer_mv: u8,
    /// `OrderHintBits`: 0 when order hints are off.
    pub(crate) order_hint_bits: u8,
    pub(crate) enable_superres: bool,
    pub(crate) enable_cdef: bool,
    pub(crate) enable_restoration: bool,
    pub(crate) color: ColorConfig,
    pub(crate) film_grain_params_present: bool,
}

impl SequenceHeader {
    /// `OperatingPointIdc` of operating point 0, the point this crate selects.
    pub(crate) fn operating_point_idc(&self) -> u16 {
        self.operating_points.first().map_or(0, |point| point.idc)
    }

    /// `idLen` of the frame header.
    pub(crate) const fn frame_id_length(&self) -> u32 {
        self.additional_frame_id_length_minus_1 as u32
            + self.delta_frame_id_length_minus_2 as u32
            + 3
    }

    /// Whether `other` belongs to the same coded video sequence. Section 7.5 lets only
    /// `operating_parameters_info()` differ inside one sequence.
    pub(crate) fn same_sequence(&self, other: &Self) -> bool {
        self.without_operating_parameters() == other.without_operating_parameters()
    }

    fn without_operating_parameters(&self) -> Self {
        let mut copy = self.clone();
        for point in &mut copy.operating_points {
            point.decoder_buffer_delay = 0;
            point.encoder_buffer_delay = 0;
            point.low_delay_mode_flag = false;
        }
        copy
    }
}

/// Parses a sequence header OBU payload.
///
/// # Errors
///
/// [`BitstreamError`] for a payload that ends early or a field out of range.
pub(crate) fn parse_sequence_header(payload: &[u8]) -> Result<SequenceHeader, BitstreamError> {
    let mut reader = BitReader::new(payload, STRUCTURE);
    let mut header = SequenceHeader {
        seq_profile: reader.u8(3, "seq_profile")?,
        ..SequenceHeader::default()
    };
    if header.seq_profile > 2 {
        return Err(reader.invalid("seq_profile", header.seq_profile.into(), "0 to 2"));
    }
    header.still_picture = reader.bit("still_picture")?;
    header.reduced_still_picture_header = reader.bit("reduced_still_picture_header")?;
    if header.reduced_still_picture_header {
        header.operating_points.push(OperatingPoint {
            seq_level_idx: reader.u8(5, "seq_level_idx")?,
            ..OperatingPoint::default()
        });
    } else {
        operating_points(&mut reader, &mut header)?;
    }
    frame_size_limits(&mut reader, &mut header)?;
    header.use_128x128_superblock = reader.bit("use_128x128_superblock")?;
    header.enable_filter_intra = reader.bit("enable_filter_intra")?;
    header.enable_intra_edge_filter = reader.bit("enable_intra_edge_filter")?;
    header.seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS;
    header.seq_force_integer_mv = SELECT_INTEGER_MV;
    if !header.reduced_still_picture_header {
        inter_tools(&mut reader, &mut header)?;
    }
    header.enable_superres = reader.bit("enable_superres")?;
    header.enable_cdef = reader.bit("enable_cdef")?;
    header.enable_restoration = reader.bit("enable_restoration")?;
    header.color = color_config(&mut reader, header.seq_profile)?;
    header.film_grain_params_present = reader.bit("film_grain_params_present")?;
    Ok(header)
}

fn frame_size_limits(
    reader: &mut BitReader<'_>,
    header: &mut SequenceHeader,
) -> Result<(), BitstreamError> {
    header.frame_width_bits_minus_1 = reader.u8(4, "frame_width_bits_minus_1")?;
    header.frame_height_bits_minus_1 = reader.u8(4, "frame_height_bits_minus_1")?;
    let width_bits = u32::from(header.frame_width_bits_minus_1) + 1;
    let height_bits = u32::from(header.frame_height_bits_minus_1) + 1;
    header.max_frame_width_minus_1 = reader.bits(width_bits, "max_frame_width_minus_1")? as u16;
    header.max_frame_height_minus_1 = reader.bits(height_bits, "max_frame_height_minus_1")? as u16;
    if !header.reduced_still_picture_header {
        header.frame_id_numbers_present_flag = reader.bit("frame_id_numbers_present_flag")?;
    }
    if header.frame_id_numbers_present_flag {
        header.delta_frame_id_length_minus_2 = reader.u8(4, "delta_frame_id_length_minus_2")?;
        header.additional_frame_id_length_minus_1 =
            reader.u8(3, "additional_frame_id_length_minus_1")?;
    }
    Ok(())
}

fn inter_tools(
    reader: &mut BitReader<'_>,
    header: &mut SequenceHeader,
) -> Result<(), BitstreamError> {
    header.enable_interintra_compound = reader.bit("enable_interintra_compound")?;
    header.enable_masked_compound = reader.bit("enable_masked_compound")?;
    header.enable_warped_motion = reader.bit("enable_warped_motion")?;
    header.enable_dual_filter = reader.bit("enable_dual_filter")?;
    header.enable_order_hint = reader.bit("enable_order_hint")?;
    if header.enable_order_hint {
        header.enable_jnt_comp = reader.bit("enable_jnt_comp")?;
        header.enable_ref_frame_mvs = reader.bit("enable_ref_frame_mvs")?;
    }
    header.seq_choose_screen_content_tools = reader.bit("seq_choose_screen_content_tools")?;
    if !header.seq_choose_screen_content_tools {
        header.seq_force_screen_content_tools =
            u8::from(reader.bit("seq_force_screen_content_tools")?);
    }
    if header.seq_force_screen_content_tools > 0 {
        header.seq_choose_integer_mv = reader.bit("seq_choose_integer_mv")?;
        if !header.seq_choose_integer_mv {
            header.seq_force_integer_mv = u8::from(reader.bit("seq_force_integer_mv")?);
        }
    }
    if header.enable_order_hint {
        header.order_hint_bits = reader.u8(3, "order_hint_bits_minus_1")? + 1;
    }
    Ok(())
}

/// Reads the timing, decoder model, and operating point fields.
fn operating_points(
    reader: &mut BitReader<'_>,
    header: &mut SequenceHeader,
) -> Result<(), BitstreamError> {
    if reader.bit("timing_info_present_flag")? {
        let mut timing = TimingInfo {
            num_units_in_display_tick: reader.bits(32, "num_units_in_display_tick")? as u32,
            time_scale: reader.bits(32, "time_scale")? as u32,
            equal_picture_interval: reader.bit("equal_picture_interval")?,
            num_ticks_per_picture_minus_1: 0,
        };
        if timing.equal_picture_interval {
            timing.num_ticks_per_picture_minus_1 = reader.uvlc("num_ticks_per_picture_minus_1")?;
        }
        header.timing_info = Some(timing);
        if reader.bit("decoder_model_info_present_flag")? {
            header.decoder_model_info = Some(DecoderModelInfo {
                buffer_delay_length_minus_1: reader.u8(5, "buffer_delay_length_minus_1")?,
                num_units_in_decoding_tick: reader.bits(32, "num_units_in_decoding_tick")? as u32,
                buffer_removal_time_length_minus_1: reader
                    .u8(5, "buffer_removal_time_length_minus_1")?,
                frame_presentation_time_length_minus_1: reader
                    .u8(5, "frame_presentation_time_length_minus_1")?,
            });
        }
    }
    header.initial_display_delay_present_flag = reader.bit("initial_display_delay_present_flag")?;
    let count = reader.bits(5, "operating_points_cnt_minus_1")? + 1;
    for _ in 0..count {
        let point = operating_point(
            reader,
            header.decoder_model_info,
            header.initial_display_delay_present_flag,
        )?;
        header.operating_points.push(point);
    }
    Ok(())
}

fn operating_point(
    reader: &mut BitReader<'_>,
    model: Option<DecoderModelInfo>,
    display_delay: bool,
) -> Result<OperatingPoint, BitstreamError> {
    let mut point = OperatingPoint {
        idc: reader.bits(12, "operating_point_idc")? as u16,
        seq_level_idx: reader.u8(5, "seq_level_idx")?,
        ..OperatingPoint::default()
    };
    point.seq_tier = point.seq_level_idx > 7 && reader.bit("seq_tier")?;
    if let Some(model) = model {
        point.decoder_model_present = reader.bit("decoder_model_present_for_this_op")?;
        if point.decoder_model_present {
            let bits = u32::from(model.buffer_delay_length_minus_1) + 1;
            point.decoder_buffer_delay = reader.bits(bits, "decoder_buffer_delay")? as u32;
            point.encoder_buffer_delay = reader.bits(bits, "encoder_buffer_delay")? as u32;
            point.low_delay_mode_flag = reader.bit("low_delay_mode_flag")?;
        }
    }
    if display_delay {
        point.initial_display_delay_present =
            reader.bit("initial_display_delay_present_for_this_op")?;
        if point.initial_display_delay_present {
            point.initial_display_delay_minus_1 = reader.u8(4, "initial_display_delay_minus_1")?;
        }
    }
    Ok(point)
}

/// Reads `color_config()` (5.5.2).
fn color_config(
    reader: &mut BitReader<'_>,
    seq_profile: u8,
) -> Result<ColorConfig, BitstreamError> {
    let high_bitdepth = reader.bit("high_bitdepth")?;
    let twelve_bit = seq_profile == 2 && high_bitdepth && reader.bit("twelve_bit")?;
    let mono_chrome = seq_profile != 1 && reader.bit("mono_chrome")?;
    let color_description_present_flag = reader.bit("color_description_present_flag")?;
    let colors = if color_description_present_flag {
        (
            reader.u8(8, "color_primaries")?,
            reader.u8(8, "transfer_characteristics")?,
            reader.u8(8, "matrix_coefficients")?,
        )
    } else {
        (2, 2, 2)
    };
    let mut color = ColorConfig {
        high_bitdepth,
        twelve_bit,
        bit_depth: bit_depth(high_bitdepth, twelve_bit),
        mono_chrome,
        color_description_present_flag,
        color_primaries: colors.0,
        transfer_characteristics: colors.1,
        matrix_coefficients: colors.2,
        subsampling_x: true,
        subsampling_y: true,
        ..ColorConfig::default()
    };
    if mono_chrome {
        color.color_range = reader.bit("color_range")?;
        return Ok(color);
    }
    if colors == BT709_SRGB_IDENTITY {
        color.color_range = true;
        color.subsampling_x = false;
        color.subsampling_y = false;
    } else {
        color.color_range = reader.bit("color_range")?;
        (color.subsampling_x, color.subsampling_y) = match seq_profile {
            0 => (true, true),
            1 => (false, false),
            _ if twelve_bit => {
                let x = reader.bit("subsampling_x")?;
                (x, x && reader.bit("subsampling_y")?)
            }
            _ => (true, false),
        };
        if color.subsampling_x && color.subsampling_y {
            color.chroma_sample_position = reader.u8(2, "chroma_sample_position")?;
        }
    }
    color.separate_uv_delta_q = reader.bit("separate_uv_delta_q")?;
    Ok(color)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::test_bits::BitWriter;

    /// A 200x100 main profile header with every optional part: timing, decoder model, two
    /// operating points, frame ids, and a colour description.
    fn full_header() -> Vec<u8> {
        let mut w = BitWriter::new();
        w.u(3, 0).u(1, 0).u(1, 0);
        w.u(1, 1).u(32, 1001).u(32, 30000).u(1, 1);
        // `uvlc` of 4: two zeros, a one, and the two bits of 4 - 3.
        w.u(3, 0b001).u(2, 0b01);
        w.u(1, 1).u(5, 9).u(32, 500).u(5, 7).u(5, 15);
        w.u(1, 1).u(5, 1);
        // Point 0: spatial 0 and 1, temporal 0. Level 8 reads a tier.
        w.u(12, 0x301).u(5, 8).u(1, 1);
        w.u(1, 1).u(10, 77).u(10, 88).u(1, 1);
        w.u(1, 1).u(4, 3);
        // Point 1: level 2 reads no tier.
        w.u(12, 0x101).u(5, 2);
        w.u(1, 0);
        w.u(1, 0);
        w.u(4, 7).u(4, 6).u(8, 199).u(7, 99);
        w.u(1, 1).u(4, 5).u(3, 2);
        w.u(1, 1).u(1, 1).u(1, 0);
        w.u(1, 1).u(1, 0).u(1, 1).u(1, 0);
        w.u(1, 1).u(1, 1).u(1, 0);
        w.u(1, 0).u(1, 1);
        w.u(1, 0).u(1, 1);
        w.u(3, 6);
        w.u(1, 1).u(1, 1).u(1, 0);
        // 10-bit, a colour description, limited range, chroma position 1, one delta.
        w.u(1, 1).u(1, 0).u(1, 1).u(8, 9).u(8, 16).u(8, 9);
        w.u(1, 0).u(2, 1).u(1, 1);
        w.u(1, 1);
        w.bytes()
    }

    #[test]
    fn a_full_header_keeps_each_field() {
        let header = parse_sequence_header(&full_header()).expect("parses");
        assert_eq!(header.seq_profile, 0);
        let timing = header.timing_info.expect("timing");
        assert_eq!(timing.num_units_in_display_tick, 1001);
        assert_eq!(timing.time_scale, 30000);
        assert_eq!(timing.num_ticks_per_picture_minus_1, 4);
        let model = header.decoder_model_info.expect("model");
        assert_eq!(model.buffer_delay_length_minus_1, 9);
        assert_eq!(model.num_units_in_decoding_tick, 500);
        assert_eq!(model.buffer_removal_time_length_minus_1, 7);
        assert_eq!(model.frame_presentation_time_length_minus_1, 15);
        assert_eq!(header.operating_points.len(), 2);
        let first = header.operating_points[0];
        assert_eq!(first.idc, 0x301);
        assert!(first.seq_tier);
        assert_eq!(first.decoder_buffer_delay, 77);
        assert_eq!(first.encoder_buffer_delay, 88);
        assert!(first.low_delay_mode_flag);
        assert_eq!(first.initial_display_delay_minus_1, 3);
        assert_eq!(header.operating_points[1].idc, 0x101);
        assert!(!header.operating_points[1].seq_tier);
        assert_eq!(header.operating_point_idc(), 0x301);
        assert_eq!(header.max_frame_width_minus_1, 199);
        assert_eq!(header.max_frame_height_minus_1, 99);
        assert_eq!(header.frame_id_length(), 10);
        assert!(header.use_128x128_superblock);
        assert!(header.enable_interintra_compound);
        assert_eq!(header.seq_force_screen_content_tools, 1);
        assert_eq!(header.seq_force_integer_mv, 1);
        assert_eq!(header.order_hint_bits, 7);
        assert!(!header.enable_restoration);
        assert_eq!(header.color.bit_depth, 10);
        assert_eq!(header.color.color_primaries, 9);
        assert_eq!(header.color.chroma_sample_position, 1);
        assert!(header.color.separate_uv_delta_q);
        assert!(header.film_grain_params_present);
    }

    #[test]
    fn a_truncation_at_every_byte_is_a_named_error() {
        let bytes = full_header();
        for end in 0..bytes.len() {
            let error = parse_sequence_header(&bytes[..end]).expect_err("truncated");
            assert!(
                matches!(error, BitstreamError::Truncated { .. }),
                "end {end}: {error}"
            );
        }
    }

    #[test]
    fn only_operating_parameters_can_differ_in_one_sequence() {
        let first = parse_sequence_header(&full_header()).expect("parses");
        let mut second = first.clone();
        second.operating_points[0].encoder_buffer_delay = 5;
        assert!(first.same_sequence(&second));
        second.max_frame_width_minus_1 = 7;
        assert!(!first.same_sequence(&second));
    }

    #[test]
    fn a_reduced_header_sets_the_select_values() {
        let mut w = BitWriter::new();
        w.u(3, 0).u(1, 1).u(1, 1).u(5, 4);
        w.u(4, 3).u(4, 3).u(4, 15).u(4, 15);
        w.u(1, 0).u(1, 1).u(1, 1);
        w.u(1, 0).u(1, 1).u(1, 1);
        w.u(1, 0).u(1, 0).u(1, 0).u(1, 0).u(2, 0).u(1, 0);
        w.u(1, 0);
        let header = parse_sequence_header(&w.bytes()).expect("parses");
        assert_eq!(
            header.seq_force_screen_content_tools,
            SELECT_SCREEN_CONTENT_TOOLS
        );
        assert_eq!(header.seq_force_integer_mv, SELECT_INTEGER_MV);
        assert_eq!(header.operating_points.len(), 1);
        assert_eq!(header.operating_points[0].seq_level_idx, 4);
        assert_eq!(header.order_hint_bits, 0);
    }
}
