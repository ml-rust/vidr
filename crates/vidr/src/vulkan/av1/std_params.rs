//! The mapping of one sequence header into `StdVideoAV1SequenceHeader`. It touches no device, so
//! tests read each field back.

use ash::vk::native::{
    StdVideoAV1ColorConfig, StdVideoAV1ColorConfigFlags, StdVideoAV1SequenceHeader,
    StdVideoAV1SequenceHeaderFlags, StdVideoAV1TimingInfo, StdVideoAV1TimingInfoFlags,
};

use crate::bitstream::av1::seq_header::SequenceHeader;
use crate::vulkan::retained::Retained;
use crate::vulkan::zeroed::{AllZero, zeroed};

// SAFETY: integers and bitfield units only.
unsafe impl AllZero for StdVideoAV1ColorConfigFlags {}
// SAFETY: as above.
unsafe impl AllZero for StdVideoAV1TimingInfoFlags {}
// SAFETY: as above.
unsafe impl AllZero for StdVideoAV1SequenceHeaderFlags {}

pub(super) fn std_color(header: &SequenceHeader) -> StdVideoAV1ColorConfig {
    let color = &header.color;
    let mut flags: StdVideoAV1ColorConfigFlags = zeroed();
    flags.set_mono_chrome(color.mono_chrome.into());
    flags.set_color_range(color.color_range.into());
    flags.set_separate_uv_delta_q(color.separate_uv_delta_q.into());
    flags.set_color_description_present_flag(color.color_description_present_flag.into());
    StdVideoAV1ColorConfig {
        flags,
        BitDepth: color.bit_depth,
        subsampling_x: color.subsampling_x.into(),
        subsampling_y: color.subsampling_y.into(),
        reserved1: 0,
        color_primaries: color.color_primaries.into(),
        transfer_characteristics: color.transfer_characteristics.into(),
        matrix_coefficients: color.matrix_coefficients.into(),
        chroma_sample_position: color.chroma_sample_position.into(),
    }
}

/// The timing info, or an all-zero one when the header holds none.
pub(super) fn std_timing(header: &SequenceHeader) -> StdVideoAV1TimingInfo {
    let timing = header.timing_info.unwrap_or_default();
    let mut flags: StdVideoAV1TimingInfoFlags = zeroed();
    flags.set_equal_picture_interval(timing.equal_picture_interval.into());
    StdVideoAV1TimingInfo {
        flags,
        num_units_in_display_tick: timing.num_units_in_display_tick,
        time_scale: timing.time_scale,
        num_ticks_per_picture_minus_1: timing.num_ticks_per_picture_minus_1,
    }
}

/// The sequence header with null pointers. The caller points them at what the functions above
/// build. `pTimingInfo` stays null when the header holds no timing info.
pub(super) fn std_sequence(header: &SequenceHeader) -> StdVideoAV1SequenceHeader {
    let mut flags: StdVideoAV1SequenceHeaderFlags = zeroed();
    flags.set_still_picture(header.still_picture.into());
    flags.set_reduced_still_picture_header(header.reduced_still_picture_header.into());
    flags.set_use_128x128_superblock(header.use_128x128_superblock.into());
    flags.set_enable_filter_intra(header.enable_filter_intra.into());
    flags.set_enable_intra_edge_filter(header.enable_intra_edge_filter.into());
    flags.set_enable_interintra_compound(header.enable_interintra_compound.into());
    flags.set_enable_masked_compound(header.enable_masked_compound.into());
    flags.set_enable_warped_motion(header.enable_warped_motion.into());
    flags.set_enable_dual_filter(header.enable_dual_filter.into());
    flags.set_enable_order_hint(header.enable_order_hint.into());
    flags.set_enable_jnt_comp(header.enable_jnt_comp.into());
    flags.set_enable_ref_frame_mvs(header.enable_ref_frame_mvs.into());
    flags.set_frame_id_numbers_present_flag(header.frame_id_numbers_present_flag.into());
    flags.set_enable_superres(header.enable_superres.into());
    flags.set_enable_cdef(header.enable_cdef.into());
    flags.set_enable_restoration(header.enable_restoration.into());
    flags.set_film_grain_params_present(header.film_grain_params_present.into());
    flags.set_timing_info_present_flag(header.timing_info.is_some().into());
    flags.set_initial_display_delay_present_flag(header.initial_display_delay_present_flag.into());
    StdVideoAV1SequenceHeader {
        flags,
        seq_profile: header.seq_profile.into(),
        frame_width_bits_minus_1: header.frame_width_bits_minus_1,
        frame_height_bits_minus_1: header.frame_height_bits_minus_1,
        max_frame_width_minus_1: header.max_frame_width_minus_1,
        max_frame_height_minus_1: header.max_frame_height_minus_1,
        delta_frame_id_length_minus_2: header.delta_frame_id_length_minus_2,
        additional_frame_id_length_minus_1: header.additional_frame_id_length_minus_1,
        order_hint_bits_minus_1: header.order_hint_bits.saturating_sub(1),
        seq_force_integer_mv: header.seq_force_integer_mv,
        seq_force_screen_content_tools: header.seq_force_screen_content_tools,
        reserved1: [0; 5],
        pColorConfig: std::ptr::null(),
        pTimingInfo: std::ptr::null(),
    }
}

/// `StdVideoAV1SequenceHeader` with what it points to, in stable heap storage. A clone shares each
/// address.
#[derive(Clone)]
pub(in crate::vulkan) struct StdSequence {
    _color: Retained<StdVideoAV1ColorConfig>,
    _timing: Retained<StdVideoAV1TimingInfo>,
    header: Retained<StdVideoAV1SequenceHeader>,
}

impl StdSequence {
    pub(in crate::vulkan) fn new(sequence: &SequenceHeader) -> Self {
        let color = Retained::new(std_color(sequence));
        let timing = Retained::new(std_timing(sequence));
        let mut header = std_sequence(sequence);
        header.pColorConfig = color.as_ptr();
        if sequence.timing_info.is_some() {
            header.pTimingInfo = timing.as_ptr();
        }
        Self {
            _color: color,
            _timing: timing,
            header: Retained::new(header),
        }
    }

    /// The header, valid while `self` lives, across moves too.
    pub(in crate::vulkan) fn as_ptr(&self) -> *const StdVideoAV1SequenceHeader {
        self.header.as_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::av1::seq_header::{ColorConfig, OperatingPoint, TimingInfo};

    fn header() -> SequenceHeader {
        SequenceHeader {
            still_picture: true,
            operating_points: vec![OperatingPoint::default()],
            frame_width_bits_minus_1: 11,
            frame_height_bits_minus_1: 10,
            max_frame_width_minus_1: 1919,
            max_frame_height_minus_1: 1079,
            frame_id_numbers_present_flag: true,
            delta_frame_id_length_minus_2: 3,
            additional_frame_id_length_minus_1: 2,
            use_128x128_superblock: true,
            enable_order_hint: true,
            order_hint_bits: 7,
            seq_force_screen_content_tools: 2,
            seq_force_integer_mv: 1,
            enable_cdef: true,
            film_grain_params_present: true,
            timing_info: Some(TimingInfo {
                num_units_in_display_tick: 1001,
                time_scale: 30000,
                equal_picture_interval: true,
                num_ticks_per_picture_minus_1: 4,
            }),
            color: ColorConfig {
                bit_depth: 10,
                color_primaries: 9,
                transfer_characteristics: 16,
                matrix_coefficients: 9,
                color_description_present_flag: true,
                color_range: true,
                subsampling_x: true,
                subsampling_y: true,
                chroma_sample_position: 2,
                separate_uv_delta_q: true,
                ..ColorConfig::default()
            },
            ..SequenceHeader::default()
        }
    }

    #[test]
    fn each_field_maps_by_name() {
        let source = header();
        let std = std_sequence(&source);
        assert_eq!(std.flags.still_picture(), 1);
        assert_eq!(std.flags.use_128x128_superblock(), 1);
        assert_eq!(std.flags.enable_order_hint(), 1);
        assert_eq!(std.flags.enable_cdef(), 1);
        assert_eq!(std.flags.enable_superres(), 0);
        assert_eq!(std.flags.film_grain_params_present(), 1);
        assert_eq!(std.flags.timing_info_present_flag(), 1);
        assert_eq!(std.flags.frame_id_numbers_present_flag(), 1);
        assert_eq!(std.seq_profile, 0);
        assert_eq!(std.frame_width_bits_minus_1, 11);
        assert_eq!(std.max_frame_width_minus_1, 1919);
        assert_eq!(std.max_frame_height_minus_1, 1079);
        assert_eq!(std.delta_frame_id_length_minus_2, 3);
        assert_eq!(std.additional_frame_id_length_minus_1, 2);
        assert_eq!(std.order_hint_bits_minus_1, 6);
        assert_eq!(std.seq_force_screen_content_tools, 2);
        assert_eq!(std.seq_force_integer_mv, 1);
    }

    #[test]
    fn colour_keeps_each_code_point() {
        let color = std_color(&header());
        assert_eq!(color.BitDepth, 10);
        assert_eq!(color.flags.color_range(), 1);
        assert_eq!(color.flags.mono_chrome(), 0);
        assert_eq!(color.flags.separate_uv_delta_q(), 1);
        assert_eq!(color.flags.color_description_present_flag(), 1);
        assert_eq!((color.subsampling_x, color.subsampling_y), (1, 1));
        assert_eq!(color.color_primaries, 9);
        assert_eq!(color.transfer_characteristics, 16);
        assert_eq!(color.matrix_coefficients, 9);
        assert_eq!(color.chroma_sample_position, 2);
    }

    #[test]
    fn timing_keeps_each_field() {
        let timing = std_timing(&header());
        assert_eq!(timing.flags.equal_picture_interval(), 1);
        assert_eq!(timing.num_units_in_display_tick, 1001);
        assert_eq!(timing.time_scale, 30000);
        assert_eq!(timing.num_ticks_per_picture_minus_1, 4);
    }

    /// The driver reads through these pointers after the create call, so a move keeps them good.
    #[test]
    fn a_moved_sequence_keeps_what_its_pointers_name() {
        let source = header();
        let owner = StdSequence::new(&source);
        let address = owner.as_ptr();
        let moved = [owner];
        assert_eq!(moved[0].as_ptr(), address);
        // SAFETY: `moved` owns the header, its colour config, and its timing info.
        let (std, color, timing) =
            unsafe { (*address, *(*address).pColorConfig, *(*address).pTimingInfo) };
        assert_eq!(std.max_frame_width_minus_1, 1919);
        assert_eq!(color.BitDepth, 10);
        assert_eq!(timing.time_scale, 30000);
    }

    #[test]
    fn a_header_with_no_timing_info_holds_a_null_pointer() {
        let source = SequenceHeader {
            timing_info: None,
            ..header()
        };
        let owner = StdSequence::new(&source);
        // SAFETY: `owner` owns the header, and it still lives.
        let std = unsafe { *owner.as_ptr() };
        assert!(std.pTimingInfo.is_null());
        assert!(!std.pColorConfig.is_null());
    }

    #[test]
    fn a_header_with_no_order_hint_states_zero_bits() {
        let source = SequenceHeader {
            order_hint_bits: 0,
            ..header()
        };
        assert_eq!(std_sequence(&source).order_hint_bits_minus_1, 0);
    }
}
