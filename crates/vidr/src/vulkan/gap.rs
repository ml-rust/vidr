//! Vulkan structures that `ash` 0.38 lacks, written from `vulkan_core.h`.
//!
//! This holds the VP9 decode session's `pNext` layouts and the `StdVideo` picture structures
//! of `vulkan_video_codec_vp9std.h` and `vulkan_video_codec_vp9std_decode.h`. It also holds the
//! AV1 encode structures of `vulkan_core.h` and `vulkan_video_codec_av1std_encode.h`. One test
//! checks the VP9 `pNext` offsets against values worked by hand from `vulkan_core.h`. The others
//! check the remaining offsets against values a C compiler reports for those headers.

#![allow(
    dead_code,
    reason = "the driver reads these fields, and vidr only writes them"
)]

use std::ffi::{CStr, c_void};

use ash::vk;
use ash::vk::native::{
    StdVideoAV1CDEF, StdVideoAV1GlobalMotion, StdVideoAV1LoopFilter, StdVideoAV1LoopRestoration,
    StdVideoAV1Quantization, StdVideoAV1Segmentation, StdVideoAV1SequenceHeader,
    StdVideoAV1TileInfo,
};

/// `VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME`.
pub(super) const VIDEO_DECODE_VP9_NAME: &CStr = c"VK_KHR_video_decode_vp9";
/// `VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR`.
pub(super) const DECODE_VP9: vk::VideoCodecOperationFlagsKHR =
    vk::VideoCodecOperationFlagsKHR::from_raw(0x0000_0008);
/// `VK_MAX_VIDEO_VP9_REFERENCES_PER_FRAME_KHR`.
pub(super) const MAX_VP9_REFERENCES: usize = 3;

/// `VkPhysicalDeviceVideoDecodeVP9FeaturesKHR`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct PhysicalDeviceVideoDecodeVp9Features {
    pub(super) s_type: vk::StructureType,
    pub(super) p_next: *mut c_void,
    pub(super) video_decode_vp9: vk::Bool32,
}

/// `VkVideoDecodeVP9ProfileInfoKHR`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct VideoDecodeVp9ProfileInfo {
    pub(super) s_type: vk::StructureType,
    pub(super) p_next: *const c_void,
    /// `StdVideoVP9Profile`.
    pub(super) std_profile: u32,
}

/// `VkVideoDecodeVP9CapabilitiesKHR`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct VideoDecodeVp9Capabilities {
    pub(super) s_type: vk::StructureType,
    pub(super) p_next: *mut c_void,
    /// `StdVideoVP9Level`.
    pub(super) max_level: u32,
}

/// `VkVideoDecodeVP9PictureInfoKHR`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct VideoDecodeVp9PictureInfo {
    pub(super) s_type: vk::StructureType,
    pub(super) p_next: *const c_void,
    pub(super) p_std_picture_info: *const StdVideoDecodeVp9PictureInfo,
    pub(super) reference_name_slot_indices: [i32; MAX_VP9_REFERENCES],
    pub(super) uncompressed_header_offset: u32,
    pub(super) compressed_header_offset: u32,
    pub(super) tiles_offset: u32,
}

// SAFETY: `p_next` stays null and nothing reads through it, so a move to another thread
// carries no shared pointer.
unsafe impl Send for VideoDecodeVp9ProfileInfo {}

/// `StdVideoVP9ColorConfig`. Bit 0 of `flags` is `color_range`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct StdVideoVp9ColorConfig {
    pub(super) flags: u32,
    pub(super) bit_depth: u8,
    pub(super) subsampling_x: u8,
    pub(super) subsampling_y: u8,
    pub(super) reserved1: u8,
    /// `StdVideoVP9ColorSpace`.
    pub(super) color_space: u32,
}

/// `StdVideoVP9LoopFilter`. In `flags`, bit 0 is `loop_filter_delta_enabled` and bit 1 is
/// `loop_filter_delta_update`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct StdVideoVp9LoopFilter {
    pub(super) flags: u32,
    pub(super) loop_filter_level: u8,
    pub(super) loop_filter_sharpness: u8,
    pub(super) update_ref_delta: u8,
    pub(super) loop_filter_ref_deltas: [i8; 4],
    pub(super) update_mode_delta: u8,
    pub(super) loop_filter_mode_deltas: [i8; 2],
}

/// `StdVideoVP9Segmentation`. In `flags`, bits 0 to 3 are `segmentation_update_map`,
/// `segmentation_temporal_update`, `segmentation_update_data`, and
/// `segmentation_abs_or_delta_update`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct StdVideoVp9Segmentation {
    pub(super) flags: u32,
    pub(super) segmentation_tree_probs: [u8; 7],
    pub(super) segmentation_pred_prob: [u8; 3],
    /// One bit per feature, for each segment.
    pub(super) feature_enabled: [u8; 8],
    pub(super) feature_data: [[i16; 4]; 8],
}

/// `StdVideoDecodeVP9PictureInfo`. In `flags`, bits 0 to 7 are `error_resilient_mode`,
/// `intra_only`, `allow_high_precision_mv`, `refresh_frame_context`,
/// `frame_parallel_decoding_mode`, `segmentation_enabled`, `show_frame`, and
/// `UsePrevFrameMvs`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct StdVideoDecodeVp9PictureInfo {
    pub(super) flags: u32,
    /// `StdVideoVP9Profile`.
    pub(super) profile: u32,
    /// `StdVideoVP9FrameType`.
    pub(super) frame_type: u32,
    pub(super) frame_context_idx: u8,
    pub(super) reset_frame_context: u8,
    pub(super) refresh_frame_flags: u8,
    pub(super) ref_frame_sign_bias_mask: u8,
    /// `StdVideoVP9InterpolationFilter`.
    pub(super) interpolation_filter: u32,
    pub(super) base_q_idx: u8,
    pub(super) delta_q_y_dc: i8,
    pub(super) delta_q_uv_dc: i8,
    pub(super) delta_q_uv_ac: i8,
    pub(super) tile_cols_log2: u8,
    pub(super) tile_rows_log2: u8,
    pub(super) reserved1: [u16; 3],
    pub(super) p_color_config: *const StdVideoVp9ColorConfig,
    pub(super) p_loop_filter: *const StdVideoVp9LoopFilter,
    pub(super) p_segmentation: *const StdVideoVp9Segmentation,
}

/// `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_DECODE_VP9_FEATURES_KHR`.
pub(super) const TYPE_VP9_FEATURES: vk::StructureType = vk::StructureType::from_raw(1_000_514_000);
/// `VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_CAPABILITIES_KHR`.
pub(super) const TYPE_VP9_CAPABILITIES: vk::StructureType =
    vk::StructureType::from_raw(1_000_514_001);
/// `VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_PICTURE_INFO_KHR`.
pub(super) const TYPE_VP9_PICTURE_INFO: vk::StructureType =
    vk::StructureType::from_raw(1_000_514_002);
/// `VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_PROFILE_INFO_KHR`.
pub(super) const TYPE_VP9_PROFILE_INFO: vk::StructureType =
    vk::StructureType::from_raw(1_000_514_003);

/// `VK_KHR_VIDEO_ENCODE_AV1_EXTENSION_NAME`.
pub(super) const VIDEO_ENCODE_AV1_NAME: &CStr = c"VK_KHR_video_encode_av1";
/// `VK_VIDEO_CODEC_OPERATION_ENCODE_AV1_BIT_KHR`.
pub(super) const ENCODE_AV1: vk::VideoCodecOperationFlagsKHR =
    vk::VideoCodecOperationFlagsKHR::from_raw(0x0004_0000);
/// `VK_MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR`.
pub(super) const MAX_AV1_REFERENCES: usize = 7;
/// `STD_VIDEO_AV1_PRIMARY_REF_NONE`.
pub(super) const AV1_PRIMARY_REF_NONE: u8 = 7;
/// `VK_VIDEO_ENCODE_AV1_PREDICTION_MODE_INTRA_ONLY_KHR`.
pub(super) const AV1_PREDICTION_INTRA_ONLY: u32 = 0;
/// `VK_VIDEO_ENCODE_AV1_PREDICTION_MODE_SINGLE_REFERENCE_KHR`.
pub(super) const AV1_PREDICTION_SINGLE_REFERENCE: u32 = 1;
/// `VK_VIDEO_ENCODE_AV1_RATE_CONTROL_GROUP_INTRA_KHR`.
pub(super) const AV1_GROUP_INTRA: u32 = 0;
/// `VK_VIDEO_ENCODE_AV1_RATE_CONTROL_GROUP_PREDICTIVE_KHR`.
pub(super) const AV1_GROUP_PREDICTIVE: u32 = 1;
/// `VK_VIDEO_ENCODE_AV1_SUPERBLOCK_SIZE_64_BIT_KHR`.
pub(super) const AV1_SUPERBLOCK_64: u32 = 1;
/// `VK_VIDEO_ENCODE_AV1_RATE_CONTROL_REGULAR_GOP_BIT_KHR`.
pub(super) const AV1_RATE_REGULAR_GOP: u32 = 1;
/// `VK_VIDEO_ENCODE_AV1_RATE_CONTROL_REFERENCE_PATTERN_FLAT_BIT_KHR`.
pub(super) const AV1_RATE_PATTERN_FLAT: u32 = 4;

/// `VkPhysicalDeviceVideoEncodeAV1FeaturesKHR`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct PhysicalDeviceVideoEncodeAv1Features {
    pub(super) s_type: vk::StructureType,
    pub(super) p_next: *mut c_void,
    pub(super) video_encode_av1: vk::Bool32,
}

/// `VkVideoEncodeAV1ProfileInfoKHR`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct VideoEncodeAv1ProfileInfo {
    pub(super) s_type: vk::StructureType,
    pub(super) p_next: *const c_void,
    /// `StdVideoAV1Profile`.
    pub(super) std_profile: u32,
}

/// `VkVideoEncodeAV1CapabilitiesKHR`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct VideoEncodeAv1Capabilities {
    pub(super) s_type: vk::StructureType,
    pub(super) p_next: *mut c_void,
    pub(super) flags: u32,
    /// `StdVideoAV1Level`.
    pub(super) max_level: u32,
    pub(super) coded_picture_alignment: vk::Extent2D,
    pub(super) max_tiles: vk::Extent2D,
    pub(super) min_tile_size: vk::Extent2D,
    pub(super) max_tile_size: vk::Extent2D,
    pub(super) superblock_sizes: u32,
    pub(super) max_single_reference_count: u32,
    pub(super) single_reference_name_mask: u32,
    pub(super) max_unidirectional_compound_reference_count: u32,
    pub(super) max_unidirectional_compound_group1_reference_count: u32,
    pub(super) unidirectional_compound_reference_name_mask: u32,
    pub(super) max_bidirectional_compound_reference_count: u32,
    pub(super) max_bidirectional_compound_group1_reference_count: u32,
    pub(super) max_bidirectional_compound_group2_reference_count: u32,
    pub(super) bidirectional_compound_reference_name_mask: u32,
    pub(super) max_temporal_layer_count: u32,
    pub(super) max_spatial_layer_count: u32,
    pub(super) max_operating_points: u32,
    pub(super) min_q_index: u32,
    pub(super) max_q_index: u32,
    pub(super) prefers_gop_remaining_frames: vk::Bool32,
    pub(super) requires_gop_remaining_frames: vk::Bool32,
    pub(super) std_syntax_flags: u32,
}

/// `VkVideoEncodeAV1SessionParametersCreateInfoKHR`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct VideoEncodeAv1SessionParametersCreateInfo {
    pub(super) s_type: vk::StructureType,
    pub(super) p_next: *const c_void,
    pub(super) p_std_sequence_header: *const StdVideoAV1SequenceHeader,
    /// `const StdVideoEncodeAV1DecoderModelInfo*`, which vidr leaves null.
    pub(super) p_std_decoder_model_info: *const c_void,
    pub(super) std_operating_point_count: u32,
    pub(super) p_std_operating_points: *const StdVideoEncodeAv1OperatingPointInfo,
}

/// `VkVideoEncodeAV1PictureInfoKHR`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct VideoEncodeAv1PictureInfo {
    pub(super) s_type: vk::StructureType,
    pub(super) p_next: *const c_void,
    /// `VkVideoEncodeAV1PredictionModeKHR`.
    pub(super) prediction_mode: u32,
    /// `VkVideoEncodeAV1RateControlGroupKHR`.
    pub(super) rate_control_group: u32,
    pub(super) constant_q_index: u32,
    pub(super) p_std_picture_info: *const StdVideoEncodeAv1PictureInfo,
    pub(super) reference_name_slot_indices: [i32; MAX_AV1_REFERENCES],
    pub(super) primary_reference_cdf_only: vk::Bool32,
    pub(super) generate_obu_extension_header: vk::Bool32,
}

/// `VkVideoEncodeAV1DpbSlotInfoKHR`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct VideoEncodeAv1DpbSlotInfo {
    pub(super) s_type: vk::StructureType,
    pub(super) p_next: *const c_void,
    pub(super) p_std_reference_info: *const StdVideoEncodeAv1ReferenceInfo,
}

/// `VkVideoEncodeAV1RateControlInfoKHR`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct VideoEncodeAv1RateControlInfo {
    pub(super) s_type: vk::StructureType,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) gop_frame_count: u32,
    pub(super) key_frame_period: u32,
    pub(super) consecutive_bipredictive_frame_count: u32,
    pub(super) temporal_layer_count: u32,
}

/// `StdVideoEncodeAV1OperatingPointInfo`. `flags` holds three bits vidr leaves at zero.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct StdVideoEncodeAv1OperatingPointInfo {
    pub(super) flags: u32,
    pub(super) operating_point_idc: u16,
    pub(super) seq_level_idx: u8,
    pub(super) seq_tier: u8,
    pub(super) decoder_buffer_delay: u32,
    pub(super) encoder_buffer_delay: u32,
    pub(super) initial_display_delay_minus_1: u8,
}

/// `show_frame` of `StdVideoEncodeAV1PictureInfoFlags`, bit 27 of its word.
pub(super) const AV1_PICTURE_SHOW_FRAME: u32 = 1 << 27;

/// `StdVideoEncodeAV1PictureInfo`. `flags` holds the header's bit field as one word.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct StdVideoEncodeAv1PictureInfo {
    pub(super) flags: u32,
    /// `StdVideoAV1FrameType`.
    pub(super) frame_type: u32,
    pub(super) frame_presentation_time: u32,
    pub(super) current_frame_id: u32,
    pub(super) order_hint: u8,
    pub(super) primary_ref_frame: u8,
    pub(super) refresh_frame_flags: u8,
    pub(super) coded_denom: u8,
    pub(super) render_width_minus_1: u16,
    pub(super) render_height_minus_1: u16,
    /// `StdVideoAV1InterpolationFilter`.
    pub(super) interpolation_filter: u32,
    /// `StdVideoAV1TxMode`.
    pub(super) tx_mode: u32,
    pub(super) delta_q_res: u8,
    pub(super) delta_lf_res: u8,
    pub(super) ref_order_hint: [u8; 8],
    pub(super) ref_frame_idx: [i8; 7],
    pub(super) reserved1: [u8; 3],
    pub(super) delta_frame_id_minus_1: [u32; 7],
    pub(super) p_tile_info: *const StdVideoAV1TileInfo,
    pub(super) p_quantization: *const StdVideoAV1Quantization,
    pub(super) p_segmentation: *const StdVideoAV1Segmentation,
    pub(super) p_loop_filter: *const StdVideoAV1LoopFilter,
    pub(super) p_cdef: *const StdVideoAV1CDEF,
    pub(super) p_loop_restoration: *const StdVideoAV1LoopRestoration,
    pub(super) p_global_motion: *const StdVideoAV1GlobalMotion,
    /// `const StdVideoEncodeAV1ExtensionHeader*`, which vidr leaves null.
    pub(super) p_extension_header: *const c_void,
    pub(super) p_buffer_removal_times: *const u32,
}

/// `StdVideoEncodeAV1ReferenceInfo`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct StdVideoEncodeAv1ReferenceInfo {
    pub(super) flags: u32,
    pub(super) ref_frame_id: u32,
    /// `StdVideoAV1FrameType`.
    pub(super) frame_type: u32,
    pub(super) order_hint: u8,
    pub(super) reserved1: [u8; 3],
    /// `const StdVideoEncodeAV1ExtensionHeader*`, which vidr leaves null.
    pub(super) p_extension_header: *const c_void,
}

/// `VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_CAPABILITIES_KHR`.
pub(super) const TYPE_AV1_ENCODE_CAPABILITIES: vk::StructureType =
    vk::StructureType::from_raw(1_000_513_000);
/// `VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_SESSION_PARAMETERS_CREATE_INFO_KHR`.
pub(super) const TYPE_AV1_ENCODE_PARAMETERS: vk::StructureType =
    vk::StructureType::from_raw(1_000_513_001);
/// `VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_PICTURE_INFO_KHR`.
pub(super) const TYPE_AV1_ENCODE_PICTURE: vk::StructureType =
    vk::StructureType::from_raw(1_000_513_002);
/// `VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_DPB_SLOT_INFO_KHR`.
pub(super) const TYPE_AV1_ENCODE_SLOT: vk::StructureType =
    vk::StructureType::from_raw(1_000_513_003);
/// `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_ENCODE_AV1_FEATURES_KHR`.
pub(super) const TYPE_AV1_ENCODE_FEATURES: vk::StructureType =
    vk::StructureType::from_raw(1_000_513_004);
/// `VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_PROFILE_INFO_KHR`.
pub(super) const TYPE_AV1_ENCODE_PROFILE: vk::StructureType =
    vk::StructureType::from_raw(1_000_513_005);
/// `VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_RATE_CONTROL_LAYER_INFO_KHR`.
pub(super) const TYPE_AV1_ENCODE_RATE_LAYER: vk::StructureType =
    vk::StructureType::from_raw(1_000_513_007);
/// `VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_RATE_CONTROL_INFO_KHR`.
pub(super) const TYPE_AV1_ENCODE_RATE: vk::StructureType =
    vk::StructureType::from_raw(1_000_513_006);

/// `VkVideoEncodeAV1RateControlLayerInfoKHR`, with every limit off.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct VideoEncodeAv1RateControlLayerInfo {
    s_type: vk::StructureType,
    p_next: *const c_void,
    use_min_q_index: vk::Bool32,
    min_q_index: [u32; 3],
    use_max_q_index: vk::Bool32,
    max_q_index: [u32; 3],
    use_max_frame_size: vk::Bool32,
    max_frame_size: [u32; 3],
}

impl VideoEncodeAv1RateControlLayerInfo {
    pub(super) const fn new() -> Self {
        Self {
            s_type: TYPE_AV1_ENCODE_RATE_LAYER,
            p_next: std::ptr::null(),
            use_min_q_index: 0,
            min_q_index: [0; 3],
            use_max_q_index: 0,
            max_q_index: [0; 3],
            use_max_frame_size: 0,
            max_frame_size: [0; 3],
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::{align_of, offset_of, size_of};

    use super::*;

    /// Offsets a C compiler reports for `vulkan_core.h`.
    #[test]
    fn av1_encode_layouts_match_the_header() {
        assert_eq!(size_of::<PhysicalDeviceVideoEncodeAv1Features>(), 24);
        assert_eq!(
            offset_of!(PhysicalDeviceVideoEncodeAv1Features, video_encode_av1),
            16
        );
        assert_eq!(size_of::<VideoEncodeAv1ProfileInfo>(), 24);
        assert_eq!(size_of::<VideoEncodeAv1Capabilities>(), 128);
        assert_eq!(offset_of!(VideoEncodeAv1Capabilities, max_level), 20);
        assert_eq!(offset_of!(VideoEncodeAv1Capabilities, superblock_sizes), 56);
        assert_eq!(
            offset_of!(VideoEncodeAv1Capabilities, max_single_reference_count),
            60
        );
        assert_eq!(
            offset_of!(VideoEncodeAv1Capabilities, single_reference_name_mask),
            64
        );
        assert_eq!(
            offset_of!(VideoEncodeAv1Capabilities, max_operating_points),
            104
        );
        assert_eq!(offset_of!(VideoEncodeAv1Capabilities, min_q_index), 108);
        assert_eq!(offset_of!(VideoEncodeAv1Capabilities, max_q_index), 112);
        assert_eq!(
            offset_of!(VideoEncodeAv1Capabilities, requires_gop_remaining_frames),
            120
        );
        assert_eq!(
            offset_of!(VideoEncodeAv1Capabilities, std_syntax_flags),
            124
        );
        assert_eq!(size_of::<VideoEncodeAv1SessionParametersCreateInfo>(), 48);
        assert_eq!(
            offset_of!(
                VideoEncodeAv1SessionParametersCreateInfo,
                std_operating_point_count
            ),
            32
        );
        assert_eq!(
            offset_of!(
                VideoEncodeAv1SessionParametersCreateInfo,
                p_std_operating_points
            ),
            40
        );
        assert_eq!(size_of::<VideoEncodeAv1PictureInfo>(), 80);
        assert_eq!(offset_of!(VideoEncodeAv1PictureInfo, constant_q_index), 24);
        assert_eq!(
            offset_of!(VideoEncodeAv1PictureInfo, p_std_picture_info),
            32
        );
        assert_eq!(
            offset_of!(VideoEncodeAv1PictureInfo, reference_name_slot_indices),
            40
        );
        assert_eq!(
            offset_of!(VideoEncodeAv1PictureInfo, primary_reference_cdf_only),
            68
        );
        assert_eq!(
            offset_of!(VideoEncodeAv1PictureInfo, generate_obu_extension_header),
            72
        );
        assert_eq!(size_of::<VideoEncodeAv1DpbSlotInfo>(), 24);
        assert_eq!(size_of::<VideoEncodeAv1RateControlInfo>(), 40);
        assert_eq!(
            offset_of!(VideoEncodeAv1RateControlInfo, temporal_layer_count),
            32
        );
    }

    /// Offsets a C compiler reports for `vulkan_video_codec_av1std_encode.h`.
    #[test]
    fn av1_encode_std_layouts_match_the_header() {
        assert_eq!(size_of::<StdVideoEncodeAv1OperatingPointInfo>(), 20);
        assert_eq!(
            offset_of!(StdVideoEncodeAv1OperatingPointInfo, seq_level_idx),
            6
        );
        assert_eq!(
            offset_of!(
                StdVideoEncodeAv1OperatingPointInfo,
                initial_display_delay_minus_1
            ),
            16
        );
        assert_eq!(size_of::<StdVideoEncodeAv1PictureInfo>(), 152);
        assert_eq!(align_of::<StdVideoEncodeAv1PictureInfo>(), 8);
        assert_eq!(offset_of!(StdVideoEncodeAv1PictureInfo, order_hint), 16);
        assert_eq!(
            offset_of!(StdVideoEncodeAv1PictureInfo, render_width_minus_1),
            20
        );
        assert_eq!(
            offset_of!(StdVideoEncodeAv1PictureInfo, interpolation_filter),
            24
        );
        assert_eq!(offset_of!(StdVideoEncodeAv1PictureInfo, tx_mode), 28);
        assert_eq!(offset_of!(StdVideoEncodeAv1PictureInfo, ref_order_hint), 34);
        assert_eq!(offset_of!(StdVideoEncodeAv1PictureInfo, ref_frame_idx), 42);
        assert_eq!(
            offset_of!(StdVideoEncodeAv1PictureInfo, delta_frame_id_minus_1),
            52
        );
        assert_eq!(offset_of!(StdVideoEncodeAv1PictureInfo, p_tile_info), 80);
        assert_eq!(offset_of!(StdVideoEncodeAv1PictureInfo, p_quantization), 88);
        assert_eq!(
            offset_of!(StdVideoEncodeAv1PictureInfo, p_global_motion),
            128
        );
        assert_eq!(
            offset_of!(StdVideoEncodeAv1PictureInfo, p_buffer_removal_times),
            144
        );
        assert_eq!(size_of::<StdVideoEncodeAv1ReferenceInfo>(), 24);
        assert_eq!(offset_of!(StdVideoEncodeAv1ReferenceInfo, order_hint), 12);
        assert_eq!(
            offset_of!(StdVideoEncodeAv1ReferenceInfo, p_extension_header),
            16
        );
    }

    /// Offsets hand-worked from `vulkan_core.h` for a 64-bit target.
    #[test]
    fn layouts_match_the_header() {
        assert_eq!(size_of::<PhysicalDeviceVideoDecodeVp9Features>(), 24);
        assert_eq!(
            offset_of!(PhysicalDeviceVideoDecodeVp9Features, video_decode_vp9),
            16
        );
        assert_eq!(size_of::<VideoDecodeVp9ProfileInfo>(), 24);
        assert_eq!(offset_of!(VideoDecodeVp9ProfileInfo, std_profile), 16);
        assert_eq!(size_of::<VideoDecodeVp9Capabilities>(), 24);
        assert_eq!(offset_of!(VideoDecodeVp9Capabilities, max_level), 16);
        assert_eq!(size_of::<VideoDecodeVp9PictureInfo>(), 48);
        assert_eq!(align_of::<VideoDecodeVp9PictureInfo>(), 8);
        assert_eq!(
            offset_of!(VideoDecodeVp9PictureInfo, p_std_picture_info),
            16
        );
        assert_eq!(
            offset_of!(VideoDecodeVp9PictureInfo, reference_name_slot_indices),
            24
        );
        assert_eq!(
            offset_of!(VideoDecodeVp9PictureInfo, uncompressed_header_offset),
            36
        );
        assert_eq!(
            offset_of!(VideoDecodeVp9PictureInfo, compressed_header_offset),
            40
        );
        assert_eq!(offset_of!(VideoDecodeVp9PictureInfo, tiles_offset), 44);
    }

    /// Offsets a C compiler reports for `vulkan_video_codec_vp9std*.h`.
    #[test]
    fn std_layouts_match_the_headers() {
        assert_eq!(size_of::<StdVideoVp9ColorConfig>(), 12);
        assert_eq!(offset_of!(StdVideoVp9ColorConfig, bit_depth), 4);
        assert_eq!(offset_of!(StdVideoVp9ColorConfig, color_space), 8);
        assert_eq!(size_of::<StdVideoVp9LoopFilter>(), 16);
        assert_eq!(offset_of!(StdVideoVp9LoopFilter, loop_filter_ref_deltas), 7);
        assert_eq!(offset_of!(StdVideoVp9LoopFilter, update_mode_delta), 11);
        assert_eq!(
            offset_of!(StdVideoVp9LoopFilter, loop_filter_mode_deltas),
            12
        );
        assert_eq!(size_of::<StdVideoVp9Segmentation>(), 88);
        assert_eq!(
            offset_of!(StdVideoVp9Segmentation, segmentation_pred_prob),
            11
        );
        assert_eq!(offset_of!(StdVideoVp9Segmentation, feature_enabled), 14);
        assert_eq!(offset_of!(StdVideoVp9Segmentation, feature_data), 22);
        assert_eq!(size_of::<StdVideoDecodeVp9PictureInfo>(), 56);
        assert_eq!(align_of::<StdVideoDecodeVp9PictureInfo>(), 8);
        assert_eq!(offset_of!(StdVideoDecodeVp9PictureInfo, profile), 4);
        assert_eq!(offset_of!(StdVideoDecodeVp9PictureInfo, frame_type), 8);
        assert_eq!(
            offset_of!(StdVideoDecodeVp9PictureInfo, frame_context_idx),
            12
        );
        assert_eq!(
            offset_of!(StdVideoDecodeVp9PictureInfo, ref_frame_sign_bias_mask),
            15
        );
        assert_eq!(
            offset_of!(StdVideoDecodeVp9PictureInfo, interpolation_filter),
            16
        );
        assert_eq!(offset_of!(StdVideoDecodeVp9PictureInfo, base_q_idx), 20);
        assert_eq!(offset_of!(StdVideoDecodeVp9PictureInfo, tile_cols_log2), 24);
        assert_eq!(offset_of!(StdVideoDecodeVp9PictureInfo, p_color_config), 32);
        assert_eq!(offset_of!(StdVideoDecodeVp9PictureInfo, p_loop_filter), 40);
        assert_eq!(offset_of!(StdVideoDecodeVp9PictureInfo, p_segmentation), 48);
    }

    #[test]
    fn structure_types_and_flag_match_the_header() {
        assert_eq!(TYPE_VP9_FEATURES.as_raw(), 1_000_514_000);
        assert_eq!(TYPE_VP9_PROFILE_INFO.as_raw(), 1_000_514_003);
        assert_eq!(DECODE_VP9.as_raw(), 8);
        assert_eq!(VIDEO_DECODE_VP9_NAME.to_bytes(), b"VK_KHR_video_decode_vp9");
    }
}
