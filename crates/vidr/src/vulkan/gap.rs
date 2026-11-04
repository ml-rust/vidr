//! Vulkan structures that `ash` 0.38 lacks, written from `vulkan_core.h`.
//!
//! This holds the VP9 decode session's `pNext` layouts and the `StdVideo` picture structures
//! of `vulkan_video_codec_vp9std.h` and `vulkan_video_codec_vp9std_decode.h`. One test checks
//! the `pNext` offsets against values worked by hand from `vulkan_core.h`. The other checks
//! the `StdVideo` offsets against values a C compiler reports for those headers.

#![allow(
    dead_code,
    reason = "the driver reads these fields, and vidr only writes them"
)]

use std::ffi::{CStr, c_void};

use ash::vk;

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

#[cfg(test)]
mod tests {
    use std::mem::{align_of, offset_of, size_of};

    use super::*;

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
