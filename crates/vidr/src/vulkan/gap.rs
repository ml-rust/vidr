//! Vulkan structures that `ash` 0.38 lacks, written from `vulkan_core.h`.
//!
//! This holds the VP9 decode session's `pNext` layouts. The VP9 unit adds the
//! `StdVideo` picture structures behind `VkVideoDecodeVP9PictureInfoKHR`, and
//! `p_std_picture_info` stays an untyped pointer until then.

#![allow(dead_code, reason = "read by the Vulkan VP9 unit")]

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
    /// `const StdVideoDecodeVP9PictureInfo*`.
    pub(super) p_std_picture_info: *const c_void,
    pub(super) reference_name_slot_indices: [i32; MAX_VP9_REFERENCES],
    pub(super) uncompressed_header_offset: u32,
    pub(super) compressed_header_offset: u32,
    pub(super) tiles_offset: u32,
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

    #[test]
    fn structure_types_and_flag_match_the_header() {
        assert_eq!(TYPE_VP9_FEATURES.as_raw(), 1_000_514_000);
        assert_eq!(TYPE_VP9_PROFILE_INFO.as_raw(), 1_000_514_003);
        assert_eq!(DECODE_VP9.as_raw(), 8);
        assert_eq!(VIDEO_DECODE_VP9_NAME.to_bytes(), b"VK_KHR_video_decode_vp9");
    }
}
