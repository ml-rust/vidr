//! Video profiles, capability and format queries, the video session with its memory,
//! and session parameters.

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use ash::vk;
use ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN;

use super::caps::DecodeCaps;
use super::device::VulkanDevice;
use super::error::{is_refusal, vk_error};
use super::gap::{self, VideoDecodeVp9Capabilities, VideoDecodeVp9ProfileInfo};
use super::objects::Memory;
use crate::decode::{DecodeCodec, PixelFormat};
use crate::error::DecodeError;

/// A decode profile: 4:2:0, with the codec's own structure behind it.
pub(super) enum VideoProfile {
    /// H.264, progressive pictures, 8 bits.
    H264(vk::VideoDecodeH264ProfileInfoKHR<'static>),
    /// HEVC at 8 or 10 bits.
    Hevc(vk::VideoDecodeH265ProfileInfoKHR<'static>, PixelFormat),
    /// VP9 at 8 or 10 bits. It uses no session parameters.
    Vp9(VideoDecodeVp9ProfileInfo, PixelFormat),
    /// AV1 Main at 8 or 10 bits.
    Av1(vk::VideoDecodeAV1ProfileInfoKHR<'static>, PixelFormat),
}

/// The Vulkan format of an image in `format`.
pub(super) const fn image_format(format: PixelFormat) -> vk::Format {
    match format {
        PixelFormat::Nv12 => vk::Format::G8_B8R8_2PLANE_420_UNORM,
        PixelFormat::P010 => vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
    }
}

impl VideoProfile {
    /// The profile of an H.264 stream with this `profile_idc`.
    pub(super) fn h264(profile_idc: u8) -> Self {
        Self::H264(
            vk::VideoDecodeH264ProfileInfoKHR::default()
                .std_profile_idc(u32::from(profile_idc))
                .picture_layout(vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE),
        )
    }

    /// The profile of a HEVC stream, by `StdVideoH265ProfileIdc`.
    pub(super) fn hevc(std_profile_idc: u32, format: PixelFormat) -> Self {
        Self::Hevc(
            vk::VideoDecodeH265ProfileInfoKHR::default().std_profile_idc(std_profile_idc),
            format,
        )
    }

    /// The profile of a VP9 stream, by `StdVideoVP9Profile`.
    pub(super) const fn vp9(std_profile: u32, format: PixelFormat) -> Self {
        Self::Vp9(
            VideoDecodeVp9ProfileInfo {
                s_type: gap::TYPE_VP9_PROFILE_INFO,
                p_next: ptr::null(),
                std_profile,
            },
            format,
        )
    }

    /// The profile of an AV1 Main stream. `film_grain` is `filmGrainSupport`.
    pub(super) fn av1(film_grain: bool, format: PixelFormat) -> Self {
        Self::Av1(
            vk::VideoDecodeAV1ProfileInfoKHR::default()
                .std_profile(StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN)
                .film_grain_support(film_grain),
            format,
        )
    }

    /// The codec's name in errors.
    pub(super) const fn codec(&self) -> DecodeCodec {
        match self {
            Self::H264(_) => DecodeCodec::H264,
            Self::Hevc(..) => DecodeCodec::Hevc,
            Self::Vp9(..) => DecodeCodec::Vp9,
            Self::Av1(..) => DecodeCodec::Av1,
        }
    }

    /// The sample layout of this profile.
    pub(super) const fn pixel_format(&self) -> PixelFormat {
        match self {
            Self::H264(_) => PixelFormat::Nv12,
            Self::Hevc(_, format) | Self::Vp9(_, format) | Self::Av1(_, format) => *format,
        }
    }

    /// Whether a coding scope of this profile binds session parameters.
    pub(super) const fn uses_parameters(&self) -> bool {
        !matches!(self, Self::Vp9(..))
    }

    /// The profile structure with the codec structure in its chain. The result borrows
    /// `self`, which holds that structure.
    pub(super) fn info(&self) -> vk::VideoProfileInfoKHR<'_> {
        let (operation, next): (_, *const c_void) = match self {
            Self::H264(h264) => (
                vk::VideoCodecOperationFlagsKHR::DECODE_H264,
                ptr::from_ref(h264).cast(),
            ),
            Self::Hevc(hevc, _) => (
                vk::VideoCodecOperationFlagsKHR::DECODE_H265,
                ptr::from_ref(hevc).cast(),
            ),
            Self::Vp9(vp9, _) => (gap::DECODE_VP9, ptr::from_ref(vp9).cast()),
            Self::Av1(av1, _) => (
                vk::VideoCodecOperationFlagsKHR::DECODE_AV1,
                ptr::from_ref(av1).cast(),
            ),
        };
        let depth = match self.pixel_format() {
            PixelFormat::Nv12 => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
            PixelFormat::P010 => vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
        };
        let mut info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(operation)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(depth)
            .chroma_bit_depth(depth);
        info.p_next = next;
        info
    }
}

/// The H.264 capabilities with the highest level the driver lists.
pub(super) fn query_h264_caps(
    device: &VulkanDevice,
    profile: &VideoProfile,
) -> Result<(DecodeCaps, u32), DecodeError> {
    let mut h264 = vk::VideoDecodeH264CapabilitiesKHR::default();
    // SAFETY: `h264` is a whole capability structure with no chain, and it outlives the call.
    let parsed = unsafe { read_caps(device, profile, ptr::from_mut(&mut h264).cast()) }?;
    Ok((parsed, h264.max_level_idc))
}

/// The HEVC capabilities with the highest level the driver lists.
pub(super) fn query_hevc_caps(
    device: &VulkanDevice,
    profile: &VideoProfile,
) -> Result<(DecodeCaps, u32), DecodeError> {
    let mut hevc = vk::VideoDecodeH265CapabilitiesKHR::default();
    // SAFETY: `hevc` is a whole capability structure with no chain, and it outlives the call.
    let parsed = unsafe { read_caps(device, profile, ptr::from_mut(&mut hevc).cast()) }?;
    Ok((parsed, hevc.max_level_idc))
}

/// The VP9 capabilities. A VP9 frame header states no level, so nothing reads `maxLevel`. The
/// query still needs its structure in the chain.
pub(super) fn query_vp9_caps(
    device: &VulkanDevice,
    profile: &VideoProfile,
) -> Result<DecodeCaps, DecodeError> {
    let mut vp9 = VideoDecodeVp9Capabilities {
        s_type: gap::TYPE_VP9_CAPABILITIES,
        p_next: ptr::null_mut(),
        max_level: 0,
    };
    // SAFETY: `vp9` is a whole capability structure with no chain, and it outlives the call.
    unsafe { read_caps(device, profile, ptr::from_mut(&mut vp9).cast()) }
}

/// The AV1 capabilities. vidr checks each frame's size against them, so nothing reads
/// `maxLevel`. A sequence header can state level 31, which names no limits. The query still
/// needs its structure in the chain.
pub(super) fn query_av1_caps(
    device: &VulkanDevice,
    profile: &VideoProfile,
) -> Result<DecodeCaps, DecodeError> {
    let mut av1 = vk::VideoDecodeAV1CapabilitiesKHR::default();
    // SAFETY: `av1` is a whole capability structure with no chain, and it outlives the call.
    unsafe { read_caps(device, profile, ptr::from_mut(&mut av1).cast()) }
}

/// Queries `profile`, with the codec's capability structure in `codec_caps` filled in.
///
/// # Safety
///
/// `codec_caps` points to a writable capability structure of `profile`'s codec, with a valid
/// chain, through the call.
unsafe fn read_caps(
    device: &VulkanDevice,
    profile: &VideoProfile,
    codec_caps: *mut c_void,
) -> Result<DecodeCaps, DecodeError> {
    // Raw pointers, because each structure gets read after the call.
    let mut decode = vk::VideoDecodeCapabilitiesKHR {
        p_next: codec_caps,
        ..Default::default()
    };
    let mut caps = vk::VideoCapabilitiesKHR {
        p_next: ptr::from_mut(&mut decode).cast(),
        ..Default::default()
    };
    let info = profile.info();
    // SAFETY: `info` and `caps` live through the call. The caller states that `codec_caps`
    // does too, and `decode` holds it at the end of the chain.
    let result = unsafe {
        (device
            .video_instance_fns
            .fp()
            .get_physical_device_video_capabilities_khr)(device.physical, &info, &mut caps)
    };
    check_profile(result, profile, "vkGetPhysicalDeviceVideoCapabilitiesKHR")?;
    Ok(DecodeCaps {
        flags: caps.flags,
        decode_flags: decode.flags,
        size_alignment: caps.min_bitstream_buffer_size_alignment,
        granularity: (
            caps.picture_access_granularity.width,
            caps.picture_access_granularity.height,
        ),
        min_coded: (caps.min_coded_extent.width, caps.min_coded_extent.height),
        max_coded: (caps.max_coded_extent.width, caps.max_coded_extent.height),
        max_dpb_slots: caps.max_dpb_slots,
        max_active_references: caps.max_active_reference_pictures,
        std_version: caps.std_header_version,
    })
}

fn check_profile(
    result: vk::Result,
    profile: &VideoProfile,
    call: &'static str,
) -> Result<(), DecodeError> {
    match result {
        vk::Result::SUCCESS => Ok(()),
        refusal if is_refusal(refusal) => Err(DecodeError::VulkanRefused {
            codec: profile.codec(),
            what: format!(
                "the stream's profile at {}-bit 4:2:0",
                profile.pixel_format().bit_depth()
            ),
            reason: format!("{call} returns {refusal:?}"),
        }),
        other => Err(vk_error(call, other, "the stream's profile")),
    }
}

/// One format and tiling the driver lists for the profile's samples at a usage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ImageFormat {
    pub(super) format: vk::Format,
    pub(super) tiling: vk::ImageTiling,
}

/// The fields of one `VkVideoFormatPropertiesKHR` that the choice reads.
#[derive(Clone, Copy, Debug)]
struct ListedFormat {
    format: vk::Format,
    image_type: vk::ImageType,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    mapping: vk::ComponentMapping,
}

/// Whether `mapping` leaves each channel in place. Each entry is either `IDENTITY` or
/// its own channel's name.
fn is_identity(mapping: vk::ComponentMapping) -> bool {
    use vk::ComponentSwizzle as Swizzle;
    let keeps = |swizzle: Swizzle, own: Swizzle| swizzle == Swizzle::IDENTITY || swizzle == own;
    keeps(mapping.r, Swizzle::R)
        && keeps(mapping.g, Swizzle::G)
        && keeps(mapping.b, Swizzle::B)
        && keeps(mapping.a, Swizzle::A)
}

/// The entry for `wanted` at `usage`, from what the driver lists, or why none fits. A plane
/// copy reads the planes as NV12 or P010 lay them out. So a mapping that moves a channel is
/// out. Pure, for tests.
fn choose_format(
    listed: &[ListedFormat],
    usage: vk::ImageUsageFlags,
    wanted: vk::Format,
) -> Result<ImageFormat, String> {
    let same = |entry: &&ListedFormat| {
        entry.format == wanted
            && entry.image_type == vk::ImageType::TYPE_2D
            && entry.usage.contains(usage)
    };
    let matches = |entry: &&ListedFormat| same(entry) && is_identity(entry.mapping);
    let found = listed
        .iter()
        .filter(matches)
        .find(|entry| entry.tiling == vk::ImageTiling::OPTIMAL)
        .or_else(|| listed.iter().find(matches));
    match found {
        Some(entry) => Ok(ImageFormat {
            format: entry.format,
            tiling: entry.tiling,
        }),
        None => Err(match listed.iter().find(same) {
            Some(entry) => format!(
                "the driver lists {wanted:?} only with componentMapping {:?}, and vidr takes \
                 the identity mapping",
                entry.mapping
            ),
            None => format!("the driver lists {} other formats", listed.len()),
        }),
    }
}

/// The call that lists video formats.
pub(super) const FORMAT_CALL: &str = "vkGetPhysicalDeviceVideoFormatPropertiesKHR";

/// The entry for `wanted` at `usage` that the driver lists for a profile, decode or encode. The
/// outer error is the call's result. The inner error states why no entry fits.
pub(super) fn listed_format(
    device: &VulkanDevice,
    info: vk::VideoProfileInfoKHR<'_>,
    usage: vk::ImageUsageFlags,
    wanted: vk::Format,
) -> Result<Result<ImageFormat, String>, vk::Result> {
    let infos = [info];
    let mut list = vk::VideoProfileListInfoKHR::default().profiles(&infos);
    let query = vk::PhysicalDeviceVideoFormatInfoKHR::default()
        .image_usage(usage)
        .push_next(&mut list);
    let query_fn = device
        .video_instance_fns
        .fp()
        .get_physical_device_video_format_properties_khr;
    let mut count = 0_u32;
    // SAFETY: `query` and its chain live through the call, and `count` is writable.
    let result = unsafe { query_fn(device.physical, &query, &mut count, ptr::null_mut()) };
    if result != vk::Result::SUCCESS {
        return Err(result);
    }
    let mut entries = vec![vk::VideoFormatPropertiesKHR::default(); count as usize];
    // SAFETY: `entries` holds `count` initialised elements.
    let result = unsafe { query_fn(device.physical, &query, &mut count, entries.as_mut_ptr()) };
    if result != vk::Result::SUCCESS {
        return Err(result);
    }
    entries.truncate(count as usize);
    let listed: Vec<_> = entries
        .iter()
        .map(|entry| ListedFormat {
            format: entry.format,
            image_type: entry.image_type,
            tiling: entry.image_tiling,
            usage: entry.image_usage_flags,
            mapping: entry.component_mapping,
        })
        .collect();
    Ok(choose_format(&listed, usage, wanted))
}

/// The format of `profile`'s samples at `usage`, or the refusal.
pub(super) fn query_format(
    device: &VulkanDevice,
    profile: &VideoProfile,
    usage: vk::ImageUsageFlags,
) -> Result<ImageFormat, DecodeError> {
    let wanted = image_format(profile.pixel_format());
    let chosen = match listed_format(device, profile.info(), usage, wanted) {
        Ok(chosen) => chosen,
        Err(result) => {
            check_profile(result, profile, FORMAT_CALL)?;
            return Err(vk_error(FORMAT_CALL, result, "the stream's profile"));
        }
    };
    chosen.map_err(|reason| DecodeError::VulkanRefused {
        codec: profile.codec(),
        what: format!("{wanted:?} images with usage {usage:?}"),
        reason,
    })
}

/// The call that creates a video session.
pub(super) const SESSION_CALL: &str = "vkCreateVideoSessionKHR";

/// What a video session opens with.
pub(super) struct SessionSpec<'a> {
    pub(super) profile: &'a VideoProfile,
    pub(super) caps: &'a DecodeCaps,
    pub(super) format: vk::Format,
    pub(super) coded: (u32, u32),
    pub(super) slots: u32,
    pub(super) active_references: u32,
}

/// What any video session opens with, decode or encode.
pub(super) struct RawSessionSpec<'a> {
    pub(super) family: u32,
    pub(super) profile: vk::VideoProfileInfoKHR<'a>,
    pub(super) picture_format: vk::Format,
    pub(super) reference_format: vk::Format,
    pub(super) coded: (u32, u32),
    pub(super) slots: u32,
    pub(super) active_references: u32,
    pub(super) std_version: &'a vk::ExtensionProperties,
}

/// A video session and the memory bound to it.
pub(super) struct VideoSession {
    device: Arc<VulkanDevice>,
    pub(super) raw: vk::VideoSessionKHR,
    memory: Vec<Memory>,
}

impl VideoSession {
    pub(super) fn new(
        device: &Arc<VulkanDevice>,
        spec: &SessionSpec<'_>,
    ) -> Result<Self, DecodeError> {
        let raw = RawSessionSpec {
            family: device.video_family,
            profile: spec.profile.info(),
            picture_format: spec.format,
            reference_format: spec.format,
            coded: spec.coded,
            slots: spec.slots,
            active_references: spec.active_references,
            std_version: &spec.caps.std_version,
        };
        Self::open(device, &raw, |result| {
            check_profile(result, spec.profile, SESSION_CALL)
        })
    }

    /// Opens a session and binds its memory. `check` turns the create call's result into an error.
    pub(super) fn open<E: From<DecodeError>>(
        device: &Arc<VulkanDevice>,
        spec: &RawSessionSpec<'_>,
        check: impl FnOnce(vk::Result) -> Result<(), E>,
    ) -> Result<Self, E> {
        let info = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(spec.family)
            .video_profile(&spec.profile)
            .picture_format(spec.picture_format)
            .max_coded_extent(vk::Extent2D {
                width: spec.coded.0,
                height: spec.coded.1,
            })
            .reference_picture_format(spec.reference_format)
            .max_dpb_slots(spec.slots)
            .max_active_reference_pictures(spec.active_references)
            .std_header_version(spec.std_version);
        let mut raw = vk::VideoSessionKHR::null();
        // SAFETY: `info` and each pointer in it live through the call.
        let result = unsafe {
            (device.video_fns.fp().create_video_session_khr)(
                device.raw.handle(),
                &info,
                ptr::null(),
                &mut raw,
            )
        };
        check(result)?;
        let mut session = Self {
            device: Arc::clone(device),
            raw,
            memory: Vec::new(),
        };
        session.bind_memory()?;
        Ok(session)
    }

    fn bind_memory(&mut self) -> Result<(), DecodeError> {
        let device = Arc::clone(&self.device);
        let fns = device.video_fns.fp();
        let handle = device.raw.handle();
        let call = "vkGetVideoSessionMemoryRequirementsKHR";
        let mut count = 0_u32;
        // SAFETY: `raw` is a live session, and `count` is writable.
        unsafe {
            (fns.get_video_session_memory_requirements_khr)(
                handle,
                self.raw,
                &mut count,
                ptr::null_mut(),
            )
        }
        .result()
        .map_err(|result| vk_error(call, result, "the video session"))?;
        let mut requirements =
            vec![vk::VideoSessionMemoryRequirementsKHR::default(); count as usize];
        // SAFETY: `requirements` holds `count` initialised elements.
        unsafe {
            (fns.get_video_session_memory_requirements_khr)(
                handle,
                self.raw,
                &mut count,
                requirements.as_mut_ptr(),
            )
        }
        .result()
        .map_err(|result| vk_error(call, result, "the video session"))?;
        requirements.truncate(count as usize);

        let mut binds = Vec::with_capacity(requirements.len());
        for requirement in &requirements {
            let memory = Memory::allocate(
                &device,
                requirement.memory_requirements,
                vk::MemoryPropertyFlags::empty(),
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                "the video session",
            )?;
            binds.push(
                vk::BindVideoSessionMemoryInfoKHR::default()
                    .memory_bind_index(requirement.memory_bind_index)
                    .memory(memory.raw)
                    .memory_offset(0)
                    .memory_size(requirement.memory_requirements.size),
            );
            self.memory.push(memory);
        }
        if binds.is_empty() {
            return Ok(());
        }
        // SAFETY: each memory comes from its requirement, and `binds` lives through the
        // call.
        unsafe {
            (fns.bind_video_session_memory_khr)(
                handle,
                self.raw,
                binds.len() as u32,
                binds.as_ptr(),
            )
        }
        .result()
        .map_err(|result| vk_error("vkBindVideoSessionMemoryKHR", result, "the video session"))
    }
}

impl Drop for VideoSession {
    fn drop(&mut self) {
        // SAFETY: this type created the session, and the owner waited for the queue. Its
        // memory goes after this call.
        unsafe {
            (self.device.video_fns.fp().destroy_video_session_khr)(
                self.device.raw.handle(),
                self.raw,
                ptr::null(),
            );
        }
    }
}

/// One session parameters object.
pub(super) struct SessionParameters {
    device: Arc<VulkanDevice>,
    pub(super) raw: vk::VideoSessionParametersKHR,
}

impl SessionParameters {
    /// Creates parameters for `session`. `codec_info` points to the codec's create info.
    ///
    /// # Safety
    ///
    /// `codec_info` points to a valid chain for this session's codec, through the call.
    pub(super) unsafe fn new(
        device: &Arc<VulkanDevice>,
        session: vk::VideoSessionKHR,
        codec_info: *const c_void,
    ) -> Result<Self, DecodeError> {
        let mut info = vk::VideoSessionParametersCreateInfoKHR::default().video_session(session);
        info.p_next = codec_info;
        let mut raw = vk::VideoSessionParametersKHR::null();
        // SAFETY: the caller states the chain is valid, and `session` is live.
        unsafe {
            (device.video_fns.fp().create_video_session_parameters_khr)(
                device.raw.handle(),
                &info,
                ptr::null(),
                &mut raw,
            )
        }
        .result()
        .map_err(|result| {
            vk_error(
                "vkCreateVideoSessionParametersKHR",
                result,
                "the stream's parameter sets",
            )
        })?;
        Ok(Self {
            device: Arc::clone(device),
            raw,
        })
    }
}

impl Drop for SessionParameters {
    fn drop(&mut self) {
        // SAFETY: this type created the parameters, and the owner waited for the queue.
        unsafe {
            (self
                .device
                .video_fns
                .fp()
                .destroy_video_session_parameters_khr)(
                self.device.raw.handle(),
                self.raw,
                ptr::null(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NV12: vk::Format = vk::Format::G8_B8R8_2PLANE_420_UNORM;

    fn listed(
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
    ) -> ListedFormat {
        ListedFormat {
            format,
            image_type: vk::ImageType::TYPE_2D,
            tiling,
            usage,
            mapping: vk::ComponentMapping::default(),
        }
    }

    #[test]
    fn nv12_prefers_optimal_tiling_at_the_needed_usage() {
        let dpb = vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR;
        let wanted = dpb | vk::ImageUsageFlags::TRANSFER_SRC;
        let listed = [
            listed(
                vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
                vk::ImageTiling::OPTIMAL,
                wanted,
            ),
            listed(NV12, vk::ImageTiling::OPTIMAL, dpb),
            listed(NV12, vk::ImageTiling::LINEAR, wanted),
            listed(NV12, vk::ImageTiling::OPTIMAL, wanted),
        ];
        assert_eq!(
            choose_format(&listed, wanted, NV12),
            Ok(ImageFormat {
                format: NV12,
                tiling: vk::ImageTiling::OPTIMAL,
            })
        );
        assert_eq!(
            choose_format(&listed[..3], wanted, NV12).map(|found| found.tiling),
            Ok(vk::ImageTiling::LINEAR)
        );
        assert_eq!(
            choose_format(&listed[..2], wanted, NV12),
            Err("the driver lists 2 other formats".to_owned())
        );
    }

    #[test]
    fn a_mapping_is_identity_only_when_each_channel_stays() {
        use vk::ComponentSwizzle as Swizzle;
        assert!(is_identity(vk::ComponentMapping::default()));
        assert!(is_identity(vk::ComponentMapping {
            r: Swizzle::R,
            g: Swizzle::G,
            b: Swizzle::B,
            a: Swizzle::A,
        }));
        assert!(is_identity(vk::ComponentMapping {
            r: Swizzle::IDENTITY,
            g: Swizzle::G,
            b: Swizzle::IDENTITY,
            a: Swizzle::A,
        }));
        assert!(!is_identity(vk::ComponentMapping {
            r: Swizzle::B,
            b: Swizzle::R,
            ..Default::default()
        }));
        assert!(!is_identity(vk::ComponentMapping {
            g: Swizzle::ONE,
            ..Default::default()
        }));
    }

    #[test]
    fn nv12_with_swapped_channels_is_refused_by_name() {
        let usage = vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR;
        let swapped = ListedFormat {
            mapping: vk::ComponentMapping {
                r: vk::ComponentSwizzle::B,
                b: vk::ComponentSwizzle::R,
                ..Default::default()
            },
            ..listed(NV12, vk::ImageTiling::OPTIMAL, usage)
        };
        let reason = choose_format(&[swapped], usage, NV12).expect_err("refused");
        assert!(reason.contains("componentMapping"), "{reason}");
        assert!(reason.contains("identity"), "{reason}");
        // An identity entry at any tiling beats a swapped optimal one.
        let plain = listed(NV12, vk::ImageTiling::LINEAR, usage);
        assert_eq!(
            choose_format(&[swapped, plain], usage, NV12).map(|found| found.tiling),
            Ok(vk::ImageTiling::LINEAR)
        );
    }

    #[test]
    fn the_h264_profile_chains_its_codec_structure() {
        let profile = VideoProfile::h264(100);
        let info = profile.info();
        assert_eq!(
            info.video_codec_operation,
            vk::VideoCodecOperationFlagsKHR::DECODE_H264
        );
        assert_eq!(
            info.chroma_subsampling,
            vk::VideoChromaSubsamplingFlagsKHR::TYPE_420
        );
        assert_eq!(
            info.luma_bit_depth,
            vk::VideoComponentBitDepthFlagsKHR::TYPE_8
        );
        let VideoProfile::H264(h264) = &profile else {
            panic!("the wrong variant");
        };
        assert_eq!(info.p_next, ptr::from_ref(h264).cast::<c_void>());
        assert_eq!(h264.std_profile_idc, 100);
        assert!(profile.uses_parameters());
    }

    #[test]
    fn a_10_bit_profile_states_its_depth_and_format() {
        let profile = VideoProfile::hevc(2, PixelFormat::P010);
        let info = profile.info();
        assert_eq!(
            info.video_codec_operation,
            vk::VideoCodecOperationFlagsKHR::DECODE_H265
        );
        assert_eq!(
            info.luma_bit_depth,
            vk::VideoComponentBitDepthFlagsKHR::TYPE_10
        );
        assert_eq!(
            info.chroma_bit_depth,
            vk::VideoComponentBitDepthFlagsKHR::TYPE_10
        );
        assert_eq!(
            image_format(profile.pixel_format()),
            vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
        );
    }

    #[test]
    fn a_vp9_profile_chains_its_structure_and_takes_no_parameters() {
        let profile = VideoProfile::vp9(0, PixelFormat::Nv12);
        let info = profile.info();
        assert_eq!(info.video_codec_operation, gap::DECODE_VP9);
        let VideoProfile::Vp9(vp9, _) = &profile else {
            panic!("the wrong variant");
        };
        assert_eq!(info.p_next, ptr::from_ref(vp9).cast::<c_void>());
        assert_eq!(vp9.s_type, gap::TYPE_VP9_PROFILE_INFO);
        assert!(!profile.uses_parameters());
    }

    #[test]
    fn an_av1_profile_chains_its_structure_and_states_film_grain() {
        for grain in [false, true] {
            let profile = VideoProfile::av1(grain, PixelFormat::P010);
            let info = profile.info();
            assert_eq!(
                info.video_codec_operation,
                vk::VideoCodecOperationFlagsKHR::DECODE_AV1
            );
            assert_eq!(
                info.luma_bit_depth,
                vk::VideoComponentBitDepthFlagsKHR::TYPE_10
            );
            let VideoProfile::Av1(av1, _) = &profile else {
                panic!("the wrong variant");
            };
            assert_eq!(info.p_next, ptr::from_ref(av1).cast::<c_void>());
            assert_eq!(av1.std_profile, 0, "STD_VIDEO_AV1_PROFILE_MAIN");
            assert_eq!(av1.film_grain_support, vk::Bool32::from(grain));
            assert!(profile.uses_parameters());
        }
    }

    #[test]
    fn p010_needs_its_own_format_entry() {
        let usage = vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR;
        let p010 = vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16;
        let listed = [listed(NV12, vk::ImageTiling::OPTIMAL, usage)];
        assert_eq!(
            choose_format(&listed, usage, p010),
            Err("the driver lists 1 other formats".to_owned())
        );
    }
}
