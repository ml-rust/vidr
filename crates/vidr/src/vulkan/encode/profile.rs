//! Encode profiles: the codec structure and usage hints behind one `VkVideoProfileInfoKHR`, the
//! capability query, and the format query.

use std::ffi::c_void;
use std::ptr;

use ash::vk;

use crate::decode::PixelFormat;
use crate::encode::{EncodeCodec, EncodeConfig, Tuning};
use crate::error::EncodeError;
use crate::vulkan::device::VulkanDevice;
use crate::vulkan::error::{is_refusal, vk_error};
use crate::vulkan::gap::{self, VideoEncodeAv1Capabilities, VideoEncodeAv1ProfileInfo};
use crate::vulkan::retained::Retained;
use crate::vulkan::session::{FORMAT_CALL, ImageFormat, image_format, listed_format};

use super::caps::{CodecCaps, EncodeCaps};

/// `STD_VIDEO_H264_PROFILE_IDC_HIGH`.
const H264_HIGH: u32 = 100;
/// `STD_VIDEO_H265_PROFILE_IDC_MAIN`.
const HEVC_MAIN: u32 = 1;
/// `STD_VIDEO_H265_PROFILE_IDC_MAIN_10`.
const HEVC_MAIN_10: u32 = 2;
/// `STD_VIDEO_AV1_PROFILE_MAIN`.
const AV1_MAIN: u32 = 0;
/// The call that lists a profile's capabilities.
const CAPS_CALL: &str = "vkGetPhysicalDeviceVideoCapabilitiesKHR";

/// The driver's tuning mode for `tuning`, or `None` for one this backend has no path for.
const fn tuning_mode(tuning: Tuning) -> Option<vk::VideoEncodeTuningModeKHR> {
    match tuning {
        Tuning::HighQuality | Tuning::UltraHighQuality => {
            Some(vk::VideoEncodeTuningModeKHR::HIGH_QUALITY)
        }
        Tuning::LowLatency => Some(vk::VideoEncodeTuningModeKHR::LOW_LATENCY),
        Tuning::UltraLowLatency => Some(vk::VideoEncodeTuningModeKHR::ULTRA_LOW_LATENCY),
        Tuning::Lossless => None,
    }
}

/// The codec structure, with the usage hints in its chain.
enum CodecProfile {
    H264(Retained<vk::VideoEncodeH264ProfileInfoKHR<'static>>),
    Hevc(Retained<vk::VideoEncodeH265ProfileInfoKHR<'static>>),
    Av1(Retained<VideoEncodeAv1ProfileInfo>),
}

/// One encode profile, 4:2:0. The structures keep their addresses across moves.
pub(super) struct EncodeProfile {
    codec: CodecProfile,
    _usage: Retained<vk::VideoEncodeUsageInfoKHR<'static>>,
    format: PixelFormat,
}

impl EncodeProfile {
    /// The profile of `config`.
    ///
    /// # Errors
    ///
    /// [`EncodeError::BitDepthUnsupported`] for 10-bit H.264, and [`EncodeError::VulkanRefused`] for
    /// lossless tuning.
    pub(super) fn new(config: &EncodeConfig) -> Result<Self, EncodeError> {
        let format = config.format;
        let tuning = tuning_mode(config.tuning).ok_or_else(|| EncodeError::VulkanRefused {
            codec: config.codec,
            what: "lossless tuning".to_owned(),
            reason: "this backend states no lossless path".to_owned(),
        })?;
        if config.codec == EncodeCodec::H264 && format == PixelFormat::P010 {
            return Err(EncodeError::BitDepthUnsupported {
                codec: config.codec,
                format,
            });
        }
        let usage = Retained::new(vk::VideoEncodeUsageInfoKHR {
            video_usage_hints: vk::VideoEncodeUsageFlagsKHR::TRANSCODING,
            video_content_hints: vk::VideoEncodeContentFlagsKHR::DEFAULT,
            tuning_mode: tuning,
            ..Default::default()
        });
        let next: *const c_void = usage.as_ptr().cast();
        let codec = match config.codec {
            EncodeCodec::H264 => {
                CodecProfile::H264(Retained::new(vk::VideoEncodeH264ProfileInfoKHR {
                    p_next: next,
                    std_profile_idc: H264_HIGH,
                    ..Default::default()
                }))
            }
            EncodeCodec::Hevc => {
                CodecProfile::Hevc(Retained::new(vk::VideoEncodeH265ProfileInfoKHR {
                    p_next: next,
                    std_profile_idc: match format {
                        PixelFormat::Nv12 => HEVC_MAIN,
                        PixelFormat::P010 => HEVC_MAIN_10,
                    },
                    ..Default::default()
                }))
            }
            EncodeCodec::Av1 => CodecProfile::Av1(Retained::new(VideoEncodeAv1ProfileInfo {
                s_type: gap::TYPE_AV1_ENCODE_PROFILE,
                p_next: next,
                std_profile: AV1_MAIN,
            })),
        };
        Ok(Self {
            codec,
            _usage: usage,
            format,
        })
    }

    pub(super) const fn format(&self) -> PixelFormat {
        self.format
    }

    const fn codec_name(&self) -> EncodeCodec {
        match self.codec {
            CodecProfile::H264(_) => EncodeCodec::H264,
            CodecProfile::Hevc(_) => EncodeCodec::Hevc,
            CodecProfile::Av1(_) => EncodeCodec::Av1,
        }
    }

    /// The profile structure with the codec structure and usage hints in its chain. Each address
    /// stays valid while `self` lives.
    pub(super) fn info(&self) -> vk::VideoProfileInfoKHR<'_> {
        let (operation, next): (_, *const c_void) = match &self.codec {
            CodecProfile::H264(h264) => (
                vk::VideoCodecOperationFlagsKHR::ENCODE_H264,
                h264.as_ptr().cast(),
            ),
            CodecProfile::Hevc(hevc) => (
                vk::VideoCodecOperationFlagsKHR::ENCODE_H265,
                hevc.as_ptr().cast(),
            ),
            CodecProfile::Av1(av1) => (gap::ENCODE_AV1, av1.as_ptr().cast()),
        };
        let depth = match self.format {
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

    /// The error of `result` for this profile. A refusal of a 10-bit profile names the bit depth.
    pub(super) fn check(&self, result: vk::Result, call: &'static str) -> Result<(), EncodeError> {
        let codec = self.codec_name();
        match result {
            vk::Result::SUCCESS => Ok(()),
            refusal if is_refusal(refusal) && self.format == PixelFormat::P010 => {
                Err(EncodeError::BitDepthUnsupported {
                    codec,
                    format: self.format,
                })
            }
            refusal if is_refusal(refusal) => Err(EncodeError::VulkanRefused {
                codec,
                what: format!("a {}-bit 4:2:0 profile", self.format.bit_depth()),
                reason: format!("{call} returns {refusal:?}"),
            }),
            other => Err(vk_error(call, other, "the encode profile").into()),
        }
    }
}

/// Queries `profile`, with the codec's capability structure in `codec_caps` filled in. The result
/// holds the common fields.
///
/// # Safety
///
/// `codec_caps` points to a writable capability structure of `profile`'s codec, with a valid chain,
/// through the call.
unsafe fn read_caps(
    device: &VulkanDevice,
    profile: &EncodeProfile,
    codec_caps: *mut c_void,
) -> Result<EncodeCaps, EncodeError> {
    let mut encode = vk::VideoEncodeCapabilitiesKHR {
        p_next: codec_caps,
        ..Default::default()
    };
    let mut caps = vk::VideoCapabilitiesKHR {
        p_next: ptr::from_mut(&mut encode).cast(),
        ..Default::default()
    };
    let info = profile.info();
    // SAFETY: `info` and `caps` live through the call. The caller states that `codec_caps` does
    // too, and `encode` holds it at the end of the chain.
    let result = unsafe {
        (device
            .video_instance_fns
            .fp()
            .get_physical_device_video_capabilities_khr)(device.physical, &info, &mut caps)
    };
    profile.check(result, CAPS_CALL)?;
    Ok(EncodeCaps {
        flags: caps.flags,
        size_alignment: caps.min_bitstream_buffer_size_alignment,
        offset_alignment: caps.min_bitstream_buffer_offset_alignment,
        granularity: (
            caps.picture_access_granularity.width,
            caps.picture_access_granularity.height,
        ),
        min_coded: (caps.min_coded_extent.width, caps.min_coded_extent.height),
        max_coded: (caps.max_coded_extent.width, caps.max_coded_extent.height),
        max_dpb_slots: caps.max_dpb_slots,
        max_active_references: caps.max_active_reference_pictures,
        std_version: caps.std_header_version,
        rate_modes: encode.rate_control_modes,
        max_rate_layers: encode.max_rate_control_layers,
        max_bitrate: encode.max_bitrate,
        max_quality_levels: encode.max_quality_levels,
        input_granularity: (
            encode.encode_input_picture_granularity.width,
            encode.encode_input_picture_granularity.height,
        ),
        feedback: encode.supported_encode_feedback_flags,
        max_p_references: 0,
        quantiser: (0, 0),
        max_level: 0,
        requires_gop_remaining: false,
        // The caller states what its codec adds.
        codec: CodecCaps::Av1 {
            reference_name_mask: 0,
            superblock_sizes: 0,
            max_operating_points: 0,
        },
    })
}

/// What the driver lists for `profile`.
///
/// # Errors
///
/// [`EncodeError::BitDepthUnsupported`] or [`EncodeError::VulkanRefused`] for a profile the driver
/// leaves out, and [`EncodeError::Vulkan`] for a failed call.
pub(super) fn query_caps(
    device: &VulkanDevice,
    profile: &EncodeProfile,
) -> Result<EncodeCaps, EncodeError> {
    match profile.codec_name() {
        EncodeCodec::H264 => {
            let mut h264 = vk::VideoEncodeH264CapabilitiesKHR::default();
            // SAFETY: `h264` is a whole capability structure with no chain, and it outlives the call.
            let caps = unsafe { read_caps(device, profile, ptr::from_mut(&mut h264).cast()) }?;
            Ok(EncodeCaps {
                max_p_references: h264.max_p_picture_l0_reference_count,
                quantiser: (h264.min_qp, h264.max_qp),
                max_level: h264.max_level_idc,
                requires_gop_remaining: h264.requires_gop_remaining_frames == vk::TRUE,
                codec: CodecCaps::H264 {
                    std_syntax: h264.std_syntax_flags,
                },
                ..caps
            })
        }
        EncodeCodec::Hevc => {
            let mut hevc = vk::VideoEncodeH265CapabilitiesKHR::default();
            // SAFETY: `hevc` is a whole capability structure with no chain, and it outlives the call.
            let caps = unsafe { read_caps(device, profile, ptr::from_mut(&mut hevc).cast()) }?;
            Ok(EncodeCaps {
                max_p_references: hevc.max_p_picture_l0_reference_count,
                quantiser: (hevc.min_qp, hevc.max_qp),
                max_level: hevc.max_level_idc,
                requires_gop_remaining: hevc.requires_gop_remaining_frames == vk::TRUE,
                codec: CodecCaps::Hevc {
                    std_syntax: hevc.std_syntax_flags,
                    ctb_sizes: hevc.ctb_sizes,
                    transform_sizes: hevc.transform_block_sizes,
                },
                ..caps
            })
        }
        EncodeCodec::Av1 => {
            // SAFETY: an all-zero value is valid: integers, an extent of integers, and a null chain.
            let mut av1: VideoEncodeAv1Capabilities = unsafe { std::mem::zeroed() };
            av1.s_type = gap::TYPE_AV1_ENCODE_CAPABILITIES;
            // SAFETY: `av1` is a whole capability structure with no chain, and it outlives the call.
            let caps = unsafe { read_caps(device, profile, ptr::from_mut(&mut av1).cast()) }?;
            let top = i32::try_from(av1.max_q_index).unwrap_or(i32::MAX);
            let bottom = i32::try_from(av1.min_q_index).unwrap_or(top);
            Ok(EncodeCaps {
                max_p_references: av1.max_single_reference_count,
                quantiser: (bottom, top),
                max_level: av1.max_level,
                requires_gop_remaining: av1.requires_gop_remaining_frames == vk::TRUE,
                codec: CodecCaps::Av1 {
                    reference_name_mask: av1.single_reference_name_mask,
                    superblock_sizes: av1.superblock_sizes,
                    max_operating_points: av1.max_operating_points,
                },
                ..caps
            })
        }
    }
}

/// The format of `profile`'s samples at `usage`, or the refusal.
pub(super) fn query_format(
    device: &VulkanDevice,
    profile: &EncodeProfile,
    usage: vk::ImageUsageFlags,
) -> Result<ImageFormat, EncodeError> {
    let wanted = image_format(profile.format);
    let chosen = match listed_format(device, profile.info(), usage, wanted) {
        Ok(chosen) => chosen,
        Err(result) => {
            profile.check(result, FORMAT_CALL)?;
            return Err(vk_error(FORMAT_CALL, result, "the encode profile").into());
        }
    };
    chosen.map_err(|reason| EncodeError::VulkanRefused {
        codec: profile.codec_name(),
        what: format!("{wanted:?} images with usage {usage:?}"),
        reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::Rational;

    fn config(codec: EncodeCodec, format: PixelFormat) -> EncodeConfig {
        let rate = Rational::new(30, 1).unwrap_or(Rational::ONE);
        let base = Rational::new(1, 30).unwrap_or(Rational::ONE);
        EncodeConfig::new(codec, 640, 360, format, rate, base)
    }

    #[test]
    fn each_codec_chains_its_structure_and_the_usage_hints() {
        let profile =
            EncodeProfile::new(&config(EncodeCodec::Hevc, PixelFormat::P010)).expect("the profile");
        let info = profile.info();
        assert_eq!(
            info.video_codec_operation,
            vk::VideoCodecOperationFlagsKHR::ENCODE_H265
        );
        assert_eq!(
            info.luma_bit_depth,
            vk::VideoComponentBitDepthFlagsKHR::TYPE_10
        );
        // SAFETY: `profile` owns the structures of the chain, and it still lives.
        let (hevc, usage) = unsafe {
            let hevc = *info.p_next.cast::<vk::VideoEncodeH265ProfileInfoKHR>();
            let usage = *hevc.p_next.cast::<vk::VideoEncodeUsageInfoKHR>();
            (hevc, usage)
        };
        assert_eq!(hevc.std_profile_idc, HEVC_MAIN_10);
        assert_eq!(
            usage.tuning_mode,
            vk::VideoEncodeTuningModeKHR::HIGH_QUALITY
        );

        let h264 =
            EncodeProfile::new(&config(EncodeCodec::H264, PixelFormat::Nv12)).expect("the profile");
        assert_eq!(
            h264.info().video_codec_operation,
            vk::VideoCodecOperationFlagsKHR::ENCODE_H264
        );
        let av1 =
            EncodeProfile::new(&config(EncodeCodec::Av1, PixelFormat::Nv12)).expect("the profile");
        assert_eq!(av1.info().video_codec_operation, gap::ENCODE_AV1);
    }

    #[test]
    fn ten_bit_h264_and_lossless_tuning_are_refused_by_name() {
        let error = EncodeProfile::new(&config(EncodeCodec::H264, PixelFormat::P010))
            .err()
            .expect("refused");
        assert_eq!(
            error,
            EncodeError::BitDepthUnsupported {
                codec: EncodeCodec::H264,
                format: PixelFormat::P010,
            }
        );
        let mut lossless = config(EncodeCodec::Hevc, PixelFormat::Nv12);
        lossless.tuning = Tuning::Lossless;
        let error = EncodeProfile::new(&lossless).err().expect("refused");
        assert!(error.to_string().contains("lossless"), "{error}");
    }

    #[test]
    fn a_refused_profile_names_the_bit_depth_or_the_codec() {
        let ten =
            EncodeProfile::new(&config(EncodeCodec::Hevc, PixelFormat::P010)).expect("the profile");
        assert_eq!(
            ten.check(
                vk::Result::ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR,
                CAPS_CALL
            ),
            Err(EncodeError::BitDepthUnsupported {
                codec: EncodeCodec::Hevc,
                format: PixelFormat::P010,
            })
        );
        let eight =
            EncodeProfile::new(&config(EncodeCodec::Hevc, PixelFormat::Nv12)).expect("the profile");
        assert!(matches!(
            eight.check(
                vk::Result::ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR,
                CAPS_CALL
            ),
            Err(EncodeError::VulkanRefused { .. })
        ));
        assert_eq!(eight.check(vk::Result::SUCCESS, CAPS_CALL), Ok(()));
        assert!(matches!(
            eight.check(vk::Result::ERROR_DEVICE_LOST, CAPS_CALL),
            Err(EncodeError::VulkanDeviceLost { .. })
        ));
    }
}
