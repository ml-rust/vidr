//! Encode capabilities as plain values, and what they allow. Nothing here touches a device, so a
//! test states any driver.

use ash::vk;

use crate::encode::{EncodeCodec, EncodeConfig, Preset};
use crate::error::EncodeError;
use crate::vulkan::caps::align_up;

use super::gop::{ACTIVE_REFERENCES, SLOTS};

/// What each codec adds to the capabilities.
#[derive(Clone, Copy, Debug)]
pub(super) enum CodecCaps {
    H264 {
        std_syntax: vk::VideoEncodeH264StdFlagsKHR,
    },
    Hevc {
        std_syntax: vk::VideoEncodeH265StdFlagsKHR,
        ctb_sizes: vk::VideoEncodeH265CtbSizeFlagsKHR,
        transform_sizes: vk::VideoEncodeH265TransformBlockSizeFlagsKHR,
    },
    Av1 {
        reference_name_mask: u32,
        superblock_sizes: u32,
        max_operating_points: u32,
    },
}

/// What the driver lists for one encode profile.
#[derive(Clone, Copy, Debug)]
pub(super) struct EncodeCaps {
    pub(super) flags: vk::VideoCapabilityFlagsKHR,
    pub(super) size_alignment: u64,
    pub(super) offset_alignment: u64,
    pub(super) granularity: (u32, u32),
    pub(super) min_coded: (u32, u32),
    pub(super) max_coded: (u32, u32),
    pub(super) max_dpb_slots: u32,
    pub(super) max_active_references: u32,
    pub(super) std_version: vk::ExtensionProperties,
    pub(super) rate_modes: vk::VideoEncodeRateControlModeFlagsKHR,
    /// `maxRateControlLayers`. CBR and VBR state one layer.
    pub(super) max_rate_layers: u32,
    pub(super) max_bitrate: u64,
    pub(super) max_quality_levels: u32,
    pub(super) input_granularity: (u32, u32),
    pub(super) feedback: vk::VideoEncodeFeedbackFlagsKHR,
    /// The most references a P picture takes.
    pub(super) max_p_references: u32,
    /// The lowest and highest quantiser: QP for H.264 and HEVC, `q_index` for AV1.
    pub(super) quantiser: (i32, i32),
    /// The highest level, as the codec's Std enum states it.
    pub(super) max_level: u32,
    /// Whether rate control needs each frame's remaining GOP.
    pub(super) requires_gop_remaining: bool,
    pub(super) codec: CodecCaps,
}

/// The frame feedback flags the encoder reads.
pub(super) const FEEDBACK_FLAGS: vk::VideoEncodeFeedbackFlagsKHR =
    vk::VideoEncodeFeedbackFlagsKHR::from_raw(
        vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BUFFER_OFFSET.as_raw()
            | vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BYTES_WRITTEN.as_raw(),
    );

/// `value` rounded up to a multiple of `step`. A step of 0 counts as 1.
fn align_side(value: u32, step: u32) -> u32 {
    align_up(u64::from(value), u64::from(step.max(1))) as u32
}

/// The coded picture's alignment: a macroblock for H.264, the smallest coding block for HEVC, and
/// the driver's granularities. AV1 has no cropping, so its coded size stays the frame size.
pub(super) fn coded_extent(codec: EncodeCodec, size: (u32, u32), caps: &EncodeCaps) -> (u32, u32) {
    let block = match codec {
        EncodeCodec::H264 => 16,
        EncodeCodec::Hevc => 8,
        EncodeCodec::Av1 => return size,
    };
    let side =
        |length: u32, access: u32, input: u32| align_side(length, block.max(access).max(input));
    (
        side(size.0, caps.granularity.0, caps.input_granularity.0),
        side(size.1, caps.granularity.1, caps.input_granularity.1),
    )
}

/// The size of the input image and each DPB image: `coded` at the driver's granularities. The
/// images follow the frame, also when the session opens larger.
pub(super) fn image_extent(coded: (u32, u32), caps: &EncodeCaps) -> (u32, u32) {
    let side = |length: u32, access: u32, input: u32| align_side(length, access.max(input).max(2));
    (
        side(coded.0, caps.granularity.0, caps.input_granularity.0),
        side(coded.1, caps.granularity.1, caps.input_granularity.1),
    )
}

/// The bitstream buffer's size: one raw frame, twice over, past any picture a quantiser allows. A
/// picture that still overruns it takes its named error.
pub(super) fn bitstream_bytes(coded: (u32, u32), bytes_per_sample: u64, caps: &EncodeCaps) -> u64 {
    let raw = u64::from(coded.0) * u64::from(coded.1) * 3 / 2 * bytes_per_sample;
    let wanted = raw.saturating_mul(2).max(1 << 20);
    align_up(wanted, caps.size_alignment.max(caps.offset_alignment))
}

/// `vk::VideoEncodeQualityLevelInfoKHR::qualityLevel` of `preset`, spread over the levels the
/// driver lists. A higher level gives better quality.
pub(super) fn quality_level(preset: Preset, max_levels: u32) -> u32 {
    let step = match preset {
        Preset::P1 => 0,
        Preset::P2 => 1,
        Preset::P3 => 2,
        Preset::P4 => 3,
        Preset::P5 => 4,
        Preset::P6 => 5,
        Preset::P7 => 6,
    };
    step * max_levels.saturating_sub(1) / 6
}

/// Refuses a size or a reference structure outside the driver's limits, by name.
pub(super) fn check_limits(
    config: &EncodeConfig,
    coded: (u32, u32),
    caps: &EncodeCaps,
) -> Result<(), EncodeError> {
    let codec = config.codec;
    let (min, max) = (caps.min_coded, caps.max_coded);
    if coded.0 < min.0 || coded.1 < min.1 || coded.0 > max.0 || coded.1 > max.1 {
        return Err(EncodeError::SizeOutOfRange {
            codec,
            width: config.width,
            height: config.height,
            minimum: min,
            maximum: max,
        });
    }
    let refused = |what: String, reason: String| EncodeError::VulkanRefused {
        codec,
        what,
        reason,
    };
    if caps.max_dpb_slots < SLOTS || caps.max_active_references < ACTIVE_REFERENCES {
        return Err(refused(
            format!("{SLOTS} DPB slots and {ACTIVE_REFERENCES} active reference"),
            format!(
                "the driver lists {} slots and {} active references",
                caps.max_dpb_slots, caps.max_active_references
            ),
        ));
    }
    if caps.max_p_references < 1 {
        return Err(refused(
            "a P picture with one reference".to_owned(),
            "the driver lists no reference for a P picture".to_owned(),
        ));
    }
    if !caps.feedback.contains(FEEDBACK_FLAGS) {
        return Err(refused(
            "bitstream offset and size feedback".to_owned(),
            format!("the driver lists {:?}", caps.feedback),
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(super) mod test_caps {
    use super::*;

    /// The values NVIDIA's driver lists for H.264 High, in shape.
    pub(in super::super) fn h264() -> EncodeCaps {
        EncodeCaps {
            flags: vk::VideoCapabilityFlagsKHR::SEPARATE_REFERENCE_IMAGES,
            size_alignment: 256,
            offset_alignment: 256,
            granularity: (16, 16),
            min_coded: (160, 64),
            max_coded: (4096, 4096),
            max_dpb_slots: 17,
            max_active_references: 16,
            std_version: vk::ExtensionProperties::default(),
            rate_modes: vk::VideoEncodeRateControlModeFlagsKHR::DISABLED
                | vk::VideoEncodeRateControlModeFlagsKHR::CBR
                | vk::VideoEncodeRateControlModeFlagsKHR::VBR,
            max_rate_layers: 1,
            max_bitrate: 120_000_000,
            max_quality_levels: 5,
            input_granularity: (16, 16),
            feedback: FEEDBACK_FLAGS,
            max_p_references: 7,
            quantiser: (0, 51),
            max_level: 15,
            requires_gop_remaining: false,
            codec: CodecCaps::H264 {
                std_syntax: vk::VideoEncodeH264StdFlagsKHR::ENTROPY_CODING_MODE_FLAG_SET
                    | vk::VideoEncodeH264StdFlagsKHR::TRANSFORM_8X8_MODE_FLAG_SET,
            },
        }
    }

    /// The values NVIDIA's driver lists for HEVC Main and Main 10, in shape.
    pub(in super::super) fn hevc() -> EncodeCaps {
        EncodeCaps {
            granularity: (32, 32),
            input_granularity: (32, 32),
            max_coded: (8192, 8192),
            max_quality_levels: 7,
            max_level: 12,
            codec: CodecCaps::Hevc {
                std_syntax: vk::VideoEncodeH265StdFlagsKHR::SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG_SET,
                ctb_sizes: vk::VideoEncodeH265CtbSizeFlagsKHR::TYPE_32,
                transform_sizes: vk::VideoEncodeH265TransformBlockSizeFlagsKHR::TYPE_4
                    | vk::VideoEncodeH265TransformBlockSizeFlagsKHR::TYPE_8
                    | vk::VideoEncodeH265TransformBlockSizeFlagsKHR::TYPE_16
                    | vk::VideoEncodeH265TransformBlockSizeFlagsKHR::TYPE_32,
            },
            ..h264()
        }
    }

    /// An AV1 profile in the shape the extension states. No driver has listed these values.
    pub(in super::super) fn av1() -> EncodeCaps {
        EncodeCaps {
            granularity: (8, 8),
            input_granularity: (8, 8),
            quantiser: (1, 255),
            max_level: 19,
            codec: CodecCaps::Av1 {
                reference_name_mask: 0b0100_0101,
                superblock_sizes: crate::vulkan::gap::AV1_SUPERBLOCK_64,
                max_operating_points: 1,
            },
            ..h264()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_caps::{av1, h264, hevc};
    use super::*;
    use crate::decode::PixelFormat;
    use crate::time::Rational;

    fn config(codec: EncodeCodec, width: u32, height: u32) -> EncodeConfig {
        let rate = Rational::new(30, 1).unwrap_or(Rational::ONE);
        let base = Rational::new(1, 30).unwrap_or(Rational::ONE);
        EncodeConfig::new(codec, width, height, PixelFormat::Nv12, rate, base)
    }

    #[test]
    fn a_coded_size_takes_the_codec_block_and_the_driver_granularity() {
        assert_eq!(
            coded_extent(EncodeCodec::H264, (640, 360), &h264()),
            (640, 368)
        );
        assert_eq!(
            coded_extent(EncodeCodec::Hevc, (640, 360), &hevc()),
            (640, 384)
        );
        let fine = EncodeCaps {
            granularity: (1, 1),
            input_granularity: (1, 1),
            ..hevc()
        };
        assert_eq!(
            coded_extent(EncodeCodec::Hevc, (642, 358), &fine),
            (648, 360)
        );
        let wide = EncodeCaps {
            granularity: (64, 16),
            ..h264()
        };
        assert_eq!(
            coded_extent(EncodeCodec::H264, (640, 360), &wide),
            (640, 368)
        );
        assert_eq!(
            coded_extent(EncodeCodec::H264, (642, 360), &wide),
            (704, 368)
        );
    }

    #[test]
    fn an_av1_coded_size_stays_the_frame_size_and_only_its_images_grow() {
        let caps = av1();
        assert_eq!(
            coded_extent(EncodeCodec::Av1, (642, 362), &caps),
            (642, 362)
        );
        assert_eq!(image_extent((642, 362), &caps), (648, 368));
    }

    #[test]
    fn the_bitstream_buffer_holds_two_raw_frames_at_the_alignment() {
        assert_eq!(bitstream_bytes((1920, 1088), 1, &h264()), 6_266_880);
        assert_eq!(bitstream_bytes((1920, 1088), 2, &h264()), 12_533_760);
        assert_eq!(bitstream_bytes((160, 64), 1, &h264()), 1 << 20);
        let coarse = EncodeCaps {
            size_alignment: 1 << 21,
            ..h264()
        };
        assert_eq!(bitstream_bytes((160, 64), 1, &coarse), 1 << 21);
    }

    #[test]
    fn presets_spread_over_the_listed_quality_levels() {
        assert_eq!(quality_level(Preset::P1, 5), 0);
        assert_eq!(quality_level(Preset::P5, 5), 2);
        assert_eq!(quality_level(Preset::P7, 5), 4);
        assert_eq!(quality_level(Preset::P7, 7), 6);
        assert_eq!(quality_level(Preset::P7, 1), 0);
        assert_eq!(quality_level(Preset::P7, 0), 0);
    }

    #[test]
    fn a_size_outside_the_driver_range_is_refused_by_name() {
        let small = config(EncodeCodec::H264, 128, 64);
        let error = check_limits(&small, (128, 64), &h264()).expect_err("refused");
        assert_eq!(
            error,
            EncodeError::SizeOutOfRange {
                codec: EncodeCodec::H264,
                width: 128,
                height: 64,
                minimum: (160, 64),
                maximum: (4096, 4096),
            }
        );
        let large = config(EncodeCodec::H264, 8192, 4096);
        assert!(check_limits(&large, (8192, 4096), &h264()).is_err());
        let fits = config(EncodeCodec::H264, 160, 64);
        assert!(check_limits(&fits, (160, 64), &h264()).is_ok());
    }

    #[test]
    fn a_driver_with_too_few_slots_or_references_is_refused_by_name() {
        let frame = config(EncodeCodec::H264, 640, 360);
        let slots = EncodeCaps {
            max_dpb_slots: 1,
            ..h264()
        };
        let text = check_limits(&frame, (640, 368), &slots)
            .expect_err("refused")
            .to_string();
        assert!(text.contains("2 DPB slots"), "{text}");
        let references = EncodeCaps {
            max_p_references: 0,
            ..h264()
        };
        let text = check_limits(&frame, (640, 368), &references)
            .expect_err("refused")
            .to_string();
        assert!(text.contains("P picture"), "{text}");
        let feedback = EncodeCaps {
            feedback: vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BYTES_WRITTEN,
            ..h264()
        };
        let text = check_limits(&frame, (640, 368), &feedback)
            .expect_err("refused")
            .to_string();
        assert!(text.contains("feedback"), "{text}");
    }
}
