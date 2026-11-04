//! Video capabilities as plain values, and what they allow. Nothing here touches a
//! device, so a test states any driver.

use ash::vk;

use crate::decode::DecodeCodec;
use crate::error::DecodeError;

/// What the driver lists for one video profile.
#[derive(Clone, Copy, Debug)]
pub(super) struct DecodeCaps {
    pub(super) flags: vk::VideoCapabilityFlagsKHR,
    pub(super) decode_flags: vk::VideoDecodeCapabilityFlagsKHR,
    pub(super) size_alignment: u64,
    pub(super) granularity: (u32, u32),
    pub(super) min_coded: (u32, u32),
    pub(super) max_coded: (u32, u32),
    pub(super) max_dpb_slots: u32,
    pub(super) max_active_references: u32,
    pub(super) std_version: vk::ExtensionProperties,
}

/// Where a decoded picture lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum DpbMode {
    /// One image per slot holds the reference and the output.
    Coincide,
    /// DPB images stay on the video queue. One more image holds each output.
    Distinct,
}

/// Which [`DpbMode`] a caller wants. A test forces each mode the driver lists.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DpbPreference {
    /// `Coincide` when the driver lists it, then `Distinct`.
    #[default]
    Auto,
    /// Only `Coincide`.
    Coincide,
    /// Only `Distinct`.
    Distinct,
}

/// The mode for `flags`, or the refusal of `codec` that names what the driver lacks.
pub(super) fn dpb_mode(
    codec: DecodeCodec,
    flags: vk::VideoDecodeCapabilityFlagsKHR,
    preference: DpbPreference,
) -> Result<DpbMode, DecodeError> {
    choose_dpb_mode(flags, preference).map_err(|reason| DecodeError::VulkanRefused {
        codec,
        what: "a DPB mode".to_owned(),
        reason,
    })
}

/// The mode for `flags`, or what the driver lacks.
fn choose_dpb_mode(
    flags: vk::VideoDecodeCapabilityFlagsKHR,
    preference: DpbPreference,
) -> Result<DpbMode, String> {
    let coincide = flags.contains(vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_COINCIDE);
    let distinct = flags.contains(vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_DISTINCT);
    match preference {
        DpbPreference::Auto | DpbPreference::Coincide if coincide => Ok(DpbMode::Coincide),
        DpbPreference::Auto | DpbPreference::Distinct if distinct => Ok(DpbMode::Distinct),
        DpbPreference::Auto => Err(format!("the driver lists neither DPB mode in {flags:?}")),
        DpbPreference::Coincide => Err(format!("the driver lists no coincide mode in {flags:?}")),
        DpbPreference::Distinct => Err(format!("the driver lists no distinct mode in {flags:?}")),
    }
}

/// Whether each DPB slot takes its own image. Otherwise every slot is a layer of one
/// image.
pub(super) fn separate_images(flags: vk::VideoCapabilityFlagsKHR) -> bool {
    flags.contains(vk::VideoCapabilityFlagsKHR::SEPARATE_REFERENCE_IMAGES)
}

/// `value` rounded up to a multiple of `alignment`. An alignment of 0 counts as 1.
pub(super) fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment.max(1)) * alignment.max(1)
}

/// The image size that holds `coded` at the driver's picture access granularity. A 4:2:0
/// image takes even sizes, and a VP9 frame can be odd.
pub(super) fn image_extent(coded: (u32, u32), granularity: (u32, u32)) -> (u32, u32) {
    let side =
        |length: u32, step: u32| (length.div_ceil(step.max(1)) * step.max(1)).next_multiple_of(2);
    (side(coded.0, granularity.0), side(coded.1, granularity.1))
}

/// Refuses a stream outside the driver's limits, by name.
pub(super) fn check_stream(
    caps: &DecodeCaps,
    codec: DecodeCodec,
    coded: (u32, u32),
    slots: u32,
    active_references: u32,
) -> Result<(), DecodeError> {
    let refused = |what: String, reason: String| DecodeError::VulkanRefused {
        codec,
        what,
        reason,
    };
    let (width, height) = coded;
    let (min, max) = (caps.min_coded, caps.max_coded);
    if width < min.0 || height < min.1 || width > max.0 || height > max.1 {
        return Err(refused(
            format!("a coded size of {width}x{height}"),
            format!(
                "the driver lists {}x{} to {}x{}",
                min.0, min.1, max.0, max.1
            ),
        ));
    }
    if slots > caps.max_dpb_slots {
        return Err(refused(
            format!("{slots} DPB slots"),
            format!("the driver lists at most {}", caps.max_dpb_slots),
        ));
    }
    if active_references > caps.max_active_references {
        return Err(refused(
            format!("{active_references} active references"),
            format!("the driver lists at most {}", caps.max_active_references),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> DecodeCaps {
        DecodeCaps {
            flags: vk::VideoCapabilityFlagsKHR::SEPARATE_REFERENCE_IMAGES,
            decode_flags: vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_COINCIDE,
            size_alignment: 256,
            granularity: (16, 16),
            min_coded: (48, 16),
            max_coded: (4096, 4096),
            max_dpb_slots: 17,
            max_active_references: 16,
            std_version: vk::ExtensionProperties::default(),
        }
    }

    #[test]
    fn auto_prefers_coincide_and_a_forced_mode_names_its_absence() {
        let both = vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_COINCIDE
            | vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_DISTINCT;
        let coincide = vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_COINCIDE;
        let distinct = vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_DISTINCT;
        assert_eq!(
            choose_dpb_mode(both, DpbPreference::Auto),
            Ok(DpbMode::Coincide)
        );
        assert_eq!(
            choose_dpb_mode(distinct, DpbPreference::Auto),
            Ok(DpbMode::Distinct)
        );
        assert_eq!(
            choose_dpb_mode(both, DpbPreference::Distinct),
            Ok(DpbMode::Distinct)
        );
        let error = choose_dpb_mode(coincide, DpbPreference::Distinct).expect_err("refused");
        assert!(error.contains("no distinct mode"), "{error}");
        let error = choose_dpb_mode(distinct, DpbPreference::Coincide).expect_err("refused");
        assert!(error.contains("no coincide mode"), "{error}");
        let none = vk::VideoDecodeCapabilityFlagsKHR::empty();
        let error = choose_dpb_mode(none, DpbPreference::Auto).expect_err("refused");
        assert!(error.contains("neither"), "{error}");
    }

    #[test]
    fn the_separate_images_bit_decides_the_slot_layout() {
        assert!(separate_images(
            vk::VideoCapabilityFlagsKHR::SEPARATE_REFERENCE_IMAGES
        ));
        assert!(!separate_images(vk::VideoCapabilityFlagsKHR::empty()));
    }

    #[test]
    fn alignment_rounds_up_and_keeps_a_multiple() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
        assert_eq!(align_up(7, 0), 7);
        assert_eq!(image_extent((320, 192), (16, 16)), (320, 192));
        assert_eq!(image_extent((322, 180), (32, 64)), (352, 192));
        assert_eq!(image_extent((321, 181), (1, 1)), (322, 182));
    }

    #[test]
    fn a_stream_outside_the_limits_is_refused_by_name() {
        let caps = caps();
        let h264 = DecodeCodec::H264;
        assert!(check_stream(&caps, h264, (320, 192), 5, 4).is_ok());
        let text = check_stream(&caps, h264, (8192, 192), 5, 4)
            .expect_err("wide")
            .to_string();
        assert!(text.contains("8192x192"), "{text}");
        assert!(text.contains("48x16 to 4096x4096"), "{text}");
        let text = check_stream(&caps, h264, (32, 16), 5, 4)
            .expect_err("small")
            .to_string();
        assert!(text.contains("32x16"), "{text}");
        let text = check_stream(&caps, h264, (320, 192), 18, 4)
            .expect_err("slots")
            .to_string();
        assert!(text.contains("18 DPB slots"), "{text}");
        let text = check_stream(&caps, h264, (320, 192), 5, 17)
            .expect_err("references")
            .to_string();
        assert!(text.contains("17 active references"), "{text}");
    }
}
