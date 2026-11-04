//! Vulkan results as decode errors.

use ash::vk;

use crate::error::DecodeError;

/// The decoder's name in errors.
pub(super) const DECODER: &str = "Vulkan Video";

/// The error of a picture at `pts` that this backend cannot decode.
pub(super) fn request_error(pts: i64, reason: String) -> DecodeError {
    DecodeError::VulkanDecodeRequest { pts, reason }
}

/// The error of `call` for `input`. A lost device has its own name.
pub(super) fn vk_error(
    call: &'static str,
    result: vk::Result,
    input: impl Into<String>,
) -> DecodeError {
    if result == vk::Result::ERROR_DEVICE_LOST {
        return DecodeError::VulkanDeviceLost {
            call,
            input: input.into(),
        };
    }
    DecodeError::Vulkan {
        call,
        code: result.as_raw(),
        name: format!("{result:?}"),
        input: input.into(),
    }
}

/// Whether a result states that the driver does not offer what the call names.
pub(super) fn is_refusal(result: vk::Result) -> bool {
    matches!(
        result,
        vk::Result::ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR
            | vk::Result::ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR
            | vk::Result::ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR
            | vk::Result::ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR
            | vk::Result::ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR
            | vk::Result::ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR
            | vk::Result::ERROR_FORMAT_NOT_SUPPORTED
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_result_keeps_its_code_and_name() {
        let error = vk_error(
            "vkCreateBuffer",
            vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            "a 4-byte buffer",
        );
        let DecodeError::Vulkan {
            call,
            code,
            name,
            input,
        } = error
        else {
            panic!("the wrong variant");
        };
        assert_eq!(call, "vkCreateBuffer");
        assert_eq!(code, -1);
        assert_eq!(name, "ERROR_OUT_OF_HOST_MEMORY");
        assert_eq!(input, "a 4-byte buffer");
    }

    #[test]
    fn a_lost_device_has_its_own_name() {
        let error = vk_error(
            "vkQueueSubmit2",
            vk::Result::ERROR_DEVICE_LOST,
            "the picture at pts 4",
        );
        assert_eq!(
            error,
            DecodeError::VulkanDeviceLost {
                call: "vkQueueSubmit2",
                input: "the picture at pts 4".to_owned(),
            }
        );
        let text = error.to_string();
        assert!(text.contains("lost Vulkan device"), "{text}");
        assert!(text.contains("pts 4"), "{text}");
    }

    #[test]
    fn only_capability_results_count_as_refusals() {
        assert!(is_refusal(
            vk::Result::ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR
        ));
        assert!(is_refusal(
            vk::Result::ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR
        ));
        assert!(!is_refusal(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY));
        assert!(!is_refusal(vk::Result::ERROR_DEVICE_LOST));
    }
}
