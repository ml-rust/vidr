//! Vulkan results as decode errors.

use ash::vk;

use crate::error::DecodeError;

/// The error of `call` for `input`.
pub(super) fn vk_error(
    call: &'static str,
    result: vk::Result,
    input: impl Into<String>,
) -> DecodeError {
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
