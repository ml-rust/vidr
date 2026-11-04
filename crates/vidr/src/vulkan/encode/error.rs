//! Errors of the shared Vulkan core as encode errors.

use crate::error::{DecodeError, EncodeError};

/// The core's objects report `DecodeError`. An encoder reports the same facts by their encode names.
impl From<DecodeError> for EncodeError {
    fn from(error: DecodeError) -> Self {
        match error {
            DecodeError::Vulkan {
                call, name, input, ..
            } if name == "TIMEOUT" => Self::VulkanWaitTimeout {
                seconds: crate::vulkan::copy::WAIT.as_secs(),
                input: format!("{call} on {input}"),
            },
            DecodeError::Vulkan {
                call,
                code,
                name,
                input,
            } => Self::Vulkan {
                call,
                code,
                name,
                input,
            },
            DecodeError::VulkanUnavailable { reason } => Self::VulkanUnavailable { reason },
            DecodeError::VulkanCopyTimeout { seconds, reason } => Self::VulkanWaitTimeout {
                seconds,
                input: format!("a copy into the encode input image: {reason}"),
            },
            DecodeError::VulkanNoMemoryType {
                what,
                required,
                type_bits,
            } => Self::VulkanNoMemoryType {
                what,
                required,
                type_bits,
            },
            DecodeError::VulkanDeviceLost { call, input } => Self::VulkanDeviceLost { call, input },
            other => Self::VulkanCore {
                reason: other.to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_core_error_keeps_its_facts_under_an_encode_name() {
        let call = EncodeError::from(DecodeError::Vulkan {
            call: "vkCreateImage",
            code: -2,
            name: "ERROR_OUT_OF_DEVICE_MEMORY".to_owned(),
            input: "a DPB image".to_owned(),
        });
        assert_eq!(
            call,
            EncodeError::Vulkan {
                call: "vkCreateImage",
                code: -2,
                name: "ERROR_OUT_OF_DEVICE_MEMORY".to_owned(),
                input: "a DPB image".to_owned(),
            }
        );
        let lost = EncodeError::from(DecodeError::VulkanDeviceLost {
            call: "vkQueueSubmit2",
            input: "the frame at pts 3".to_owned(),
        });
        assert!(matches!(lost, EncodeError::VulkanDeviceLost { .. }));
        assert!(lost.to_string().contains("lost Vulkan device"), "{lost}");
    }

    #[test]
    fn a_timed_out_wait_has_its_own_name() {
        let timeout = EncodeError::from(DecodeError::Vulkan {
            call: "vkWaitSemaphores",
            code: 2,
            name: "TIMEOUT".to_owned(),
            input: "timeline value 7".to_owned(),
        });
        let text = timeout.to_string();
        assert!(matches!(timeout, EncodeError::VulkanWaitTimeout { .. }));
        assert!(text.contains("timeline value 7"), "{text}");
    }
}
