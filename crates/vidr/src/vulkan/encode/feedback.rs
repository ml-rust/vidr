//! The encode feedback query: where a picture's bytes landed, how many, and its status.

use std::ptr;
use std::sync::Arc;

use ash::vk;

use crate::error::EncodeError;
use crate::vulkan::device::VulkanDevice;
use crate::vulkan::engine::{STATUS_COMPLETE, status_name};
use crate::vulkan::error::vk_error;

use super::caps::FEEDBACK_FLAGS;

/// `VK_QUERY_RESULT_STATUS_INSUFFICIENT_BITSTREAM_BUFFER_RANGE_KHR`.
const STATUS_OVERRUN: i32 = -1_000_299_000;

/// One query's result: the two flags in their bit order, then the status.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct FeedbackWords {
    pub(super) offset: u32,
    pub(super) written: u32,
    pub(super) status: i32,
}

/// The byte range of a finished picture, or the error that names why it has none.
pub(super) fn picture_range(
    words: FeedbackWords,
    capacity: u64,
    pts: i64,
) -> Result<(u64, usize), EncodeError> {
    match words.status {
        STATUS_COMPLETE => {}
        STATUS_OVERRUN => return Err(EncodeError::VulkanBitstreamOverrun { pts, capacity }),
        other => {
            return Err(EncodeError::VulkanEncodeStatus {
                status: other,
                name: status_name(other),
                pts,
            });
        }
    }
    let (offset, written) = (u64::from(words.offset), u64::from(words.written));
    if written == 0 || offset.saturating_add(written) > capacity {
        return Err(EncodeError::VulkanFeedbackRange {
            pts,
            offset: words.offset,
            written: words.written,
            capacity,
        });
    }
    Ok((offset, words.written as usize))
}

/// A query pool of one encode feedback query.
pub(super) struct FeedbackQuery {
    device: Arc<VulkanDevice>,
    pub(super) raw: vk::QueryPool,
}

impl FeedbackQuery {
    pub(super) fn new(
        device: &Arc<VulkanDevice>,
        profile: &vk::VideoProfileInfoKHR<'_>,
    ) -> Result<Self, EncodeError> {
        let feedback = vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR {
            p_next: ptr::from_ref(profile).cast(),
            encode_feedback_flags: FEEDBACK_FLAGS,
            ..Default::default()
        };
        let mut info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::VIDEO_ENCODE_FEEDBACK_KHR)
            .query_count(1);
        // `push_next` writes through the whole chain, and a shared reference holds the codec
        // structure at its end.
        info.p_next = ptr::from_ref(&feedback).cast();
        // SAFETY: `info`, `feedback`, and the profile chain live through the call.
        let raw = unsafe { device.raw.create_query_pool(&info, None) }
            .map_err(|result| vk_error("vkCreateQueryPool", result, "one feedback query"))?;
        Ok(Self {
            device: Arc::clone(device),
            raw,
        })
    }

    /// The result of the query, after its submit finished.
    pub(super) fn read(&self, pts: i64) -> Result<FeedbackWords, EncodeError> {
        let mut words = [FeedbackWords::default(); 1];
        // SAFETY: the pool holds one query with two feedback values, and its submit finished.
        // `FeedbackWords` holds those values and the status word, so its size is the stride.
        unsafe {
            self.device.raw.get_query_pool_results(
                self.raw,
                0,
                &mut words,
                vk::QueryResultFlags::WAIT | vk::QueryResultFlags::WITH_STATUS_KHR,
            )
        }
        .map_err(|result| {
            vk_error(
                "vkGetQueryPoolResults",
                result,
                format!("the frame at pts {pts}"),
            )
        })?;
        Ok(words[0])
    }
}

impl Drop for FeedbackQuery {
    fn drop(&mut self) {
        // SAFETY: this type created the pool, and the owner waited for the queue.
        unsafe { self.device.raw.destroy_query_pool(self.raw, None) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(status: i32) -> FeedbackWords {
        FeedbackWords {
            offset: 16,
            written: 1000,
            status,
        }
    }

    #[test]
    fn a_complete_status_gives_the_range() {
        assert_eq!(picture_range(words(1), 4096, 7), Ok((16, 1000)));
    }

    #[test]
    fn a_failed_status_keeps_its_value_and_name() {
        assert_eq!(
            picture_range(words(-1), 4096, 7),
            Err(EncodeError::VulkanEncodeStatus {
                status: -1,
                name: "ERROR",
                pts: 7,
            })
        );
        assert_eq!(
            picture_range(words(0), 4096, 7),
            Err(EncodeError::VulkanEncodeStatus {
                status: 0,
                name: "NOT_READY",
                pts: 7,
            })
        );
    }

    #[test]
    fn an_overrun_names_the_buffer() {
        let error = picture_range(words(STATUS_OVERRUN), 4096, 9).expect_err("refused");
        assert_eq!(
            error,
            EncodeError::VulkanBitstreamOverrun {
                pts: 9,
                capacity: 4096,
            }
        );
        assert!(error.to_string().contains("4096-byte"), "{error}");
    }

    #[test]
    fn a_range_outside_the_buffer_or_empty_is_an_error() {
        assert!(matches!(
            picture_range(words(1), 1015, 7),
            Err(EncodeError::VulkanFeedbackRange { .. })
        ));
        let empty = FeedbackWords {
            offset: 0,
            written: 0,
            status: 1,
        };
        assert!(matches!(
            picture_range(empty, 4096, 7),
            Err(EncodeError::VulkanFeedbackRange { .. })
        ));
        assert_eq!(picture_range(words(1), 1016, 7), Ok((16, 1000)));
    }

    #[test]
    fn the_words_hold_three_values_with_no_padding() {
        assert_eq!(std::mem::size_of::<FeedbackWords>(), 12);
        assert_eq!(FEEDBACK_FLAGS.as_raw(), 0b11);
    }
}
