//! The checks of one decode request, before any driver call of the picture.

use ash::vk;

use super::engine::DecodeRequest;
use crate::error::DecodeError;

/// What a request checks against.
pub(super) struct RequestLimits {
    /// The session's DPB slots.
    pub(super) slots: usize,
    /// `maxActiveReferencePictures` of the session.
    pub(super) active_references: u32,
    /// Whether the session holds an output image.
    pub(super) output_image: bool,
    /// The session parameters in place.
    pub(super) parameters: Option<vk::VideoSessionParametersKHR>,
    /// Whether the codec uses session parameters.
    pub(super) uses_parameters: bool,
}

/// The parameters a coding scope binds. A decode scope of a codec with session parameters
/// needs them, and the session's reset runs in one. So that codec sets them before its first
/// picture. VP9 uses none, so `uses_parameters` is false and the handle stays null.
pub(super) fn bound_parameters(
    parameters: Option<vk::VideoSessionParametersKHR>,
    uses_parameters: bool,
    pts: i64,
) -> Result<vk::VideoSessionParametersKHR, DecodeError> {
    if !uses_parameters {
        return Ok(vk::VideoSessionParametersKHR::null());
    }
    parameters.ok_or_else(|| DecodeError::VulkanDecodeRequest {
        pts,
        reason: "the session has no parameters".to_owned(),
    })
}

/// Refuses a request the driver takes as invalid usage, whatever its codec:
///
/// - a slot past the session's `slots`.
/// - more references than `limit`, `VUID-vkCmdDecodeVideoKHR-activeReferencePictureCount-07150`.
/// - a slot named twice, `VUID-VkVideoBeginCodingInfoKHR-pPictureResource-07238`,
///   `VUID-vkCmdDecodeVideoKHR-pDecodeInfo-07264`, and
///   `VUID-vkCmdDecodeVideoKHR-dpbFrameUseCount-07176`.
///
/// The setup slot counts as named, because a coding scope binds it too. No path here decodes a
/// second field, which shares its first field's slot. So a valid stream never names a slot twice.
fn check_slots(request: &DecodeRequest<'_>, slots: usize, limit: u32) -> Result<(), DecodeError> {
    let named = || request.references.iter().chain([&request.setup]);
    for slot_ref in named() {
        if slot_ref.slot as usize >= slots {
            return Err(absent_slot(slot_ref.slot, slots, request.pts));
        }
    }
    if request.references.len() > limit as usize {
        return Err(DecodeError::VulkanTooManyReferences {
            pts: request.pts,
            references: request.references.len(),
            limit,
        });
    }
    for (index, slot_ref) in named().enumerate() {
        if named()
            .take(index)
            .any(|earlier| earlier.slot == slot_ref.slot)
        {
            return Err(DecodeError::VulkanDuplicateSlot {
                pts: request.pts,
                slot: slot_ref.slot,
            });
        }
    }
    Ok(())
}

/// Refuses a byte range outside the picture's bytes, on what the driver takes as
/// `VUID-vkCmdDecodeVideoKHR-pTileOffsets-09253` and
/// `VUID-vkCmdDecodeVideoKHR-pTileOffsets-09252`.
fn check_ranges(request: &DecodeRequest<'_>) -> Result<(), DecodeError> {
    let bytes: u64 = request.bitstream.iter().map(|part| part.len() as u64).sum();
    for &(offset, size) in request.byte_ranges {
        if u64::from(offset) >= bytes || u64::from(offset) + u64::from(size) > bytes {
            return Err(DecodeError::VulkanTileRange {
                pts: request.pts,
                offset,
                size,
                bytes,
            });
        }
    }
    Ok(())
}

/// The error of a slot index past the session's slots.
pub(super) fn absent_slot(slot: u32, slots: usize, pts: i64) -> DecodeError {
    DecodeError::VulkanDecodeRequest {
        pts,
        reason: format!("DPB slot {slot} does not exist in {slots} slots"),
    }
}

/// Refuses a request with slots this session cannot bind, or a session with no parameters.
/// The refusal comes before any driver call of the picture.
pub(super) fn check_request(
    request: &DecodeRequest<'_>,
    limits: &RequestLimits,
) -> Result<(), DecodeError> {
    check_slots(request, limits.slots, limits.active_references)?;
    check_ranges(request)?;
    if request.separate_output && !limits.output_image {
        return Err(DecodeError::VulkanDecodeRequest {
            pts: request.pts,
            reason: "the picture needs an output image, and the session has none".to_owned(),
        });
    }
    bound_parameters(limits.parameters, limits.uses_parameters, request.pts).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::super::engine::SlotRef;
    use super::*;

    fn slot_ref(slot: u32) -> SlotRef {
        SlotRef {
            slot,
            codec_info: std::ptr::null(),
            coded: (64, 64),
        }
    }

    fn checked(setup: u32, references: &[u32], limit: u32) -> Result<(), DecodeError> {
        let references: Vec<_> = references.iter().copied().map(slot_ref).collect();
        let request = DecodeRequest {
            bitstream: &[],
            picture_info: std::ptr::null(),
            setup: slot_ref(setup),
            references: &references,
            reference: true,
            output: true,
            separate_output: false,
            byte_ranges: &[],
            pts: 9,
        };
        check_slots(&request, 6, limit)
    }

    fn ranges_checked(ranges: &[(u32, u32)]) -> Result<(), DecodeError> {
        let request = DecodeRequest {
            bitstream: &[&[0; 6], &[0; 4]],
            picture_info: std::ptr::null(),
            setup: slot_ref(0),
            references: &[],
            reference: true,
            output: true,
            separate_output: false,
            byte_ranges: ranges,
            pts: 9,
        };
        check_ranges(&request)
    }

    #[test]
    fn a_byte_range_inside_the_picture_passes() {
        assert_eq!(ranges_checked(&[]), Ok(()));
        assert_eq!(ranges_checked(&[(0, 10)]), Ok(()));
        assert_eq!(ranges_checked(&[(6, 4), (9, 1)]), Ok(()));
    }

    #[test]
    fn a_byte_range_past_the_picture_is_refused_by_name() {
        for (offset, size) in [(10, 0), (10, 1), (9, 2), (u32::MAX, u32::MAX)] {
            let error = ranges_checked(&[(0, 1), (offset, size)]).expect_err("refused");
            assert_eq!(
                error,
                DecodeError::VulkanTileRange {
                    pts: 9,
                    offset,
                    size,
                    bytes: 10,
                }
            );
            let text = error.to_string();
            assert!(text.contains("pts 9"), "{text}");
            assert!(text.contains("10 bytes"), "{text}");
        }
    }

    #[test]
    fn distinct_slots_up_to_the_limit_pass() {
        assert_eq!(checked(0, &[], 1), Ok(()));
        assert_eq!(checked(5, &[1, 2, 3], 3), Ok(()));
    }

    #[test]
    fn more_references_than_the_session_holds_are_refused_by_name() {
        let error = checked(5, &[1, 2, 3], 2).expect_err("refused");
        assert_eq!(
            error,
            DecodeError::VulkanTooManyReferences {
                pts: 9,
                references: 3,
                limit: 2,
            }
        );
        let text = error.to_string();
        assert!(text.contains("pts 9"), "{text}");
        assert!(text.contains("binds 3"), "{text}");
        assert!(text.contains("holds 2"), "{text}");
        assert!(text.contains("re-encode"), "{text}");
    }

    #[test]
    fn a_slot_named_twice_is_refused_by_name() {
        for (setup, references, slot) in [(5, &[1, 2, 1], 1), (2, &[1, 2, 3], 2)] {
            let error = checked(setup, references, 3).expect_err("refused");
            assert_eq!(error, DecodeError::VulkanDuplicateSlot { pts: 9, slot });
            let text = error.to_string();
            assert!(text.contains(&format!("DPB slot {slot} twice")), "{text}");
        }
    }

    #[test]
    fn a_slot_past_the_session_comes_first() {
        let error = checked(5, &[7, 7, 7], 1).expect_err("refused");
        assert!(matches!(error, DecodeError::VulkanDecodeRequest { .. }));
    }

    #[test]
    fn a_decode_needs_session_parameters() {
        let raw = vk::VideoSessionParametersKHR::null();
        assert_eq!(bound_parameters(Some(raw), true, 4), Ok(raw));
        let error = bound_parameters(None, true, 4).expect_err("refused");
        assert!(matches!(
            error,
            DecodeError::VulkanDecodeRequest { pts: 4, .. }
        ));
    }

    #[test]
    fn a_codec_with_no_parameters_binds_a_null_handle() {
        assert_eq!(
            bound_parameters(None, false, 4),
            Ok(vk::VideoSessionParametersKHR::null())
        );
    }

    #[test]
    fn an_absent_slot_names_itself_with_no_codec() {
        let text = absent_slot(7, 5, 9).to_string();
        assert!(text.contains("DPB slot 7"), "{text}");
        assert!(text.contains("pts 9"), "{text}");
        assert!(!text.contains("H.264"), "{text}");
    }

    /// The output image and the parameters come after the slots and the ranges.
    #[test]
    fn a_request_checks_its_session_in_order() {
        let raw = vk::VideoSessionParametersKHR::null();
        let references = [slot_ref(1)];
        let request = DecodeRequest {
            bitstream: &[],
            picture_info: std::ptr::null(),
            setup: slot_ref(0),
            references: &references,
            reference: true,
            output: true,
            separate_output: true,
            byte_ranges: &[],
            pts: 9,
        };
        let mut limits = RequestLimits {
            slots: 2,
            active_references: 1,
            output_image: true,
            parameters: Some(raw),
            uses_parameters: true,
        };
        assert_eq!(check_request(&request, &limits), Ok(()));
        limits.parameters = None;
        let text = check_request(&request, &limits)
            .expect_err("refused")
            .to_string();
        assert!(text.contains("no parameters"), "{text}");
        limits.output_image = false;
        let text = check_request(&request, &limits)
            .expect_err("refused")
            .to_string();
        assert!(text.contains("output image"), "{text}");
        limits.slots = 1;
        let error = check_request(&request, &limits).expect_err("refused");
        assert!(error.to_string().contains("DPB slot 1"), "{error}");
    }
}
