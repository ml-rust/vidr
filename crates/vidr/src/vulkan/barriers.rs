//! Image barriers of a decode and a plane copy, as plain values. Nothing here touches a
//! device, so tests read each field back.

use ash::vk;

use super::objects::color_range;
use super::pictures::{PictureImage, SlotState};

/// The barrier a decode needs before it reads a slot in `state`, or `None`. `families`
/// is `(general, video)` when the queue families differ, and both ignored otherwise.
///
/// - `Written`: the decode that wrote the slot, against this read.
/// - `Returning`: the slot comes back from the copy layout. With a transfer, this is the
///   acquire of the copy queue's release, and an acquire has no source scope. With no
///   transfer, the source scope chains to the barrier that ended the writing decode.
pub(super) fn read_barrier(
    state: SlotState,
    picture: PictureImage,
    families: (u32, u32),
) -> Option<vk::ImageMemoryBarrier2<'static>> {
    let read = (
        vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
        vk::AccessFlags2::VIDEO_DECODE_READ_KHR,
        vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
    );
    match state {
        SlotState::Empty | SlotState::Ready => None,
        SlotState::Written => Some(image_barrier(
            picture,
            (
                vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
                vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
                vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
            ),
            read,
            (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED),
        )),
        SlotState::Returning => {
            let stage = if families.0 == vk::QUEUE_FAMILY_IGNORED {
                vk::PipelineStageFlags2::ALL_COMMANDS
            } else {
                vk::PipelineStageFlags2::NONE
            };
            Some(image_barrier(
                picture,
                (
                    stage,
                    vk::AccessFlags2::NONE,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                ),
                read,
                families,
            ))
        }
    }
}

/// One image barrier: `(stage, access, layout)` on each side, and `(from, to)` families.
pub(super) fn image_barrier(
    picture: PictureImage,
    src: (vk::PipelineStageFlags2, vk::AccessFlags2, vk::ImageLayout),
    dst: (vk::PipelineStageFlags2, vk::AccessFlags2, vk::ImageLayout),
    families: (u32, u32),
) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src.0)
        .src_access_mask(src.1)
        .old_layout(src.2)
        .dst_stage_mask(dst.0)
        .dst_access_mask(dst.1)
        .new_layout(dst.2)
        .src_queue_family_index(families.0)
        .dst_queue_family_index(families.1)
        .image(picture.image)
        .subresource_range(color_range(picture.layer))
}

/// A barrier that discards an image's contents and takes it for a decode. The source scope
/// holds an earlier decode's write, so that write cannot land on the new picture.
pub(super) fn discard_barrier(
    picture: PictureImage,
    layout: vk::ImageLayout,
    access: vk::AccessFlags2,
) -> vk::ImageMemoryBarrier2<'static> {
    image_barrier(
        picture,
        (
            vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
            vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
            vk::ImageLayout::UNDEFINED,
        ),
        (vk::PipelineStageFlags2::VIDEO_DECODE_KHR, access, layout),
        (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_discard_starts_from_undefined_with_no_transfer() {
        let picture = PictureImage {
            image: vk::Image::null(),
            layer: 3,
        };
        let barrier = discard_barrier(
            picture,
            vk::ImageLayout::VIDEO_DECODE_DST_KHR,
            vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
        );
        assert_eq!(barrier.old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(barrier.new_layout, vk::ImageLayout::VIDEO_DECODE_DST_KHR);
        assert_eq!(barrier.src_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
        assert_eq!(barrier.dst_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
        assert_eq!(barrier.subresource_range.base_array_layer, 3);
        assert_eq!(
            barrier.src_access_mask,
            vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR
        );
    }

    const NO_TRANSFER: (u32, u32) = (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED);
    const PICTURE: PictureImage = PictureImage {
        image: vk::Image::null(),
        layer: 1,
    };

    #[test]
    fn a_written_slot_takes_a_memory_dependency_before_a_read() {
        let barrier = read_barrier(SlotState::Written, PICTURE, NO_TRANSFER).expect("a barrier");
        assert_eq!(
            barrier.src_stage_mask,
            vk::PipelineStageFlags2::VIDEO_DECODE_KHR
        );
        assert_eq!(
            barrier.src_access_mask,
            vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR
        );
        assert_eq!(
            barrier.dst_stage_mask,
            vk::PipelineStageFlags2::VIDEO_DECODE_KHR
        );
        assert_eq!(
            barrier.dst_access_mask,
            vk::AccessFlags2::VIDEO_DECODE_READ_KHR
        );
        assert_eq!(barrier.old_layout, vk::ImageLayout::VIDEO_DECODE_DPB_KHR);
        assert_eq!(barrier.new_layout, vk::ImageLayout::VIDEO_DECODE_DPB_KHR);
        assert_eq!(barrier.src_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
    }

    #[test]
    fn a_ready_or_empty_slot_takes_no_barrier() {
        assert!(read_barrier(SlotState::Ready, PICTURE, NO_TRANSFER).is_none());
        assert!(read_barrier(SlotState::Empty, PICTURE, NO_TRANSFER).is_none());
    }

    #[test]
    fn a_returning_slot_acquires_or_chains_on_one_family() {
        let barrier = read_barrier(SlotState::Returning, PICTURE, (0, 3)).expect("a barrier");
        assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::NONE);
        assert_eq!(barrier.old_layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        assert_eq!(barrier.new_layout, vk::ImageLayout::VIDEO_DECODE_DPB_KHR);
        assert_eq!(
            barrier.dst_access_mask,
            vk::AccessFlags2::VIDEO_DECODE_READ_KHR
        );
        assert_eq!(
            (
                barrier.src_queue_family_index,
                barrier.dst_queue_family_index
            ),
            (0, 3)
        );

        let barrier = read_barrier(SlotState::Returning, PICTURE, NO_TRANSFER).expect("a barrier");
        assert_eq!(
            barrier.src_stage_mask,
            vk::PipelineStageFlags2::ALL_COMMANDS
        );
        assert_eq!(
            barrier.dst_access_mask,
            vk::AccessFlags2::VIDEO_DECODE_READ_KHR
        );
    }
}
