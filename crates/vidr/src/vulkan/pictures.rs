//! The decode images: DPB slots in the form the capabilities require, and the output
//! image of distinct mode.

use std::sync::Arc;

use ash::vk;

use super::caps::DpbMode;
use super::device::VulkanDevice;
use super::objects::{Image, ImageView};
use super::session::{ImageFormat, VideoProfile, query_format};
use crate::error::DecodeError;

/// Where a slot's image stands between decodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SlotState {
    /// Holds nothing the stream needs. The next use discards the contents.
    Empty,
    /// In the DPB layout, owned by the video queue, and visible to a decode read.
    Ready,
    /// In the DPB layout, owned by the video queue. A decode wrote it, and the next read
    /// needs a memory dependency on that write.
    Written,
    /// In the copy layout. When the families differ, the copy queue also released it to
    /// the video queue. The next decode takes it back.
    Returning,
}

/// The state of the setup slot after its decode and copy.
pub(super) const fn state_after_copy(mode: DpbMode, reference: bool) -> SlotState {
    match (mode, reference) {
        (_, false) => SlotState::Empty,
        (DpbMode::Coincide, true) => SlotState::Returning,
        (DpbMode::Distinct, true) => SlotState::Written,
    }
}

/// Whether the copy hands the image back to the video queue.
pub(super) const fn copy_returns_image(mode: DpbMode, reference: bool) -> bool {
    matches!(mode, DpbMode::Coincide) && reference
}

/// The image usage of a slot.
fn slot_usage(mode: DpbMode) -> vk::ImageUsageFlags {
    match mode {
        DpbMode::Coincide => {
            vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR
                | vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
                | vk::ImageUsageFlags::TRANSFER_SRC
        }
        DpbMode::Distinct => vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR,
    }
}

/// The image usage of the output image in distinct mode.
fn output_usage() -> vk::ImageUsageFlags {
    vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR | vk::ImageUsageFlags::TRANSFER_SRC
}

/// One DPB slot's picture resource.
pub(super) struct Slot {
    pub(super) image: usize,
    pub(super) layer: u32,
    pub(super) view: ImageView,
    pub(super) state: SlotState,
}

/// One image layer a decode writes and a copy reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PictureImage {
    pub(super) image: vk::Image,
    pub(super) layer: u32,
}

/// Every image of one session.
pub(super) struct PictureStore {
    pub(super) mode: DpbMode,
    pub(super) slots: Vec<Slot>,
    /// The view comes first, so it drops before its image.
    pub(super) output: Option<(ImageView, Image)>,
    images: Vec<Image>,
}

impl PictureStore {
    /// `extent` is the size of each image. With `separate` false, every slot is a layer
    /// of one image.
    pub(super) fn new(
        device: &Arc<VulkanDevice>,
        profile: &VideoProfile,
        mode: DpbMode,
        slot_count: u32,
        separate: bool,
        extent: (u32, u32),
    ) -> Result<Self, DecodeError> {
        let slot_format = query_format(device, profile, slot_usage(mode))?;
        let info = profile.info();
        let infos = [info];
        let make = |layers: u32,
                    usage: vk::ImageUsageFlags,
                    format: ImageFormat,
                    what: &str|
         -> Result<Image, DecodeError> {
            let mut list = vk::VideoProfileListInfoKHR::default().profiles(&infos);
            let create = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format.format)
                .extent(vk::Extent3D {
                    width: extent.0,
                    height: extent.1,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(layers)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(format.tiling)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .push_next(&mut list);
            Image::new(device, &create, what)
        };

        let usage = slot_usage(mode);
        let (image_count, layers) = if separate {
            (slot_count, 1)
        } else {
            (1, slot_count)
        };
        let mut images = Vec::with_capacity(image_count as usize);
        for _ in 0..image_count {
            images.push(make(layers, usage, slot_format, "a DPB image")?);
        }
        let mut slots = Vec::with_capacity(slot_count as usize);
        for index in 0..slot_count {
            let (image, layer) = if separate {
                (index as usize, 0)
            } else {
                (0, index)
            };
            let view = ImageView::new(device, images[image].raw, slot_format.format, layer)?;
            slots.push(Slot {
                image,
                layer,
                view,
                state: SlotState::Empty,
            });
        }
        let output = match mode {
            DpbMode::Coincide => None,
            DpbMode::Distinct => {
                let format = query_format(device, profile, output_usage())?;
                let image = make(1, output_usage(), format, "the output image")?;
                let view = ImageView::new(device, image.raw, format.format, 0)?;
                Some((view, image))
            }
        };
        Ok(Self {
            mode,
            slots,
            output,
            images,
        })
    }

    /// The image of `slot`.
    pub(super) fn slot_image(&self, slot: usize) -> Option<PictureImage> {
        let entry = self.slots.get(slot)?;
        Some(PictureImage {
            image: self.images.get(entry.image)?.raw,
            layer: entry.layer,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_coincide_reference_goes_back_through_the_copy_queue() {
        assert!(copy_returns_image(DpbMode::Coincide, true));
        assert!(!copy_returns_image(DpbMode::Coincide, false));
        assert!(!copy_returns_image(DpbMode::Distinct, true));
        assert_eq!(
            state_after_copy(DpbMode::Coincide, true),
            SlotState::Returning
        );
        assert_eq!(
            state_after_copy(DpbMode::Distinct, true),
            SlotState::Written
        );
        assert_eq!(state_after_copy(DpbMode::Coincide, false), SlotState::Empty);
        assert_eq!(state_after_copy(DpbMode::Distinct, false), SlotState::Empty);
    }

    #[test]
    fn each_image_holds_the_usage_of_its_role() {
        assert!(slot_usage(DpbMode::Coincide).contains(vk::ImageUsageFlags::TRANSFER_SRC));
        assert!(!slot_usage(DpbMode::Distinct).contains(vk::ImageUsageFlags::TRANSFER_SRC));
        assert!(output_usage().contains(vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR));
    }
}
