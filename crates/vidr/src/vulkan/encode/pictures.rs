//! The encode images: one input image that a plane copy fills, and the DPB slots. Each stays on the
//! encode queue between pictures, and only the input image crosses to the copy queue.

use std::sync::Arc;

use ash::vk;

use crate::error::EncodeError;
use crate::vulkan::caps::separate_images;
use crate::vulkan::device::VulkanDevice;
use crate::vulkan::objects::{Image, ImageView};
use crate::vulkan::pictures::PictureImage;
use crate::vulkan::session::ImageFormat;

use super::caps::EncodeCaps;
use super::gop::SLOTS;
use super::profile::{EncodeProfile, query_format};

/// The input image's usage: a plane copy writes it, and the encode reads it.
const INPUT_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR.as_raw() | vk::ImageUsageFlags::TRANSFER_DST.as_raw(),
);

/// Every image of one session. Each view comes first, so it drops before its image.
pub(super) struct EncodePictures {
    pub(super) input_view: ImageView,
    pub(super) input: Image,
    pub(super) input_format: vk::Format,
    pub(super) slot_views: Vec<ImageView>,
    pub(super) reference_format: vk::Format,
    images: Vec<Image>,
    layers_per_image: u32,
}

impl EncodePictures {
    /// `extent` is the size of each image. Without separate reference images, each slot is a layer
    /// of one image.
    pub(super) fn new(
        device: &Arc<VulkanDevice>,
        profile: &EncodeProfile,
        caps: &EncodeCaps,
        extent: (u32, u32),
    ) -> Result<Self, EncodeError> {
        let info = profile.info();
        let infos = [info];
        let make = |layers: u32,
                    usage: vk::ImageUsageFlags,
                    format: ImageFormat,
                    what: &str|
         -> Result<Image, EncodeError> {
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
            Ok(Image::new(device, &create, what)?)
        };

        let input_format = query_format(device, profile, INPUT_USAGE)?;
        let input = make(1, INPUT_USAGE, input_format, "the encode input image")?;
        let input_view = ImageView::new(device, input.raw, input_format.format, 0)?;

        let dpb_usage = vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR;
        let slot_format = query_format(device, profile, dpb_usage)?;
        let separate = separate_images(caps.flags);
        let (image_count, layers) = if separate { (SLOTS, 1) } else { (1, SLOTS) };
        let mut images = Vec::with_capacity(image_count as usize);
        for _ in 0..image_count {
            images.push(make(layers, dpb_usage, slot_format, "a DPB image")?);
        }
        let mut slot_views = Vec::with_capacity(SLOTS as usize);
        for slot in 0..SLOTS {
            let (image, layer) = if separate {
                (slot as usize, 0)
            } else {
                (0, slot)
            };
            slot_views.push(ImageView::new(
                device,
                images[image].raw,
                slot_format.format,
                layer,
            )?);
        }
        Ok(Self {
            input_view,
            input,
            input_format: input_format.format,
            slot_views,
            reference_format: slot_format.format,
            images,
            layers_per_image: layers,
        })
    }

    /// The input image.
    pub(super) const fn input_picture(&self) -> PictureImage {
        PictureImage {
            image: self.input.raw,
            layer: 0,
        }
    }

    /// The image of `slot`.
    pub(super) fn slot_image(&self, slot: u32) -> Option<PictureImage> {
        let (image, layer) = if self.layers_per_image > 1 {
            (0, slot)
        } else {
            (slot as usize, 0)
        };
        Some(PictureImage {
            image: self.images.get(image)?.raw,
            layer,
        })
    }
}
