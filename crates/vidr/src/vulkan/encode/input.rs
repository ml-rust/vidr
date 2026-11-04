//! A frame's planes into the encode input image, with no host round trip.
//!
//! A numr tensor on WebGPU is a storage buffer in numr's registry. So the plane copy is
//! `vkCmdCopyBufferToImage` from that buffer, recorded on a wgpu command encoder and submitted on
//! wgpu's queue, after each kernel that wrote the frame. When the encode family differs, the copy
//! releases the image to it, and the encode's command buffer acquires it.

use ash::vk;
use numr::runtime::Device as _;
use numr::runtime::wgpu::{WgpuRuntime, get_buffer};
use wgpu::hal::api::Vulkan;
use wgpu::hal::vulkan as hal_vulkan;

use crate::encode::EncodeFrame;
use crate::error::EncodeError;
use crate::vulkan::barriers::image_barrier;
use crate::vulkan::copy::{raw_buffer, whole_buffer};
use crate::vulkan::device::VulkanDevice;
use crate::vulkan::pictures::PictureImage;

use super::planes::PlaneRegions;

/// The barrier that ends the copy. When the families differ, it releases the image.
pub(super) fn release_barrier(
    picture: PictureImage,
    families: Option<(u32, u32)>,
) -> vk::ImageMemoryBarrier2<'static> {
    let (dst_stage, dst_access, indices) = match families {
        Some(indices) => (
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
            indices,
        ),
        None => (
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ,
            (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED),
        ),
    };
    image_barrier(
        picture,
        (
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ),
        (dst_stage, dst_access, vk::ImageLayout::VIDEO_ENCODE_SRC_KHR),
        indices,
    )
}

/// The encode's acquire of the image. It states the release's two layouts, and the pair runs one
/// transition.
pub(super) fn acquire_barrier(
    picture: PictureImage,
    families: (u32, u32),
) -> vk::ImageMemoryBarrier2<'static> {
    image_barrier(
        picture,
        (
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ),
        (
            vk::PipelineStageFlags2::VIDEO_ENCODE_KHR,
            vk::AccessFlags2::VIDEO_ENCODE_READ_KHR,
            vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
        ),
        families,
    )
}

/// The barrier that starts the copy. The last encode finished, because each encode waits, so the
/// contents can go.
fn discard_barrier(picture: PictureImage) -> vk::ImageMemoryBarrier2<'static> {
    image_barrier(
        picture,
        (
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
            vk::ImageLayout::UNDEFINED,
        ),
        (
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ),
        (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED),
    )
}

/// What one frame's copy touches.
struct InputCopy<'a> {
    y: vk::Buffer,
    uv: vk::Buffer,
    picture: PictureImage,
    regions: &'a PlaneRegions,
    /// `(general, encode)` when the families differ.
    families: Option<(u32, u32)>,
}

/// Records the barriers and both plane copies on `cmd`.
///
/// # Safety
///
/// `cmd` is recording on the general family. The image and both buffers stay live until the copy
/// finishes.
unsafe fn record_input_copy(device: &VulkanDevice, cmd: vk::CommandBuffer, copy: &InputCopy<'_>) {
    let written = [copy.y, copy.uv].map(|buffer| {
        whole_buffer(
            buffer,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_WRITE,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_READ,
        )
    });
    let discard = [discard_barrier(copy.picture)];
    let before = vk::DependencyInfo::default()
        .buffer_memory_barriers(&written)
        .image_memory_barriers(&discard);
    // SAFETY: the caller states `cmd` is recording, and each barrier names a live object.
    unsafe { device.sync2.cmd_pipeline_barrier2(cmd, &before) };

    for (buffer, regions) in [
        (copy.y, &copy.regions.luma),
        (copy.uv, &copy.regions.chroma),
    ] {
        // SAFETY: the image sits in the transfer layout, and each region stays inside the buffer,
        // whose rows hold `row_texels` texels. `FrameLayout::check_frame` checked each plane.
        unsafe {
            device.raw.cmd_copy_buffer_to_image(
                cmd,
                buffer,
                copy.picture.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                regions,
            );
        }
    }

    let release = [release_barrier(copy.picture, copy.families)];
    let after = vk::DependencyInfo::default().image_memory_barriers(&release);
    // SAFETY: as above.
    unsafe { device.sync2.cmd_pipeline_barrier2(cmd, &after) };
}

/// Copies `frame` into the input image and waits for the copy. The image then sits in the encode
/// source layout, released to the encode family when the families differ.
///
/// # Errors
///
/// [`EncodeError::SurfaceMismatch`] for a plane on another device, and the errors of the wait.
pub(super) fn copy_frame(
    device: &VulkanDevice,
    frame: &EncodeFrame<'_, WgpuRuntime>,
    regions: &PlaneRegions,
    picture: PictureImage,
    encode_family: u32,
) -> Result<(), EncodeError> {
    let device_id = device.numr_device().id();
    for (plane, tensor) in [("luma", frame.y()), ("chroma", frame.uv())] {
        if tensor.device().id() != device_id {
            return Err(EncodeError::SurfaceMismatch {
                plane,
                expected: format!("a tensor on numr WebGPU device {device_id}"),
                found: format!("a tensor on numr WebGPU device {}", tensor.device().id()),
            });
        }
    }
    let missing = |plane: &'static str| EncodeError::SurfaceMismatch {
        plane,
        expected: "a tensor with a buffer in numr's registry".to_owned(),
        found: "a tensor with no buffer".to_owned(),
    };
    let y_buffer = get_buffer(frame.y().ptr()).ok_or_else(|| missing("luma"))?;
    let uv_buffer = get_buffer(frame.uv().ptr()).ok_or_else(|| missing("chroma"))?;

    let families =
        (encode_family != device.general_family).then_some((device.general_family, encode_family));
    let copy = InputCopy {
        y: raw_buffer(&y_buffer)?,
        uv: raw_buffer(&uv_buffer)?,
        picture,
        regions,
        families,
    };
    let mut encoder = device
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vidr encode input copy"),
        });
    let record = |encoder: Option<&mut hal_vulkan::CommandEncoder>| {
        encoder.map(|encoder| {
            // SAFETY: vidr never destroys this handle, and wgpu ends the buffer in `finish`.
            let cmd = unsafe { encoder.raw_handle() };
            // SAFETY: `cmd` is recording on wgpu's queue family. The image and both buffers stay
            // live until the wait below returns.
            unsafe { record_input_copy(device, cmd, &copy) };
        })
    };
    // SAFETY: the callback records on the open encoder, ends nothing, destroys nothing, and this
    // encoder takes no other command.
    let recorded = unsafe { encoder.as_hal_mut::<Vulkan, _, _>(record) };
    if recorded.is_none() {
        return Err(EncodeError::VulkanUnavailable {
            reason: "wgpu's command encoder is not recording on Vulkan".to_owned(),
        });
    }
    let submission = device.wgpu_queue.submit([encoder.finish()]);
    device.wait_general_queue(Some(submission), "a copy into the encode input image")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PICTURE: PictureImage = PictureImage {
        image: vk::Image::null(),
        layer: 0,
    };

    /// `VUID-VkImageMemoryBarrier2`: a release and its acquire state the same layouts and families.
    #[test]
    fn a_release_and_its_acquire_state_one_transition_across_families() {
        let release = release_barrier(PICTURE, Some((0, 4)));
        let acquire = acquire_barrier(PICTURE, (0, 4));
        assert_eq!(release.old_layout, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        assert_eq!(release.new_layout, vk::ImageLayout::VIDEO_ENCODE_SRC_KHR);
        assert_eq!(acquire.old_layout, release.old_layout);
        assert_eq!(acquire.new_layout, release.new_layout);
        assert_eq!(release.src_queue_family_index, 0);
        assert_eq!(release.dst_queue_family_index, 4);
        assert_eq!(acquire.src_queue_family_index, 0);
        assert_eq!(acquire.dst_queue_family_index, 4);
        assert_eq!(release.dst_stage_mask, vk::PipelineStageFlags2::NONE);
        assert_eq!(acquire.src_stage_mask, vk::PipelineStageFlags2::NONE);
        assert_eq!(
            acquire.dst_access_mask,
            vk::AccessFlags2::VIDEO_ENCODE_READ_KHR
        );
    }

    #[test]
    fn one_family_takes_no_transfer_and_makes_the_write_visible() {
        let release = release_barrier(PICTURE, None);
        assert_eq!(release.src_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
        assert_eq!(release.dst_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
        assert_eq!(
            release.dst_stage_mask,
            vk::PipelineStageFlags2::ALL_COMMANDS
        );
        assert_eq!(release.dst_access_mask, vk::AccessFlags2::MEMORY_READ);
    }

    #[test]
    fn a_new_frame_discards_the_image_with_no_transfer() {
        let discard = discard_barrier(PICTURE);
        assert_eq!(discard.old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(discard.new_layout, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        assert_eq!(discard.src_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
    }
}
