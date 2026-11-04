//! Image planes into numr buffers, with no host round trip.
//!
//! A numr tensor on WebGPU is a storage buffer in numr's registry. So the plane copy
//! is `vkCmdCopyImageToBuffer` into that buffer, recorded on a wgpu command encoder and
//! submitted on wgpu's queue. numr's kernels read the buffer on the same queue.
//!
//! wgpu tracks whether each buffer holds data, and it zeroes a buffer before its first
//! read otherwise. A raw copy is invisible to that tracking. So `clear_buffer` first
//! marks the buffer as written. It also gives the zeros past each row that the word
//! form states. A second encoder then holds the raw copy, because wgpu refuses both
//! kinds of command in one encoder.

use std::time::Duration;

use ash::vk;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::wgpu::{WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::hal::api::Vulkan;
use wgpu::hal::vulkan as hal_vulkan;

use super::barriers::image_barrier;
use super::device::VulkanDevice;
use super::engine::CopySource;
use crate::decode::{ColorFacts, DecodedFrame, PixelFormat};
use crate::error::DecodeError;
use crate::geometry::Dimensions;

/// How long a copy, or the end of a session, waits before it stops waiting.
pub(super) const WAIT: Duration = Duration::from_secs(60);
const WORD_BYTES: usize = 4;

/// The picture area of one frame, inside the coded image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FrameArea {
    pub(super) left: u32,
    pub(super) top: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

impl FrameArea {
    /// The display size of this area, or the refusal that names it inside `coded`.
    pub(super) fn display(self, coded: (u32, u32)) -> Result<Dimensions, DecodeError> {
        Dimensions::new(self.width, self.height).map_err(|_| DecodeError::DisplayArea {
            rect: (
                self.left as i32,
                self.top as i32,
                (self.left + self.width) as i32,
                (self.top + self.height) as i32,
            ),
            coded,
        })
    }
}

/// One plane's shape as words and its copy region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PlaneLayout {
    pub(super) shape: [usize; 2],
    pub(super) row_texels: u32,
    pub(super) offset: (i32, i32),
    pub(super) extent: (u32, u32),
    pub(super) aspect: vk::ImageAspectFlags,
}

/// The luma and chroma layouts of 8-bit 4:2:0 in `PlaneForm::Words`. A word row holds
/// `4 * words` bytes, so `row_texels` counts 1-byte luma and 2-byte chroma texels.
pub(super) fn nv12_layout(area: FrameArea) -> [PlaneLayout; 2] {
    let (width, height) = (area.width as usize, area.height as usize);
    let luma_words = width.div_ceil(WORD_BYTES);
    let chroma_width = width.div_ceil(2) * 2;
    let chroma_words = chroma_width.div_ceil(WORD_BYTES);
    [
        PlaneLayout {
            shape: [height, luma_words],
            row_texels: (luma_words * WORD_BYTES) as u32,
            offset: (area.left as i32, area.top as i32),
            extent: (area.width, area.height),
            aspect: vk::ImageAspectFlags::PLANE_0,
        },
        PlaneLayout {
            shape: [height.div_ceil(2), chroma_words],
            row_texels: (chroma_words * WORD_BYTES / 2) as u32,
            offset: (area.left as i32 / 2, area.top as i32 / 2),
            extent: (area.width.div_ceil(2), area.height.div_ceil(2)),
            aspect: vk::ImageAspectFlags::PLANE_1,
        },
    ]
}

/// The luma and chroma layouts of 10-bit 4:2:0 in `PlaneForm::Words`. Each luma texel holds
/// 2 bytes and each chroma texel 4, as P010 lays them out.
pub(super) fn p010_layout(area: FrameArea) -> [PlaneLayout; 2] {
    let (width, height) = (area.width as usize, area.height as usize);
    let luma_words = (width * 2).div_ceil(WORD_BYTES);
    let chroma_words = width.div_ceil(2);
    [
        PlaneLayout {
            shape: [height, luma_words],
            row_texels: (luma_words * WORD_BYTES / 2) as u32,
            offset: (area.left as i32, area.top as i32),
            extent: (area.width, area.height),
            aspect: vk::ImageAspectFlags::PLANE_0,
        },
        PlaneLayout {
            shape: [height.div_ceil(2), chroma_words],
            row_texels: chroma_words as u32,
            offset: (area.left as i32 / 2, area.top as i32 / 2),
            extent: (area.width.div_ceil(2), area.height.div_ceil(2)),
            aspect: vk::ImageAspectFlags::PLANE_1,
        },
    ]
}

fn allocate(
    device: &VulkanDevice,
    plane: &'static str,
    shape: [usize; 2],
) -> Result<Tensor<WgpuRuntime>, DecodeError> {
    Tensor::empty(&shape, DType::U32, device.numr_device()).map_err(|error| {
        DecodeError::Allocation {
            plane,
            shape,
            reason: error.to_string(),
        }
    })
}

pub(super) fn raw_buffer(buffer: &wgpu::Buffer) -> Result<vk::Buffer, DecodeError> {
    // SAFETY: vidr destroys nothing behind the guard, and `buffer` outlives the copy.
    let hal = unsafe { buffer.as_hal::<Vulkan>() };
    hal.map(|hal| {
        // SAFETY: as above.
        unsafe { hal.raw_handle() }
    })
    .ok_or_else(|| DecodeError::VulkanUnavailable {
        reason: "numr's buffer is not a Vulkan buffer".to_owned(),
    })
}

/// Returns `waited`, and on an error keeps each tensor allocated for the life of the process.
///
/// A wait that fails leaves the copy in flight, and wgpu tracks no raw-recorded command. A tensor
/// that drops then frees a buffer the GPU still uses, which `VUID-vkDestroyBuffer-buffer-00922`
/// forbids. A forgotten clone holds the shared storage, so numr never frees the buffer. The named
/// error stays with the decoder or encoder, which then stops. So each stream leaks at most once.
pub(super) fn keep_on_stall<R: Runtime, T, E>(
    waited: Result<T, E>,
    tensors: [&Tensor<R>; 2],
) -> Result<T, E> {
    if waited.is_err() {
        for tensor in tensors {
            std::mem::forget(tensor.clone());
        }
    }
    waited
}

/// A frame of size `coded`, from the top left of what `source` names. Its planes take one copy
/// on the device.
pub(super) fn whole_frame(
    device: &VulkanDevice,
    source: &CopySource,
    format: PixelFormat,
    coded: (u32, u32),
    color: ColorFacts,
    pts: i64,
) -> Result<DecodedFrame<WgpuRuntime>, DecodeError> {
    let area = FrameArea {
        left: 0,
        top: 0,
        width: coded.0,
        height: coded.1,
    };
    let display = area.display(coded)?;
    let (y, uv) = copy_planes(device, source, area, format)?;
    Ok(DecodedFrame::from_packed_planes(
        y, uv, format, pts, display, color,
    ))
}

/// Copies both planes of an image in `format` to new tensors and waits for the copy.
pub(super) fn copy_planes(
    device: &VulkanDevice,
    source: &CopySource,
    area: FrameArea,
    format: PixelFormat,
) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>), DecodeError> {
    device.check_live("a plane copy")?;
    let [luma, chroma] = match format {
        PixelFormat::Nv12 => nv12_layout(area),
        PixelFormat::P010 => p010_layout(area),
    };
    let y = allocate(device, "luma", luma.shape)?;
    let uv = allocate(device, "chroma", chroma.shape)?;
    let missing = || DecodeError::VulkanUnavailable {
        reason: "numr's registry holds no buffer for a new tensor".to_owned(),
    };
    let y_buffer = get_buffer(y.ptr()).ok_or_else(missing)?;
    let uv_buffer = get_buffer(uv.ptr()).ok_or_else(missing)?;

    let wgpu_device = &device.wgpu_device;
    let mut clear = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vidr frame clear"),
    });
    clear.clear_buffer(&y_buffer, 0, None);
    clear.clear_buffer(&uv_buffer, 0, None);

    let copies = [
        (raw_buffer(&y_buffer)?, luma),
        (raw_buffer(&uv_buffer)?, chroma),
    ];
    let mut copy = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vidr frame copy"),
    });
    let record = |encoder: Option<&mut hal_vulkan::CommandEncoder>| {
        encoder.map(|encoder| {
            // SAFETY: vidr never destroys this handle, and wgpu ends the buffer in `finish`.
            let cmd = unsafe { encoder.raw_handle() };
            // SAFETY: `cmd` is recording on wgpu's queue family. The image and both
            // buffers stay live through the wait below. A failed wait keeps both tensors.
            unsafe { record_copy(device, cmd, source, &copies) };
        })
    };
    // SAFETY: the callback records on the open encoder, ends nothing, destroys nothing,
    // and this encoder takes no other command.
    let recorded = unsafe { copy.as_hal_mut::<Vulkan, _, _>(record) };
    if recorded.is_none() {
        return Err(DecodeError::VulkanUnavailable {
            reason: "wgpu's command encoder is not recording on Vulkan".to_owned(),
        });
    }

    // SAFETY: vidr destroys nothing behind the guard.
    if let Some(queue) = unsafe { device.wgpu_queue.as_hal::<Vulkan>() } {
        queue.add_wait_semaphore(
            source.timeline,
            Some(source.timeline_value),
            vk::PipelineStageFlags::ALL_COMMANDS,
        );
    }
    let submission = device.wgpu_queue.submit([clear.finish(), copy.finish()]);
    let waited = device.wait_general_queue(Some(submission), "a plane copy");
    keep_on_stall(waited, [&y, &uv])?;
    Ok((y, uv))
}

/// Records the barriers and both plane copies on `cmd`.
///
/// # Safety
///
/// `cmd` is recording on the general family, and the image and buffers are live.
unsafe fn record_copy(
    device: &VulkanDevice,
    cmd: vk::CommandBuffer,
    source: &CopySource,
    copies: &[(vk::Buffer, PlaneLayout); 2],
) {
    // SAFETY: the caller states `cmd` is recording, and each object is live.
    unsafe { record_before_copy(device, cmd, source, copies) };
    for (buffer, plane) in copies {
        let region = plane_region(plane, source.picture.layer);
        // SAFETY: the image sits in the transfer layout, and the region fits the buffer.
        // The buffer's rows hold `row_texels` texels over `shape[0]` rows.
        unsafe {
            device.raw.cmd_copy_image_to_buffer(
                cmd,
                source.picture.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                *buffer,
                &[region],
            );
        }
    }
    // SAFETY: as above.
    unsafe { record_after_copy(device, cmd, source, copies) };
}

/// The barrier after the clear, and the image's acquire when the families differ.
///
/// # Safety
///
/// As [`record_copy`].
unsafe fn record_before_copy(
    device: &VulkanDevice,
    cmd: vk::CommandBuffer,
    source: &CopySource,
    copies: &[(vk::Buffer, PlaneLayout); 2],
) {
    let cleared = copies.map(|(buffer, _)| {
        whole_buffer(
            buffer,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
        )
    });
    let acquire = device.transfers_ownership().then(|| {
        image_barrier(
            source.picture,
            // The decode's release states the same two layouts. The pair runs one
            // transition.
            (
                vk::PipelineStageFlags2::NONE,
                vk::AccessFlags2::NONE,
                source.decode_layout,
            ),
            (
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            ),
            (device.video_family, device.general_family),
        )
    });
    let info = vk::DependencyInfo::default()
        .buffer_memory_barriers(&cleared)
        .image_memory_barriers(acquire.as_slice());
    // SAFETY: the caller states `cmd` is recording, and each barrier names a live object.
    unsafe { device.sync2.cmd_pipeline_barrier2(cmd, &info) };
}

/// The copy of one plane of the image layer.
fn plane_region(plane: &PlaneLayout, layer: u32) -> vk::BufferImageCopy {
    vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: plane.row_texels,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: plane.aspect,
            mip_level: 0,
            base_array_layer: layer,
            layer_count: 1,
        },
        image_offset: vk::Offset3D {
            x: plane.offset.0,
            y: plane.offset.1,
            z: 0,
        },
        image_extent: vk::Extent3D {
            width: plane.extent.0,
            height: plane.extent.1,
            depth: 1,
        },
    }
}

/// The barrier after the copy, and the image's release when it returns across families.
///
/// # Safety
///
/// As [`record_copy`].
unsafe fn record_after_copy(
    device: &VulkanDevice,
    cmd: vk::CommandBuffer,
    source: &CopySource,
    copies: &[(vk::Buffer, PlaneLayout); 2],
) {
    let written = copies.map(|(buffer, _)| {
        whole_buffer(
            buffer,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        )
    });
    let release = (source.returns_image && device.transfers_ownership()).then(|| {
        image_barrier(
            source.picture,
            (
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            ),
            (
                vk::PipelineStageFlags2::NONE,
                vk::AccessFlags2::NONE,
                vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
            ),
            (device.general_family, device.video_family),
        )
    });
    let info = vk::DependencyInfo::default()
        .buffer_memory_barriers(&written)
        .image_memory_barriers(release.as_slice());
    // SAFETY: the caller states `cmd` is recording, and each barrier names a live object.
    unsafe { device.sync2.cmd_pipeline_barrier2(cmd, &info) };
}

pub(super) fn whole_buffer(
    buffer: vk::Buffer,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) -> vk::BufferMemoryBarrier2<'static> {
    vk::BufferMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(0)
        .size(vk::WHOLE_SIZE)
}

#[cfg(test)]
mod tests {
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    use super::*;

    #[test]
    fn a_failed_wait_keeps_both_tensors_and_a_finished_wait_keeps_none() {
        let device = CpuDevice::new();
        let tensor = |shape: &[usize]| {
            Tensor::<CpuRuntime>::empty(shape, DType::U32, &device).expect("a tensor")
        };
        let (y, uv) = (tensor(&[2, 2]), tensor(&[1, 2]));
        assert_eq!(keep_on_stall(Ok::<_, ()>(()), [&y, &uv]), Ok(()));
        assert_eq!((y.ref_count(), uv.ref_count()), (1, 1));
        assert_eq!(keep_on_stall(Err::<(), _>(7), [&y, &uv]), Err(7));
        assert_eq!((y.ref_count(), uv.ref_count()), (2, 2));
    }

    #[test]
    fn word_rows_hold_each_plane_at_the_crop_offset() {
        // 320x180 in a 320x192 coded image: rows fill whole words.
        let [luma, chroma] = nv12_layout(FrameArea {
            left: 0,
            top: 0,
            width: 320,
            height: 180,
        });
        assert_eq!(luma.shape, [180, 80]);
        assert_eq!(luma.row_texels, 320);
        assert_eq!(luma.extent, (320, 180));
        assert_eq!(luma.aspect, vk::ImageAspectFlags::PLANE_0);
        assert_eq!(chroma.shape, [90, 80]);
        assert_eq!(chroma.row_texels, 160);
        assert_eq!(chroma.extent, (160, 90));
        assert_eq!(chroma.aspect, vk::ImageAspectFlags::PLANE_1);
    }

    #[test]
    fn p010_rows_hold_two_byte_luma_and_four_byte_chroma_texels() {
        let [luma, chroma] = p010_layout(FrameArea {
            left: 0,
            top: 0,
            width: 320,
            height: 180,
        });
        assert_eq!(luma.shape, [180, 160]);
        assert_eq!(luma.row_texels, 320);
        assert_eq!(chroma.shape, [90, 160]);
        assert_eq!(chroma.row_texels, 160);
        assert_eq!(chroma.extent, (160, 90));
    }

    #[test]
    fn an_odd_p010_size_pads_each_row_to_a_word() {
        let [luma, chroma] = p010_layout(FrameArea {
            left: 4,
            top: 2,
            width: 325,
            height: 181,
        });
        assert_eq!(luma.shape, [181, 163]);
        assert_eq!(luma.row_texels, 326);
        assert_eq!(luma.offset, (4, 2));
        assert_eq!(chroma.shape, [91, 163]);
        assert_eq!(chroma.row_texels, 163);
        assert_eq!(chroma.offset, (2, 1));
        assert_eq!(chroma.extent, (163, 91));
    }

    #[test]
    fn a_width_past_a_word_pads_its_rows_and_an_offset_halves_for_chroma() {
        // 326 bytes round up to 82 words, 328 bytes a row.
        let [luma, chroma] = nv12_layout(FrameArea {
            left: 4,
            top: 2,
            width: 326,
            height: 184,
        });
        assert_eq!(luma.shape, [184, 82]);
        assert_eq!(luma.row_texels, 328);
        assert_eq!(luma.offset, (4, 2));
        assert_eq!(chroma.shape, [92, 82]);
        assert_eq!(chroma.row_texels, 164);
        assert_eq!(chroma.offset, (2, 1));
        assert_eq!(chroma.extent, (163, 92));
    }
}
