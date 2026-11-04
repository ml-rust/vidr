//! One owner per Vulkan object. Each frees its object on drop. A caller waits for the
//! queue before it drops an object a submit used.

use std::sync::Arc;

use ash::vk;

use super::device::VulkanDevice;
use super::error::vk_error;
use crate::error::DecodeError;

/// A memory type that holds every flag of `required`, with all of `preferred` when
/// one exists.
fn pick_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
    preferred: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let holds = |flags: vk::MemoryPropertyFlags| {
        (0..properties
            .memory_type_count
            .min(vk::MAX_MEMORY_TYPES as u32))
            .find(|&index| {
                type_bits & (1 << index) != 0
                    && properties.memory_types[index as usize]
                        .property_flags
                        .contains(flags)
            })
    };
    holds(required | preferred).or_else(|| holds(required))
}

/// One device memory allocation.
pub(super) struct Memory {
    device: Arc<VulkanDevice>,
    pub(super) raw: vk::DeviceMemory,
    /// Whether a write reaches the device with no flush.
    coherent: bool,
}

impl Memory {
    pub(super) fn allocate(
        device: &Arc<VulkanDevice>,
        requirements: vk::MemoryRequirements,
        required: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
        what: &str,
    ) -> Result<Self, DecodeError> {
        // SAFETY: `physical` belongs to this instance, and the call reads nothing else.
        let properties = unsafe {
            device
                .instance
                .get_physical_device_memory_properties(device.physical)
        };
        let index = pick_memory_type(
            &properties,
            requirements.memory_type_bits,
            required,
            preferred,
        )
        .ok_or_else(|| DecodeError::VulkanNoMemoryType {
            what: what.to_owned(),
            required: format!("{required:?}"),
            type_bits: requirements.memory_type_bits,
        })?;
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(index);
        // SAFETY: `info` is fully initialised, and the index comes from this device.
        let raw = unsafe { device.raw.allocate_memory(&info, None) }.map_err(|result| {
            vk_error(
                "vkAllocateMemory",
                result,
                format!("{} bytes for {what}", requirements.size),
            )
        })?;
        Ok(Self {
            device: Arc::clone(device),
            raw,
            coherent: properties.memory_types[index as usize]
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_COHERENT),
        })
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        // SAFETY: this type allocated the memory, and each object bound to it goes first.
        unsafe { self.device.raw.free_memory(self.raw, None) };
    }
}

/// An image and its memory.
pub(super) struct Image {
    device: Arc<VulkanDevice>,
    pub(super) raw: vk::Image,
    _memory: Memory,
}

impl Image {
    pub(super) fn new(
        device: &Arc<VulkanDevice>,
        info: &vk::ImageCreateInfo<'_>,
        what: &str,
    ) -> Result<Self, DecodeError> {
        // SAFETY: `info` and its chain are valid for the call.
        let raw = unsafe { device.raw.create_image(info, None) }
            .map_err(|result| vk_error("vkCreateImage", result, what))?;
        // A guard for the paths that return early.
        let owner = RawImage { device, raw };
        // SAFETY: `raw` is a live image of this device.
        let requirements = unsafe { device.raw.get_image_memory_requirements(raw) };
        let memory = Memory::allocate(
            device,
            requirements,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
            what,
        )?;
        // SAFETY: the memory comes from the image's requirements and is unbound.
        unsafe { device.raw.bind_image_memory(raw, memory.raw, 0) }
            .map_err(|result| vk_error("vkBindImageMemory", result, what))?;
        std::mem::forget(owner);
        Ok(Self {
            device: Arc::clone(device),
            raw,
            _memory: memory,
        })
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        // SAFETY: this type created the image. Its memory goes after this call.
        unsafe { self.device.raw.destroy_image(self.raw, None) };
    }
}

struct RawImage<'a> {
    device: &'a VulkanDevice,
    raw: vk::Image,
}

impl Drop for RawImage<'_> {
    fn drop(&mut self) {
        // SAFETY: no submit used this image, and nothing else owns it.
        unsafe { self.device.raw.destroy_image(self.raw, None) };
    }
}

/// A view of one image layer.
pub(super) struct ImageView {
    device: Arc<VulkanDevice>,
    pub(super) raw: vk::ImageView,
}

impl ImageView {
    pub(super) fn new(
        device: &Arc<VulkanDevice>,
        image: vk::Image,
        format: vk::Format,
        layer: u32,
    ) -> Result<Self, DecodeError> {
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(color_range(layer));
        // SAFETY: `image` is a live image of this device with this layer.
        let raw = unsafe { device.raw.create_image_view(&info, None) }.map_err(|result| {
            vk_error(
                "vkCreateImageView",
                result,
                format!("layer {layer} of {format:?}"),
            )
        })?;
        Ok(Self {
            device: Arc::clone(device),
            raw,
        })
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        // SAFETY: this type created the view, and the owner waited for the queue.
        unsafe { self.device.raw.destroy_image_view(self.raw, None) };
    }
}

/// All planes of one layer.
pub(super) fn color_range(layer: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: layer,
        layer_count: 1,
    }
}

/// Copies `parts` one after another to the start of `target`, then zeros the rest. The
/// copy stops at the end of `target`.
fn fill_parts(target: &mut [u8], parts: &[&[u8]]) {
    let mut written = 0;
    for part in parts {
        let kept = part.len().min(target.len() - written);
        target[written..written + kept].copy_from_slice(&part[..kept]);
        written += kept;
    }
    target[written..].fill(0);
}

/// A host-visible buffer of compressed bytes. Its memory stays mapped while it lives.
pub(super) struct HostBuffer {
    device: Arc<VulkanDevice>,
    pub(super) raw: vk::Buffer,
    pub(super) size: u64,
    /// The start of the memory's one map.
    mapped: *mut u8,
    memory: Memory,
}

// SAFETY: this value alone holds the map pointer, and a map stays valid on any thread.
unsafe impl Send for HostBuffer {}

impl HostBuffer {
    /// A buffer the host writes.
    pub(super) fn new(
        device: &Arc<VulkanDevice>,
        info: &vk::BufferCreateInfo<'_>,
        what: &str,
    ) -> Result<Self, DecodeError> {
        Self::with_memory(device, info, vk::MemoryPropertyFlags::HOST_COHERENT, what)
    }

    /// A buffer the host reads. It prefers cached memory, so a read stays fast.
    pub(super) fn for_reading(
        device: &Arc<VulkanDevice>,
        info: &vk::BufferCreateInfo<'_>,
        what: &str,
    ) -> Result<Self, DecodeError> {
        Self::with_memory(device, info, vk::MemoryPropertyFlags::HOST_CACHED, what)
    }

    fn with_memory(
        device: &Arc<VulkanDevice>,
        info: &vk::BufferCreateInfo<'_>,
        preferred: vk::MemoryPropertyFlags,
        what: &str,
    ) -> Result<Self, DecodeError> {
        // SAFETY: `info` and its chain are valid for the call.
        let raw = unsafe { device.raw.create_buffer(info, None) }
            .map_err(|result| vk_error("vkCreateBuffer", result, what))?;
        let owner = RawBuffer { device, raw };
        // SAFETY: `raw` is a live buffer of this device.
        let requirements = unsafe { device.raw.get_buffer_memory_requirements(raw) };
        let memory = Memory::allocate(
            device,
            requirements,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            preferred,
            what,
        )?;
        // SAFETY: the memory comes from the buffer's requirements and is unbound.
        unsafe { device.raw.bind_buffer_memory(raw, memory.raw, 0) }
            .map_err(|result| vk_error("vkBindBufferMemory", result, what))?;
        // SAFETY: the memory is host visible and unmapped.
        let mapped = unsafe {
            device
                .raw
                .map_memory(memory.raw, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .map_err(|result| vk_error("vkMapMemory", result, what))?;
        std::mem::forget(owner);
        Ok(Self {
            device: Arc::clone(device),
            raw,
            size: info.size,
            mapped: mapped.cast(),
            memory,
        })
    }

    /// Writes `parts` one after another from offset 0, then zeros up to `total` bytes. The
    /// write stops at `size`. A write to non-coherent memory ends with a flush.
    pub(super) fn write(&mut self, parts: &[&[u8]], total: u64) -> Result<(), DecodeError> {
        let length = total.min(self.size) as usize;
        // SAFETY: the map covers the buffer's `size` bytes, and `length` stays within it.
        // `&mut self` is the only path to the map, and no submit reads it, because the
        // engine waits for each decode.
        let target = unsafe { std::slice::from_raw_parts_mut(self.mapped, length) };
        fill_parts(target, parts);
        if self.memory.coherent {
            return Ok(());
        }
        let range = vk::MappedMemoryRange::default()
            .memory(self.memory.raw)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        // SAFETY: the range lies in the mapped memory.
        unsafe { self.device.raw.flush_mapped_memory_ranges(&[range]) }.map_err(|result| {
            vk_error(
                "vkFlushMappedMemoryRanges",
                result,
                format!("{total} bitstream bytes"),
            )
        })
    }

    /// Copies `length` bytes from `offset`. A read of non-coherent memory starts with an
    /// invalidate. The caller checks the range against `size`.
    pub(super) fn read(&self, offset: u64, length: usize) -> Result<Vec<u8>, DecodeError> {
        if !self.memory.coherent {
            let range = vk::MappedMemoryRange::default()
                .memory(self.memory.raw)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            // SAFETY: the range lies in the mapped memory.
            unsafe { self.device.raw.invalidate_mapped_memory_ranges(&[range]) }.map_err(
                |result| {
                    vk_error(
                        "vkInvalidateMappedMemoryRanges",
                        result,
                        format!("{length} bitstream bytes"),
                    )
                },
            )?;
        }
        // SAFETY: the caller keeps `offset` and `length` inside the buffer, the map covers it, and
        // the engine waits for each encode, so no submit writes these bytes during the copy.
        let source =
            unsafe { std::slice::from_raw_parts(self.mapped.add(offset as usize), length) };
        Ok(source.to_vec())
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        // SAFETY: this type created the buffer and the map, and no reference to the map
        // outlives `write`. The memory goes after this call.
        unsafe {
            self.device.raw.unmap_memory(self.memory.raw);
            self.device.raw.destroy_buffer(self.raw, None);
        }
    }
}

struct RawBuffer<'a> {
    device: &'a VulkanDevice,
    raw: vk::Buffer,
}

impl Drop for RawBuffer<'_> {
    fn drop(&mut self) {
        // SAFETY: no submit used this buffer, and nothing else owns it.
        unsafe { self.device.raw.destroy_buffer(self.raw, None) };
    }
}

/// A command pool. Its command buffers go with it.
pub(super) struct CommandPool {
    device: Arc<VulkanDevice>,
    raw: vk::CommandPool,
    pub(super) buffer: vk::CommandBuffer,
}

impl CommandPool {
    pub(super) fn new(device: &Arc<VulkanDevice>, family: u32) -> Result<Self, DecodeError> {
        let info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(family);
        // SAFETY: `family` is one this device opened a queue on.
        let raw = unsafe { device.raw.create_command_pool(&info, None) }.map_err(|result| {
            vk_error(
                "vkCreateCommandPool",
                result,
                format!("queue family {family}"),
            )
        })?;
        let mut pool = Self {
            device: Arc::clone(device),
            raw,
            buffer: vk::CommandBuffer::null(),
        };
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(raw)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: `raw` is the live pool above, and the count matches the result.
        let buffers = unsafe { device.raw.allocate_command_buffers(&info) }.map_err(|result| {
            vk_error(
                "vkAllocateCommandBuffers",
                result,
                format!("queue family {family}"),
            )
        })?;
        if let Some(&buffer) = buffers.first() {
            pool.buffer = buffer;
        }
        Ok(pool)
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        // SAFETY: this type created the pool, and the owner waited for the queue.
        unsafe { self.device.raw.destroy_command_pool(self.raw, None) };
    }
}

/// A timeline semaphore and the last value submitted for a signal.
pub(super) struct Timeline {
    device: Arc<VulkanDevice>,
    pub(super) raw: vk::Semaphore,
    pub(super) last: u64,
}

impl Timeline {
    pub(super) fn new(device: &Arc<VulkanDevice>) -> Result<Self, DecodeError> {
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
        // SAFETY: `info` and its chain live through the call.
        let raw = unsafe { device.raw.create_semaphore(&info, None) }
            .map_err(|result| vk_error("vkCreateSemaphore", result, "a timeline semaphore"))?;
        Ok(Self {
            device: Arc::clone(device),
            raw,
            last: 0,
        })
    }

    /// Waits on the host until the semaphore reaches `value`.
    pub(super) fn wait(&self, value: u64, timeout_ns: u64) -> Result<(), DecodeError> {
        let semaphores = [self.raw];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // SAFETY: `raw` is a live timeline semaphore, and each array holds one value.
        unsafe { self.device.raw.wait_semaphores(&info, timeout_ns) }.map_err(|result| {
            vk_error(
                "vkWaitSemaphores",
                result,
                format!("timeline value {value}"),
            )
        })
    }
}

impl Drop for Timeline {
    fn drop(&mut self) {
        // SAFETY: this type created the semaphore, and the owner waited for each queue.
        unsafe { self.device.raw.destroy_semaphore(self.raw, None) };
    }
}

/// A query pool of one decode-status query.
pub(super) struct StatusQuery {
    device: Arc<VulkanDevice>,
    pub(super) raw: vk::QueryPool,
}

impl StatusQuery {
    pub(super) fn new(
        device: &Arc<VulkanDevice>,
        profile: &vk::VideoProfileInfoKHR<'_>,
    ) -> Result<Self, DecodeError> {
        let mut info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::RESULT_STATUS_ONLY_KHR)
            .query_count(1);
        // `push_next` writes through the whole chain, and a shared reference holds the
        // codec structure at its end.
        info.p_next = std::ptr::from_ref(profile).cast();
        // SAFETY: `info` and the profile chain live through the call.
        let raw = unsafe { device.raw.create_query_pool(&info, None) }
            .map_err(|result| vk_error("vkCreateQueryPool", result, "one status query"))?;
        Ok(Self {
            device: Arc::clone(device),
            raw,
        })
    }
}

impl Drop for StatusQuery {
    fn drop(&mut self) {
        // SAFETY: this type created the pool, and the owner waited for the queue.
        unsafe { self.device.raw.destroy_query_pool(self.raw, None) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory_properties(flags: &[vk::MemoryPropertyFlags]) -> vk::PhysicalDeviceMemoryProperties {
        let mut properties = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: flags.len() as u32,
            ..Default::default()
        };
        for (slot, &property_flags) in properties.memory_types.iter_mut().zip(flags) {
            slot.property_flags = property_flags;
        }
        properties
    }

    #[test]
    fn parts_land_in_order_and_zeros_fill_the_rest() {
        let mut target = [0xFF; 8];
        fill_parts(&mut target, &[&[1, 2], &[], &[3]]);
        assert_eq!(target, [1, 2, 3, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn parts_stop_at_the_end_of_the_target() {
        let mut target = [0xFF; 3];
        fill_parts(&mut target, &[&[1, 2], &[3, 4], &[5]]);
        assert_eq!(target, [1, 2, 3]);
    }

    #[test]
    fn a_memory_type_takes_the_preference_when_it_can() {
        let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let visible = vk::MemoryPropertyFlags::HOST_VISIBLE;
        let coherent = vk::MemoryPropertyFlags::HOST_COHERENT;
        let properties = memory_properties(&[local, visible, visible | coherent]);
        assert_eq!(
            pick_memory_type(&properties, 0b111, visible, coherent),
            Some(2)
        );
        assert_eq!(
            pick_memory_type(&properties, 0b011, visible, coherent),
            Some(1)
        );
        assert_eq!(
            pick_memory_type(&properties, 0b001, visible, coherent),
            None
        );
        assert_eq!(
            pick_memory_type(&properties, 0b111, local, vk::MemoryPropertyFlags::empty()),
            Some(0)
        );
    }
}
