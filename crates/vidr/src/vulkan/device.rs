//! [`VulkanDevice`]: one Vulkan device that numr's WebGPU backend and the video
//! queue share.
//!
//! wgpu-hal opens the device with the video extensions and the decode queue added.
//! wgpu wraps it, and `WgpuClient::install` makes it numr's device 0. So a tensor's
//! buffer and a decoded image live on one `VkDevice`, and a plane copy stays on it.

use std::ffi::CStr;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use ash::vk;
use numr::runtime::wgpu::{WgpuClient, WgpuDevice};
use wgpu::hal::api::Vulkan;
use wgpu::hal::vulkan as hal_vulkan;

use super::probe::{self, DeviceFacts, Placement};
use super::validation::{self, ValidationCapture};
use crate::error::DecodeError;

/// The numr device index that vidr installs.
const NUMR_INDEX: usize = 0;
/// One queue, at full priority.
static ONE_QUEUE: [f32; 1] = [1.0];
/// wgpu's queue and the decode queue, in one family.
static TWO_QUEUES: [f32; 2] = [1.0, 1.0];

static SHARED: Mutex<Option<Arc<VulkanDevice>>> = Mutex::new(None);

/// How to open a [`VulkanDevice`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VulkanOptions {
    /// Turns on `VK_LAYER_KHRONOS_validation` when the loader lists it, and keeps
    /// each error for [`VulkanDevice::validation_errors`].
    pub validation: bool,
}

/// The Vulkan device behind numr's WebGPU device 0, with a video decode queue.
pub struct VulkanDevice {
    pub(super) instance: ash::Instance,
    pub(super) raw: ash::Device,
    pub(super) physical: vk::PhysicalDevice,
    pub(super) wgpu_device: wgpu::Device,
    pub(super) wgpu_queue: wgpu::Queue,
    pub(super) general_family: u32,
    pub(super) video_family: u32,
    pub(super) status_queries: bool,
    pub(super) video_instance_fns: ash::khr::video_queue::Instance,
    pub(super) video_fns: ash::khr::video_queue::Device,
    pub(super) decode_fns: ash::khr::video_decode_queue::Device,
    pub(super) sync2: ash::khr::synchronization2::Device,
    video_queue: Mutex<vk::Queue>,
    client: WgpuClient,
    numr_device: WgpuDevice,
    name: String,
    /// Declared before `_wgpu_instance`, so the messenger goes first.
    validation: Option<ValidationCapture>,
    _wgpu_instance: wgpu::Instance,
}

impl VulkanDevice {
    /// The process's device: opens it on the first call and returns it after that.
    /// `options` count only on a call that opens.
    ///
    /// # Errors
    ///
    /// [`DecodeError::VulkanUnavailable`] when no device decodes H.264, when the open
    /// fails, or when numr already holds a device at index 0. Call this before the
    /// first tensor on numr's WebGPU backend.
    pub fn open_shared(options: VulkanOptions) -> Result<Arc<Self>, DecodeError> {
        let mut slot = SHARED.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(device) = &*slot {
            return Ok(Arc::clone(device));
        }
        let device = Arc::new(Self::open(options)?);
        *slot = Some(Arc::clone(&device));
        Ok(device)
    }

    fn open(options: VulkanOptions) -> Result<Self, DecodeError> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::VULKAN;
        descriptor.flags = if options.validation {
            wgpu::InstanceFlags::VALIDATION | wgpu::InstanceFlags::DEBUG
        } else {
            wgpu::InstanceFlags::empty()
        };
        let wgpu_instance = wgpu::Instance::new(descriptor);
        let (adapter, facts, placement) = choose_adapter(&wgpu_instance)?;

        let features = adapter.features() & wgpu::Features::SUBGROUP;
        let limits = adapter.limits();
        let open_device = open_hal_device(&adapter, &placement, features, &limits)?;
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("vidr Vulkan Video device"),
            required_features: features,
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::default(),
        };
        // SAFETY: wgpu-hal opened `open_device` from this adapter, with these features
        // and limits.
        let (wgpu_device, wgpu_queue) =
            unsafe { adapter.create_device_from_hal::<Vulkan>(open_device, &descriptor) }
                .map_err(|error| unavailable(format!("wgpu refuses the device: {error}")))?;

        // SAFETY: vidr destroys nothing behind the guard, and it keeps `wgpu_device`,
        // which owns the raw device, in the returned value.
        let hal = unsafe { wgpu_device.as_hal::<Vulkan>() }
            .ok_or_else(|| unavailable("wgpu reports no Vulkan device".to_owned()))?;
        let raw = hal.raw_device().clone();
        let physical = hal.raw_physical_device();
        let general_family = hal.queue_family_index();
        let shared = hal.shared_instance();
        let instance = shared.raw_instance().clone();
        let validation = capture(options, shared)?;
        let video_instance_fns = ash::khr::video_queue::Instance::new(shared.entry(), &instance);
        drop(hal);

        // SAFETY: the device was created with this family and queue index.
        let video_queue =
            unsafe { raw.get_device_queue(placement.video_family, placement.video_queue_index) };
        let client = WgpuClient::install(
            NUMR_INDEX,
            &adapter,
            wgpu_device.clone(),
            wgpu_queue.clone(),
        )
        .map_err(|error| {
            unavailable(format!(
                "numr refuses the device: {error}. Open Vulkan Video before the first WebGPU \
                 tensor"
            ))
        })?;
        Ok(Self {
            video_fns: ash::khr::video_queue::Device::new(&instance, &raw),
            decode_fns: ash::khr::video_decode_queue::Device::new(&instance, &raw),
            sync2: ash::khr::synchronization2::Device::new(&instance, &raw),
            video_instance_fns,
            instance,
            raw,
            physical,
            wgpu_device,
            wgpu_queue,
            general_family,
            video_family: placement.video_family,
            status_queries: placement.status_queries,
            video_queue: Mutex::new(video_queue),
            client,
            numr_device: WgpuDevice::new(NUMR_INDEX),
            name: facts.name,
            validation,
            _wgpu_instance: wgpu_instance,
        })
    }

    /// The numr client on this device.
    pub fn client(&self) -> &WgpuClient {
        &self.client
    }

    /// The numr device that frames live on.
    pub fn numr_device(&self) -> &WgpuDevice {
        &self.numr_device
    }

    /// The GPU's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether the validation layer runs on this device.
    pub fn validation_active(&self) -> bool {
        self.validation.is_some()
    }

    /// Every validation error so far. Empty when validation does not run.
    pub fn validation_errors(&self) -> Vec<String> {
        self.validation
            .as_ref()
            .map_or_else(Vec::new, ValidationCapture::errors)
    }

    /// Whether video and general work run in different queue families.
    pub(super) fn transfers_ownership(&self) -> bool {
        self.video_family != self.general_family
    }

    /// The decode queue. The guard is its external synchronisation.
    pub(super) fn video_queue(&self) -> MutexGuard<'_, vk::Queue> {
        self.video_queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

fn unavailable(reason: String) -> DecodeError {
    DecodeError::VulkanUnavailable { reason }
}

/// A discrete GPU first, then the driver's order.
fn choose_adapter(
    instance: &wgpu::Instance,
) -> Result<(wgpu::Adapter, DeviceFacts, Placement), DecodeError> {
    let adapters = probe::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
    let mut reasons = Vec::new();
    let mut chosen = None;
    for adapter in adapters {
        // SAFETY: the guard's handles stay live through this loop body, and vidr
        // destroys none of them.
        let facts = match unsafe { adapter.as_hal::<Vulkan>() } {
            Some(hal) => probe::read_facts(
                hal.shared_instance().raw_instance(),
                hal.raw_physical_device(),
            ),
            None => continue,
        };
        let placed = facts.and_then(|facts| probe::place(&facts).map(|placed| (facts, placed)));
        match placed {
            Ok((facts, placement)) => {
                let discrete = facts.discrete;
                if chosen.is_none() || discrete {
                    chosen = Some((adapter, facts, placement));
                }
                if discrete {
                    break;
                }
            }
            Err(reason) => reasons.push(reason),
        }
    }
    chosen.ok_or_else(|| unavailable(probe::no_device_reason(&reasons)))
}

fn open_hal_device(
    adapter: &wgpu::Adapter,
    placement: &Placement,
    features: wgpu::Features,
    limits: &wgpu::Limits,
) -> Result<wgpu::hal::OpenDevice<Vulkan>, DecodeError> {
    let mut synchronization2 =
        vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
    let callback = device_callback(*placement, &mut synchronization2);
    // SAFETY: the guard's adapter stays live through the call. The callback only adds
    // extensions and a feature `place` checked, and adds or widens one queue create info.
    unsafe {
        let hal = adapter
            .as_hal::<Vulkan>()
            .ok_or_else(|| unavailable("wgpu reports no Vulkan adapter".to_owned()))?;
        hal.open_with_callback(
            features,
            limits,
            &wgpu::MemoryHints::Performance,
            Some(callback),
        )
    }
    .map_err(|error| unavailable(format!("vkCreateDevice fails: {error}")))
}

/// What the device adds to wgpu-hal's: the video extensions, the queue, and
/// `synchronization2`, which the video extensions need.
fn device_callback<'this>(
    placement: Placement,
    synchronization2: &'this mut vk::PhysicalDeviceSynchronization2Features<'this>,
) -> Box<hal_vulkan::CreateDeviceCallback<'this>> {
    Box::new(move |args| {
        for name in probe::REQUIRED_EXTENSIONS {
            add_extension(args.extensions, name);
        }
        let priorities: &'static [f32] = if placement.video_family == probe::GENERAL_FAMILY {
            &TWO_QUEUES
        } else {
            &ONE_QUEUE
        };
        match args
            .queue_create_infos
            .iter_mut()
            .find(|info| info.queue_family_index == placement.video_family)
        {
            Some(info) => *info = info.queue_priorities(priorities),
            None => args.queue_create_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(placement.video_family)
                    .queue_priorities(priorities),
            ),
        }
        let info = std::mem::take(args.create_info);
        *args.create_info = info.push_next(synchronization2);
    })
}

fn add_extension(extensions: &mut Vec<&'static CStr>, name: &'static CStr) {
    if !extensions.contains(&name) {
        extensions.push(name);
    }
}

fn capture(
    options: VulkanOptions,
    shared: &hal_vulkan::InstanceShared,
) -> Result<Option<ValidationCapture>, DecodeError> {
    let wanted = options.validation
        && validation::layer_present(shared.entry())
        && shared.extensions().contains(&ash::ext::debug_utils::NAME);
    if !wanted {
        return Ok(None);
    }
    ValidationCapture::new(shared.entry(), shared.raw_instance()).map(Some)
}
