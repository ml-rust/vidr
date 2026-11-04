//! What a physical device offers, and the choice of device and queue families.
//!
//! [`place`] is pure, so a test states any device. wgpu-hal gives its own queue
//! family 0, so vidr needs a family with transfer and compute there.

use std::ffi::CStr;
use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

use ash::vk;
use wgpu::hal::api::Vulkan;

use super::gap::{self, PhysicalDeviceVideoDecodeVp9Features};
use crate::error::DecodeError;

/// The family wgpu-hal opens its queue on.
pub(super) const GENERAL_FAMILY: u32 = 0;

/// Device extensions H.264 decode needs.
pub(super) const REQUIRED_EXTENSIONS: [&CStr; 4] = [
    ash::khr::video_queue::NAME,
    ash::khr::video_decode_queue::NAME,
    ash::khr::video_decode_h264::NAME,
    ash::khr::synchronization2::NAME,
];

/// Device extensions for one more codec. A device can lack them, and a stream of that codec
/// is then refused by name.
pub(super) const CODEC_EXTENSIONS: [(&CStr, vk::VideoCodecOperationFlagsKHR); 3] = [
    (
        ash::khr::video_decode_h265::NAME,
        vk::VideoCodecOperationFlagsKHR::DECODE_H265,
    ),
    (gap::VIDEO_DECODE_VP9_NAME, gap::DECODE_VP9),
    (
        vk::KHR_VIDEO_DECODE_AV1_NAME,
        vk::VideoCodecOperationFlagsKHR::DECODE_AV1,
    ),
];

/// One queue family as the driver lists it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FamilyFacts {
    pub(super) flags: vk::QueueFlags,
    pub(super) count: u32,
    pub(super) video_ops: vk::VideoCodecOperationFlagsKHR,
    pub(super) status_queries: bool,
}

/// One physical device as the driver lists it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DeviceFacts {
    pub(super) name: String,
    pub(super) discrete: bool,
    pub(super) api_version: u32,
    pub(super) extensions: Vec<String>,
    pub(super) families: Vec<FamilyFacts>,
    pub(super) synchronization2: bool,
    pub(super) timeline_semaphore: bool,
    /// The `videoDecodeVP9` feature.
    pub(super) video_decode_vp9: bool,
}

/// Where a device's video work runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Placement {
    /// The family of the decode queue.
    pub(super) video_family: u32,
    /// The queue index inside that family.
    pub(super) video_queue_index: u32,
    /// Whether the family reports decode status through queries.
    pub(super) status_queries: bool,
    /// The codecs that family decodes and whose extensions the device lists.
    pub(super) decode_ops: vk::VideoCodecOperationFlagsKHR,
}

/// Places H.264 decode on `facts`, or states what the device lacks.
pub(super) fn place(facts: &DeviceFacts) -> Result<Placement, String> {
    if facts.api_version < vk::API_VERSION_1_2 {
        return Err(format!(
            "{} offers Vulkan {}.{}, and vidr needs 1.2",
            facts.name,
            vk::api_version_major(facts.api_version),
            vk::api_version_minor(facts.api_version)
        ));
    }
    for required in REQUIRED_EXTENSIONS {
        let required = required.to_string_lossy();
        if !facts.extensions.iter().any(|name| *name == *required) {
            return Err(format!("{} lacks {required}", facts.name));
        }
    }
    if !facts.synchronization2 {
        return Err(format!("{} lacks the synchronization2 feature", facts.name));
    }
    if !facts.timeline_semaphore {
        return Err(format!(
            "{} lacks the timelineSemaphore feature",
            facts.name
        ));
    }
    let Some(general) = facts.families.first() else {
        return Err(format!("{} lists no queue family", facts.name));
    };
    if !general
        .flags
        .contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER)
    {
        return Err(format!(
            "queue family {GENERAL_FAMILY} of {} lacks compute or transfer",
            facts.name
        ));
    }
    let video = facts.families.iter().enumerate().find(|(_, family)| {
        family.flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR)
            && family
                .video_ops
                .contains(vk::VideoCodecOperationFlagsKHR::DECODE_H264)
    });
    let Some((index, family)) = video else {
        return Err(format!("{} has no H.264 decode queue", facts.name));
    };
    let video_family = index as u32;
    let shares_general = video_family == GENERAL_FAMILY;
    if shares_general && family.count < 2 {
        return Err(format!(
            "{} decodes on queue family {GENERAL_FAMILY}, which holds one queue, and wgpu holds it",
            facts.name
        ));
    }
    Ok(Placement {
        video_family,
        video_queue_index: u32::from(shares_general),
        status_queries: family.status_queries,
        decode_ops: usable_ops(facts, family.video_ops),
    })
}

/// The codecs of `family_ops` that the device can turn on: H.264, which [`place`] requires,
/// and each other codec whose extension the device lists. VP9 also needs its feature.
fn usable_ops(
    facts: &DeviceFacts,
    family_ops: vk::VideoCodecOperationFlagsKHR,
) -> vk::VideoCodecOperationFlagsKHR {
    let mut ops = vk::VideoCodecOperationFlagsKHR::DECODE_H264;
    for (name, op) in CODEC_EXTENSIONS {
        let name = name.to_string_lossy();
        let listed = facts.extensions.iter().any(|listed| *listed == *name);
        let enabled = op != gap::DECODE_VP9 || facts.video_decode_vp9;
        if listed && enabled && family_ops.contains(op) {
            ops |= op;
        }
    }
    ops
}

/// Runs a future that never waits on an event. wgpu's adapter list on native is one.
pub(super) fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        std::thread::yield_now();
    }
}

/// Reads what the driver lists for `physical`.
pub(super) fn read_facts(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> Result<DeviceFacts, String> {
    // SAFETY: `physical` comes from `instance`, which stays live through this call.
    let properties = unsafe { instance.get_physical_device_properties(physical) };
    let name = properties.device_name_as_c_str().map_or_else(
        |_| "an unnamed GPU".to_owned(),
        |text| text.to_string_lossy().into_owned(),
    );
    // SAFETY: as above.
    let listed = unsafe { instance.enumerate_device_extension_properties(physical) }
        .map_err(|result| format!("{name} lists no extensions: {result:?}"))?;
    let extensions = listed
        .iter()
        .filter_map(|extension| extension.extension_name_as_c_str().ok())
        .map(|text| text.to_string_lossy().into_owned())
        .collect();

    let mut synchronization2 = vk::PhysicalDeviceSynchronization2Features::default();
    let mut timeline = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
    let mut vp9 = PhysicalDeviceVideoDecodeVp9Features {
        s_type: gap::TYPE_VP9_FEATURES,
        p_next: std::ptr::null_mut(),
        video_decode_vp9: vk::FALSE,
    };
    // A driver without the extension does not know this structure, so it stays out.
    let lists_vp9 = listed
        .iter()
        .filter_map(|extension| extension.extension_name_as_c_str().ok())
        .any(|name| name == gap::VIDEO_DECODE_VP9_NAME);
    if lists_vp9 {
        timeline.p_next = std::ptr::from_mut(&mut vp9).cast();
    }
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut synchronization2)
        .push_next(&mut timeline);
    // SAFETY: `physical` is valid, and each chained structure lives through the call.
    unsafe { instance.get_physical_device_features2(physical, &mut features) };

    Ok(DeviceFacts {
        name,
        discrete: properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
        api_version: properties.api_version,
        extensions,
        families: read_families(instance, physical),
        synchronization2: synchronization2.synchronization2 == vk::TRUE,
        timeline_semaphore: timeline.timeline_semaphore == vk::TRUE,
        video_decode_vp9: vp9.video_decode_vp9 == vk::TRUE,
    })
}

fn read_families(instance: &ash::Instance, physical: vk::PhysicalDevice) -> Vec<FamilyFacts> {
    // SAFETY: `physical` comes from `instance`, and the call writes one count.
    let count = unsafe { instance.get_physical_device_queue_family_properties2_len(physical) };
    let mut video = vec![vk::QueueFamilyVideoPropertiesKHR::default(); count];
    let mut status = vec![vk::QueueFamilyQueryResultStatusPropertiesKHR::default(); count];
    let mut families: Vec<vk::QueueFamilyProperties2<'_>> = video
        .iter_mut()
        .zip(status.iter_mut())
        .map(|(video, status)| {
            vk::QueueFamilyProperties2::default()
                .push_next(video)
                .push_next(status)
        })
        .collect();
    // SAFETY: `families` holds `count` elements, and each chain lives through the call.
    unsafe { instance.get_physical_device_queue_family_properties2(physical, &mut families) };
    let listed: Vec<(vk::QueueFlags, u32)> = families
        .iter()
        .map(|family| {
            let properties = family.queue_family_properties;
            (properties.queue_flags, properties.queue_count)
        })
        .collect();
    drop(families);
    listed
        .into_iter()
        .zip(video.iter().zip(&status))
        .map(|((flags, count), (video, status))| FamilyFacts {
            flags,
            count,
            video_ops: video.video_codec_operations,
            status_queries: status.query_result_status_support == vk::TRUE,
        })
        .collect()
}

/// Checks that a device on this machine decodes H.264 on Vulkan Video. It opens no
/// device, so numr can still open its own.
///
/// # Errors
///
/// [`DecodeError::VulkanUnavailable`] states what each device lacks.
pub fn vulkan_video_available() -> Result<(), DecodeError> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    descriptor.flags = wgpu::InstanceFlags::empty();
    let instance = wgpu::Instance::new(descriptor);
    let adapters = block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
    let mut reasons = Vec::new();
    for adapter in &adapters {
        // SAFETY: the guard's handles stay live through this loop body, and vidr
        // destroys none of them.
        let Some(hal) = (unsafe { adapter.as_hal::<Vulkan>() }) else {
            continue;
        };
        let raw = hal.shared_instance().raw_instance();
        match read_facts(raw, hal.raw_physical_device()).and_then(|facts| place(&facts)) {
            Ok(_) => return Ok(()),
            Err(reason) => reasons.push(reason),
        }
    }
    Err(DecodeError::VulkanUnavailable {
        reason: no_device_reason(&reasons),
    })
}

/// Why no device fits, from what each device lacks.
pub(super) fn no_device_reason(reasons: &[String]) -> String {
    if reasons.is_empty() {
        "no Vulkan device answers".to_owned()
    } else {
        reasons.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn general() -> FamilyFacts {
        FamilyFacts {
            flags: vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
            count: 16,
            video_ops: vk::VideoCodecOperationFlagsKHR::empty(),
            status_queries: false,
        }
    }

    fn decode_family() -> FamilyFacts {
        FamilyFacts {
            flags: vk::QueueFlags::TRANSFER | vk::QueueFlags::VIDEO_DECODE_KHR,
            count: 1,
            video_ops: vk::VideoCodecOperationFlagsKHR::DECODE_H264,
            status_queries: true,
        }
    }

    fn device(families: Vec<FamilyFacts>) -> DeviceFacts {
        DeviceFacts {
            name: "test GPU".to_owned(),
            discrete: true,
            api_version: vk::API_VERSION_1_3,
            extensions: REQUIRED_EXTENSIONS
                .iter()
                .map(|name| name.to_string_lossy().into_owned())
                .collect(),
            families,
            synchronization2: true,
            timeline_semaphore: true,
            video_decode_vp9: false,
        }
    }

    #[test]
    fn decode_goes_on_the_first_family_with_h264() {
        let encode = FamilyFacts {
            flags: vk::QueueFlags::VIDEO_ENCODE_KHR,
            video_ops: vk::VideoCodecOperationFlagsKHR::ENCODE_H264,
            ..decode_family()
        };
        let facts = device(vec![general(), encode, decode_family()]);
        assert_eq!(
            place(&facts),
            Ok(Placement {
                video_family: 2,
                video_queue_index: 0,
                status_queries: true,
                decode_ops: vk::VideoCodecOperationFlagsKHR::DECODE_H264,
            })
        );
    }

    #[test]
    fn a_family_shared_with_wgpu_takes_the_second_queue() {
        let both = FamilyFacts {
            flags: general().flags | vk::QueueFlags::VIDEO_DECODE_KHR,
            video_ops: vk::VideoCodecOperationFlagsKHR::DECODE_H264,
            ..general()
        };
        let placed = place(&device(vec![both])).expect("placed");
        assert_eq!((placed.video_family, placed.video_queue_index), (0, 1));

        let single = FamilyFacts { count: 1, ..both };
        let refused = place(&device(vec![single])).expect_err("refused");
        assert!(refused.contains("one queue"), "{refused}");
    }

    /// AV1 has no feature structure: its extension and its family decide.
    #[test]
    fn av1_needs_the_family_and_the_extension() {
        let av1 = vk::VideoCodecOperationFlagsKHR::DECODE_AV1;
        let family = FamilyFacts {
            video_ops: vk::VideoCodecOperationFlagsKHR::DECODE_H264 | av1,
            ..decode_family()
        };
        let mut facts = device(vec![general(), family]);
        let ops = |facts: &DeviceFacts| place(facts).expect("placed").decode_ops;
        assert!(!ops(&facts).contains(av1));
        facts.extensions.push("VK_KHR_video_decode_av1".to_owned());
        assert!(ops(&facts).contains(av1));
        let narrow = device(vec![general(), decode_family()]);
        assert!(!ops(&narrow).contains(av1));
    }

    #[test]
    fn other_codecs_need_the_family_the_extension_and_the_feature() {
        let hevc_vp9 = vk::VideoCodecOperationFlagsKHR::DECODE_H264
            | vk::VideoCodecOperationFlagsKHR::DECODE_H265
            | gap::DECODE_VP9;
        let family = FamilyFacts {
            video_ops: hevc_vp9,
            ..decode_family()
        };
        let mut facts = device(vec![general(), family]);
        let ops = |facts: &DeviceFacts| place(facts).expect("placed").decode_ops;
        assert_eq!(ops(&facts), vk::VideoCodecOperationFlagsKHR::DECODE_H264);

        facts.extensions.push("VK_KHR_video_decode_h265".to_owned());
        facts.extensions.push("VK_KHR_video_decode_vp9".to_owned());
        assert_eq!(
            ops(&facts),
            vk::VideoCodecOperationFlagsKHR::DECODE_H264
                | vk::VideoCodecOperationFlagsKHR::DECODE_H265,
            "VP9 waits for its feature"
        );
        facts.video_decode_vp9 = true;
        assert_eq!(ops(&facts), hevc_vp9);

        // A family that lists fewer codecs than the device's extensions gives fewer.
        let mut narrow = facts.clone();
        narrow.families[1].video_ops = vk::VideoCodecOperationFlagsKHR::DECODE_H264;
        assert_eq!(ops(&narrow), vk::VideoCodecOperationFlagsKHR::DECODE_H264);
    }

    #[test]
    fn each_missing_part_is_named() {
        let mut facts = device(vec![general(), decode_family()]);
        facts
            .extensions
            .retain(|name| name != "VK_KHR_video_decode_h264");
        let refused = place(&facts).expect_err("refused");
        assert!(refused.contains("VK_KHR_video_decode_h264"), "{refused}");

        let facts = device(vec![general()]);
        assert!(
            place(&facts)
                .expect_err("refused")
                .contains("H.264 decode queue")
        );

        let mut facts = device(vec![general(), decode_family()]);
        facts.synchronization2 = false;
        assert!(
            place(&facts)
                .expect_err("refused")
                .contains("synchronization2")
        );

        let mut facts = device(vec![general(), decode_family()]);
        facts.api_version = vk::API_VERSION_1_1;
        assert!(place(&facts).expect_err("refused").contains("1.1"));
    }
}
