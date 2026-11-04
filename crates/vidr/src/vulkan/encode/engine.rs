//! One encode per call: the barriers, the coding scope, the submit, the wait, the feedback, and the
//! bitstream read. Codecs supply the picture and slot structures.
//!
//! The input image takes these steps when the encode and general queue families differ:
//!
//! 1. The general queue discards it, fills it from numr's buffers, and releases it to the encode
//!    family. `input::copy_frame` waits for this on the host.
//! 2. The encode queue acquires it and encodes from it.
//!
//! A DPB slot stays on the encode queue. A picture that reads a slot takes a barrier against the
//! encode that wrote it, and a picture that writes a slot discards what it held.
//!
//! Each coding scope binds the session parameters. The first scope's control command resets the
//! session and sets the rate control and quality level. Each later scope states that rate control
//! again.

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::ptr;
use std::sync::Arc;

use ash::vk;
use numr::runtime::wgpu::WgpuRuntime;

use crate::encode::EncodeFrame;
use crate::error::EncodeError;
use crate::vulkan::barriers::image_barrier;
use crate::vulkan::copy::whole_buffer;
use crate::vulkan::device::VulkanDevice;
use crate::vulkan::engine::{WAIT_NS, queues_finished};
use crate::vulkan::error::vk_error;
use crate::vulkan::objects::{CommandPool, HostBuffer, Timeline};
use crate::vulkan::pictures::PictureImage;
use crate::vulkan::recording::{Recorder, resource};
use crate::vulkan::session::{RawSessionSpec, SESSION_CALL, SessionParameters, VideoSession};

use super::caps::{EncodeCaps, bitstream_bytes, image_extent};
use super::feedback::{FeedbackQuery, picture_range};
use super::gop::{ACTIVE_REFERENCES, SLOTS};
use super::input::{acquire_barrier, copy_frame};
use super::pictures::EncodePictures;
use super::planes::{FrameLayout, PlaneRegions};
use super::profile::EncodeProfile;
use super::rate::RateState;

/// One DPB slot in an encode, with its codec structure.
#[derive(Clone, Copy, Debug)]
pub(super) struct SlotInfo {
    pub(super) slot: u32,
    pub(super) codec_info: *const c_void,
}

/// One picture to encode from the input image.
pub(super) struct EncodeRequest {
    /// The codec's picture structure.
    pub(super) picture_info: *const c_void,
    /// The slot the picture writes.
    pub(super) setup: SlotInfo,
    /// The slot the picture reads.
    pub(super) reference: Option<SlotInfo>,
    /// The picture's pts, for errors.
    pub(super) pts: i64,
}

/// A barrier that takes a slot for a read, against the encode that wrote it.
pub(super) fn read_barrier(picture: PictureImage) -> vk::ImageMemoryBarrier2<'static> {
    image_barrier(
        picture,
        (
            vk::PipelineStageFlags2::VIDEO_ENCODE_KHR,
            vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR,
            vk::ImageLayout::VIDEO_ENCODE_DPB_KHR,
        ),
        (
            vk::PipelineStageFlags2::VIDEO_ENCODE_KHR,
            vk::AccessFlags2::VIDEO_ENCODE_READ_KHR,
            vk::ImageLayout::VIDEO_ENCODE_DPB_KHR,
        ),
        (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED),
    )
}

/// A barrier that discards a slot and takes it for the next picture.
pub(super) fn setup_barrier(picture: PictureImage) -> vk::ImageMemoryBarrier2<'static> {
    image_barrier(
        picture,
        (
            vk::PipelineStageFlags2::VIDEO_ENCODE_KHR,
            vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR,
            vk::ImageLayout::UNDEFINED,
        ),
        (
            vk::PipelineStageFlags2::VIDEO_ENCODE_KHR,
            vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR | vk::AccessFlags2::VIDEO_ENCODE_READ_KHR,
            vk::ImageLayout::VIDEO_ENCODE_DPB_KHR,
        ),
        (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED),
    )
}

/// What the first coding scope's control command states.
const fn first_control_flags() -> vk::VideoCodingControlFlagsKHR {
    vk::VideoCodingControlFlagsKHR::from_raw(
        vk::VideoCodingControlFlagsKHR::RESET.as_raw()
            | vk::VideoCodingControlFlagsKHR::ENCODE_RATE_CONTROL.as_raw()
            | vk::VideoCodingControlFlagsKHR::ENCODE_QUALITY_LEVEL.as_raw(),
    )
}

/// Every object a submit or a plane copy can use. The fields drop in this order, and each comes
/// after what uses it.
struct EngineObjects {
    parameters: Option<SessionParameters>,
    session: VideoSession,
    pictures: EncodePictures,
    bitstream: HostBuffer,
    feedback: FeedbackQuery,
    pool: CommandPool,
    timeline: Timeline,
}

impl EngineObjects {
    /// The session parameters' handle, or null before they exist.
    fn parameters_handle(&self) -> vk::VideoSessionParametersKHR {
        self.parameters
            .as_ref()
            .map_or_else(vk::VideoSessionParametersKHR::null, |parameters| {
                parameters.raw
            })
    }
}

/// The error of a request that names a slot the session lacks.
fn absent_slot(slot: u32, what: &str) -> EncodeError {
    EncodeError::VulkanCore {
        reason: format!("{what} names DPB slot {slot} of {SLOTS}"),
    }
}

/// What a new engine holds.
pub(super) struct EngineSpec<'a> {
    pub(super) profile: &'a EncodeProfile,
    pub(super) caps: &'a EncodeCaps,
    pub(super) coded: (u32, u32),
    pub(super) layout: FrameLayout,
    pub(super) rate: RateState,
    pub(super) family: u32,
}

/// The session and everything one encode needs. Every codec shares it.
pub(super) struct EncodeEngine {
    // `Drop` frees these only once both queues are done with them.
    objects: ManuallyDrop<EngineObjects>,
    coded: (u32, u32),
    regions: PlaneRegions,
    rate: RateState,
    family: u32,
    /// Whether the session still needs its reset. Its first encode runs it.
    needs_reset: bool,
    device: Arc<VulkanDevice>,
}

impl EncodeEngine {
    /// Opens a session and its images. The session's reset waits for its first picture.
    pub(super) fn new(
        device: &Arc<VulkanDevice>,
        spec: EngineSpec<'_>,
    ) -> Result<Self, EncodeError> {
        let EngineSpec {
            profile,
            caps,
            coded,
            layout,
            rate,
            family,
        } = spec;
        let image = image_extent(coded, caps);
        let pictures = EncodePictures::new(device, profile, caps, image)?;
        let raw = RawSessionSpec {
            family,
            profile: profile.info(),
            picture_format: pictures.input_format,
            reference_format: pictures.reference_format,
            coded,
            slots: SLOTS,
            active_references: ACTIVE_REFERENCES,
            std_version: &caps.std_version,
        };
        let session =
            VideoSession::open(device, &raw, |result| profile.check(result, SESSION_CALL))?;
        let infos = [profile.info()];
        let mut list = vk::VideoProfileListInfoKHR::default().profiles(&infos);
        let size = bitstream_bytes(coded, profile.format().bytes_per_sample() as u64, caps);
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::VIDEO_ENCODE_DST_KHR)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut list);
        let bitstream = HostBuffer::for_reading(device, &info, "the encode bitstream buffer")?;
        let feedback = FeedbackQuery::new(device, &profile.info())?;
        let pool = CommandPool::new(device, family)?;
        let timeline = Timeline::new(device)?;
        Ok(Self {
            objects: ManuallyDrop::new(EngineObjects {
                parameters: None,
                session,
                pictures,
                bitstream,
                feedback,
                pool,
                timeline,
            }),
            coded,
            regions: layout.regions(image),
            rate,
            family,
            needs_reset: true,
            device: Arc::clone(device),
        })
    }

    pub(super) const fn rate(&self) -> &RateState {
        &self.rate
    }

    /// Creates the session parameters. `codec_info` points to the codec's create info, whose chain
    /// the quality level goes before.
    ///
    /// # Safety
    ///
    /// `codec_info` points to a valid chain for this session's codec, through the call.
    pub(super) unsafe fn create_parameters(
        &mut self,
        codec_info: *const c_void,
    ) -> Result<(), EncodeError> {
        let quality = vk::VideoEncodeQualityLevelInfoKHR {
            p_next: codec_info,
            quality_level: self.rate.quality_level(),
            ..Default::default()
        };
        // SAFETY: the caller states the codec's chain is valid, and `quality` lives through the
        // call.
        let parameters = unsafe {
            SessionParameters::new(
                &self.device,
                self.objects.session.raw,
                ptr::from_ref(&quality).cast(),
            )
        }?;
        self.objects.parameters = Some(parameters);
        Ok(())
    }

    /// The bytes of the parameter sets the session parameters hold, as the driver writes them.
    /// `codec_info` points to the codec's get info, or null for AV1.
    ///
    /// # Safety
    ///
    /// `codec_info` is null or points to a valid chain for this session's codec, through the call.
    pub(super) unsafe fn encoded_parameters(
        &self,
        codec_info: *const c_void,
    ) -> Result<Vec<u8>, EncodeError> {
        const CALL: &str = "vkGetEncodedVideoSessionParametersKHR";
        let info = vk::VideoEncodeSessionParametersGetInfoKHR {
            p_next: codec_info,
            video_session_parameters: self.objects.parameters_handle(),
            ..Default::default()
        };
        let mut feedback = vk::VideoEncodeSessionParametersFeedbackInfoKHR::default();
        let call = self
            .device
            .encode_fns
            .fp()
            .get_encoded_video_session_parameters_khr;
        let handle = self.device.raw.handle();
        let mut size = 0_usize;
        // SAFETY: `info` and `feedback` live through the call, `size` is writable, and a null data
        // pointer asks for the size.
        unsafe { call(handle, &info, &mut feedback, &mut size, ptr::null_mut()) }
            .result()
            .map_err(|result| vk_error(CALL, result, "the session parameters"))?;
        let mut bytes = vec![0_u8; size];
        // SAFETY: as above, and `bytes` holds `size` writable bytes.
        unsafe {
            call(
                handle,
                &info,
                &mut feedback,
                &mut size,
                bytes.as_mut_ptr().cast(),
            )
        }
        .result()
        .map_err(|result| vk_error(CALL, result, "the session parameters"))?;
        bytes.truncate(size);
        Ok(bytes)
    }

    /// Copies `frame` into the input image and waits for the copy.
    pub(super) fn load_frame(
        &mut self,
        frame: &EncodeFrame<'_, WgpuRuntime>,
    ) -> Result<(), EncodeError> {
        copy_frame(
            &self.device,
            frame,
            &self.regions,
            self.objects.pictures.input_picture(),
            self.family,
        )
    }

    /// Encodes the input image and waits for it. The result holds the picture's bytes.
    ///
    /// # Safety
    ///
    /// Each pointer in `request` points to a valid chain for this session's codec. Each referenced
    /// structure lives through the call. [`EncodeEngine::load_frame`] ran for this picture.
    pub(super) unsafe fn encode(
        &mut self,
        request: &EncodeRequest,
    ) -> Result<Vec<u8>, EncodeError> {
        let pts = request.pts;
        let what = format!("the frame at pts {pts}");
        let cmd = self.objects.pool.buffer;
        let recorder = Recorder {
            device: &self.device,
            cmd,
        };
        recorder.begin(&what)?;
        // SAFETY: the buffer is recording outside a coding scope, and the pool holds one query.
        unsafe {
            self.device
                .raw
                .cmd_reset_query_pool(cmd, self.objects.feedback.raw, 0, 1);
        }
        self.record_barriers(recorder, request, &what)?;
        // SAFETY: the caller states the codec chains are valid, and the buffer is recording.
        unsafe { self.record_scope(cmd, request, &what) }?;
        self.record_host_read(cmd);

        let value = {
            let Some(queue) = self.device.encode_queue() else {
                return Err(EncodeError::VulkanUnavailable {
                    reason: format!("{} opened with no encode queue", self.device.name()),
                });
            };
            recorder.end_and_submit(*queue, &mut self.objects.timeline, &what)?
        };
        self.objects.timeline.wait(value, WAIT_NS)?;
        self.needs_reset = false;

        let words = self.objects.feedback.read(pts)?;
        let capacity = self.objects.bitstream.size;
        let (offset, length) = picture_range(words, capacity, pts)?;
        Ok(self.objects.bitstream.read(offset, length)?)
    }

    fn slot_image(&self, slot: u32, what: &str) -> Result<PictureImage, EncodeError> {
        self.objects
            .pictures
            .slot_image(slot)
            .ok_or_else(|| absent_slot(slot, what))
    }

    fn record_barriers(
        &self,
        recorder: Recorder<'_>,
        request: &EncodeRequest,
        what: &str,
    ) -> Result<(), EncodeError> {
        let mut barriers = Vec::with_capacity(3);
        if self.family != self.device.general_family {
            barriers.push(acquire_barrier(
                self.objects.pictures.input_picture(),
                (self.device.general_family, self.family),
            ));
        }
        if let Some(reference) = &request.reference {
            barriers.push(read_barrier(self.slot_image(reference.slot, what)?));
        }
        barriers.push(setup_barrier(self.slot_image(request.setup.slot, what)?));
        recorder.pipeline_barrier(&barriers);
        Ok(())
    }

    /// Records the commands of one coding scope, in their one order.
    ///
    /// # Safety
    ///
    /// As [`EncodeEngine::encode`], with `cmd` recording on the encode family.
    unsafe fn record_scope(
        &self,
        cmd: vk::CommandBuffer,
        request: &EncodeRequest,
        what: &str,
    ) -> Result<(), EncodeError> {
        let objects = &*self.objects;
        let views = &objects.pictures.slot_views;
        let view = |slot: u32| {
            views
                .get(slot as usize)
                .map(|view| view.raw)
                .ok_or_else(|| absent_slot(slot, what))
        };
        let setup_resource = resource(view(request.setup.slot)?, self.coded);
        let reference_resource = match &request.reference {
            Some(reference) => Some((reference, resource(view(reference.slot)?, self.coded))),
            None => None,
        };
        // A slot with a picture goes in by its index. The slot of the new picture goes in
        // unassigned, and the encode assigns it.
        let mut bound: Vec<vk::VideoReferenceSlotInfoKHR<'_>> = Vec::with_capacity(2);
        if let Some((reference, picture)) = &reference_resource {
            bound.push(
                vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(reference.slot as i32)
                    .picture_resource(picture),
            );
        }
        bound.push(
            vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(-1)
                .picture_resource(&setup_resource),
        );
        let mut begin = vk::VideoBeginCodingInfoKHR::default()
            .video_session(objects.session.raw)
            .video_session_parameters(objects.parameters_handle())
            .reference_slots(&bound);
        if !self.needs_reset {
            begin.p_next = self.rate.begin_chain();
        }

        let mut control = vk::VideoCodingControlInfoKHR::default().flags(first_control_flags());
        control.p_next = self.rate.control_chain();

        let mut setup_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(request.setup.slot as i32)
            .picture_resource(&setup_resource);
        setup_slot.p_next = request.setup.codec_info;
        let references: Vec<vk::VideoReferenceSlotInfoKHR<'_>> = reference_resource
            .iter()
            .map(|(reference, picture)| {
                let mut slot = vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(reference.slot as i32)
                    .picture_resource(picture);
                slot.p_next = reference.codec_info;
                slot
            })
            .collect();
        let mut info = vk::VideoEncodeInfoKHR {
            p_next: request.picture_info,
            dst_buffer: objects.bitstream.raw,
            dst_buffer_offset: 0,
            dst_buffer_range: objects.bitstream.size,
            src_picture_resource: resource(objects.pictures.input_view.raw, self.coded),
            p_setup_reference_slot: &setup_slot,
            ..Default::default()
        };
        info.reference_slot_count = references.len() as u32;
        info.p_reference_slots = references.as_ptr();

        let end = vk::VideoEndCodingInfoKHR::default();
        let device = &self.device;
        let fns = device.video_fns.fp();
        let query = objects.feedback.raw;
        // SAFETY: the buffer is recording on the encode family, the barriers set each layout, and
        // the caller states the codec chains are valid. Each structure here lives through these
        // calls. The first picture binds no active slot, so its reset invalidates nothing it reads.
        unsafe {
            (fns.cmd_begin_video_coding_khr)(cmd, &begin);
            if self.needs_reset {
                (fns.cmd_control_video_coding_khr)(cmd, &control);
            }
            device
                .raw
                .cmd_begin_query(cmd, query, 0, vk::QueryControlFlags::empty());
            (device.encode_fns.fp().cmd_encode_video_khr)(cmd, &info);
            device.raw.cmd_end_query(cmd, query, 0);
            (fns.cmd_end_video_coding_khr)(cmd, &end);
        }
        Ok(())
    }

    /// Makes the encode's write to the bitstream buffer visible to the host.
    fn record_host_read(&self, cmd: vk::CommandBuffer) {
        let barrier = [whole_buffer(
            self.objects.bitstream.raw,
            vk::PipelineStageFlags2::VIDEO_ENCODE_KHR,
            vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR,
            vk::PipelineStageFlags2::HOST,
            vk::AccessFlags2::HOST_READ,
        )];
        let info = vk::DependencyInfo::default().buffer_memory_barriers(&barrier);
        // SAFETY: the buffer is recording outside a coding scope, and the barrier names a live
        // buffer.
        unsafe { self.device.sync2.cmd_pipeline_barrier2(cmd, &info) };
    }
}

impl Drop for EncodeEngine {
    /// Waits for both queues, each for a fixed time. See `queues_finished`.
    ///
    /// When a wait times out or the device is lost, the objects stay allocated for the life of the
    /// process. A GPU can still use them, and a destroy in flight is undefined behaviour. The caller
    /// already holds the error of the stall, so `drop` reports nothing.
    fn drop(&mut self) {
        if queues_finished(&self.device, &self.objects.timeline) {
            // SAFETY: both queues finished with the objects, `drop` runs once, and nothing reads
            // `objects` after this line.
            unsafe { ManuallyDrop::drop(&mut self.objects) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PICTURE: PictureImage = PictureImage {
        image: vk::Image::null(),
        layer: 1,
    };

    #[test]
    fn a_read_waits_for_the_write_in_the_dpb_layout() {
        let barrier = read_barrier(PICTURE);
        assert_eq!(barrier.old_layout, vk::ImageLayout::VIDEO_ENCODE_DPB_KHR);
        assert_eq!(barrier.new_layout, vk::ImageLayout::VIDEO_ENCODE_DPB_KHR);
        assert_eq!(
            barrier.src_access_mask,
            vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR
        );
        assert_eq!(
            barrier.dst_access_mask,
            vk::AccessFlags2::VIDEO_ENCODE_READ_KHR
        );
        assert_eq!(barrier.subresource_range.base_array_layer, 1);
    }

    #[test]
    fn a_setup_slot_discards_what_it_held() {
        let barrier = setup_barrier(PICTURE);
        assert_eq!(barrier.old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(barrier.new_layout, vk::ImageLayout::VIDEO_ENCODE_DPB_KHR);
        assert!(
            barrier
                .dst_access_mask
                .contains(vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR)
        );
    }

    #[test]
    fn the_first_control_resets_and_sets_rate_and_quality() {
        let flags = first_control_flags();
        assert!(flags.contains(vk::VideoCodingControlFlagsKHR::RESET));
        assert!(flags.contains(vk::VideoCodingControlFlagsKHR::ENCODE_RATE_CONTROL));
        assert!(flags.contains(vk::VideoCodingControlFlagsKHR::ENCODE_QUALITY_LEVEL));
    }
}
