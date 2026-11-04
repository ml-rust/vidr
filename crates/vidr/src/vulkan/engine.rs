//! One decode per call: the bitstream write, the barriers, the coding scope, the submit,
//! the wait, and the status. Codecs supply the bytes, the slots, and the `pNext` chains.
//!
//! A slot's image takes these steps in coincide mode, when the video and general queue
//! families differ:
//!
//! 1. The video queue decodes into it, then releases it to the general family.
//! 2. The general queue acquires it and copies each plane to a numr buffer.
//! 3. For a reference, the general queue releases it, and the next decode acquires it.
//!
//! When the families match, step 1 makes the decode's write visible to later reads on
//! the queue. No transfer runs. In distinct mode only the output image takes steps 1
//! and 2. A slot stays in the DPB layout on the video queue. So the next decode that reads
//! it takes a barrier against the decode that wrote it. A decode with no output also skips
//! steps 1 and 2. [`DecodeEngine::read_slot`] takes them later for a frame shown again.
//! A use of an empty image starts from `UNDEFINED`, which discards its contents and needs
//! no transfer.
//!
//! An AV1 frame with film grain writes the output image in either mode. Its slot acts as in
//! distinct mode. So each step above reads the mode of its request.
//!
//! A new session takes its reset inside the coding scope of its first picture. A decode
//! scope needs session parameters bound, and that scope has them. So the same call order
//! holds for every codec.

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::sync::Arc;

use ash::vk;

use super::barriers::read_barrier;
use super::caps::{DecodeCaps, DpbMode, align_up, image_extent, separate_images};
use super::copy::WAIT;
use super::device::VulkanDevice;
use super::error::vk_error;
use super::objects::{CommandPool, HostBuffer, StatusQuery, Timeline};
use super::pictures::{
    PictureImage, PictureStore, SlotState, copy_returns_image, state_after_copy,
};
use super::recording::{Recorder, ScopeObjects, decode_layout, request_mode};
use super::request_checks::{RequestLimits, absent_slot, check_request};
use super::session::{SessionParameters, SessionSpec, VideoProfile, VideoSession, image_format};
use crate::error::DecodeError;

/// How long a host wait lasts before it reports an error.
pub(super) const WAIT_NS: u64 = WAIT.as_nanos() as u64;
/// The smallest bitstream buffer.
const MIN_BITSTREAM_BYTES: u64 = 1 << 20;
/// `VK_QUERY_RESULT_STATUS_COMPLETE_KHR`.
pub(super) const STATUS_COMPLETE: i32 = 1;

/// The name of a `VkQueryResultStatusKHR` value.
pub(super) const fn status_name(status: i32) -> &'static str {
    match status {
        -1 => "ERROR",
        0 => "NOT_READY",
        1 => "COMPLETE",
        -1_000_299_000 => "INSUFFICIENT_BITSTREAM_BUFFER_RANGE",
        _ => "unknown",
    }
}

/// One DPB slot in a decode, with its codec structure. A codec with no slot structure
/// passes null.
#[derive(Clone, Copy, Debug)]
pub(super) struct SlotRef {
    pub(super) slot: u32,
    pub(super) codec_info: *const c_void,
    /// The coded size of the picture this slot holds, or takes.
    pub(super) coded: (u32, u32),
}

/// One picture to decode.
pub(super) struct DecodeRequest<'a> {
    /// The picture's bytes in the form the codec needs, in pieces.
    pub(super) bitstream: &'a [&'a [u8]],
    /// The codec's picture structure.
    pub(super) picture_info: *const c_void,
    /// The slot the picture takes.
    pub(super) setup: SlotRef,
    /// Every active reference.
    pub(super) references: &'a [SlotRef],
    /// Whether the picture stays a reference.
    pub(super) reference: bool,
    /// Whether a plane copy follows this decode. A decode with no output moves no image to
    /// the copy queue.
    pub(super) output: bool,
    /// Whether the decode writes the output image, whatever the session's DPB mode. An AV1
    /// frame with film grain does: `VUID-vkCmdDecodeVideoKHR-pDecodeInfo-09249` forbids its
    /// output to match its slot.
    pub(super) separate_output: bool,
    /// Each byte range, as offset and size, that `picture_info` names in `bitstream`.
    pub(super) byte_ranges: &'a [(u32, u32)],
    /// The picture's pts, for errors.
    pub(super) pts: i64,
}

/// What a plane copy reads.
#[derive(Clone, Copy, Debug)]
pub(super) struct CopySource {
    pub(super) picture: PictureImage,
    /// The timeline value of the decode.
    pub(super) timeline_value: u64,
    pub(super) timeline: vk::Semaphore,
    /// Whether the copy releases the image back to the video queue.
    pub(super) returns_image: bool,
    /// The image's layout in the decode. A release and its acquire both state it.
    pub(super) decode_layout: vk::ImageLayout,
}

/// Every object a submit or a plane copy can use. The fields drop in this order, and each
/// comes after what uses it.
struct EngineObjects {
    parameters: Option<SessionParameters>,
    session: VideoSession,
    pictures: PictureStore,
    bitstream: HostBuffer,
    status: Option<StatusQuery>,
    pool: CommandPool,
    timeline: Timeline,
}

/// The session and everything one decode needs. Every codec shares it.
pub(super) struct DecodeEngine {
    // `Drop` frees these only once both queues are done with them.
    objects: ManuallyDrop<EngineObjects>,
    /// The largest coded size this session takes.
    max_coded: (u32, u32),
    /// `maxActiveReferencePictures` of the session.
    active_references: u32,
    size_alignment: u64,
    // `new_bitstream` needs it when a picture outgrows the buffer.
    profile: VideoProfile,
    /// Whether the session still needs its reset. Its first decode runs it.
    needs_reset: bool,
    device: Arc<VulkanDevice>,
}

fn new_bitstream(
    device: &Arc<VulkanDevice>,
    profile: &VideoProfile,
    size: u64,
) -> Result<HostBuffer, DecodeError> {
    let infos = [profile.info()];
    let mut list = vk::VideoProfileListInfoKHR::default().profiles(&infos);
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut list);
    HostBuffer::new(device, &info, "the bitstream buffer")
}

/// What a new engine holds.
pub(super) struct EngineSpec<'a> {
    pub(super) caps: &'a DecodeCaps,
    pub(super) mode: DpbMode,
    pub(super) coded: (u32, u32),
    pub(super) slots: u32,
    pub(super) active_references: u32,
    /// Whether a plane copy can read a slot's image: see [`DecodeEngine::read_slot`].
    pub(super) readable_slots: bool,
    /// Whether a request can set [`DecodeRequest::separate_output`] in coincide mode.
    pub(super) output_image: bool,
}

impl DecodeEngine {
    /// Opens a session and its images. The session's reset waits for its first picture.
    pub(super) fn new(
        device: &Arc<VulkanDevice>,
        profile: VideoProfile,
        spec: &EngineSpec<'_>,
    ) -> Result<Self, DecodeError> {
        let caps = spec.caps;
        let pictures = PictureStore::new(
            device,
            &profile,
            spec.mode,
            spec.slots,
            separate_images(caps.flags),
            image_extent(spec.coded, caps.granularity),
            spec.readable_slots,
            spec.output_image,
        )?;
        let session = VideoSession::new(
            device,
            &SessionSpec {
                profile: &profile,
                caps,
                format: image_format(profile.pixel_format()),
                coded: spec.coded,
                slots: spec.slots,
                active_references: spec.active_references,
            },
        )?;
        let bitstream = new_bitstream(
            device,
            &profile,
            align_up(MIN_BITSTREAM_BYTES, caps.size_alignment),
        )?;
        let status = if device.status_queries {
            Some(StatusQuery::new(device, &profile.info())?)
        } else {
            None
        };
        let pool = CommandPool::new(device, device.video_family)?;
        let timeline = Timeline::new(device)?;
        Ok(Self {
            objects: ManuallyDrop::new(EngineObjects {
                parameters: None,
                session,
                pictures,
                bitstream,
                status,
                pool,
                timeline,
            }),
            max_coded: spec.coded,
            active_references: spec.active_references,
            size_alignment: caps.size_alignment,
            profile,
            needs_reset: true,
            device: Arc::clone(device),
        })
    }

    /// The largest coded size this session takes.
    pub(super) const fn max_coded(&self) -> (u32, u32) {
        self.max_coded
    }

    /// The session, for its parameters.
    pub(super) fn session(&self) -> vk::VideoSessionKHR {
        self.objects.session.raw
    }

    /// Replaces the session parameters. The next decode uses them.
    pub(super) fn set_parameters(&mut self, parameters: SessionParameters) {
        self.objects.parameters = Some(parameters);
    }

    /// The command buffer, for recording.
    fn recorder(&self) -> Recorder<'_> {
        Recorder {
            device: &self.device,
            cmd: self.objects.pool.buffer,
        }
    }

    /// What a request checks against, as this session stands.
    fn request_limits(&self) -> RequestLimits {
        let objects = &*self.objects;
        RequestLimits {
            slots: objects.pictures.slots.len(),
            active_references: self.active_references,
            output_image: objects.pictures.output.is_some(),
            parameters: objects.parameters.as_ref().map(|parameters| parameters.raw),
            uses_parameters: self.profile.uses_parameters(),
        }
    }

    /// Ends the command buffer, submits it, and waits for it.
    fn submit_and_wait(&mut self, what: &str) -> Result<u64, DecodeError> {
        let recorder = Recorder {
            device: &self.device,
            cmd: self.objects.pool.buffer,
        };
        let value = {
            let queue = self.device.video_queue();
            recorder.end_and_submit(*queue, &mut self.objects.timeline, what)?
        };
        self.objects.timeline.wait(value, WAIT_NS)?;
        Ok(value)
    }

    /// Writes at offset 0, a multiple of every `minBitstreamBufferOffsetAlignment`. The
    /// range rounds up to `minBitstreamBufferSizeAlignment`.
    fn write_bitstream(&mut self, parts: &[&[u8]]) -> Result<u64, DecodeError> {
        let bytes: u64 = parts.iter().map(|part| part.len() as u64).sum();
        let range = align_up(bytes, self.size_alignment);
        if range > self.objects.bitstream.size {
            // Every earlier submit finished, so the old buffer has no reader.
            self.objects.bitstream =
                new_bitstream(&self.device, &self.profile, range.saturating_mul(2))?;
        }
        self.objects.bitstream.write(parts, range)?;
        Ok(range)
    }

    /// Decodes one picture and waits for it. The result names what a copy reads.
    ///
    /// # Safety
    ///
    /// Each pointer in `request` points to a valid chain for this session's codec. Each
    /// referenced structure lives through the call.
    pub(super) unsafe fn decode(
        &mut self,
        request: &DecodeRequest<'_>,
    ) -> Result<CopySource, DecodeError> {
        check_request(request, &self.request_limits())?;
        let range = self.write_bitstream(request.bitstream)?;
        let what = format!("the picture at pts {}", request.pts);
        let recorder = self.recorder();
        recorder.begin(&what)?;
        if let Some(status) = &self.objects.status {
            // SAFETY: the buffer is recording, and the pool holds one query.
            unsafe {
                self.device
                    .raw
                    .cmd_reset_query_pool(self.objects.pool.buffer, status.raw, 0, 1);
            }
        }
        let mode = request_mode(self.objects.pictures.mode, request.separate_output);
        let objects = &*self.objects;
        recorder.record_acquires(&objects.pictures, request, mode);
        let scope = ScopeObjects {
            session: objects.session.raw,
            parameters: objects.parameters.as_ref().map(|parameters| parameters.raw),
            uses_parameters: self.profile.uses_parameters(),
            pictures: &objects.pictures,
            bitstream: objects.bitstream.raw,
            status: objects.status.as_ref(),
            needs_reset: self.needs_reset,
        };
        // SAFETY: the caller states the codec chains are valid, and the buffer is recording.
        let target = unsafe { recorder.record_decode(&scope, request, range, mode) }?;
        if request.output {
            recorder.record_release(target, decode_layout(mode));
        }

        let timeline_value = self.submit_and_wait(&what)?;
        self.needs_reset = false;
        self.check_status(request.pts)?;

        let pictures = &mut self.objects.pictures;
        for reference in request.references {
            if let Some(entry) = pictures.slots.get_mut(reference.slot as usize) {
                entry.state = SlotState::Ready;
            }
        }
        if let Some(entry) = pictures.slots.get_mut(request.setup.slot as usize) {
            entry.state = state_after_copy(mode, request.reference, request.output);
        }
        Ok(CopySource {
            picture: target,
            timeline_value,
            timeline: self.objects.timeline.raw,
            returns_image: copy_returns_image(mode, request.reference),
            decode_layout: decode_layout(mode),
        })
    }

    /// Prepares a stored slot for a plane copy, for a picture a codec shows again. The slot
    /// stays a reference: the copy hands it back, and the next decode takes it. The engine
    /// must open with `readable_slots`.
    pub(super) fn read_slot(&mut self, slot: u32, pts: i64) -> Result<CopySource, DecodeError> {
        let pictures = &self.objects.pictures;
        let count = pictures.slots.len();
        let state = pictures
            .slots
            .get(slot as usize)
            .map(|entry| entry.state)
            .ok_or_else(|| absent_slot(slot, count, pts))?;
        if state == SlotState::Empty {
            return Err(DecodeError::VulkanDecodeRequest {
                pts,
                reason: format!("DPB slot {slot} holds no picture to show"),
            });
        }
        let picture = pictures
            .slot_image(slot as usize)
            .ok_or_else(|| absent_slot(slot, count, pts))?;
        let what = format!("DPB slot {slot} at pts {pts}");
        let recorder = self.recorder();
        recorder.begin(&what)?;
        let families = recorder.families_to_video(self.device.transfers_ownership());
        let acquire = read_barrier(state, picture, families);
        recorder.pipeline_barrier(acquire.as_slice());
        recorder.record_release(picture, vk::ImageLayout::VIDEO_DECODE_DPB_KHR);
        let timeline_value = self.submit_and_wait(&what)?;
        if let Some(entry) = self.objects.pictures.slots.get_mut(slot as usize) {
            entry.state = SlotState::Returning;
        }
        Ok(CopySource {
            picture,
            timeline_value,
            timeline: self.objects.timeline.raw,
            returns_image: true,
            decode_layout: vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
        })
    }

    /// Marks `slot` free. Its contents go when the next decode takes it.
    pub(super) fn release_slot(&mut self, slot: u32) {
        if let Some(entry) = self.objects.pictures.slots.get_mut(slot as usize) {
            entry.state = SlotState::Empty;
        }
    }

    fn check_status(&self, pts: i64) -> Result<(), DecodeError> {
        let Some(status) = &self.objects.status else {
            return Ok(());
        };
        let mut result = [0_i32; 1];
        // SAFETY: the pool holds one query of status type, and its submit finished.
        unsafe {
            self.device.raw.get_query_pool_results(
                status.raw,
                0,
                &mut result,
                vk::QueryResultFlags::WAIT | vk::QueryResultFlags::WITH_STATUS_KHR,
            )
        }
        .map_err(|error| {
            vk_error(
                "vkGetQueryPoolResults",
                error,
                format!("the picture at pts {pts}"),
            )
        })?;
        match result[0] {
            STATUS_COMPLETE => Ok(()),
            other => Err(DecodeError::VulkanDecodeStatus {
                status: other,
                name: status_name(other),
                pts,
            }),
        }
    }
}

/// Whether the objects can go: the video queue and the copy queue both finished with them.
const fn safe_to_destroy(video_done: bool, copies_done: bool) -> bool {
    video_done && copies_done
}

/// Waits for `timeline`'s last value, then for wgpu's queue, which holds any copy that a timed-out
/// call left behind. Each wait stops after [`WAIT`]. The result states whether an engine can free
/// its objects.
pub(super) fn queues_finished(device: &VulkanDevice, timeline: &Timeline) -> bool {
    let video_done = timeline.wait(timeline.last, WAIT_NS).is_ok();
    // A stalled video queue also stalls each copy that waits on it. So the second wait
    // runs only after the first passes.
    let copies_done = video_done
        && device
            .wait_general_queue(None, "the end of a session")
            .is_ok();
    safe_to_destroy(video_done, copies_done)
}

impl Drop for DecodeEngine {
    /// Waits for both queues. See [`queues_finished`].
    ///
    /// When a wait times out or the device is lost, the objects stay allocated for the
    /// life of the process. A GPU can still use them, and a destroy in flight is undefined
    /// behaviour. The caller already holds the error of the stall, so `drop` reports nothing.
    fn drop(&mut self) {
        if queues_finished(&self.device, &self.objects.timeline) {
            // SAFETY: both queues finished with the objects, `drop` runs once, and nothing
            // reads `objects` after this line.
            unsafe { ManuallyDrop::drop(&mut self.objects) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_status_has_its_name() {
        assert_eq!(status_name(-1), "ERROR");
        assert_eq!(status_name(0), "NOT_READY");
        assert_eq!(status_name(1), "COMPLETE");
        assert_eq!(
            status_name(-1_000_299_000),
            "INSUFFICIENT_BITSTREAM_BUFFER_RANGE"
        );
        assert_eq!(status_name(7), "unknown");
    }

    #[test]
    fn objects_go_only_after_both_queues_finish() {
        assert!(safe_to_destroy(true, true));
        assert!(!safe_to_destroy(false, true));
        assert!(!safe_to_destroy(true, false));
        assert!(!safe_to_destroy(false, false));
    }
}
