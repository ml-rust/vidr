//! One decode per call: the bitstream write, the barriers, the coding scope, the submit,
//! the wait, and the status. Codecs supply `pNext` chains and nothing else.
//!
//! A slot's image takes these steps in coincide mode, when the video and general queue
//! families differ:
//!
//! 1. The video queue decodes into it, then releases it to the general family.
//! 2. The general queue acquires it and copies each plane to a numr buffer.
//! 3. For a reference, the general queue releases it, and the next decode acquires it.
//!
//! When the families match, step 1 makes the decode's write visible to later reads on
//! the queue, and no transfer runs. In distinct mode only the output image takes steps 1
//! and 2. A slot stays in the DPB layout on the video queue, so the next decode that reads
//! it takes a barrier against the decode that wrote it. A use of an empty image starts
//! from `UNDEFINED`, which discards its contents and needs no transfer.
//!
//! A new session takes its reset inside the coding scope of its first picture. A decode
//! scope needs session parameters bound, and that scope has them. So the same call order
//! holds for every codec.

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::sync::Arc;

use ash::vk;

use super::caps::{DecodeCaps, DpbMode, align_up, image_extent, separate_images};
use super::copy::WAIT;
use super::device::VulkanDevice;
use super::error::vk_error;
use super::objects::{CommandPool, HostBuffer, StatusQuery, Timeline};
use super::pictures::{
    PictureImage, PictureStore, SlotState, copy_returns_image, state_after_copy,
};
use super::session::{SessionParameters, SessionSpec, VideoProfile, VideoSession};
use crate::error::DecodeError;

/// How long a host wait lasts before it reports an error.
const WAIT_NS: u64 = WAIT.as_nanos() as u64;
/// The smallest bitstream buffer.
const MIN_BITSTREAM_BYTES: u64 = 1 << 20;
/// `VK_QUERY_RESULT_STATUS_COMPLETE_KHR`.
const STATUS_COMPLETE: i32 = 1;

/// The name of a `VkQueryResultStatusKHR` value.
const fn status_name(status: i32) -> &'static str {
    match status {
        -1 => "ERROR",
        0 => "NOT_READY",
        1 => "COMPLETE",
        -1_000_299_000 => "INSUFFICIENT_BITSTREAM_BUFFER_RANGE",
        _ => "unknown",
    }
}

/// One DPB slot in a decode, with its codec structure.
#[derive(Clone, Copy, Debug)]
pub(super) struct SlotRef {
    pub(super) slot: u32,
    pub(super) codec_info: *const c_void,
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

/// The session and everything one decode needs. Later codecs reuse it whole.
pub(super) struct DecodeEngine {
    // `Drop` frees these only once both queues are done with them.
    objects: ManuallyDrop<EngineObjects>,
    coded: vk::Extent2D,
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
}

/// The parameters a coding scope binds. Each decode scope needs them, and the session's
/// reset runs in one, so a codec sets them before its first picture.
fn bound_parameters(
    parameters: Option<vk::VideoSessionParametersKHR>,
    pts: i64,
) -> Result<vk::VideoSessionParametersKHR, DecodeError> {
    parameters.ok_or_else(|| DecodeError::VulkanDecodeRequest {
        pts,
        reason: "the session has no parameters".to_owned(),
    })
}

/// The error of a slot index past the session's slots.
fn absent_slot(slot: u32, slots: usize, pts: i64) -> DecodeError {
    DecodeError::VulkanDecodeRequest {
        pts,
        reason: format!("DPB slot {slot} does not exist in {slots} slots"),
    }
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
        )?;
        let session = VideoSession::new(
            device,
            &SessionSpec {
                profile: &profile,
                caps,
                format: vk::Format::G8_B8R8_2PLANE_420_UNORM,
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
            coded: vk::Extent2D {
                width: spec.coded.0,
                height: spec.coded.1,
            },
            size_alignment: caps.size_alignment,
            profile,
            needs_reset: true,
            device: Arc::clone(device),
        })
    }

    /// The session, for its parameters.
    pub(super) fn session(&self) -> vk::VideoSessionKHR {
        self.objects.session.raw
    }

    /// Replaces the session parameters. The next decode uses them.
    pub(super) fn set_parameters(&mut self, parameters: SessionParameters) {
        self.objects.parameters = Some(parameters);
    }

    fn begin_commands(&self, what: &str) -> Result<(), DecodeError> {
        let info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the buffer belongs to this pool, and every earlier submit finished,
        // because each one waits on the timeline.
        unsafe {
            self.device.raw.reset_command_buffer(
                self.objects.pool.buffer,
                vk::CommandBufferResetFlags::empty(),
            )
        }
        .map_err(|result| vk_error("vkResetCommandBuffer", result, what))?;
        // SAFETY: the reset above leaves the buffer ready to record.
        unsafe {
            self.device
                .raw
                .begin_command_buffer(self.objects.pool.buffer, &info)
        }
        .map_err(|result| vk_error("vkBeginCommandBuffer", result, what))
    }

    /// Ends the command buffer, submits it, and waits for it.
    fn submit_and_wait(&mut self, what: &str) -> Result<u64, DecodeError> {
        let cmd = self.objects.pool.buffer;
        // SAFETY: the buffer is recording, and each scope in it is closed.
        unsafe { self.device.raw.end_command_buffer(cmd) }
            .map_err(|result| vk_error("vkEndCommandBuffer", result, what))?;
        let value = self.objects.timeline.last + 1;
        let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
        let signals = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.objects.timeline.raw)
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&commands)
            .signal_semaphore_infos(&signals);
        {
            let queue = self.device.video_queue();
            // SAFETY: the guard holds the queue's external synchronisation. The buffer
            // was recorded for this family, and each array lives through the call.
            unsafe {
                self.device
                    .sync2
                    .queue_submit2(*queue, &[submit], vk::Fence::null())
            }
            .map_err(|result| vk_error("vkQueueSubmit2", result, what))?;
        }
        self.objects.timeline.last = value;
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

    fn resource(&self, view: vk::ImageView) -> vk::VideoPictureResourceInfoKHR<'static> {
        vk::VideoPictureResourceInfoKHR::default()
            .coded_extent(self.coded)
            .image_view_binding(view)
    }

    /// Refuses a request that names no slot of this session, or a session with no
    /// parameters.
    fn check_request(&self, request: &DecodeRequest<'_>) -> Result<(), DecodeError> {
        let slots = self.objects.pictures.slots.len();
        for slot_ref in request.references.iter().chain([&request.setup]) {
            if slot_ref.slot as usize >= slots {
                return Err(absent_slot(slot_ref.slot, slots, request.pts));
            }
        }
        let parameters = self.objects.parameters.as_ref();
        bound_parameters(parameters.map(|parameters| parameters.raw), request.pts).map(|_| ())
    }

    /// Decodes one picture and waits for it. The result names what a copy reads.
    ///
    /// # Safety
    ///
    /// Each pointer in `request` points to a valid chain for this session's codec, and
    /// each referenced structure lives through the call.
    pub(super) unsafe fn decode(
        &mut self,
        request: &DecodeRequest<'_>,
    ) -> Result<CopySource, DecodeError> {
        self.check_request(request)?;
        let range = self.write_bitstream(request.bitstream)?;
        let what = format!("the picture at pts {}", request.pts);
        self.begin_commands(&what)?;
        if let Some(status) = &self.objects.status {
            // SAFETY: the buffer is recording, and the pool holds one query.
            unsafe {
                self.device
                    .raw
                    .cmd_reset_query_pool(self.objects.pool.buffer, status.raw, 0, 1);
            }
        }
        self.record_acquires(request);
        // SAFETY: the caller states the codec chains are valid, and the buffer is recording.
        let target = unsafe { self.record_decode(request, range) }?;
        self.record_release(target);

        let timeline_value = self.submit_and_wait(&what)?;
        self.needs_reset = false;
        self.check_status(request.pts)?;

        let mode = self.objects.pictures.mode;
        for reference in request.references {
            self.objects.pictures.slots[reference.slot as usize].state = SlotState::Ready;
        }
        self.objects.pictures.slots[request.setup.slot as usize].state =
            state_after_copy(mode, request.reference);
        Ok(CopySource {
            picture: target,
            timeline_value,
            timeline: self.objects.timeline.raw,
            returns_image: copy_returns_image(mode, request.reference),
            decode_layout: match mode {
                DpbMode::Coincide => vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
                DpbMode::Distinct => vk::ImageLayout::VIDEO_DECODE_DST_KHR,
            },
        })
    }

    /// Records the coding scope: the first picture's reset, the query, and the decode.
    /// Every structure the commands read lives in this function. The result is the image
    /// the decode writes for output.
    ///
    /// # Safety
    ///
    /// As [`Self::decode`], and the buffer is recording on the video family.
    unsafe fn record_decode(
        &self,
        request: &DecodeRequest<'_>,
        range: u64,
    ) -> Result<PictureImage, DecodeError> {
        let objects = &*self.objects;
        let pts = request.pts;
        let slots = objects.pictures.slots.len();
        let slot_view = |slot: u32| {
            objects
                .pictures
                .slots
                .get(slot as usize)
                .map(|entry| entry.view.raw)
                .ok_or_else(|| absent_slot(slot, slots, pts))
        };
        let setup_resource = self.resource(slot_view(request.setup.slot)?);
        let mut reference_resources = Vec::with_capacity(request.references.len());
        for reference in request.references {
            reference_resources.push(self.resource(slot_view(reference.slot)?));
        }
        let bound: Vec<_> = request
            .references
            .iter()
            .zip(&reference_resources)
            .map(|(reference, resource)| {
                vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(reference.slot as i32)
                    .picture_resource(resource)
            })
            // The slot of the new picture goes in unassigned. The decode assigns it.
            .chain([vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(-1)
                .picture_resource(&setup_resource)])
            .collect();
        let parameters = objects.parameters.as_ref().map(|parameters| parameters.raw);
        let begin = vk::VideoBeginCodingInfoKHR::default()
            .video_session(objects.session.raw)
            .video_session_parameters(bound_parameters(parameters, pts)?)
            .reference_slots(&bound);
        let control =
            vk::VideoCodingControlInfoKHR::default().flags(vk::VideoCodingControlFlagsKHR::RESET);

        let decode_references: Vec<_> = bound[..request.references.len()]
            .iter()
            .zip(request.references)
            .map(|(slot, reference)| {
                let mut slot = *slot;
                slot.p_next = reference.codec_info;
                slot
            })
            .collect();
        let mut setup_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(request.setup.slot as i32)
            .picture_resource(&setup_resource);
        setup_slot.p_next = request.setup.codec_info;

        let (dst_resource, target) = match &objects.pictures.output {
            Some((view, image)) => (
                self.resource(view.raw),
                PictureImage {
                    image: image.raw,
                    layer: 0,
                },
            ),
            None => {
                let setup = request.setup.slot;
                let target = objects
                    .pictures
                    .slot_image(setup as usize)
                    .ok_or_else(|| absent_slot(setup, slots, pts))?;
                (setup_resource, target)
            }
        };
        let mut info = vk::VideoDecodeInfoKHR::default()
            .src_buffer(objects.bitstream.raw)
            .src_buffer_offset(0)
            .src_buffer_range(range)
            .dst_picture_resource(dst_resource)
            .setup_reference_slot(&setup_slot)
            .reference_slots(&decode_references);
        info.p_next = request.picture_info;

        let device = &self.device;
        let cmd = objects.pool.buffer;
        let fns = device.video_fns.fp();
        let end = vk::VideoEndCodingInfoKHR::default();
        // SAFETY: the buffer is recording on the video family, each resource layout was
        // set by `record_acquires`, and the caller states the codec chains are valid. The
        // first picture binds no active slot, so its reset invalidates nothing it reads.
        unsafe {
            (fns.cmd_begin_video_coding_khr)(cmd, &begin);
            if self.needs_reset {
                (fns.cmd_control_video_coding_khr)(cmd, &control);
            }
            if let Some(status) = &objects.status {
                device
                    .raw
                    .cmd_begin_query(cmd, status.raw, 0, vk::QueryControlFlags::empty());
            }
            (device.decode_fns.fp().cmd_decode_video_khr)(cmd, &info);
            if let Some(status) = &objects.status {
                device.raw.cmd_end_query(cmd, status.raw, 0);
            }
            (fns.cmd_end_video_coding_khr)(cmd, &end);
        }
        Ok(target)
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

    /// Puts each image in the layout its decode needs, with each write visible.
    fn record_acquires(&self, request: &DecodeRequest<'_>) {
        let pictures = &self.objects.pictures;
        let families = self.families_to_video(self.device.transfers_ownership());
        let mut barriers = Vec::with_capacity(request.references.len() + 2);
        for reference in request.references {
            let state = pictures.slots[reference.slot as usize].state;
            let barrier = pictures
                .slot_image(reference.slot as usize)
                .and_then(|picture| read_barrier(state, picture, families));
            barriers.extend(barrier);
        }
        if let Some(picture) = pictures.slot_image(request.setup.slot as usize) {
            barriers.push(discard_barrier(
                picture,
                vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
                vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR | vk::AccessFlags2::VIDEO_DECODE_READ_KHR,
            ));
        }
        if let Some((_, image)) = &pictures.output {
            barriers.push(discard_barrier(
                PictureImage {
                    image: image.raw,
                    layer: 0,
                },
                vk::ImageLayout::VIDEO_DECODE_DST_KHR,
                vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
            ));
        }
        self.pipeline_barrier(&barriers);
    }

    /// Moves the decoded image to the copy layout, and to the general family.
    fn record_release(&self, target: PictureImage) {
        let old = match self.objects.pictures.mode {
            DpbMode::Coincide => vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
            DpbMode::Distinct => vk::ImageLayout::VIDEO_DECODE_DST_KHR,
        };
        let transfers = self.device.transfers_ownership();
        let (dst_stage, dst_access) = if transfers {
            (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE)
        } else {
            (
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ,
            )
        };
        let barrier = image_barrier(
            target,
            (
                vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
                vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
                old,
            ),
            (dst_stage, dst_access, vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
            self.families_to_general(transfers),
        );
        self.pipeline_barrier(&[barrier]);
    }

    fn families_to_video(&self, transfers: bool) -> (u32, u32) {
        if transfers {
            (self.device.general_family, self.device.video_family)
        } else {
            (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
        }
    }

    fn families_to_general(&self, transfers: bool) -> (u32, u32) {
        if transfers {
            (self.device.video_family, self.device.general_family)
        } else {
            (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
        }
    }

    fn pipeline_barrier(&self, barriers: &[vk::ImageMemoryBarrier2<'_>]) {
        if barriers.is_empty() {
            return;
        }
        let info = vk::DependencyInfo::default().image_memory_barriers(barriers);
        // SAFETY: the buffer is recording, and each barrier names a live image.
        unsafe {
            self.device
                .sync2
                .cmd_pipeline_barrier2(self.objects.pool.buffer, &info);
        }
    }
}

/// Whether the objects can go: the video queue and the copy queue both finished with them.
const fn safe_to_destroy(video_done: bool, copies_done: bool) -> bool {
    video_done && copies_done
}

impl Drop for DecodeEngine {
    /// Waits for this engine's last decode, then for wgpu's queue, which holds any copy
    /// that a timed-out `copy_planes` left behind. Each wait stops after [`WAIT`].
    ///
    /// When a wait times out or the device is lost, the objects stay allocated for the
    /// life of the process. A GPU can still use them, and a destroy in flight is undefined
    /// behaviour. The caller already holds the error of the stall, so `drop` reports nothing.
    fn drop(&mut self) {
        let timeline = &self.objects.timeline;
        let video_done = timeline.wait(timeline.last, WAIT_NS).is_ok();
        // A stalled video queue also stalls each copy that waits on it, so the second wait
        // runs only after the first passes.
        let copies_done = video_done
            && self
                .device
                .wgpu_device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(WAIT),
                })
                .is_ok();
        if safe_to_destroy(video_done, copies_done) {
            // SAFETY: both queues finished with the objects, `drop` runs once, and nothing
            // reads `objects` after this line.
            unsafe { ManuallyDrop::drop(&mut self.objects) };
        }
    }
}

/// The barrier a decode needs before it reads a slot in `state`, or `None`. `families`
/// is `(general, video)` when the queue families differ, and both ignored otherwise.
///
/// - `Written`: the decode that wrote the slot, against this read.
/// - `Returning`: the slot comes back from the copy layout. With a transfer, this is the
///   acquire of the copy queue's release, and an acquire has no source scope. With no
///   transfer, the source scope chains to the barrier that ended the writing decode.
fn read_barrier(
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
        .subresource_range(super::objects::color_range(picture.layer))
}

/// A barrier that discards an image's contents and takes it for a decode. The source scope
/// holds an earlier decode's write, so that write cannot land on the new picture.
fn discard_barrier(
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

    #[test]
    fn a_decode_needs_session_parameters() {
        let raw = vk::VideoSessionParametersKHR::null();
        assert_eq!(bound_parameters(Some(raw), 4), Ok(raw));
        let error = bound_parameters(None, 4).expect_err("refused");
        assert!(matches!(
            error,
            DecodeError::VulkanDecodeRequest { pts: 4, .. }
        ));
    }

    #[test]
    fn an_absent_slot_names_itself_with_no_codec() {
        let text = absent_slot(7, 5, 9).to_string();
        assert!(text.contains("DPB slot 7"), "{text}");
        assert!(text.contains("pts 9"), "{text}");
        assert!(!text.contains("H.264"), "{text}");
    }

    #[test]
    fn objects_go_only_after_both_queues_finish() {
        assert!(safe_to_destroy(true, true));
        assert!(!safe_to_destroy(false, true));
        assert!(!safe_to_destroy(true, false));
        assert!(!safe_to_destroy(false, false));
    }
}
