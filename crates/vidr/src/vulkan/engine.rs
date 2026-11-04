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
//! A new session takes its reset inside the coding scope of its first picture. A decode
//! scope needs session parameters bound, and that scope has them. So the same call order
//! holds for every codec.

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::sync::Arc;

use ash::vk;

use super::barriers::{discard_barrier, image_barrier, read_barrier};
use super::caps::{DecodeCaps, DpbMode, align_up, image_extent, separate_images};
use super::copy::WAIT;
use super::device::VulkanDevice;
use super::error::vk_error;
use super::objects::{CommandPool, HostBuffer, StatusQuery, Timeline};
use super::pictures::{
    PictureImage, PictureStore, SlotState, copy_returns_image, state_after_copy,
};
use super::session::{SessionParameters, SessionSpec, VideoProfile, VideoSession, image_format};
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
}

/// The parameters a coding scope binds. A decode scope of a codec with session parameters
/// needs them, and the session's reset runs in one. So that codec sets them before its first
/// picture. VP9 uses none, so `uses_parameters` is false and the handle stays null.
fn bound_parameters(
    parameters: Option<vk::VideoSessionParametersKHR>,
    uses_parameters: bool,
    pts: i64,
) -> Result<vk::VideoSessionParametersKHR, DecodeError> {
    if !uses_parameters {
        return Ok(vk::VideoSessionParametersKHR::null());
    }
    parameters.ok_or_else(|| DecodeError::VulkanDecodeRequest {
        pts,
        reason: "the session has no parameters".to_owned(),
    })
}

/// Refuses a request the driver takes as invalid usage, whatever its codec:
///
/// - a slot past the session's `slots`.
/// - more references than `limit`, `VUID-vkCmdDecodeVideoKHR-activeReferencePictureCount-07150`.
/// - a slot named twice, `VUID-VkVideoBeginCodingInfoKHR-pPictureResource-07238`,
///   `VUID-vkCmdDecodeVideoKHR-pDecodeInfo-07264`, and
///   `VUID-vkCmdDecodeVideoKHR-dpbFrameUseCount-07176`.
///
/// The setup slot counts as named, because a coding scope binds it too. No path here decodes a
/// second field, which shares its first field's slot. So a valid stream never names a slot twice.
fn check_slots(request: &DecodeRequest<'_>, slots: usize, limit: u32) -> Result<(), DecodeError> {
    let named = || request.references.iter().chain([&request.setup]);
    for slot_ref in named() {
        if slot_ref.slot as usize >= slots {
            return Err(absent_slot(slot_ref.slot, slots, request.pts));
        }
    }
    if request.references.len() > limit as usize {
        return Err(DecodeError::VulkanTooManyReferences {
            pts: request.pts,
            references: request.references.len(),
            limit,
        });
    }
    for (index, slot_ref) in named().enumerate() {
        if named()
            .take(index)
            .any(|earlier| earlier.slot == slot_ref.slot)
        {
            return Err(DecodeError::VulkanDuplicateSlot {
                pts: request.pts,
                slot: slot_ref.slot,
            });
        }
    }
    Ok(())
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
            spec.readable_slots,
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

    fn resource(
        view: vk::ImageView,
        coded: (u32, u32),
    ) -> vk::VideoPictureResourceInfoKHR<'static> {
        vk::VideoPictureResourceInfoKHR::default()
            .coded_extent(vk::Extent2D {
                width: coded.0,
                height: coded.1,
            })
            .image_view_binding(view)
    }

    /// Refuses a request with slots this session cannot bind, or a session with no parameters.
    /// The refusal comes before any driver call of the picture.
    fn check_request(&self, request: &DecodeRequest<'_>) -> Result<(), DecodeError> {
        check_slots(
            request,
            self.objects.pictures.slots.len(),
            self.active_references,
        )?;
        let parameters = self.objects.parameters.as_ref();
        bound_parameters(
            parameters.map(|parameters| parameters.raw),
            self.profile.uses_parameters(),
            request.pts,
        )
        .map(|_| ())
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
        if request.output {
            self.record_release(target, self.decode_layout());
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
        let mode = pictures.mode;
        if let Some(entry) = pictures.slots.get_mut(request.setup.slot as usize) {
            entry.state = state_after_copy(mode, request.reference, request.output);
        }
        Ok(CopySource {
            picture: target,
            timeline_value,
            timeline: self.objects.timeline.raw,
            returns_image: copy_returns_image(mode, request.reference),
            decode_layout: self.decode_layout(),
        })
    }

    /// The layout the output image holds during a decode.
    fn decode_layout(&self) -> vk::ImageLayout {
        match self.objects.pictures.mode {
            DpbMode::Coincide => vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
            DpbMode::Distinct => vk::ImageLayout::VIDEO_DECODE_DST_KHR,
        }
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
        self.begin_commands(&what)?;
        let families = self.families_to_video(self.device.transfers_ownership());
        let acquire = read_barrier(state, picture, families);
        self.pipeline_barrier(acquire.as_slice());
        self.record_release(picture, vk::ImageLayout::VIDEO_DECODE_DPB_KHR);
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

    /// The view of `slot`.
    fn slot_view(&self, slot: u32, pts: i64) -> Result<vk::ImageView, DecodeError> {
        let slots = &self.objects.pictures.slots;
        slots
            .get(slot as usize)
            .map(|entry| entry.view.raw)
            .ok_or_else(|| absent_slot(slot, slots.len(), pts))
    }

    /// The resource of each reference, in request order.
    fn reference_resources(
        &self,
        request: &DecodeRequest<'_>,
    ) -> Result<Vec<vk::VideoPictureResourceInfoKHR<'static>>, DecodeError> {
        request
            .references
            .iter()
            .map(|reference| {
                self.slot_view(reference.slot, request.pts)
                    .map(|view| Self::resource(view, reference.coded))
            })
            .collect()
    }

    /// The resource the decode writes for output, and the image behind it.
    fn output_target(
        &self,
        request: &DecodeRequest<'_>,
        setup_resource: vk::VideoPictureResourceInfoKHR<'static>,
    ) -> Result<(vk::VideoPictureResourceInfoKHR<'static>, PictureImage), DecodeError> {
        let pictures = &self.objects.pictures;
        match &pictures.output {
            Some((view, image)) => Ok((
                Self::resource(view.raw, request.setup.coded),
                PictureImage {
                    image: image.raw,
                    layer: 0,
                },
            )),
            None => {
                let setup = request.setup.slot;
                let target = pictures
                    .slot_image(setup as usize)
                    .ok_or_else(|| absent_slot(setup, pictures.slots.len(), request.pts))?;
                Ok((setup_resource, target))
            }
        }
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
        let setup_view = self.slot_view(request.setup.slot, pts)?;
        let setup_resource = Self::resource(setup_view, request.setup.coded);
        let reference_resources = self.reference_resources(request)?;
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
            .video_session_parameters(bound_parameters(
                parameters,
                self.profile.uses_parameters(),
                pts,
            )?)
            .reference_slots(&bound);

        // `bound` ends in the new picture's slot, and the zip stops before it.
        let decode_references: Vec<_> = bound
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

        let (dst_resource, target) = self.output_target(request, setup_resource)?;
        let mut info = vk::VideoDecodeInfoKHR::default()
            .src_buffer(objects.bitstream.raw)
            .src_buffer_offset(0)
            .src_buffer_range(range)
            .dst_picture_resource(dst_resource)
            .setup_reference_slot(&setup_slot)
            .reference_slots(&decode_references);
        info.p_next = request.picture_info;

        // SAFETY: the caller states the codec chains are valid, and the buffer is recording.
        // `begin`, `info`, and all they point to live in this function.
        unsafe { self.record_scope(&begin, &info) };
        Ok(target)
    }

    /// Records the commands of one coding scope, in their one order.
    ///
    /// # Safety
    ///
    /// The buffer is recording on the video family. `begin` and `info` hold valid chains.
    unsafe fn record_scope(
        &self,
        begin: &vk::VideoBeginCodingInfoKHR<'_>,
        info: &vk::VideoDecodeInfoKHR<'_>,
    ) {
        let objects = &*self.objects;
        let device = &self.device;
        let cmd = objects.pool.buffer;
        let fns = device.video_fns.fp();
        let control =
            vk::VideoCodingControlInfoKHR::default().flags(vk::VideoCodingControlFlagsKHR::RESET);
        let end = vk::VideoEndCodingInfoKHR::default();
        // SAFETY: the buffer is recording on the video family. `record_acquires` set each
        // resource layout, and the caller states the codec chains are valid. The first
        // picture binds no active slot, so its reset invalidates nothing it reads.
        unsafe {
            (fns.cmd_begin_video_coding_khr)(cmd, begin);
            if self.needs_reset {
                (fns.cmd_control_video_coding_khr)(cmd, &control);
            }
            if let Some(status) = &objects.status {
                device
                    .raw
                    .cmd_begin_query(cmd, status.raw, 0, vk::QueryControlFlags::empty());
            }
            (device.decode_fns.fp().cmd_decode_video_khr)(cmd, info);
            if let Some(status) = &objects.status {
                device.raw.cmd_end_query(cmd, status.raw, 0);
            }
            (fns.cmd_end_video_coding_khr)(cmd, &end);
        }
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
            let barrier = pictures
                .slots
                .get(reference.slot as usize)
                .zip(pictures.slot_image(reference.slot as usize))
                .and_then(|(entry, picture)| read_barrier(entry.state, picture, families));
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
    fn record_release(&self, target: PictureImage, old: vk::ImageLayout) {
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
        // A stalled video queue also stalls each copy that waits on it. So the second wait
        // runs only after the first passes.
        let copies_done = video_done
            && self
                .device
                .wait_general_queue(None, "the end of a session")
                .is_ok();
        if safe_to_destroy(video_done, copies_done) {
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

    fn slot_ref(slot: u32) -> SlotRef {
        SlotRef {
            slot,
            codec_info: std::ptr::null(),
            coded: (64, 64),
        }
    }

    fn checked(setup: u32, references: &[u32], limit: u32) -> Result<(), DecodeError> {
        let references: Vec<_> = references.iter().copied().map(slot_ref).collect();
        let request = DecodeRequest {
            bitstream: &[],
            picture_info: std::ptr::null(),
            setup: slot_ref(setup),
            references: &references,
            reference: true,
            output: true,
            pts: 9,
        };
        check_slots(&request, 6, limit)
    }

    #[test]
    fn distinct_slots_up_to_the_limit_pass() {
        assert_eq!(checked(0, &[], 1), Ok(()));
        assert_eq!(checked(5, &[1, 2, 3], 3), Ok(()));
    }

    #[test]
    fn more_references_than_the_session_holds_are_refused_by_name() {
        let error = checked(5, &[1, 2, 3], 2).expect_err("refused");
        assert_eq!(
            error,
            DecodeError::VulkanTooManyReferences {
                pts: 9,
                references: 3,
                limit: 2,
            }
        );
        let text = error.to_string();
        assert!(text.contains("pts 9"), "{text}");
        assert!(text.contains("binds 3"), "{text}");
        assert!(text.contains("holds 2"), "{text}");
        assert!(text.contains("re-encode"), "{text}");
    }

    #[test]
    fn a_slot_named_twice_is_refused_by_name() {
        for (setup, references, slot) in [(5, &[1, 2, 1], 1), (2, &[1, 2, 3], 2)] {
            let error = checked(setup, references, 3).expect_err("refused");
            assert_eq!(error, DecodeError::VulkanDuplicateSlot { pts: 9, slot });
            let text = error.to_string();
            assert!(text.contains(&format!("DPB slot {slot} twice")), "{text}");
        }
    }

    #[test]
    fn a_slot_past_the_session_comes_first() {
        let error = checked(5, &[7, 7, 7], 1).expect_err("refused");
        assert!(matches!(error, DecodeError::VulkanDecodeRequest { .. }));
    }

    #[test]
    fn a_decode_needs_session_parameters() {
        let raw = vk::VideoSessionParametersKHR::null();
        assert_eq!(bound_parameters(Some(raw), true, 4), Ok(raw));
        let error = bound_parameters(None, true, 4).expect_err("refused");
        assert!(matches!(
            error,
            DecodeError::VulkanDecodeRequest { pts: 4, .. }
        ));
    }

    #[test]
    fn a_codec_with_no_parameters_binds_a_null_handle() {
        assert_eq!(
            bound_parameters(None, false, 4),
            Ok(vk::VideoSessionParametersKHR::null())
        );
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
