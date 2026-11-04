//! The commands of one decode, in their one order: barriers, the coding scope, and the release.
//! Each function borrows the objects it records against. Encode shares the begin and submit of
//! `Recorder`, and `resource`.

use ash::vk;

use super::barriers::{discard_barrier, image_barrier, read_barrier};
use super::caps::DpbMode;
use super::device::VulkanDevice;
use super::engine::DecodeRequest;
use super::error::vk_error;
use super::objects::{StatusQuery, Timeline};
use super::pictures::{PictureImage, PictureStore};
use super::request_checks::{absent_slot, bound_parameters};
use crate::error::DecodeError;

/// The DPB mode of one request: a request with its own output acts as distinct mode.
pub(super) const fn request_mode(mode: DpbMode, separate_output: bool) -> DpbMode {
    if separate_output {
        DpbMode::Distinct
    } else {
        mode
    }
}

/// The layout the output image holds during a decode.
pub(super) const fn decode_layout(mode: DpbMode) -> vk::ImageLayout {
    match mode {
        DpbMode::Coincide => vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
        DpbMode::Distinct => vk::ImageLayout::VIDEO_DECODE_DST_KHR,
    }
}

pub(super) fn resource(
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

/// The view of `slot`.
fn slot_view(pictures: &PictureStore, slot: u32, pts: i64) -> Result<vk::ImageView, DecodeError> {
    let slots = &pictures.slots;
    slots
        .get(slot as usize)
        .map(|entry| entry.view.raw)
        .ok_or_else(|| absent_slot(slot, slots.len(), pts))
}

/// The resource of each reference, in request order.
fn reference_resources(
    pictures: &PictureStore,
    request: &DecodeRequest<'_>,
) -> Result<Vec<vk::VideoPictureResourceInfoKHR<'static>>, DecodeError> {
    request
        .references
        .iter()
        .map(|reference| {
            slot_view(pictures, reference.slot, request.pts)
                .map(|view| resource(view, reference.coded))
        })
        .collect()
}

/// The resource the decode writes for output, and the image behind it.
fn output_target(
    pictures: &PictureStore,
    request: &DecodeRequest<'_>,
    setup_resource: vk::VideoPictureResourceInfoKHR<'static>,
    mode: DpbMode,
) -> Result<(vk::VideoPictureResourceInfoKHR<'static>, PictureImage), DecodeError> {
    match (mode, &pictures.output) {
        (DpbMode::Distinct, Some((view, image))) => Ok((
            resource(view.raw, request.setup.coded),
            PictureImage {
                image: image.raw,
                layer: 0,
            },
        )),
        _ => {
            let setup = request.setup.slot;
            let target = pictures
                .slot_image(setup as usize)
                .ok_or_else(|| absent_slot(setup, pictures.slots.len(), request.pts))?;
            Ok((setup_resource, target))
        }
    }
}

/// What one coding scope binds and reads.
pub(super) struct ScopeObjects<'a> {
    pub(super) session: vk::VideoSessionKHR,
    pub(super) parameters: Option<vk::VideoSessionParametersKHR>,
    pub(super) uses_parameters: bool,
    pub(super) pictures: &'a PictureStore,
    pub(super) bitstream: vk::Buffer,
    pub(super) status: Option<&'a StatusQuery>,
    /// Whether the session still needs its reset.
    pub(super) needs_reset: bool,
}

/// One command buffer in recording, on one device.
#[derive(Clone, Copy)]
pub(super) struct Recorder<'a> {
    pub(super) device: &'a VulkanDevice,
    pub(super) cmd: vk::CommandBuffer,
}

impl Recorder<'_> {
    /// Resets the buffer and begins it.
    pub(super) fn begin(self, what: &str) -> Result<(), DecodeError> {
        let info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the buffer belongs to this pool, and every earlier submit finished,
        // because each one waits on the timeline.
        unsafe {
            self.device
                .raw
                .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
        }
        .map_err(|result| vk_error("vkResetCommandBuffer", result, what))?;
        // SAFETY: the reset above leaves the buffer ready to record.
        unsafe { self.device.raw.begin_command_buffer(self.cmd, &info) }
            .map_err(|result| vk_error("vkBeginCommandBuffer", result, what))
    }

    /// Ends the buffer and submits it on `queue`, with a signal of the next value of `timeline`.
    /// The caller holds the queue's guard through this call.
    pub(super) fn end_and_submit(
        self,
        queue: vk::Queue,
        timeline: &mut Timeline,
        what: &str,
    ) -> Result<u64, DecodeError> {
        // SAFETY: the buffer is recording, and each scope in it is closed.
        unsafe { self.device.raw.end_command_buffer(self.cmd) }
            .map_err(|result| vk_error("vkEndCommandBuffer", result, what))?;
        let value = timeline.last + 1;
        let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(self.cmd)];
        let signals = [vk::SemaphoreSubmitInfo::default()
            .semaphore(timeline.raw)
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&commands)
            .signal_semaphore_infos(&signals);
        // SAFETY: the caller's guard holds the queue's external synchronisation. The buffer was
        // recorded for this queue's family, and each array lives through the call.
        unsafe {
            self.device
                .sync2
                .queue_submit2(queue, &[submit], vk::Fence::null())
        }
        .map_err(|result| vk_error("vkQueueSubmit2", result, what))?;
        timeline.last = value;
        Ok(value)
    }

    /// Records the coding scope: the first picture's reset, the query, and the decode.
    /// Every structure the commands read lives in this function. The result is the image
    /// the decode writes for output.
    ///
    /// # Safety
    ///
    /// Each pointer in `request` points to a valid chain for this session's codec. Each referenced
    /// structure lives through the call. The buffer is recording on the video family.
    pub(super) unsafe fn record_decode(
        self,
        objects: &ScopeObjects<'_>,
        request: &DecodeRequest<'_>,
        range: u64,
        mode: DpbMode,
    ) -> Result<PictureImage, DecodeError> {
        let pts = request.pts;
        let setup_view = slot_view(objects.pictures, request.setup.slot, pts)?;
        let setup_resource = resource(setup_view, request.setup.coded);
        let reference_resources = reference_resources(objects.pictures, request)?;
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
        let begin = vk::VideoBeginCodingInfoKHR::default()
            .video_session(objects.session)
            .video_session_parameters(bound_parameters(
                objects.parameters,
                objects.uses_parameters,
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

        let (dst_resource, target) =
            output_target(objects.pictures, request, setup_resource, mode)?;
        let mut info = vk::VideoDecodeInfoKHR::default()
            .src_buffer(objects.bitstream)
            .src_buffer_offset(0)
            .src_buffer_range(range)
            .dst_picture_resource(dst_resource)
            .setup_reference_slot(&setup_slot)
            .reference_slots(&decode_references);
        info.p_next = request.picture_info;

        // SAFETY: the caller states the codec chains are valid, and the buffer is recording.
        // `begin`, `info`, and all they point to live in this function.
        unsafe { self.record_scope(objects, &begin, &info) };
        Ok(target)
    }

    /// Records the commands of one coding scope, in their one order.
    ///
    /// # Safety
    ///
    /// The buffer is recording on the video family. `begin` and `info` hold valid chains.
    unsafe fn record_scope(
        self,
        objects: &ScopeObjects<'_>,
        begin: &vk::VideoBeginCodingInfoKHR<'_>,
        info: &vk::VideoDecodeInfoKHR<'_>,
    ) {
        let device = self.device;
        let cmd = self.cmd;
        let fns = device.video_fns.fp();
        let control =
            vk::VideoCodingControlInfoKHR::default().flags(vk::VideoCodingControlFlagsKHR::RESET);
        let end = vk::VideoEndCodingInfoKHR::default();
        // SAFETY: the buffer is recording on the video family. `record_acquires` set each
        // resource layout, and the caller states the codec chains are valid. The first
        // picture binds no active slot, so its reset invalidates nothing it reads.
        unsafe {
            (fns.cmd_begin_video_coding_khr)(cmd, begin);
            if objects.needs_reset {
                (fns.cmd_control_video_coding_khr)(cmd, &control);
            }
            if let Some(status) = objects.status {
                device
                    .raw
                    .cmd_begin_query(cmd, status.raw, 0, vk::QueryControlFlags::empty());
            }
            (device.decode_fns.fp().cmd_decode_video_khr)(cmd, info);
            if let Some(status) = objects.status {
                device.raw.cmd_end_query(cmd, status.raw, 0);
            }
            (fns.cmd_end_video_coding_khr)(cmd, &end);
        }
    }

    /// Puts each image in the layout its decode needs, with each write visible.
    pub(super) fn record_acquires(
        self,
        pictures: &PictureStore,
        request: &DecodeRequest<'_>,
        mode: DpbMode,
    ) {
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
        if let (DpbMode::Distinct, Some((_, image))) = (mode, &pictures.output) {
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
    pub(super) fn record_release(self, target: PictureImage, old: vk::ImageLayout) {
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

    pub(super) fn families_to_video(self, transfers: bool) -> (u32, u32) {
        if transfers {
            (self.device.general_family, self.device.video_family)
        } else {
            (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
        }
    }

    fn families_to_general(self, transfers: bool) -> (u32, u32) {
        if transfers {
            (self.device.video_family, self.device.general_family)
        } else {
            (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
        }
    }

    pub(super) fn pipeline_barrier(self, barriers: &[vk::ImageMemoryBarrier2<'_>]) {
        if barriers.is_empty() {
            return;
        }
        let info = vk::DependencyInfo::default().image_memory_barriers(barriers);
        // SAFETY: the buffer is recording, and each barrier names a live image.
        unsafe {
            self.device.sync2.cmd_pipeline_barrier2(self.cmd, &info);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_with_its_own_output_acts_as_distinct_mode() {
        assert_eq!(request_mode(DpbMode::Coincide, false), DpbMode::Coincide);
        assert_eq!(request_mode(DpbMode::Coincide, true), DpbMode::Distinct);
        assert_eq!(request_mode(DpbMode::Distinct, false), DpbMode::Distinct);
        assert_eq!(
            decode_layout(DpbMode::Distinct),
            vk::ImageLayout::VIDEO_DECODE_DST_KHR
        );
        assert_eq!(
            decode_layout(DpbMode::Coincide),
            vk::ImageLayout::VIDEO_DECODE_DPB_KHR
        );
    }
}
