//! [`VulkanDecoder`]: Vulkan Video behind the backend-neutral [`Decoder`] trait.

use std::sync::Arc;

use ash::vk;
use numr::runtime::wgpu::WgpuRuntime;

use super::av1::Av1Decoder;
use super::caps::DpbPreference;
use super::device::VulkanDevice;
use super::gap;
use super::h264::H264Decoder;
use super::hevc::HevcDecoder;
use super::vp9::Vp9Decoder;
use crate::decode::{DecodeCodec, DecodedFrame, Decoder};
use crate::error::DecodeError;
use crate::route::FrameRoute;

/// One codec on the Vulkan core.
enum Backend {
    H264(H264Decoder),
    Hevc(HevcDecoder),
    Vp9(Vp9Decoder),
    Av1(Box<Av1Decoder>),
}

/// Refuses a codec the device opened without, by name.
fn check_device_codec(device: &VulkanDevice, codec: DecodeCodec) -> Result<(), DecodeError> {
    let op = match codec {
        DecodeCodec::Hevc => vk::VideoCodecOperationFlagsKHR::DECODE_H265,
        DecodeCodec::Vp9 => gap::DECODE_VP9,
        DecodeCodec::Av1 => vk::VideoCodecOperationFlagsKHR::DECODE_AV1,
        _ => return Ok(()),
    };
    if device.decode_ops.contains(op) {
        return Ok(());
    }
    Err(DecodeError::VulkanRefused {
        codec,
        what: "the codec".to_owned(),
        reason: format!(
            "{} lists no decode queue, extension, or feature for it. Use another decoder",
            device.name()
        ),
    })
}

/// Hardware decode on Vulkan Video, into numr WebGPU tensors.
///
/// Frames hold [`PlaneForm::Words`](crate::decode::PlaneForm::Words) NV12 or P010 planes on
/// [`VulkanDevice::numr_device`]. Each frame takes one device-to-device copy from a
/// video image to numr buffers, and no host copy.
///
/// # Input
///
/// One packet holds one access unit, or one VP9 chunk.
///
/// - H.264: `config` is the track's `avcC` payload. With it, `send` takes container samples.
///   With an empty `config`, `send` takes Annex-B access units.
/// - HEVC: the same, with `hvcC`.
/// - VP9: `config` goes unread, and `send` takes a frame or a superframe.
/// - AV1: `config` is the track's `av1C` payload, or empty. `send` takes low-overhead OBUs: a
///   container sample or a temporal unit, as `NvdecDecoder` takes them. Annex B framing is refused
///   by name.
///
/// # Refused
///
/// A refusal names what it refuses, and what the driver's capabilities leave out.
///
/// - H.264: 10 bits, other chroma formats, and field and MBAFF coding. Profiles other than
///   Baseline, Main, and High. Slice groups, data partitioning, and gaps in `frame_num`.
/// - HEVC: profiles other than Main, Main 10, and Main Still Picture. Unequal luma and chroma
///   depth, other depths, and other chroma formats. A picture that binds a reference the DPB
///   lacks, because Vulkan Video activates a DPB slot only through a decode.
/// - VP9: profiles other than 0 and 2, and a frame that grows past its session while it keeps
///   references. [`VulkanDecoder::with_max_coded_size`] sizes that session.
/// - AV1: profiles other than Main, 4:0:0, and depths other than 8 and 10. Large-scale tile
///   coding. A frame that reads a reference name with no frame, or one `ref_order_hint` states
///   as lost. A size outside what the driver lists, for each frame. Film grain, when the driver lists no `filmGrainSupport` profile.
///
/// # Output
///
/// An H.264 IDR picture with `no_output_of_prior_pics_flag` 1 removes each frame that still
/// waits for output, as ITU-T H.264 C.4.4 states. An IDR picture that changes the frame size
/// or `max_dec_frame_buffering` does the same. So a stream with either can give fewer frames
/// than packets.
///
/// HEVC follows ITU-T H.265 C.5.2: `NoOutputOfPriorPicsFlag` drops what waits, a RASL picture
/// of a first CRA picture does not decode, and `pic_output_flag` 0 outputs nothing.
///
/// VP9 returns the last shown frame of a chunk, as libvpx does, and a hidden frame only
/// fills a slot. `show_existing_frame` copies the slot again.
///
/// AV1 returns the last shown frame of a temporal unit, the highest spatial layer of operating
/// point 0. It drops the OBUs outside that point. A frame takes `UpscaledWidth` by `FrameHeight`,
/// as dav1d gives, and the render size goes unread. A frame with `apply_grain` 1 takes film grain
/// from the driver, in an output image of its own. Its slot stays free of grain. Its grained
/// planes stay with the slot, because a later `show_existing_frame` repeats those planes.
///
/// # Threads
///
/// The decoder is `Send`. Each picture's decode and copy finish before `send` returns.
///
/// # Errors
///
/// After an error, each later `send` and `flush` returns that error. A wait on the GPU stops
/// after a fixed time and returns an error. So does a lost device, by name, as
/// [`DecodeError::VulkanDeviceLost`]. A drop then waits a fixed time, and keeps each object a
/// stalled queue can still use.
pub struct VulkanDecoder {
    device: Arc<VulkanDevice>,
    backend: Backend,
    flushed: bool,
    error: Option<DecodeError>,
}

impl VulkanDecoder {
    /// Opens a decoder for `codec` on `device`.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::VulkanRefused`]: a codec or stream the driver or this backend leaves out.
    /// - [`DecodeError::Bitstream`]: a record that does not parse.
    /// - [`DecodeError::Vulkan`]: a call that fails.
    /// - [`DecodeError::VulkanDeviceLost`]: a device that is lost. A call to
    ///   [`VulkanDevice::open_shared`] opens a new one when the machine allows it.
    pub fn new(
        device: &Arc<VulkanDevice>,
        codec: DecodeCodec,
        config: &[u8],
    ) -> Result<Self, DecodeError> {
        Self::with_dpb_preference(device, codec, config, DpbPreference::Auto)
    }

    /// As [`VulkanDecoder::new`], with a VP9 session of at least `max_coded`, up to what the
    /// driver lists. A frame that keeps references can then grow to that size. The other codecs
    /// take their size from their parameter sets or sequence header, and ignore this.
    ///
    /// # Errors
    ///
    /// As [`VulkanDecoder::new`].
    pub fn with_max_coded_size(
        device: &Arc<VulkanDevice>,
        codec: DecodeCodec,
        config: &[u8],
        max_coded: (u32, u32),
    ) -> Result<Self, DecodeError> {
        Self::open(device, codec, config, DpbPreference::Auto, max_coded)
    }

    /// As [`VulkanDecoder::new`], with a stated DPB mode. A test runs each mode the
    /// driver lists.
    ///
    /// # Errors
    ///
    /// As [`VulkanDecoder::new`]. [`DecodeError::VulkanRefused`] when the driver lacks
    /// the mode.
    pub fn with_dpb_preference(
        device: &Arc<VulkanDevice>,
        codec: DecodeCodec,
        config: &[u8],
        preference: DpbPreference,
    ) -> Result<Self, DecodeError> {
        Self::open(device, codec, config, preference, (0, 0))
    }

    fn open(
        device: &Arc<VulkanDevice>,
        codec: DecodeCodec,
        config: &[u8],
        preference: DpbPreference,
        max_coded: (u32, u32),
    ) -> Result<Self, DecodeError> {
        device.check_live("a new decoder")?;
        check_device_codec(device, codec)?;
        let backend = match codec {
            DecodeCodec::H264 => {
                Backend::H264(H264Decoder::new(Arc::clone(device), config, preference)?)
            }
            DecodeCodec::Hevc => {
                Backend::Hevc(HevcDecoder::new(Arc::clone(device), config, preference)?)
            }
            DecodeCodec::Vp9 => {
                Backend::Vp9(Vp9Decoder::new(Arc::clone(device), preference, max_coded))
            }
            DecodeCodec::Av1 => Backend::Av1(Box::new(Av1Decoder::new(
                Arc::clone(device),
                config,
                preference,
            )?)),
        };
        Ok(Self {
            device: Arc::clone(device),
            backend,
            flushed: false,
            error: None,
        })
    }

    fn keep_error(&mut self, result: Result<(), DecodeError>) -> Result<(), DecodeError> {
        if let Err(error) = &result {
            if matches!(error, DecodeError::VulkanDeviceLost { .. }) {
                self.device.mark_lost();
            }
            self.error = Some(error.clone());
        }
        result
    }

    fn stored_error(&self) -> Result<(), DecodeError> {
        match &self.error {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }
}

impl Decoder<WgpuRuntime> for VulkanDecoder {
    fn send(&mut self, packet: &[u8], pts: i64) -> Result<(), DecodeError> {
        self.stored_error()?;
        if self.flushed {
            return Err(DecodeError::AfterFlush { pts });
        }
        if packet.is_empty() {
            return Err(DecodeError::EmptyPacket { pts });
        }
        let live = self.device.check_live(&format!("the packet at pts {pts}"));
        let result = live.and_then(|()| match &mut self.backend {
            Backend::H264(decoder) => decoder.send(packet, pts),
            Backend::Hevc(decoder) => decoder.send(packet, pts),
            Backend::Vp9(decoder) => decoder.send(packet, pts),
            Backend::Av1(decoder) => decoder.send(packet, pts),
        });
        self.keep_error(result)
    }

    fn flush(&mut self) -> Result<(), DecodeError> {
        self.stored_error()?;
        if self.flushed {
            return Ok(());
        }
        self.flushed = true;
        match &mut self.backend {
            Backend::H264(decoder) => decoder.flush(),
            Backend::Hevc(decoder) => decoder.flush(),
            // VP9 and AV1 output has no reorder, so nothing waits.
            Backend::Vp9(_) | Backend::Av1(_) => {}
        }
        Ok(())
    }

    fn receive(&mut self) -> Option<DecodedFrame<WgpuRuntime>> {
        match &mut self.backend {
            Backend::H264(decoder) => decoder.receive(),
            Backend::Hevc(decoder) => decoder.receive(),
            Backend::Vp9(decoder) => decoder.receive(),
            Backend::Av1(decoder) => decoder.receive(),
        }
    }

    fn route(&self) -> FrameRoute {
        FrameRoute::Device
    }
}
