//! [`VulkanDecoder`]: Vulkan Video behind the backend-neutral [`Decoder`] trait.

use std::sync::Arc;

use numr::runtime::wgpu::WgpuRuntime;

use super::caps::DpbPreference;
use super::device::VulkanDevice;
use super::h264::H264Decoder;
use crate::decode::{DecodeCodec, DecodedFrame, Decoder};
use crate::error::DecodeError;
use crate::route::FrameRoute;

/// One codec on the Vulkan core. Later codecs add a variant.
enum Backend {
    H264(H264Decoder),
}

/// Hardware decode on Vulkan Video, into numr WebGPU tensors.
///
/// Frames hold [`PlaneForm::Words`](crate::decode::PlaneForm::Words) NV12 planes on
/// [`VulkanDevice::numr_device`]. Each frame takes one device-to-device copy from a
/// video image to numr buffers, and no host copy.
///
/// # Input
///
/// `config` is the track's `avcC` payload. With it, `send` takes container samples.
/// With an empty `config`, `send` takes Annex-B access units.
///
/// # Refused
///
/// A refusal names what it refuses: other codecs, 10 bits, other chroma formats, field
/// and MBAFF coding, profiles other than Baseline, Main, and High, slice groups, data
/// partitioning, gaps in `frame_num`, and what the driver's capabilities leave out.
///
/// # Output
///
/// An IDR picture with `no_output_of_prior_pics_flag` 1 removes each frame that still waits
/// for output, as ITU-T H.264 C.4.4 states. An IDR picture that changes the frame size or
/// `max_dec_frame_buffering` does the same. So a stream with either can give fewer frames
/// than packets.
///
/// # Threads
///
/// The decoder is `Send`. Each picture's decode and copy finish before `send` returns.
pub struct VulkanDecoder {
    backend: Backend,
    flushed: bool,
    error: Option<DecodeError>,
}

impl VulkanDecoder {
    /// Opens a decoder for `codec` on `device`.
    ///
    /// # Errors
    ///
    /// [`DecodeError::VulkanRefused`] for a codec or stream the driver or this backend
    /// leaves out, [`DecodeError::Bitstream`] for a record that does not parse, and
    /// [`DecodeError::Vulkan`] for a call that fails.
    pub fn new(
        device: &Arc<VulkanDevice>,
        codec: DecodeCodec,
        config: &[u8],
    ) -> Result<Self, DecodeError> {
        Self::with_dpb_preference(device, codec, config, DpbPreference::Auto)
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
        let backend = match codec {
            DecodeCodec::H264 => {
                Backend::H264(H264Decoder::new(Arc::clone(device), config, preference)?)
            }
            other => {
                return Err(DecodeError::VulkanRefused {
                    codec: other,
                    what: "the codec".to_owned(),
                    reason: "this release decodes H.264 only on Vulkan Video, and this codec is \
                             not supported yet"
                        .to_owned(),
                });
            }
        };
        Ok(Self {
            backend,
            flushed: false,
            error: None,
        })
    }

    fn keep_error(&mut self, result: Result<(), DecodeError>) -> Result<(), DecodeError> {
        if let Err(error) = &result {
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
        if self.flushed {
            return Err(DecodeError::AfterFlush { pts });
        }
        if packet.is_empty() {
            return Err(DecodeError::EmptyPacket { pts });
        }
        self.stored_error()?;
        let Backend::H264(decoder) = &mut self.backend;
        let result = decoder.send(packet, pts);
        self.keep_error(result)
    }

    fn flush(&mut self) -> Result<(), DecodeError> {
        if self.flushed {
            return Ok(());
        }
        self.flushed = true;
        self.stored_error()?;
        let Backend::H264(decoder) = &mut self.backend;
        decoder.flush();
        Ok(())
    }

    fn receive(&mut self) -> Option<DecodedFrame<WgpuRuntime>> {
        let Backend::H264(decoder) = &mut self.backend;
        decoder.receive()
    }

    fn route(&self) -> FrameRoute {
        FrameRoute::Device
    }
}
