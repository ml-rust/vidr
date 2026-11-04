//! [`VulkanEncoder`]: Vulkan Video behind the backend-neutral [`Encoder`] trait.

use std::collections::VecDeque;
use std::sync::Arc;

use ash::vk;
use numr::runtime::wgpu::WgpuRuntime;

use crate::encode::dts::DtsClock;
use crate::encode::sample_filter::SampleFilter;
use crate::encode::{EncodeCodec, EncodeConfig, EncodeFrame, EncodedPacket, Encoder};
use crate::error::EncodeError;
use crate::route::FrameRoute;
use crate::vulkan::device::VulkanDevice;
use crate::vulkan::gap;

use super::av1::Av1Stream;
use super::caps::{EncodeCaps, check_limits, coded_extent, quality_level};
use super::engine::{EncodeEngine, EngineSpec};
use super::gop::GopState;
use super::h264::H264Stream;
use super::hevc::HevcStream;
use super::planes::FrameLayout;
use super::profile::{EncodeProfile, query_caps};
use super::rate::{RatePlan, RateState, plan_rate};

/// One codec's structures.
enum CodecStream {
    H264(H264Stream),
    Hevc(HevcStream),
    Av1(Box<Av1Stream>),
}

/// The codecs of `ops`, for a refusal.
fn listed_codecs(ops: vk::VideoCodecOperationFlagsKHR) -> String {
    let names: Vec<&str> = [
        (vk::VideoCodecOperationFlagsKHR::ENCODE_H264, "H.264"),
        (vk::VideoCodecOperationFlagsKHR::ENCODE_H265, "HEVC"),
        (gap::ENCODE_AV1, "AV1"),
    ]
    .into_iter()
    .filter(|(op, _)| ops.contains(*op))
    .map(|(_, name)| name)
    .collect();
    if names.is_empty() {
        "no codec vidr names".to_owned()
    } else {
        names.join(", ")
    }
}

/// The operation that encodes `codec`.
const fn codec_op(codec: EncodeCodec) -> vk::VideoCodecOperationFlagsKHR {
    match codec {
        EncodeCodec::H264 => vk::VideoCodecOperationFlagsKHR::ENCODE_H264,
        EncodeCodec::Hevc => vk::VideoCodecOperationFlagsKHR::ENCODE_H265,
        EncodeCodec::Av1 => gap::ENCODE_AV1,
    }
}

/// Refuses what no driver call can change: B-frames, and an odd AV1 size.
fn check_config(config: &EncodeConfig) -> Result<(), EncodeError> {
    config.validate()?;
    if config.b_frames > 0 {
        return Err(EncodeError::BFramesUnsupported {
            codec: config.codec,
            requested: config.b_frames,
            maximum: 0,
        });
    }
    // `validate` refuses this for H.264 and HEVC. The input image is 4:2:0 with even sides, and an
    // AV1 frame has no cropping that hides one more column.
    if config.width % 2 == 1 || config.height % 2 == 1 {
        return Err(EncodeError::OddSize {
            codec: config.codec,
            width: config.width,
            height: config.height,
        });
    }
    Ok(())
}

/// Hardware encode on Vulkan Video, from numr WebGPU tensors.
///
/// Frames hold [`PlaneForm::Words`](crate::decode::PlaneForm::Words) NV12 or P010 planes on
/// [`VulkanDevice::numr_device`]. Each frame takes one device-to-device copy from numr's buffers to
/// an input image, and no host copy. Only the compressed bitstream reaches the host.
///
/// # Codecs
///
/// - H.264 High at 8 bits.
/// - HEVC Main at 8 bits, and Main 10 at 10 bits where the driver lists it.
/// - AV1 Main at 8 or 10 bits on a driver with `VK_KHR_video_encode_av1`. No GPU has run this path.
///
/// # Structure
///
/// Each stream holds one previous frame as its reference: an IDR picture every `gop_length` frames
/// and P pictures between them, so `dts` equals `pts`. `b_frames` above 0 is refused by name.
///
/// # Rate control
///
/// [`RateControl::VbrQuality`](crate::encode::RateControl::VbrQuality) runs at a constant quantiser
/// where the driver lists its disabled mode: `cq` is the QP for H.264 and HEVC, and maps to
/// `q_index` for AV1. A driver with no such mode takes VBR, then CBR, at a bitrate mapped from the
/// frame size, the frame rate, and `cq`. [`VulkanEncoder::rate_control`] states which one a stream
/// runs. [`Preset`](crate::encode::Preset) spreads over the driver's quality levels.
///
/// # Output
///
/// Packets are MP4 and Matroska samples, as [`EncodedPacket`] states. The session parameters give
/// the driver's own SPS, PPS, and VPS, or its AV1 sequence header. The config record comes from
/// those bytes once the first packet exists. The driver's overrides of a parameter set reach the
/// record, so a decoder reads what the driver coded.
///
/// # Threads
///
/// The encoder is `Send`. Each frame's copy and encode finish before `send` returns.
///
/// # Errors
///
/// After a driver error, each later `send` and `flush` returns that error. A refused frame leaves
/// the encoder usable. A wait on the GPU stops after a fixed time and returns
/// [`EncodeError::VulkanWaitTimeout`]. A lost device returns [`EncodeError::VulkanDeviceLost`]. A
/// drop then waits a fixed time, and keeps each object a stalled queue can still use.
pub struct VulkanEncoder {
    device: Arc<VulkanDevice>,
    config: EncodeConfig,
    layout: FrameLayout,
    gop: GopState,
    clock: DtsClock,
    filter: SampleFilter,
    /// The driver's parameter sets or sequence header. The first sample takes them.
    parameters: Vec<u8>,
    config_record: Option<Vec<u8>>,
    packets: VecDeque<EncodedPacket>,
    last_pts: Option<i64>,
    flushed: bool,
    error: Option<EncodeError>,
    /// Dropped before `stream`, which holds what the session parameters point to.
    engine: EncodeEngine,
    stream: CodecStream,
}

impl VulkanEncoder {
    /// Opens an encoder on `device`.
    ///
    /// # Errors
    ///
    /// - [`EncodeError::OddSize`] or [`EncodeError::InvalidConfig`] for a config no GPU can take.
    /// - [`EncodeError::BFramesUnsupported`] for `b_frames` above 0.
    /// - [`EncodeError::CodecUnsupported`], [`EncodeError::BitDepthUnsupported`],
    ///   [`EncodeError::SizeOutOfRange`], or [`EncodeError::VulkanRefused`] for what this driver
    ///   leaves out.
    /// - [`EncodeError::Vulkan`], [`EncodeError::VulkanNoMemoryType`], or
    ///   [`EncodeError::VulkanDeviceLost`] when a driver call fails.
    pub fn new(device: &Arc<VulkanDevice>, config: EncodeConfig) -> Result<Self, EncodeError> {
        check_config(&config)?;
        device.check_live("a new encoder")?;
        let codec = config.codec;
        let Some(placement) = device.encode else {
            return Err(EncodeError::CodecUnsupported {
                codec,
                available: "no codec, because the device has no encode queue".to_owned(),
            });
        };
        if !placement.ops.contains(codec_op(codec)) {
            return Err(EncodeError::CodecUnsupported {
                codec,
                available: listed_codecs(placement.ops),
            });
        }
        let profile = EncodeProfile::new(&config)?;
        let caps = query_caps(device, &profile)?;
        let coded = coded_extent(codec, (config.width, config.height), &caps);
        check_limits(&config, coded, &caps)?;
        let plan = plan_rate(&config, &caps)?;
        let rate = RateState::new(
            &config,
            plan,
            quality_level(config.preset, caps.max_quality_levels),
        )?;
        let layout = FrameLayout::new(&config);
        let stream = open_stream(&config, coded, &caps, plan)?;
        let mut engine = EncodeEngine::new(
            device,
            EngineSpec {
                profile: &profile,
                caps: &caps,
                coded,
                layout,
                rate,
                family: placement.family,
            },
        )?;
        let parameters = match &stream {
            CodecStream::H264(stream) => stream.parameters(&mut engine),
            CodecStream::Hevc(stream) => stream.parameters(&mut engine),
            CodecStream::Av1(stream) => stream.parameters(&mut engine),
        }?;
        Ok(Self {
            device: Arc::clone(device),
            layout,
            gop: GopState::new(config.gop_length),
            clock: DtsClock::new(0),
            filter: SampleFilter::new(codec),
            parameters,
            config_record: None,
            packets: VecDeque::new(),
            last_pts: None,
            flushed: false,
            error: None,
            engine,
            stream,
            config,
        })
    }

    /// The config the encoder opened with.
    pub const fn config(&self) -> &EncodeConfig {
        &self.config
    }

    /// The rate control this stream runs: `constant quantiser`, `VBR`, or `CBR`.
    pub fn rate_control(&self) -> &'static str {
        self.engine.rate().plan().name()
    }

    fn stored_error(&self) -> Result<(), EncodeError> {
        match &self.error {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    fn keep_error(&mut self, result: Result<(), EncodeError>) -> Result<(), EncodeError> {
        if let Err(error) = &result {
            if matches!(error, EncodeError::VulkanDeviceLost { .. }) {
                self.device.mark_lost();
            }
            self.error = Some(error.clone());
        }
        result
    }

    fn encode_frame(
        &mut self,
        frame: &EncodeFrame<'_, WgpuRuntime>,
        pts: i64,
        duration: i64,
    ) -> Result<(), EncodeError> {
        self.device.check_live(&format!("the frame at pts {pts}"))?;
        self.engine.load_frame(frame)?;
        let plan = self.gop.next();
        let bytes = match &mut self.stream {
            CodecStream::H264(stream) => stream.encode(&mut self.engine, &plan, pts),
            CodecStream::Hevc(stream) => stream.encode(&mut self.engine, &plan, pts),
            CodecStream::Av1(stream) => stream.encode(&mut self.engine, &plan, pts),
        }?;
        self.last_pts = Some(pts);
        self.clock.push(pts, duration);
        let data = self.sample(&bytes, plan.idr)?;
        if self.config_record.is_none() {
            self.config_record = self.filter.config_record()?;
        }
        self.packets.push_back(EncodedPacket {
            data,
            pts,
            dts: self.clock.next(pts)?,
            duration,
            keyframe: plan.idr,
        });
        Ok(())
    }

    /// The MP4 sample of one picture. The first picture of H.264 and HEVC takes the parameter sets
    /// ahead of it, and the filter moves them to the record. Each AV1 key frame takes its sequence
    /// header, because the AV1 ISO-BMFF binding asks it of sync samples.
    fn sample(&mut self, bytes: &[u8], idr: bool) -> Result<Vec<u8>, EncodeError> {
        let first = self.config_record.is_none();
        let with_parameters = needs_parameters(self.config.codec, first, idr);
        if !with_parameters {
            return self.filter.sample(bytes);
        }
        let mut unit = Vec::with_capacity(self.parameters.len() + bytes.len());
        unit.extend_from_slice(&self.parameters);
        unit.extend_from_slice(bytes);
        self.filter.sample(&unit)
    }
}

/// Whether a picture's output starts with the driver's parameter bytes.
const fn needs_parameters(codec: EncodeCodec, first: bool, idr: bool) -> bool {
    match codec {
        EncodeCodec::H264 | EncodeCodec::Hevc => first,
        EncodeCodec::Av1 => idr,
    }
}

fn open_stream(
    config: &EncodeConfig,
    coded: (u32, u32),
    caps: &EncodeCaps,
    plan: RatePlan,
) -> Result<CodecStream, EncodeError> {
    Ok(match config.codec {
        EncodeCodec::H264 => CodecStream::H264(H264Stream::new(config, coded, caps)?),
        EncodeCodec::Hevc => {
            let constant = matches!(plan, RatePlan::Constant { .. });
            CodecStream::Hevc(HevcStream::new(config, coded, caps, constant)?)
        }
        EncodeCodec::Av1 => CodecStream::Av1(Box::new(Av1Stream::new(config, caps)?)),
    })
}

impl Encoder<WgpuRuntime> for VulkanEncoder {
    fn send(
        &mut self,
        frame: &EncodeFrame<'_, WgpuRuntime>,
        pts: i64,
        duration: i64,
    ) -> Result<(), EncodeError> {
        self.stored_error()?;
        if self.flushed {
            return Err(EncodeError::AfterFlush { pts });
        }
        if duration <= 0 {
            return Err(EncodeError::NonPositiveDuration { pts, duration });
        }
        if let Some(previous) = self.last_pts
            && pts <= previous
        {
            return Err(EncodeError::PtsNotIncreasing { pts, previous });
        }
        self.layout.check_frame(frame)?;
        let result = self.encode_frame(frame, pts, duration);
        self.keep_error(result)
    }

    fn flush(&mut self) -> Result<(), EncodeError> {
        self.stored_error()?;
        // Each frame's encode finished in its `send`, so nothing waits.
        self.flushed = true;
        Ok(())
    }

    fn receive(&mut self) -> Option<EncodedPacket> {
        self.packets.pop_front()
    }

    fn config_record(&self) -> Option<&[u8]> {
        self.config_record.as_deref()
    }

    fn route(&self) -> FrameRoute {
        FrameRoute::Device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::PixelFormat;
    use crate::time::Rational;

    fn config(codec: EncodeCodec) -> EncodeConfig {
        let rate = Rational::new(30, 1).unwrap_or(Rational::ONE);
        let base = Rational::new(1, 30).unwrap_or(Rational::ONE);
        EncodeConfig::new(codec, 640, 360, PixelFormat::Nv12, rate, base)
    }

    #[test]
    fn b_frames_are_refused_by_name() {
        let mut asks = config(EncodeCodec::H264);
        asks.b_frames = 2;
        assert_eq!(
            check_config(&asks),
            Err(EncodeError::BFramesUnsupported {
                codec: EncodeCodec::H264,
                requested: 2,
                maximum: 0,
            })
        );
        assert_eq!(check_config(&config(EncodeCodec::H264)), Ok(()));
    }

    #[test]
    fn an_odd_size_is_refused_for_each_codec() {
        for codec in [EncodeCodec::H264, EncodeCodec::Hevc, EncodeCodec::Av1] {
            let mut odd = config(codec);
            odd.width = 641;
            assert_eq!(
                check_config(&odd),
                Err(EncodeError::OddSize {
                    codec,
                    width: 641,
                    height: 360,
                })
            );
        }
    }

    #[test]
    fn a_refusal_names_the_listed_codecs() {
        let both = vk::VideoCodecOperationFlagsKHR::ENCODE_H264
            | vk::VideoCodecOperationFlagsKHR::ENCODE_H265;
        assert_eq!(listed_codecs(both), "H.264, HEVC");
        assert_eq!(
            listed_codecs(vk::VideoCodecOperationFlagsKHR::empty()),
            "no codec vidr names"
        );
        assert_eq!(codec_op(EncodeCodec::Av1), gap::ENCODE_AV1);
    }

    #[test]
    fn parameter_bytes_lead_the_first_sample_and_each_av1_key_frame() {
        assert!(needs_parameters(EncodeCodec::H264, true, true));
        assert!(!needs_parameters(EncodeCodec::H264, false, true));
        assert!(!needs_parameters(EncodeCodec::Hevc, false, true));
        assert!(needs_parameters(EncodeCodec::Av1, false, true));
        assert!(!needs_parameters(EncodeCodec::Av1, false, false));
    }

    const fn assert_send<T: Send>() {}
    const _: () = assert_send::<VulkanEncoder>();
}
