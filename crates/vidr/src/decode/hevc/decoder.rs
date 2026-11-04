//! `HevcDecoder` and its path with every stage on the calling thread. The `threaded` module
//! holds the path over the worker pool.

use std::collections::{BTreeMap, VecDeque};
use std::num::NonZero;

use numr::dtype::DType;
use numr::runtime::Runtime;

use super::super::cpu::dsp::sample::Sample;
use super::super::cpu::hevc::dpb::buffer::Frame;
use super::super::cpu::hevc::error::HevcError;
use super::super::cpu::hevc::params::MAX_LUMA_PICTURE_SIZE;
use super::super::cpu::hevc::picture::{DisplayWindow, HostScratch};
use super::super::cpu::hevc::stream::{
    DecodeOptions, DecodedPicture, StartedPicture, StreamDecoder,
};
use super::super::{
    ChromaFormat, ColorFacts, DecodeCodec, DecodedFrame, Decoder, PixelFormat, vui_color,
};
use crate::bitstream::hevc_sps_full::SeqParameterSet;
use crate::bitstream::{HevcConfig, HevcNalType, LengthSize, annexb, length_prefixed};
use crate::error::DecodeError;
use crate::frame::{upload_planes, upload_route};
use crate::geometry::Dimensions;
use crate::route::FrameRoute;

use super::threaded::ThreadedEngine;

pub(super) const DECODER: &str = "HevcDecoder";
/// The luma samples of the frames one `send` can output: 16 pictures, the largest
/// `MaxDpbSize` (A-2), at `MaxLumaPs` of level 6.2. One access unit stays below it.
const MAX_SEND_LUMA_SAMPLES: u64 = 16 * MAX_LUMA_PICTURE_SIZE;
/// The most workers the decoder picks by itself. Past this, more pictures in flight cost
/// memory and gain little. [`HevcOptions::threads`] sets any other count.
const MAX_AUTO_THREADS: usize = 32;

/// What a picture's packet stated when the picture started decoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PendingPicture {
    pts: i64,
    color: ColorFacts,
}

/// What [`HevcDecoder::with_options`] takes. The default picks the thread count and runs
/// no hash check.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HevcOptions {
    /// Worker threads. `None` picks them from [`std::thread::available_parallelism`], at
    /// most 32. `Some(1)` decodes on the calling thread, with no worker and no channel.
    /// `Some(0)` is an error.
    pub threads: Option<usize>,
    /// Compares each picture with its decoded picture hash SEI message (D.3.20). A
    /// difference is a decode error. A picture with no such message passes unchecked.
    pub check_picture_hash: bool,
}

/// The code that runs the core.
pub(super) enum Engine {
    /// Every stage on the calling thread.
    Serial(Box<StreamDecoder>),
    /// Picture stages on a worker pool.
    Threaded(Box<ThreadedEngine>),
}

impl Engine {
    /// Takes one unit of the `hvcC` record.
    fn config_unit(&mut self, nal: &[u8]) -> Result<(), HevcError> {
        match self {
            Self::Serial(stream) => stream.decode_nal(nal),
            Self::Threaded(engine) => engine.config_unit(nal),
        }
    }
}

/// HEVC software decode into numr tensors on any backend.
///
/// # Threads
///
/// [`Self::new`] runs on a pool of worker threads. Header parsing, reference marking,
/// and output order stay on the calling thread. Each picture's parse starts on a worker when
/// its slices have arrived, and its reconstruction starts when the pictures it reads have
/// finished. Output bytes, order, and pts are the same for every thread count.
///
/// `send` returns while pictures run, and later `send` calls deliver their frames. A worker's
/// error comes back from a later `send` or from `flush`, in decode order: the frames before
/// it arrive first, as with one thread. `flush` waits for every picture.
///
/// # Input
///
/// `config` is the track's `hvcC` payload. Its parameter sets go in at open. With it,
/// `send` takes length-prefixed container samples. With an empty `config`, `send`
/// takes Annex-B data. Pass one access unit per packet, or a part of one. Two
/// pictures in one packet share its pts.
///
/// # Output
///
/// Main and Main 10, 4:2:0 at 8 or 10 bits, as NV12 or P010 in the form the backend's
/// colour kernels take. The display size is the SPS conformance window.
///
/// Frames leave in output order (C.5.2). Each frame carries the pts of the packet
/// that held the picture's first slice segment. Its colour comes from the VUI of the
/// SPS active at that point, so a later SPS never relabels a held frame. A frame
/// always reports progressive, because this decoder reads no picture timing SEI.
///
/// A skipped RASL picture, a picture with `PicOutputFlag` 0, and a picture that
/// C.5.2.2 drops at an IRAP picture give no frame.
///
/// # Refusals
///
/// | Stream | Error |
/// | --- | --- |
/// | Not 4:2:0, unequal luma and chroma depths, or a depth other than 8 or 10 | [`DecodeError::UnsupportedLayout`] |
/// | A multilayer, 3D, or tiles with wavefront tool | [`DecodeError::HevcTool`] |
/// | A range extension or screen content tool | [`DecodeError::Bitstream`] naming the flag |
///
/// `new` checks each SPS of `hvcC` for the layout, and `send` checks each SPS in a
/// packet. The other refusals come at the first picture that activates the SPS.
///
/// # Limits
///
/// Bad input ends in an error, and memory stays tied to these limits:
///
/// | Limit | Value | Error |
/// | --- | --- | --- |
/// | Luma samples of one picture | 35651584, `MaxLumaPs` of level 6.2 (Table A.8) | [`DecodeError::Software`] |
/// | Samples on one picture side | 16888 (A.4.1) | [`DecodeError::Software`] |
/// | Pictures in the buffer | `MaxDpbSize` of level 6.2 at the picture size: 6 to 16 (A-2) | [`DecodeError::Software`] |
/// | Luma samples of the frames from one `send` | 16 pictures of the largest size | [`DecodeError::Software`] |
/// | Pictures in flight on workers | Worker threads plus the picture buffer size, at most 16 | `send` blocks |
///
/// A caller holds the frame queue to its own size: call [`Decoder::receive`] until
/// it returns `None` after each `send`. With workers, one call can queue the frames of the
/// pictures in flight on top of its own.
///
/// # Errors after the first
///
/// After a decode error or a refusal, every later `send` and `flush` returns the same
/// error. A packet whose NAL units do not split leaves the decoder usable.
pub struct HevcDecoder<R: Runtime> {
    pub(super) device: R::Device,
    pub(super) engine: Engine,
    /// The size of each sample's length fields. `None` for Annex-B input.
    length_size: Option<LengthSize>,
    /// Pictures in the core that give a frame later, by [`Frame::decode_order`].
    pub(super) pending: BTreeMap<u64, PendingPicture>,
    pub(super) scratch: HostScratch,
    pub(super) frames: VecDeque<DecodedFrame<R>>,
    /// The pts of the packet in `send`.
    send_pts: i64,
    /// The luma samples of the frames the packet in `send` has output.
    send_luma: u64,
    /// The largest `send_luma`.
    pub(super) send_luma_limit: u64,
    flushed: bool,
    error: Option<DecodeError>,
}

impl<R> HevcDecoder<R>
where
    R: Runtime<DType = DType>,
{
    /// Opens a decoder whose frames go to tensors on `device`. It picks the worker threads
    /// from [`std::thread::available_parallelism`].
    ///
    /// `config` is the track's `hvcC` payload, or empty for raw Annex-B input.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::Bitstream`] for a config record or parameter set that does
    ///   not parse.
    /// - [`DecodeError::UnsupportedLayout`] for a layout an SPS in `hvcC` states.
    /// - [`DecodeError::Software`] when a worker thread does not start.
    pub fn new(device: &R::Device, config: &[u8]) -> Result<Self, DecodeError> {
        Self::with_options(device, config, HevcOptions::default())
    }

    /// As [`Self::new`], with `threads` workers. `1` decodes on the calling thread, with no
    /// worker and no channel.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], and [`DecodeError::Software`] for a `threads` of 0.
    pub fn with_threads(
        device: &R::Device,
        config: &[u8],
        threads: usize,
    ) -> Result<Self, DecodeError> {
        let options = HevcOptions {
            threads: Some(threads),
            ..HevcOptions::default()
        };
        Self::with_options(device, config, options)
    }

    /// As [`Self::new`], and also compares each picture with its decoded picture hash
    /// SEI message (D.3.20). A difference is a decode error. A picture with no such
    /// message passes unchecked.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn with_picture_hash_check(device: &R::Device, config: &[u8]) -> Result<Self, DecodeError> {
        let options = HevcOptions {
            check_picture_hash: true,
            ..HevcOptions::default()
        };
        Self::with_options(device, config, options)
    }

    /// As [`Self::new`], with the choices of `options`.
    ///
    /// # Errors
    ///
    /// As [`Self::with_threads`].
    pub fn with_options(
        device: &R::Device,
        config: &[u8],
        options: HevcOptions,
    ) -> Result<Self, DecodeError> {
        let threads = match options.threads {
            Some(0) => {
                return Err(DecodeError::Software {
                    decoder: DECODER,
                    call: "HevcDecoder::with_options",
                    input: "a thread count of 0".to_owned(),
                    reason: "the thread count starts at 1".to_owned(),
                });
            }
            Some(threads) => threads,
            None => std::thread::available_parallelism()
                .map_or(1, NonZero::get)
                .min(MAX_AUTO_THREADS),
        };
        let core = DecodeOptions {
            check_picture_hash: options.check_picture_hash,
        };
        let engine = if threads == 1 {
            Engine::Serial(Box::new(StreamDecoder::new(core)))
        } else {
            let started = ThreadedEngine::start(threads, core);
            Engine::Threaded(Box::new(started.map_err(|error| {
                let input = format!("{threads} threads");
                core_error("HevcDecoder::with_options", &input, error)
            })?))
        };
        Self::open(device, config, engine)
    }

    pub(super) fn open(
        device: &R::Device,
        config: &[u8],
        mut engine: Engine,
    ) -> Result<Self, DecodeError> {
        let mut length_size = None;
        if !config.is_empty() {
            let parsed = HevcConfig::parse(config)?;
            length_size = Some(parsed.length_size);
            for nal in parsed.nal_units() {
                check_sps(nal)?;
                engine.config_unit(nal).map_err(|error| {
                    core_error("StreamDecoder::decode_nal", "the hvcC record", error)
                })?;
            }
        }
        Ok(Self {
            device: device.clone(),
            engine,
            length_size,
            pending: BTreeMap::new(),
            scratch: HostScratch::default(),
            frames: VecDeque::new(),
            send_pts: 0,
            send_luma: 0,
            send_luma_limit: MAX_SEND_LUMA_SAMPLES,
            flushed: false,
            error: None,
        })
    }

    /// Frames decoded and not yet received.
    pub fn queued(&self) -> usize {
        self.frames.len()
    }

    fn stored_error(&self) -> Result<(), DecodeError> {
        match &self.error {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    /// Stores a first error. It ends the workers, because no later picture can matter.
    fn keep_error(&mut self, result: Result<(), DecodeError>) -> Result<(), DecodeError> {
        if let Err(error) = &result {
            self.error = Some(error.clone());
            self.stop_workers();
        }
        result
    }

    /// Joins the workers. A decoder with none does nothing.
    fn stop_workers(&mut self) {
        if let Engine::Threaded(engine) = &mut self.engine {
            engine.stop();
        }
    }

    fn decode_units(&mut self, units: &[&[u8]], pts: i64) -> Result<(), DecodeError> {
        for nal in units {
            self.decode_unit(nal, pts)?;
        }
        Ok(())
    }

    /// Decodes one NAL unit, and queues every picture it completes.
    fn decode_unit(&mut self, nal: &[u8], pts: i64) -> Result<(), DecodeError> {
        let (result, started) = match &mut self.engine {
            Engine::Serial(stream) => {
                check_sps(nal)?;
                let result = stream.decode_nal(nal);
                (result, stream.take_started())
            }
            Engine::Threaded(_) => return self.threaded_unit(nal, pts),
        };
        self.note_started(started, pts);
        self.drain()?;
        self.drop_dropped(started);
        result.map_err(|error| {
            let input = format!("the {}-byte NAL unit at pts {pts}", nal.len());
            core_error("StreamDecoder::decode_nal", &input, error)
        })
    }

    /// Records the packet facts of a picture that gives a frame later.
    pub(super) fn note_started(&mut self, started: Option<StartedPicture>, pts: i64) {
        if let Some(picture) = started.filter(|picture| picture.output) {
            let pending = PendingPicture {
                pts,
                color: vui_color(&picture.signal),
            };
            self.pending.insert(picture.decode_order, pending);
        }
    }

    /// Forgets the pictures an IRAP picture drops. The core has output or dropped each
    /// earlier picture, and the caller took the output ones. So an earlier entry left is a
    /// dropped picture.
    pub(super) fn drop_dropped(&mut self, started: Option<StartedPicture>) {
        if let Some(picture) = started.filter(|picture| picture.irap_no_rasl) {
            self.pending = self.pending.split_off(&picture.decode_order);
        }
    }

    /// Moves every output picture of the core to the frame queue.
    fn drain(&mut self) -> Result<(), DecodeError> {
        while let Engine::Serial(stream) = &mut self.engine
            && let Some(picture) = stream.next_output()
        {
            let window = match &picture {
                DecodedPicture::Eight(frame) => frame.picture.window(),
                DecodedPicture::Ten(frame) => frame.picture.window(),
            };
            self.charge_send(window)?;
            let frame = match &picture {
                DecodedPicture::Eight(frame) => self.convert(frame, PixelFormat::Nv12)?,
                DecodedPicture::Ten(frame) => self.convert(frame, PixelFormat::P010)?,
            };
            self.frames.push_back(frame);
        }
        Ok(())
    }

    /// Counts one output picture against the packet in `send`, before its upload.
    pub(super) fn charge_send(&mut self, window: DisplayWindow) -> Result<(), DecodeError> {
        let luma = (window.width as u64).saturating_mul(window.height as u64);
        self.send_luma = self.send_luma.saturating_add(luma);
        if self.send_luma <= self.send_luma_limit {
            return Ok(());
        }
        Err(DecodeError::Software {
            decoder: DECODER,
            call: "HevcDecoder::send",
            input: format!("the packet at pts {}", self.send_pts),
            reason: format!(
                "the packet outputs {} luma samples and the limit is {}, so pass one access unit \
                 per packet",
                self.send_luma, self.send_luma_limit
            ),
        })
    }

    fn convert<S: Sample>(
        &mut self,
        frame: &Frame<S>,
        format: PixelFormat,
    ) -> Result<DecodedFrame<R>, DecodeError> {
        let pending = self
            .pending
            .remove(&frame.decode_order)
            .ok_or(DecodeError::UnmatchedPicture { decoder: DECODER })?;
        to_frame::<R, S>(&self.device, frame, pending, format, &mut self.scratch)
    }

    /// Ends the stream. A picture still pending here is a lost frame.
    fn finish(&mut self) -> Result<(), DecodeError> {
        match &mut self.engine {
            Engine::Serial(stream) => {
                let ended = stream.flush();
                self.drain()?;
                ended.map_err(end_error)?;
            }
            Engine::Threaded(_) => self.threaded_finish()?,
        }
        match self.pending.values().map(|picture| picture.pts).min() {
            None => Ok(()),
            Some(first_pts) => Err(DecodeError::MissingPictures {
                decoder: DECODER,
                count: self.pending.len(),
                first_pts,
            }),
        }
    }
}

/// Refuses an SPS of the base layer with no vidr frame. Every other NAL unit passes,
/// and the core drops the other layers.
pub(super) fn check_sps(nal: &[u8]) -> Result<(), DecodeError> {
    let (Some(&first), Some(&second)) = (nal.first(), nal.get(1)) else {
        return Ok(());
    };
    let base_layer = first & 1 == 0 && second >> 3 == 0;
    if !base_layer || HevcNalType::from_header(first) != HevcNalType::SPS {
        return Ok(());
    }
    let sps = SeqParameterSet::parse(nal)?;
    check_layout(sps.chroma_format_idc, sps.bit_depth_y(), sps.bit_depth_c())
}

/// Refuses a layout that has no vidr frame.
fn check_layout(chroma_format_idc: u32, luma: u32, chroma: u32) -> Result<(), DecodeError> {
    let depth = |bits: u32| u8::try_from(bits).unwrap_or(u8::MAX);
    let (bit_depth_luma, bit_depth_chroma) = (depth(luma), depth(chroma));
    let format = match chroma_format_idc {
        0 => ChromaFormat::Monochrome,
        1 => ChromaFormat::Yuv420,
        2 => ChromaFormat::Yuv422,
        _ => ChromaFormat::Yuv444,
    };
    let takes = format == ChromaFormat::Yuv420
        && luma == chroma
        && PixelFormat::for_bit_depth(bit_depth_luma).is_some();
    if takes {
        return Ok(());
    }
    Err(DecodeError::UnsupportedLayout {
        codec: DecodeCodec::Hevc,
        bit_depth_luma,
        bit_depth_chroma,
        chroma: format,
    })
}

/// Names an error of the core in the crate's public error type.
pub(super) fn core_error(call: &'static str, input: &str, error: HevcError) -> DecodeError {
    match error {
        HevcError::Header(error) => DecodeError::Bitstream(error),
        HevcError::Unsupported {
            field,
            value,
            supported,
        } => DecodeError::HevcTool {
            decoder: DECODER,
            field,
            value,
            supported,
        },
        other => DecodeError::Software {
            decoder: DECODER,
            call,
            input: input.to_owned(),
            // `Software` adds its own full stop and next action.
            reason: other.to_string().trim_end_matches('.').to_owned(),
        },
    }
}

/// Names an error of the core at the end of the stream.
pub(super) fn end_error(error: HevcError) -> DecodeError {
    core_error("StreamDecoder::flush", "the end of the stream", error)
}

/// Copies the conformance window of one picture to tensors on `device`.
pub(super) fn to_frame<R, S>(
    device: &R::Device,
    frame: &Frame<S>,
    pending: PendingPicture,
    format: PixelFormat,
    scratch: &mut HostScratch,
) -> Result<DecodedFrame<R>, DecodeError>
where
    R: Runtime<DType = DType>,
    S: Sample,
{
    let window = frame.picture.window();
    let invalid = || DecodeError::DisplayArea {
        rect: (
            window.x as i32,
            window.y as i32,
            (window.x + window.width) as i32,
            (window.y + window.height) as i32,
        ),
        coded: (
            frame.picture.luma().width() as u32,
            frame.picture.luma().height() as u32,
        ),
    };
    let display = u32::try_from(window.width)
        .ok()
        .zip(u32::try_from(window.height).ok())
        .and_then(|(width, height)| Dimensions::new(width, height).ok())
        .ok_or_else(invalid)?;
    let planes = frame.picture.host_planes(scratch).map_err(|error| {
        let input = format!("the picture with POC {}", frame.poc);
        core_error("Picture::host_planes", &input, error)
    })?;
    let (luma, chroma) = upload_planes::<R>(device, planes, format)?;
    let PendingPicture { pts, color } = pending;
    Ok(if R::packed_color_planes() {
        DecodedFrame::from_packed_planes(luma, chroma, format, pts, display, color)
    } else {
        DecodedFrame::from_planes(luma, chroma, format, pts, display, color)
    })
}

impl<R> Decoder<R> for HevcDecoder<R>
where
    R: Runtime<DType = DType>,
{
    fn send(&mut self, packet: &[u8], pts: i64) -> Result<(), DecodeError> {
        self.stored_error()?;
        if self.flushed {
            return Err(DecodeError::AfterFlush { pts });
        }
        if packet.is_empty() {
            return Err(DecodeError::EmptyPacket { pts });
        }
        self.send_pts = pts;
        self.send_luma = 0;
        let units: Vec<&[u8]> = match self.length_size {
            Some(size) => length_prefixed::split(packet, size).collect::<Result<_, _>>()?,
            None => annexb::split(packet)?.collect(),
        };
        let result = self.decode_units(&units, pts);
        self.keep_error(result)
    }

    fn flush(&mut self) -> Result<(), DecodeError> {
        self.stored_error()?;
        if self.flushed {
            return Ok(());
        }
        self.flushed = true;
        self.send_luma = 0;
        let result = self.finish();
        self.stop_workers();
        self.keep_error(result)
    }

    fn receive(&mut self) -> Option<DecodedFrame<R>> {
        if self.frames.is_empty() && self.error.is_none() {
            let result = self.advance_ready();
            // The next `send` or `flush` returns this error.
            let _ = self.keep_error(result);
        }
        self.frames.pop_front()
    }

    fn route(&self) -> FrameRoute {
        upload_route::<R>()
    }
}

#[cfg(test)]
mod tests {
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    use super::*;
    use crate::decode::cpu::hevc::stream_tests;

    const fn refused(
        bit_depth_luma: u8,
        bit_depth_chroma: u8,
        chroma: ChromaFormat,
    ) -> DecodeError {
        DecodeError::UnsupportedLayout {
            codec: DecodeCodec::Hevc,
            bit_depth_luma,
            bit_depth_chroma,
            chroma,
        }
    }

    #[test]
    fn only_equal_depths_of_8_or_10_at_420_pass() {
        assert_eq!(check_layout(1, 8, 8), Ok(()));
        assert_eq!(check_layout(1, 10, 10), Ok(()));
        assert_eq!(
            check_layout(1, 9, 9),
            Err(refused(9, 9, ChromaFormat::Yuv420))
        );
        assert_eq!(
            check_layout(1, 12, 12),
            Err(refused(12, 12, ChromaFormat::Yuv420))
        );
        assert_eq!(
            check_layout(1, 8, 10),
            Err(refused(8, 10, ChromaFormat::Yuv420))
        );
        assert_eq!(
            check_layout(0, 8, 8),
            Err(refused(8, 8, ChromaFormat::Monochrome))
        );
        assert_eq!(
            check_layout(2, 10, 10),
            Err(refused(10, 10, ChromaFormat::Yuv422))
        );
        assert_eq!(
            check_layout(3, 8, 8),
            Err(refused(8, 8, ChromaFormat::Yuv444))
        );
    }

    #[test]
    fn units_of_other_layers_and_types_pass_the_layout_check() {
        assert_eq!(check_sps(&[]), Ok(()));
        assert_eq!(check_sps(&[0x42]), Ok(()));
        // nuh_layer_id 1 on an SPS with no payload.
        assert_eq!(check_sps(&[0x42, 0x09]), Ok(()));
        // A PPS with no payload.
        assert_eq!(check_sps(&[0x44, 0x01]), Ok(()));
    }

    #[test]
    fn core_errors_keep_their_names() {
        let tool = HevcError::Unsupported {
            field: "chroma_format_idc",
            value: 3,
            supported: "1, which is 4:2:0",
        };
        assert_eq!(
            core_error("call", "input", tool),
            DecodeError::HevcTool {
                decoder: DECODER,
                field: "chroma_format_idc",
                value: 3,
                supported: "1, which is 4:2:0",
            }
        );
        let missing = HevcError::MissingReference { poc: 7 };
        let DecodeError::Software {
            call,
            input,
            reason,
            ..
        } = core_error("call", "input", missing)
        else {
            panic!("a lost reference is a software error");
        };
        assert_eq!((call, input.as_str()), ("call", "input"));
        assert!(reason.contains("POC 7"));
        assert!(!reason.ends_with('.'));
    }

    #[test]
    fn the_cycle_refuses_what_the_trait_states() {
        let mut decoder = HevcDecoder::<CpuRuntime>::new(&CpuDevice::new(), &[]).unwrap();
        assert_eq!(decoder.route(), FrameRoute::HostOnly);
        assert_eq!(
            decoder.send(&[], 3),
            Err(DecodeError::EmptyPacket { pts: 3 })
        );
        assert_eq!(decoder.flush(), Ok(()));
        assert_eq!(decoder.flush(), Ok(()));
        assert_eq!(
            decoder.send(&[0, 0, 1, 0x40, 0x01], 4),
            Err(DecodeError::AfterFlush { pts: 4 })
        );
        assert!(decoder.receive().is_none());
    }

    /// TRAIL_R, first_slice_segment_in_pic_flag 1, slice_pic_parameter_set_id 0.
    const SLICE_WITH_NO_PPS: [u8; 7] = [0, 0, 1, 0x02, 0x01, 0xc0, 0x00];

    #[test]
    fn the_first_error_stays() {
        let mut decoder = HevcDecoder::<CpuRuntime>::new(&CpuDevice::new(), &[]).unwrap();
        let first = decoder.send(&SLICE_WITH_NO_PPS, 0).unwrap_err();
        assert!(first.to_string().contains("PPS 0"), "{first}");
        assert_eq!(decoder.send(&SLICE_WITH_NO_PPS, 1), Err(first.clone()));
        assert_eq!(decoder.flush(), Err(first));
    }

    #[test]
    fn data_with_no_start_code_leaves_the_decoder_usable() {
        let mut decoder = HevcDecoder::<CpuRuntime>::new(&CpuDevice::new(), &[]).unwrap();
        assert!(matches!(
            decoder.send(&[0x40, 0x01], 0),
            Err(DecodeError::Bitstream(_))
        ));
        assert_eq!(decoder.flush(), Ok(()));
    }

    /// A packet of many pictures stops at its limit, before the frames pile up.
    #[test]
    fn a_packet_past_its_luma_limit_is_a_named_error() {
        let mut decoder = HevcDecoder::<CpuRuntime>::new(&CpuDevice::new(), &[]).unwrap();
        // Two output pictures of 16x16 fit, and a third does not.
        decoder.send_luma_limit = 2 * 16 * 16;
        let stream = stream_tests::annexb_of(&stream_tests::intra_pictures(4));
        let error = decoder.send(&stream, 9).unwrap_err();
        let DecodeError::Software { call, input, .. } = &error else {
            panic!("a packet over its limit is a software error: {error}");
        };
        assert_eq!(*call, "HevcDecoder::send");
        assert_eq!(input, "the packet at pts 9");
        assert_eq!(decoder.queued(), 2);
    }

    #[test]
    fn a_packet_under_its_luma_limit_decodes() {
        let mut decoder = HevcDecoder::<CpuRuntime>::new(&CpuDevice::new(), &[]).unwrap();
        let stream = stream_tests::annexb_of(&stream_tests::intra_pictures(4));
        assert_eq!(decoder.send(&stream, 9), Ok(()));
        assert_eq!(decoder.flush(), Ok(()));
        assert_eq!(decoder.queued(), 4);
    }

    #[test]
    fn the_decoder_moves_between_threads() {
        const fn assert_send<T: Send>() {}
        assert_send::<HevcDecoder<CpuRuntime>>();
    }

    #[test]
    fn a_thread_count_of_zero_is_a_named_error() {
        let opened = HevcDecoder::<CpuRuntime>::with_threads(&CpuDevice::new(), &[], 0);
        let Err(DecodeError::Software { input, .. }) = opened else {
            panic!("a thread count of 0 opens no decoder");
        };
        assert_eq!(input, "a thread count of 0");
    }

    /// A stored error also outranks the refusals of a later call.
    #[test]
    fn a_stored_error_stays_after_a_flush() {
        let mut decoder =
            HevcDecoder::<CpuRuntime>::with_threads(&CpuDevice::new(), &[], 1).unwrap();
        let first = decoder.send(&SLICE_WITH_NO_PPS, 0).unwrap_err();
        assert_eq!(decoder.flush(), Err(first.clone()));
        assert_eq!(decoder.flush(), Err(first.clone()));
        assert_eq!(decoder.send(&SLICE_WITH_NO_PPS, 1), Err(first));
    }
}
