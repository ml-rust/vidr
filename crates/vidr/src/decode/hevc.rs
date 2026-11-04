//! HEVC software decode on this crate's own core, into numr tensors through host
//! memory.
//!
//! Each frame takes one copy to the tensors, and [`Decoder::route`] declares it. A
//! 10-bit picture first copies once on the host, into little-endian bytes. HEVC has
//! patent pools: the README states what a binary distribution needs.

use std::collections::{BTreeMap, VecDeque};

use numr::dtype::DType;
use numr::runtime::Runtime;

use super::cpu::dsp::sample::Sample;
use super::cpu::hevc::dpb::buffer::Frame;
use super::cpu::hevc::error::HevcError;
use super::cpu::hevc::params::MAX_LUMA_PICTURE_SIZE;
use super::cpu::hevc::picture::HostScratch;
use super::cpu::hevc::stream::{DecodeOptions, DecodedPicture, StreamDecoder};
use super::{ChromaFormat, ColorFacts, DecodeCodec, DecodedFrame, Decoder, PixelFormat, vui_color};
use crate::bitstream::hevc_sps_full::SeqParameterSet;
use crate::bitstream::{HevcConfig, HevcNalType, LengthSize, annexb, length_prefixed};
use crate::error::DecodeError;
use crate::frame::{upload_planes, upload_route};
use crate::geometry::Dimensions;
use crate::route::FrameRoute;

const DECODER: &str = "HevcDecoder";
/// The luma samples of the frames one `send` can output: 16 pictures, the largest
/// `MaxDpbSize` (A-2), at `MaxLumaPs` of level 6.2. One access unit stays below it.
const MAX_SEND_LUMA_SAMPLES: u64 = 16 * MAX_LUMA_PICTURE_SIZE;

/// What a picture's packet stated when the picture started decoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingPicture {
    pts: i64,
    color: ColorFacts,
}

/// HEVC software decode into numr tensors on any backend.
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
///
/// A caller holds the frame queue to its own size: call [`Decoder::receive`] until
/// it returns `None` after each `send`.
///
/// # Errors after the first
///
/// After a decode error or a refusal, every later `send` and `flush` returns the same
/// error. A packet whose NAL units do not split leaves the decoder usable.
pub struct HevcDecoder<R: Runtime> {
    device: R::Device,
    stream: StreamDecoder,
    /// The size of each sample's length fields. `None` for Annex-B input.
    length_size: Option<LengthSize>,
    /// Pictures in the core that give a frame later, by [`Frame::decode_order`].
    pending: BTreeMap<u64, PendingPicture>,
    scratch: HostScratch,
    frames: VecDeque<DecodedFrame<R>>,
    /// The pts of the packet in `send`.
    send_pts: i64,
    /// The luma samples of the frames the packet in `send` has output.
    send_luma: u64,
    /// The largest `send_luma`.
    send_luma_limit: u64,
    flushed: bool,
    error: Option<DecodeError>,
}

impl<R> HevcDecoder<R>
where
    R: Runtime<DType = DType>,
{
    /// Opens a decoder whose frames go to tensors on `device`.
    ///
    /// `config` is the track's `hvcC` payload, or empty for raw Annex-B input.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::Bitstream`] for a config record or parameter set that does
    ///   not parse.
    /// - [`DecodeError::UnsupportedLayout`] for a layout an SPS in `hvcC` states.
    pub fn new(device: &R::Device, config: &[u8]) -> Result<Self, DecodeError> {
        Self::open(device, config, DecodeOptions::default())
    }

    /// As [`Self::new`], and also compares each picture with its decoded picture hash
    /// SEI message (D.3.20). A difference is a decode error. A picture with no such
    /// message passes unchecked.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn with_picture_hash_check(device: &R::Device, config: &[u8]) -> Result<Self, DecodeError> {
        let options = DecodeOptions {
            check_picture_hash: true,
        };
        Self::open(device, config, options)
    }

    fn open(
        device: &R::Device,
        config: &[u8],
        options: DecodeOptions,
    ) -> Result<Self, DecodeError> {
        let mut stream = StreamDecoder::new(options);
        let mut length_size = None;
        if !config.is_empty() {
            let parsed = HevcConfig::parse(config)?;
            length_size = Some(parsed.length_size);
            for nal in parsed.nal_units() {
                check_sps(nal)?;
                stream.decode_nal(nal).map_err(|error| {
                    core_error("StreamDecoder::decode_nal", "the hvcC record", error)
                })?;
            }
        }
        Ok(Self {
            device: device.clone(),
            stream,
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

    fn keep_error(&mut self, result: Result<(), DecodeError>) -> Result<(), DecodeError> {
        if let Err(error) = &result {
            self.error = Some(error.clone());
        }
        result
    }

    fn decode_units(&mut self, units: &[&[u8]], pts: i64) -> Result<(), DecodeError> {
        for nal in units {
            self.decode_unit(nal, pts)?;
        }
        Ok(())
    }

    /// Decodes one NAL unit, and queues every picture it completes.
    fn decode_unit(&mut self, nal: &[u8], pts: i64) -> Result<(), DecodeError> {
        check_sps(nal)?;
        let result = self.stream.decode_nal(nal);
        let started = self.stream.take_started();
        if let Some(picture) = started.filter(|picture| picture.output) {
            let pending = PendingPicture {
                pts,
                color: vui_color(&picture.signal),
            };
            self.pending.insert(picture.decode_order, pending);
        }
        self.drain()?;
        // The core has output or dropped each earlier picture, and the drain took the
        // output ones. So an earlier entry left is a dropped picture.
        if let Some(picture) = started.filter(|picture| picture.irap_no_rasl) {
            self.pending = self.pending.split_off(&picture.decode_order);
        }
        result.map_err(|error| {
            let input = format!("the {}-byte NAL unit at pts {pts}", nal.len());
            core_error("StreamDecoder::decode_nal", &input, error)
        })
    }

    /// Moves every output picture of the core to the frame queue.
    fn drain(&mut self) -> Result<(), DecodeError> {
        while let Some(picture) = self.stream.next_output() {
            self.charge_send(&picture)?;
            let frame = match &picture {
                DecodedPicture::Eight(frame) => self.convert(frame, PixelFormat::Nv12)?,
                DecodedPicture::Ten(frame) => self.convert(frame, PixelFormat::P010)?,
            };
            self.frames.push_back(frame);
        }
        Ok(())
    }

    /// Counts one output picture against the packet in `send`, before its upload.
    fn charge_send(&mut self, picture: &DecodedPicture) -> Result<(), DecodeError> {
        let window = match picture {
            DecodedPicture::Eight(frame) => frame.picture.window(),
            DecodedPicture::Ten(frame) => frame.picture.window(),
        };
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
        let ended = self.stream.flush();
        self.drain()?;
        ended
            .map_err(|error| core_error("StreamDecoder::flush", "the end of the stream", error))?;
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
fn check_sps(nal: &[u8]) -> Result<(), DecodeError> {
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
fn core_error(call: &'static str, input: &str, error: HevcError) -> DecodeError {
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

/// Copies the conformance window of one picture to tensors on `device`.
fn to_frame<R, S>(
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
        if self.flushed {
            return Err(DecodeError::AfterFlush { pts });
        }
        if packet.is_empty() {
            return Err(DecodeError::EmptyPacket { pts });
        }
        self.stored_error()?;
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
        if self.flushed {
            return Ok(());
        }
        self.flushed = true;
        self.stored_error()?;
        self.send_luma = 0;
        let result = self.finish();
        self.keep_error(result)
    }

    fn receive(&mut self) -> Option<DecodedFrame<R>> {
        self.frames.pop_front()
    }

    fn route(&self) -> FrameRoute {
        upload_route::<R>()
    }
}

#[cfg(test)]
mod tests {
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    use super::super::cpu::hevc::stream_tests;
    use super::*;

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
}
