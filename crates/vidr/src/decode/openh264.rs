//! H.264 Baseline software decode on OpenH264, into numr tensors through host
//! memory. Each frame takes one copy through host memory, and
//! [`Decoder::route`] declares it.
//!
//! # Licence position
//!
//! OpenH264's code is BSD-2-Clause. H.264 patents are a separate matter:
//!
//! - The `openh264` feature compiles OpenH264 from source. A source build gets no
//!   Cisco patent cover. Whoever ships it answers for the H.264 patent licence.
//! - The `openh264-cisco` feature loads Cisco's binary from a path the caller
//!   supplies, after the `openh264` crate checks its hash. Cisco's cover applies
//!   only to its own binary. Cisco's FAQ requires the binary to download when the
//!   product installs on the user's device. vidr holds no download code.
//!
//! # Baseline only
//!
//! The upstream README states Constrained Baseline. Its parser also accepts Main,
//! Extended, and High, and its B-slice output differs from the reference decode.
//! So vidr passes it `profile_idc` 66 only, and refuses the rest by name.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use ::openh264::OpenH264API;
use ::openh264::decoder::{DecodedYUV, Decoder as Library, DecoderConfig, Flush};
use ::openh264::formats::YUVSource;
use numr::dtype::DType;
use numr::runtime::Runtime;

use super::{
    ChromaFormat, ColorFacts, DecodeCodec, DecodedFrame, Decoder, PixelFormat, unspecified_color,
    vui_color,
};
use crate::bitstream::{
    AvcConfig, H264NalType, H264SpsFacts, ToAnnexB, annexb, h264_sps_facts, h264_sps_vui,
};
use crate::error::DecodeError;
use crate::frame::{HostPlanes, upload_planes, upload_route};
use crate::geometry::Dimensions;
use crate::route::FrameRoute;

const DECODER: &str = "OpenH264";
/// `profile_idc` of Baseline.
const PROFILE_BASELINE: u8 = 66;

/// The pts of pictures the library holds.
///
/// The wrapper's own timestamp field carries an open question upstream, so this
/// queue supplies the pts. Frames leave in display order, so the lowest pts
/// waiting belongs to the next frame. A heap keeps a repeated pts.
#[derive(Debug, Default)]
struct DisplayPts(BinaryHeap<Reverse<i64>>);

impl DisplayPts {
    fn push(&mut self, pts: i64) {
        self.0.push(Reverse(pts));
    }

    fn pop(&mut self) -> Result<i64, DecodeError> {
        self.0
            .pop()
            .map(|Reverse(pts)| pts)
            .ok_or(DecodeError::UnmatchedPicture { decoder: DECODER })
    }

    /// Refuses a stream end with pictures that never came out.
    fn check_empty(&self) -> Result<(), DecodeError> {
        match self.0.iter().map(|Reverse(pts)| *pts).min() {
            None => Ok(()),
            Some(first_pts) => Err(DecodeError::MissingPictures {
                decoder: DECODER,
                count: self.0.len(),
                first_pts,
            }),
        }
    }
}

/// H.264 Baseline software decode into numr tensors on any backend.
///
/// # Input
///
/// `config` is the track's `avcC` payload. With it, `send` takes container
/// samples and converts them with [`ToAnnexB`]. With an empty `config`, `send`
/// takes Annex-B access units. Each packet holds one access unit.
///
/// # Output
///
/// NV12, in the form the backend's colour kernels take. Colour comes from the VUI
/// of the SPS in effect. The display size comes from each picture.
///
/// # Refusals
///
/// An SPS that fails these checks stops the stream before the C++ decoder reads
/// it. `new` checks the SPS of `avcC`, and `send` checks each SPS in a packet.
///
/// | Stream | Error |
/// | --- | --- |
/// | Above 8 bits | [`DecodeError::H264BitDepth`] |
/// | 4:0:0, 4:2:2, 4:4:4 | [`DecodeError::UnsupportedLayout`] |
/// | `frame_mbs_only_flag` 0 | [`DecodeError::H264Interlaced`] |
/// | `profile_idc` other than 66 | [`DecodeError::H264Profile`] |
///
/// # Errors after the first
///
/// After a decode error or a refused SPS, every later `send` and `flush` returns
/// the same error. A packet the bitstream filter refuses leaves the decoder usable.
pub struct Openh264Decoder<R: Runtime> {
    device: R::Device,
    library: Library,
    filter: Option<ToAnnexB>,
    scratch: Vec<u8>,
    color: ColorFacts,
    pending: DisplayPts,
    frames: VecDeque<DecodedFrame<R>>,
    flushed: bool,
    error: Option<DecodeError>,
}

impl<R> Openh264Decoder<R>
where
    R: Runtime<DType = DType>,
{
    /// Opens a decoder on OpenH264 compiled from source. See the module
    /// documentation for the licence position.
    ///
    /// `config` is the track's `avcC` payload, or empty for raw Annex-B input.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Bitstream`] for a config record that does not parse, a
    /// refusal from the type documentation, and [`DecodeError::Software`] when
    /// OpenH264 does not open.
    #[cfg(feature = "openh264")]
    pub fn new(device: &R::Device, config: &[u8]) -> Result<Self, DecodeError> {
        Self::open(device, config, OpenH264API::from_source())
    }

    /// Opens a decoder on Cisco's binary at `library`. The `openh264` crate reads
    /// the file and checks its hash against the list it holds. See the module
    /// documentation for the licence position.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Openh264Library`] for a binary that does not load or a hash
    /// not on the list, and the errors of `new`.
    #[cfg(feature = "openh264-cisco")]
    pub fn with_cisco_binary(
        device: &R::Device,
        config: &[u8],
        library: impl AsRef<std::path::Path>,
    ) -> Result<Self, DecodeError> {
        let path = library.as_ref();
        let api =
            OpenH264API::from_blob_path(path).map_err(|error| DecodeError::Openh264Library {
                path: path.display().to_string(),
                reason: error.to_string(),
            })?;
        Self::open(device, config, api)
    }

    fn open(device: &R::Device, config: &[u8], api: OpenH264API) -> Result<Self, DecodeError> {
        let mut color = unspecified_color();
        let filter = if config.is_empty() {
            None
        } else {
            let parsed = AvcConfig::parse(config)?;
            for sps in &parsed.sps {
                check_sps(&h264_sps_facts(sps)?)?;
            }
            if let Some(first) = parsed.sps.first() {
                color = vui_color(&h264_sps_vui(first)?);
            }
            Some(ToAnnexB::h264(&parsed))
        };
        // The wrapper's default flushes a frame after a decode with no output,
        // which breaks display order. `flush` drains the rest at the stream end.
        let settings = DecoderConfig::new().flush_after_decode(Flush::NoFlush);
        let library = Library::with_api_config(api, settings).map_err(|error| {
            software("WelsCreateDecoder", "the default config".to_owned(), &error)
        })?;
        Ok(Self {
            device: device.clone(),
            library,
            filter,
            scratch: Vec::new(),
            color,
            pending: DisplayPts::default(),
            frames: VecDeque::new(),
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

    /// Checks every SPS of `unit`, and returns whether `unit` holds a picture.
    fn read_parameter_sets(&mut self, unit: &[u8]) -> Result<bool, DecodeError> {
        let mut holds_picture = false;
        for nal in annexb::split(unit)? {
            let Some(kind) = H264NalType::of(nal) else {
                continue;
            };
            holds_picture |= kind.is_vcl();
            if kind == H264NalType::SPS {
                check_sps(&h264_sps_facts(nal)?)?;
                self.color = vui_color(&h264_sps_vui(nal)?);
            }
        }
        Ok(holds_picture)
    }

    fn decode(&mut self, unit: &[u8], pts: i64) -> Result<(), DecodeError> {
        if self.read_parameter_sets(unit)? {
            self.pending.push(pts);
        }
        let picture = self.library.decode(unit).map_err(|error| {
            let input = format!("the {}-byte packet at pts {pts}", unit.len());
            software("DecodeFrameNoDelay", input, &error)
        })?;
        if let Some(picture) = picture {
            let pts = self.pending.pop()?;
            let frame = to_frame::<R>(&self.device, &picture, pts, self.color)?;
            self.frames.push_back(frame);
        }
        Ok(())
    }

    fn drain(&mut self) -> Result<(), DecodeError> {
        let pictures = self
            .library
            .flush_remaining()
            .map_err(|error| software("FlushFrame", "the end of the stream".to_owned(), &error))?;
        for picture in &pictures {
            let pts = self.pending.pop()?;
            let frame = to_frame::<R>(&self.device, picture, pts, self.color)?;
            self.frames.push_back(frame);
        }
        self.pending.check_empty()
    }
}

/// Refuses an SPS by name, most specific first.
fn check_sps(sps: &H264SpsFacts) -> Result<(), DecodeError> {
    if sps.bit_depth_luma != 8 || sps.bit_depth_chroma != 8 {
        return Err(DecodeError::H264BitDepth {
            decoder: DECODER,
            luma: sps.bit_depth_luma,
            chroma: sps.bit_depth_chroma,
        });
    }
    let chroma = match sps.chroma_format_idc {
        0 => Some(ChromaFormat::Monochrome),
        2 => Some(ChromaFormat::Yuv422),
        3 => Some(ChromaFormat::Yuv444),
        _ => None,
    };
    if let Some(chroma) = chroma {
        return Err(DecodeError::UnsupportedLayout {
            codec: DecodeCodec::H264,
            bit_depth_luma: sps.bit_depth_luma,
            bit_depth_chroma: sps.bit_depth_chroma,
            chroma,
        });
    }
    if !sps.frame_mbs_only {
        return Err(DecodeError::H264Interlaced { decoder: DECODER });
    }
    if sps.profile_idc != PROFILE_BASELINE {
        return Err(DecodeError::H264Profile {
            decoder: DECODER,
            profile_idc: sps.profile_idc,
        });
    }
    Ok(())
}

fn software(call: &'static str, input: String, error: &::openh264::Error) -> DecodeError {
    DecodeError::Software {
        decoder: DECODER,
        call,
        input,
        reason: error.to_string(),
    }
}

/// Copies one picture to tensors on `device`.
fn to_frame<R>(
    device: &R::Device,
    picture: &DecodedYUV<'_>,
    pts: i64,
    color: ColorFacts,
) -> Result<DecodedFrame<R>, DecodeError>
where
    R: Runtime<DType = DType>,
{
    let (width, height) = picture.dimensions();
    let (y_stride, u_stride, v_stride) = picture.strides();
    let invalid = || DecodeError::DisplayArea {
        rect: (0, 0, width as i32, height as i32),
        coded: (width as u32, height as u32),
    };
    let display = u32::try_from(width)
        .ok()
        .zip(u32::try_from(height).ok())
        .and_then(|(width, height)| Dimensions::new(width, height).ok())
        .ok_or_else(invalid)?;
    let planes = HostPlanes {
        y: picture.y(),
        y_stride,
        u: picture.u(),
        u_stride,
        v: picture.v(),
        v_stride,
        width,
        height,
    };
    let format = PixelFormat::Nv12;
    let (luma, chroma) = upload_planes::<R>(device, planes, format)?;
    Ok(if R::packed_color_planes() {
        DecodedFrame::from_packed_planes(luma, chroma, format, pts, display, color)
    } else {
        DecodedFrame::from_planes(luma, chroma, format, pts, display, color)
    })
}

impl<R> Decoder<R> for Openh264Decoder<R>
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
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.clear();
        let unit: &[u8] = match &mut self.filter {
            Some(filter) => {
                let filtered = filter.filter_into(packet, &mut scratch);
                if let Err(error) = filtered {
                    self.scratch = scratch;
                    return Err(error.into());
                }
                &scratch
            }
            None => packet,
        };
        let result = self.decode(unit, pts);
        self.scratch = scratch;
        self.keep_error(result)
    }

    fn flush(&mut self) -> Result<(), DecodeError> {
        if self.flushed {
            return Ok(());
        }
        self.flushed = true;
        self.stored_error()?;
        let result = self.drain();
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
    use super::*;

    const fn baseline() -> H264SpsFacts {
        H264SpsFacts {
            profile_idc: PROFILE_BASELINE,
            constraint_flags: 0xc0,
            chroma_format_idc: 1,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            frame_mbs_only: true,
        }
    }

    #[test]
    fn baseline_passes() {
        assert_eq!(check_sps(&baseline()), Ok(()));
    }

    /// The specific fact names the refusal, whatever the profile says.
    #[test]
    fn each_refusal_has_its_name() {
        let mut high_10 = baseline();
        high_10.profile_idc = 110;
        high_10.bit_depth_luma = 10;
        high_10.bit_depth_chroma = 10;
        assert_eq!(
            check_sps(&high_10),
            Err(DecodeError::H264BitDepth {
                decoder: DECODER,
                luma: 10,
                chroma: 10,
            })
        );

        let mut monochrome = baseline();
        monochrome.profile_idc = 100;
        monochrome.chroma_format_idc = 0;
        assert_eq!(
            check_sps(&monochrome),
            Err(DecodeError::UnsupportedLayout {
                codec: DecodeCodec::H264,
                bit_depth_luma: 8,
                bit_depth_chroma: 8,
                chroma: ChromaFormat::Monochrome,
            })
        );

        let mut interlaced = baseline();
        interlaced.profile_idc = 100;
        interlaced.frame_mbs_only = false;
        assert_eq!(
            check_sps(&interlaced),
            Err(DecodeError::H264Interlaced { decoder: DECODER })
        );

        for profile_idc in [77, 88, 100] {
            let mut other = baseline();
            other.profile_idc = profile_idc;
            assert_eq!(
                check_sps(&other),
                Err(DecodeError::H264Profile {
                    decoder: DECODER,
                    profile_idc,
                })
            );
        }
    }

    /// The pop order for B-frames in decode order: each frame takes the lowest
    /// pts sent so far, which yields display order.
    #[test]
    fn pts_leave_lowest_first() {
        let mut pending = DisplayPts::default();
        let mut shown = Vec::new();
        pending.push(0);
        pending.push(1536);
        shown.push(pending.pop());
        pending.push(512);
        shown.push(pending.pop());
        pending.push(1024);
        shown.push(pending.pop());
        shown.push(pending.pop());
        assert_eq!(shown, [Ok(0), Ok(512), Ok(1024), Ok(1536)]);
        assert_eq!(pending.check_empty(), Ok(()));
    }

    #[test]
    fn a_repeated_pts_stays_in_the_queue() {
        let mut pending = DisplayPts::default();
        pending.push(7);
        pending.push(7);
        assert_eq!(pending.pop(), Ok(7));
        assert_eq!(pending.pop(), Ok(7));
    }

    #[test]
    fn a_picture_with_no_packet_is_an_error() {
        let mut pending = DisplayPts::default();
        assert_eq!(
            pending.pop(),
            Err(DecodeError::UnmatchedPicture { decoder: DECODER })
        );
    }

    #[test]
    fn a_packet_with_no_frame_is_an_error_at_the_stream_end() {
        let mut pending = DisplayPts::default();
        pending.push(30);
        pending.push(20);
        assert_eq!(
            pending.check_empty(),
            Err(DecodeError::MissingPictures {
                decoder: DECODER,
                count: 2,
                first_pts: 20,
            })
        );
    }
}
