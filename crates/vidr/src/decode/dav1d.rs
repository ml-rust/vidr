//! AV1 software decode on libdav1d, into numr tensors through host memory.
//!
//! `libdav1d` 1.3.0 or newer links through pkg-config. Each frame takes one copy
//! through host memory, and [`Decoder::route`] declares it.

use std::collections::VecDeque;

use ::dav1d::{
    Decoder as Library, Error as LibraryError, PixelLayout, PlanarImageComponent, Settings,
};
use numr::dtype::DType;
use numr::runtime::Runtime;

use super::av1_frame::{check_facts, check_layout, color_facts};
use super::{ChromaFormat, ColorFacts, DecodeCodec, DecodedFrame, Decoder, PixelFormat};
use crate::bitstream::av1::seq_header::bit_depth;
use crate::bitstream::av1::{Av1Config, Av1ToDecoder, ObuType, av1_sequence_facts, obu};
use crate::error::DecodeError;
use crate::frame::{HostPlanes, upload_planes, upload_route};
use crate::geometry::Dimensions;
use crate::route::FrameRoute;

const DECODER: &str = "dav1d";

/// A decoded picture, on the library's own allocator.
type Picture = ::dav1d::Picture;

/// AV1 software decode into numr tensors on any backend.
///
/// # Input
///
/// `config` is the track's `av1C` payload. With it, `send` takes container
/// samples and converts them with [`Av1ToDecoder`]. With an empty `config`,
/// `send` takes AV1 temporal units.
///
/// # Output
///
/// 4:2:0 at 8 or 10 bits, as NV12 or P010 in the form the backend's colour
/// kernels take. Film grain applies. The display size comes from each picture,
/// and odd sizes work. Colour comes from the sequence header in effect when
/// the picture's packet went in, so a later sequence never relabels a held frame.
///
/// # Refusals
///
/// 12 bits, 4:0:0, 4:2:2, and 4:4:4 return [`DecodeError::UnsupportedLayout`]
/// from `new` when `av1C` states them, and from `send` for the sequence header
/// that states them.
///
/// # Errors after the first
///
/// After a decode error, every later `send` and `flush` returns the same error.
/// A packet the bitstream filter refuses leaves the decoder usable.
pub struct Dav1dDecoder<R: Runtime> {
    device: R::Device,
    library: Library,
    filter: Option<Av1ToDecoder>,
    /// The colour of each sequence so far. A packet's offset names its entry.
    sequences: Vec<ColorFacts>,
    frames: VecDeque<DecodedFrame<R>>,
    flushed: bool,
    error: Option<DecodeError>,
}

impl<R> Dav1dDecoder<R>
where
    R: Runtime<DType = DType>,
{
    /// Opens a decoder whose frames go to tensors on `device`.
    ///
    /// `config` is the track's `av1C` payload, or empty for raw temporal units.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::Bitstream`] for a config record that does not parse.
    /// - [`DecodeError::UnsupportedLayout`] for a layout `av1C` states and vidr
    ///   has no frame for.
    /// - [`DecodeError::Software`] when libdav1d does not open.
    pub fn new(device: &R::Device, config: &[u8]) -> Result<Self, DecodeError> {
        let mut sequences = Vec::new();
        let filter = if config.is_empty() {
            None
        } else {
            let parsed = Av1Config::parse(config)?;
            check_layout(
                bit_depth(parsed.high_bitdepth, parsed.twelve_bit),
                parsed.monochrome,
                parsed.chroma_subsampling_x,
                parsed.chroma_subsampling_y,
            )?;
            if let Some(header) = parsed.sequence_header()? {
                let facts = av1_sequence_facts(header.payload)?;
                check_facts(&facts)?;
                sequences.push(color_facts(&facts));
            }
            Some(Av1ToDecoder::new(&parsed)?)
        };

        let mut settings = Settings::new();
        settings.set_apply_grain(true);
        // One frame per temporal unit, the highest spatial layer. ffmpeg's
        // libdav1d wrapper defaults to the same.
        settings.set_all_layers(false);
        let library = Library::with_settings(&settings)
            .map_err(|error| software("dav1d_open", "the default settings".to_owned(), error))?;
        Ok(Self {
            device: device.clone(),
            library,
            filter,
            sequences,
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

    /// Reads every sequence header of `unit`, in stream order.
    fn read_sequences(&mut self, unit: &[u8]) -> Result<(), DecodeError> {
        for parsed in obu::split(unit) {
            let parsed = parsed?;
            if parsed.header.obu_type != ObuType::SEQUENCE_HEADER {
                continue;
            }
            let facts = av1_sequence_facts(parsed.payload)?;
            check_facts(&facts)?;
            let color = color_facts(&facts);
            if self.sequences.last() != Some(&color) {
                self.sequences.push(color);
            }
        }
        Ok(())
    }

    fn decode(&mut self, unit: Vec<u8>, pts: i64) -> Result<(), DecodeError> {
        let size = unit.len();
        let input = || format!("the {size}-byte packet at pts {pts}");
        let sequence = self.sequences.len().saturating_sub(1) as i64;
        match self
            .library
            .send_data(unit, Some(sequence), Some(pts), None)
        {
            Ok(()) => {}
            Err(LibraryError::Again) => loop {
                // libdav1d holds a finished picture and the rest of the data.
                self.take_picture()?;
                match self.library.send_pending_data() {
                    Ok(()) => break,
                    Err(LibraryError::Again) => {}
                    Err(error) => return Err(software("dav1d_send_data", input(), error)),
                }
            },
            Err(error) => return Err(software("dav1d_send_data", input(), error)),
        }
        // One call per packet keeps frame threading at work. A loop here turns
        // into a drain, which waits for every frame thread.
        self.take_picture()?;
        Ok(())
    }

    /// Queues the next picture, and returns whether one exists.
    fn take_picture(&mut self) -> Result<bool, DecodeError> {
        match self.library.get_picture() {
            Ok(picture) => {
                let frame = to_frame::<R>(&self.device, &self.sequences, &picture)?;
                self.frames.push_back(frame);
                Ok(true)
            }
            Err(LibraryError::Again) => Ok(false),
            Err(error) => Err(software(
                "dav1d_get_picture",
                "the next picture".to_owned(),
                error,
            )),
        }
    }

    /// Takes every delayed picture. The wrapper's own `flush` calls `dav1d_flush`,
    /// which discards them, so this never calls it. Every `send` ends in
    /// `get_picture`, so here libdav1d drains its frame threads.
    fn drain(&mut self) -> Result<(), DecodeError> {
        while self.take_picture()? {}
        Ok(())
    }
}

fn software(call: &'static str, input: String, error: LibraryError) -> DecodeError {
    DecodeError::Software {
        decoder: DECODER,
        call,
        input,
        reason: error.to_string(),
    }
}

/// The picture's sample layout, or the refusal for one with no vidr frame.
fn picture_format(picture: &Picture) -> Result<PixelFormat, DecodeError> {
    let bit_depth = picture
        .bits_per_component()
        .map_or(0, |bits| u8::try_from(bits.0).unwrap_or(u8::MAX));
    let chroma = match picture.pixel_layout() {
        PixelLayout::I400 => ChromaFormat::Monochrome,
        PixelLayout::I420 => ChromaFormat::Yuv420,
        PixelLayout::I422 => ChromaFormat::Yuv422,
        PixelLayout::I444 => ChromaFormat::Yuv444,
    };
    match (chroma, PixelFormat::for_bit_depth(bit_depth)) {
        (ChromaFormat::Yuv420, Some(format)) => Ok(format),
        _ => Err(DecodeError::UnsupportedLayout {
            codec: DecodeCodec::Av1,
            bit_depth_luma: bit_depth,
            bit_depth_chroma: bit_depth,
            chroma,
        }),
    }
}

/// Copies one picture to tensors on `device`, with its pts and colour.
fn to_frame<R>(
    device: &R::Device,
    sequences: &[ColorFacts],
    picture: &Picture,
) -> Result<DecodedFrame<R>, DecodeError>
where
    R: Runtime<DType = DType>,
{
    let format = picture_format(picture)?;
    let color = usize::try_from(picture.offset())
        .ok()
        .and_then(|index| sequences.get(index))
        .copied()
        .ok_or(DecodeError::NoSequence {
            callback: "dav1d_get_picture",
        })?;
    let (width, height) = (picture.width(), picture.height());
    let display = Dimensions::new(width, height).map_err(|_| DecodeError::DisplayArea {
        rect: (0, 0, width as i32, height as i32),
        coded: (width, height),
    })?;
    let (y, u, v) = (
        picture.plane(PlanarImageComponent::Y),
        picture.plane(PlanarImageComponent::U),
        picture.plane(PlanarImageComponent::V),
    );
    let planes = HostPlanes {
        y: &y,
        y_stride: picture.stride(PlanarImageComponent::Y) as usize,
        u: &u,
        u_stride: picture.stride(PlanarImageComponent::U) as usize,
        v: &v,
        v_stride: picture.stride(PlanarImageComponent::V) as usize,
        width: width as usize,
        height: height as usize,
    };
    let (luma, chroma) = upload_planes::<R>(device, planes, format)?;
    // The wrapper reads back an unset timestamp as `None`. Every packet sets one,
    // so `None` means the packet's pts was the sentinel itself.
    let pts = picture.timestamp().unwrap_or(i64::MIN);
    Ok(if R::packed_color_planes() {
        DecodedFrame::from_packed_planes(luma, chroma, format, pts, display, color)
    } else {
        DecodedFrame::from_planes(luma, chroma, format, pts, display, color)
    })
}

impl<R> Decoder<R> for Dav1dDecoder<R>
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
        let unit = match &mut self.filter {
            Some(filter) => filter.filter(packet)?,
            None => packet.to_vec(),
        };
        let mut result = self.read_sequences(&unit);
        if result.is_ok() {
            result = self.decode(unit, pts);
        }
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
