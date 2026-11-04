//! VP9 software decode on the `rusty_vp9` crate, into numr tensors through host
//! memory.
//!
//! Each frame takes one copy through host memory, and [`Decoder::route`] declares
//! it. The crate also crops each picture into its own buffers, so a frame moves
//! twice on the host. No crate type crosses this module's public API, so another
//! VP9 decoder can replace the crate here.

use std::collections::VecDeque;

use ::rusty_vp9::{DecodedFrame as Picture, Error as LibraryError, Vp9Decoder as Library};
use numr::dtype::DType;
use numr::runtime::Runtime;

use super::vp9_frame::{VP9_UNSTATED, check_layout, vp9_color_facts};
use super::{ColorFacts, DecodedFrame, Decoder, PixelFormat};
use crate::bitstream::{
    NUM_REF_FRAMES, RefFrameTable, UncompressedHeader, Vp9HeaderParser, split_superframe,
};
use crate::error::DecodeError;
use crate::frame::{HostPlanes, upload_planes, upload_route};
use crate::geometry::Dimensions;
use crate::route::FrameRoute;

const DECODER: &str = "rusty_vp9";

/// VP9 software decode into numr tensors on any backend.
///
/// # Input
///
/// `send` takes one chunk: one frame, or a superframe. VP9 needs no config
/// record, so a track's `vpcC` stays out of this type.
///
/// # Output
///
/// 4:2:0 at 8 or 10 bits, as NV12 or P010 in the form the backend's colour
/// kernels take. The display size is each frame's coded size, as libvpx reports
/// it, so a stream that resizes changes size mid-stream, and odd sizes work.
///
/// A hidden frame gives no output. A chunk gives at most one frame: its last shown
/// frame, as libvpx does. So the lower spatial layers of a superframe give none, and a
/// superframe with two shown frames keeps one frame per packet and each packet's timing.
/// The frame carries the chunk's pts. `show_existing_frame` repeats a slot, with
/// the colour that slot's frame stated.
///
/// Colour comes from this crate's header parser. A VP9 header states the matrix
/// and the range, so transfer and primaries stay unspecified for the container
/// to fill. The chroma location stays `None`.
///
/// # Refusals
///
/// Profiles 1 and 3, 12 bits, and anything but 4:2:0 return
/// [`DecodeError::Vp9Layout`] from `send`, before the crate reads the frame.
///
/// # Errors after the first
///
/// After a decode error or refusal, every later `send` and `flush` returns the
/// same error. A chunk whose superframe index does not parse leaves the decoder
/// usable.
pub struct RustyVp9Decoder<R: Runtime> {
    device: R::Device,
    library: Library,
    parser: Vp9HeaderParser,
    refs: RefFrameTable,
    /// The colour of the frame in each reference slot.
    slots: [ColorFacts; NUM_REF_FRAMES],
    frames: VecDeque<DecodedFrame<R>>,
    flushed: bool,
    error: Option<DecodeError>,
}

impl<R> RustyVp9Decoder<R>
where
    R: Runtime<DType = DType>,
{
    /// Opens a decoder whose frames go to tensors on `device`.
    pub fn new(device: &R::Device) -> Self {
        Self {
            device: device.clone(),
            library: Library::new(),
            parser: Vp9HeaderParser::new(),
            refs: RefFrameTable::default(),
            slots: [VP9_UNSTATED; NUM_REF_FRAMES],
            frames: VecDeque::new(),
            flushed: false,
            error: None,
        }
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

    /// Decodes each coded frame of a chunk, and queues the last shown one.
    fn decode(&mut self, coded: &[&[u8]], pts: i64) -> Result<(), DecodeError> {
        let mut shown = None;
        for frame in coded {
            let header = self.parser.parse(frame, &self.refs)?;
            check_layout(DECODER, &header)?;
            let color = if header.show_existing_frame {
                self.slots[usize::from(header.frame_to_show_map_idx)]
            } else {
                vp9_color_facts(&header.color_config)
            };
            if let Some(picture) = self.decode_one(frame, &header, pts)? {
                shown = Some((picture, color));
            }
            if !header.show_existing_frame {
                self.keep_references(&header, color);
            }
        }
        if let Some((picture, color)) = shown {
            let frame = to_frame::<R>(&self.device, &picture, pts, color)?;
            self.frames.push_back(frame);
        }
        Ok(())
    }

    /// Runs one coded frame through the crate. The crate takes one queue entry
    /// per call, so a call per frame leaves its queue empty.
    fn decode_one(
        &mut self,
        frame: &[u8],
        header: &UncompressedHeader,
        pts: i64,
    ) -> Result<Option<Picture>, DecodeError> {
        let input = || format!("the {}-byte frame at pts {pts}", frame.len());
        self.library
            .push(frame, Some(pts))
            .map_err(|error| software("Vp9Decoder::push", input(), &error))?;
        let shows = header.show_existing_frame || header.show_frame;
        match self.library.next_frame() {
            Ok(picture) if shows => Ok(Some(picture)),
            Ok(_) => Err(DecodeError::UnmatchedPicture { decoder: DECODER }),
            Err(LibraryError::Again) if !shows => Ok(None),
            Err(LibraryError::Again) => Err(DecodeError::MissingPictures {
                decoder: DECODER,
                count: 1,
                first_pts: pts,
            }),
            Err(error) => Err(software("Vp9Decoder::next_frame", input(), &error)),
        }
    }

    /// Applies the reference update of a frame that finished decoding (8.10).
    fn keep_references(&mut self, header: &UncompressedHeader, color: ColorFacts) {
        self.refs.update(header);
        for (slot, stored) in self.slots.iter_mut().enumerate() {
            if (header.refresh_frame_flags >> slot) & 1 == 1 {
                *stored = color;
            }
        }
    }
}

fn software(call: &'static str, input: String, error: &LibraryError) -> DecodeError {
    DecodeError::Software {
        decoder: DECODER,
        call,
        input,
        reason: error.to_string(),
    }
}

/// The picture's sample layout, or the refusal for one with no vidr frame.
fn picture_format(picture: &Picture) -> Result<PixelFormat, DecodeError> {
    let bit_depth = u8::try_from(picture.bit_depth).unwrap_or(u8::MAX);
    let subsampled = picture.subsampling_x == 1 && picture.subsampling_y == 1;
    match PixelFormat::for_bit_depth(bit_depth) {
        Some(format) if subsampled => Ok(format),
        _ => Err(DecodeError::Vp9Layout {
            decoder: DECODER,
            profile: if bit_depth > 8 { 2 } else { 0 },
            bit_depth,
            subsampling_x: picture.subsampling_x == 1,
            subsampling_y: picture.subsampling_y == 1,
        }),
    }
}

/// Copies one picture to tensors on `device`.
fn to_frame<R>(
    device: &R::Device,
    picture: &Picture,
    pts: i64,
    color: ColorFacts,
) -> Result<DecodedFrame<R>, DecodeError>
where
    R: Runtime<DType = DType>,
{
    let format = picture_format(picture)?;
    let (width, height) = (picture.width, picture.height);
    let display = Dimensions::new(width, height).map_err(|_| DecodeError::DisplayArea {
        rect: (0, 0, width as i32, height as i32),
        coded: (width, height),
    })?;
    let ([y, u, v], [y_stride, u_stride, v_stride]) =
        (picture.planes.as_slice(), picture.strides.as_slice())
    else {
        return Err(DecodeError::Software {
            decoder: DECODER,
            call: "Vp9Decoder::next_frame",
            input: format!("the frame at pts {pts}"),
            reason: format!(
                "the picture has {} planes and {} strides, and 3 of each are needed",
                picture.planes.len(),
                picture.strides.len()
            ),
        });
    };
    let planes = HostPlanes {
        y,
        y_stride: *y_stride,
        u,
        u_stride: *u_stride,
        v,
        v_stride: *v_stride,
        width: width as usize,
        height: height as usize,
    };
    let (luma, chroma) = upload_planes::<R>(device, planes, format)?;
    Ok(if R::packed_color_planes() {
        DecodedFrame::from_packed_planes(luma, chroma, format, pts, display, color)
    } else {
        DecodedFrame::from_planes(luma, chroma, format, pts, display, color)
    })
}

impl<R> Decoder<R> for RustyVp9Decoder<R>
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
        let coded = split_superframe(packet)?;
        let result = self.decode(&coded, pts);
        if let Err(error) = &result {
            self.error = Some(error.clone());
        }
        result
    }

    fn flush(&mut self) -> Result<(), DecodeError> {
        if self.flushed {
            return Ok(());
        }
        self.flushed = true;
        self.stored_error()?;
        self.library.flush();
        // `send` leaves nothing queued in the crate, so it reports the end at once.
        let result = match self.library.next_frame() {
            Err(LibraryError::Eof) => Ok(()),
            Ok(_) => Err(DecodeError::UnmatchedPicture { decoder: DECODER }),
            Err(error) => Err(software(
                "Vp9Decoder::next_frame",
                "the end of the stream".to_owned(),
                &error,
            )),
        };
        if let Err(error) = &result {
            self.error = Some(error.clone());
        }
        result
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
    use numr::runtime::cpu::CpuRuntime;

    use super::*;
    use crate::bitstream::ColorConfig;
    use crate::decode::{PRIMARIES_UNSPECIFIED, TRANSFER_UNSPECIFIED};

    fn header(profile: u8, bit_depth: u8, x: bool, y: bool) -> UncompressedHeader {
        UncompressedHeader {
            profile,
            color_config: ColorConfig {
                bit_depth,
                color_space: 2,
                color_range: false,
                subsampling_x: x,
                subsampling_y: y,
            },
            ..UncompressedHeader::default()
        }
    }

    #[test]
    fn profiles_0_and_2_at_8_or_10_bits_pass() {
        assert_eq!(check_layout(DECODER, &header(0, 8, true, true)), Ok(()));
        assert_eq!(check_layout(DECODER, &header(2, 10, true, true)), Ok(()));
    }

    #[test]
    fn other_layouts_are_refused_by_name() {
        for (profile, depth, x, y) in [
            (1, 8, false, false),
            (1, 8, true, false),
            (1, 8, false, true),
            (2, 12, true, true),
            (3, 10, false, false),
            (3, 12, true, false),
        ] {
            assert_eq!(
                check_layout(DECODER, &header(profile, depth, x, y)),
                Err(DecodeError::Vp9Layout {
                    decoder: DECODER,
                    profile,
                    bit_depth: depth,
                    subsampling_x: x,
                    subsampling_y: y,
                })
            );
        }
    }

    #[test]
    fn a_repeated_frame_states_no_layout() {
        let repeat = UncompressedHeader {
            profile: 1,
            show_existing_frame: true,
            ..UncompressedHeader::default()
        };
        assert_eq!(check_layout(DECODER, &repeat), Ok(()));
    }

    #[test]
    fn colour_states_matrix_and_range_only() {
        let color = ColorConfig {
            bit_depth: 8,
            color_space: 1,
            color_range: true,
            subsampling_x: true,
            subsampling_y: true,
        };
        assert_eq!(
            vp9_color_facts(&color),
            ColorFacts {
                matrix_coefficients: 5,
                transfer_characteristics: TRANSFER_UNSPECIFIED,
                colour_primaries: PRIMARIES_UNSPECIFIED,
                full_range: true,
                chroma_location: None,
            }
        );
    }

    #[test]
    fn the_decoder_moves_between_threads() {
        const fn assert_send<T: Send>() {}
        assert_send::<RustyVp9Decoder<CpuRuntime>>();
    }
}
