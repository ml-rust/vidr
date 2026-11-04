//! The sequence callback's decisions: what the stream needs, whether the
//! GPU decodes it, and whether the current decoder still fits.
//!
//! Everything here is plain data in and out, so the unit tests run without
//! a GPU.

use std::ffi::{c_int, c_ulong};

use super::ffi::{
    CHROMA_420, CHROMA_422, CHROMA_444, CHROMA_MONOCHROME, CODEC_AV1, CODEC_H264, CODEC_HEVC,
    CODEC_VP9, CREATE_PREFER_CUVID, CuVideoCtxLock, CuVideoFormat, CudaVideoChromaFormat,
    CudaVideoCodec, CuvidDecodeCaps, CuvidDecodeCreateInfo, DEINTERLACE_WEAVE, SURFACE_NV12,
    SURFACE_P016, ShortRect,
};
use super::vp9_color;
use crate::bitstream::Vp9Color;
use crate::decode::{ChromaFormat, ChromaLocation, ColorFacts, DecodeCodec, PixelFormat};
use crate::error::DecodeError;
use crate::geometry::Dimensions;

/// Output surfaces a decoder can hold mapped at once. vidr keeps a surface
/// mapped until the copy out of it completes on numr's stream.
pub(super) const OUTPUT_SURFACES: u32 = 4;

/// Decode surfaces when the parser reports a minimum of 0. 20 holds the
/// largest H.264 and HEVC DPB of 16 pictures, the current picture, and
/// three pictures waiting for display.
const FALLBACK_DECODE_SURFACES: u32 = 20;

/// The `cudaVideoCodec` value of a codec.
pub(super) const fn cuvid_codec(codec: DecodeCodec) -> CudaVideoCodec {
    match codec {
        DecodeCodec::H264 => CODEC_H264,
        DecodeCodec::Hevc => CODEC_HEVC,
        DecodeCodec::Av1 => CODEC_AV1,
        DecodeCodec::Vp9 => CODEC_VP9,
    }
}

const fn chroma_of(value: CudaVideoChromaFormat) -> Option<ChromaFormat> {
    match value {
        CHROMA_MONOCHROME => Some(ChromaFormat::Monochrome),
        CHROMA_420 => Some(ChromaFormat::Yuv420),
        CHROMA_422 => Some(ChromaFormat::Yuv422),
        CHROMA_444 => Some(ChromaFormat::Yuv444),
        _ => None,
    }
}

/// The facts of one sequence header that decide the decoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Sequence {
    pub(super) codec: DecodeCodec,
    pub(super) format: PixelFormat,
    /// Coded `(width, height)`.
    pub(super) coded: (u32, u32),
    /// The rectangle NVDEC crops to, with even width and height so the
    /// target surface needs no scaling.
    pub(super) crop: ShortRect,
    /// The display size, odd sides kept.
    pub(super) display: Dimensions,
    pub(super) decode_surfaces: u32,
    pub(super) color: ColorFacts,
}

impl Sequence {
    /// Reads and checks a sequence header for a decoder opened on `codec`.
    ///
    /// For VP9, NVDEC's signal description is all zero, so `vp9` supplies
    /// the matrix and range from the latest key frame header. `None` gives
    /// an unspecified matrix. The argument is ignored for other codecs.
    ///
    /// # Errors
    ///
    /// [`DecodeError::CodecMismatch`], [`DecodeError::UnsupportedLayout`]
    /// for a stream that is not 4:2:0 at 8 or 10 bits, or
    /// [`DecodeError::DisplayArea`] for a display rectangle that does not
    /// fit the coded frame.
    pub(super) fn read(
        codec: DecodeCodec,
        format: &CuVideoFormat,
        chroma_location: Option<ChromaLocation>,
        vp9: Option<Vp9Color>,
    ) -> Result<Self, DecodeError> {
        if format.codec != cuvid_codec(codec) {
            return Err(DecodeError::CodecMismatch {
                expected: codec,
                found: format.codec,
            });
        }
        let luma = format.bit_depth_luma_minus8.saturating_add(8);
        let chroma_depth = format.bit_depth_chroma_minus8.saturating_add(8);
        let chroma = chroma_of(format.chroma_format);
        let layout_error = || DecodeError::UnsupportedLayout {
            codec,
            bit_depth_luma: luma,
            bit_depth_chroma: chroma_depth,
            chroma: chroma.unwrap_or(ChromaFormat::Yuv420),
        };
        if chroma != Some(ChromaFormat::Yuv420) || luma != chroma_depth {
            return Err(layout_error());
        }
        let pixel = PixelFormat::for_bit_depth(luma).ok_or_else(layout_error)?;

        let coded = (format.coded_width, format.coded_height);
        let (crop, display) = crop_of(format.display_area, coded)?;
        let decode_surfaces = match format.min_num_decode_surfaces {
            0 => FALLBACK_DECODE_SURFACES,
            minimum => u32::from(minimum),
        };
        let signal = format.video_signal_description;
        let mut color = ColorFacts {
            matrix_coefficients: signal.matrix_coefficients,
            transfer_characteristics: signal.transfer_characteristics,
            colour_primaries: signal.color_primaries,
            full_range: signal.full_range(),
            chroma_location,
        };
        if codec == DecodeCodec::Vp9 {
            vp9_color::apply(&mut color, vp9);
        }
        Ok(Self {
            codec,
            format: pixel,
            coded,
            crop,
            display,
            decode_surfaces,
            color,
        })
    }

    /// The chroma format, always 4:2:0 after [`Sequence::read`].
    pub(super) const fn chroma(&self) -> ChromaFormat {
        ChromaFormat::Yuv420
    }

    /// True when a decoder created for `self` also decodes `next`. Colour
    /// facts can change without a new decoder.
    pub(super) fn same_decoder(&self, next: &Self) -> bool {
        self.codec == next.codec
            && self.format == next.format
            && self.coded == next.coded
            && self.crop == next.crop
            && self.decode_surfaces >= next.decode_surfaces
    }

    /// Target surface `(width, height)`: the crop size rounded up to even,
    /// because NVDEC surfaces have even sides.
    pub(super) fn target(&self) -> (u32, u32) {
        (
            self.display.width().next_multiple_of(2),
            self.display.height().next_multiple_of(2),
        )
    }

    /// Checks the GPU's reply to `cuvidGetDecoderCaps` for this sequence.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Unsupported`] when the GPU does not decode the codec,
    /// depth, and chroma format, or lacks the output surface format, and
    /// [`DecodeError::CodedSize`] when the coded size is out of range.
    pub(super) fn check_caps(&self, caps: &CuvidDecodeCaps) -> Result<(), DecodeError> {
        let unsupported = |reason: String| DecodeError::Unsupported {
            codec: self.codec,
            bit_depth: self.format.bit_depth(),
            chroma: self.chroma(),
            reason,
        };
        if caps.is_supported == 0 {
            return Err(unsupported(
                "cuvidGetDecoderCaps reports bIsSupported 0".to_owned(),
            ));
        }
        let surface = self.surface_format();
        if caps.output_format_mask & (1 << surface) == 0 {
            return Err(unsupported(format!(
                "nOutputFormatMask {:#06x} lacks {}",
                caps.output_format_mask,
                self.surface_name()
            )));
        }
        let (width, height) = self.coded;
        let macroblocks = u64::from(width.div_ceil(16)) * u64::from(height.div_ceil(16));
        if width < u32::from(caps.min_width)
            || height < u32::from(caps.min_height)
            || width > caps.max_width
            || height > caps.max_height
            || macroblocks > u64::from(caps.max_mb_count)
        {
            return Err(DecodeError::CodedSize {
                codec: self.codec,
                width,
                height,
                min: (u32::from(caps.min_width), u32::from(caps.min_height)),
                max: (caps.max_width, caps.max_height),
                max_macroblocks: caps.max_mb_count,
            });
        }
        Ok(())
    }

    /// The caps query for this sequence.
    pub(super) fn caps_query(&self) -> CuvidDecodeCaps {
        CuvidDecodeCaps {
            codec_type: cuvid_codec(self.codec),
            chroma_format: CHROMA_420,
            bit_depth_minus8: u32::from(self.format.bit_depth() - 8),
            ..CuvidDecodeCaps::default()
        }
    }

    /// The `cuvidCreateDecoder` parameters: weave, no scaling, the crop as
    /// display area, and `vid_lock` as the context lock.
    pub(super) fn create_info(&self, vid_lock: CuVideoCtxLock) -> CuvidDecodeCreateInfo {
        let (target_width, target_height) = self.target();
        let mut info = CuvidDecodeCreateInfo::zeroed();
        info.width = c_ulong::from(self.coded.0);
        info.height = c_ulong::from(self.coded.1);
        info.num_decode_surfaces = c_ulong::from(self.decode_surfaces);
        info.codec_type = cuvid_codec(self.codec);
        info.chroma_format = CHROMA_420;
        info.creation_flags = CREATE_PREFER_CUVID;
        info.bit_depth_minus8 = c_ulong::from(self.format.bit_depth() - 8);
        info.max_width = c_ulong::from(self.coded.0);
        info.max_height = c_ulong::from(self.coded.1);
        info.display_area = self.crop;
        info.output_format = self.surface_format();
        info.deinterlace_mode = DEINTERLACE_WEAVE;
        info.target_width = c_ulong::from(target_width);
        info.target_height = c_ulong::from(target_height);
        info.num_output_surfaces = c_ulong::from(OUTPUT_SURFACES);
        info.vid_lock = vid_lock;
        info
    }

    /// A description of the sequence for driver error messages.
    pub(super) fn describe(&self) -> String {
        format!(
            "{} {}-bit {} coded {}x{}, {} decode surfaces, {} output",
            self.codec,
            self.format.bit_depth(),
            self.chroma(),
            self.coded.0,
            self.coded.1,
            self.decode_surfaces,
            self.surface_name()
        )
    }

    const fn surface_format(&self) -> c_int {
        match self.format {
            PixelFormat::Nv12 => SURFACE_NV12,
            PixelFormat::P010 => SURFACE_P016,
        }
    }

    const fn surface_name(&self) -> &'static str {
        match self.format {
            PixelFormat::Nv12 => "cudaVideoSurfaceFormat_NV12",
            PixelFormat::P010 => "cudaVideoSurfaceFormat_P016",
        }
    }
}

/// The NVDEC crop and the display size for a display rectangle.
///
/// An odd side grows by one pixel, so the crop matches the even target
/// surface and NVDEC does not scale. The growth can pass an odd coded edge:
/// NVDEC then fills the surface without moving a visible pixel, which the
/// byte-exact tests prove for AV1 and VP9. A rounded coded size or an odd
/// target rectangle does not work: the first moves AV1 pixels and the second
/// blanks the last column. The display size keeps the odd side, and the frame
/// copy reads only that size.
fn crop_of(
    area: super::ffi::IntRect,
    coded: (u32, u32),
) -> Result<(ShortRect, Dimensions), DecodeError> {
    let error = || DecodeError::DisplayArea {
        rect: (area.left, area.top, area.right, area.bottom),
        coded,
    };
    let fits = |low: c_int, high: c_int, limit: u32| {
        low >= 0 && high > low && i64::from(high) <= i64::from(limit)
    };
    if !fits(area.left, area.right, coded.0) || !fits(area.top, area.bottom, coded.1) {
        return Err(error());
    }
    let width = (area.right - area.left) as u32;
    let height = (area.bottom - area.top) as u32;
    let crop = ShortRect {
        left: i16::try_from(area.left).map_err(|_| error())?,
        top: i16::try_from(area.top).map_err(|_| error())?,
        right: i16::try_from(area.right + (width % 2) as c_int).map_err(|_| error())?,
        bottom: i16::try_from(area.bottom + (height % 2) as c_int).map_err(|_| error())?,
    };
    let display = Dimensions::new(width, height).map_err(|_| error())?;
    Ok((crop, display))
}
