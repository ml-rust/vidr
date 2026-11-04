//! A decoded frame: two dense planes in device memory, its pts, and its
//! colour facts.

use std::fmt;

use numr::ops::YuvPlanes;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::codec::PixelFormat;
use super::color::ColorFacts;
use crate::geometry::Dimensions;

/// The picture structure flags a decoder reports for one frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(feature = "nvdec"), allow(dead_code))]
pub(crate) struct PictureStructure {
    pub(crate) progressive: bool,
    pub(crate) top_field_first: bool,
}

/// One decoded 4:2:0 frame in display order.
///
/// For a display size of `W x H`, with `Wc = 2 * ceil(W / 2)` and
/// `Hc = ceil(H / 2)`:
///
/// | Plane | Shape | Dtype |
/// | --- | --- | --- |
/// | `y` | `[H, W]` | U8 for `Nv12`, U16 for `P010` |
/// | `uv` | `[Hc, Wc]`, `U, V` interleaved | as `y` |
///
/// Both planes are dense. Their bytes equal ffmpeg's `-f rawvideo` output in
/// `nv12` or `p010le` for the same frame, including odd sizes: a 321x181
/// AV1 or VP9 frame has a chroma plane of 91 rows and 322 columns. H.264 and
/// HEVC cannot produce an odd 4:2:0 size, because their cropping works in
/// units of two samples.
pub struct DecodedFrame<R: Runtime> {
    y: Tensor<R>,
    uv: Tensor<R>,
    format: PixelFormat,
    pts: i64,
    display: Dimensions,
    color: ColorFacts,
    progressive: bool,
    top_field_first: bool,
}

impl<R: Runtime> DecodedFrame<R> {
    /// Assembles a frame. Decoders call this after they write both planes.
    /// `structure` holds the decoder's picture structure flags for this
    /// frame.
    #[cfg_attr(not(feature = "nvdec"), allow(dead_code))]
    pub(crate) fn new(
        y: Tensor<R>,
        uv: Tensor<R>,
        format: PixelFormat,
        pts: i64,
        display: Dimensions,
        color: ColorFacts,
        structure: PictureStructure,
    ) -> Self {
        Self {
            y,
            uv,
            format,
            pts,
            display,
            color,
            progressive: structure.progressive,
            top_field_first: structure.top_field_first,
        }
    }

    /// Assembles a progressive frame from planes the caller holds: a decoder
    /// outside this crate, or a test.
    ///
    /// This function checks nothing. `vidr::frame::to_rgb` checks each plane
    /// against `format` and `display`, and an encoder checks them in `send`.
    pub fn from_planes(
        y: Tensor<R>,
        uv: Tensor<R>,
        format: PixelFormat,
        pts: i64,
        display: Dimensions,
        color: ColorFacts,
    ) -> Self {
        let structure = PictureStructure {
            progressive: true,
            top_field_first: false,
        };
        Self::new(y, uv, format, pts, display, color, structure)
    }

    /// The luma plane.
    pub fn y(&self) -> &Tensor<R> {
        &self.y
    }

    /// The interleaved chroma plane.
    pub fn uv(&self) -> &Tensor<R> {
        &self.uv
    }

    /// The sample layout.
    pub fn format(&self) -> PixelFormat {
        self.format
    }

    /// Presentation timestamp, as passed to `send` for the packet that
    /// carried this frame.
    pub fn pts(&self) -> i64 {
        self.pts
    }

    /// Display size in pixels, after the stream's cropping.
    pub fn display(&self) -> Dimensions {
        self.display
    }

    /// Colour facts from the sequence header.
    pub fn color(&self) -> &ColorFacts {
        &self.color
    }

    /// Whether the decoder reports this frame as progressive.
    ///
    /// NVDEC supplies the flag from `CUVIDPARSERDISPINFO.progressive_frame`.
    /// It describes how the picture was coded, not what the content shows.
    /// To decide whether a stream is interlaced, read its SPS.
    pub fn progressive(&self) -> bool {
        self.progressive
    }

    /// Whether the top field comes first, from
    /// `CUVIDPARSERDISPINFO.top_field_first`.
    ///
    /// It carries meaning only when [`DecodedFrame::progressive`] is
    /// `false`.
    pub fn top_field_first(&self) -> bool {
        self.top_field_first
    }

    /// The planes as numr's `ColorOps::yuv_to_rgb` takes them: `Nv12` or
    /// `P010`.
    ///
    /// numr converts even sizes only. A frame with an odd display width or
    /// height holds ffmpeg's plane layout, and `yuv_to_rgb` returns
    /// `Error::InvalidArgument` for the `planes` argument. `vidr::frame::to_rgb`
    /// converts every size.
    pub fn planes(&self) -> YuvPlanes<'_, R> {
        match self.format {
            PixelFormat::Nv12 => YuvPlanes::Nv12 {
                y: &self.y,
                uv: &self.uv,
            },
            PixelFormat::P010 => YuvPlanes::P010 {
                y: &self.y,
                uv: &self.uv,
            },
        }
    }

    /// The luma and chroma planes, consuming the frame.
    pub fn into_planes(self) -> (Tensor<R>, Tensor<R>) {
        (self.y, self.uv)
    }
}

// Manual impls: a derive bounds `R: Debug`, which `Runtime` does not require.
impl<R: Runtime> Clone for DecodedFrame<R> {
    fn clone(&self) -> Self {
        Self {
            y: self.y.clone(),
            uv: self.uv.clone(),
            format: self.format,
            pts: self.pts,
            display: self.display,
            color: self.color,
            progressive: self.progressive,
            top_field_first: self.top_field_first,
        }
    }
}

impl<R: Runtime> fmt::Debug for DecodedFrame<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecodedFrame")
            .field("y", &self.y)
            .field("uv", &self.uv)
            .field("format", &self.format)
            .field("pts", &self.pts)
            .field("display", &self.display)
            .field("color", &self.color)
            .field("progressive", &self.progressive)
            .field("top_field_first", &self.top_field_first)
            .finish()
    }
}

/// Shape `[rows, columns]` of the chroma plane for a display size.
#[cfg_attr(not(feature = "nvdec"), allow(dead_code))]
pub(crate) const fn chroma_shape(width: u32, height: u32) -> [usize; 2] {
    [
        height.div_ceil(2) as usize,
        (width.div_ceil(2) * 2) as usize,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chroma_plane_rounds_odd_sizes_up() {
        assert_eq!(chroma_shape(320, 180), [90, 320]);
        assert_eq!(chroma_shape(129, 75), [38, 130]);
    }
}
