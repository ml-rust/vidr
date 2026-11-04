//! A frame an encoder takes, and a packet it returns.

use std::fmt;

use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::decode::{DecodedFrame, PixelFormat, PlaneForm};
use crate::geometry::Dimensions;

/// One 4:2:0 frame to encode.
///
/// For a config of `W x H`, with `Wc = 2 * ceil(W / 2)` and `Hc = ceil(H / 2)`:
///
/// | Plane | Shape | Dtype |
/// | --- | --- | --- |
/// | `y` | `[H, W]` | U8 for `Nv12`, U16 for `P010` |
/// | `uv` | `[Hc, Wc]`, `U, V` interleaved | as `y` |
///
/// Each plane is dense or has a row pitch: a column stride of 1 and a row stride
/// of at least the row width. `Tensor::narrow` of a wider surface gives this
/// form. These are the planes numr's `ColorOps::rgb_to_yuv_into` writes and
/// [`DecodedFrame`] holds.
///
/// A frame in [`PlaneForm::Words`] holds those planes as U32 words, in the
/// shapes [`DecodedFrame`] documents. Words record neither sample size nor
/// row width, so the frame states its format and display size.
pub struct EncodeFrame<'a, R: Runtime> {
    y: &'a Tensor<R>,
    uv: &'a Tensor<R>,
    words: Option<(PixelFormat, Dimensions)>,
}

impl<'a, R: Runtime> EncodeFrame<'a, R> {
    /// Creates a frame from U8 or U16 planes. The encoder checks them in `send`.
    pub const fn new(y: &'a Tensor<R>, uv: &'a Tensor<R>) -> Self {
        Self { y, uv, words: None }
    }

    /// Creates a frame from packed U32 planes. The encoder checks them in `send`.
    pub const fn from_packed_planes(
        y: &'a Tensor<R>,
        uv: &'a Tensor<R>,
        format: PixelFormat,
        display: Dimensions,
    ) -> Self {
        Self {
            y,
            uv,
            words: Some((format, display)),
        }
    }

    /// How the planes hold their samples.
    pub const fn plane_form(&self) -> PlaneForm {
        match self.words {
            Some(_) => PlaneForm::Words,
            None => PlaneForm::Samples,
        }
    }

    /// The format and display size of a word-form frame, or `None` for samples.
    pub const fn word_layout(&self) -> Option<(PixelFormat, Dimensions)> {
        self.words
    }

    /// The luma plane.
    pub const fn y(&self) -> &'a Tensor<R> {
        self.y
    }

    /// The interleaved chroma plane.
    pub const fn uv(&self) -> &'a Tensor<R> {
        self.uv
    }
}

impl<'a, R: Runtime> From<&'a DecodedFrame<R>> for EncodeFrame<'a, R> {
    fn from(frame: &'a DecodedFrame<R>) -> Self {
        match frame.plane_form() {
            PlaneForm::Samples => Self::new(frame.y(), frame.uv()),
            PlaneForm::Words => {
                Self::from_packed_planes(frame.y(), frame.uv(), frame.format(), frame.display())
            }
        }
    }
}

// A manual impl: a derive bounds `R: Debug`, which `Runtime` does not require.
impl<R: Runtime> fmt::Debug for EncodeFrame<'_, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncodeFrame")
            .field("y", self.y)
            .field("uv", self.uv)
            .field("form", &self.plane_form())
            .field("words", &self.words)
            .finish()
    }
}

/// One encoded picture in decode order, as an MP4 or Matroska sample.
///
/// H.264 and HEVC samples hold length-prefixed NAL units with 4-byte lengths,
/// and no parameter sets or access unit delimiters. AV1 samples hold low-overhead
/// OBUs without temporal delimiters. The config record carries the rest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedPacket {
    /// The sample bytes.
    pub data: Vec<u8>,
    /// Presentation timestamp, as passed to `send`, in the config's time base.
    pub pts: i64,
    /// Decode timestamp in the config's time base. It increases strictly from
    /// packet to packet and never exceeds `pts`. With B-frames, the first
    /// `b_frames` packets precede the first pts, so a dts can be negative.
    pub dts: i64,
    /// Duration, as passed to `send`, in the config's time base.
    pub duration: i64,
    /// True for a picture a decoder can start from.
    pub keyframe: bool,
}
