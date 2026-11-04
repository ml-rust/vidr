//! A frame an encoder takes, and a packet it returns.

use std::fmt;

use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::decode::DecodedFrame;

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
pub struct EncodeFrame<'a, R: Runtime> {
    y: &'a Tensor<R>,
    uv: &'a Tensor<R>,
}

impl<'a, R: Runtime> EncodeFrame<'a, R> {
    /// Creates a frame from its planes. The encoder checks them in `send`.
    pub const fn new(y: &'a Tensor<R>, uv: &'a Tensor<R>) -> Self {
        Self { y, uv }
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
        Self::new(frame.y(), frame.uv())
    }
}

// A manual impl: a derive bounds `R: Debug`, which `Runtime` does not require.
impl<R: Runtime> fmt::Debug for EncodeFrame<'_, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncodeFrame")
            .field("y", self.y)
            .field("uv", self.uv)
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
