//! Video for Rust. This release holds the exact frame timing model, frame
//! geometry, output checks, and, behind the `h264`, `hevc`, `av1`, and `vp9`
//! features, the bitstream filters between container samples and hardware
//! codecs. Behind `nvdec`, `decode::nvdec::NvdecDecoder` decodes H.264,
//! HEVC, AV1, and VP9 on NVDEC into numr CUDA tensors. Behind `nvenc`,
//! `encode::nvenc::NvencEncoder` encodes numr CUDA tensors to H.264, HEVC, and
//! AV1 samples on NVENC. Behind `frame`, `frame::to_rgb` and `frame::from_rgb`
//! convert between NV12 or P010 planes and float RGB. Software codecs, other
//! hardware codecs, and the filter graph are roadmap items.

#[cfg(any(feature = "h264", feature = "hevc", feature = "av1", feature = "vp9"))]
pub mod bitstream;
pub mod check;
#[cfg(feature = "decode")]
pub mod decode;
#[cfg(feature = "encode")]
pub mod encode;
pub mod error;
#[cfg(feature = "frame")]
pub mod frame;
pub mod geometry;
pub mod time;

pub use check::{CheckError, DURATION_TOLERANCE, Deviation, Side, StreamFacts, check_output};
#[cfg(any(feature = "h264", feature = "hevc", feature = "av1", feature = "vp9"))]
pub use error::BitstreamError;
#[cfg(feature = "decode")]
pub use error::DecodeError;
#[cfg(feature = "encode")]
pub use error::EncodeError;
#[cfg(feature = "frame")]
pub use error::FrameError;
pub use error::{EditListError, Error, GeometryError, Operation, TimeError, TimingError};
pub use geometry::{Dimensions, Geometry, Rotation, SampleAspectRatio};
pub use time::{
    DurationHints, EditEntry, EditTimeline, FrameTiming, LastDurationSource, OrderViolation,
    Rational, Rounding, StartOffset, TrackTiming, presentation_order_violations, rescale,
};
