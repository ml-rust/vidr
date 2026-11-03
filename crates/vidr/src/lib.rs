//! Video for Rust. This release holds the exact frame timing model, frame
//! geometry, output checks, and, behind the `h264`, `hevc`, and `av1`
//! features, the bitstream filters between container samples and hardware
//! codecs. Behind `nvdec`, `decode::nvdec::NvdecDecoder` decodes H.264,
//! HEVC, AV1, and VP9 on NVDEC into numr CUDA tensors. Software codecs,
//! other hardware decoders, encoding, and the filter graph are roadmap
//! items.

#[cfg(any(feature = "h264", feature = "hevc", feature = "av1"))]
pub mod bitstream;
pub mod check;
#[cfg(feature = "decode")]
pub mod decode;
pub mod error;
pub mod geometry;
pub mod time;

pub use check::{CheckError, DURATION_TOLERANCE, Deviation, Side, StreamFacts, check_output};
#[cfg(any(feature = "h264", feature = "hevc", feature = "av1"))]
pub use error::BitstreamError;
#[cfg(feature = "decode")]
pub use error::DecodeError;
pub use error::{EditListError, Error, GeometryError, Operation, TimeError, TimingError};
pub use geometry::{Dimensions, Geometry, Rotation, SampleAspectRatio};
pub use time::{
    DurationHints, EditEntry, EditTimeline, FrameTiming, LastDurationSource, OrderViolation,
    Rational, Rounding, StartOffset, TrackTiming, presentation_order_violations, rescale,
};
