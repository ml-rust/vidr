//! Crate error types.
//!
//! Every message names what failed, the inputs, and the next action.

mod any;
#[cfg(any(feature = "h264", feature = "hevc", feature = "av1", feature = "vp9"))]
mod bitstream;
#[cfg(feature = "decode")]
mod decode;
#[cfg(feature = "encode")]
mod encode;
#[cfg(feature = "frame")]
mod frame;
mod geometry;
mod support;
mod time;
mod timing;

pub use any::Error;
#[cfg(any(feature = "h264", feature = "hevc", feature = "av1", feature = "vp9"))]
pub use bitstream::BitstreamError;
#[cfg(feature = "decode")]
pub use decode::DecodeError;
#[cfg(feature = "encode")]
pub use encode::EncodeError;
#[cfg(feature = "frame")]
pub use frame::FrameError;
pub use geometry::GeometryError;
pub use time::{Operation, TimeError};
pub use timing::{EditListError, TimingError};
