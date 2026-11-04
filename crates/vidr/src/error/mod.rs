//! Crate error types.
//!
//! Every message names what failed, the inputs, and the next action.

mod any;
#[cfg(any(feature = "h264", feature = "hevc", feature = "av1"))]
mod bitstream;
#[cfg(feature = "decode")]
mod decode;
#[cfg(feature = "encode")]
mod encode;
mod geometry;
mod support;
mod time;
mod timing;

pub use any::Error;
#[cfg(any(feature = "h264", feature = "hevc", feature = "av1"))]
pub use bitstream::BitstreamError;
#[cfg(feature = "decode")]
pub use decode::DecodeError;
#[cfg(feature = "encode")]
pub use encode::EncodeError;
pub use geometry::GeometryError;
pub use time::{Operation, TimeError};
pub use timing::{EditListError, TimingError};
