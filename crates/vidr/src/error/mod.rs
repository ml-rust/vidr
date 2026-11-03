//! Crate error types.
//!
//! Every message names what failed, the inputs, and the next action.

mod any;
#[cfg(any(feature = "h264", feature = "hevc", feature = "av1"))]
mod bitstream;
mod geometry;
mod support;
mod time;
mod timing;

pub use any::Error;
#[cfg(any(feature = "h264", feature = "hevc", feature = "av1"))]
pub use bitstream::BitstreamError;
pub use geometry::GeometryError;
pub use time::{Operation, TimeError};
pub use timing::{EditListError, TimingError};
