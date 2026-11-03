//! Crate error types.
//!
//! Every message names what failed, the inputs, and the next action.

mod any;
mod geometry;
mod support;
mod time;
mod timing;

pub use any::Error;
pub use geometry::GeometryError;
pub use time::{Operation, TimeError};
pub use timing::{EditListError, TimingError};
