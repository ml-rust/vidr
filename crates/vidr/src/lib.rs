//! Video for Rust. This release holds the exact frame timing model, frame
//! geometry, and output checks. Codecs, hardware decode and encode, and the
//! filter graph are roadmap items.

pub mod check;
pub mod error;
pub mod geometry;
pub mod time;

pub use check::{CheckError, DURATION_TOLERANCE, Deviation, Side, StreamFacts, check_output};
pub use error::{EditListError, Error, GeometryError, Operation, TimeError, TimingError};
pub use geometry::{Dimensions, Geometry, Rotation, SampleAspectRatio};
pub use time::{
    DurationHints, EditEntry, EditTimeline, FrameTiming, LastDurationSource, OrderViolation,
    Rational, Rounding, StartOffset, TrackTiming, presentation_order_violations, rescale,
};
