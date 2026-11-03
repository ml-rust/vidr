//! Exact time: rationals, rounding, frame timing, and edit lists.
//!
//! No timestamp passes through floating point.

mod chain;
mod edit_list;
mod rational;
mod rounding;
mod ticks;
mod timing;

pub use edit_list::{EditEntry, EditTimeline};
pub use rational::Rational;
pub use rounding::{Rounding, rescale};
pub use timing::{
    DurationHints, FrameTiming, LastDurationSource, OrderViolation, StartOffset, TrackTiming,
    presentation_order_violations,
};

pub(crate) use chain::{Durations, durations};
pub(crate) use ticks::{checked_tick_add, checked_tick_sub};
