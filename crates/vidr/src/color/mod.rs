//! Colour policy: which facts a stream has, whether vidr takes them, and
//! which conversion and tags each frame gets.

mod policy;

#[cfg(any(feature = "mp4", feature = "mkv"))]
pub use policy::merge_container;
pub use policy::{ColorPlan, check_sdr, color_plan};
