//! Frame conversion between 4:2:0 planes and float RGB.
//!
//! CPU tests run on every build with the `frame` feature. Tests that need
//! ffmpeg, a fixture, or a GPU skip with a notice when it is missing.

mod analytic;
#[cfg(all(feature = "nvdec", feature = "nvenc"))]
mod encode_loop;
#[cfg(feature = "nvdec")]
mod gpu;
mod odd;
mod refusals;
mod reuse;
mod roundtrip;
mod support;
mod swscale;
