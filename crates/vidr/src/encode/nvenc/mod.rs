//! NVENC hardware encode on numr's CUDA context and stream.
//!
//! `libnvidia-encode.so.1` loads at run time, so the build needs no NVIDIA
//! library and a machine without the driver gets
//! [`EncodeError::LibraryMissing`](crate::error::EncodeError::LibraryMissing).

mod dts;
mod encoder;
mod ffi;
mod lock;
mod output;
mod params;
#[cfg(test)]
mod params_tests;
mod session;
mod surface;
mod surface_layout;

pub use encoder::{NvencEncoder, nvenc_available};
