//! NVDEC hardware decode on numr's CUDA context and stream.
//!
//! `libnvcuvid.so.1` loads at run time, so the build needs no NVIDIA
//! library and a machine without the driver gets
//! [`DecodeError::LibraryMissing`](crate::error::DecodeError::LibraryMissing).

mod callbacks;
mod decoder;
mod ffi;
mod input;
mod sequence;
#[cfg(test)]
mod sequence_tests;
mod session;
mod signal;
mod surface;
mod vp9_color;

pub use decoder::{NvdecDecoder, nvdec_available};
