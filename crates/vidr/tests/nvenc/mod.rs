//! NVENC encode on the local GPU, checked by decoding the output back.
//!
//! Every test skips with a notice when the NVIDIA library, a CUDA device, or
//! ffmpeg is missing.

mod bframes;
mod encode;
mod refusals;
mod support;
