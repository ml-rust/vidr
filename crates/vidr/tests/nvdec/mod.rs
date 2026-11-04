//! NVDEC decode on the local GPU against ffmpeg's software decode.
//!
//! Every test skips with a notice when the NVIDIA library, a CUDA device,
//! or its fixture is missing.

mod decode;
pub(crate) mod fixtures;
mod refusals;
mod structure;
mod vp9_color;
