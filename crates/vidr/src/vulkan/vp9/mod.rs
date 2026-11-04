//! VP9 on Vulkan Video. VP9 uses no session parameters, and its picture structure holds the
//! whole frame header.
//!
//! | Module | Holds |
//! | --- | --- |
//! | `std_video` | The mapping of one frame header into `StdVideoDecodeVP9*` structures |
//! | `refs` | The eight reference names and the DPB slots they share |
//! | `decoder` | One stream: superframes, sessions, frames, and frames shown again |
//!
//! Every module but `decoder` is pure, so tests run with no GPU.

mod decoder;
mod refs;
mod std_video;

pub(super) use decoder::Vp9Decoder;
