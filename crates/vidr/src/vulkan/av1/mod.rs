//! AV1 on Vulkan Video. Session parameters hold the sequence header. Each picture states its
//! whole frame header, with one offset and size per tile.
//!
//! | Module | Holds |
//! | --- | --- |
//! | `std_params` | The sequence header as `StdVideoAV1SequenceHeader` |
//! | `std_picture` | One frame header as `StdVideoDecodeAV1*` structures |
//! | `refs` | The eight reference names and the DPB slots they share |
//! | `decoder` | One stream: sequences, sessions, frames, and frames shown again |
//!
//! Every module but `decoder` is pure, so tests run with no GPU.

mod decoder;
mod refs;
pub(super) mod std_params;
mod std_picture;

pub(super) use decoder::Av1Decoder;
