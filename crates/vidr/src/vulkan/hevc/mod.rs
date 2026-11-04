//! HEVC on Vulkan Video.
//!
//! Picture order, the reference picture set, marking, and bumping come from
//! [`crate::decode::hevc_order`], shared with the CPU decoder.
//!
//! | Module | Holds |
//! | --- | --- |
//! | `sequence` | Refusals by name, and what a sequence asks of a session |
//! | `std_params` | The mapping of VPS, SPS, and PPS into `StdVideoH265*` structures |
//! | `std_common` | The structures both the VPS and the SPS point to |
//! | `std_picture` | The mapping of one picture into `StdVideoDecodeH265*` structures |
//! | `slots` | DPB slots of stored pictures, and the current picture's slot lists |
//! | `picture` | The slice segments of one picture |
//! | `decoder` | One stream: parameter sets, sessions, and pictures |
//!
//! Every module but `decoder` is pure, so tests run with no GPU.

mod decoder;
mod picture;
pub(super) mod sequence;
mod slots;
pub(super) mod std_common;
mod std_params;
#[cfg(test)]
mod std_params_tests;
mod std_picture;

pub(super) use decoder::HevcDecoder;
