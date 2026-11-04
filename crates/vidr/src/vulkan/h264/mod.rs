//! H.264 on Vulkan Video.
//!
//! | Module | Holds |
//! | --- | --- |
//! | `params` | SPS and PPS parsing and refusals, and facts of a sequence |
//! | `std_video` | The mapping into `StdVideoH264*` structures |
//! | `slice` | Slice headers, picture boundaries, and reference marking syntax |
//! | `poc` | Picture order count, types 0, 1, and 2 (ITU-T H.264 8.2.1) |
//! | `dpb` | Reference marking and DPB slots (8.2.5) |
//! | `output` | Output order bumping (C.4) |
//! | `decoder` | One stream: parameter sets, sessions, and pictures |
//!
//! Every module but `decoder` is pure, so tests run with no GPU.

mod decoder;
mod dpb;
mod output;
mod params;
mod poc;
mod slice;
pub(super) mod std_video;
#[cfg(test)]
mod std_video_tests;
#[cfg(test)]
mod test_nal;

pub(super) use decoder::H264Decoder;
