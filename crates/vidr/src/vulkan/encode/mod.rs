//! Hardware encode on Vulkan Video, on numr's WebGPU device.
//!
//! | Module | Holds |
//! | --- | --- |
//! | `error` | Core errors as encode errors |
//! | `caps` | Encode capabilities as plain values, and what they allow |
//! | `gop` | Each picture's type, slots, and counters |
//! | `level` | The lowest level that holds a frame size, rate, and bitrate |
//! | `rate` | Rate control from the config and the capabilities |
//! | `feedback` | The encode feedback query and its results |
//! | `profile` | Encode profiles, their capability query, and their formats |
//! | `planes` | Frame checks and the buffer-to-image regions of each plane |
//! | `input` | The device-to-device copy of a frame into the input image |
//! | `pictures` | The input image and the DPB images |
//! | `engine` | One submit per picture: the coding scope, the wait, and the bitstream read |
//! | `h264`, `hevc`, `av1` | Each codec's parameter sets and picture structures |
//! | `encoder` | [`VulkanEncoder`] behind the `Encoder` trait |
//!
//! The modules `caps`, `gop`, `level`, `rate`, and `planes`, and the mapping functions of each
//! codec, are pure, so tests run with no GPU.

mod av1;
mod caps;
mod encoder;
mod engine;
mod error;
mod feedback;
mod gop;
mod h264;
mod hevc;
mod input;
mod level;
mod pictures;
mod planes;
mod profile;
mod rate;

pub use encoder::VulkanEncoder;
