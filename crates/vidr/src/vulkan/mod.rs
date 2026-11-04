//! Hardware decode on Vulkan Video, on numr's WebGPU device.
//!
//! | Module | Holds |
//! | --- | --- |
//! | `probe` | What a physical device offers, and the choice of device and queues |
//! | `device` | [`VulkanDevice`]: the device through wgpu-hal, handed to numr |
//! | `validation` | The capture of validation errors |
//! | `objects` | One owner per Vulkan object, each freed on drop |
//! | `caps` | Capabilities as plain values, and what they allow |
//! | `session` | Profiles, queries, the video session, its memory, and its parameters |
//! | `retained` | Stable heap storage for what session parameters point to |
//! | `pictures` | DPB images in the form the capabilities require |
//! | `engine` | One submit per picture: bitstream, images, status |
//! | `request_checks` | The checks of one decode request, before any driver call |
//! | `recording` | The commands of one decode: barriers, the coding scope, and the release |
//! | `barriers` | Image barriers of a decode and a plane copy, as plain values |
//! | `copy` | Image planes into numr buffers, on the device |
//! | `h264` | H.264 parameter sets, picture order, DPB, and output order |
//! | `hevc` | HEVC parameter sets, sessions, and pictures on the shared picture order |
//! | `ref_pool` | The eight reference names of VP9 and AV1, and the DPB slots they share |
//! | `vp9` | VP9 frame headers, the reference pool, and frames shown again |
//! | `av1` | AV1 sequence and frame headers, the reference pool, tiles, and film grain |
//! | `decoder` | [`VulkanDecoder`] behind the `Decoder` trait |
//! | `gap` | Structures that `ash` 0.38 lacks |
//! | `zeroed` | All-zero values of plain C structures |
//!
//! All `unsafe` of this backend sits in this tree.

mod av1;
mod barriers;
mod caps;
mod copy;
mod decoder;
mod device;
mod engine;
mod error;
mod gap;
mod h264;
mod hevc;
mod objects;
mod pictures;
mod probe;
mod recording;
mod ref_pool;
mod request_checks;
mod retained;
mod session;
mod validation;
mod vp9;
mod zeroed;

pub use caps::DpbPreference;
pub use decoder::VulkanDecoder;
pub use device::{VulkanDevice, VulkanOptions};
pub use probe::vulkan_video_available;
