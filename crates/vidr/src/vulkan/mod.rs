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
//! | `pictures` | DPB images in the form the capabilities require |
//! | `engine` | One submit per picture: bitstream, images, barriers, status |
//! | `copy` | Image planes into numr buffers, on the device |
//! | `h264` | H.264 parameter sets, picture order, DPB, and output order |
//! | `decoder` | [`VulkanDecoder`] behind the `Decoder` trait |
//! | `gap` | Structures that `ash` 0.38 lacks |
//!
//! All `unsafe` of this backend sits in this tree.

mod caps;
mod copy;
mod decoder;
mod device;
mod engine;
mod error;
mod gap;
mod h264;
mod objects;
mod pictures;
mod probe;
mod session;
mod validation;

pub use caps::DpbPreference;
pub use decoder::VulkanDecoder;
pub use device::{VulkanDevice, VulkanOptions};
pub use probe::vulkan_video_available;
