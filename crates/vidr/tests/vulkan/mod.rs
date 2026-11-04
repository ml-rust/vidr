//! Vulkan Video decode on the local GPU against ffmpeg's software decode.
//!
//! Every test skips with a notice when no device decodes H.264 on Vulkan Video, or
//! when its fixture is missing. Each test turns on `VK_LAYER_KHRONOS_validation` when
//! the loader lists it, and any validation error fails the test.

mod decode;
mod refusals;
mod support;
