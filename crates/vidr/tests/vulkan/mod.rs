//! Vulkan Video decode on the local GPU against a software reference decode.
//!
//! Every test skips with a notice when no device runs Vulkan Video decode, or when its
//! fixture is missing. An HEVC or VP9 test also skips when the device lacks the codec. Each
//! test turns on `VK_LAYER_KHRONOS_validation` when the loader lists it, and any validation
//! error fails the test.

mod decode;
mod hevc;
mod refusals;
pub(crate) mod support;
mod vp9;
