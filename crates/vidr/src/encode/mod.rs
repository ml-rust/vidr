//! Video encoding from numr tensors.
//!
//! [`Encoder`] is the interface every backend implements. Each backend is a
//! Cargo feature:
//!
//! | Backend | Feature | Type |
//! | --- | --- | --- |
//! | NVENC | `nvenc` | `nvenc::NvencEncoder` |
//!
//! Frames go in as [`EncodeFrame`]: Y and UV planes in NV12 or P010, dense or
//! with a row pitch. Packets come out as [`EncodedPacket`]: MP4 and Matroska
//! samples in decode order, with exact timestamps.

mod codec;
mod config;
mod frame;
mod traits;

#[cfg(feature = "nvenc")]
pub mod nvenc;

pub use crate::decode::{ChromaLocation, ColorFacts, PixelFormat};
pub use codec::{EncodeCodec, Preset, RateControl, Tuning};
pub use config::EncodeConfig;
pub use frame::{EncodeFrame, EncodedPacket};
pub use traits::Encoder;
