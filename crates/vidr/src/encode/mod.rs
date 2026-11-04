//! Video encoding from numr tensors.
//!
//! [`Encoder`] is the interface every backend implements. Each backend is a
//! Cargo feature:
//!
//! | Backend | Feature | Type |
//! | --- | --- | --- |
//! | NVENC | `nvenc` | `nvenc::NvencEncoder` |
//! | OpenH264, H.264 on the CPU | `openh264`, `openh264-cisco` | `openh264::Openh264Encoder` |
//! | rav1r, AV1 on the CPU | `cpu-av1` | `rav1r::Rav1rEncoder` |
//!
//! Frames go in as [`EncodeFrame`]: Y and UV planes in NV12 or P010, dense or
//! with a row pitch, as samples or packed words. Packets come out as
//! [`EncodedPacket`]: MP4 and Matroska samples in decode order, with exact
//! timestamps.

mod codec;
mod config;
#[cfg_attr(
    not(any(
        feature = "nvenc",
        feature = "openh264",
        feature = "openh264-cisco",
        feature = "cpu-av1"
    )),
    allow(dead_code)
)]
mod dts;
mod frame;
mod traits;

#[cfg(feature = "nvenc")]
pub mod nvenc;
#[cfg(any(feature = "openh264", feature = "openh264-cisco"))]
pub mod openh264;
#[cfg(feature = "cpu-av1")]
pub mod rav1r;

pub use crate::decode::{ChromaLocation, ColorFacts, PixelFormat, PlaneForm};
pub use codec::{EncodeCodec, Preset, RateControl, Tuning};
pub use config::EncodeConfig;
pub use frame::{EncodeFrame, EncodedPacket};
pub use traits::Encoder;
