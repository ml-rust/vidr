//! Video decoding into numr tensors.
//!
//! [`Decoder`] is the interface every backend implements. Each backend is a
//! Cargo feature:
//!
//! | Backend | Feature | Type |
//! | --- | --- | --- |
//! | NVDEC | `nvdec` | `nvdec::NvdecDecoder` |
//! | dav1d, AV1 on the CPU | `dav1d` | `dav1d::Dav1dDecoder` |
//! | OpenH264, H.264 Baseline on the CPU | `openh264`, `openh264-cisco` | `openh264::Openh264Decoder` |
//! | rusty_vp9, VP9 on the CPU | `cpu-vp9` | `rusty_vp9::RustyVp9Decoder` |
//! | This crate's own HEVC core, Main and Main 10 on the CPU | `cpu-hevc` | `hevc::HevcDecoder` |
//! | Vulkan Video, H.264 8-bit 4:2:0 on numr's WebGPU device | `vulkan-video` | `crate::vulkan::VulkanDecoder` |
//!
//! Frames come out as [`DecodedFrame`]: NV12 or P010 planes on the decoder's
//! device, as samples or packed words, ready for numr's `ColorOps::yuv_to_rgb`.

mod codec;
mod color;
pub(crate) mod frame;
mod traits;

#[cfg(feature = "cpu-hevc")]
pub(crate) mod cpu;
#[cfg(feature = "dav1d")]
pub mod dav1d;
#[cfg(feature = "cpu-hevc")]
pub mod hevc;
#[cfg(any(feature = "cpu-hevc", feature = "vulkan-video"))]
pub(crate) mod hevc_order;
#[cfg(feature = "nvdec")]
pub mod nvdec;
#[cfg(any(feature = "openh264", feature = "openh264-cisco"))]
pub mod openh264;
#[cfg(feature = "cpu-vp9")]
pub mod rusty_vp9;
#[cfg(any(feature = "cpu-vp9", feature = "vulkan-video"))]
pub(crate) mod vp9_frame;

pub use codec::{ChromaFormat, DecodeCodec, PixelFormat};
#[cfg(any(feature = "openh264", feature = "openh264-cisco"))]
pub(crate) use color::unspecified_color;
#[cfg(any(feature = "nvdec", feature = "cpu-vp9", feature = "vulkan-video"))]
pub(crate) use color::vp9_matrix;
#[cfg(any(
    feature = "openh264",
    feature = "openh264-cisco",
    feature = "cpu-hevc",
    feature = "vulkan-video"
))]
pub(crate) use color::vui_color;
pub use color::{
    ChromaLocation, ColorFacts, MATRIX_BT470BG, MATRIX_BT709, MATRIX_FCC, MATRIX_SMPTE170M,
    MATRIX_SMPTE240M, MATRIX_UNSPECIFIED, PRIMARIES_UNSPECIFIED, TRANSFER_UNSPECIFIED,
};
pub use frame::{DecodedFrame, PlaneForm};
pub use traits::Decoder;
