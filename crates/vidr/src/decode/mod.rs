//! Video decoding into numr tensors.
//!
//! [`Decoder`] is the interface every backend implements. Each backend is a
//! Cargo feature:
//!
//! | Backend | Feature | Type |
//! | --- | --- | --- |
//! | NVDEC | `nvdec` | `nvdec::NvdecDecoder` |
//!
//! Frames come out as [`DecodedFrame`]: dense NV12 or P010 planes on the
//! decoder's device, ready for numr's `ColorOps::yuv_to_rgb`.

mod codec;
mod color;
pub(crate) mod frame;
mod traits;

#[cfg(feature = "nvdec")]
pub mod nvdec;

pub use codec::{ChromaFormat, DecodeCodec, PixelFormat};
pub use color::{
    ChromaLocation, ColorFacts, MATRIX_BT470BG, MATRIX_BT709, MATRIX_FCC, MATRIX_SMPTE170M,
    MATRIX_SMPTE240M, MATRIX_UNSPECIFIED, PRIMARIES_UNSPECIFIED, TRANSFER_UNSPECIFIED,
};
pub use frame::DecodedFrame;
pub use traits::Decoder;
