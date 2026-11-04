//! Conversion between decoded 4:2:0 planes and float RGB, and host copies of
//! those planes for a CPU codec.
//!
//! [`to_rgb`] converts a [`DecodedFrame`](crate::decode::DecodedFrame) to
//! float RGB. [`from_rgb`] converts float RGB to [`OutputPlanes`] for an
//! encoder. Both run on the device of their input, on any numr backend with
//! colour kernels. WebGPU takes planes as packed words, so its frames hold
//! [`PlaneForm::Words`]. [`upload_planes`] and [`download_planes`] copy planes
//! between host memory and tensors.

mod convert;
mod host;

pub use crate::decode::PlaneForm;
pub use convert::{OutputPlanes, from_rgb, reuse_tensor, to_rgb, to_rgb_reusing};
pub use host::{HostFrame, HostPlanes, download_planes, upload_planes};
