//! Conversion between decoded 4:2:0 planes and float RGB.
//!
//! [`to_rgb`] converts a [`DecodedFrame`](crate::decode::DecodedFrame) to
//! float RGB. [`from_rgb`] converts float RGB to [`OutputPlanes`] for an
//! encoder. Both run on the device of their input, on any numr backend whose
//! `ColorOps` takes U8 and U16 planes.

mod convert;

pub use convert::{OutputPlanes, from_rgb, to_rgb};
