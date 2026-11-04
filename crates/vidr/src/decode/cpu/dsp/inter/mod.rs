//! Inter sample prediction kernels (8.5.3.3): interpolation with clamped reference
//! coordinates, and default and explicit weighting.

pub(crate) mod fir;
pub(crate) mod interp;
pub(crate) mod weight;
