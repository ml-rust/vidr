//! Block-level kernels of the CPU decoders. Each runs on scalar Rust, written so
//! the compiler vectorises the inner loops, with exact integer output.

pub(crate) mod sample;
pub(crate) mod scale;
pub(crate) mod transform;
pub(crate) mod transform_matrix;
