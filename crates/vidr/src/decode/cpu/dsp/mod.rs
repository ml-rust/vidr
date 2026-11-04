//! Block-level kernels of the CPU decoders. Each runs on scalar Rust, written so
//! the compiler vectorises the inner loops, with exact integer output.

pub(crate) mod deblock;
pub(crate) mod inter;
pub(crate) mod intra;
pub(crate) mod sample;
pub(crate) mod sao;
pub(crate) mod scale;
pub(crate) mod transform;
pub(crate) mod transform_matrix;
