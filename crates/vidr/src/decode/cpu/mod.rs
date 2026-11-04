//! CPU video decoders written in this crate, from the codec specifications.
//!
//! `hevc` holds the HEVC intra decoding core: syntax parser, reconstruction, in-loop
//! filters, and picture hash check. `dsp` holds the block-level kernels. No module
//! here names a numr type, and a later adapter connects the core to `DecodedFrame`.

#![forbid(unsafe_code)]

#[cfg(test)]
mod adapter_tests;
pub(crate) mod dsp;
pub(crate) mod hevc;
