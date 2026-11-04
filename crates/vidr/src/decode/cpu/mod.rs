//! CPU video decoders written in this crate, from the codec specifications.
//!
//! `hevc` holds the HEVC decoding core: syntax parser, reconstruction, inter prediction,
//! in-loop filters, picture hash check, and decoded picture buffer. `dsp` holds the
//! block-level kernels. No module here names a numr type. `decode::hevc::HevcDecoder`
//! connects the core to `DecodedFrame`.

#![forbid(unsafe_code)]

#[cfg(test)]
mod adapter_tests;
pub(crate) mod dsp;
pub(crate) mod hevc;
