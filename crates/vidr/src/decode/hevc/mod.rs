//! HEVC software decode on this crate's own core, into numr tensors through host
//! memory.
//!
//! Each frame takes one copy to the tensors, and [`Decoder::route`] declares it. A
//! 10-bit picture first copies once on the host, into little-endian bytes. HEVC has
//! patent pools: the README states what a binary distribution needs.
//!
//! | Module | Holds |
//! | --- | --- |
//! | `decoder` | `HevcDecoder`, its options, and the calling-thread path |
//! | `threaded` | The same decoder over the worker pool |
//!
//! [`Decoder::route`]: crate::decode::Decoder::route

mod decoder;
mod threaded;

pub use decoder::{HevcDecoder, HevcOptions};
