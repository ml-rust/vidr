//! `Rav1rEncoder` on numr's CPU backend, decoded back by independent decoders.
//!
//! A decoder command line tool that is absent skips its test with a notice.

#[cfg(feature = "dav1d")]
mod dav1d;
mod encode;
mod external;
mod support;
