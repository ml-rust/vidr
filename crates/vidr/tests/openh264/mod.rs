//! OpenH264 decode and encode on numr's CPU backend.
//!
//! A build with `openh264` runs on the source build. A build with only
//! `openh264-cisco` runs on the binary at `VIDR_OPENH264_CISCO`, and skips with a
//! notice when it is unset. A test whose fixture is missing also skips.

#[cfg(feature = "openh264-cisco")]
mod cisco;
mod decode;
mod encode;
mod support;
