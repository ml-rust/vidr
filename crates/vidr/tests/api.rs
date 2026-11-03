//! Integration tests over vidr's public API.

#[cfg(any(feature = "h264", feature = "hevc", feature = "av1"))]
mod bitstream;
#[cfg(feature = "nvdec")]
mod nvdec;
mod timing;
