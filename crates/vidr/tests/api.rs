//! Integration tests over vidr's public API.

#[cfg(any(feature = "h264", feature = "hevc", feature = "av1"))]
mod bitstream;
#[cfg(feature = "frame")]
mod frame;
#[cfg(feature = "nvdec")]
mod nvdec;
#[cfg(feature = "nvenc")]
mod nvenc;
mod timing;
