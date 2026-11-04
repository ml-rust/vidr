//! Integration tests over vidr's public API.

#[cfg(any(feature = "h264", feature = "hevc", feature = "av1"))]
mod bitstream;
#[cfg(feature = "dav1d")]
mod dav1d;
#[cfg(any(
    feature = "nvdec",
    feature = "dav1d",
    feature = "openh264",
    feature = "openh264-cisco"
))]
mod decode_fixtures;
#[cfg(feature = "frame")]
mod frame;
#[cfg(any(feature = "mp4", feature = "mkv"))]
mod media;
#[cfg(feature = "nvdec")]
mod nvdec;
#[cfg(feature = "nvenc")]
mod nvenc;
#[cfg(any(feature = "openh264", feature = "openh264-cisco"))]
mod openh264;
#[cfg(feature = "pipeline")]
mod pipeline;
mod timing;
