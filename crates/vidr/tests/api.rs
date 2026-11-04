//! Integration tests over vidr's public API.

#[cfg(any(feature = "h264", feature = "hevc", feature = "av1"))]
mod bitstream;
mod conformance;
#[cfg(feature = "dav1d")]
mod dav1d;
#[cfg(any(
    feature = "nvdec",
    feature = "dav1d",
    feature = "openh264",
    feature = "openh264-cisco",
    feature = "cpu-vp9"
))]
mod decode_fixtures;
#[cfg(feature = "frame")]
mod frame;
#[cfg(any(
    feature = "h264",
    feature = "hevc",
    feature = "av1",
    feature = "cpu-vp9"
))]
mod framing;
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
#[cfg(feature = "cpu-av1")]
mod rav1r;
#[cfg(feature = "cpu-vp9")]
mod rusty_vp9;
mod timing;
