//! Bitstream filters against ffmpeg on generated fixtures, and robustness on
//! truncated and corrupt input.

#[cfg(feature = "av1")]
mod av1;
#[cfg(feature = "h264")]
mod avcc;
mod fixtures;
#[cfg(feature = "h264")]
mod h264;
#[cfg(feature = "hevc")]
mod hevc;
#[cfg(feature = "hevc")]
mod hvcc;
mod robustness;
