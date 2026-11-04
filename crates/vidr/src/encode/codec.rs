//! The codec, preset, tuning, and rate control names of the encoder config.

use std::fmt;

use crate::decode::DecodeCodec;

/// A compressed video format an encoder writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EncodeCodec {
    /// H.264 / AVC. The config record is `avcC`.
    H264,
    /// H.265 / HEVC. The config record is `hvcC`.
    Hevc,
    /// AV1. The config record is `av1C`.
    Av1,
}

impl EncodeCodec {
    /// The codec name in messages: `H.264`, `HEVC`, or `AV1`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::H264 => "H.264",
            Self::Hevc => "HEVC",
            Self::Av1 => "AV1",
        }
    }
}

impl fmt::Display for EncodeCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl From<EncodeCodec> for DecodeCodec {
    fn from(codec: EncodeCodec) -> Self {
        match codec {
            EncodeCodec::H264 => Self::H264,
            EncodeCodec::Hevc => Self::Hevc,
            EncodeCodec::Av1 => Self::Av1,
        }
    }
}

/// Encoder effort, from `P1`, the fastest, to `P7`, the best quality.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Preset {
    /// Fastest.
    P1,
    /// Faster.
    P2,
    /// Fast.
    P3,
    /// Medium.
    P4,
    /// Slow.
    P5,
    /// Slower.
    P6,
    /// Slowest, best quality.
    P7,
}

/// What the encoder optimises for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Tuning {
    /// High quality, for stored video.
    HighQuality,
    /// Low latency.
    LowLatency,
    /// Ultra low latency.
    UltraLowLatency,
    /// Lossless. The GPU must support lossless encode.
    Lossless,
    /// Ultra high quality. The GPU and driver must support it.
    UltraHighQuality,
}

/// Rate control.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RateControl {
    /// Variable bitrate at a target quality, with no bitrate target. This is
    /// ffmpeg's `-rc vbr -cq N -b:v 0`.
    VbrQuality {
        /// Target quality, lower is better: 1 to 51 for H.264 and HEVC, 1 to
        /// 63 for AV1.
        cq: u8,
    },
    /// One quantiser per picture type: at most 51 for H.264 and HEVC, any
    /// value for AV1.
    ConstQp {
        /// Quantiser of I pictures.
        intra: u8,
        /// Quantiser of P pictures.
        inter_p: u8,
        /// Quantiser of B pictures.
        inter_b: u8,
    },
    /// Variable bitrate at a target bitrate.
    Bitrate {
        /// Average bits per second, above zero.
        average: u32,
        /// Peak bits per second, at least `average`. `None` lets the encoder
        /// choose.
        maximum: Option<u32>,
    },
}
