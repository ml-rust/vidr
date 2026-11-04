//! Codec, chroma format, and pixel format names shared by every decoder.

use std::fmt;

use numr::dtype::DType;

/// A compressed video format a decoder takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DecodeCodec {
    /// H.264 / AVC. The config record is `avcC`.
    H264,
    /// H.265 / HEVC. The config record is `hvcC`.
    Hevc,
    /// AV1. The config record is `av1C`.
    Av1,
    /// VP9. Decoders ignore the `vpcC` config record.
    Vp9,
}

impl DecodeCodec {
    /// The codec name in messages: `H.264`, `HEVC`, `AV1`, or `VP9`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::H264 => "H.264",
            Self::Hevc => "HEVC",
            Self::Av1 => "AV1",
            Self::Vp9 => "VP9",
        }
    }
}

impl fmt::Display for DecodeCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Chroma subsampling of a coded stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ChromaFormat {
    /// Luma only, 4:0:0.
    Monochrome,
    /// 4:2:0.
    Yuv420,
    /// 4:2:2.
    Yuv422,
    /// 4:4:4.
    Yuv444,
}

impl fmt::Display for ChromaFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Monochrome => "4:0:0",
            Self::Yuv420 => "4:2:0",
            Self::Yuv422 => "4:2:2",
            Self::Yuv444 => "4:4:4",
        })
    }
}

/// Sample layout of a decoded 4:2:0 frame.
///
/// | Format | Bits | Sample | Planes |
/// | --- | --- | --- | --- |
/// | `Nv12` | 8 | U8 | Y, then interleaved U and V |
/// | `P010` | 10 | U16, value in the high 10 bits | Y, then interleaved U and V |
///
/// `P010` samples equal ffmpeg's `p010le`: `raw = value << 6`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PixelFormat {
    /// 8-bit NV12.
    Nv12,
    /// 10-bit P010.
    P010,
}

impl PixelFormat {
    /// The format for a coded bit depth, or `None` for a depth vidr does not
    /// output.
    pub const fn for_bit_depth(bit_depth: u8) -> Option<Self> {
        match bit_depth {
            8 => Some(Self::Nv12),
            10 => Some(Self::P010),
            _ => None,
        }
    }

    /// Bits per coded sample.
    pub const fn bit_depth(self) -> u8 {
        match self {
            Self::Nv12 => 8,
            Self::P010 => 10,
        }
    }

    /// Bytes per stored sample.
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::Nv12 => 1,
            Self::P010 => 2,
        }
    }

    /// The numr dtype of a stored sample: U8 for `Nv12`, U16 for `P010`.
    pub const fn sample_dtype(self) -> DType {
        match self {
            Self::Nv12 => DType::U8,
            Self::P010 => DType::U16,
        }
    }
}

impl fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Nv12 => "NV12",
            Self::P010 => "P010",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_format_follows_bit_depth() {
        assert_eq!(PixelFormat::for_bit_depth(8), Some(PixelFormat::Nv12));
        assert_eq!(PixelFormat::for_bit_depth(10), Some(PixelFormat::P010));
        assert_eq!(PixelFormat::for_bit_depth(12), None);
        assert_eq!(PixelFormat::P010.bytes_per_sample(), 2);
        assert_eq!(PixelFormat::Nv12.bit_depth(), 8);
    }

    #[test]
    fn names_match_messages() {
        assert_eq!(DecodeCodec::Hevc.to_string(), "HEVC");
        assert_eq!(ChromaFormat::Yuv420.to_string(), "4:2:0");
        assert_eq!(PixelFormat::P010.to_string(), "P010");
    }
}
