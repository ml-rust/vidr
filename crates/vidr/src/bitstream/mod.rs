//! Bitstream filters between container samples and hardware codecs.
//!
//! MP4 and Matroska store H.264 and HEVC as length-prefixed NAL units with
//! an `avcC` or `hvcC` config record, and AV1 as low-overhead OBUs with an
//! `av1C` record. NVDEC takes Annex-B access units and AV1 temporal units.
//! NVENC writes Annex-B. The filters convert in both directions:
//!
//! | Filter | From | To |
//! | --- | --- | --- |
//! | `ToAnnexB` | H.264 or HEVC samples and config | Annex-B access units |
//! | `FromAnnexB` | H.264 or HEVC Annex-B access units | samples and config |
//! | `av1::Av1ToDecoder` | AV1 samples and `av1C` | temporal units |
//! | `av1::Av1FromEncoder` | AV1 temporal units | samples and `av1C` |
//!
//! The filters read raw sample and config bytes, and depend on no container
//! crate. Each codec is a Cargo feature: `h264`, `hevc`, and `av1`.

mod bits;
mod bytes;

#[cfg(any(feature = "h264", feature = "hevc"))]
pub mod annexb;
#[cfg(any(feature = "h264", feature = "hevc"))]
mod from_annexb;
#[cfg(any(feature = "h264", feature = "hevc"))]
pub mod length_prefixed;
#[cfg(any(feature = "h264", feature = "hevc"))]
mod nal;
#[cfg(any(feature = "h264", feature = "hevc"))]
mod rbsp;
#[cfg(any(feature = "h264", feature = "hevc"))]
mod to_annexb;
#[cfg(feature = "h264")]
mod to_annexb_h264;
#[cfg(feature = "hevc")]
mod to_annexb_hevc;

#[cfg(feature = "h264")]
mod avcc;
#[cfg(feature = "h264")]
mod h264_sps;
#[cfg(feature = "h264")]
mod h264_vui;

#[cfg(feature = "hevc")]
mod hevc_params;
#[cfg(feature = "hevc")]
mod hevc_ptl;
#[cfg(feature = "hevc")]
mod hevc_sps;
#[cfg(feature = "hevc")]
mod hevc_vui;
#[cfg(feature = "hevc")]
mod hvcc;
#[cfg(feature = "hevc")]
mod hvcc_build;

#[cfg(any(feature = "h264", feature = "hevc"))]
mod vui;

#[cfg(feature = "av1")]
pub mod av1;

#[cfg(any(feature = "h264", feature = "hevc"))]
pub use annexb::StartCode;
#[cfg(feature = "h264")]
pub use avcc::{AvcConfig, AvcHighProfile, avcc_has_extension};
#[cfg(any(feature = "h264", feature = "hevc"))]
pub use from_annexb::{FromAnnexB, NalCodec, NalConfig};
#[cfg(feature = "h264")]
pub use h264_vui::h264_sps_vui;
#[cfg(feature = "hevc")]
pub use hevc_sps::hevc_sps_vui;
#[cfg(feature = "hevc")]
pub use hvcc::{HevcConfig, HvccArray};
#[cfg(any(feature = "h264", feature = "hevc"))]
pub use length_prefixed::LengthSize;
#[cfg(feature = "h264")]
pub use nal::H264NalType;
#[cfg(feature = "hevc")]
pub use nal::HevcNalType;
#[cfg(any(feature = "h264", feature = "hevc"))]
pub use nal::ParameterSetKind;
#[cfg(any(feature = "h264", feature = "hevc"))]
pub use to_annexb::ToAnnexB;
#[cfg(any(feature = "h264", feature = "hevc"))]
pub use vui::{ChromaSampleLoc, ColourDescription, VideoSignalType, VuiSignal};
