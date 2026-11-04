//! The refusals of `execution_for`.

use std::fmt;

use crate::request::Codec;

use super::select::SourceCodec;

/// Why no runtime, decoder, and encoder fit a request.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExecutionError {
    /// The request names two sides that do not work together.
    Pair {
        /// One side, as its flag.
        first: String,
        /// The other side, as its flag.
        second: String,
        /// Why the two do not fit.
        reason: &'static str,
        /// The flags that work.
        working: &'static str,
    },
    /// This build leaves out what the request needs.
    NotBuilt {
        /// What the request needs.
        what: String,
        /// The Cargo feature to enable.
        feature: &'static str,
    },
    /// This machine lacks what the request names.
    Unavailable {
        /// The flag that names it.
        what: String,
        /// The probe's text.
        reason: String,
        /// The next action.
        next: &'static str,
    },
    /// `auto` found no hardware path and no CPU path.
    NoPath {
        /// `decoder` or `encoder`.
        side: &'static str,
        /// Why the hardware path is out.
        hardware: String,
        /// Why the CPU path is out.
        cpu: Box<Self>,
    },
    /// `--decoder vulkan` names a codec that has no Vulkan Video path yet.
    VulkanCodec {
        /// The video's codec.
        codec: SourceCodec,
    },
    /// OpenH264 takes Baseline only, and the source states another profile.
    H264NotBaseline {
        /// `profile_idc` of the source.
        profile_idc: u8,
    },
    /// No CPU encoder exists for HEVC, by decision.
    HevcCpuEncode,
    /// This build holds only Cisco's OpenH264 path, and no binary path is set.
    CiscoLibraryMissing,
    /// No decoder reads the output back for the output check.
    NoOutputCheck {
        /// The output codec.
        codec: Codec,
        /// Why none of the decoders fits, and the next action.
        reason: &'static str,
    },
    /// A file holds a codec this release does not name.
    UnknownCodec {
        /// The codec's name.
        name: String,
    },
    /// A decoder or encoder opened on a runtime it has no path for.
    WrongRuntime {
        /// The backend.
        backend: &'static str,
        /// The runtime.
        runtime: &'static str,
    },
}

impl ExecutionError {
    pub(super) fn not_built_decoder(codec: SourceCodec, feature: &'static str) -> Self {
        Self::NotBuilt {
            what: format!("{} CPU decode", codec.name()),
            feature,
        }
    }
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pair {
                first,
                second,
                reason,
                working,
            } => write!(
                f,
                "{first} with {second} is refused, because {reason}. Pass {working}."
            ),
            Self::NotBuilt { what, feature } => write!(
                f,
                "{what} is not in this build, because the {feature} feature is off. Rebuild \
                 vidr-remaster with --features {feature}."
            ),
            Self::Unavailable { what, reason, next } => {
                write!(f, "{what} is unavailable on this machine: {reason}. {next}")
            }
            Self::NoPath {
                side,
                hardware,
                cpu,
            } => write!(
                f,
                "No {side} fits this run. Hardware path: {hardware}. {cpu}"
            ),
            Self::VulkanCodec { codec } => write!(
                f,
                "{} on Vulkan Video is not supported yet. --decoder vulkan decodes H.264, HEVC, \
                 and VP9 in this release. Pass --decoder auto or --decoder cpu.",
                codec.name()
            ),
            Self::H264NotBaseline { profile_idc } => write!(
                f,
                "OpenH264 decodes H.264 Baseline only, profile_idc 66, and this source states \
                 profile_idc {profile_idc}. A source with B-frames is refused, because \
                 OpenH264's B-slice output differs from the reference decode. Pass --decoder \
                 nvdec on a cuda build, --decoder vulkan on a vulkan-video build, or re-encode \
                 the source as Baseline."
            ),
            Self::HevcCpuEncode => f.write_str(
                "HEVC has no CPU encoder here, by decision. Pass --codec av1 with the cpu-av1 \
                 feature, or --encoder nvenc on a cuda build.",
            ),
            Self::CiscoLibraryMissing => f.write_str(
                "This build holds OpenH264 through Cisco's binary only, and no path names it. \
                 Pass --openh264-library PATH, or rebuild with --features openh264.",
            ),
            Self::NoOutputCheck { codec, reason } => write!(
                f,
                "The output check decodes the {codec} output, and no decoder here reads it: \
                 {reason}"
            ),
            Self::UnknownCodec { name } => write!(
                f,
                "This release names no decoder for codec {name}. Transcode the source to AV1, \
                 VP9, HEVC, or H.264 first."
            ),
            Self::WrongRuntime { backend, runtime } => write!(
                f,
                "{backend} has no path on the {runtime} runtime. Pass --runtime cuda for NVDEC \
                 and NVENC, --runtime wgpu for Vulkan Video, or --decoder cpu and --encoder cpu."
            ),
        }
    }
}

impl std::error::Error for ExecutionError {}
