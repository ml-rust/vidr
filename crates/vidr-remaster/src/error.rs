//! Errors of a remaster run.

use std::fmt;
use std::io;
use std::path::PathBuf;

#[cfg(feature = "transform")]
use vidr::PipelineError;
use vidr::{CheckError, GeometryError, TimeError, TimingError};
#[cfg(any(feature = "mp4", feature = "mkv"))]
use vidr::{ColorError, DecodeError, EncodeError, MediaError};

use crate::catalog::CatalogError;
use crate::execution::ExecutionError;
use crate::provenance::ProvenanceError;
use crate::publish::PublishError;
use crate::report::ReportError;
use crate::request::RequestError;
use crate::resolution::ResolutionError;
use crate::reuse::ReuseError;
#[cfg(any(feature = "mp4", feature = "mkv"))]
use crate::scan::ScanError;
#[cfg(feature = "transform")]
use crate::transform::TransformError;

/// An error that ends a remaster run.
#[derive(Debug)]
#[non_exhaustive]
pub enum RemasterError {
    /// The request does not pass validation. The run writes no report.
    Request(RequestError),
    /// The run report does not create or write.
    Report(ReportError),
    /// The temporary output does not create or publish.
    Publish(PublishError),
    /// A model is not in this build or not on disk.
    Catalog(CatalogError),
    /// An enhanced video is refused.
    Reuse(ReuseError),
    /// The provenance sidecar does not write before the publish.
    Provenance(ProvenanceError),
    /// No runtime, decoder, and encoder fit the request.
    Execution(ExecutionError),
    /// This build leaves out what a run needs.
    MissingFeatures {
        /// The Cargo features to enable.
        missing: Vec<&'static str>,
    },
    /// A stop request ends a step. `remaster` returns `RunOutcome::Stopped` in
    /// place of this value.
    Stopped,
    /// The output size does not compute.
    Resolution(ResolutionError),
    /// Exact time arithmetic does not fit.
    Time(TimeError),
    /// A track's frame timing breaks a rule.
    Timing(TimingError),
    /// A track's geometry does not evaluate.
    Geometry(GeometryError),
    /// The temporary output differs from the source.
    Check(CheckError),
    /// A file of the run does not open, sync, or read.
    File {
        /// The file.
        path: PathBuf,
        /// The operating-system error.
        error: io::Error,
    },
    /// No CUDA device answers.
    NoGpu,
    /// The CUDA device does not open.
    Gpu {
        /// The message from numr or the driver library.
        reason: String,
    },
    /// The pre-scan refuses a video.
    #[cfg(any(feature = "mp4", feature = "mkv"))]
    Scan(ScanError),
    /// Media input or output fails.
    #[cfg(any(feature = "mp4", feature = "mkv"))]
    Media(MediaError),
    /// The colour of a video has no plan.
    #[cfg(any(feature = "mp4", feature = "mkv"))]
    Color(ColorError),
    /// The decoder fails during a scan.
    #[cfg(any(feature = "mp4", feature = "mkv"))]
    Decode {
        /// The video.
        path: PathBuf,
        /// The decoder's error.
        error: DecodeError,
    },
    /// The encoder refuses its config or fails.
    #[cfg(any(feature = "mp4", feature = "mkv"))]
    Encode(EncodeError),
    /// The hardware encoder has no 10-bit H.264, and the source is 10-bit.
    #[cfg(any(feature = "mp4", feature = "mkv"))]
    H264TenBit,
    /// OpenH264 encodes 8 bits only, and the source is 10-bit.
    #[cfg(any(feature = "mp4", feature = "mkv"))]
    Openh264TenBit,
    /// The pump fails.
    #[cfg(feature = "transform")]
    Pipeline(PipelineError),
    /// The frame transform does not load.
    #[cfg(feature = "transform")]
    Transform(TransformError),
}

impl RemasterError {
    /// Returns `true` for the result of a stop request.
    pub fn is_stop(&self) -> bool {
        #[cfg(feature = "transform")]
        if let Self::Pipeline(error) = self {
            return error.is_stop();
        }
        matches!(self, Self::Stopped)
    }
}

impl fmt::Display for RemasterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Request(error) => fmt::Display::fmt(error, f),
            Self::Report(error) => fmt::Display::fmt(error, f),
            Self::Publish(error) => fmt::Display::fmt(error, f),
            Self::Catalog(error) => fmt::Display::fmt(error, f),
            Self::Reuse(error) => fmt::Display::fmt(error, f),
            Self::Provenance(error) => fmt::Display::fmt(error, f),
            Self::Execution(error) => fmt::Display::fmt(error, f),
            Self::MissingFeatures { missing } => write!(
                f,
                "This build of vidr-remaster cannot run a video, because these features are \
                 off: {}. Rebuild with --features cuda,mp4,mkv,model-srvgg,model-rrdb, or with \
                 CPU codecs such as --features mp4,dav1d,cpu-av1.",
                missing.join(", ")
            ),
            Self::Stopped => f.write_str("The run stops on request."),
            Self::Resolution(error) => fmt::Display::fmt(error, f),
            Self::Time(error) => fmt::Display::fmt(error, f),
            Self::Timing(error) => fmt::Display::fmt(error, f),
            Self::Geometry(error) => fmt::Display::fmt(error, f),
            Self::Check(error) => write!(
                f,
                "{error}\nThe temporary output is removed and nothing is published. Check the \
                 source against the mismatches above."
            ),
            Self::File { path, error } => write!(
                f,
                "Cannot use {}: {error}. Check the path, its permissions, and free space.",
                path.display()
            ),
            Self::NoGpu => {
                f.write_str("CUDA is unavailable. Run from a terminal with NVIDIA GPU access.")
            }
            Self::Gpu { reason } => f.write_str(reason),
            #[cfg(any(feature = "mp4", feature = "mkv"))]
            Self::Scan(error) => fmt::Display::fmt(error, f),
            #[cfg(any(feature = "mp4", feature = "mkv"))]
            Self::Media(error) => fmt::Display::fmt(error, f),
            #[cfg(any(feature = "mp4", feature = "mkv"))]
            Self::Color(error) => fmt::Display::fmt(error, f),
            #[cfg(any(feature = "mp4", feature = "mkv"))]
            Self::Decode { path, error } => write!(
                f,
                "Frame scan failed for {}: {error} Check the source file.",
                path.display()
            ),
            #[cfg(any(feature = "mp4", feature = "mkv"))]
            Self::Encode(error) => fmt::Display::fmt(error, f),
            #[cfg(any(feature = "mp4", feature = "mkv"))]
            Self::H264TenBit => f.write_str(
                "This hardware encoder does not encode H.264 at 10 bits, and this video is \
                 10-bit. Pass --codec hevc.",
            ),
            #[cfg(any(feature = "mp4", feature = "mkv"))]
            Self::Openh264TenBit => f.write_str(
                "OpenH264 encodes 8 bits only, and this video is 10-bit. Pass --codec av1 with \
                 the cpu-av1 feature.",
            ),
            #[cfg(feature = "transform")]
            Self::Pipeline(error) => fmt::Display::fmt(error, f),
            #[cfg(feature = "transform")]
            Self::Transform(error) => fmt::Display::fmt(error, f),
        }
    }
}

impl std::error::Error for RemasterError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Request(error) => Some(error),
            Self::Report(error) => Some(error),
            Self::Publish(error) => Some(error),
            Self::Catalog(error) => Some(error),
            Self::Reuse(error) => Some(error),
            Self::Provenance(error) => Some(error),
            Self::Execution(error) => Some(error),
            Self::Resolution(error) => Some(error),
            Self::Time(error) => Some(error),
            Self::Timing(error) => Some(error),
            Self::Geometry(error) => Some(error),
            Self::Check(error) => Some(error),
            Self::File { error, .. } => Some(error),
            #[cfg(any(feature = "mp4", feature = "mkv"))]
            Self::Scan(error) => Some(error),
            #[cfg(any(feature = "mp4", feature = "mkv"))]
            Self::Media(error) => Some(error),
            #[cfg(any(feature = "mp4", feature = "mkv"))]
            Self::Color(error) => Some(error),
            #[cfg(any(feature = "mp4", feature = "mkv"))]
            Self::Decode { error, .. } => Some(error),
            #[cfg(any(feature = "mp4", feature = "mkv"))]
            Self::Encode(error) => Some(error),
            #[cfg(feature = "transform")]
            Self::Pipeline(error) => Some(error),
            #[cfg(feature = "transform")]
            Self::Transform(error) => Some(error),
            _ => None,
        }
    }
}

macro_rules! from_error {
    ($($(#[$gate:meta])* $variant:ident($error:ty),)*) => {
        $(
            $(#[$gate])*
            impl From<$error> for RemasterError {
                fn from(error: $error) -> Self {
                    Self::$variant(error)
                }
            }
        )*
    };
}

from_error! {
    Request(RequestError),
    Report(ReportError),
    Publish(PublishError),
    Catalog(CatalogError),
    Reuse(ReuseError),
    Provenance(ProvenanceError),
    Execution(ExecutionError),
    Resolution(ResolutionError),
    Time(TimeError),
    Timing(TimingError),
    Geometry(GeometryError),
    Check(CheckError),
    #[cfg(any(feature = "mp4", feature = "mkv"))]
    Scan(ScanError),
    #[cfg(any(feature = "mp4", feature = "mkv"))]
    Media(MediaError),
    #[cfg(any(feature = "mp4", feature = "mkv"))]
    Color(ColorError),
    #[cfg(any(feature = "mp4", feature = "mkv"))]
    Encode(EncodeError),
    #[cfg(feature = "transform")]
    Pipeline(PipelineError),
    #[cfg(feature = "transform")]
    Transform(TransformError),
}
