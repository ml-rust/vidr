//! The CPU codecs, on any numr runtime. Each declares its host copy through
//! `FrameRoute`, and the pump counts it.
//!
//! `execution_for` chooses only what this build holds. So the arm of a feature that
//! is off returns the same refusal, for a caller that skips the choice.

use std::path::Path;

use numr::dtype::DType;
use numr::runtime::Runtime;
use vidr::decode::Decoder;
use vidr::encode::{EncodeConfig, Encoder};
use vidr::media::DecodeConfig;

use crate::error::RemasterError;
use crate::execution::{DecoderBackend, EncoderBackend, ExecutionError};

/// A decoder of any backend, for the pump's `D`.
pub(super) type BoxDecoder<R> = Box<dyn Decoder<R> + Send>;
/// An encoder of any backend, for the pump's `E`.
pub(super) type BoxEncoder<R> = Box<dyn Encoder<R> + Send>;

/// The refusal of a codec this build leaves out.
#[allow(
    dead_code,
    reason = "a build with every CPU codec has no arm that calls this"
)]
fn not_built(what: &str, feature: &'static str) -> RemasterError {
    ExecutionError::NotBuilt {
        what: what.to_owned(),
        feature,
    }
    .into()
}

/// The refusal of a hardware codec on a runtime with no path for it.
pub(super) fn wrong_runtime<R: Runtime>(backend: &'static str) -> RemasterError {
    ExecutionError::WrongRuntime {
        backend,
        runtime: R::name(),
    }
    .into()
}

pub(super) fn decode_error(path: &Path, error: vidr::DecodeError) -> RemasterError {
    RemasterError::Decode {
        path: path.to_path_buf(),
        error,
    }
}

/// Opens the CPU decoder `backend`, whose frames go to tensors on `device`.
///
/// # Errors
///
/// [`RemasterError::Decode`] when the decoder refuses the track's config, and
/// [`RemasterError::Execution`] for a backend this build or runtime leaves out.
#[allow(
    unused_variables,
    reason = "a build with no CPU decoder reads no argument"
)]
pub(super) fn open_cpu_decoder<R>(
    device: &R::Device,
    library: Option<&Path>,
    backend: DecoderBackend,
    decode: &DecodeConfig,
    path: &Path,
) -> Result<BoxDecoder<R>, RemasterError>
where
    R: Runtime<DType = DType>,
{
    let failed = |error| decode_error(path, error);
    match backend {
        DecoderBackend::Nvdec | DecoderBackend::Vulkan => Err(wrong_runtime::<R>(backend.name())),
        #[cfg(feature = "dav1d")]
        DecoderBackend::Dav1d => {
            let decoder = vidr::decode::dav1d::Dav1dDecoder::<R>::new(device, &decode.config);
            Ok(Box::new(decoder.map_err(failed)?))
        }
        #[cfg(not(feature = "dav1d"))]
        DecoderBackend::Dav1d => Err(not_built("AV1 CPU decode", "dav1d")),
        #[cfg(feature = "cpu-vp9")]
        DecoderBackend::RustyVp9 => Ok(Box::new(
            vidr::decode::rusty_vp9::RustyVp9Decoder::<R>::new(device),
        )),
        #[cfg(not(feature = "cpu-vp9"))]
        DecoderBackend::RustyVp9 => Err(not_built("VP9 CPU decode", "cpu-vp9")),
        #[cfg(feature = "cpu-hevc")]
        DecoderBackend::Hevc => {
            let decoder = vidr::decode::hevc::HevcDecoder::<R>::new(device, &decode.config);
            Ok(Box::new(decoder.map_err(failed)?))
        }
        #[cfg(not(feature = "cpu-hevc"))]
        DecoderBackend::Hevc => Err(not_built("HEVC CPU decode", "cpu-hevc")),
        #[cfg(feature = "openh264")]
        DecoderBackend::Openh264(crate::execution::Openh264Build::Source) => {
            let decoder = vidr::decode::openh264::Openh264Decoder::<R>::new(device, &decode.config);
            Ok(Box::new(decoder.map_err(failed)?))
        }
        #[cfg(not(feature = "openh264"))]
        DecoderBackend::Openh264(crate::execution::Openh264Build::Source) => {
            Err(not_built("H.264 CPU decode", "openh264"))
        }
        #[cfg(feature = "openh264-cisco")]
        DecoderBackend::Openh264(crate::execution::Openh264Build::Cisco) => {
            let Some(library) = library else {
                return Err(ExecutionError::CiscoLibraryMissing.into());
            };
            let decoder = vidr::decode::openh264::Openh264Decoder::<R>::with_cisco_binary(
                device,
                &decode.config,
                library,
            );
            Ok(Box::new(decoder.map_err(failed)?))
        }
        #[cfg(not(feature = "openh264-cisco"))]
        DecoderBackend::Openh264(crate::execution::Openh264Build::Cisco) => {
            Err(not_built("H.264 CPU decode", "openh264-cisco"))
        }
    }
}

/// Opens the CPU encoder `backend`.
///
/// # Errors
///
/// `plan::encoder_open_error` for a config the encoder refuses, and
/// [`RemasterError::Execution`] for a backend this build or runtime leaves out.
#[allow(
    unused_variables,
    reason = "a build with no CPU encoder reads no argument"
)]
pub(super) fn open_cpu_encoder<R>(
    library: Option<&Path>,
    backend: EncoderBackend,
    config: EncodeConfig,
) -> Result<BoxEncoder<R>, RemasterError>
where
    R: Runtime<DType = DType>,
{
    match backend {
        EncoderBackend::Nvenc => Err(wrong_runtime::<R>(backend.name())),
        #[cfg(feature = "cpu-av1")]
        EncoderBackend::Rav1r => {
            let encoder = vidr::encode::rav1r::Rav1rEncoder::<R>::new(config);
            Ok(Box::new(encoder.map_err(crate::plan::encoder_open_error)?))
        }
        #[cfg(not(feature = "cpu-av1"))]
        EncoderBackend::Rav1r => Err(not_built("AV1 CPU encode", "cpu-av1")),
        #[cfg(feature = "openh264")]
        EncoderBackend::Openh264(crate::execution::Openh264Build::Source) => {
            let encoder = vidr::encode::openh264::Openh264Encoder::<R>::new(config);
            Ok(Box::new(encoder.map_err(crate::plan::encoder_open_error)?))
        }
        #[cfg(not(feature = "openh264"))]
        EncoderBackend::Openh264(crate::execution::Openh264Build::Source) => {
            Err(not_built("H.264 CPU encode", "openh264"))
        }
        #[cfg(feature = "openh264-cisco")]
        EncoderBackend::Openh264(crate::execution::Openh264Build::Cisco) => {
            let Some(library) = library else {
                return Err(ExecutionError::CiscoLibraryMissing.into());
            };
            let encoder =
                vidr::encode::openh264::Openh264Encoder::<R>::with_cisco_binary(config, library);
            Ok(Box::new(encoder.map_err(crate::plan::encoder_open_error)?))
        }
        #[cfg(not(feature = "openh264-cisco"))]
        EncoderBackend::Openh264(crate::execution::Openh264Build::Cisco) => {
            Err(not_built("H.264 CPU encode", "openh264-cisco"))
        }
    }
}
