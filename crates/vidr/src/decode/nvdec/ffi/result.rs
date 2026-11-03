//! `CUresult` codes to [`DecodeError`].

use super::types::{CUDA_SUCCESS, CuResult};
use crate::error::DecodeError;

/// The `cudaError_enum` name of the codes NVDEC and CUDA driver calls
/// return, from the CUDA toolkit's `cuda.h`.
pub(crate) const fn result_name(code: CuResult) -> &'static str {
    match code {
        0 => "CUDA_SUCCESS",
        1 => "CUDA_ERROR_INVALID_VALUE",
        2 => "CUDA_ERROR_OUT_OF_MEMORY",
        3 => "CUDA_ERROR_NOT_INITIALIZED",
        4 => "CUDA_ERROR_DEINITIALIZED",
        100 => "CUDA_ERROR_NO_DEVICE",
        101 => "CUDA_ERROR_INVALID_DEVICE",
        201 => "CUDA_ERROR_INVALID_CONTEXT",
        205 => "CUDA_ERROR_MAP_FAILED",
        206 => "CUDA_ERROR_UNMAP_FAILED",
        209 => "CUDA_ERROR_NO_BINARY_FOR_GPU",
        214 => "CUDA_ERROR_ECC_UNCORRECTABLE",
        400 => "CUDA_ERROR_INVALID_HANDLE",
        500 => "CUDA_ERROR_NOT_FOUND",
        600 => "CUDA_ERROR_NOT_READY",
        700 => "CUDA_ERROR_ILLEGAL_ADDRESS",
        719 => "CUDA_ERROR_LAUNCH_FAILED",
        801 => "CUDA_ERROR_NOT_SUPPORTED",
        999 => "CUDA_ERROR_UNKNOWN",
        _ => "unknown",
    }
}

/// `Ok` for `CUDA_SUCCESS`, else [`DecodeError::Driver`] with the call
/// name and the inputs `input` describes.
pub(crate) fn check(
    call: &'static str,
    code: CuResult,
    input: impl FnOnce() -> String,
) -> Result<(), DecodeError> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(DecodeError::Driver {
            call,
            code,
            name: result_name(code),
            input: input(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn success_passes_and_errors_name_the_call() {
        assert_eq!(check("cuvidCreateDecoder", 0, String::new), Ok(()));
        let error = check("cuvidCreateDecoder", 801, || "HEVC 10-bit".to_owned());
        let message = error.map_err(|error| error.to_string()).err();
        let message = message.unwrap_or_default();
        assert!(message.contains("cuvidCreateDecoder returns CUresult 801"));
        assert!(message.contains("CUDA_ERROR_NOT_SUPPORTED"));
        assert!(message.contains("HEVC 10-bit"));
    }

    #[test]
    fn unnamed_codes_say_unknown() {
        assert_eq!(result_name(12345), "unknown");
    }
}
