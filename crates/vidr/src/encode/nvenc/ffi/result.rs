//! `NVENCSTATUS` codes to [`EncodeError`].

use super::types::NvEncStatus;
use crate::error::EncodeError;

/// `NV_ENC_SUCCESS`.
pub(crate) const SUCCESS: NvEncStatus = 0;
/// `NV_ENC_ERR_OUT_OF_MEMORY`.
pub(crate) const ERR_OUT_OF_MEMORY: NvEncStatus = 10;
/// `NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY`.
pub(crate) const ERR_INCOMPATIBLE_CLIENT_KEY: NvEncStatus = 21;
/// `NV_ENC_ERR_NEED_MORE_INPUT`. In synchronous mode this is no error: the
/// encoder holds the picture until a later one completes its reorder group.
pub(crate) const NEED_MORE_INPUT: NvEncStatus = 17;

/// The `NVENCSTATUS` name of a code, in the header's order.
pub(crate) const fn status_name(code: NvEncStatus) -> &'static str {
    match code {
        0 => "NV_ENC_SUCCESS",
        1 => "NV_ENC_ERR_NO_ENCODE_DEVICE",
        2 => "NV_ENC_ERR_UNSUPPORTED_DEVICE",
        3 => "NV_ENC_ERR_INVALID_ENCODERDEVICE",
        4 => "NV_ENC_ERR_INVALID_DEVICE",
        5 => "NV_ENC_ERR_DEVICE_NOT_EXIST",
        6 => "NV_ENC_ERR_INVALID_PTR",
        7 => "NV_ENC_ERR_INVALID_EVENT",
        8 => "NV_ENC_ERR_INVALID_PARAM",
        9 => "NV_ENC_ERR_INVALID_CALL",
        10 => "NV_ENC_ERR_OUT_OF_MEMORY",
        11 => "NV_ENC_ERR_ENCODER_NOT_INITIALIZED",
        12 => "NV_ENC_ERR_UNSUPPORTED_PARAM",
        13 => "NV_ENC_ERR_LOCK_BUSY",
        14 => "NV_ENC_ERR_NOT_ENOUGH_BUFFER",
        15 => "NV_ENC_ERR_INVALID_VERSION",
        16 => "NV_ENC_ERR_MAP_FAILED",
        17 => "NV_ENC_ERR_NEED_MORE_INPUT",
        18 => "NV_ENC_ERR_ENCODER_BUSY",
        19 => "NV_ENC_ERR_EVENT_NOT_REGISTERD",
        20 => "NV_ENC_ERR_GENERIC",
        21 => "NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY",
        22 => "NV_ENC_ERR_UNIMPLEMENTED",
        23 => "NV_ENC_ERR_RESOURCE_REGISTER_FAILED",
        24 => "NV_ENC_ERR_RESOURCE_NOT_REGISTERED",
        25 => "NV_ENC_ERR_RESOURCE_NOT_MAPPED",
        26 => "NV_ENC_ERR_NEED_MORE_OUTPUT",
        _ => "unknown",
    }
}

/// `Ok` for `NV_ENC_SUCCESS`, else [`EncodeError::Nvenc`] with the call name,
/// the inputs `input` describes, and the driver text `detail` returns.
pub(crate) fn check(
    call: &'static str,
    code: NvEncStatus,
    input: impl FnOnce() -> String,
    detail: impl FnOnce() -> String,
) -> Result<(), EncodeError> {
    if code == SUCCESS {
        Ok(())
    } else {
        Err(EncodeError::Nvenc {
            call,
            code,
            name: status_name(code),
            input: input(),
            detail: detail(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn success_passes_and_errors_name_the_call() {
        assert_eq!(
            check("nvEncEncodePicture", 0, String::new, String::new),
            Ok(())
        );
        let error = check(
            "nvEncRegisterResource",
            23,
            || "640x360 NV12".to_owned(),
            || "driver text".to_owned(),
        );
        let message = error.map_err(|error| error.to_string()).err();
        let message = message.unwrap_or_default();
        assert!(message.contains("nvEncRegisterResource returns NVENCSTATUS 23"));
        assert!(message.contains("NV_ENC_ERR_RESOURCE_REGISTER_FAILED"));
        assert!(message.contains("640x360 NV12"));
        assert!(message.contains("driver text"));
    }

    #[test]
    fn status_names_follow_the_header_order() {
        assert_eq!(status_name(NEED_MORE_INPUT), "NV_ENC_ERR_NEED_MORE_INPUT");
        assert_eq!(status_name(20), "NV_ENC_ERR_GENERIC");
        assert_eq!(status_name(26), "NV_ENC_ERR_NEED_MORE_OUTPUT");
        assert_eq!(status_name(27), "unknown");
    }
}
