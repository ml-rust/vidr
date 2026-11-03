//! The parser callbacks. Each one catches panics and errors, stores the
//! first in the session, and returns 0 so the parser stops. No unwind
//! crosses into the driver.

use std::any::Any;
use std::ffi::{c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};

use super::ffi::{CuVideoFormat, CuvidParserDispInfo, CuvidPicParams};
use super::session::Session;
use crate::error::DecodeError;

/// Callback bodies over some state that can hold an error.
pub(super) trait Fallible {
    /// True when an error is stored.
    fn failed(&self) -> bool;
    /// Stores `error` unless an error is stored already.
    fn fail(&mut self, error: DecodeError);
}

impl Fallible for Session {
    fn failed(&self) -> bool {
        self.error.is_some()
    }

    fn fail(&mut self, error: DecodeError) {
        if self.error.is_none() {
            self.error = Some(error);
        }
    }
}

/// Runs `body` on the state behind `user`, and turns an error or a panic
/// into a stored error and a return value of 0.
///
/// # Safety
///
/// `user` is null, or points to a live `S` that nothing else borrows for
/// the duration of the call.
pub(super) unsafe fn guarded<S: Fallible>(
    user: *mut c_void,
    callback: &'static str,
    body: impl FnOnce(&mut S) -> Result<c_int, DecodeError>,
) -> c_int {
    let state = user.cast::<S>();
    if state.is_null() {
        return 0;
    }
    // SAFETY: the caller guarantees `state` points to a live `S` with no
    // other borrow. The reference ends before the next one below.
    if unsafe { &*state }.failed() {
        return 0;
    }
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: as above. This is the only reference during `body`.
        body(unsafe { &mut *state })
    }));
    let error = match outcome {
        Ok(Ok(value)) => return value,
        Ok(Err(error)) => error,
        Err(payload) => DecodeError::CallbackPanic {
            callback,
            message: panic_message(payload.as_ref()),
        },
    };
    // SAFETY: as above. `body` and its reference have ended.
    unsafe { &mut *state }.fail(error);
    0
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "a non-string panic payload".to_owned()
    }
}

/// Borrows the parser's argument, or returns `NullCallbackArgument`.
///
/// # Safety
///
/// `pointer` is null, or points to a live `T` for `'a`.
unsafe fn non_null<'a, T>(
    pointer: *mut T,
    callback: &'static str,
    argument: &'static str,
) -> Result<&'a T, DecodeError> {
    // SAFETY: the caller guarantees the pointer is null or valid for `'a`.
    unsafe { pointer.as_ref() }.ok_or(DecodeError::NullCallbackArgument { callback, argument })
}

/// `pfnSequenceCallback`.
///
/// # Safety
///
/// The parser calls it with the `Session` pointer it was created with and
/// a valid `CUVIDEOFORMAT`.
pub(super) unsafe extern "C" fn on_sequence(
    user: *mut c_void,
    format: *mut CuVideoFormat,
) -> c_int {
    let body = |session: &mut Session| {
        // SAFETY: the parser owns `format` and keeps it valid for this call.
        // A null pointer is refused.
        let format = unsafe { non_null(format, "sequence", "format") }?;
        session.on_sequence(format)
    };
    // SAFETY: the parser passes the session it was created with, and only
    // from inside `cuvidParseVideoData`, while `NvdecDecoder` holds no
    // reference to it.
    unsafe { guarded(user, "sequence", body) }
}

/// `pfnDecodePicture`.
///
/// # Safety
///
/// As [`on_sequence`], with a valid `CUVIDPICPARAMS`.
pub(super) unsafe extern "C" fn on_decode(
    user: *mut c_void,
    picture: *mut CuvidPicParams,
) -> c_int {
    // SAFETY: as in `on_sequence`.
    unsafe {
        guarded(user, "decode", |session: &mut Session| {
            // The parser owns `picture`. Only the null check uses it here.
            non_null(picture, "decode", "picture")?;
            session.on_decode(picture)
        })
    }
}

/// `pfnDisplayPicture`. A null `info` marks the end of the stream, which
/// only `CUVID_PKT_NOTIFY_EOS` requests and vidr never sets.
///
/// # Safety
///
/// As [`on_sequence`], with a valid or null `CUVIDPARSERDISPINFO`.
pub(super) unsafe extern "C" fn on_display(
    user: *mut c_void,
    info: *mut CuvidParserDispInfo,
) -> c_int {
    let body = |session: &mut Session| {
        // SAFETY: the parser owns `info` and keeps it valid for this call.
        match unsafe { info.as_ref() } {
            Some(info) => session.on_display(info),
            None => Ok(1),
        }
    };
    // SAFETY: as in `on_sequence`.
    unsafe { guarded(user, "display", body) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Probe {
        error: Option<DecodeError>,
        calls: u32,
    }

    impl Fallible for Probe {
        fn failed(&self) -> bool {
            self.error.is_some()
        }

        fn fail(&mut self, error: DecodeError) {
            if self.error.is_none() {
                self.error = Some(error);
            }
        }
    }

    fn call(
        probe: &mut Probe,
        body: impl FnOnce(&mut Probe) -> Result<c_int, DecodeError>,
    ) -> c_int {
        let user = std::ptr::from_mut(probe).cast::<c_void>();
        // SAFETY: `user` points to `probe`, which the closure alone borrows
        // during the call.
        unsafe { guarded(user, "test", body) }
    }

    #[test]
    fn success_returns_the_body_value() {
        let mut probe = Probe::default();
        assert_eq!(
            call(&mut probe, |p| {
                p.calls += 1;
                Ok(7)
            }),
            7
        );
        assert_eq!(probe.calls, 1);
        assert_eq!(probe.error, None);
    }

    #[test]
    fn an_error_is_stored_and_later_calls_do_nothing() {
        let mut probe = Probe::default();
        let first = call(&mut probe, |_| Err(DecodeError::EmptyPacket { pts: 3 }));
        assert_eq!(first, 0);
        let second = call(&mut probe, |p| {
            p.calls += 1;
            Ok(1)
        });
        assert_eq!(second, 0);
        assert_eq!(probe.calls, 0);
        assert_eq!(probe.error, Some(DecodeError::EmptyPacket { pts: 3 }));
    }

    #[test]
    fn a_panic_is_caught_and_stored() {
        let mut probe = Probe::default();
        let result = call(&mut probe, |_| panic!("surface lost"));
        assert_eq!(result, 0);
        assert_eq!(
            probe.error,
            Some(DecodeError::CallbackPanic {
                callback: "test",
                message: "surface lost".to_owned(),
            })
        );
    }

    #[test]
    fn a_null_argument_is_refused_with_its_name() {
        let null: *mut u32 = std::ptr::null_mut();
        // SAFETY: a null pointer is allowed.
        let error = unsafe { non_null(null, "sequence", "format") };
        assert_eq!(
            error,
            Err(DecodeError::NullCallbackArgument {
                callback: "sequence",
                argument: "format",
            })
        );
    }

    #[test]
    fn a_valid_argument_is_borrowed() {
        let mut value = 5_u32;
        // SAFETY: the pointer is to a live local.
        let borrowed = unsafe { non_null(&raw mut value, "decode", "picture") };
        assert_eq!(borrowed, Ok(&5));
    }

    #[test]
    fn a_null_user_pointer_returns_zero() {
        // SAFETY: a null pointer is allowed.
        let result = unsafe { guarded::<Probe>(std::ptr::null_mut(), "test", |_| Ok(1)) };
        assert_eq!(result, 0);
    }
}
