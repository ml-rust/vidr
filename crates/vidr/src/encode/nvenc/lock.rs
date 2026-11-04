//! Reading one picture's output: `nvEncLockBitstream` to `nvEncUnlockBitstream`.

use super::ffi::{LOCK_BITSTREAM_VER, NvEncLockBitstream, NvEncOutputPtr, PIC_TYPE_IDR};
use super::session::Session;
use crate::error::EncodeError;

/// The facts the lock reports about a picture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PictureFacts {
    pub(super) pts: i64,
    pub(super) duration: i64,
    pub(super) keyframe: bool,
}

fn lock(session: &Session, bitstream: NvEncOutputPtr) -> (NvEncLockBitstream, i32) {
    let mut lock = NvEncLockBitstream::zeroed();
    lock.version = LOCK_BITSTREAM_VER;
    lock.output_bitstream = bitstream;
    // SAFETY: the session is live, `bitstream` belongs to it and holds a
    // submitted picture, and `lock` is fully initialised with zeroed reserved
    // fields. Waiting is the default, so the output is complete on success.
    let code = unsafe { (session.nvenc.lock_bitstream)(session.handle, &mut lock) };
    (lock, code)
}

/// Waits for the picture in `bitstream`, passes its bytes to `read`, and
/// unlocks. The bytes are the one host copy on the encode path: the compressed
/// bitstream.
pub(super) fn read_output<T>(
    session: &Session,
    bitstream: NvEncOutputPtr,
    read: impl FnOnce(&[u8]) -> T,
) -> Result<(T, PictureFacts), EncodeError> {
    let describe = || format!("output buffer {bitstream:?}");
    let (locked, code) = lock(session, bitstream);
    session.check("nvEncLockBitstream", code, describe)?;
    // SAFETY: the lock returned success, so `bitstream_buffer_ptr` points to
    // `bitstream_size_in_bytes` readable bytes until the unlock below.
    let output = unsafe {
        std::slice::from_raw_parts(
            locked.bitstream_buffer_ptr.cast::<u8>().cast_const(),
            locked.bitstream_size_in_bytes as usize,
        )
    };
    let value = read(output);
    // SAFETY: `bitstream` is locked by the call above, and `output` is not used
    // after this.
    let code = unsafe { (session.nvenc.unlock_bitstream)(session.handle, bitstream) };
    session.check("nvEncUnlockBitstream", code, describe)?;
    Ok((
        value,
        PictureFacts {
            // NVENC echoes the opaque bits `send` passed, so a negative pts
            // survives.
            pts: locked.output_time_stamp as i64,
            duration: locked.output_duration as i64,
            keyframe: locked.picture_type == PIC_TYPE_IDR,
        },
    ))
}

/// Waits for the picture in `bitstream` and drops its bytes. A driver error here
/// has no caller to reach, so this ignores it.
pub(super) fn discard_output(session: &Session, bitstream: NvEncOutputPtr) {
    let (_, code) = lock(session, bitstream);
    if code == 0 {
        // SAFETY: the lock above returned success on this buffer.
        let _ = unsafe { (session.nvenc.unlock_bitstream)(session.handle, bitstream) };
    }
}
