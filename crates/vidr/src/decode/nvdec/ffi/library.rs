//! `libnvcuvid` loaded at run time, with the functions vidr calls.

use std::ffi::{c_int, c_uint};
use std::sync::OnceLock;

use libloading::Library;

use super::types::{
    CuContext, CuResult, CuVideoCtxLock, CuVideoDecoder, CuVideoParser, CuvidDecodeCaps,
    CuvidDecodeCreateInfo, CuvidParserParams, CuvidPicParams, CuvidProcParams,
    CuvidSourceDataPacket,
};
use crate::error::DecodeError;

/// The NVDEC library file. The NVIDIA driver installs it.
#[cfg(windows)]
pub(crate) const LIBRARY: &str = "nvcuvid.dll";
/// The NVDEC library file. The NVIDIA driver installs it.
#[cfg(not(windows))]
pub(crate) const LIBRARY: &str = "libnvcuvid.so.1";

/// The NVDEC entry points, resolved once per process.
///
/// Each field has the signature of the `tcuvid*` typedef of the same name
/// in `dynlink_cuviddec.h` or `dynlink_nvcuvid.h`. `CUDAAPI` is the C
/// calling convention on every 64-bit target.
pub(crate) struct Cuvid {
    pub(crate) get_decoder_caps: unsafe extern "C" fn(*mut CuvidDecodeCaps) -> CuResult,
    pub(crate) create_decoder:
        unsafe extern "C" fn(*mut CuVideoDecoder, *mut CuvidDecodeCreateInfo) -> CuResult,
    pub(crate) destroy_decoder: unsafe extern "C" fn(CuVideoDecoder) -> CuResult,
    pub(crate) decode_picture:
        unsafe extern "C" fn(CuVideoDecoder, *mut CuvidPicParams) -> CuResult,
    pub(crate) map_video_frame: unsafe extern "C" fn(
        CuVideoDecoder,
        c_int,
        *mut u64,
        *mut c_uint,
        *mut CuvidProcParams,
    ) -> CuResult,
    pub(crate) unmap_video_frame: unsafe extern "C" fn(CuVideoDecoder, u64) -> CuResult,
    pub(crate) create_video_parser:
        unsafe extern "C" fn(*mut CuVideoParser, *mut CuvidParserParams) -> CuResult,
    pub(crate) parse_video_data:
        unsafe extern "C" fn(CuVideoParser, *mut CuvidSourceDataPacket) -> CuResult,
    pub(crate) destroy_video_parser: unsafe extern "C" fn(CuVideoParser) -> CuResult,
    pub(crate) ctx_lock_create: unsafe extern "C" fn(*mut CuVideoCtxLock, CuContext) -> CuResult,
    pub(crate) ctx_lock_destroy: unsafe extern "C" fn(CuVideoCtxLock) -> CuResult,
    /// Keeps the library mapped, so every pointer above stays valid. The
    /// `OnceLock` in [`cuvid`] never drops it.
    _library: Library,
}

/// The process-wide library, loaded on first use.
///
/// # Errors
///
/// [`DecodeError::LibraryMissing`] when the library does not load, and
/// [`DecodeError::SymbolMissing`] when it lacks a function. The first
/// result is kept, so a later call returns the same error.
pub(crate) fn cuvid() -> Result<&'static Cuvid, DecodeError> {
    static CUVID: OnceLock<Result<Cuvid, DecodeError>> = OnceLock::new();
    CUVID.get_or_init(load).as_ref().map_err(Clone::clone)
}

fn load() -> Result<Cuvid, DecodeError> {
    // SAFETY: loading runs the library's initialisers. libnvcuvid is the
    // NVIDIA driver's own library, whose initialisers have no preconditions
    // on the caller.
    let library =
        unsafe { Library::new(LIBRARY) }.map_err(|error| DecodeError::LibraryMissing {
            library: LIBRARY,
            reason: error.to_string(),
        })?;

    // Resolves `$name` as the function pointer type of the field it fills.
    macro_rules! symbol {
        ($name:literal) => {{
            // SAFETY: the field's type is the `tcuvid*` typedef of `$name`
            // in the NVIDIA headers, so the pointer has the right signature.
            // The pointer stays valid while `library` stays loaded, which
            // `Cuvid` guarantees by owning it for the life of the process.
            let symbol =
                unsafe { library.get($name) }.map_err(|error| DecodeError::SymbolMissing {
                    library: LIBRARY,
                    symbol: $name,
                    reason: error.to_string(),
                })?;
            *symbol
        }};
    }

    Ok(Cuvid {
        get_decoder_caps: symbol!("cuvidGetDecoderCaps"),
        create_decoder: symbol!("cuvidCreateDecoder"),
        destroy_decoder: symbol!("cuvidDestroyDecoder"),
        decode_picture: symbol!("cuvidDecodePicture"),
        map_video_frame: symbol!("cuvidMapVideoFrame64"),
        unmap_video_frame: symbol!("cuvidUnmapVideoFrame64"),
        create_video_parser: symbol!("cuvidCreateVideoParser"),
        parse_video_data: symbol!("cuvidParseVideoData"),
        destroy_video_parser: symbol!("cuvidDestroyVideoParser"),
        ctx_lock_create: symbol!("cuvidCtxLockCreate"),
        ctx_lock_destroy: symbol!("cuvidCtxLockDestroy"),
        _library: library,
    })
}
