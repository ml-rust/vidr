//! `libnvidia-encode` loaded at run time, with the functions vidr calls.

use std::ffi::{c_char, c_int, c_void};
use std::sync::OnceLock;

use libloading::{Library, Symbol};

use super::config_types::NvEncPresetConfig;
use super::guid::Guid;
use super::result::SUCCESS;
use super::types::{
    API_MAJOR, API_MINOR, FUNCTION_LIST_VER, NvEncCapsParam, NvEncCreateBitstreamBuffer,
    NvEncHandle, NvEncInitializeParams, NvEncInputPtr, NvEncLockBitstream, NvEncMapInputResource,
    NvEncOpenSessionParams, NvEncOutputPtr, NvEncPicParams, NvEncRegisterResource,
    NvEncRegisteredPtr, NvEncStatus,
};
use crate::error::EncodeError;

/// The NVENC library file. The NVIDIA driver installs it.
#[cfg(windows)]
pub(crate) const LIBRARY: &str = "nvEncodeAPI64.dll";
/// The NVENC library file. The NVIDIA driver installs it.
#[cfg(not(windows))]
pub(crate) const LIBRARY: &str = "libnvidia-encode.so.1";

type GetEncodeGuidCount = unsafe extern "C" fn(NvEncHandle, *mut u32) -> NvEncStatus;
type GetEncodeGuids = unsafe extern "C" fn(NvEncHandle, *mut Guid, u32, *mut u32) -> NvEncStatus;
type GetEncodeCaps =
    unsafe extern "C" fn(NvEncHandle, Guid, *mut NvEncCapsParam, *mut c_int) -> NvEncStatus;
type GetEncodePresetConfigEx =
    unsafe extern "C" fn(NvEncHandle, Guid, Guid, c_int, *mut NvEncPresetConfig) -> NvEncStatus;
type InitializeEncoder =
    unsafe extern "C" fn(NvEncHandle, *mut NvEncInitializeParams) -> NvEncStatus;
type CreateBitstreamBuffer =
    unsafe extern "C" fn(NvEncHandle, *mut NvEncCreateBitstreamBuffer) -> NvEncStatus;
type DestroyBitstreamBuffer = unsafe extern "C" fn(NvEncHandle, NvEncOutputPtr) -> NvEncStatus;
type EncodePicture = unsafe extern "C" fn(NvEncHandle, *mut NvEncPicParams) -> NvEncStatus;
type LockBitstream = unsafe extern "C" fn(NvEncHandle, *mut NvEncLockBitstream) -> NvEncStatus;
type UnlockBitstream = unsafe extern "C" fn(NvEncHandle, NvEncOutputPtr) -> NvEncStatus;
type MapInputResource =
    unsafe extern "C" fn(NvEncHandle, *mut NvEncMapInputResource) -> NvEncStatus;
type UnmapInputResource = unsafe extern "C" fn(NvEncHandle, NvEncInputPtr) -> NvEncStatus;
type DestroyEncoder = unsafe extern "C" fn(NvEncHandle) -> NvEncStatus;
type OpenEncodeSessionEx =
    unsafe extern "C" fn(*mut NvEncOpenSessionParams, *mut NvEncHandle) -> NvEncStatus;
type RegisterResource =
    unsafe extern "C" fn(NvEncHandle, *mut NvEncRegisterResource) -> NvEncStatus;
type UnregisterResource = unsafe extern "C" fn(NvEncHandle, NvEncRegisteredPtr) -> NvEncStatus;
type GetLastErrorString = unsafe extern "C" fn(NvEncHandle) -> *const c_char;

/// `NV_ENCODE_API_FUNCTION_LIST`. `NvEncodeAPICreateInstance` fills it. Entries
/// vidr does not call stay untyped pointers.
#[repr(C)]
#[allow(dead_code, reason = "FFI mirrors keep every header field")]
pub(super) struct NvEncodeApiFunctionList {
    pub(super) version: u32,
    pub(super) reserved: u32,
    pub(super) open_encode_session: *mut c_void,
    pub(super) get_encode_guid_count: Option<GetEncodeGuidCount>,
    pub(super) get_encode_profile_guid_count: *mut c_void,
    pub(super) get_encode_profile_guids: *mut c_void,
    pub(super) get_encode_guids: Option<GetEncodeGuids>,
    pub(super) get_input_format_count: *mut c_void,
    pub(super) get_input_formats: *mut c_void,
    pub(super) get_encode_caps: Option<GetEncodeCaps>,
    pub(super) get_encode_preset_count: *mut c_void,
    pub(super) get_encode_preset_guids: *mut c_void,
    pub(super) get_encode_preset_config: *mut c_void,
    pub(super) initialize_encoder: Option<InitializeEncoder>,
    pub(super) create_input_buffer: *mut c_void,
    pub(super) destroy_input_buffer: *mut c_void,
    pub(super) create_bitstream_buffer: Option<CreateBitstreamBuffer>,
    pub(super) destroy_bitstream_buffer: Option<DestroyBitstreamBuffer>,
    pub(super) encode_picture: Option<EncodePicture>,
    pub(super) lock_bitstream: Option<LockBitstream>,
    pub(super) unlock_bitstream: Option<UnlockBitstream>,
    pub(super) lock_input_buffer: *mut c_void,
    pub(super) unlock_input_buffer: *mut c_void,
    pub(super) get_encode_stats: *mut c_void,
    pub(super) get_sequence_params: *mut c_void,
    pub(super) register_async_event: *mut c_void,
    pub(super) unregister_async_event: *mut c_void,
    pub(super) map_input_resource: Option<MapInputResource>,
    pub(super) unmap_input_resource: Option<UnmapInputResource>,
    pub(super) destroy_encoder: Option<DestroyEncoder>,
    pub(super) invalidate_ref_frames: *mut c_void,
    pub(super) open_encode_session_ex: Option<OpenEncodeSessionEx>,
    pub(super) register_resource: Option<RegisterResource>,
    pub(super) unregister_resource: Option<UnregisterResource>,
    pub(super) reconfigure_encoder: *mut c_void,
    pub(super) reserved1: *mut c_void,
    pub(super) create_mv_buffer: *mut c_void,
    pub(super) destroy_mv_buffer: *mut c_void,
    pub(super) run_motion_estimation_only: *mut c_void,
    pub(super) get_last_error_string: Option<GetLastErrorString>,
    pub(super) set_io_cuda_streams: *mut c_void,
    pub(super) get_encode_preset_config_ex: Option<GetEncodePresetConfigEx>,
    pub(super) get_sequence_param_ex: *mut c_void,
    pub(super) restore_encoder_state: *mut c_void,
    pub(super) lookahead_picture: *mut c_void,
    pub(super) reserved2: [*mut c_void; 275],
}

impl_zeroed!(NvEncodeApiFunctionList);

/// The NVENC entry points, resolved once per process.
///
/// Each field has the signature of the `PNVENC*` typedef of the same name in
/// `nvEncodeAPI.h`. `NVENCAPI` is the C calling convention on every 64-bit
/// target.
pub(crate) struct Nvenc {
    pub(crate) get_encode_guid_count: GetEncodeGuidCount,
    pub(crate) get_encode_guids: GetEncodeGuids,
    pub(crate) get_encode_caps: GetEncodeCaps,
    pub(crate) get_encode_preset_config_ex: GetEncodePresetConfigEx,
    pub(crate) initialize_encoder: InitializeEncoder,
    pub(crate) create_bitstream_buffer: CreateBitstreamBuffer,
    pub(crate) destroy_bitstream_buffer: DestroyBitstreamBuffer,
    pub(crate) encode_picture: EncodePicture,
    pub(crate) lock_bitstream: LockBitstream,
    pub(crate) unlock_bitstream: UnlockBitstream,
    pub(crate) map_input_resource: MapInputResource,
    pub(crate) unmap_input_resource: UnmapInputResource,
    pub(crate) destroy_encoder: DestroyEncoder,
    pub(crate) open_encode_session_ex: OpenEncodeSessionEx,
    pub(crate) register_resource: RegisterResource,
    pub(crate) unregister_resource: UnregisterResource,
    pub(crate) get_last_error_string: GetLastErrorString,
    /// Keeps the library mapped, so every pointer above stays valid. The
    /// `OnceLock` in [`nvenc`] never drops it.
    _library: Library,
}

/// The process-wide library, loaded on first use.
///
/// # Errors
///
/// [`EncodeError::LibraryMissing`] when the library does not load,
/// [`EncodeError::SymbolMissing`] when it lacks a function, and
/// [`EncodeError::DriverTooOld`] when the driver's API is older than the pinned
/// one. The first result is kept, so a later call returns the same error.
pub(crate) fn nvenc() -> Result<&'static Nvenc, EncodeError> {
    static NVENC: OnceLock<Result<Nvenc, EncodeError>> = OnceLock::new();
    NVENC.get_or_init(load).as_ref().map_err(Clone::clone)
}

/// Splits `NvEncodeAPIGetMaxSupportedVersion`'s value, `(major << 4) | minor`.
const fn split_version(version: u32) -> (u32, u32) {
    (version >> 4, version & 0xf)
}

/// Whether a driver at `found` serves the pinned API version.
const fn serves_pinned_api(found: (u32, u32)) -> bool {
    found.0 > API_MAJOR || (found.0 == API_MAJOR && found.1 >= API_MINOR)
}

fn symbol_error(symbol: &'static str, reason: String) -> EncodeError {
    EncodeError::SymbolMissing {
        library: LIBRARY,
        symbol,
        reason,
    }
}

fn load() -> Result<Nvenc, EncodeError> {
    // SAFETY: loading runs the library's initialisers. libnvidia-encode is the
    // NVIDIA driver's own library, whose initialisers have no preconditions on
    // the caller.
    let library =
        unsafe { Library::new(LIBRARY) }.map_err(|error| EncodeError::LibraryMissing {
            library: LIBRARY,
            reason: error.to_string(),
        })?;

    let mut version = 0_u32;
    {
        const NAME: &str = "NvEncodeAPIGetMaxSupportedVersion";
        // SAFETY: the type is the header's declaration of this function.
        let get_version: Symbol<unsafe extern "C" fn(*mut u32) -> NvEncStatus> =
            unsafe { library.get(NAME) }.map_err(|error| symbol_error(NAME, error.to_string()))?;
        // SAFETY: `version` is a valid, writable `uint32_t`.
        let status = unsafe { get_version(&mut version) };
        if status != SUCCESS {
            return Err(symbol_error(NAME, format!("it returns status {status}")));
        }
    }
    let found = split_version(version);
    if !serves_pinned_api(found) {
        return Err(EncodeError::DriverTooOld {
            found,
            required: (API_MAJOR, API_MINOR),
        });
    }

    let mut list = NvEncodeApiFunctionList::zeroed();
    list.version = FUNCTION_LIST_VER;
    {
        const NAME: &str = "NvEncodeAPICreateInstance";
        // SAFETY: the type is the header's declaration of this function.
        let create: Symbol<unsafe extern "C" fn(*mut NvEncodeApiFunctionList) -> NvEncStatus> =
            unsafe { library.get(NAME) }.map_err(|error| symbol_error(NAME, error.to_string()))?;
        // SAFETY: `list` is a zeroed `NV_ENCODE_API_FUNCTION_LIST` with its
        // version set, and `layout.rs` checks its layout against the header.
        let status = unsafe { create(&mut list) };
        if status != SUCCESS {
            return Err(symbol_error(NAME, format!("it returns status {status}")));
        }
    }

    // Takes one entry from the filled list.
    macro_rules! entry {
        ($field:ident, $name:literal) => {
            list.$field.ok_or_else(|| {
                symbol_error(
                    $name,
                    "the driver leaves its function list entry null".to_owned(),
                )
            })?
        };
    }

    Ok(Nvenc {
        get_encode_guid_count: entry!(get_encode_guid_count, "nvEncGetEncodeGUIDCount"),
        get_encode_guids: entry!(get_encode_guids, "nvEncGetEncodeGUIDs"),
        get_encode_caps: entry!(get_encode_caps, "nvEncGetEncodeCaps"),
        get_encode_preset_config_ex: entry!(
            get_encode_preset_config_ex,
            "nvEncGetEncodePresetConfigEx"
        ),
        initialize_encoder: entry!(initialize_encoder, "nvEncInitializeEncoder"),
        create_bitstream_buffer: entry!(create_bitstream_buffer, "nvEncCreateBitstreamBuffer"),
        destroy_bitstream_buffer: entry!(destroy_bitstream_buffer, "nvEncDestroyBitstreamBuffer"),
        encode_picture: entry!(encode_picture, "nvEncEncodePicture"),
        lock_bitstream: entry!(lock_bitstream, "nvEncLockBitstream"),
        unlock_bitstream: entry!(unlock_bitstream, "nvEncUnlockBitstream"),
        map_input_resource: entry!(map_input_resource, "nvEncMapInputResource"),
        unmap_input_resource: entry!(unmap_input_resource, "nvEncUnmapInputResource"),
        destroy_encoder: entry!(destroy_encoder, "nvEncDestroyEncoder"),
        open_encode_session_ex: entry!(open_encode_session_ex, "nvEncOpenEncodeSessionEx"),
        register_resource: entry!(register_resource, "nvEncRegisterResource"),
        unregister_resource: entry!(unregister_resource, "nvEncUnregisterResource"),
        get_last_error_string: entry!(get_last_error_string, "nvEncGetLastErrorString"),
        _library: library,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_splits_into_major_and_minor() {
        // This value is what driver 615.71.09 reports.
        assert_eq!(split_version(209), (13, 1));
    }

    #[test]
    fn older_drivers_fail_the_pin() {
        assert!(serves_pinned_api((13, 1)));
        assert!(serves_pinned_api((13, 2)));
        assert!(serves_pinned_api((14, 0)));
        assert!(!serves_pinned_api((13, 0)));
        assert!(!serves_pinned_api((12, 2)));
    }
}
