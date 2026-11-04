//! `#[repr(C)]` mirrors of the NVENC session, input, and output types.
//!
//! The layout reference is NVIDIA's MIT-licensed `nvEncodeAPI.h` as
//! nv-codec-headers ships it, NVENC API 13.1. `layout.rs` asserts every size and
//! the offsets of the fields vidr reads or writes. C enums are `int` sized, so
//! each enum is a `c_int` alias with constants.

// The driver reads or writes every field, including the ones vidr never
// touches, so each mirror keeps all of them.
#![allow(dead_code, reason = "FFI mirrors keep every header field and constant")]

use std::ffi::{c_int, c_void};

use super::config_types::NvEncConfig;
use super::guid::Guid;

/// `NVENCAPI_MAJOR_VERSION`.
pub(crate) const API_MAJOR: u32 = 13;
/// `NVENCAPI_MINOR_VERSION`.
pub(crate) const API_MINOR: u32 = 1;
/// `NVENCAPI_VERSION`.
pub(crate) const API_VERSION: u32 = API_MAJOR | (API_MINOR << 24);

/// `NVENCAPI_STRUCT_VERSION(ver)`.
pub(crate) const fn struct_version(ver: u32) -> u32 {
    API_VERSION | (ver << 16) | (0x7 << 28)
}

/// `NVENCAPI_STRUCT_VERSION(ver) | (1u << 31)`, the form of the larger structs.
pub(crate) const fn long_struct_version(ver: u32) -> u32 {
    struct_version(ver) | (1 << 31)
}

/// `NV_ENC_CAPS_PARAM_VER`.
pub(crate) const CAPS_PARAM_VER: u32 = struct_version(1);
/// `NV_ENC_PRESET_CONFIG_VER`.
pub(crate) const PRESET_CONFIG_VER: u32 = long_struct_version(5);
/// `NV_ENC_INITIALIZE_PARAMS_VER`.
pub(crate) const INITIALIZE_PARAMS_VER: u32 = long_struct_version(7);
/// `NV_ENC_PIC_PARAMS_VER`.
pub(crate) const PIC_PARAMS_VER: u32 = long_struct_version(7);
/// `NV_ENC_LOCK_BITSTREAM_VER`.
pub(crate) const LOCK_BITSTREAM_VER: u32 = long_struct_version(2);
/// `NV_ENC_MAP_INPUT_RESOURCE_VER`.
pub(crate) const MAP_INPUT_RESOURCE_VER: u32 = struct_version(4);
/// `NV_ENC_REGISTER_RESOURCE_VER`.
pub(crate) const REGISTER_RESOURCE_VER: u32 = struct_version(5);
/// `NV_ENC_CREATE_BITSTREAM_BUFFER_VER`.
pub(crate) const CREATE_BITSTREAM_BUFFER_VER: u32 = struct_version(1);
/// `NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER`.
pub(crate) const OPEN_SESSION_VER: u32 = struct_version(1);
/// `NV_ENCODE_API_FUNCTION_LIST_VER`.
pub(crate) const FUNCTION_LIST_VER: u32 = struct_version(2);

/// `NVENCSTATUS`: 0 is `NV_ENC_SUCCESS`.
pub(crate) type NvEncStatus = c_int;
/// The encoder session handle.
pub(crate) type NvEncHandle = *mut c_void;
/// `NV_ENC_REGISTERED_PTR`.
pub(crate) type NvEncRegisteredPtr = *mut c_void;
/// `NV_ENC_INPUT_PTR`.
pub(crate) type NvEncInputPtr = *mut c_void;
/// `NV_ENC_OUTPUT_PTR`.
pub(crate) type NvEncOutputPtr = *mut c_void;

/// `NV_ENC_DEVICE_TYPE_CUDA`.
pub(crate) const DEVICE_TYPE_CUDA: c_int = 1;
/// `NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR`.
pub(crate) const RESOURCE_CUDA_DEVICE_PTR: c_int = 1;
/// `NV_ENC_INPUT_IMAGE`.
pub(crate) const BUFFER_USAGE_INPUT_IMAGE: c_int = 0;
/// `NV_ENC_BUFFER_FORMAT_NV12`.
pub(crate) const BUFFER_FORMAT_NV12: c_int = 0x1;
/// `NV_ENC_BUFFER_FORMAT_YUV420_10BIT`.
pub(crate) const BUFFER_FORMAT_YUV420_10BIT: c_int = 0x1_0000;
/// `NV_ENC_PIC_STRUCT_FRAME`.
pub(crate) const PIC_STRUCT_FRAME: c_int = 1;
/// `NV_ENC_PIC_TYPE_IDR`.
pub(crate) const PIC_TYPE_IDR: c_int = 3;
/// `NV_ENC_PIC_FLAG_EOS`.
pub(crate) const PIC_FLAG_EOS: u32 = 0x8;

/// `NV_ENC_CAPS_NUM_MAX_BFRAMES`.
pub(crate) const CAPS_NUM_MAX_BFRAMES: c_int = 0;
/// `NV_ENC_CAPS_WIDTH_MAX`.
pub(crate) const CAPS_WIDTH_MAX: c_int = 16;
/// `NV_ENC_CAPS_HEIGHT_MAX`.
pub(crate) const CAPS_HEIGHT_MAX: c_int = 17;
/// `NV_ENC_CAPS_SUPPORT_10BIT_ENCODE`.
pub(crate) const CAPS_SUPPORT_10BIT: c_int = 39;
/// `NV_ENC_CAPS_WIDTH_MIN`.
pub(crate) const CAPS_WIDTH_MIN: c_int = 45;
/// `NV_ENC_CAPS_HEIGHT_MIN`.
pub(crate) const CAPS_HEIGHT_MIN: c_int = 46;

/// `NV_ENC_CAPS_PARAM`.
#[repr(C)]
pub(crate) struct NvEncCapsParam {
    pub(crate) version: u32,
    pub(crate) caps_to_query: c_int,
    pub(crate) reserved: [u32; 62],
}

/// `NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE`. Its bit fields stay zero.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct MeHintCounts(pub(crate) [u32; 4]);

/// `NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS`.
#[repr(C)]
pub(crate) struct NvEncOpenSessionParams {
    pub(crate) version: u32,
    pub(crate) device_type: c_int,
    pub(crate) device: *mut c_void,
    pub(crate) reserved: *mut c_void,
    pub(crate) api_version: u32,
    pub(crate) reserved1: [u32; 253],
    pub(crate) reserved2: [*mut c_void; 64],
}

/// `NV_ENC_INITIALIZE_PARAMS`.
#[repr(C)]
pub(crate) struct NvEncInitializeParams {
    pub(crate) version: u32,
    pub(crate) encode_guid: Guid,
    pub(crate) preset_guid: Guid,
    pub(crate) encode_width: u32,
    pub(crate) encode_height: u32,
    pub(crate) dar_width: u32,
    pub(crate) dar_height: u32,
    pub(crate) frame_rate_num: u32,
    pub(crate) frame_rate_den: u32,
    pub(crate) enable_encode_async: u32,
    pub(crate) enable_ptd: u32,
    /// `reportSliceOffsets` to `reservedBitFields`.
    pub(crate) flags: u32,
    pub(crate) priv_data_size: u32,
    pub(crate) reserved: u32,
    pub(crate) priv_data: *mut c_void,
    pub(crate) encode_config: *mut NvEncConfig,
    pub(crate) max_encode_width: u32,
    pub(crate) max_encode_height: u32,
    pub(crate) max_me_hint_counts_per_block: [MeHintCounts; 2],
    pub(crate) tuning_info: c_int,
    pub(crate) buffer_format: c_int,
    pub(crate) num_state_buffers: u32,
    pub(crate) output_stats_level: c_int,
    pub(crate) reserved1: [u32; 284],
    pub(crate) reserved2: [*mut c_void; 64],
}

/// `NV_ENC_CODEC_PIC_PARAMS`, opaque: vidr leaves picture type and references
/// to the encoder, so this union stays zero.
#[repr(C, align(8))]
pub(crate) struct NvEncCodecPicParams(pub(crate) [u8; 1544]);

/// `NV_ENC_PIC_PARAMS`.
#[repr(C)]
pub(crate) struct NvEncPicParams {
    pub(crate) version: u32,
    pub(crate) input_width: u32,
    pub(crate) input_height: u32,
    pub(crate) input_pitch: u32,
    pub(crate) encode_pic_flags: u32,
    pub(crate) frame_idx: u32,
    pub(crate) input_time_stamp: u64,
    pub(crate) input_duration: u64,
    pub(crate) input_buffer: NvEncInputPtr,
    pub(crate) output_bitstream: NvEncOutputPtr,
    pub(crate) completion_event: *mut c_void,
    pub(crate) buffer_fmt: c_int,
    pub(crate) picture_struct: c_int,
    pub(crate) picture_type: c_int,
    pub(crate) codec_pic_params: NvEncCodecPicParams,
    pub(crate) me_hint_counts_per_block: [MeHintCounts; 2],
    pub(crate) me_external_hints: *mut c_void,
    pub(crate) reserved2: [u32; 7],
    pub(crate) reserved5: [*mut c_void; 2],
    pub(crate) qp_delta_map: *mut i8,
    pub(crate) qp_delta_map_size: u32,
    pub(crate) reserved_bit_fields: u32,
    pub(crate) me_hint_ref_pic_dist: [u16; 2],
    pub(crate) diff_pic_num_hint: i32,
    pub(crate) alpha_buffer: NvEncInputPtr,
    pub(crate) me_external_sb_hints: *mut c_void,
    pub(crate) me_sb_hints_count: u32,
    pub(crate) state_buffer_idx: u32,
    pub(crate) output_recon_buffer: NvEncOutputPtr,
    pub(crate) reserved3: [u32; 284],
    pub(crate) reserved6: [*mut c_void; 57],
}

/// `NV_ENC_LOCK_BITSTREAM`.
#[repr(C)]
pub(crate) struct NvEncLockBitstream {
    pub(crate) version: u32,
    /// `doNotWait` to `reservedBitFields`.
    pub(crate) flags: u32,
    pub(crate) output_bitstream: NvEncOutputPtr,
    pub(crate) slice_offsets: *mut u32,
    pub(crate) frame_idx: u32,
    pub(crate) hw_encode_status: u32,
    pub(crate) num_slices: u32,
    pub(crate) bitstream_size_in_bytes: u32,
    pub(crate) output_time_stamp: u64,
    pub(crate) output_duration: u64,
    pub(crate) bitstream_buffer_ptr: *mut c_void,
    pub(crate) picture_type: c_int,
    pub(crate) picture_struct: c_int,
    pub(crate) frame_avg_qp: u32,
    pub(crate) frame_satd: u32,
    pub(crate) ltr_frame_idx: u32,
    pub(crate) ltr_frame_bitmap: u32,
    pub(crate) temporal_id: u32,
    pub(crate) intra_mb_count: u32,
    pub(crate) inter_mb_count: u32,
    pub(crate) average_mvx: i32,
    pub(crate) average_mvy: i32,
    pub(crate) alpha_layer_size_in_bytes: u32,
    pub(crate) output_stats_ptr_size: u32,
    pub(crate) reserved: u32,
    pub(crate) output_stats_ptr: *mut c_void,
    pub(crate) frame_idx_display: u32,
    pub(crate) reserved1: [u32; 219],
    pub(crate) reserved2: [*mut c_void; 63],
    pub(crate) reserved_internal: [u32; 8],
}

/// `NV_ENC_MAP_INPUT_RESOURCE`.
#[repr(C)]
pub(crate) struct NvEncMapInputResource {
    pub(crate) version: u32,
    pub(crate) sub_resource_index: u32,
    pub(crate) input_resource: *mut c_void,
    pub(crate) registered_resource: NvEncRegisteredPtr,
    pub(crate) mapped_resource: NvEncInputPtr,
    pub(crate) mapped_buffer_fmt: c_int,
    pub(crate) reserved1: [u32; 251],
    pub(crate) reserved2: [*mut c_void; 63],
}

/// `NV_ENC_REGISTER_RESOURCE`.
#[repr(C)]
pub(crate) struct NvEncRegisterResource {
    pub(crate) version: u32,
    pub(crate) resource_type: c_int,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pitch: u32,
    pub(crate) sub_resource_index: u32,
    pub(crate) resource_to_register: *mut c_void,
    pub(crate) registered_resource: NvEncRegisteredPtr,
    pub(crate) buffer_format: c_int,
    pub(crate) buffer_usage: c_int,
    pub(crate) input_fence_point: *mut c_void,
    pub(crate) chroma_offset: [u32; 2],
    pub(crate) chroma_offset_in: [u32; 2],
    pub(crate) reserved1: [u32; 244],
    pub(crate) reserved2: [*mut c_void; 61],
}

/// `NV_ENC_CREATE_BITSTREAM_BUFFER`.
#[repr(C)]
pub(crate) struct NvEncCreateBitstreamBuffer {
    pub(crate) version: u32,
    pub(crate) size: u32,
    pub(crate) memory_heap: c_int,
    pub(crate) reserved: u32,
    pub(crate) bitstream_buffer: NvEncOutputPtr,
    pub(crate) bitstream_buffer_ptr: *mut c_void,
    pub(crate) reserved1: [u32; 58],
    pub(crate) reserved2: [*mut c_void; 64],
}

impl_zeroed!(
    NvEncCapsParam,
    NvEncOpenSessionParams,
    NvEncInitializeParams,
    NvEncPicParams,
    NvEncLockBitstream,
    NvEncMapInputResource,
    NvEncRegisterResource,
    NvEncCreateBitstreamBuffer,
);
