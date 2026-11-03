//! `#[repr(C)]` mirrors of the NVDEC types vidr uses.
//!
//! The layout reference is NVIDIA's MIT-licensed headers as nv-codec-headers
//! ships them, NVDEC API 13.1: `dynlink_cuviddec.h` and `dynlink_nvcuvid.h`.
//! `layout.rs` asserts every size and the offsets of the fields vidr reads
//! or writes. C enums are `int` sized, so each enum is a `c_int` alias with
//! constants. `tcu_ulong` is C `unsigned long`, which is `c_ulong` on every
//! target except Cygwin.

// The driver reads or writes every field, including the ones vidr never
// touches, so each mirror keeps all of them.
#![allow(dead_code, reason = "FFI mirrors keep every header field and constant")]

use std::ffi::{c_int, c_short, c_uint, c_ulong, c_void};

/// `CUresult`: 0 is `CUDA_SUCCESS`.
pub(crate) type CuResult = c_int;
/// `CUDA_SUCCESS`.
pub(crate) const CUDA_SUCCESS: CuResult = 0;

/// `CUcontext`.
pub(crate) type CuContext = *mut c_void;
/// `CUstream`.
pub(crate) type CuStream = *mut c_void;
/// `CUvideodecoder`.
pub(crate) type CuVideoDecoder = *mut c_void;
/// `CUvideoparser`.
pub(crate) type CuVideoParser = *mut c_void;
/// `CUvideoctxlock`.
pub(crate) type CuVideoCtxLock = *mut c_void;
/// `CUvideotimestamp`: `long long`.
pub(crate) type CuVideoTimestamp = i64;

/// `cudaVideoCodec`.
pub(crate) type CudaVideoCodec = c_int;
/// `cudaVideoCodec_H264`.
pub(crate) const CODEC_H264: CudaVideoCodec = 4;
/// `cudaVideoCodec_HEVC`.
pub(crate) const CODEC_HEVC: CudaVideoCodec = 8;
/// `cudaVideoCodec_VP9`.
pub(crate) const CODEC_VP9: CudaVideoCodec = 10;
/// `cudaVideoCodec_AV1`.
pub(crate) const CODEC_AV1: CudaVideoCodec = 11;

/// `cudaVideoChromaFormat`.
pub(crate) type CudaVideoChromaFormat = c_int;
/// `cudaVideoChromaFormat_Monochrome`.
pub(crate) const CHROMA_MONOCHROME: CudaVideoChromaFormat = 0;
/// `cudaVideoChromaFormat_420`.
pub(crate) const CHROMA_420: CudaVideoChromaFormat = 1;
/// `cudaVideoChromaFormat_422`.
pub(crate) const CHROMA_422: CudaVideoChromaFormat = 2;
/// `cudaVideoChromaFormat_444`.
pub(crate) const CHROMA_444: CudaVideoChromaFormat = 3;

/// `cudaVideoSurfaceFormat`.
pub(crate) type CudaVideoSurfaceFormat = c_int;
/// `cudaVideoSurfaceFormat_NV12`.
pub(crate) const SURFACE_NV12: CudaVideoSurfaceFormat = 0;
/// `cudaVideoSurfaceFormat_P016`.
pub(crate) const SURFACE_P016: CudaVideoSurfaceFormat = 1;

/// `cudaVideoDeinterlaceMode_Weave`: both fields as one frame, unchanged.
pub(crate) const DEINTERLACE_WEAVE: c_int = 0;
/// `cudaVideoCreate_PreferCUVID`: the dedicated decode engines.
pub(crate) const CREATE_PREFER_CUVID: c_ulong = 0x04;

/// `CUVID_PKT_ENDOFSTREAM`.
pub(crate) const PKT_ENDOFSTREAM: c_ulong = 0x01;
/// `CUVID_PKT_TIMESTAMP`.
pub(crate) const PKT_TIMESTAMP: c_ulong = 0x02;

/// Mirrors `CUVIDDECODECAPS` (`dynlink_cuviddec.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CuvidDecodeCaps {
    pub(crate) codec_type: CudaVideoCodec,
    pub(crate) chroma_format: CudaVideoChromaFormat,
    pub(crate) bit_depth_minus8: c_uint,
    pub(crate) reserved1: [c_uint; 3],
    pub(crate) is_supported: u8,
    pub(crate) num_nvdecs: u8,
    pub(crate) output_format_mask: u16,
    pub(crate) max_width: c_uint,
    pub(crate) max_height: c_uint,
    pub(crate) max_mb_count: c_uint,
    pub(crate) min_width: u16,
    pub(crate) min_height: u16,
    pub(crate) is_histogram_supported: u8,
    pub(crate) counter_bit_depth: u8,
    pub(crate) max_histogram_bins: u16,
    pub(crate) is_decode_stats_supported: u8,
    pub(crate) reserved4: [u8; 3],
    pub(crate) reserved3: [c_uint; 9],
}

/// Mirrors the `short` rectangles `display_area` and `target_rect` of
/// `CUVIDDECODECREATEINFO`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ShortRect {
    pub(crate) left: c_short,
    pub(crate) top: c_short,
    pub(crate) right: c_short,
    pub(crate) bottom: c_short,
}

/// Mirrors `CUVIDDECODECREATEINFO` (`dynlink_cuviddec.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CuvidDecodeCreateInfo {
    pub(crate) width: c_ulong,
    pub(crate) height: c_ulong,
    pub(crate) num_decode_surfaces: c_ulong,
    pub(crate) codec_type: CudaVideoCodec,
    pub(crate) chroma_format: CudaVideoChromaFormat,
    pub(crate) creation_flags: c_ulong,
    pub(crate) bit_depth_minus8: c_ulong,
    pub(crate) intra_decode_only: c_ulong,
    pub(crate) max_width: c_ulong,
    pub(crate) max_height: c_ulong,
    pub(crate) reserved1: c_ulong,
    pub(crate) display_area: ShortRect,
    pub(crate) output_format: CudaVideoSurfaceFormat,
    pub(crate) deinterlace_mode: c_int,
    pub(crate) target_width: c_ulong,
    pub(crate) target_height: c_ulong,
    pub(crate) num_output_surfaces: c_ulong,
    pub(crate) vid_lock: CuVideoCtxLock,
    pub(crate) target_rect: ShortRect,
    pub(crate) enable_histogram: c_ulong,
    pub(crate) enable_decode_features: c_ulong,
    pub(crate) reserved2: [c_ulong; 3],
}

/// Mirrors `CUVIDPROCPARAMS` (`dynlink_cuviddec.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CuvidProcParams {
    pub(crate) progressive_frame: c_int,
    pub(crate) second_field: c_int,
    pub(crate) top_field_first: c_int,
    pub(crate) unpaired_field: c_int,
    pub(crate) reserved_flags: c_uint,
    pub(crate) reserved_zero: c_uint,
    pub(crate) raw_input_dptr: u64,
    pub(crate) raw_input_pitch: c_uint,
    pub(crate) raw_input_format: c_uint,
    pub(crate) raw_output_dptr: u64,
    pub(crate) raw_output_pitch: c_uint,
    pub(crate) reserved1: c_uint,
    pub(crate) output_stream: CuStream,
    pub(crate) reserved: [c_uint; 46],
    pub(crate) histogram_dptr: *mut u64,
    pub(crate) cuvid_proc_ext: *mut c_void,
}

/// Opaque `CUVIDPICPARAMS`: the parser fills it, and vidr passes the
/// pointer to `cuvidDecodePicture` unread.
#[repr(C)]
pub(crate) struct CuvidPicParams {
    _opaque: [u8; 0],
}

/// Mirrors the `frame_rate` member of `CUVIDEOFORMAT`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FrameRate {
    pub(crate) numerator: c_uint,
    pub(crate) denominator: c_uint,
}

/// Mirrors the `int` rectangle `display_area` of `CUVIDEOFORMAT`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct IntRect {
    pub(crate) left: c_int,
    pub(crate) top: c_int,
    pub(crate) right: c_int,
    pub(crate) bottom: c_int,
}

/// Mirrors the `display_aspect_ratio` member of `CUVIDEOFORMAT`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct AspectRatio {
    pub(crate) x: c_int,
    pub(crate) y: c_int,
}

/// Mirrors the `video_signal_description` member of `CUVIDEOFORMAT`.
///
/// Byte 0 holds the C bit fields from bit 0 up: `video_format` in bits 0
/// to 2, `video_full_range_flag` in bit 3.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VideoSignalDescription {
    pub(crate) flags: u8,
    pub(crate) color_primaries: u8,
    pub(crate) transfer_characteristics: u8,
    pub(crate) matrix_coefficients: u8,
}

impl VideoSignalDescription {
    /// `video_format`.
    pub(crate) const fn video_format(self) -> u8 {
        self.flags & 0x07
    }

    /// `video_full_range_flag`.
    pub(crate) const fn full_range(self) -> bool {
        self.flags & 0x08 != 0
    }
}

/// Mirrors `CUVIDEOFORMAT` (`dynlink_nvcuvid.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CuVideoFormat {
    pub(crate) codec: CudaVideoCodec,
    pub(crate) frame_rate: FrameRate,
    pub(crate) progressive_sequence: u8,
    pub(crate) bit_depth_luma_minus8: u8,
    pub(crate) bit_depth_chroma_minus8: u8,
    pub(crate) min_num_decode_surfaces: u8,
    pub(crate) coded_width: c_uint,
    pub(crate) coded_height: c_uint,
    pub(crate) display_area: IntRect,
    pub(crate) chroma_format: CudaVideoChromaFormat,
    pub(crate) bitrate: c_uint,
    pub(crate) display_aspect_ratio: AspectRatio,
    pub(crate) video_signal_description: VideoSignalDescription,
    pub(crate) seqhdr_data_length: c_uint,
}

/// Mirrors `CUVIDSOURCEDATAPACKET` (`dynlink_nvcuvid.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CuvidSourceDataPacket {
    pub(crate) flags: c_ulong,
    pub(crate) payload_size: c_ulong,
    pub(crate) payload: *const u8,
    pub(crate) timestamp: CuVideoTimestamp,
}

/// Mirrors `CUVIDPARSERDISPINFO` (`dynlink_nvcuvid.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CuvidParserDispInfo {
    pub(crate) picture_index: c_int,
    pub(crate) progressive_frame: c_int,
    pub(crate) top_field_first: c_int,
    pub(crate) repeat_first_field: c_int,
    pub(crate) timestamp: CuVideoTimestamp,
}

/// `PFNVIDSEQUENCECALLBACK`.
pub(crate) type SequenceCallback = unsafe extern "C" fn(*mut c_void, *mut CuVideoFormat) -> c_int;
/// `PFNVIDDECODECALLBACK`.
pub(crate) type DecodeCallback = unsafe extern "C" fn(*mut c_void, *mut CuvidPicParams) -> c_int;
/// `PFNVIDDISPLAYCALLBACK`.
pub(crate) type DisplayCallback =
    unsafe extern "C" fn(*mut c_void, *mut CuvidParserDispInfo) -> c_int;

/// Mirrors `CUVIDPARSERPARAMS` (`dynlink_nvcuvid.h`).
///
/// `flags` holds the C bit fields from bit 0 up: `bAnnexb` in bit 0,
/// `bMemoryOptimize` in bit 1. vidr leaves both 0. The operating point and
/// SEI callbacks stay null, so the parser decodes operating point 0.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CuvidParserParams {
    pub(crate) codec_type: CudaVideoCodec,
    pub(crate) max_num_decode_surfaces: c_uint,
    pub(crate) clock_rate: c_uint,
    pub(crate) error_threshold: c_uint,
    pub(crate) max_display_delay: c_uint,
    pub(crate) flags: c_uint,
    pub(crate) reserved1: [c_uint; 4],
    pub(crate) user_data: *mut c_void,
    pub(crate) sequence_callback: Option<SequenceCallback>,
    pub(crate) decode_picture: Option<DecodeCallback>,
    pub(crate) display_picture: Option<DisplayCallback>,
    pub(crate) get_operating_point: *mut c_void,
    pub(crate) get_sei_msg: *mut c_void,
    pub(crate) reserved2: [*mut c_void; 5],
    pub(crate) ext_video_info: *mut c_void,
}

impl CuvidDecodeCreateInfo {
    /// Every field zero and every pointer null, as the header requires of
    /// reserved fields.
    pub(crate) const fn zeroed() -> Self {
        Self {
            width: 0,
            height: 0,
            num_decode_surfaces: 0,
            codec_type: 0,
            chroma_format: 0,
            creation_flags: 0,
            bit_depth_minus8: 0,
            intra_decode_only: 0,
            max_width: 0,
            max_height: 0,
            reserved1: 0,
            display_area: ShortRect {
                left: 0,
                top: 0,
                right: 0,
                bottom: 0,
            },
            output_format: 0,
            deinterlace_mode: 0,
            target_width: 0,
            target_height: 0,
            num_output_surfaces: 0,
            vid_lock: std::ptr::null_mut(),
            target_rect: ShortRect {
                left: 0,
                top: 0,
                right: 0,
                bottom: 0,
            },
            enable_histogram: 0,
            enable_decode_features: 0,
            reserved2: [0; 3],
        }
    }
}

impl CuvidProcParams {
    /// Every field zero and every pointer null.
    pub(crate) const fn zeroed() -> Self {
        Self {
            progressive_frame: 0,
            second_field: 0,
            top_field_first: 0,
            unpaired_field: 0,
            reserved_flags: 0,
            reserved_zero: 0,
            raw_input_dptr: 0,
            raw_input_pitch: 0,
            raw_input_format: 0,
            raw_output_dptr: 0,
            raw_output_pitch: 0,
            reserved1: 0,
            output_stream: std::ptr::null_mut(),
            reserved: [0; 46],
            histogram_dptr: std::ptr::null_mut(),
            cuvid_proc_ext: std::ptr::null_mut(),
        }
    }
}

impl CuvidParserParams {
    /// Every field zero and every pointer null.
    pub(crate) const fn zeroed() -> Self {
        Self {
            codec_type: 0,
            max_num_decode_surfaces: 0,
            clock_rate: 0,
            error_threshold: 0,
            max_display_delay: 0,
            flags: 0,
            reserved1: [0; 4],
            user_data: std::ptr::null_mut(),
            sequence_callback: None,
            decode_picture: None,
            display_picture: None,
            get_operating_point: std::ptr::null_mut(),
            get_sei_msg: std::ptr::null_mut(),
            reserved2: [std::ptr::null_mut(); 5],
            ext_video_info: std::ptr::null_mut(),
        }
    }
}
