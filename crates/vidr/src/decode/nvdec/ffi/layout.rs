//! Compile-time layout checks of the FFI mirrors.
//!
//! The values come from `sizeof` and `offsetof` in a C program compiled
//! against nv-codec-headers (NVDEC API 13.1) with GCC on x86-64 Linux, an
//! LP64 target. Windows is LLP64: `tcu_ulong` is 4 bytes there, so the
//! `CUVIDDECODECREATEINFO` and `CUVIDSOURCEDATAPACKET` values differ and
//! these checks do not run.

#![cfg(all(target_os = "linux", target_pointer_width = "64"))]

use std::mem::{align_of, offset_of, size_of};

use super::types::{
    CuVideoFormat, CuvidDecodeCaps, CuvidDecodeCreateInfo, CuvidParserDispInfo, CuvidParserParams,
    CuvidPicParams, CuvidProcParams, CuvidSourceDataPacket, VideoSignalDescription,
};

const _: () = assert!(offset_of!(CuvidPicParams, curr_pic_idx) == 8);

macro_rules! check {
    ($type:ty, size $size:expr, align $align:expr, $($field:ident @ $offset:expr),* $(,)?) => {
        const _: () = assert!(size_of::<$type>() == $size);
        const _: () = assert!(align_of::<$type>() == $align);
        $(const _: () = assert!(offset_of!($type, $field) == $offset);)*
    };
}

check!(CuvidDecodeCaps, size 88, align 4,
    codec_type @ 0,
    bit_depth_minus8 @ 8,
    is_supported @ 24,
    output_format_mask @ 26,
    max_width @ 28,
    max_height @ 32,
    max_mb_count @ 36,
    min_width @ 40,
    min_height @ 42,
);

check!(CuvidDecodeCreateInfo, size 176, align 8,
    num_decode_surfaces @ 16,
    codec_type @ 24,
    chroma_format @ 28,
    creation_flags @ 32,
    max_width @ 56,
    display_area @ 80,
    output_format @ 88,
    deinterlace_mode @ 92,
    target_width @ 96,
    num_output_surfaces @ 112,
    vid_lock @ 120,
    target_rect @ 128,
    enable_histogram @ 136,
    reserved2 @ 152,
);

check!(CuvidProcParams, size 264, align 8,
    progressive_frame @ 0,
    unpaired_field @ 12,
    raw_input_dptr @ 24,
    raw_output_dptr @ 40,
    output_stream @ 56,
    reserved @ 64,
    histogram_dptr @ 248,
    cuvid_proc_ext @ 256,
);

check!(CuVideoFormat, size 64, align 4,
    codec @ 0,
    frame_rate @ 4,
    progressive_sequence @ 12,
    bit_depth_luma_minus8 @ 13,
    bit_depth_chroma_minus8 @ 14,
    min_num_decode_surfaces @ 15,
    coded_width @ 16,
    coded_height @ 20,
    display_area @ 24,
    chroma_format @ 40,
    bitrate @ 44,
    display_aspect_ratio @ 48,
    video_signal_description @ 56,
    seqhdr_data_length @ 60,
);

// `color_primaries` sits at 57, `transfer_characteristics` at 58, and
// `matrix_coefficients` at 59 of `CUVIDEOFORMAT`.
check!(VideoSignalDescription, size 4, align 1,
    color_primaries @ 1,
    transfer_characteristics @ 2,
    matrix_coefficients @ 3,
);

check!(CuvidSourceDataPacket, size 32, align 8,
    payload_size @ 8,
    payload @ 16,
    timestamp @ 24,
);

check!(CuvidParserDispInfo, size 24, align 8,
    progressive_frame @ 4,
    top_field_first @ 8,
    repeat_first_field @ 12,
    timestamp @ 16,
);

check!(CuvidParserParams, size 136, align 8,
    max_num_decode_surfaces @ 4,
    clock_rate @ 8,
    error_threshold @ 12,
    max_display_delay @ 16,
    flags @ 20,
    reserved1 @ 24,
    user_data @ 40,
    sequence_callback @ 48,
    decode_picture @ 56,
    display_picture @ 64,
    get_operating_point @ 72,
    get_sei_msg @ 80,
    reserved2 @ 88,
    ext_video_info @ 128,
);

#[cfg(test)]
mod tests {
    use super::super::types::VideoSignalDescription;

    /// GCC packs `video_format : 3` then `video_full_range_flag : 1` from
    /// bit 0: setting only the flag stores 8, and `video_format` 5 alone
    /// stores 5.
    #[test]
    fn signal_bit_fields_match_gcc() {
        let full = VideoSignalDescription {
            flags: 8,
            ..VideoSignalDescription::default()
        };
        assert!(full.full_range());
        assert_eq!(full.video_format(), 0);
        let pal = VideoSignalDescription {
            flags: 5,
            ..VideoSignalDescription::default()
        };
        assert!(!pal.full_range());
        assert_eq!(pal.video_format(), 5);
    }
}
