//! Compile-time layout checks of the FFI mirrors.
//!
//! The values come from `sizeof` and `offsetof` in a C program compiled against
//! nv-codec-headers (NVENC API 13.1) with GCC on x86-64 Linux, an LP64 target.
//! Other targets do not run these checks.

#![cfg(all(target_os = "linux", target_pointer_width = "64"))]

use std::mem::{align_of, offset_of, size_of};

use super::config_types::{
    NvEncCodecConfig, NvEncConfig, NvEncConfigAv1, NvEncConfigH264, NvEncConfigHevc,
    NvEncPresetConfig, NvEncQp, NvEncRcParams, NvEncVuiParameters,
};
use super::guid::Guid;
use super::library::NvEncodeApiFunctionList;
use super::types::{
    MeHintCounts, NvEncCapsParam, NvEncCodecPicParams, NvEncCreateBitstreamBuffer,
    NvEncInitializeParams, NvEncLockBitstream, NvEncMapInputResource, NvEncOpenSessionParams,
    NvEncPicParams, NvEncRegisterResource,
};

macro_rules! check {
    ($type:ty, size $size:expr, align $align:expr $(, $field:ident @ $offset:expr)* $(,)?) => {
        const _: () = assert!(size_of::<$type>() == $size);
        const _: () = assert!(align_of::<$type>() == $align);
        $(const _: () = assert!(offset_of!($type, $field) == $offset);)*
    };
}

check!(Guid, size 16, align 4);
check!(NvEncQp, size 12, align 4);
check!(MeHintCounts, size 16, align 4);
check!(NvEncCodecPicParams, size 1544, align 8);
check!(NvEncCodecConfig, size 1792, align 8);

check!(NvEncCapsParam, size 256, align 4, caps_to_query @ 4, reserved @ 8);

check!(NvEncRcParams, size 128, align 4,
    rate_control_mode @ 4,
    const_qp @ 8,
    average_bit_rate @ 20,
    max_bit_rate @ 24,
    vbv_buffer_size @ 28,
    vbv_initial_delay @ 32,
    min_qp @ 40,
    temporal_layer_qp @ 80,
    target_quality @ 88,
    target_quality_lsb @ 89,
    lookahead_depth @ 90,
    multi_pass @ 100,
    lookahead_level @ 112,
    view_bitrate_ratios @ 116,
    reserved3 @ 123,
    reserved1 @ 124,
);

check!(NvEncVuiParameters, size 112, align 4,
    video_signal_type_present_flag @ 8,
    colour_matrix @ 32,
    chroma_sample_location_flag @ 36,
    bitstream_restriction_flag @ 48,
    time_scale @ 60,
    reserved @ 64,
);

check!(NvEncConfigH264, size 1792, align 8,
    level @ 4,
    idr_period @ 8,
    vui @ 72,
    ltr_num_frames @ 184,
    chroma_format_idc @ 192,
    use_b_frames_as_ref @ 200,
    output_bit_depth @ 212,
    input_bit_depth @ 216,
    tf_level @ 220,
    reserved1 @ 224,
    reserved2 @ 1280,
);

check!(NvEncConfigHevc, size 1560, align 8,
    idr_period @ 20,
    vui @ 64,
    ltr_trust_mode @ 176,
    use_b_frames_as_ref @ 180,
    output_bit_depth @ 200,
    input_bit_depth @ 204,
    num_views @ 212,
    reserved1 @ 216,
    reserved2 @ 1048,
);

check!(NvEncConfigAv1, size 1552, align 8,
    idr_period @ 20,
    tile_widths @ 48,
    max_temporal_layers_minus1 @ 64,
    color_primaries @ 68,
    color_range @ 80,
    chroma_sample_position @ 84,
    use_b_frames_as_ref @ 88,
    film_grain_params @ 96,
    num_fwd_refs @ 104,
    output_bit_depth @ 112,
    input_bit_depth @ 116,
    tf_level @ 128,
    reserved1 @ 132,
    reserved3 @ 1056,
);

check!(NvEncConfig, size 3584, align 8,
    profile_guid @ 4,
    gop_length @ 20,
    frame_interval_p @ 24,
    mono_chrome_encoding @ 28,
    frame_field_mode @ 32,
    mv_precision @ 36,
    rc_params @ 40,
    codec @ 168,
    reserved @ 1960,
    reserved2 @ 3072,
);

check!(NvEncPresetConfig, size 5128, align 8,
    preset_cfg @ 8,
    reserved1 @ 3592,
    reserved2 @ 4616,
);

check!(NvEncInitializeParams, size 1800, align 8,
    encode_guid @ 4,
    preset_guid @ 20,
    encode_width @ 36,
    dar_width @ 44,
    frame_rate_num @ 52,
    enable_encode_async @ 60,
    enable_ptd @ 64,
    priv_data_size @ 72,
    priv_data @ 80,
    encode_config @ 88,
    max_encode_width @ 96,
    max_me_hint_counts_per_block @ 104,
    tuning_info @ 136,
    buffer_format @ 140,
    reserved1 @ 152,
    reserved2 @ 1288,
);

check!(NvEncPicParams, size 3360, align 8,
    input_width @ 4,
    input_pitch @ 12,
    encode_pic_flags @ 16,
    frame_idx @ 20,
    input_time_stamp @ 24,
    input_duration @ 32,
    input_buffer @ 40,
    output_bitstream @ 48,
    completion_event @ 56,
    buffer_fmt @ 64,
    picture_struct @ 68,
    picture_type @ 72,
    codec_pic_params @ 80,
    me_hint_counts_per_block @ 1624,
    me_external_hints @ 1656,
    reserved3 @ 1768,
    reserved6 @ 2904,
);

check!(NvEncLockBitstream, size 1544, align 8,
    output_bitstream @ 8,
    slice_offsets @ 16,
    frame_idx @ 24,
    hw_encode_status @ 28,
    num_slices @ 32,
    bitstream_size_in_bytes @ 36,
    output_time_stamp @ 40,
    output_duration @ 48,
    bitstream_buffer_ptr @ 56,
    picture_type @ 64,
    picture_struct @ 68,
    frame_avg_qp @ 72,
    alpha_layer_size_in_bytes @ 108,
    output_stats_ptr @ 120,
    frame_idx_display @ 128,
    reserved1 @ 132,
    reserved2 @ 1008,
    reserved_internal @ 1512,
);

check!(NvEncMapInputResource, size 1544, align 8,
    input_resource @ 8,
    registered_resource @ 16,
    mapped_resource @ 24,
    mapped_buffer_fmt @ 32,
    reserved1 @ 36,
    reserved2 @ 1040,
);

check!(NvEncRegisterResource, size 1536, align 8,
    resource_type @ 4,
    pitch @ 16,
    resource_to_register @ 24,
    registered_resource @ 32,
    buffer_format @ 40,
    buffer_usage @ 44,
    input_fence_point @ 48,
    chroma_offset @ 56,
    chroma_offset_in @ 64,
    reserved1 @ 72,
    reserved2 @ 1048,
);

check!(NvEncCreateBitstreamBuffer, size 776, align 8,
    bitstream_buffer @ 16,
    bitstream_buffer_ptr @ 24,
    reserved1 @ 32,
    reserved2 @ 264,
);

check!(NvEncOpenSessionParams, size 1552, align 8,
    device_type @ 4,
    device @ 8,
    api_version @ 24,
    reserved1 @ 28,
    reserved2 @ 1040,
);

check!(NvEncodeApiFunctionList, size 2552, align 8,
    get_encode_guid_count @ 16,
    get_encode_guids @ 40,
    get_encode_caps @ 64,
    initialize_encoder @ 96,
    create_bitstream_buffer @ 120,
    destroy_bitstream_buffer @ 128,
    encode_picture @ 136,
    lock_bitstream @ 144,
    unlock_bitstream @ 152,
    map_input_resource @ 208,
    unmap_input_resource @ 216,
    destroy_encoder @ 224,
    open_encode_session_ex @ 240,
    register_resource @ 248,
    unregister_resource @ 256,
    get_last_error_string @ 304,
    get_encode_preset_config_ex @ 320,
    reserved2 @ 352,
);

#[cfg(test)]
mod tests {
    use super::super::config_types::{
        AV1_CHROMA_FORMAT_SHIFT, AV1_FLAG_ANNEXB, AV1_FLAG_DISABLE_SEQ_HDR,
        AV1_FLAG_REPEAT_SEQ_HDR, CONFIG_VER, H264_FLAG_DISABLE_SPSPPS, H264_FLAG_OUTPUT_AUD,
        H264_FLAG_REPEAT_SPSPPS, HEVC_CHROMA_FORMAT_SHIFT, HEVC_FLAG_DISABLE_SPSPPS,
        HEVC_FLAG_OUTPUT_AUD, HEVC_FLAG_REPEAT_SPSPPS, RC_FLAG_LOOKAHEAD, RC_PARAMS_VER,
    };
    use super::super::types::{
        API_VERSION, CAPS_PARAM_VER, CREATE_BITSTREAM_BUFFER_VER, FUNCTION_LIST_VER,
        INITIALIZE_PARAMS_VER, LOCK_BITSTREAM_VER, MAP_INPUT_RESOURCE_VER, OPEN_SESSION_VER,
        PIC_PARAMS_VER, PRESET_CONFIG_VER, REGISTER_RESOURCE_VER,
    };

    /// The values the header's macros give, printed by the GCC probe.
    #[test]
    fn struct_versions_match_the_header_macros() {
        assert_eq!(API_VERSION, 0x0100_000d);
        assert_eq!(CAPS_PARAM_VER, 0x7101_000d);
        assert_eq!(RC_PARAMS_VER, 0x7101_000d);
        assert_eq!(CONFIG_VER, 0xf109_000d);
        assert_eq!(INITIALIZE_PARAMS_VER, 0xf107_000d);
        assert_eq!(PRESET_CONFIG_VER, 0xf105_000d);
        assert_eq!(PIC_PARAMS_VER, 0xf107_000d);
        assert_eq!(LOCK_BITSTREAM_VER, 0xf102_000d);
        assert_eq!(MAP_INPUT_RESOURCE_VER, 0x7104_000d);
        assert_eq!(REGISTER_RESOURCE_VER, 0x7105_000d);
        assert_eq!(CREATE_BITSTREAM_BUFFER_VER, 0x7101_000d);
        assert_eq!(OPEN_SESSION_VER, 0x7101_000d);
        assert_eq!(FUNCTION_LIST_VER, 0x7102_000d);
    }

    /// GCC packs bit fields from bit 0. These words are what the probe stores
    /// after setting each field alone.
    #[test]
    fn bit_field_masks_match_gcc() {
        assert_eq!(RC_FLAG_LOOKAHEAD, 0x20);
        assert_eq!(H264_FLAG_OUTPUT_AUD, 0x40);
        assert_eq!(H264_FLAG_DISABLE_SPSPPS, 0x80);
        assert_eq!(H264_FLAG_REPEAT_SPSPPS, 0x1000);
        assert_eq!(HEVC_FLAG_OUTPUT_AUD, 0x10);
        assert_eq!(HEVC_FLAG_DISABLE_SPSPPS, 0x40);
        assert_eq!(HEVC_FLAG_REPEAT_SPSPPS, 0x80);
        assert_eq!(1_u32 << HEVC_CHROMA_FORMAT_SHIFT, 0x200);
        assert_eq!(AV1_FLAG_ANNEXB, 0x1);
        assert_eq!(AV1_FLAG_DISABLE_SEQ_HDR, 0x10);
        assert_eq!(AV1_FLAG_REPEAT_SEQ_HDR, 0x20);
        assert_eq!(1_u32 << AV1_CHROMA_FORMAT_SHIFT, 0x80);
    }
}
