//! [`EncodeConfig`] to the NVENC structures. No driver calls.
//!
//! The preset config from `nvEncGetEncodePresetConfigEx` is the base. [`apply`]
//! overwrites the fields the config owns and keeps the rest of the preset.

use std::ffi::c_int;

use super::ffi::{
    AV1_CHROMA_FORMAT_SHIFT, AV1_FLAG_ANNEXB, AV1_FLAG_DISABLE_SEQ_HDR, AV1_FLAG_REPEAT_SEQ_HDR,
    BFRAME_REF_DISABLED, BUFFER_FORMAT_NV12, BUFFER_FORMAT_YUV420_10BIT, CHROMA_FORMAT_420,
    CODEC_AV1, CODEC_H264, CODEC_HEVC, Guid, H264_FLAG_DISABLE_SPSPPS, H264_FLAG_OUTPUT_AUD,
    H264_FLAG_REPEAT_SPSPPS, HEVC_CHROMA_FORMAT_SHIFT, HEVC_FLAG_DISABLE_SPSPPS,
    HEVC_FLAG_OUTPUT_AUD, HEVC_FLAG_REPEAT_SPSPPS, INITIALIZE_PARAMS_VER, NvEncConfig,
    NvEncInitializeParams, NvEncRcParams, NvEncVuiParameters, PRESETS, PROFILE_AV1_MAIN,
    PROFILE_H264_HIGH, PROFILE_H264_HIGH_10, PROFILE_HEVC_MAIN, PROFILE_HEVC_MAIN10, RC_CONSTQP,
    RC_FLAG_LOOKAHEAD, RC_PARAMS_VER, RC_VBR, VIDEO_FORMAT_UNSPECIFIED,
};
use crate::decode::{ColorFacts, PixelFormat};
use crate::encode::{EncodeCodec, EncodeConfig, Preset, RateControl, Tuning};
use crate::error::EncodeError;

/// The codec's GUID.
pub(super) const fn codec_guid(codec: EncodeCodec) -> Guid {
    match codec {
        EncodeCodec::H264 => CODEC_H264,
        EncodeCodec::Hevc => CODEC_HEVC,
        EncodeCodec::Av1 => CODEC_AV1,
    }
}

/// The codec names of `guids`, for a refusal.
pub(super) fn codec_names(guids: &[Guid]) -> String {
    let names: Vec<&str> = [EncodeCodec::H264, EncodeCodec::Hevc, EncodeCodec::Av1]
        .into_iter()
        .filter(|&codec| guids.contains(&codec_guid(codec)))
        .map(EncodeCodec::name)
        .collect();
    if names.is_empty() {
        "no codec vidr knows".to_owned()
    } else {
        names.join(", ")
    }
}

/// The preset's GUID.
pub(super) const fn preset_guid(preset: Preset) -> Guid {
    PRESETS[match preset {
        Preset::P1 => 0,
        Preset::P2 => 1,
        Preset::P3 => 2,
        Preset::P4 => 3,
        Preset::P5 => 4,
        Preset::P6 => 5,
        Preset::P7 => 6,
    }]
}

/// The `NV_ENC_TUNING_INFO` value.
pub(super) const fn tuning_info(tuning: Tuning) -> c_int {
    match tuning {
        Tuning::HighQuality => 1,
        Tuning::LowLatency => 2,
        Tuning::UltraLowLatency => 3,
        Tuning::Lossless => 4,
        Tuning::UltraHighQuality => 5,
    }
}

/// The `NV_ENC_BUFFER_FORMAT` value.
pub(super) const fn buffer_format(format: PixelFormat) -> c_int {
    match format {
        PixelFormat::Nv12 => BUFFER_FORMAT_NV12,
        PixelFormat::P010 => BUFFER_FORMAT_YUV420_10BIT,
    }
}

const fn profile_guid(codec: EncodeCodec, format: PixelFormat) -> Guid {
    match (codec, format) {
        (EncodeCodec::H264, PixelFormat::Nv12) => PROFILE_H264_HIGH,
        (EncodeCodec::H264, PixelFormat::P010) => PROFILE_H264_HIGH_10,
        (EncodeCodec::Hevc, PixelFormat::Nv12) => PROFILE_HEVC_MAIN,
        (EncodeCodec::Hevc, PixelFormat::P010) => PROFILE_HEVC_MAIN10,
        (EncodeCodec::Av1, _) => PROFILE_AV1_MAIN,
    }
}

fn apply_rate_control(rate_control: RateControl, rc: &mut NvEncRcParams) {
    rc.version = RC_PARAMS_VER;
    rc.flags &= !RC_FLAG_LOOKAHEAD;
    rc.target_quality = 0;
    rc.target_quality_lsb = 0;
    rc.average_bit_rate = 0;
    rc.max_bit_rate = 0;
    rc.vbv_buffer_size = 0;
    rc.vbv_initial_delay = 0;
    match rate_control {
        RateControl::VbrQuality { cq } => {
            rc.rate_control_mode = RC_VBR;
            rc.target_quality = cq;
        }
        RateControl::ConstQp {
            intra,
            inter_p,
            inter_b,
        } => {
            rc.rate_control_mode = RC_CONSTQP;
            rc.const_qp.qp_intra = u32::from(intra);
            rc.const_qp.qp_inter_p = u32::from(inter_p);
            rc.const_qp.qp_inter_b = u32::from(inter_b);
        }
        RateControl::Bitrate { average, maximum } => {
            rc.rate_control_mode = RC_VBR;
            rc.average_bit_rate = average;
            rc.max_bit_rate = maximum.unwrap_or(0);
        }
    }
}

fn apply_vui(color: &ColorFacts, intra_only: bool, vui: &mut NvEncVuiParameters) {
    vui.video_signal_type_present_flag = 1;
    vui.video_format = VIDEO_FORMAT_UNSPECIFIED;
    vui.video_full_range_flag = u32::from(color.full_range);
    vui.colour_description_present_flag = 1;
    vui.colour_primaries = c_int::from(color.colour_primaries);
    vui.transfer_characteristics = c_int::from(color.transfer_characteristics);
    vui.colour_matrix = c_int::from(color.matrix_coefficients);
    let location = color.chroma_location.map(|location| location.to_h273());
    vui.chroma_sample_location_flag = u32::from(location.is_some());
    vui.chroma_sample_location_top = u32::from(location.unwrap_or(0));
    vui.chroma_sample_location_bot = u32::from(location.unwrap_or(0));
    vui.bitstream_restriction_flag = u32::from(!intra_only);
}

/// Overwrites what `config` owns in the preset's `cfg`.
///
/// Parameter sets come only with the first picture: the config record carries
/// them, and the samples drop them. The AV1 sequence header repeats at each key
/// frame, because the AV1 ISO-BMFF binding asks it of sync samples.
pub(super) fn apply(config: &EncodeConfig, cfg: &mut NvEncConfig) {
    let depth = c_int::from(config.format.bit_depth());
    let intra_only = config.gop_length == 1;
    let b_frame_reference = config.b_frames >= 2;

    cfg.profile_guid = profile_guid(config.codec, config.format);
    cfg.gop_length = config.gop_length;
    cfg.frame_interval_p = if intra_only {
        0
    } else {
        // `validate` bounds `b_frames` below `gop_length`, and the GPU's caps
        // bound it to single digits before this runs.
        i32::try_from(config.b_frames.saturating_add(1)).unwrap_or(i32::MAX)
    };
    apply_rate_control(config.rate_control, &mut cfg.rc_params);

    match config.codec {
        EncodeCodec::H264 => {
            // SAFETY: every member of the union holds only integers and raw
            // pointers, so any bytes are a valid value of each.
            let mut h264 = unsafe { cfg.codec.h264 };
            h264.flags &=
                !(H264_FLAG_OUTPUT_AUD | H264_FLAG_DISABLE_SPSPPS | H264_FLAG_REPEAT_SPSPPS);
            h264.idr_period = config.gop_length;
            h264.chroma_format_idc = CHROMA_FORMAT_420;
            h264.input_bit_depth = depth;
            h264.output_bit_depth = depth;
            if !b_frame_reference {
                h264.use_b_frames_as_ref = BFRAME_REF_DISABLED;
            }
            apply_vui(&config.color, intra_only, &mut h264.vui);
            cfg.codec.h264 = h264;
        }
        EncodeCodec::Hevc => {
            // SAFETY: every member of the union holds only integers and raw
            // pointers, so any bytes are a valid value of each.
            let mut hevc = unsafe { cfg.codec.hevc };
            hevc.flags &=
                !(HEVC_FLAG_OUTPUT_AUD | HEVC_FLAG_DISABLE_SPSPPS | HEVC_FLAG_REPEAT_SPSPPS);
            hevc.flags &= !(0b11 << HEVC_CHROMA_FORMAT_SHIFT);
            hevc.flags |= CHROMA_FORMAT_420 << HEVC_CHROMA_FORMAT_SHIFT;
            hevc.idr_period = config.gop_length;
            hevc.input_bit_depth = depth;
            hevc.output_bit_depth = depth;
            if !b_frame_reference {
                hevc.use_b_frames_as_ref = BFRAME_REF_DISABLED;
            }
            apply_vui(&config.color, intra_only, &mut hevc.vui);
            cfg.codec.hevc = hevc;
        }
        EncodeCodec::Av1 => {
            // SAFETY: every member of the union holds only integers and raw
            // pointers, so any bytes are a valid value of each.
            let mut av1 = unsafe { cfg.codec.av1 };
            av1.flags &= !(AV1_FLAG_ANNEXB | AV1_FLAG_DISABLE_SEQ_HDR);
            av1.flags |= AV1_FLAG_REPEAT_SEQ_HDR;
            av1.flags &= !(0b11 << AV1_CHROMA_FORMAT_SHIFT);
            av1.flags |= CHROMA_FORMAT_420 << AV1_CHROMA_FORMAT_SHIFT;
            av1.idr_period = config.gop_length;
            av1.input_bit_depth = depth;
            av1.output_bit_depth = depth;
            if !b_frame_reference {
                av1.use_b_frames_as_ref = BFRAME_REF_DISABLED;
            }
            av1.color_primaries = c_int::from(config.color.colour_primaries);
            av1.transfer_characteristics = c_int::from(config.color.transfer_characteristics);
            av1.matrix_coefficients = c_int::from(config.color.matrix_coefficients);
            av1.color_range = u32::from(config.color.full_range);
            av1.chroma_sample_position = config
                .color
                .chroma_location
                .and_then(|location| location.to_av1())
                .map_or(0, u32::from);
            cfg.codec.av1 = av1;
        }
    }
}

/// The `nvEncInitializeEncoder` input. `cfg` must outlive the call that takes it.
///
/// # Errors
///
/// [`EncodeError::InvalidConfig`] from the frame rate or the aspect ratio.
pub(super) fn initialize_params(
    config: &EncodeConfig,
    cfg: *mut NvEncConfig,
) -> Result<NvEncInitializeParams, EncodeError> {
    let (frame_rate_num, frame_rate_den) = config.frame_rate_parts()?;
    let (dar_width, dar_height) = config.display_aspect_ratio()?;
    let mut params = NvEncInitializeParams::zeroed();
    params.version = INITIALIZE_PARAMS_VER;
    params.encode_guid = codec_guid(config.codec);
    params.preset_guid = preset_guid(config.preset);
    params.encode_width = config.width;
    params.encode_height = config.height;
    params.dar_width = dar_width;
    params.dar_height = dar_height;
    params.frame_rate_num = frame_rate_num;
    params.frame_rate_den = frame_rate_den;
    // Synchronous mode: asynchronous encode needs Windows events.
    params.enable_encode_async = 0;
    // The encoder picks each picture's type.
    params.enable_ptd = 1;
    params.encode_config = cfg;
    params.max_encode_width = config.width;
    params.max_encode_height = config.height;
    params.tuning_info = tuning_info(config.tuning);
    params.buffer_format = buffer_format(config.format);
    Ok(params)
}
