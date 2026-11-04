//! NVENC output decoded by NVDEC and ffmpeg.
//!
//! The PSNR floors sit below measured values, so a failure means the frames
//! reach the encoder wrong. They are not targets to tune.

use numr::runtime::cuda::CudaRuntime;
use vidr::bitstream::{AvcConfig, HevcConfig};
use vidr::encode::nvenc::NvencEncoder;
use vidr::encode::{EncodeCodec, EncodeConfig, Encoder, PixelFormat, RateControl};

use super::support::{self, HEIGHT, WIDTH};

/// Luma PSNR floor at cq 20 for 8-bit samples, peak 255.
const FLOOR_8BIT_DB: f64 = 45.0;
/// Luma PSNR floor at cq 20 for 10-bit samples, peak 1023.
const FLOOR_10BIT_DB: f64 = 50.0;

const FRAMES: usize = 12;

fn run(codec: EncodeCodec, format: PixelFormat, row_padding: usize, name: &str) {
    let Some(client) = support::gpu() else {
        return;
    };
    let config = support::config(codec, format);
    let mut encoder =
        NvencEncoder::new(&client, config).unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let encoded = {
        let encoder: &mut dyn Encoder<CudaRuntime> = &mut encoder;
        support::encode_gradient(&client, encoder, format, FRAMES, row_padding, name)
    };
    support::check_packets(&encoded, name);
    assert!(
        encoded
            .packets
            .iter()
            .all(|packet| packet.dts == packet.pts),
        "{name}: dts differs from pts without B-frames"
    );

    let floor = match format {
        PixelFormat::P010 => FLOOR_10BIT_DB,
        _ => FLOOR_8BIT_DB,
    };
    support::check_nvdec(&client, codec, format, &encoded, floor, name);

    let path = support::write_annexb(codec, &encoded, name);
    let Some(facts) = support::ffmpeg_facts(&path, name) else {
        return;
    };
    let fact = |key: &str| facts.get(key).map_or("", String::as_str);
    assert_eq!(
        fact("nb_read_frames"),
        FRAMES.to_string(),
        "{name}: ffmpeg frames"
    );
    assert_eq!(fact("width"), WIDTH.to_string(), "{name}: ffprobe width");
    assert_eq!(fact("height"), HEIGHT.to_string(), "{name}: ffprobe height");
    assert_eq!(fact("color_range"), "tv", "{name}: range");
    assert_eq!(fact("color_space"), "bt709", "{name}: matrix");
    assert_eq!(fact("color_transfer"), "bt709", "{name}: transfer");
    assert_eq!(fact("color_primaries"), "bt709", "{name}: primaries");
    assert_eq!(fact("chroma_location"), "left", "{name}: siting");
    let (codec_name, profile, pixel_format) = match (codec, format) {
        (EncodeCodec::H264, _) => ("h264", "High", "yuv420p"),
        (EncodeCodec::Hevc, PixelFormat::P010) => ("hevc", "Main 10", "yuv420p10le"),
        _ => ("hevc", "Main", "yuv420p"),
    };
    assert_eq!(fact("codec_name"), codec_name, "{name}: codec");
    assert_eq!(fact("profile"), profile, "{name}: profile");
    assert_eq!(fact("pix_fmt"), pixel_format, "{name}: bit depth");
}

#[test]
fn h264_nv12_decodes_back() {
    run(EncodeCodec::H264, PixelFormat::Nv12, 0, "nvenc_h264_nv12");
}

#[test]
fn h264_nv12_from_pitched_planes_decodes_back() {
    run(
        EncodeCodec::H264,
        PixelFormat::Nv12,
        64,
        "nvenc_h264_pitched",
    );
}

#[test]
fn hevc_nv12_decodes_back() {
    run(EncodeCodec::Hevc, PixelFormat::Nv12, 0, "nvenc_hevc_nv12");
}

#[test]
fn hevc_p010_decodes_back() {
    run(EncodeCodec::Hevc, PixelFormat::P010, 0, "nvenc_hevc_p010");
}

fn encode_short(config: EncodeConfig, name: &str) -> Option<Vec<u8>> {
    let client = support::gpu()?;
    let format = config.format;
    let mut encoder =
        NvencEncoder::new(&client, config).unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let encoded = support::encode_gradient(&client, &mut encoder, format, 2, 0, name);
    Some(encoded.config_record)
}

#[test]
fn avcc_parses_and_describes_the_stream() {
    let name = "nvenc_avcc";
    let config = support::config(EncodeCodec::H264, PixelFormat::Nv12);
    let Some(record) = encode_short(config, name) else {
        return;
    };
    let parsed = AvcConfig::parse(&record).unwrap_or_else(|error| panic!("{name}: {error}"));
    assert_eq!(parsed.profile_indication, 100, "{name}: High profile");
    assert_eq!(parsed.sps.len(), 1, "{name}: SPS count");
    assert_eq!(parsed.pps.len(), 1, "{name}: PPS count");
    let extension = parsed.high_profile.as_ref();
    assert_eq!(
        extension.map(|high| high.chroma_format),
        Some(1),
        "{name}: 4:2:0"
    );
}

#[test]
fn hvcc_parses_and_describes_the_stream() {
    let name = "nvenc_hvcc";
    let config = support::config(EncodeCodec::Hevc, PixelFormat::P010);
    let Some(record) = encode_short(config, name) else {
        return;
    };
    let parsed = HevcConfig::parse(&record).unwrap_or_else(|error| panic!("{name}: {error}"));
    assert_eq!(parsed.general_profile_idc, 2, "{name}: Main 10 profile");
    assert_eq!(parsed.nal_units().count(), 3, "{name}: VPS, SPS, PPS");
}

#[test]
fn const_qp_and_bitrate_modes_encode() {
    let modes = [
        RateControl::ConstQp {
            intra: 22,
            inter_p: 24,
            inter_b: 26,
        },
        RateControl::Bitrate {
            average: 2_000_000,
            maximum: Some(4_000_000),
        },
    ];
    for rate_control in modes {
        let mut config = support::config(EncodeCodec::H264, PixelFormat::Nv12);
        config.rate_control = rate_control;
        if encode_short(config, "nvenc_rate_control").is_none() {
            return;
        }
    }
}

#[test]
fn keyframes_follow_the_gop_length() {
    let name = "nvenc_gop";
    let Some(client) = support::gpu() else {
        return;
    };
    let mut config = support::config(EncodeCodec::H264, PixelFormat::Nv12);
    config.gop_length = 4;
    let mut encoder =
        NvencEncoder::new(&client, config).unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let encoded = support::encode_gradient(&client, &mut encoder, config.format, 9, 0, name);
    let keys: Vec<usize> = encoded
        .packets
        .iter()
        .enumerate()
        .filter(|(_, packet)| packet.keyframe)
        .map(|(index, _)| index)
        .collect();
    assert_eq!(keys, [0, 4, 8], "{name}: keyframe indices");
}
