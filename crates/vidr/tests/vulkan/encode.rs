//! Vulkan Video encode on the local GPU: ffmpeg as a black box, a round trip through
//! `VulkanDecoder`, the crate's parsers on the driver's parameter sets, and an MP4 that ffprobe
//! reads.
//!
//! The PSNR floors sit below what a correct stream gives at quantiser 20, so a failure means the
//! frames reach the encoder wrong. They are not targets to tune. Each run prints what it measures.

use std::sync::Arc;

use numr::runtime::wgpu::WgpuRuntime;
use vidr::bitstream::{AvcConfig, HevcConfig, h264_sps_facts, h264_sps_vui, hevc_sps_vui};
use vidr::decode::{DecodeCodec, Decoder, PixelFormat};
use vidr::encode::{EncodeCodec, EncodeFrame, Encoder, RateControl};
use vidr::frame::{HostFrame, download_planes};
use vidr::vulkan::{VulkanDecoder, VulkanDevice, VulkanEncoder};
use vidr::{DecodeError, EncodeError, FrameRoute};

use super::encode_support::{
    DURATION, Encoded, HEIGHT, WIDTH, check_packets, config, encode_device, encode_pattern,
    ffmpeg_luma, lowest_psnr, open, write_annexb, write_obu,
};
use super::support::assert_no_validation_errors;

/// Luma PSNR floor at quantiser 20 for 8-bit samples, peak 255.
const FLOOR_8BIT_DB: f64 = 50.0;
/// Luma PSNR floor at quantiser 20 for 10-bit samples, peak 1023.
const FLOOR_10BIT_DB: f64 = 55.0;
/// Luma PSNR floor for a small frame, where the test picture's gradient is steep.
const SMALL_FLOOR_DB: f64 = 35.0;
const FRAMES: usize = 12;
const GOP: u32 = 5;

fn floor_and_peak(format: PixelFormat) -> (f64, f64) {
    match format {
        PixelFormat::P010 => (FLOOR_10BIT_DB, 1023.0),
        _ => (FLOOR_8BIT_DB, 255.0),
    }
}

/// One stream through ffmpeg: frame count, a silent decode, and the PSNR floor.
fn run_through_ffmpeg(codec: EncodeCodec, format: PixelFormat, name: &str) {
    let Some((device, _guard)) = encode_device() else {
        return;
    };
    let mut wanted = config(codec, format);
    wanted.gop_length = GOP;
    let Some(mut encoder) = open(&device, wanted, name) else {
        return;
    };
    assert_eq!(Encoder::<WgpuRuntime>::route(&encoder), FrameRoute::Device);
    eprintln!("{name}: rate control {}", encoder.rate_control());
    let encoded = encode_pattern(&device, &mut encoder, &wanted, FRAMES, name);
    check_packets(&encoded, GOP as usize, name);
    assert_no_validation_errors(&device, name);

    let path = match codec {
        EncodeCodec::Av1 => write_obu(&encoded, name),
        _ => write_annexb(codec, &encoded, name),
    };
    let Some(decoded) = ffmpeg_luma(&path, format, (WIDTH, HEIGHT), name) else {
        return;
    };
    let (floor, peak) = floor_and_peak(format);
    let lowest = lowest_psnr(&decoded, &encoded, peak, name);
    assert!(lowest >= floor, "{name}: lowest PSNR {lowest:.2} dB");
}

#[test]
fn h264_8_bit_decodes_in_ffmpeg() {
    run_through_ffmpeg(EncodeCodec::H264, PixelFormat::Nv12, "enc_h264");
}

#[test]
fn hevc_8_bit_decodes_in_ffmpeg() {
    run_through_ffmpeg(EncodeCodec::Hevc, PixelFormat::Nv12, "enc_hevc");
}

#[test]
fn hevc_10_bit_decodes_in_ffmpeg() {
    run_through_ffmpeg(EncodeCodec::Hevc, PixelFormat::P010, "enc_hevc_main10");
}

/// This skips with a notice on a driver with no `VK_KHR_video_encode_av1`.
#[test]
fn av1_decodes_in_ffmpeg_where_the_driver_lists_it() {
    run_through_ffmpeg(EncodeCodec::Av1, PixelFormat::Nv12, "enc_av1");
}

/// One encode at the shared config, or `None` after a notice.
fn encode_once(
    device: &Arc<VulkanDevice>,
    codec: EncodeCodec,
    format: PixelFormat,
    name: &str,
) -> Option<Encoded> {
    let mut wanted = config(codec, format);
    wanted.gop_length = GOP;
    let mut encoder = open(device, wanted, name)?;
    Some(encode_pattern(device, &mut encoder, &wanted, FRAMES, name))
}

#[test]
fn the_driver_avcc_parses_and_states_the_config() {
    let Some((device, _guard)) = encode_device() else {
        return;
    };
    let name = "enc_avcc";
    let Some(encoded) = encode_once(&device, EncodeCodec::H264, PixelFormat::Nv12, name) else {
        return;
    };
    let record =
        AvcConfig::parse(&encoded.config_record).unwrap_or_else(|error| panic!("avcC: {error}"));
    assert_eq!(record.length_size.bytes(), 4, "length-prefixed samples");
    let sps = record.sps.first().expect("one SPS");
    let facts = h264_sps_facts(sps).unwrap_or_else(|error| panic!("SPS: {error}"));
    assert_eq!(facts.profile_idc, 100, "High");
    assert_eq!(facts.chroma_format_idc, 1);
    assert_eq!(facts.bit_depth_luma, 8);
    assert!(facts.frame_mbs_only);
    let vui = h264_sps_vui(sps).unwrap_or_else(|error| panic!("VUI: {error}"));
    let signal = vui.video_signal_type.expect("the signal type");
    let colour = signal.colour.expect("the colour description");
    assert_eq!(colour.colour_primaries, 1);
    assert_eq!(colour.transfer_characteristics, 1);
    assert_eq!(colour.matrix_coefficients, 1);
    assert_eq!(record.pps.len(), 1);
}

#[test]
fn the_driver_hvcc_parses_and_states_the_config() {
    let Some((device, _guard)) = encode_device() else {
        return;
    };
    let name = "enc_hvcc";
    let Some(encoded) = encode_once(&device, EncodeCodec::Hevc, PixelFormat::Nv12, name) else {
        return;
    };
    let record =
        HevcConfig::parse(&encoded.config_record).unwrap_or_else(|error| panic!("hvcC: {error}"));
    assert_eq!(record.general_profile_idc, 1, "Main");
    let parsed: Vec<&[u8]> = record.nal_units().collect();
    assert!(parsed.len() >= 3, "VPS, SPS, and PPS");
    let sps = parsed
        .iter()
        .find(|unit| (unit[0] >> 1) & 0x3F == 33)
        .expect("an SPS");
    let vui = hevc_sps_vui(sps).unwrap_or_else(|error| panic!("VUI: {error}"));
    let signal = vui.video_signal_type.expect("the signal type");
    let colour = signal.colour.expect("the colour description");
    assert_eq!(colour.colour_primaries, 1);
    assert_eq!(colour.transfer_characteristics, 1);
    assert_eq!(colour.matrix_coefficients, 1);
}

/// Each picture of a GOP of one is a keyframe, and a longer GOP keeps its cadence past one wrap.
#[test]
fn keyframes_follow_the_gop_length() {
    let Some((device, _guard)) = encode_device() else {
        return;
    };
    for (gop, frames, name) in [(1_u32, 4, "enc_gop_1"), (3, 10, "enc_gop_3")] {
        let mut wanted = config(EncodeCodec::H264, PixelFormat::Nv12);
        wanted.gop_length = gop;
        let Some(mut encoder) = open(&device, wanted, name) else {
            return;
        };
        let encoded = encode_pattern(&device, &mut encoder, &wanted, frames, name);
        check_packets(&encoded, gop as usize, name);
        let path = write_annexb(EncodeCodec::H264, &encoded, name);
        if let Some(decoded) = ffmpeg_luma(&path, PixelFormat::Nv12, (WIDTH, HEIGHT), name) {
            assert_eq!(decoded.len(), frames, "{name}: frame count");
        }
    }
    assert_no_validation_errors(&device, "keyframes_follow_the_gop_length");
}

/// `VulkanDecoder` reads what `VulkanEncoder` writes.
fn round_trip(codec: EncodeCodec, format: PixelFormat, name: &str) {
    let Some((device, _guard)) = encode_device() else {
        return;
    };
    let Some(encoded) = encode_once(&device, codec, format, name) else {
        return;
    };
    let (floor, _) = floor_and_peak(format);
    check_round_trip(&device, codec, format, &encoded, floor, name);
}

/// A small frame through the encoder and back: `size`, or the driver's minimum on each side the
/// driver refuses. Each H.264 and HEVC session opens at the shared floor.
fn small_round_trip(codec: EncodeCodec, size: (u32, u32), name: &str) {
    let Some((device, _guard)) = encode_device() else {
        return;
    };
    let mut wanted = config(codec, PixelFormat::Nv12);
    wanted.gop_length = GOP;
    (wanted.width, wanted.height) = size;
    let mut encoder = match VulkanEncoder::new(&device, wanted) {
        Ok(encoder) => encoder,
        Err(EncodeError::SizeOutOfRange { minimum, .. }) => {
            eprintln!(
                "notice: {name}: the driver's minimum is {minimum:?}, and this test runs there"
            );
            wanted.width = size.0.max(minimum.0);
            wanted.height = size.1.max(minimum.1);
            VulkanEncoder::new(&device, wanted)
                .unwrap_or_else(|error| panic!("{name}: open at the minimum: {error}"))
        }
        Err(error) => panic!("{name}: open: {error}"),
    };
    let encoded = encode_pattern(&device, &mut encoder, &wanted, FRAMES, name);
    check_packets(&encoded, GOP as usize, name);
    check_round_trip(
        &device,
        codec,
        PixelFormat::Nv12,
        &encoded,
        SMALL_FLOOR_DB,
        name,
    );
}

#[test]
fn h264_at_64x64_round_trips_where_the_driver_lists_it() {
    small_round_trip(EncodeCodec::H264, (64, 64), "small_h264_64");
}

#[test]
fn h264_at_128x128_round_trips_where_the_driver_lists_it() {
    small_round_trip(EncodeCodec::H264, (128, 128), "small_h264_128");
}

#[test]
fn hevc_at_64x64_round_trips_where_the_driver_lists_it() {
    small_round_trip(EncodeCodec::Hevc, (64, 64), "small_hevc_64");
}

#[test]
fn hevc_at_128x128_round_trips_where_the_driver_lists_it() {
    small_round_trip(EncodeCodec::Hevc, (128, 128), "small_hevc_128");
}

/// Decodes `encoded` through `VulkanDecoder`, and holds its lowest luma PSNR to `floor`.
fn check_round_trip(
    device: &Arc<VulkanDevice>,
    codec: EncodeCodec,
    format: PixelFormat,
    encoded: &Encoded,
    floor: f64,
    name: &str,
) {
    let decode_codec = match codec {
        EncodeCodec::H264 => DecodeCodec::H264,
        EncodeCodec::Hevc => DecodeCodec::Hevc,
        _ => DecodeCodec::Av1,
    };
    let (_, peak) = floor_and_peak(format);
    let mut decoder = match VulkanDecoder::new(device, decode_codec, &encoded.config_record) {
        Ok(decoder) => decoder,
        // A driver's decode range can start above its encode range, so ffmpeg reads this size.
        Err(DecodeError::VulkanRefused { .. }) => {
            eprintln!("notice: {name}: the decoder refuses this size, so ffmpeg reads it.");
            let path = write_annexb(codec, encoded, name);
            if let Some(decoded) = ffmpeg_luma(&path, format, encoded.size, name) {
                let lowest = lowest_psnr(&decoded, encoded, peak, name);
                assert!(lowest >= floor, "{name}: lowest PSNR {lowest:.2} dB");
            }
            return;
        }
        Err(error) => panic!("{name}: decoder: {error}"),
    };
    let mut host = HostFrame::new();
    let mut decoded = Vec::new();
    let mut take = |frame: vidr::decode::DecodedFrame<WgpuRuntime>| {
        download_planes(&EncodeFrame::from(&frame), &mut host)
            .unwrap_or_else(|error| panic!("{name}: download: {error}"));
        let luma: Vec<u16> = match format {
            PixelFormat::P010 => host
                .y()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
                .collect(),
            _ => host.y().iter().map(|&byte| u16::from(byte)).collect(),
        };
        decoded.push(luma);
    };
    for packet in &encoded.packets {
        decoder
            .send(&packet.data, packet.pts)
            .unwrap_or_else(|error| panic!("{name}: send: {error}"));
        while let Some(frame) = decoder.receive() {
            take(frame);
        }
    }
    decoder
        .flush()
        .unwrap_or_else(|error| panic!("{name}: flush: {error}"));
    while let Some(frame) = decoder.receive() {
        take(frame);
    }
    let lowest = lowest_psnr(&decoded, encoded, peak, name);
    assert!(lowest >= floor, "{name}: lowest PSNR {lowest:.2} dB");
    assert_no_validation_errors(device, name);
}

#[test]
fn h264_round_trips_through_the_vulkan_decoder() {
    round_trip(EncodeCodec::H264, PixelFormat::Nv12, "round_h264");
}

#[test]
fn hevc_round_trips_through_the_vulkan_decoder() {
    round_trip(EncodeCodec::Hevc, PixelFormat::Nv12, "round_hevc");
}

#[test]
fn hevc_10_bit_round_trips_through_the_vulkan_decoder() {
    round_trip(EncodeCodec::Hevc, PixelFormat::P010, "round_hevc_main10");
}

#[test]
fn av1_round_trips_where_the_driver_lists_it() {
    round_trip(EncodeCodec::Av1, PixelFormat::Nv12, "round_av1");
}

#[test]
fn a_size_below_the_driver_minimum_is_refused_by_name() {
    let Some((device, _guard)) = encode_device() else {
        return;
    };
    for codec in [EncodeCodec::H264, EncodeCodec::Hevc] {
        let mut tiny = config(codec, PixelFormat::Nv12);
        tiny.width = 16;
        tiny.height = 16;
        match VulkanEncoder::new(&device, tiny) {
            Err(EncodeError::SizeOutOfRange { .. }) => {}
            Err(error) => panic!("{codec}: the wrong refusal: {error}"),
            Ok(_) => eprintln!("notice: this driver takes {codec} at 16x16"),
        }
    }
}

#[test]
fn b_frames_and_a_wrong_frame_are_refused_and_a_refused_frame_leaves_the_encoder_usable() {
    let Some((device, _guard)) = encode_device() else {
        return;
    };
    let mut asks = config(EncodeCodec::H264, PixelFormat::Nv12);
    asks.b_frames = 1;
    assert!(matches!(
        VulkanEncoder::new(&device, asks).err(),
        Some(EncodeError::BFramesUnsupported { .. })
    ));

    let name = "enc_refusals";
    let wanted = config(EncodeCodec::H264, PixelFormat::Nv12);
    let Some(mut encoder) = open(&device, wanted, name) else {
        return;
    };
    let empty = |shape: [usize; 2]| {
        numr::tensor::Tensor::<WgpuRuntime>::empty(
            &shape,
            numr::dtype::DType::U32,
            device.numr_device(),
        )
        .unwrap_or_else(|error| panic!("{name}: plane: {error}"))
    };
    let (short, uv) = (empty([10, 10]), empty([10, 10]));
    let display = vidr::Dimensions::new(WIDTH, HEIGHT).unwrap_or_else(|error| panic!("{error}"));
    let bad = EncodeFrame::from_packed_planes(&short, &uv, PixelFormat::Nv12, display);
    assert!(matches!(
        Encoder::<WgpuRuntime>::send(&mut encoder, &bad, 0, DURATION),
        Err(EncodeError::SurfaceMismatch { plane: "luma", .. })
    ));
    let encoded = encode_pattern(&device, &mut encoder, &wanted, 2, name);
    assert_eq!(encoded.packets.len(), 2, "{name}: the refusal stays local");
    assert!(matches!(
        Encoder::<WgpuRuntime>::send(&mut encoder, &bad, 1_000_000, DURATION),
        Err(EncodeError::AfterFlush { .. })
    ));
}

/// A rate-controlled stream runs too, and its encoder states the mode.
#[test]
fn a_target_bitrate_runs_under_rate_control() {
    let Some((device, _guard)) = encode_device() else {
        return;
    };
    let name = "enc_bitrate";
    let mut wanted = config(EncodeCodec::Hevc, PixelFormat::Nv12);
    wanted.rate_control = RateControl::Bitrate {
        average: 2_000_000,
        maximum: None,
    };
    wanted.gop_length = GOP;
    let Some(mut encoder) = open(&device, wanted, name) else {
        return;
    };
    assert_ne!(encoder.rate_control(), "constant quantiser");
    let encoded = encode_pattern(&device, &mut encoder, &wanted, FRAMES, name);
    check_packets(&encoded, GOP as usize, name);
    assert_no_validation_errors(&device, name);
}

/// Dropping an encoder mid-stream frees its objects, and the next encoder opens.
#[test]
fn a_dropped_encoder_lets_the_next_one_open() {
    let Some((device, _guard)) = encode_device() else {
        return;
    };
    let name = "enc_drop";
    let wanted = config(EncodeCodec::H264, PixelFormat::Nv12);
    for _ in 0..2 {
        let Some(mut encoder) = open(&device, wanted, name) else {
            return;
        };
        let encoded = encode_pattern(&device, &mut encoder, &wanted, 2, name);
        assert_eq!(encoded.packets.len(), 2);
    }
    assert_no_validation_errors(&device, name);
}

/// The record and samples go through containr's MP4 writer. ffprobe reads the codec, the size, and
/// the packet count, and ffmpeg decodes the file silently.
#[cfg(feature = "mp4")]
fn mux_and_probe(codec: EncodeCodec, ffprobe_name: &str, name: &str) {
    use std::fs::OpenOptions;
    use std::path::PathBuf;
    use std::process::Command;

    use vidr::media::{Input, OutputKind, OutputMux, VideoOutput, plan_copy};

    let Some((device, _guard)) = encode_device() else {
        return;
    };
    let Some(encoded) = encode_once(&device, codec, PixelFormat::Nv12, name) else {
        return;
    };
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("vulkan_encode");
    std::fs::create_dir_all(&directory).unwrap_or_else(|error| panic!("{name}: {error}"));
    let source = directory.join(format!("{name}_source.mp4"));
    let made = Command::new("ffmpeg")
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=25",
        ])
        .args(["-frames:v", "2", "-c:v", "mpeg4"])
        .arg(&source)
        .output();
    match made {
        Ok(output) if output.status.success() => {}
        other => {
            eprintln!("skipping the MP4 check of {name}: ffmpeg gives {other:?}");
            return;
        }
    }
    let mut input = Input::open(&source).unwrap_or_else(|error| panic!("{name}: {error}"));
    let plan =
        plan_copy(&mut input, OutputKind::Mp4).unwrap_or_else(|error| panic!("{name}: {error}"));
    let wanted = config(codec, PixelFormat::Nv12);
    let video = VideoOutput::new(
        codec,
        encoded.config_record.clone(),
        WIDTH,
        HEIGHT,
        wanted.time_base,
    );
    let output = directory.join(format!("{name}.mp4"));
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&output)
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    let mut mux =
        OutputMux::start(file, plan, video).unwrap_or_else(|error| panic!("{name}: {error}"));
    for packet in &encoded.packets {
        mux.write_video(packet)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
    }
    mux.finish()
        .unwrap_or_else(|error| panic!("{name}: {error}"));

    let probed = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-count_packets"])
        .args([
            "-show_entries",
            "stream=codec_name,width,height,nb_read_packets",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(&output)
        .output()
        .unwrap_or_else(|error| panic!("{name}: ffprobe: {error}"));
    let text = String::from_utf8_lossy(&probed.stdout);
    assert!(
        probed.status.success() && probed.stderr.is_empty(),
        "{name}: ffprobe: {}",
        String::from_utf8_lossy(&probed.stderr)
    );
    assert!(
        text.contains(&format!("codec_name={ffprobe_name}")),
        "{name}: {text}"
    );
    assert!(text.contains(&format!("width={WIDTH}")), "{name}: {text}");
    assert!(text.contains(&format!("height={HEIGHT}")), "{name}: {text}");
    assert!(
        text.contains(&format!("nb_read_packets={FRAMES}")),
        "{name}: {text}"
    );
    let decoded = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&output)
        .args(["-f", "null", "-"])
        .output()
        .unwrap_or_else(|error| panic!("{name}: ffmpeg: {error}"));
    assert!(
        decoded.status.success() && decoded.stderr.is_empty(),
        "{name}: ffmpeg decode: {}",
        String::from_utf8_lossy(&decoded.stderr)
    );
}

#[cfg(feature = "mp4")]
#[test]
fn avcc_goes_through_the_mp4_mux_and_ffprobe_reads_it() {
    mux_and_probe(EncodeCodec::H264, "h264", "mux_h264");
}

#[cfg(feature = "mp4")]
#[test]
fn hvcc_goes_through_the_mp4_mux_and_ffprobe_reads_it() {
    mux_and_probe(EncodeCodec::Hevc, "hevc", "mux_hevc");
}
