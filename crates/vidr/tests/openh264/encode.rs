//! `Openh264Encoder` output decoded back by ffmpeg and `Openh264Decoder`.
//!
//! The PSNR floor sits below what the QP step gives at QP 20, so a failure means
//! the frames reach the encoder wrong. It is not a target to tune.

use std::path::PathBuf;
use std::process::Command;

use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
use vidr::bitstream::{AvcConfig, LengthSize, ToAnnexB};
use vidr::decode::ChromaLocation;
use vidr::encode::openh264::Openh264Encoder;
use vidr::encode::{
    ColorFacts, EncodeCodec, EncodeConfig, EncodeFrame, EncodedPacket, Encoder, PixelFormat,
    RateControl,
};
use vidr::{
    Dimensions, DurationHints, EncodeError, FrameRoute, FrameTiming, Geometry, Rational, Rotation,
    SampleAspectRatio, StreamFacts, check_output,
};

use super::support::{self, plane, psnr};
use crate::decode_fixtures::cpu::decode_all;

const WIDTH: usize = 640;
const HEIGHT: usize = 360;
/// One frame at 30000/1001 in a time base of 1/30000.
const DURATION: i64 = 1001;
/// The first pts. It is not zero, so a test catches a dropped offset.
const FIRST_PTS: i64 = 7;
const FRAMES: usize = 12;
/// PSNR floor at `cq` 20, peak 255, over every plane.
const FLOOR_DB: f64 = 36.0;

fn rate() -> Rational {
    Rational::new(30000, 1001).unwrap_or_else(|error| panic!("rate: {error}"))
}

fn base() -> Rational {
    Rational::new(1, 30000).unwrap_or_else(|error| panic!("base: {error}"))
}

/// `640x360` at 30000/1001, BT.709 limited, `cq` 20.
fn config() -> EncodeConfig {
    let mut config = EncodeConfig::new(
        EncodeCodec::H264,
        WIDTH as u32,
        HEIGHT as u32,
        PixelFormat::Nv12,
        rate(),
        base(),
    );
    config.rate_control = RateControl::VbrQuality { cq: 20 };
    config.color = ColorFacts {
        matrix_coefficients: 1,
        transfer_characteristics: 1,
        colour_primaries: 1,
        full_range: false,
        chroma_location: Some(ChromaLocation::Left),
    };
    config
}

/// Frame `t` as NV12: luma moves across the frame, and chroma runs each way.
fn nv12_frame(t: usize) -> (Vec<u8>, Vec<u8>) {
    let mut y = Vec::with_capacity(WIDTH * HEIGHT);
    for row in 0..HEIGHT {
        for column in 0..WIDTH {
            let across = (column + 4 * t) as f32 / WIDTH as f32;
            let down = row as f32 / HEIGHT as f32;
            y.push((16.0 + 219.0 * (0.7 * across.fract() + 0.3 * down)) as u8);
        }
    }
    let mut uv = Vec::with_capacity(WIDTH * HEIGHT / 2);
    for row in 0..HEIGHT / 2 {
        for column in 0..WIDTH / 2 {
            uv.push((64 + column * 128 / (WIDTH / 2)) as u8);
            uv.push((64 + row * 128 / (HEIGHT / 2)) as u8);
        }
    }
    (y, uv)
}

fn pts_of(index: usize) -> i64 {
    FIRST_PTS + index as i64 * DURATION
}

fn send_frame(encoder: &mut dyn Encoder<CpuRuntime>, t: usize, name: &str) {
    let (y, uv) = nv12_frame(t);
    let y = plane(&y, [HEIGHT, WIDTH], DType::U8);
    let uv = plane(&uv, [HEIGHT / 2, WIDTH], DType::U8);
    encoder
        .send(&EncodeFrame::new(&y, &uv), pts_of(t), DURATION)
        .unwrap_or_else(|error| panic!("{name}: send frame {t}: {error}"));
}

/// Encodes `frames` frames through the backend-neutral trait.
fn encode(
    config: EncodeConfig,
    frames: usize,
    name: &str,
) -> Option<(Vec<EncodedPacket>, Vec<u8>)> {
    let mut encoder =
        support::encoder(config)?.unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let encoder: &mut dyn Encoder<CpuRuntime> = &mut encoder;
    assert_eq!(encoder.route(), FrameRoute::HostOnly, "{name}: route");
    assert!(
        encoder.config_record().is_none(),
        "{name}: a config record exists before the first frame"
    );
    let mut packets = Vec::new();
    for t in 0..frames {
        send_frame(encoder, t, name);
        // Each frame gives its packet in the same call.
        let Some(packet) = encoder.receive() else {
            panic!("{name}: no packet after frame {t}");
        };
        packets.push(packet);
        assert!(
            encoder.config_record().is_some(),
            "{name}: no config record"
        );
    }
    encoder
        .flush()
        .unwrap_or_else(|error| panic!("{name}: flush: {error}"));
    assert!(encoder.receive().is_none(), "{name}: a packet after flush");
    let Some(record) = encoder.config_record().map(<[u8]>::to_vec) else {
        panic!("{name}: no config record after the stream");
    };
    Some((packets, record))
}

/// Pts and durations as sent, dts equal to pts, and the first packet a keyframe.
#[test]
fn packets_keep_their_timing() {
    let name = "openh264_timing";
    let Some((packets, _)) = encode(config(), FRAMES, name) else {
        return;
    };
    assert_eq!(packets.len(), FRAMES, "{name}: packet count");
    assert!(packets[0].keyframe, "{name}: packet 0 is not a keyframe");
    for (index, packet) in packets.iter().enumerate() {
        assert_eq!(packet.pts, pts_of(index), "{name}: packet {index} pts");
        assert_eq!(packet.dts, packet.pts, "{name}: packet {index} dts");
        assert_eq!(packet.duration, DURATION, "{name}: packet {index} duration");
        assert!(!packet.data.is_empty(), "{name}: packet {index} is empty");
    }
}

/// The frame count, size, aspect ratio, pts, and total duration pass the same
/// check a run's output takes.
#[test]
fn packets_pass_the_output_check() {
    let name = "openh264_check";
    let Some((packets, _)) = encode(config(), FRAMES, name) else {
        return;
    };
    let size = Dimensions::new(WIDTH as u32, HEIGHT as u32)
        .unwrap_or_else(|error| panic!("{name}: size: {error}"));
    let geometry = Geometry::new(size, Rotation::default(), SampleAspectRatio::SQUARE);
    let hints = DurationHints {
        packet_duration: Some(DURATION),
        average_frame_rate: Some(rate()),
        base_frame_rate: Some(rate()),
    };
    let facts = |frames: Vec<FrameTiming>| StreamFacts::new(geometry, base(), frames, hints);
    let source: Vec<FrameTiming> = (0..FRAMES)
        .map(|index| FrameTiming::new(pts_of(index), None, DURATION))
        .collect();
    let output: Vec<FrameTiming> = packets
        .iter()
        .map(|packet| FrameTiming::new(packet.pts, Some(packet.dts), packet.duration))
        .collect();
    check_output(&facts(source), &facts(output), size)
        .unwrap_or_else(|error| panic!("{name}: {error}"));
}

/// `avcC` holds one SPS and one PPS, Baseline, and four-byte lengths.
#[test]
fn config_record_describes_the_stream() {
    let name = "openh264_avcc";
    let Some((_, record)) = encode(config(), 2, name) else {
        return;
    };
    let parsed = AvcConfig::parse(&record).unwrap_or_else(|error| panic!("{name}: {error}"));
    assert_eq!(parsed.profile_indication, 66, "{name}: profile");
    assert_eq!(parsed.length_size, LengthSize::Four, "{name}: lengths");
    assert_eq!(parsed.sps.len(), 1, "{name}: SPS count");
    assert_eq!(parsed.pps.len(), 1, "{name}: PPS count");
}

/// Keyframes fall on the config's period, with no scene-change keyframes.
#[test]
fn keyframes_follow_the_gop_length() {
    let name = "openh264_gop";
    let mut short = config();
    short.gop_length = 4;
    let Some((packets, _)) = encode(short, 9, name) else {
        return;
    };
    let keys: Vec<usize> = packets
        .iter()
        .enumerate()
        .filter_map(|(index, packet)| packet.keyframe.then_some(index))
        .collect();
    assert_eq!(keys, [0, 4, 8], "{name}: keyframe positions");
}

/// PSNR of each decoded plane against the input of frame `index`.
fn check_planes(y: &[u8], uv: &[u8], index: usize, floor: f64, name: &str) -> f64 {
    let (input_y, input_uv) = nv12_frame(index);
    let luma = psnr(y, &input_y);
    let chroma = psnr(uv, &input_uv);
    assert!(luma > floor, "{name}: frame {index} luma PSNR {luma:.2} dB");
    assert!(
        chroma > floor,
        "{name}: frame {index} chroma PSNR {chroma:.2} dB"
    );
    luma
}

/// Decodes the samples with `Openh264Decoder`, checks pts, size, colour, and PSNR
/// above `floor`, and returns the mean luma PSNR.
fn decode_back(packets: &[EncodedPacket], record: &[u8], floor: f64, name: &str) -> Option<f64> {
    let mut decoder =
        support::decoder(record)?.unwrap_or_else(|error| panic!("{name}: decoder open: {error}"));
    let units = packets
        .iter()
        .map(|packet| (packet.data.clone(), packet.pts));
    let frames = decode_all(&mut decoder, units, name);
    assert_eq!(frames.len(), packets.len(), "{name}: decoded frames");
    let mut total = 0.0;
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(frame.pts(), pts_of(index), "{name}: frame {index} pts");
        assert_eq!(frame.format(), PixelFormat::Nv12, "{name}: frame {index}");
        assert_eq!(frame.display().width(), WIDTH as u32, "{name}: width");
        assert_eq!(frame.display().height(), HEIGHT as u32, "{name}: height");
        let color = frame.color();
        assert_eq!(color.matrix_coefficients, 1, "{name}: matrix");
        assert_eq!(color.transfer_characteristics, 1, "{name}: transfer");
        assert_eq!(color.colour_primaries, 1, "{name}: primaries");
        assert!(!color.full_range, "{name}: range");
        let read = |plane: &numr::tensor::Tensor<CpuRuntime>| {
            plane
                .to_bytes()
                .unwrap_or_else(|error| panic!("{name}: reading a plane: {error}"))
        };
        total += check_planes(&read(frame.y()), &read(frame.uv()), index, floor, name);
    }
    Some(total / frames.len() as f64)
}

#[test]
fn decodes_back_through_openh264() {
    let name = "openh264_decode_back";
    let Some((packets, record)) = encode(config(), FRAMES, name) else {
        return;
    };
    decode_back(&packets, &record, FLOOR_DB, name);
}

/// The Annex-B stream, parameter sets restored, in a directory under `target/`.
fn write_annexb(packets: &[EncodedPacket], record: &[u8], name: &str) -> PathBuf {
    let parsed = AvcConfig::parse(record).unwrap_or_else(|error| panic!("{name}: avcC: {error}"));
    let mut filter = ToAnnexB::h264(&parsed);
    let mut stream = Vec::new();
    for packet in packets {
        filter
            .filter_into(&packet.data, &mut stream)
            .unwrap_or_else(|error| panic!("{name}: ToAnnexB: {error}"));
    }
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("openh264");
    std::fs::create_dir_all(&directory)
        .unwrap_or_else(|error| panic!("creating {}: {error}", directory.display()));
    let path = directory.join(format!("{name}.h264"));
    std::fs::write(&path, stream)
        .unwrap_or_else(|error| panic!("writing {}: {error}", path.display()));
    path
}

/// ffmpeg decodes the stream silently, above the floor, and ffprobe reads the
/// profile and the colour tags.
#[test]
fn decodes_back_through_ffmpeg() {
    let name = "openh264_ffmpeg";
    let Some((packets, record)) = encode(config(), FRAMES, name) else {
        return;
    };
    let path = write_annexb(&packets, &record, name);
    let decode = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&path)
        .args(["-f", "rawvideo", "-pix_fmt", "nv12", "-"])
        .output();
    let decode = match decode {
        Ok(output) => output,
        Err(error) => {
            eprintln!("skipping the ffmpeg checks of {name}: {error}");
            return;
        }
    };
    assert!(
        decode.status.success() && decode.stderr.is_empty(),
        "{name}: ffmpeg: {}",
        String::from_utf8_lossy(&decode.stderr)
    );
    let frame_bytes = WIDTH * HEIGHT * 3 / 2;
    assert_eq!(
        decode.stdout.len(),
        frame_bytes * FRAMES,
        "{name}: ffmpeg frame bytes"
    );
    for (index, frame) in decode.stdout.chunks_exact(frame_bytes).enumerate() {
        let (y, uv) = frame.split_at(WIDTH * HEIGHT);
        check_planes(y, uv, index, FLOOR_DB, name);
    }

    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
        .args(["stream", "-of", "default=noprint_wrappers=1"])
        .arg(&path)
        .output()
        .unwrap_or_else(|error| panic!("{name}: ffprobe: {error}"));
    let text = String::from_utf8_lossy(&probe.stdout);
    let fact = |key: &str| {
        text.lines()
            .filter_map(|line| line.split_once('='))
            .find_map(|(found, value)| (found == key).then(|| value.to_owned()))
            .unwrap_or_default()
    };
    assert_eq!(fact("codec_name"), "h264", "{name}: codec");
    assert!(fact("profile").contains("Baseline"), "{name}: profile");
    assert_eq!(fact("width"), WIDTH.to_string(), "{name}: width");
    assert_eq!(fact("height"), HEIGHT.to_string(), "{name}: height");
    assert_eq!(fact("pix_fmt"), "yuv420p", "{name}: pixel format");
    assert_eq!(fact("color_range"), "tv", "{name}: range");
    assert_eq!(fact("color_space"), "bt709", "{name}: matrix");
    assert_eq!(fact("color_transfer"), "bt709", "{name}: transfer");
    assert_eq!(fact("color_primaries"), "bt709", "{name}: primaries");
}

fn stream_bytes(packets: &[EncodedPacket]) -> usize {
    packets.iter().map(|packet| packet.data.len()).sum()
}

/// A lower `cq` spends more bytes and gives a higher PSNR, so the QP reaches the
/// encoder. `ConstQp` with one value is the same mapping, so it gives the same size.
#[test]
fn quality_settings_reach_the_encoder() {
    let name = "openh264_quality";
    let with = |rate_control| {
        let mut config = config();
        config.rate_control = rate_control;
        config
    };
    let Some((fine, fine_record)) = encode(with(RateControl::VbrQuality { cq: 12 }), FRAMES, name)
    else {
        return;
    };
    let Some((coarse, coarse_record)) =
        encode(with(RateControl::VbrQuality { cq: 40 }), FRAMES, name)
    else {
        return;
    };
    assert!(
        stream_bytes(&fine) > stream_bytes(&coarse),
        "{name}: cq 12 spends {} bytes, cq 40 spends {}",
        stream_bytes(&fine),
        stream_bytes(&coarse)
    );
    let Some(fine_db) = decode_back(&fine, &fine_record, 0.0, name) else {
        return;
    };
    let Some(coarse_db) = decode_back(&coarse, &coarse_record, 0.0, name) else {
        return;
    };
    assert!(
        fine_db > coarse_db,
        "{name}: cq 12 gives {fine_db:.2} dB, cq 40 gives {coarse_db:.2} dB"
    );

    let constant = RateControl::ConstQp {
        intra: 40,
        inter_p: 40,
        inter_b: 40,
    };
    let Some((same, _)) = encode(with(constant), FRAMES, name) else {
        return;
    };
    assert_eq!(
        stream_bytes(&same),
        stream_bytes(&coarse),
        "{name}: ConstQp"
    );
}

/// A higher target spends more bytes, and every frame still gives a picture.
#[test]
fn bitrate_reaches_the_encoder() {
    let name = "openh264_bitrate";
    let with = |average| {
        let mut config = config();
        config.rate_control = RateControl::Bitrate {
            average,
            maximum: None,
        };
        config
    };
    let Some((low, low_record)) = encode(with(100_000), FRAMES, name) else {
        return;
    };
    let Some((high, _)) = encode(with(8_000_000), FRAMES, name) else {
        return;
    };
    assert_eq!(low.len(), FRAMES, "{name}: no frame is skipped");
    assert!(
        stream_bytes(&high) > stream_bytes(&low),
        "{name}: 8 Mb/s spends {} bytes, 100 kb/s spends {}",
        stream_bytes(&high),
        stream_bytes(&low)
    );
    decode_back(&low, &low_record, 0.0, name);
}

fn open_or_skip(name: &str) -> Option<Openh264Encoder<CpuRuntime>> {
    Some(support::encoder(config())?.unwrap_or_else(|error| panic!("{name}: open: {error}")))
}

fn refused_field(config: EncodeConfig) -> Option<&'static str> {
    match support::encoder(config)? {
        Err(EncodeError::InvalidConfig { field, .. }) => Some(field),
        Err(other) => panic!("expected InvalidConfig, got {other:?}: {other}"),
        Ok(_) => panic!("the config opens, and the test expects a refusal"),
    }
}

#[test]
fn a_config_openh264_cannot_take_is_refused_by_name() {
    let mut p010 = config();
    p010.format = PixelFormat::P010;
    let mut b_frames = config();
    b_frames.b_frames = 2;
    let mut hevc = config();
    hevc.codec = EncodeCodec::Hevc;
    let mut split = config();
    split.rate_control = RateControl::ConstQp {
        intra: 20,
        inter_p: 24,
        inter_b: 28,
    };
    for (config, field) in [
        (p010, "format"),
        (b_frames, "b_frames"),
        (hevc, "codec"),
        (split, "rate_control"),
    ] {
        let Some(found) = refused_field(config) else {
            return;
        };
        assert_eq!(found, field);
    }
}

/// An odd size has its own error, from the shared config check.
#[test]
fn an_odd_size_is_refused() {
    let mut odd = config();
    odd.width = 641;
    let Some(result) = support::encoder(odd) else {
        return;
    };
    assert!(matches!(
        result,
        Err(EncodeError::OddSize { width: 641, .. })
    ));
}

#[test]
fn a_frame_that_differs_from_the_config_leaves_the_encoder_usable() {
    let name = "openh264_refused_frames";
    let Some(mut encoder) = open_or_skip(name) else {
        return;
    };
    let encoder: &mut dyn Encoder<CpuRuntime> = &mut encoder;

    // Half the width, then P010 samples at the right size.
    let small_y = plane(&vec![0; HEIGHT * WIDTH / 2], [HEIGHT, WIDTH / 2], DType::U8);
    let small_uv = plane(
        &vec![128; HEIGHT * WIDTH / 4],
        [HEIGHT / 2, WIDTH / 2],
        DType::U8,
    );
    let result = encoder.send(&EncodeFrame::new(&small_y, &small_uv), 0, DURATION);
    assert!(
        matches!(result, Err(EncodeError::SurfaceMismatch { .. })),
        "{name}: small frame: {result:?}"
    );
    let wide_y = plane(&vec![0; 2 * HEIGHT * WIDTH], [HEIGHT, WIDTH], DType::U16);
    let wide_uv = plane(&vec![0; HEIGHT * WIDTH], [HEIGHT / 2, WIDTH], DType::U16);
    let result = encoder.send(&EncodeFrame::new(&wide_y, &wide_uv), 0, DURATION);
    assert!(
        matches!(result, Err(EncodeError::SurfaceMismatch { .. })),
        "{name}: P010 frame: {result:?}"
    );

    send_frame(encoder, 0, name);
    assert!(encoder.receive().is_some(), "{name}: no packet");
}

#[test]
fn calls_out_of_order_are_refused() {
    let name = "openh264_call_order";
    let Some(mut encoder) = open_or_skip(name) else {
        return;
    };
    let encoder: &mut dyn Encoder<CpuRuntime> = &mut encoder;
    let (y, uv) = nv12_frame(0);
    let y = plane(&y, [HEIGHT, WIDTH], DType::U8);
    let uv = plane(&uv, [HEIGHT / 2, WIDTH], DType::U8);
    let frame = EncodeFrame::new(&y, &uv);
    assert_eq!(
        encoder.send(&frame, 0, 0),
        Err(EncodeError::NonPositiveDuration {
            pts: 0,
            duration: 0,
        })
    );
    assert_eq!(encoder.send(&frame, 10, DURATION), Ok(()));
    assert_eq!(
        encoder.send(&frame, 10, DURATION),
        Err(EncodeError::PtsNotIncreasing {
            pts: 10,
            previous: 10,
        })
    );
    assert_eq!(encoder.flush(), Ok(()));
    assert_eq!(encoder.flush(), Ok(()));
    assert_eq!(
        encoder.send(&frame, 20, DURATION),
        Err(EncodeError::AfterFlush { pts: 20 })
    );
}
