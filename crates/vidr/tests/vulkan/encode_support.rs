//! A moving test picture on the device, a full encode, and the checks of its output through
//! ffmpeg as a black box.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, MutexGuard};

use numr::runtime::wgpu::WgpuRuntime;
use vidr::bitstream::{AvcConfig, HevcConfig, ToAnnexB};
use vidr::decode::{ChromaLocation, ColorFacts, PixelFormat};
use vidr::encode::{EncodeCodec, EncodeConfig, EncodeFrame, EncodedPacket, Encoder, RateControl};
use vidr::frame::{HostPlanes, upload_planes};
use vidr::vulkan::{VulkanDevice, VulkanEncoder, vulkan_encode_available};
use vidr::{Dimensions, EncodeError, Rational};

use super::support;

pub(crate) const WIDTH: u32 = 640;
pub(crate) const HEIGHT: u32 = 360;
/// One frame at 30000/1001 in a time base of 1/30000.
pub(crate) const DURATION: i64 = 1001;
/// The first pts. It is not zero, so a test catches a dropped offset.
pub(crate) const FIRST_PTS: i64 = 7;

/// The device with an encode queue and the guard for this test. Prints a notice and returns `None`
/// with no device or no encode queue.
pub(crate) fn encode_device() -> Option<(Arc<VulkanDevice>, MutexGuard<'static, ()>)> {
    if let Err(error) = vulkan_encode_available() {
        eprintln!("skipping: {error}");
        return None;
    }
    support::device()
}

/// `WIDTH x HEIGHT` at 30000/1001, BT.709 limited with left chroma, constant quantiser 20.
pub(crate) fn config(codec: EncodeCodec, format: PixelFormat) -> EncodeConfig {
    let rate = Rational::new(30000, 1001).unwrap_or_else(|error| panic!("rate: {error}"));
    let base = Rational::new(1, 30000).unwrap_or_else(|error| panic!("base: {error}"));
    let mut config = EncodeConfig::new(codec, WIDTH, HEIGHT, format, rate, base);
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

/// Opens an encoder, or prints a notice and returns `None` when the driver leaves the codec or
/// depth out.
pub(crate) fn open(
    device: &Arc<VulkanDevice>,
    config: EncodeConfig,
    name: &str,
) -> Option<VulkanEncoder> {
    match VulkanEncoder::new(device, config) {
        Ok(encoder) => Some(encoder),
        Err(
            error
            @ (EncodeError::CodecUnsupported { .. } | EncodeError::BitDepthUnsupported { .. }),
        ) => {
            eprintln!("skipping {name}: {error}");
            None
        }
        Err(error) => panic!("{name}: open: {error}"),
    }
}

/// The luma sample at one position of frame `t`, from 0 to 1.
fn luma_at(column: usize, row: usize, t: usize, width: usize, height: usize) -> f32 {
    let across = column as f32 / width as f32;
    let down = row as f32 / height as f32;
    (0.5 * across + 0.3 * down + t as f32 / 64.0).fract()
}

/// The I420 or I010 planes of frame `t` and its luma at the coded bit depth.
fn test_picture(
    format: PixelFormat,
    size: (u32, u32),
    t: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u16>) {
    let (width, height) = (size.0 as usize, size.1 as usize);
    let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
    let peak = match format {
        PixelFormat::P010 => 1023.0,
        _ => 255.0,
    };
    let mut luma = Vec::with_capacity(width * height);
    for row in 0..height {
        for column in 0..width {
            luma.push((luma_at(column, row, t, width, height) * 0.8 * peak + 0.1 * peak) as u16);
        }
    }
    let flat = |value: f32| -> Vec<u16> {
        let sample = (value * peak) as u16;
        vec![sample; chroma_width * chroma_height]
    };
    let (u, v) = (flat(0.4), flat(0.6));
    let bytes = |samples: &[u16]| -> Vec<u8> {
        match format {
            PixelFormat::P010 => samples
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect(),
            _ => samples.iter().map(|&sample| sample as u8).collect(),
        }
    };
    (bytes(&luma), bytes(&u), bytes(&v), luma)
}

/// What one encode produced.
pub(crate) struct Encoded {
    pub(crate) packets: Vec<EncodedPacket>,
    pub(crate) config_record: Vec<u8>,
    /// Input luma of each frame, at the coded bit depth.
    pub(crate) luma: Vec<Vec<u16>>,
    /// Input pts of each frame.
    pub(crate) pts: Vec<i64>,
}

/// Encodes `frames` frames of the test picture through the backend-neutral trait.
pub(crate) fn encode_pattern(
    device: &VulkanDevice,
    encoder: &mut dyn Encoder<WgpuRuntime>,
    config: &EncodeConfig,
    frames: usize,
    name: &str,
) -> Encoded {
    assert!(
        encoder.config_record().is_none(),
        "{name}: a config record exists before the first frame"
    );
    let format = config.format;
    let size = (config.width, config.height);
    let display =
        Dimensions::new(size.0, size.1).unwrap_or_else(|error| panic!("{name}: size: {error}"));
    let sample = format.bytes_per_sample();
    let mut packets = Vec::new();
    let mut luma = Vec::with_capacity(frames);
    let mut pts_list = Vec::with_capacity(frames);
    for t in 0..frames {
        let (y, u, v, samples) = test_picture(format, size, t);
        let width = size.0 as usize;
        let planes = HostPlanes {
            y: &y,
            y_stride: width * sample,
            u: &u,
            u_stride: width.div_ceil(2) * sample,
            v: &v,
            v_stride: width.div_ceil(2) * sample,
            width,
            height: size.1 as usize,
        };
        let (y_plane, uv_plane) =
            upload_planes::<WgpuRuntime>(device.numr_device(), planes, format)
                .unwrap_or_else(|error| panic!("{name}: upload: {error}"));
        let frame = EncodeFrame::from_packed_planes(&y_plane, &uv_plane, format, display);
        let pts = FIRST_PTS + t as i64 * DURATION;
        pts_list.push(pts);
        luma.push(samples);
        encoder
            .send(&frame, pts, DURATION)
            .unwrap_or_else(|error| panic!("{name}: send at pts {pts}: {error}"));
        while let Some(packet) = encoder.receive() {
            packets.push(packet);
        }
    }
    encoder
        .flush()
        .unwrap_or_else(|error| panic!("{name}: flush: {error}"));
    while let Some(packet) = encoder.receive() {
        packets.push(packet);
    }
    let Some(config_record) = encoder.config_record().map(<[u8]>::to_vec) else {
        panic!("{name}: no config record after the stream");
    };
    Encoded {
        packets,
        config_record,
        luma,
        pts: pts_list,
    }
}

/// Luma PSNR at the coded bit depth.
pub(crate) fn psnr(a: &[u16], b: &[u16], peak: f64) -> f64 {
    assert_eq!(a.len(), b.len(), "luma sizes differ");
    let squares: f64 = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    let mean = squares / a.len() as f64;
    if mean == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (peak * peak / mean).log10()
    }
}

/// What every stream shares: count, timing, and the keyframe cadence of `gop`.
pub(crate) fn check_packets(encoded: &Encoded, gop: usize, name: &str) {
    let packets = &encoded.packets;
    assert_eq!(packets.len(), encoded.pts.len(), "{name}: packet count");
    for (index, (packet, pts)) in packets.iter().zip(&encoded.pts).enumerate() {
        assert_eq!(packet.pts, *pts, "{name}: pts of packet {index}");
        assert_eq!(packet.dts, *pts, "{name}: dts of packet {index}");
        assert_eq!(packet.duration, DURATION, "{name}: duration");
        assert!(!packet.data.is_empty(), "{name}: empty packet {index}");
        assert_eq!(
            packet.keyframe,
            index.is_multiple_of(gop),
            "{name}: keyframe flag of packet {index}"
        );
    }
}

fn scratch(name: &str, extension: &str) -> PathBuf {
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("vulkan_encode");
    std::fs::create_dir_all(&directory)
        .unwrap_or_else(|error| panic!("creating {}: {error}", directory.display()));
    directory.join(format!("{name}.{extension}"))
}

/// The H.264 or HEVC stream in Annex-B, with the record's parameter sets restored.
pub(crate) fn write_annexb(codec: EncodeCodec, encoded: &Encoded, name: &str) -> PathBuf {
    let (mut filter, extension) = match codec {
        EncodeCodec::H264 => {
            let record = AvcConfig::parse(&encoded.config_record)
                .unwrap_or_else(|error| panic!("{name}: avcC: {error}"));
            (ToAnnexB::h264(&record), "h264")
        }
        EncodeCodec::Hevc => {
            let record = HevcConfig::parse(&encoded.config_record)
                .unwrap_or_else(|error| panic!("{name}: hvcC: {error}"));
            (ToAnnexB::hevc(&record), "hevc")
        }
        other => panic!("{name}: no Annex-B form for {other}"),
    };
    let mut stream = Vec::new();
    for packet in &encoded.packets {
        filter
            .filter_into(&packet.data, &mut stream)
            .unwrap_or_else(|error| panic!("{name}: ToAnnexB: {error}"));
    }
    let path = scratch(name, extension);
    std::fs::write(&path, stream)
        .unwrap_or_else(|error| panic!("writing {}: {error}", path.display()));
    path
}

/// The AV1 stream as low-overhead OBUs, with a temporal delimiter before each sample.
pub(crate) fn write_obu(encoded: &Encoded, name: &str) -> PathBuf {
    let mut stream = Vec::new();
    for packet in &encoded.packets {
        stream.extend_from_slice(&[0x12, 0x00]);
        stream.extend_from_slice(&packet.data);
    }
    let path = scratch(name, "obu");
    std::fs::write(&path, stream)
        .unwrap_or_else(|error| panic!("writing {}: {error}", path.display()));
    path
}

/// Decodes `path` with ffmpeg to raw luma samples, one list per frame. ffmpeg must take the stream
/// silently. Prints a notice and returns `None` when ffmpeg does not run.
pub(crate) fn ffmpeg_luma(
    path: &Path,
    format: PixelFormat,
    size: (u32, u32),
    name: &str,
) -> Option<Vec<Vec<u16>>> {
    let pixel_format = match format {
        // A gray format makes ffmpeg rescale limited range to full, so read whole frames.
        PixelFormat::P010 => "yuv420p10le",
        _ => "yuv420p",
    };
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-f", "rawvideo", "-pix_fmt", pixel_format, "-"])
        .output();
    let output = match output {
        Ok(output) => output,
        Err(error) => {
            eprintln!("skipping the ffmpeg checks of {name}: {error}");
            return None;
        }
    };
    assert!(
        output.status.success() && output.stderr.is_empty(),
        "{name}: ffmpeg decode: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let samples: Vec<u16> = match format {
        PixelFormat::P010 => output
            .stdout
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect(),
        _ => output.stdout.iter().map(|&byte| u16::from(byte)).collect(),
    };
    let luma = size.0 as usize * size.1 as usize;
    let frame = luma + 2 * (size.0 as usize).div_ceil(2) * (size.1 as usize).div_ceil(2);
    assert!(
        samples.len().is_multiple_of(frame),
        "{name}: {} samples is no whole count of {frame}-sample frames",
        samples.len()
    );
    Some(
        samples
            .chunks(frame)
            .map(|picture| picture[..luma].to_vec())
            .collect(),
    )
}

/// The lowest PSNR of each frame against `encoded`'s input.
pub(crate) fn lowest_psnr(decoded: &[Vec<u16>], encoded: &Encoded, peak: f64, name: &str) -> f64 {
    assert_eq!(decoded.len(), encoded.luma.len(), "{name}: frame count");
    let lowest = decoded
        .iter()
        .zip(&encoded.luma)
        .map(|(got, wanted)| psnr(got, wanted, peak))
        .fold(f64::INFINITY, f64::min);
    eprintln!("{name}: lowest luma PSNR {lowest:.2} dB");
    lowest
}
