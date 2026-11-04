//! RGB through `from_rgb`, NVENC, NVDEC, and `to_rgb`.

use numr::dtype::DType;
use numr::ops::{ChromaSiting, ColorMatrix, ColorRange, ConvLayout, YuvSpec};
use numr::runtime::RuntimeClient;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use vidr::decode::nvdec::NvdecDecoder;
use vidr::decode::{ChromaLocation, PixelFormat};
use vidr::encode::nvenc::{NvencEncoder, nvenc_available};
use vidr::encode::{
    ColorFacts, EncodeCodec, EncodeConfig, EncodeFrame, EncodedPacket, Encoder, RateControl,
};
use vidr::frame::{from_rgb, to_rgb};
use vidr::{EncodeError, Rational};

use super::support::{pattern, psnr, rgb_shape, rgb_tensor, rgb_values};
use crate::nvdec::fixtures;

const WIDTH: usize = 640;
const HEIGHT: usize = 360;
const FRAMES: usize = 12;
/// One frame at 30000/1001 in a time base of 1/30000.
const DURATION: i64 = 1001;
/// RGB PSNR floor at cq 20, peak 1. It sits below measured values, so a
/// failure means the frames reach the encoder or decoder wrong.
const FLOOR_DB: f64 = 40.0;

fn config(codec: EncodeCodec, format: PixelFormat) -> EncodeConfig {
    let rate = Rational::new(30000, 1001).unwrap_or_else(|error| panic!("rate: {error}"));
    let base = Rational::new(1, 30000).unwrap_or_else(|error| panic!("base: {error}"));
    let mut config = EncodeConfig::new(codec, WIDTH as u32, HEIGHT as u32, format, rate, base);
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

/// Encodes the moving pattern, or a notice and `None` when the GPU has no
/// such encoder.
fn encode(
    client: &CudaClient,
    codec: EncodeCodec,
    format: PixelFormat,
    spec: &YuvSpec,
    name: &str,
) -> Option<(Vec<EncodedPacket>, Vec<u8>)> {
    let mut encoder = match NvencEncoder::new(client, config(codec, format)) {
        Ok(encoder) => encoder,
        Err(
            error
            @ (EncodeError::CodecUnsupported { .. } | EncodeError::BitDepthUnsupported { .. }),
        ) => {
            eprintln!("skipping {name}: {error}");
            return None;
        }
        Err(error) => panic!("{name}: open: {error}"),
    };
    let encoder: &mut dyn Encoder<CudaRuntime> = &mut encoder;
    let layout = ConvLayout::Nchw;
    let mut packets = Vec::new();
    for t in 0..FRAMES {
        let values = pattern(WIDTH, HEIGHT, t);
        let rgb = rgb_tensor::<CudaRuntime>(&values, WIDTH, HEIGHT, layout, client.device());
        let planes = from_rgb(client, &rgb, spec, format, layout)
            .unwrap_or_else(|error| panic!("{name}: from_rgb {t}: {error}"));
        encoder
            .send(&EncodeFrame::from(&planes), t as i64 * DURATION, DURATION)
            .unwrap_or_else(|error| panic!("{name}: send {t}: {error}"));
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
    let Some(record) = encoder.config_record().map(<[u8]>::to_vec) else {
        panic!("{name}: no config record after the stream");
    };
    Some((packets, record))
}

fn run(codec: EncodeCodec, format: PixelFormat, name: &str) {
    if let Err(error) = nvenc_available() {
        eprintln!("skipping {name}: {error}");
        return;
    }
    let Some(client) = fixtures::gpu() else {
        return;
    };
    let spec = YuvSpec::new(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left);
    let Some((packets, record)) = encode(&client, codec, format, &spec, name) else {
        return;
    };

    let mut decoder = NvdecDecoder::new(&client, codec.into(), &record)
        .unwrap_or_else(|error| panic!("{name}: NVDEC open: {error}"));
    let mut frames = Vec::new();
    for packet in &packets {
        decoder
            .send(&packet.data, packet.pts)
            .unwrap_or_else(|error| panic!("{name}: NVDEC at pts {}: {error}", packet.pts));
        while let Some(frame) = decoder.receive() {
            frames.push(frame);
        }
    }
    decoder
        .flush()
        .unwrap_or_else(|error| panic!("{name}: NVDEC flush: {error}"));
    while let Some(frame) = decoder.receive() {
        frames.push(frame);
    }
    assert_eq!(frames.len(), FRAMES, "{name}: decoded frames");

    for (t, frame) in frames.iter().enumerate() {
        assert_eq!(frame.pts(), t as i64 * DURATION, "{name}: display order");
        assert_eq!(frame.format(), format, "{name}: format");
        let layout = ConvLayout::Nhwc;
        let rgb = to_rgb(&client, frame, &spec, DType::F32, layout)
            .unwrap_or_else(|error| panic!("{name}: to_rgb {t}: {error}"));
        assert_eq!(rgb.shape(), rgb_shape(WIDTH, HEIGHT, layout), "{name}");
        let db = psnr(&pattern(WIDTH, HEIGHT, t), &rgb_values(&rgb, layout), 1.0);
        assert!(db > FLOOR_DB, "{name}: frame {t} RGB PSNR {db:.2} dB");
    }
}

#[test]
fn h264_nv12_loop_stays_above_the_floor() {
    run(EncodeCodec::H264, PixelFormat::Nv12, "frame_h264_nv12");
}

#[test]
fn hevc_p010_loop_stays_above_the_floor() {
    run(EncodeCodec::Hevc, PixelFormat::P010, "frame_hevc_p010");
}

#[test]
fn av1_nv12_loop_stays_above_the_floor() {
    run(EncodeCodec::Av1, PixelFormat::Nv12, "frame_av1_nv12");
}
