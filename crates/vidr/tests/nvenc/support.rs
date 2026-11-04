//! The GPU, a numr-generated moving gradient, and the checks of encoded output.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use numr::dtype::DType;
use numr::ops::{ChromaSiting, ColorMatrix, ColorOps, ColorRange, ConvLayout, YuvPlanes, YuvSpec};
use numr::runtime::RuntimeClient;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime, is_cuda_available};
use numr::tensor::Tensor;
use vidr::Rational;
use vidr::bitstream::{AvcConfig, HevcConfig, ToAnnexB};
use vidr::decode::ChromaLocation;
use vidr::encode::nvenc::nvenc_available;
use vidr::encode::{
    ColorFacts, EncodeCodec, EncodeConfig, EncodeFrame, EncodedPacket, Encoder, PixelFormat,
    RateControl,
};

pub(crate) const WIDTH: u32 = 640;
pub(crate) const HEIGHT: u32 = 360;
/// One frame at 30000/1001 in a time base of 1/30000.
pub(crate) const DURATION: i64 = 1001;
/// The first pts. It is not zero, so a test catches a dropped offset.
pub(crate) const FIRST_PTS: i64 = 7;

/// numr's client on CUDA device 0, or a notice and `None` when the NVIDIA
/// library or a CUDA device is missing.
pub(crate) fn gpu() -> Option<CudaClient> {
    if let Err(error) = nvenc_available() {
        eprintln!("skipping: {error}");
        return None;
    }
    if !is_cuda_available() {
        eprintln!("skipping: no CUDA device 0. Run the NVENC tests on a machine with a GPU.");
        return None;
    }
    match CudaClient::new(CudaDevice::new(0)) {
        Ok(client) => Some(client),
        Err(error) => {
            eprintln!("skipping: numr cannot open CUDA device 0: {error}");
            None
        }
    }
}

/// `640x360` at 30000/1001, BT.709 limited with left chroma, cq 20.
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

/// Float RGB of frame `t`, `[1, 3, H, W]`: red moves across the frame, green
/// falls down it, and blue runs against red.
fn gradient(client: &CudaClient, t: usize) -> Tensor<CudaRuntime> {
    let (width, height) = (WIDTH as usize, HEIGHT as usize);
    let mut rgb = vec![0.0_f32; 3 * width * height];
    for row in 0..height {
        for column in 0..width {
            let across = column as f32 / width as f32;
            let index = row * width + column;
            rgb[index] = (across + t as f32 / 64.0).fract();
            rgb[width * height + index] = row as f32 / height as f32;
            rgb[2 * width * height + index] = 1.0 - across;
        }
    }
    Tensor::from_slice(&rgb, &[1, 3, height, width], client.device())
        .unwrap_or_else(|error| panic!("gradient upload: {error}"))
}

/// The planes numr writes, narrowed from wider surfaces when `row_padding` is
/// above zero.
pub(crate) struct Planes {
    pub(crate) y: Tensor<CudaRuntime>,
    pub(crate) uv: Tensor<CudaRuntime>,
}

fn plane(
    client: &CudaClient,
    rows: usize,
    columns: usize,
    dtype: DType,
    row_padding: usize,
) -> Tensor<CudaRuntime> {
    let surface =
        Tensor::<CudaRuntime>::empty(&[rows, columns + row_padding], dtype, client.device())
            .unwrap_or_else(|error| panic!("plane allocation: {error}"));
    surface
        .narrow(1, 0, columns)
        .unwrap_or_else(|error| panic!("plane narrow: {error}"))
}

/// Writes frame `t` through `ColorOps::rgb_to_yuv_into`.
pub(crate) fn frame_planes(
    client: &CudaClient,
    format: PixelFormat,
    t: usize,
    row_padding: usize,
) -> Planes {
    let (width, height) = (WIDTH as usize, HEIGHT as usize);
    let dtype = match format {
        PixelFormat::Nv12 => DType::U8,
        PixelFormat::P010 => DType::U16,
        other => panic!("unexpected format {other}"),
    };
    let y = plane(client, height, width, dtype, row_padding);
    let uv = plane(client, height / 2, width, dtype, row_padding);
    let spec = YuvSpec::new(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left);
    let planes = match format {
        PixelFormat::P010 => YuvPlanes::P010 { y: &y, uv: &uv },
        _ => YuvPlanes::Nv12 { y: &y, uv: &uv },
    };
    client
        .rgb_to_yuv_into(&gradient(client, t), planes, &spec, ConvLayout::Nchw)
        .unwrap_or_else(|error| panic!("rgb_to_yuv_into: {error}"));
    Planes { y, uv }
}

/// Luma samples at the coded bit depth.
pub(crate) fn luma_samples(y: &Tensor<CudaRuntime>, format: PixelFormat) -> Vec<u16> {
    // `try_to_vec` refuses a pitched view.
    let y = y
        .contiguous()
        .unwrap_or_else(|error| panic!("packing luma: {error}"));
    match format {
        PixelFormat::P010 => y
            .try_to_vec::<u16>()
            .unwrap_or_else(|error| panic!("reading luma: {error}"))
            .iter()
            .map(|sample| sample >> 6)
            .collect(),
        _ => y
            .try_to_vec::<u8>()
            .unwrap_or_else(|error| panic!("reading luma: {error}"))
            .iter()
            .map(|&sample| u16::from(sample))
            .collect(),
    }
}

/// What one encode produced.
pub(crate) struct Encoded {
    pub(crate) packets: Vec<EncodedPacket>,
    pub(crate) config_record: Vec<u8>,
    /// Input luma of each frame, in display order.
    #[cfg_attr(
        not(feature = "nvdec"),
        allow(dead_code, reason = "only the NVDEC check reads it")
    )]
    pub(crate) luma: Vec<Vec<u16>>,
    /// Input pts of each frame, in display order.
    pub(crate) pts: Vec<i64>,
}

/// Encodes `frames` gradient frames through the backend-neutral trait.
pub(crate) fn encode_gradient(
    client: &CudaClient,
    encoder: &mut dyn Encoder<CudaRuntime>,
    format: PixelFormat,
    frames: usize,
    row_padding: usize,
    name: &str,
) -> Encoded {
    assert!(
        encoder.config_record().is_none(),
        "{name}: a config record exists before the first frame"
    );
    let mut packets = Vec::new();
    let mut luma = Vec::with_capacity(frames);
    let mut pts_list = Vec::with_capacity(frames);
    for t in 0..frames {
        let planes = frame_planes(client, format, t, row_padding);
        luma.push(luma_samples(&planes.y, format));
        let pts = FIRST_PTS + t as i64 * DURATION;
        pts_list.push(pts);
        encoder
            .send(&EncodeFrame::new(&planes.y, &planes.uv), pts, DURATION)
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
#[cfg_attr(
    not(feature = "nvdec"),
    allow(dead_code, reason = "only the NVDEC check reads it")
)]
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

/// The checks every stream shares: count, pts, durations, and the first key.
pub(crate) fn check_packets(encoded: &Encoded, name: &str) {
    let packets = &encoded.packets;
    assert_eq!(packets.len(), encoded.pts.len(), "{name}: packet count");
    assert!(packets[0].keyframe, "{name}: packet 0 is not a keyframe");
    assert!(
        packets.iter().all(|packet| packet.duration == DURATION),
        "{name}: a duration differs from {DURATION}"
    );
    assert!(
        packets.iter().all(|packet| !packet.data.is_empty()),
        "{name}: an empty packet"
    );
    let mut presented: Vec<i64> = packets.iter().map(|packet| packet.pts).collect();
    presented.sort_unstable();
    assert_eq!(presented, encoded.pts, "{name}: pts differ from the input");
    assert!(
        packets.windows(2).all(|pair| pair[0].dts < pair[1].dts),
        "{name}: dts does not increase strictly"
    );
    assert!(
        packets.iter().all(|packet| packet.dts <= packet.pts),
        "{name}: a dts exceeds its pts"
    );
}

/// The Annex-B stream, parameter sets restored, in a gitignored directory under
/// `target/`.
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
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("nvenc");
    std::fs::create_dir_all(&directory)
        .unwrap_or_else(|error| panic!("creating {}: {error}", directory.display()));
    let path = directory.join(format!("{name}.{extension}"));
    std::fs::write(&path, stream)
        .unwrap_or_else(|error| panic!("writing {}: {error}", path.display()));
    path
}

/// ffprobe's stream facts after counting frames, or a notice and `None` when
/// ffmpeg is missing. ffmpeg's software decoder must take the stream silently.
pub(crate) fn ffmpeg_facts(path: &Path, name: &str) -> Option<HashMap<String, String>> {
    let decode = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-f", "null", "-"])
        .output();
    let decode = match decode {
        Ok(output) => output,
        Err(error) => {
            eprintln!("skipping the ffmpeg checks of {name}: {error}");
            return None;
        }
    };
    assert!(
        decode.status.success() && decode.stderr.is_empty(),
        "{name}: ffmpeg decode: {}",
        String::from_utf8_lossy(&decode.stderr)
    );
    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-count_frames", "-select_streams", "v:0"])
        .args([
            "-show_entries",
            "stream",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(path)
        .output()
        .unwrap_or_else(|error| panic!("{name}: ffprobe: {error}"));
    assert!(
        probe.status.success(),
        "{name}: ffprobe: {}",
        String::from_utf8_lossy(&probe.stderr)
    );
    Some(
        String::from_utf8_lossy(&probe.stdout)
            .lines()
            .filter_map(|line| line.split_once('='))
            .map(|(key, value)| (key.to_owned(), value.to_owned()))
            .collect(),
    )
}

/// Decodes the samples on NVDEC and checks pts, size, colour, and luma PSNR.
#[cfg(feature = "nvdec")]
pub(crate) fn check_nvdec(
    client: &CudaClient,
    codec: EncodeCodec,
    format: PixelFormat,
    encoded: &Encoded,
    floor_db: f64,
    name: &str,
) {
    use vidr::decode::nvdec::NvdecDecoder;

    let mut decoder = NvdecDecoder::new(client, codec.into(), &encoded.config_record)
        .unwrap_or_else(|error| panic!("{name}: NVDEC open: {error}"));
    let mut frames = Vec::new();
    for packet in &encoded.packets {
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

    let pts: Vec<i64> = frames.iter().map(|frame| frame.pts()).collect();
    assert_eq!(pts, encoded.pts, "{name}: NVDEC display-order pts");
    let peak = f64::from((1_u32 << format.bit_depth()) - 1);
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(frame.format(), format, "{name}: frame {index} format");
        assert_eq!(
            frame.display().width(),
            WIDTH,
            "{name}: frame {index} width"
        );
        assert_eq!(
            frame.display().height(),
            HEIGHT,
            "{name}: frame {index} height"
        );
        let color = frame.color();
        assert_eq!(color.matrix_coefficients, 1, "{name}: matrix");
        assert_eq!(color.transfer_characteristics, 1, "{name}: transfer");
        assert_eq!(color.colour_primaries, 1, "{name}: primaries");
        assert!(!color.full_range, "{name}: range");
        assert_eq!(
            color.chroma_location,
            Some(ChromaLocation::Left),
            "{name}: siting"
        );
        let decoded = luma_samples(frame.y(), format);
        let db = psnr(&decoded, &encoded.luma[index], peak);
        assert!(db > floor_db, "{name}: frame {index} luma PSNR {db:.2} dB");
    }
}

/// Without the `nvdec` feature, ffmpeg's checks stand alone.
#[cfg(not(feature = "nvdec"))]
pub(crate) fn check_nvdec(
    _client: &CudaClient,
    _codec: EncodeCodec,
    _format: PixelFormat,
    _encoded: &Encoded,
    _floor_db: f64,
    name: &str,
) {
    eprintln!("skipping the NVDEC checks of {name}: build with the `nvdec` feature.");
}
