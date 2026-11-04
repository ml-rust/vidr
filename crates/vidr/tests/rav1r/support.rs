//! A tiny synthetic clip, the encode loop, and PSNR.
//!
//! The PSNR floor sits well below what the quantizer gives this clip, so a failure
//! means the frames reach the encoder wrong. It is not a target to tune.

use numr::dtype::DType;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use vidr::Rational;
use vidr::encode::rav1r::Rav1rEncoder;
use vidr::encode::{
    EncodeCodec, EncodeConfig, EncodeFrame, EncodedPacket, Encoder, PixelFormat, Preset,
    RateControl,
};

/// One frame at 30000/1001 in a time base of 1/30000.
pub(crate) const DURATION: i64 = 1001;
/// The first pts. It is not zero, so a test catches a dropped offset.
pub(crate) const FIRST_PTS: i64 = 7;
/// Frames in a clip.
pub(crate) const FRAMES: usize = 12;
/// PSNR floor of a whole frame at `cq` 20, at each format's own peak.
pub(crate) const FLOOR_DB: f64 = 38.0;

pub(crate) fn rate() -> Rational {
    Rational::new(30000, 1001).unwrap_or_else(|error| panic!("rate: {error}"))
}

pub(crate) fn base() -> Rational {
    Rational::new(1, 30000).unwrap_or_else(|error| panic!("base: {error}"))
}

pub(crate) const fn pts_of(index: usize) -> i64 {
    FIRST_PTS + index as i64 * DURATION
}

/// A clip's size and sample layout.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Clip {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) format: PixelFormat,
    /// The first frame of a second scene, which holds other luma.
    pub(crate) cut: Option<usize>,
}

/// The clips of the decode-back tests: 8 and 10 bits, even and odd sizes.
pub(crate) const CLIPS: [Clip; 4] = [
    Clip::new(96, 64, PixelFormat::Nv12),
    Clip::new(96, 64, PixelFormat::P010),
    Clip::new(97, 65, PixelFormat::Nv12),
    Clip::new(97, 65, PixelFormat::P010),
];

impl Clip {
    pub(crate) const fn new(width: usize, height: usize, format: PixelFormat) -> Self {
        Self {
            width,
            height,
            format,
            cut: None,
        }
    }

    pub(crate) const fn chroma_size(&self) -> (usize, usize) {
        (self.width.div_ceil(2), self.height.div_ceil(2))
    }

    /// `(peak sample, 8-bit value scale)` of the format.
    const fn scale(&self) -> (u16, u16) {
        match self.format {
            PixelFormat::P010 => (1023, 4),
            _ => (255, 1),
        }
    }

    pub(crate) const fn peak(&self) -> u16 {
        self.scale().0
    }

    /// Samples in one frame.
    pub(crate) const fn samples(&self) -> usize {
        let (chroma_width, chroma_height) = self.chroma_size();
        self.width * self.height + 2 * chroma_width * chroma_height
    }

    /// The Y, U, and V samples of frame `t`, in that order: luma moves across the
    /// frame, and chroma runs each way.
    pub(crate) fn frame(&self, t: usize) -> Vec<u16> {
        let scale = f64::from(self.scale().1);
        let (width, height) = (self.width as f64, self.height as f64);
        let (chroma_width, chroma_height) = self.chroma_size();
        let second_scene = self.cut.is_some_and(|cut| t >= cut);
        let mut samples = Vec::with_capacity(self.samples());
        for row in 0..self.height {
            for column in 0..self.width {
                let across = (((column + 4 * t) as f64) / width).fract();
                let down = row as f64 / height;
                let value = if second_scene {
                    235.0 - 219.0 * (0.3 * across + 0.7 * down)
                } else {
                    16.0 + 219.0 * (0.7 * across + 0.3 * down)
                };
                samples.push((value * scale) as u16);
            }
        }
        for _row in 0..chroma_height {
            for column in 0..chroma_width {
                let value = 64.0 + (column * 128) as f64 / chroma_width as f64;
                samples.push((value * scale) as u16);
            }
        }
        for row in 0..chroma_height {
            for _column in 0..chroma_width {
                let value = 64.0 + (row * 128) as f64 / chroma_height as f64;
                samples.push((value * scale) as u16);
            }
        }
        samples
    }

    fn stored(&self, samples: &[u16]) -> Vec<u8> {
        match self.format {
            PixelFormat::P010 => samples
                .iter()
                .flat_map(|sample| (sample << 6).to_le_bytes())
                .collect(),
            _ => samples.iter().map(|sample| *sample as u8).collect(),
        }
    }

    /// The Y and UV planes of frame `t`, as an encoder takes them.
    pub(crate) fn planes(&self, t: usize) -> (Tensor<CpuRuntime>, Tensor<CpuRuntime>) {
        let samples = self.frame(t);
        let luma = self.width * self.height;
        let (chroma_width, chroma_height) = self.chroma_size();
        let chroma = chroma_width * chroma_height;
        let (y, rest) = samples.split_at(luma);
        let (u, v) = rest.split_at(chroma);
        let interleaved: Vec<u16> = u.iter().zip(v).flat_map(|(cb, cr)| [*cb, *cr]).collect();
        let dtype = match self.format {
            PixelFormat::P010 => DType::U16,
            _ => DType::U8,
        };
        let plane = |bytes: Vec<u8>, shape: [usize; 2]| {
            Tensor::<CpuRuntime>::from_bytes(&bytes, &shape, dtype, &CpuDevice::new())
                .unwrap_or_else(|error| panic!("plane of shape {shape:?}: {error}"))
        };
        (
            plane(self.stored(y), [self.height, self.width]),
            plane(self.stored(&interleaved), [chroma_height, 2 * chroma_width]),
        )
    }

    /// The test config: the fastest preset, `cq` 20, no reorder.
    pub(crate) fn config(&self) -> EncodeConfig {
        let mut config = EncodeConfig::new(
            EncodeCodec::Av1,
            self.width as u32,
            self.height as u32,
            self.format,
            rate(),
            base(),
        );
        config.preset = Preset::P1;
        config.rate_control = RateControl::VbrQuality { cq: 20 };
        config
    }
}

/// Encodes `frames` frames through the backend-neutral trait, and returns the
/// packets and the config record.
pub(crate) fn encode(
    clip: &Clip,
    config: EncodeConfig,
    frames: usize,
    name: &str,
) -> (Vec<EncodedPacket>, Vec<u8>) {
    let mut encoder = Rav1rEncoder::<CpuRuntime>::new(config)
        .unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let encoder: &mut dyn Encoder<CpuRuntime> = &mut encoder;
    assert!(
        encoder.config_record().is_none(),
        "{name}: a config record exists before the first frame"
    );
    let mut packets = Vec::new();
    for t in 0..frames {
        let (y, uv) = clip.planes(t);
        encoder
            .send(&EncodeFrame::new(&y, &uv), pts_of(t), DURATION)
            .unwrap_or_else(|error| panic!("{name}: send frame {t}: {error}"));
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
    (packets, record)
}

/// PSNR of two sample runs at `peak`.
pub(crate) fn psnr(a: &[u16], b: &[u16], peak: u16) -> f64 {
    assert_eq!(a.len(), b.len(), "sample counts differ");
    let squares: f64 = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    let mean = squares / a.len() as f64;
    if mean == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (f64::from(peak) * f64::from(peak) / mean).log10()
    }
}

/// Checks every decoded frame of `decoded` against the clip, above the floor.
pub(crate) fn check_psnr(clip: &Clip, decoded: &[Vec<u16>], name: &str) {
    assert_eq!(decoded.len(), FRAMES, "{name}: decoded frames");
    for (t, samples) in decoded.iter().enumerate() {
        let quality = psnr(samples, &clip.frame(t), clip.peak());
        assert!(
            quality > FLOOR_DB,
            "{name}: frame {t} PSNR {quality:.2} dB, floor {FLOOR_DB} dB"
        );
    }
}
