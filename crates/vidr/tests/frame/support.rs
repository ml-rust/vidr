//! Sample planes, RGB patterns, and readback shared by the frame tests.

use numr::dtype::DType;
use numr::ops::{ChromaSiting, ColorMatrix, ColorRange, ConvLayout, YuvSpec};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use vidr::Dimensions;
use vidr::decode::{ColorFacts, DecodedFrame, PixelFormat};

pub(crate) const FORMATS: [PixelFormat; 2] = [PixelFormat::Nv12, PixelFormat::P010];
pub(crate) const LAYOUTS: [ConvLayout; 2] = [ConvLayout::Nchw, ConvLayout::Nhwc];
pub(crate) const DTYPES: [DType; 2] = [DType::F32, DType::F16];
/// Every odd shape: both sides odd, then each side alone.
pub(crate) const ODD_SIZES: [(usize, usize); 3] = [(321, 181), (321, 180), (320, 181)];

/// numr's CPU client and device.
pub(crate) fn cpu() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    (client, device)
}

/// Left-sited specs for BT.601 and BT.709 in limited and full range.
pub(crate) fn specs() -> [YuvSpec; 4] {
    let spec = |matrix, range| YuvSpec::new(matrix, range, ChromaSiting::Left);
    [
        spec(ColorMatrix::Bt601, ColorRange::Limited),
        spec(ColorMatrix::Bt601, ColorRange::Full),
        spec(ColorMatrix::Bt709, ColorRange::Limited),
        spec(ColorMatrix::Bt709, ColorRange::Full),
    ]
}

/// Sample values of one frame at the format's bit depth, in the plane shapes
/// `DecodedFrame` documents.
pub(crate) struct Samples {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) format: PixelFormat,
    pub(crate) y: Vec<u16>,
    pub(crate) uv: Vec<u16>,
}

impl Samples {
    /// Builds a frame from functions of `(row, column)` that return 8-bit
    /// codes. `chroma` takes chroma coordinates and returns `(U, V)`. P010
    /// holds each code times 4.
    pub(crate) fn from_fn(
        (width, height): (usize, usize),
        format: PixelFormat,
        luma: impl Fn(usize, usize) -> u16,
        chroma: impl Fn(usize, usize) -> (u16, u16),
    ) -> Self {
        let scale = 1_u16 << (format.bit_depth() - 8);
        let mut y = Vec::with_capacity(width * height);
        for row in 0..height {
            y.extend((0..width).map(|column| luma(row, column) * scale));
        }
        let (rows, columns) = (height.div_ceil(2), width.div_ceil(2));
        let mut uv = Vec::with_capacity(rows * columns * 2);
        for row in 0..rows {
            for column in 0..columns {
                let (u, v) = chroma(row, column);
                uv.extend([u * scale, v * scale]);
            }
        }
        Self {
            width,
            height,
            format,
            y,
            uv,
        }
    }

    /// Sinusoids in luma and chroma, with no sharp chroma edge.
    pub(crate) fn smooth(size: (usize, usize), format: PixelFormat) -> Self {
        let (width, height) = (size.0 as f64, size.1 as f64);
        let (rows, columns) = ((height / 2.0).ceil(), (width / 2.0).ceil());
        let tau = std::f64::consts::TAU;
        Self::from_fn(
            size,
            format,
            |row, column| {
                let across = (tau * column as f64 / width + 0.3).sin();
                let down = (tau * row as f64 / height).cos();
                code(16.0 + 219.0 * (0.5 + 0.4 * across * down))
            },
            |row, column| {
                let u = 128.0 + 70.0 * (tau * column as f64 / columns + 0.5).sin();
                let v = 128.0 + 70.0 * (tau * row as f64 / rows + 0.2).cos();
                (code(u), code(v))
            },
        )
    }

    /// Splits one frame of ffmpeg's `nv12` raw bytes.
    pub(crate) fn from_nv12(bytes: &[u8], width: usize, height: usize) -> Self {
        let (y, uv) = bytes.split_at(width * height);
        Self {
            width,
            height,
            format: PixelFormat::Nv12,
            y: y.iter().map(|&sample| u16::from(sample)).collect(),
            uv: uv.iter().map(|&sample| u16::from(sample)).collect(),
        }
    }

    /// The chroma plane's `[rows, columns]`.
    pub(crate) fn chroma_shape(&self) -> [usize; 2] {
        [self.height.div_ceil(2), self.width.div_ceil(2) * 2]
    }

    /// The even frame with the same chroma, and luma extended by replication
    /// on the odd side.
    pub(crate) fn replicated(&self) -> Self {
        let width = self.width.div_ceil(2) * 2;
        let height = self.height.div_ceil(2) * 2;
        let mut y = Vec::with_capacity(width * height);
        for row in 0..height {
            let start = row.min(self.height - 1) * self.width;
            let source = &self.y[start..start + self.width];
            y.extend_from_slice(source);
            y.extend(std::iter::repeat_n(
                source[self.width - 1],
                width - self.width,
            ));
        }
        Self {
            width,
            height,
            format: self.format,
            y,
            uv: self.uv.clone(),
        }
    }

    /// The planes as ffmpeg's `nv12` or `p010le` raw bytes.
    pub(crate) fn raw(&self) -> Vec<u8> {
        let samples = self.y.iter().chain(&self.uv);
        match self.format {
            PixelFormat::Nv12 => samples.map(|&sample| sample as u8).collect(),
            _ => samples
                .flat_map(|&sample| (sample << 6).to_le_bytes())
                .collect(),
        }
    }

    /// One plane as stored samples on `device`.
    pub(crate) fn plane<R: Runtime<DType = DType>>(
        &self,
        samples: &[u16],
        shape: [usize; 2],
        device: &R::Device,
    ) -> Tensor<R> {
        let plane = match self.format {
            PixelFormat::Nv12 => {
                let bytes: Vec<u8> = samples.iter().map(|&sample| sample as u8).collect();
                Tensor::<R>::from_slice(&bytes, &shape, device)
            }
            _ => {
                let words: Vec<u16> = samples.iter().map(|&sample| sample << 6).collect();
                Tensor::<R>::from_slice(&words, &shape, device)
            }
        };
        plane.unwrap_or_else(|error| panic!("plane upload: {error}"))
    }

    /// The samples as a decoded frame on `device`.
    pub(crate) fn frame<R: Runtime<DType = DType>>(&self, device: &R::Device) -> DecodedFrame<R> {
        let y = self.plane(&self.y, [self.height, self.width], device);
        let uv = self.plane(&self.uv, self.chroma_shape(), device);
        planes_frame(y, uv, self.format, self.width, self.height)
    }
}

/// A frame from planes the test built, with unspecified colour facts.
pub(crate) fn planes_frame<R: Runtime>(
    y: Tensor<R>,
    uv: Tensor<R>,
    format: PixelFormat,
    width: usize,
    height: usize,
) -> DecodedFrame<R> {
    let display = Dimensions::new(width as u32, height as u32)
        .unwrap_or_else(|error| panic!("display size: {error}"));
    let color = ColorFacts {
        matrix_coefficients: 2,
        transfer_characteristics: 2,
        colour_primaries: 2,
        full_range: false,
        chroma_location: None,
    };
    DecodedFrame::from_planes(y, uv, format, 0, display, color)
}

fn code(value: f64) -> u16 {
    value.round_ties_even() as u16
}

/// Smooth in-gamut RGB at position `(row, column)`, moving with `t`.
fn pattern_pixel(
    (width, height): (usize, usize),
    (row, column): (usize, usize),
    t: usize,
) -> [f32; 3] {
    let tau = std::f64::consts::TAU;
    let phase = tau * t as f64 / 64.0;
    let (w, h) = (width as f64, height as f64);
    let (r, c) = (row as f64, column as f64);
    [
        (0.5 + 0.4 * (tau * c / w + 0.3 + phase).sin()) as f32,
        (0.5 + 0.4 * (tau * r / h).cos()) as f32,
        (0.5 + 0.3 * (tau * (c + r) / (w + h) - phase).sin()) as f32,
    ]
}

/// The pattern of [`pattern_pixel`], with channels last.
pub(crate) fn pattern(width: usize, height: usize, t: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(3 * width * height);
    for row in 0..height {
        for column in 0..width {
            values.extend(pattern_pixel((width, height), (row, column), t));
        }
    }
    values
}

/// F32 RGB of shape `[1, 3, H, W]` or `[1, H, W, 3]` from channels-last values.
pub(crate) fn rgb_tensor<R: Runtime<DType = DType>>(
    values: &[f32],
    width: usize,
    height: usize,
    layout: ConvLayout,
    device: &R::Device,
) -> Tensor<R> {
    let made = match layout {
        ConvLayout::Nhwc => Tensor::<R>::from_slice(values, &[1, height, width, 3], device),
        ConvLayout::Nchw => {
            let mut planar: Vec<f32> = Vec::with_capacity(values.len());
            for channel in 0..3 {
                planar.extend(values.iter().skip(channel).step_by(3));
            }
            Tensor::<R>::from_slice(&planar, &[1, 3, height, width], device)
        }
    };
    made.unwrap_or_else(|error| panic!("RGB upload: {error}"))
}

/// The value of an F16 bit pattern.
fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let fraction = f32::from(bits & 0x3ff);
    match exponent {
        0 => sign * fraction * 2.0_f32.powi(-24),
        31 => panic!("F16 bits {bits:#06x} are not finite"),
        _ => sign * (1.0 + fraction / 1024.0) * 2.0_f32.powi(exponent - 15),
    }
}

/// F32 or F16 RGB as channels-last `f32` values on the host.
pub(crate) fn rgb_values<R: Runtime<DType = DType>>(
    rgb: &Tensor<R>,
    layout: ConvLayout,
) -> Vec<f32> {
    let dense = rgb
        .contiguous()
        .unwrap_or_else(|error| panic!("packing RGB: {error}"));
    let values: Vec<f32> = if dense.dtype() == DType::F16 {
        dense
            .try_to_vec::<u16>()
            .unwrap_or_else(|error| panic!("reading RGB: {error}"))
            .into_iter()
            .map(half_to_f32)
            .collect()
    } else {
        dense
            .try_to_vec::<f32>()
            .unwrap_or_else(|error| panic!("reading RGB: {error}"))
    };
    if layout == ConvLayout::Nhwc {
        return values;
    }
    let (height, width) = (dense.shape()[2], dense.shape()[3]);
    let mut packed = Vec::with_capacity(values.len());
    for row in 0..height {
        for column in 0..width {
            for channel in 0..3 {
                packed.push(values[(channel * height + row) * width + column]);
            }
        }
    }
    packed
}

/// The shape of one RGB frame in `layout`.
pub(crate) fn rgb_shape(width: usize, height: usize, layout: ConvLayout) -> [usize; 4] {
    match layout {
        ConvLayout::Nchw => [1, 3, height, width],
        ConvLayout::Nhwc => [1, height, width, 3],
    }
}

/// Sample values of a plane on the host: `raw >> 6` for P010.
pub(crate) fn plane_values<R: Runtime<DType = DType>>(
    plane: &Tensor<R>,
    format: PixelFormat,
) -> Vec<u16> {
    let dense = plane
        .contiguous()
        .unwrap_or_else(|error| panic!("packing a plane: {error}"));
    match format {
        PixelFormat::Nv12 => dense
            .try_to_vec::<u8>()
            .unwrap_or_else(|error| panic!("reading a plane: {error}"))
            .into_iter()
            .map(u16::from)
            .collect(),
        _ => dense
            .try_to_vec::<u16>()
            .unwrap_or_else(|error| panic!("reading a plane: {error}"))
            .into_iter()
            .map(|sample| sample >> 6)
            .collect(),
    }
}

/// Channels-last rows of `[H, W]` from channels-last values of `even_width`
/// columns.
pub(crate) fn crop_rows<T: Copy>(
    values: &[T],
    even_width: usize,
    width: usize,
    height: usize,
) -> Vec<T> {
    let mut cropped = Vec::with_capacity(3 * width * height);
    for row in 0..height {
        let start = 3 * row * even_width;
        cropped.extend_from_slice(&values[start..start + 3 * width]);
    }
    cropped
}

/// RGB in `[0, 1]` as 8-bit codes.
pub(crate) fn to_codes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .map(|value| (value * 255.0).round_ties_even().clamp(0.0, 255.0) as u8)
        .collect()
}

/// PSNR in dB for samples of peak value `peak`.
pub(crate) fn psnr<T: Copy + Into<f64>>(a: &[T], b: &[T], peak: f64) -> f64 {
    assert_eq!(a.len(), b.len(), "PSNR inputs differ in length");
    let squares: f64 = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| {
            let (a, b): (f64, f64) = (a.into(), b.into());
            (a - b).powi(2)
        })
        .sum();
    let mean = squares / a.len() as f64;
    if mean == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (peak * peak / mean).log10()
    }
}

/// The largest absolute difference between two sets of 8-bit codes.
pub(crate) fn max_difference(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len(), "code slices differ in length");
    a.iter()
        .zip(b)
        .map(|(&a, &b)| a.abs_diff(b))
        .max()
        .unwrap_or(0)
}
