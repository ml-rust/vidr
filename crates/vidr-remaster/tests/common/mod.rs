//! Synthetic frames, plane readback, and PSNR shared by the transform tests.

// Each test crate calls only some of these helpers.
#![allow(dead_code)]

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use vidr::Dimensions;
use vidr::color::{ColorPlan, color_plan};
use vidr::decode::{ColorFacts, DecodedFrame, MATRIX_BT709, PixelFormat};

pub const FORMATS: [PixelFormat; 2] = [PixelFormat::Nv12, PixelFormat::P010];

pub fn dims(width: usize, height: usize) -> Dimensions {
    Dimensions::new(width as u32, height as u32).expect("nonzero size")
}

/// BT.709, limited range, left-sited.
pub fn facts() -> ColorFacts {
    ColorFacts {
        matrix_coefficients: MATRIX_BT709,
        transfer_characteristics: 1,
        colour_primaries: 1,
        full_range: false,
        chroma_location: None,
    }
}

pub fn plan() -> ColorPlan {
    color_plan(&facts()).expect("BT.709 plan")
}

/// Largest code of a format: the PSNR peak.
pub fn peak(format: PixelFormat) -> f64 {
    f64::from((1_u32 << format.bit_depth()) - 1)
}

/// A plane from 8-bit codes. P010 holds each code times 4, in the high bits.
fn plane<R: Runtime<DType = DType>>(
    codes: &[u16],
    shape: [usize; 2],
    format: PixelFormat,
    device: &R::Device,
) -> Tensor<R> {
    let plane = match format {
        PixelFormat::Nv12 => {
            let bytes: Vec<u8> = codes.iter().map(|&code| code as u8).collect();
            Tensor::<R>::from_slice(&bytes, &shape, device)
        }
        _ => {
            let words: Vec<u16> = codes.iter().map(|&code| (code * 4) << 6).collect();
            Tensor::<R>::from_slice(&words, &shape, device)
        }
    };
    plane.expect("plane upload")
}

/// A frame from functions of `(row, column)` that return 8-bit codes.
fn frame_from<R: Runtime<DType = DType>>(
    (width, height): (usize, usize),
    format: PixelFormat,
    device: &R::Device,
    luma: impl Fn(usize, usize) -> u16,
    chroma: impl Fn(usize, usize) -> (u16, u16),
) -> DecodedFrame<R> {
    let mut y = Vec::with_capacity(width * height);
    for row in 0..height {
        y.extend((0..width).map(|column| luma(row, column)));
    }
    let (rows, columns) = (height.div_ceil(2), width.div_ceil(2));
    let mut uv = Vec::with_capacity(rows * columns * 2);
    for row in 0..rows {
        for column in 0..columns {
            let (u, v) = chroma(row, column);
            uv.extend([u, v]);
        }
    }
    DecodedFrame::from_planes(
        plane(&y, [height, width], format, device),
        plane(&uv, [rows, columns * 2], format, device),
        format,
        0,
        dims(width, height),
        facts(),
    )
}

/// One colour across the frame, from 8-bit codes.
pub fn flat_frame<R: Runtime<DType = DType>>(
    size: (usize, usize),
    format: PixelFormat,
    (y, u, v): (u16, u16, u16),
    device: &R::Device,
) -> DecodedFrame<R> {
    frame_from(size, format, device, |_, _| y, |_, _| (u, v))
}

/// Diagonal ramps in luma and chroma, inside the limited range.
pub fn ramp_frame<R: Runtime<DType = DType>>(
    size: (usize, usize),
    format: PixelFormat,
    device: &R::Device,
) -> DecodedFrame<R> {
    frame_from(
        size,
        format,
        device,
        |row, column| 40 + ((row * 3 + column * 2) % 160) as u16,
        |row, column| {
            (
                100 + ((row * 5) % 56) as u16,
                100 + ((column * 5) % 56) as u16,
            )
        },
    )
}

/// Sample values of a plane on the host, at the format's bit depth.
pub fn samples<R: Runtime<DType = DType>>(plane: &Tensor<R>, format: PixelFormat) -> Vec<u16> {
    let dense = plane.contiguous().expect("packing a plane");
    match format {
        PixelFormat::Nv12 => dense
            .try_to_vec::<u8>()
            .expect("reading a plane")
            .into_iter()
            .map(u16::from)
            .collect(),
        _ => dense
            .try_to_vec::<u16>()
            .expect("reading a plane")
            .into_iter()
            .map(|sample| sample >> 6)
            .collect(),
    }
}

/// PSNR in dB for samples of peak value `peak`.
pub fn psnr(a: &[u16], b: &[u16], peak: f64) -> f64 {
    assert_eq!(a.len(), b.len(), "PSNR inputs differ in length");
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
