//! Frames and sources the crop tests share.

use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

use super::rect::CropRect;
use super::window::Window;
use crate::decode::{
    ColorFacts, DecodedFrame, MATRIX_UNSPECIFIED, PRIMARIES_UNSPECIFIED, PixelFormat, PlaneForm,
    TRANSFER_UNSPECIFIED,
};
use crate::geometry::Dimensions;

/// Colour that no stream states. A crop carries colour through unread.
pub(super) const UNSTATED: ColorFacts = ColorFacts {
    matrix_coefficients: MATRIX_UNSPECIFIED,
    transfer_characteristics: TRANSFER_UNSPECIFIED,
    colour_primaries: PRIMARIES_UNSPECIFIED,
    full_range: false,
    chroma_location: None,
};

pub(super) const WIDTH: u32 = 11;
pub(super) const HEIGHT: u32 = 7;

/// A plane of `rows x columns` values, each unique: `row * 100 + column`.
pub(super) fn values(rows: usize, columns: usize) -> Vec<u16> {
    (0..rows)
        .flat_map(|row| (0..columns).map(move |column| (row * 100 + column) as u16))
        .collect()
}

pub(super) fn bytes_of(values: &[u16], sample: usize) -> Vec<u8> {
    match sample {
        1 => values.iter().map(|value| *value as u8).collect(),
        _ => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    }
}

/// Bytes of `values` (`rows x columns`) cut to a window, by hand.
pub(super) fn slice(values: &[u16], columns: usize, window: Window, sample: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for row in window.row..window.row + window.rows {
        let start = row * columns + window.column;
        out.extend(bytes_of(&values[start..start + window.columns], sample));
    }
    out
}

/// Packs each row of `bytes` (`rows` rows) into words, padding rows to 4 bytes.
pub(super) fn words_of(bytes: &[u8], rows: usize) -> (Vec<u32>, usize) {
    let row_bytes = bytes.len() / rows;
    let words = row_bytes.div_ceil(4);
    let mut out = Vec::new();
    for row in bytes.chunks(row_bytes) {
        let mut padded = row.to_vec();
        padded.resize(words * 4, 0xEE);
        out.extend(
            padded
                .chunks(4)
                .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]])),
        );
    }
    (out, words)
}

/// The first `row_bytes` bytes of each row of a packed plane.
pub(super) fn unpack(plane: &Tensor<CpuRuntime>, row_bytes: usize) -> Vec<u8> {
    let words = plane.shape()[1];
    let data: Vec<u32> = plane.to_vec();
    data.chunks(words)
        .flat_map(|row| {
            let bytes: Vec<u8> = row.iter().flat_map(|word| word.to_le_bytes()).collect();
            bytes[..row_bytes].to_vec()
        })
        .collect()
}

pub(super) fn plane_bytes(plane: &Tensor<CpuRuntime>, format: PixelFormat) -> Vec<u8> {
    match format {
        PixelFormat::Nv12 => plane.to_vec::<u8>(),
        _ => plane
            .to_vec::<u16>()
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    }
}

pub(super) struct Source {
    pub(super) luma: Vec<u16>,
    pub(super) chroma: Vec<u16>,
    pub(super) chroma_shape: [usize; 2],
}

pub(super) fn source(format: PixelFormat) -> Source {
    let shape = crate::decode::frame::chroma_shape(WIDTH, HEIGHT);
    let mask = if format == PixelFormat::Nv12 {
        0xFF
    } else {
        0xFFFF
    };
    let masked = |plane: Vec<u16>| plane.into_iter().map(|value| value & mask).collect();
    Source {
        luma: masked(values(HEIGHT as usize, WIDTH as usize)),
        chroma: masked(values(shape[0], shape[1])),
        chroma_shape: shape,
    }
}

pub(super) fn frame(format: PixelFormat, form: PlaneForm) -> (DecodedFrame<CpuRuntime>, Source) {
    let device = CpuDevice::new();
    let source = source(format);
    let display = Dimensions::new(WIDTH, HEIGHT).unwrap();
    let (width, height) = (WIDTH as usize, HEIGHT as usize);
    let [chroma_rows, chroma_columns] = source.chroma_shape;
    let sample = format.bytes_per_sample();
    let built = match form {
        PlaneForm::Samples => {
            let (y, uv) = match format {
                PixelFormat::Nv12 => {
                    let y: Vec<u8> = bytes_of(&source.luma, 1);
                    let uv: Vec<u8> = bytes_of(&source.chroma, 1);
                    (
                        Tensor::from_slice(&y, &[height, width], &device).unwrap(),
                        Tensor::from_slice(&uv, &[chroma_rows, chroma_columns], &device).unwrap(),
                    )
                }
                _ => (
                    Tensor::from_slice(&source.luma, &[height, width], &device).unwrap(),
                    Tensor::from_slice(&source.chroma, &[chroma_rows, chroma_columns], &device)
                        .unwrap(),
                ),
            };
            DecodedFrame::from_planes(y, uv, format, 7, display, UNSTATED)
        }
        PlaneForm::Words => {
            let (y, y_words) = words_of(&bytes_of(&source.luma, sample), height);
            let (uv, uv_words) = words_of(&bytes_of(&source.chroma, sample), chroma_rows);
            DecodedFrame::from_packed_planes(
                Tensor::from_slice(&y, &[height, y_words], &device).unwrap(),
                Tensor::from_slice(&uv, &[chroma_rows, uv_words], &device).unwrap(),
                format,
                7,
                display,
                UNSTATED,
            )
        }
    };
    (built, source)
}

pub(super) const fn rect(top: u32, bottom: u32, left: u32, right: u32) -> CropRect {
    CropRect {
        top,
        bottom,
        left,
        right,
    }
}
