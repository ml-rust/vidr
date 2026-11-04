//! `to_rgb` against values computed from numr's documented formulas.

use numr::dtype::DType;
use numr::ops::{ChromaSiting, ColorMatrix, ColorRange, YuvSpec};
use numr::runtime::cpu::CpuRuntime;
use vidr::decode::PixelFormat;
use vidr::frame::to_rgb;

use super::support::{DTYPES, FORMATS, LAYOUTS, Samples, cpu, rgb_shape, rgb_values, specs};

/// Allowed F32 difference, about 0.003 of an 8-bit code. It covers the f32
/// rounding of numr's steps against this file's f64 arithmetic.
const F32_TOLERANCE: f64 = 1e-5;
/// Allowed F16 difference: half an F16 step below 1, plus [`F32_TOLERANCE`].
const F16_TOLERANCE: f64 = 1.0 / 4096.0 + F32_TOLERANCE;
/// Allowed distance of a coded primary from its ideal value. The 8-bit code
/// table rounds each primary, which moves a channel by up to this much.
const PRIMARY_TOLERANCE: f32 = 0.01;

/// RGB of one pixel from sample values, by the formulas of `ColorOps`.
fn decode_pixel(y: f64, u: f64, v: f64, format: PixelFormat, spec: &YuvSpec) -> [f64; 3] {
    let bits = u32::from(format.bit_depth());
    let scale = f64::from(1_u32 << (bits - 8));
    let peak = f64::from((1_u32 << bits) - 1);
    let (luma, cb, cr) = match spec.range {
        ColorRange::Limited => (
            (y - 16.0 * scale) / (219.0 * scale),
            (u - 128.0 * scale) / (224.0 * scale),
            (v - 128.0 * scale) / (224.0 * scale),
        ),
        _ => {
            let middle = f64::from(1_u32 << (bits - 1));
            (y / peak, (u - middle) / peak, (v - middle) / peak)
        }
    };
    let (kr, kb) = spec.matrix.kr_kb();
    let kg = 1.0 - kr - kb;
    let red = luma + 2.0 * (1.0 - kr) * cr;
    let blue = luma + 2.0 * (1.0 - kb) * cb;
    let green = (luma - kr * red - kb * blue) / kg;
    [red, green, blue].map(|value| value.clamp(0.0, 1.0))
}

/// `U` at a luma column under left siting: even columns take their chroma
/// column, and odd columns average two.
fn left_sited(column: usize, chroma: &[f64]) -> f64 {
    let k = column / 2;
    if column.is_multiple_of(2) {
        chroma[k]
    } else {
        (chroma[k] + chroma[(k + 1).min(chroma.len() - 1)]) / 2.0
    }
}

/// `V` at a luma row: weights 1/4 and 3/4 on the two nearest chroma rows.
fn midway(row: usize, chroma: &[f64]) -> f64 {
    let k = row / 2;
    if row.is_multiple_of(2) {
        (chroma[k.saturating_sub(1)] + 3.0 * chroma[k]) / 4.0
    } else {
        (3.0 * chroma[k] + chroma[(k + 1).min(chroma.len() - 1)]) / 4.0
    }
}

fn tolerance(dtype: DType) -> f64 {
    if dtype == DType::F16 {
        F16_TOLERANCE
    } else {
        F32_TOLERANCE
    }
}

/// A luma ramp, `U` rising across chroma columns, and `V` rising down chroma
/// rows, so every pixel has a known value and each filter direction shows.
fn check_ramp(size: (usize, usize), format: PixelFormat) {
    let (width, height) = size;
    let (client, device) = cpu();
    let scale = f64::from(1_u32 << (format.bit_depth() - 8));
    let luma = |row: usize, column: usize| (30 + 10 * column + row) as u16;
    let u_code = |column: usize| (96 + 4 * column) as u16;
    let v_code = |row: usize| (80 + 8 * row) as u16;
    let samples = Samples::from_fn(size, format, luma, |row, column| {
        (u_code(column), v_code(row))
    });
    let u_plane: Vec<f64> = (0..width.div_ceil(2))
        .map(|column| f64::from(u_code(column)) * scale)
        .collect();
    let v_plane: Vec<f64> = (0..height.div_ceil(2))
        .map(|row| f64::from(v_code(row)) * scale)
        .collect();
    let frame = samples.frame::<CpuRuntime>(&device);
    for spec in specs() {
        for dtype in DTYPES {
            for layout in LAYOUTS {
                let rgb = to_rgb(&client, &frame, &spec, dtype, layout)
                    .unwrap_or_else(|error| panic!("{format} {spec:?}: {error}"));
                assert_eq!(rgb.shape(), rgb_shape(width, height, layout));
                let values = rgb_values(&rgb, layout);
                let limit = tolerance(dtype);
                for (index, pixel) in values.as_chunks::<3>().0.iter().enumerate() {
                    let (row, column) = (index / width, index % width);
                    let expected = decode_pixel(
                        f64::from(luma(row, column)) * scale,
                        left_sited(column, &u_plane),
                        midway(row, &v_plane),
                        format,
                        &spec,
                    );
                    for (got, want) in pixel.iter().zip(expected) {
                        assert!(
                            (f64::from(*got) - want).abs() <= limit,
                            "{format} {spec:?} {dtype:?} {layout:?} {width}x{height} at row \
                             {row}, column {column}: got {got}, want {want}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn even_ramp_matches_the_formulas() {
    for format in FORMATS {
        check_ramp((16, 12), format);
    }
}

#[test]
fn odd_ramp_matches_the_formulas() {
    for format in FORMATS {
        check_ramp((15, 11), format);
        check_ramp((15, 12), format);
        check_ramp((16, 11), format);
    }
}

#[test]
fn one_pixel_frame_matches_the_formulas() {
    for format in FORMATS {
        check_ramp((1, 1), format);
    }
}

fn flat(
    format: PixelFormat,
    spec: &YuvSpec,
    (y, u, v): (u16, u16, u16),
    expected: [f32; 3],
    limit: f32,
) {
    let (client, device) = cpu();
    let samples = Samples::from_fn((8, 6), format, |_, _| y, |_, _| (u, v));
    let frame = samples.frame::<CpuRuntime>(&device);
    for layout in LAYOUTS {
        let rgb = to_rgb(&client, &frame, spec, DType::F32, layout)
            .unwrap_or_else(|error| panic!("{format} {spec:?}: {error}"));
        for pixel in rgb_values(&rgb, layout).as_chunks::<3>().0 {
            for (got, want) in pixel.iter().zip(expected) {
                assert!(
                    (got - want).abs() <= limit,
                    "{format} {spec:?} codes ({y}, {u}, {v}): got {pixel:?}, want {expected:?}"
                );
            }
        }
    }
}

#[test]
fn limited_range_white_and_black_are_exact() {
    for format in FORMATS {
        for matrix in [ColorMatrix::Bt601, ColorMatrix::Bt709] {
            let spec = YuvSpec::new(matrix, ColorRange::Limited, ChromaSiting::Left);
            flat(format, &spec, (235, 128, 128), [1.0; 3], 0.0);
            flat(format, &spec, (16, 128, 128), [0.0; 3], 0.0);
        }
    }
}

#[test]
fn limited_range_red_uses_the_named_matrix() {
    for format in FORMATS {
        let bt601 = YuvSpec::new(ColorMatrix::Bt601, ColorRange::Limited, ChromaSiting::Left);
        let bt709 = YuvSpec::new(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left);
        let red = [1.0, 0.0, 0.0];
        flat(format, &bt601, (81, 90, 240), red, PRIMARY_TOLERANCE);
        flat(format, &bt709, (63, 102, 240), red, PRIMARY_TOLERANCE);
    }
}
