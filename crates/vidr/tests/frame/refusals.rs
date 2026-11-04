//! Named refusals of `to_rgb` and `from_rgb`.

use numr::dtype::DType;
use numr::ops::{ChromaSiting, ColorMatrix, ColorRange, ConvLayout, YuvSpec};
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;
use vidr::decode::PixelFormat;
use vidr::frame::{from_rgb, to_rgb};
use vidr::{Error, FrameError};

use super::support::{FORMATS, LAYOUTS, ODD_SIZES, Samples, cpu, planes_frame, rgb_shape};

fn spec() -> YuvSpec {
    YuvSpec::new(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left)
}

fn blank(shape: &[usize], dtype: DType) -> Tensor<CpuRuntime> {
    let (_, device) = cpu();
    Tensor::<CpuRuntime>::empty(shape, dtype, &device)
        .unwrap_or_else(|error| panic!("allocating {shape:?}: {error}"))
}

#[test]
fn from_rgb_refuses_odd_sizes_by_name() {
    let (client, _) = cpu();
    for (width, height) in ODD_SIZES {
        for layout in LAYOUTS {
            let rgb = blank(&rgb_shape(width, height, layout), DType::F32);
            for format in FORMATS {
                let error = from_rgb(&client, &rgb, &spec(), format, layout)
                    .expect_err("an odd size converts");
                assert_eq!(error, FrameError::OddRgbSize { width, height });
                let message = error.to_string();
                assert!(
                    message.contains(&format!("{width}x{height}")) && message.contains("even"),
                    "{message}"
                );
            }
        }
    }
}

#[test]
fn from_rgb_refuses_anything_but_one_frame() {
    let (client, _) = cpu();
    let cases: [(&[usize], ConvLayout); 5] = [
        (&[3, 180, 320], ConvLayout::Nchw),
        (&[2, 3, 180, 320], ConvLayout::Nchw),
        (&[1, 4, 180, 320], ConvLayout::Nchw),
        (&[1, 3, 180, 320], ConvLayout::Nhwc),
        (&[1, 3, 0, 320], ConvLayout::Nchw),
    ];
    for (shape, layout) in cases {
        let rgb = blank(shape, DType::F32);
        let error = from_rgb(&client, &rgb, &spec(), PixelFormat::Nv12, layout)
            .expect_err("a bad shape converts");
        assert_eq!(
            error,
            FrameError::RgbShape {
                shape: shape.to_vec(),
                layout
            }
        );
    }
}

#[test]
fn to_rgb_refuses_planes_that_do_not_match_the_frame() {
    let (client, _) = cpu();
    let (width, height) = (321, 181);
    // Each case swaps one plane. The last two floor the chroma shape.
    // Luma shape, luma dtype, chroma shape, chroma dtype, expected plane.
    type Case<'a> = (&'a [usize], DType, &'a [usize], DType, &'a str);
    let cases: [Case<'_>; 4] = [
        (&[181, 320], DType::U8, &[91, 322], DType::U8, "luma"),
        (&[181, 321], DType::U16, &[91, 322], DType::U8, "luma"),
        (&[181, 321], DType::U8, &[90, 322], DType::U8, "chroma"),
        (&[181, 321], DType::U8, &[91, 320], DType::U8, "chroma"),
    ];
    for (y_shape, y_dtype, uv_shape, uv_dtype, named) in cases {
        let frame = planes_frame(
            blank(y_shape, y_dtype),
            blank(uv_shape, uv_dtype),
            PixelFormat::Nv12,
            width,
            height,
        );
        let error = to_rgb(&client, &frame, &spec(), DType::F32, ConvLayout::Nchw)
            .expect_err("mismatched planes convert");
        match &error {
            FrameError::PlaneMismatch { plane, .. } => assert_eq!(*plane, named),
            other => panic!("unexpected error: {other}"),
        }
    }
}

#[test]
fn to_rgb_passes_numr_refusals_with_the_input() {
    let (client, device) = cpu();
    let frame = Samples::smooth((320, 180), PixelFormat::Nv12).frame::<CpuRuntime>(&device);
    let error = to_rgb(&client, &frame, &spec(), DType::U8, ConvLayout::Nchw)
        .expect_err("an integer RGB dtype converts");
    match &error {
        FrameError::Numr {
            operation, input, ..
        } => {
            assert_eq!(*operation, "ColorOps::yuv_to_rgb");
            assert!(input.contains("U8"), "{input}");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn frame_errors_join_the_crate_error() {
    let error = FrameError::OddRgbSize {
        width: 321,
        height: 181,
    };
    assert_eq!(Error::from(error.clone()), Error::Frame(error));
}
