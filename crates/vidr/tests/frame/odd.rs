//! Odd display sizes against the replicated even frame.

use numr::dtype::DType;
use numr::ops::{ChromaSiting, ColorMatrix, ColorOps, ColorRange, YuvSpec};
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use vidr::frame::to_rgb;

use super::support::{
    DTYPES, FORMATS, LAYOUTS, ODD_SIZES, Samples, cpu, crop_rows, planes_frame, rgb_shape,
    rgb_values, specs,
};

/// Every pixel is bit-identical to the same pixel of the even conversion:
/// both runs read the same samples through the same kernel.
#[test]
fn odd_frames_equal_the_replicated_even_frame_bit_for_bit() {
    let (client, device) = cpu();
    for size in ODD_SIZES {
        let (width, height) = size;
        for format in FORMATS {
            let samples = Samples::smooth(size, format);
            let even = samples.replicated();
            let odd_frame = samples.frame::<CpuRuntime>(&device);
            let even_frame = even.frame::<CpuRuntime>(&device);
            for spec in specs() {
                for dtype in DTYPES {
                    for layout in LAYOUTS {
                        let label =
                            format!("{width}x{height} {format} {spec:?} {dtype:?} {layout:?}");
                        let rgb = to_rgb(&client, &odd_frame, &spec, dtype, layout)
                            .unwrap_or_else(|error| panic!("{label}: {error}"));
                        assert_eq!(
                            rgb.shape(),
                            rgb_shape(width, height, layout),
                            "{label}: shape"
                        );
                        assert!(rgb.is_contiguous(), "{label}: not dense");
                        assert_eq!(rgb.dtype(), dtype, "{label}: dtype");
                        let reference = to_rgb(&client, &even_frame, &spec, dtype, layout)
                            .unwrap_or_else(|error| panic!("{label}: even: {error}"));
                        let expected =
                            crop_rows(&rgb_values(&reference, layout), even.width, width, height);
                        let got = rgb_values(&rgb, layout);
                        assert!(
                            got.iter()
                                .zip(&expected)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "{label}: a pixel differs from the replicated even frame"
                        );
                    }
                }
            }
        }
    }
}

/// An even frame goes to numr unchanged.
#[test]
fn even_frames_match_numr_directly() {
    let (client, device) = cpu();
    let spec = YuvSpec::new(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left);
    for format in FORMATS {
        let frame = Samples::smooth((320, 180), format).frame::<CpuRuntime>(&device);
        for layout in LAYOUTS {
            let rgb = to_rgb(&client, &frame, &spec, DType::F32, layout)
                .unwrap_or_else(|error| panic!("{format}: {error}"));
            let direct = client
                .yuv_to_rgb(frame.planes(), &spec, DType::F32, layout)
                .unwrap_or_else(|error| panic!("{format}: numr: {error}"));
            let (got, want) = (rgb_values(&rgb, layout), rgb_values(&direct, layout));
            assert!(
                got.iter()
                    .zip(&want)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{format} {layout:?}: `to_rgb` differs from `yuv_to_rgb`"
            );
        }
    }
}

/// A view of `values` with a row pitch: `extra` samples past each row.
fn pitched(
    samples: &Samples,
    values: &[u16],
    [rows, columns]: [usize; 2],
    extra: usize,
    device: &CpuDevice,
) -> Tensor<CpuRuntime> {
    let mut wide = Vec::with_capacity(rows * (columns + extra));
    for row in values.chunks_exact(columns) {
        wide.extend_from_slice(row);
        wide.extend(std::iter::repeat_n(200_u16, extra));
    }
    let view = samples
        .plane::<CpuRuntime>(&wide, [rows, columns + extra], device)
        .narrow(1, 0, columns)
        .unwrap_or_else(|error| panic!("pitched plane: {error}"));
    assert!(!view.is_contiguous(), "the view has no row pitch");
    view
}

/// A plane with a row pitch converts like a dense plane.
#[test]
fn pitched_planes_match_dense_planes() {
    let (client, device) = cpu();
    let spec = YuvSpec::new(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left);
    for size in ODD_SIZES {
        let (width, height) = size;
        for format in FORMATS {
            let samples = Samples::smooth(size, format);
            let dense = samples.frame::<CpuRuntime>(&device);
            let [chroma_rows, chroma_columns] = samples.chroma_shape();
            let frame = planes_frame(
                pitched(&samples, &samples.y, [height, width], 13, &device),
                pitched(
                    &samples,
                    &samples.uv,
                    [chroma_rows, chroma_columns],
                    7,
                    &device,
                ),
                format,
                width,
                height,
            );
            for layout in LAYOUTS {
                let label = format!("{width}x{height} {format} {layout:?}");
                let got = to_rgb(&client, &frame, &spec, DType::F32, layout)
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
                let want = to_rgb(&client, &dense, &spec, DType::F32, layout)
                    .unwrap_or_else(|error| panic!("{label}: dense: {error}"));
                assert!(
                    rgb_values(&got, layout) == rgb_values(&want, layout),
                    "{label}: pitched planes differ from dense planes"
                );
            }
        }
    }
}
