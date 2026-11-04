//! `to_rgb_reusing` and `reuse_tensor` keep one RGB buffer across frames, with
//! the bits `to_rgb` gives.

use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;
use vidr::frame::{reuse_tensor, to_rgb, to_rgb_reusing};

use super::support::{DTYPES, FORMATS, LAYOUTS, ODD_SIZES, Samples, cpu, rgb_values, specs};

const SIZE: (usize, usize) = (64, 36);

#[test]
fn an_even_frame_matches_to_rgb_bit_for_bit() {
    let (client, device) = cpu();
    let spec = specs()[2];
    for format in FORMATS {
        let frame = Samples::smooth(SIZE, format).frame::<CpuRuntime>(&device);
        for dtype in DTYPES {
            for layout in LAYOUTS {
                let label = format!("{format} {dtype:?} {layout:?}");
                let mut buffer = None;
                let got = to_rgb_reusing(&client, &frame, &spec, dtype, layout, &mut buffer)
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
                let want = to_rgb(&client, &frame, &spec, dtype, layout)
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
                assert_eq!(got.shape(), want.shape(), "{label}: shape");
                assert_eq!(got.dtype(), dtype, "{label}: dtype");
                let (got, want) = (rgb_values(&got, layout), rgb_values(&want, layout));
                assert!(
                    got.iter()
                        .zip(&want)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{label}: a pixel differs from to_rgb"
                );
            }
        }
    }
}

#[test]
fn frames_of_one_size_write_one_buffer() {
    let (client, device) = cpu();
    let (spec, layout) = (specs()[2], LAYOUTS[1]);
    let frame = Samples::smooth(SIZE, FORMATS[0]).frame::<CpuRuntime>(&device);
    let mut buffer = None;
    let first = to_rgb_reusing(&client, &frame, &spec, DType::F16, layout, &mut buffer)
        .unwrap_or_else(|error| panic!("first frame: {error}"));
    let second = to_rgb_reusing(&client, &frame, &spec, DType::F16, layout, &mut buffer)
        .unwrap_or_else(|error| panic!("second frame: {error}"));
    assert_eq!(first.ptr(), second.ptr());
}

#[test]
fn an_odd_frame_leaves_the_buffer_unchanged() {
    let (client, device) = cpu();
    let (spec, layout) = (specs()[2], LAYOUTS[1]);
    for size in ODD_SIZES {
        let frame = Samples::smooth(size, FORMATS[0]).frame::<CpuRuntime>(&device);
        let mut buffer = None;
        let rgb = to_rgb_reusing(&client, &frame, &spec, DType::F32, layout, &mut buffer)
            .unwrap_or_else(|error| panic!("{size:?}: {error}"));
        assert_eq!(rgb.shape(), [1, size.1, size.0, 3], "{size:?}");
        assert!(buffer.is_none(), "{size:?}: the buffer was written");
    }
}

#[test]
fn another_shape_or_dtype_replaces_the_tensor() {
    let (_, device) = cpu();
    let mut slot: Option<Tensor<CpuRuntime>> = None;
    let first = reuse_tensor(&mut slot, &[2, 4], DType::F32, &device)
        .unwrap_or_else(|error| panic!("allocation: {error}"));
    let same = reuse_tensor(&mut slot, &[2, 4], DType::F32, &device)
        .unwrap_or_else(|error| panic!("reuse: {error}"));
    assert_eq!(first.ptr(), same.ptr());
    let shape = reuse_tensor(&mut slot, &[4, 4], DType::F32, &device)
        .unwrap_or_else(|error| panic!("new shape: {error}"));
    assert_eq!(shape.shape(), [4, 4]);
    let dtype = reuse_tensor(&mut slot, &[4, 4], DType::F16, &device)
        .unwrap_or_else(|error| panic!("new dtype: {error}"));
    assert_eq!(dtype.dtype(), DType::F16);
}
