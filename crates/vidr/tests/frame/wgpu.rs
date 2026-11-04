//! The WebGPU path against the CPU path, on packed word planes.
//!
//! numr's colour backend parity compares WebGPU and CPU bytes for equality, so
//! these tests allow no difference. Every test skips with a printed notice when
//! the machine has no WebGPU adapter.

use std::panic::catch_unwind;
use std::sync::{Mutex, PoisonError};

use numr::dtype::DType;
use numr::ops::ConvLayout;
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime, is_wgpu_available};
use numr::tensor::Tensor;
use vidr::FrameError;
use vidr::decode::PlaneForm;
use vidr::frame::{from_rgb, to_rgb, to_rgb_reusing};

use super::host::{
    check_buffer_reuse, check_pitched_download, check_round_trip, check_size_change,
};
use super::support::{FORMATS, LAYOUTS, Samples, cpu, pattern, rgb_tensor, specs, word_rows};

/// Even sizes, one with a partial last word in NV12.
const EVEN_SIZES: [(usize, usize); 3] = [(32, 18), (34, 20), (6, 4)];
/// Each odd shape.
const ODD_SIZES: [(usize, usize); 4] = [(33, 19), (33, 18), (32, 19), (3, 3)];

/// Concurrent use loses a WebGPU device, so these tests run one at a time.
static BACKEND: Mutex<()> = Mutex::new(());

/// Runs `check` on WebGPU device 0, or prints a notice when there is none.
fn with_wgpu(check: impl FnOnce(&WgpuClient, &WgpuDevice)) {
    if !is_wgpu_available() {
        eprintln!("skipping: no WebGPU adapter. Run the frame WebGPU tests on a machine with one.");
        return;
    }
    let _guard = BACKEND.lock().unwrap_or_else(PoisonError::into_inner);
    // numr's `default_client` panics when the device does not open.
    let opened = catch_unwind(|| {
        let device = WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);
        (client, device)
    });
    let Ok((client, device)) = opened else {
        eprintln!("skipping: numr cannot open WebGPU device 0.");
        return;
    };
    check(&client, &device);
}

fn bytes<R: Runtime>(tensor: &Tensor<R>, label: &str) -> Vec<u8> {
    tensor
        .to_bytes()
        .unwrap_or_else(|error| panic!("{label}: read: {error}"))
}

/// `to_rgb` on word planes equals `to_rgb` on CPU samples, bit for bit, at even
/// and odd sizes in NV12 and P010. An odd size runs numr's `pad_packed_luma`.
#[test]
fn to_rgb_matches_the_cpu_path() {
    with_wgpu(|client, device| {
        let (cpu_client, cpu_device) = cpu();
        for size in EVEN_SIZES.into_iter().chain(ODD_SIZES) {
            for format in FORMATS {
                let samples = Samples::smooth(size, format);
                let frame = samples.words_frame::<WgpuRuntime>(device);
                assert_eq!(frame.plane_form(), PlaneForm::Words);
                let reference = samples.frame::<CpuRuntime>(&cpu_device);
                for spec in specs() {
                    for layout in LAYOUTS {
                        let label = format!("{size:?} {format} {spec:?} {layout:?}");
                        let got = to_rgb(client, &frame, &spec, DType::F32, layout)
                            .unwrap_or_else(|error| panic!("{label}: {error}"));
                        let want = to_rgb(&cpu_client, &reference, &spec, DType::F32, layout)
                            .unwrap_or_else(|error| panic!("{label}: CPU: {error}"));
                        assert_eq!(got.shape(), want.shape(), "{label}: shape");
                        assert!(got.is_contiguous(), "{label}: not dense");
                        assert!(
                            bytes(&got, &label) == bytes(&want, &label),
                            "{label}: RGB differs from the CPU path"
                        );
                    }
                }
            }
        }
    });
}

/// `to_rgb_reusing` on word planes writes the CPU path's bits into its buffer,
/// and a second frame keeps that buffer.
#[test]
fn to_rgb_reusing_matches_the_cpu_path() {
    with_wgpu(|client, device| {
        let (cpu_client, cpu_device) = cpu();
        let spec = specs()[2];
        for format in FORMATS {
            let samples = Samples::smooth((32, 18), format);
            let frame = samples.words_frame::<WgpuRuntime>(device);
            let reference = samples.frame::<CpuRuntime>(&cpu_device);
            let label = format!("{format}");
            let want = to_rgb(&cpu_client, &reference, &spec, DType::F32, ConvLayout::Nchw)
                .unwrap_or_else(|error| panic!("{label}: CPU: {error}"));
            let mut buffer = None;
            let got = to_rgb_reusing(
                client,
                &frame,
                &spec,
                DType::F32,
                ConvLayout::Nchw,
                &mut buffer,
            )
            .unwrap_or_else(|error| panic!("{label}: {error}"));
            assert!(bytes(&got, &label) == bytes(&want, &label), "{label}: RGB");
            let again = to_rgb_reusing(
                client,
                &frame,
                &spec,
                DType::F32,
                ConvLayout::Nchw,
                &mut buffer,
            )
            .unwrap_or_else(|error| panic!("{label}: {error}"));
            assert_eq!(got.ptr(), again.ptr(), "{label}: the buffer moved");
        }
    });
}

/// `from_rgb` allocates U32 `[rows, row_words]` on WebGPU, and the bytes of each
/// row equal the CPU path's samples.
#[test]
fn from_rgb_matches_the_cpu_path() {
    with_wgpu(|client, device| {
        let (cpu_client, cpu_device) = cpu();
        for (width, height) in EVEN_SIZES {
            let values = pattern(width, height, 3);
            for format in FORMATS {
                let sample = format.bytes_per_sample();
                let row_bytes = width * sample;
                let row_words = row_bytes.div_ceil(4);
                for spec in specs() {
                    for layout in LAYOUTS {
                        let label = format!("{width}x{height} {format} {spec:?} {layout:?}");
                        let rgb = rgb_tensor::<WgpuRuntime>(&values, width, height, layout, device);
                        let planes = from_rgb(client, &rgb, &spec, format, layout)
                            .unwrap_or_else(|error| panic!("{label}: {error}"));
                        assert_eq!(planes.plane_form(), PlaneForm::Words, "{label}");
                        assert_eq!(planes.y.dtype(), DType::U32, "{label}: luma dtype");
                        assert_eq!(planes.y.shape(), [height, row_words], "{label}: luma");
                        assert_eq!(planes.uv.dtype(), DType::U32, "{label}: chroma dtype");
                        assert_eq!(
                            planes.uv.shape(),
                            [height / 2, row_words],
                            "{label}: chroma"
                        );

                        let cpu_rgb =
                            rgb_tensor::<CpuRuntime>(&values, width, height, layout, &cpu_device);
                        let reference = from_rgb(&cpu_client, &cpu_rgb, &spec, format, layout)
                            .unwrap_or_else(|error| panic!("{label}: CPU: {error}"));
                        assert!(
                            word_rows(&planes.y, row_bytes) == bytes(&reference.y, &label),
                            "{label}: luma differs from the CPU path"
                        );
                        assert!(
                            word_rows(&planes.uv, row_bytes) == bytes(&reference.uv, &label),
                            "{label}: chroma differs from the CPU path"
                        );
                    }
                }
            }
        }
    });
}

/// `from_rgb` refuses an odd size on WebGPU by name, as it does on CPU.
#[test]
fn from_rgb_refuses_an_odd_size() {
    with_wgpu(|client, device| {
        let values = pattern(33, 18, 0);
        let rgb = rgb_tensor::<WgpuRuntime>(&values, 33, 18, ConvLayout::Nhwc, device);
        let spec = specs()[0];
        let error = from_rgb(client, &rgb, &spec, FORMATS[0], ConvLayout::Nhwc)
            .expect_err("an odd size converts");
        assert_eq!(
            error,
            FrameError::OddRgbSize {
                width: 33,
                height: 18
            }
        );
    });
}

#[test]
fn host_planes_round_trip_byte_exact() {
    with_wgpu(|_, device| check_round_trip::<WgpuRuntime>(device));
}

#[test]
fn pitched_and_offset_word_views_download_byte_exact() {
    with_wgpu(|_, device| check_pitched_download::<WgpuRuntime>(device));
}

#[test]
fn a_second_frame_reuses_the_host_buffers() {
    with_wgpu(|_, device| check_buffer_reuse::<WgpuRuntime>(device));
}

#[test]
fn a_size_change_downloads() {
    with_wgpu(|_, device| check_size_change::<WgpuRuntime>(device));
}
