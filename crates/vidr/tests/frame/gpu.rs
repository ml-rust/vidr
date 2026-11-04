//! The CUDA path against the CPU path, on synthetic and NVDEC-decoded frames.
//!
//! Every test skips with a notice when the NVIDIA library, a CUDA device, or
//! its fixture is missing.

use numr::dtype::DType;
use numr::ops::ChromaSiting;
use numr::runtime::RuntimeClient;
use numr::runtime::cpu::CpuRuntime;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime, is_cuda_available};
use vidr::decode::DecodedFrame;
use vidr::decode::nvdec::NvdecDecoder;
use vidr::frame::to_rgb;

use super::support::{
    DTYPES, FORMATS, LAYOUTS, ODD_SIZES, Samples, cpu, rgb_shape, rgb_values, specs,
};
use crate::nvdec::fixtures::{self, Stream};

/// Allowed F32 difference between backends: numr runs the same rounded steps
/// on each, and four such steps stay within `4 * 2^-23` of the reference.
const F32_TOLERANCE: f32 = 4.0 * f32::EPSILON;
/// Allowed F16 difference between backends: one F16 step below 1.
const F16_TOLERANCE: f32 = 1.0 / 2048.0;

fn tolerance(dtype: DType) -> f32 {
    if dtype == DType::F16 {
        F16_TOLERANCE
    } else {
        F32_TOLERANCE
    }
}

fn assert_close(got: &[f32], want: &[f32], limit: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: lengths");
    for (index, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            (a - b).abs() <= limit,
            "{label}: value {index}: CUDA {a}, CPU {b}"
        );
    }
}

/// numr's client on CUDA device 0, or a notice and `None`. This path needs no
/// NVIDIA video library.
fn cuda_only() -> Option<CudaClient> {
    if !is_cuda_available() {
        eprintln!("skipping: no CUDA device 0. Run the frame GPU tests on a machine with a GPU.");
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

/// Odd NV12 and P010 planes run the U8 and U16 pad on CUDA, in both layouts and
/// both dtypes, and match the CPU path.
#[test]
fn synthetic_odd_frames_match_the_cpu_path() {
    let Some(client) = cuda_only() else {
        return;
    };
    let (cpu_client, cpu_device) = cpu();
    for size in ODD_SIZES {
        let (width, height) = size;
        for format in FORMATS {
            let samples = Samples::smooth(size, format);
            let gpu_frame = samples.frame::<CudaRuntime>(client.device());
            let cpu_frame = samples.frame::<CpuRuntime>(&cpu_device);
            for spec in specs() {
                for dtype in DTYPES {
                    for layout in LAYOUTS {
                        let label =
                            format!("{width}x{height} {format} {spec:?} {dtype:?} {layout:?}");
                        let rgb = to_rgb(&client, &gpu_frame, &spec, dtype, layout)
                            .unwrap_or_else(|error| panic!("{label}: {error}"));
                        assert_eq!(rgb.shape(), rgb_shape(width, height, layout), "{label}");
                        assert!(rgb.is_contiguous(), "{label}: not dense");
                        let reference = to_rgb(&cpu_client, &cpu_frame, &spec, dtype, layout)
                            .unwrap_or_else(|error| panic!("{label}: CPU: {error}"));
                        assert_close(
                            &rgb_values(&rgb, layout),
                            &rgb_values(&reference, layout),
                            tolerance(dtype),
                            &label,
                        );
                    }
                }
            }
        }
    }
}

fn decode_stream(client: &CudaClient, stream: &Stream) -> Vec<DecodedFrame<CudaRuntime>> {
    let name = stream.name;
    let mut decoder = NvdecDecoder::new(client, stream.codec(), &stream.config)
        .unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let mut frames = Vec::new();
    for (sample, pts) in stream.samples.iter().zip(&stream.pts) {
        decoder
            .send(sample, *pts)
            .unwrap_or_else(|error| panic!("{name}: send at pts {pts}: {error}"));
        while let Some(frame) = decoder.receive() {
            frames.push(frame);
        }
    }
    decoder
        .flush()
        .unwrap_or_else(|error| panic!("{name}: flush: {error}"));
    while let Some(frame) = decoder.receive() {
        frames.push(frame);
    }
    frames
}

/// NVDEC frames of an odd stream, converted on CUDA, match ffmpeg's decode of
/// the same stream converted on the CPU.
fn check_decoded_odd(name: &'static str) {
    let Some(stream) = fixtures::load(name) else {
        return;
    };
    let Some(client) = fixtures::gpu() else {
        return;
    };
    let Some(yuv) = &stream.yuv else {
        panic!("{name}: the fixture has no .yuv");
    };
    let (width, height) = (
        stream.number("width") as usize,
        stream.number("height") as usize,
    );
    assert!(
        !width.is_multiple_of(2) && !height.is_multiple_of(2),
        "{name}: {width}x{height} is not odd"
    );
    let frames = decode_stream(&client, &stream);
    let frame_bytes = width * height + height.div_ceil(2) * width.div_ceil(2) * 2;
    assert_eq!(yuv.len(), frame_bytes * frames.len(), "{name}: frame count");
    let (cpu_client, cpu_device) = cpu();
    for (index, (frame, raw)) in frames.iter().zip(yuv.chunks_exact(frame_bytes)).enumerate() {
        let Some(spec) = frame.color().yuv_spec(ChromaSiting::Left) else {
            panic!(
                "{name} frame {index}: matrix {}",
                frame.color().matrix_coefficients
            );
        };
        let cpu_frame = Samples::from_nv12(raw, width, height).frame::<CpuRuntime>(&cpu_device);
        for dtype in DTYPES {
            for layout in LAYOUTS {
                let label = format!("{name} frame {index} {dtype:?} {layout:?}");
                let rgb = to_rgb(&client, frame, &spec, dtype, layout)
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
                assert_eq!(rgb.shape(), rgb_shape(width, height, layout), "{label}");
                assert!(rgb.is_contiguous(), "{label}: not dense");
                let reference = to_rgb(&cpu_client, &cpu_frame, &spec, dtype, layout)
                    .unwrap_or_else(|error| panic!("{label}: CPU: {error}"));
                assert_close(
                    &rgb_values(&rgb, layout),
                    &rgb_values(&reference, layout),
                    tolerance(dtype),
                    &label,
                );
            }
        }
    }
}

#[test]
fn decoded_odd_av1_matches_the_cpu_path() {
    check_decoded_odd("av1_odd");
}

#[test]
fn decoded_odd_vp9_matches_the_cpu_path() {
    check_decoded_odd("vp9_odd");
}
