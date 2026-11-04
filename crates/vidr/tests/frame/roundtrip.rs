//! RGB to planes and back on the CPU, and the borrow into an `EncodeFrame`.

use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
use vidr::decode::PixelFormat;
use vidr::encode::EncodeFrame;
use vidr::frame::{OutputPlanes, from_rgb, to_rgb};

use super::support::{
    DTYPES, FORMATS, LAYOUTS, Samples, cpu, pattern, plane_values, planes_frame, psnr, rgb_tensor,
    rgb_values, specs,
};

const WIDTH: usize = 320;
const HEIGHT: usize = 180;
/// RGB PSNR floor of RGB to 8-bit planes and back, peak 1.
const RGB_FLOOR_8BIT_DB: f64 = 50.0;
/// RGB PSNR floor of RGB to 10-bit planes and back, peak 1.
const RGB_FLOOR_10BIT_DB: f64 = 60.0;
/// Plane PSNR floor of planes to RGB and back, at the format's peak. Chroma
/// passes through one upsample and one downsample.
const PLANE_FLOOR_DB: f64 = 60.0;

fn floor(format: PixelFormat) -> f64 {
    match format {
        PixelFormat::P010 => RGB_FLOOR_10BIT_DB,
        _ => RGB_FLOOR_8BIT_DB,
    }
}

/// The floors sit below measured values. A failure means the conversion is
/// wrong, and is not a target to tune.
#[test]
fn planes_and_rgb_round_trip_above_their_floors() {
    let (client, device) = cpu();
    let source = pattern(WIDTH, HEIGHT, 0);
    for format in FORMATS {
        let peak = f64::from((1_u32 << format.bit_depth()) - 1);
        for spec in specs() {
            for layout in LAYOUTS {
                let label = format!("{format} {spec:?} {layout:?}");
                let rgb = rgb_tensor::<CpuRuntime>(&source, WIDTH, HEIGHT, layout, &device);
                let planes = from_rgb(&client, &rgb, &spec, format, layout)
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
                assert_eq!(planes.y.shape(), &[HEIGHT, WIDTH], "{label}: luma shape");
                assert_eq!(planes.uv.shape(), &[HEIGHT / 2, WIDTH], "{label}: uv shape");
                assert_eq!(planes.y.dtype(), format.sample_dtype(), "{label}: dtype");
                assert_eq!(planes.uv.dtype(), format.sample_dtype(), "{label}: dtype");
                let frame =
                    planes_frame(planes.y.clone(), planes.uv.clone(), format, WIDTH, HEIGHT);
                for dtype in DTYPES {
                    let label = format!("{label} {dtype:?}");
                    let decoded = to_rgb(&client, &frame, &spec, dtype, layout)
                        .unwrap_or_else(|error| panic!("{label}: {error}"));
                    let values = rgb_values(&decoded, layout);
                    let db = psnr(&source, &values, 1.0);
                    assert!(db > floor(format), "{label}: RGB PSNR {db:.2} dB");

                    let again = from_rgb(&client, &decoded, &spec, format, layout)
                        .unwrap_or_else(|error| panic!("{label}: again: {error}"));
                    let db = psnr(
                        &plane_values(&planes.y, format),
                        &plane_values(&again.y, format),
                        peak,
                    );
                    assert!(db > PLANE_FLOOR_DB, "{label}: luma PSNR {db:.2} dB");
                    let db = psnr(
                        &plane_values(&planes.uv, format),
                        &plane_values(&again.uv, format),
                        peak,
                    );
                    assert!(db > PLANE_FLOOR_DB, "{label}: chroma PSNR {db:.2} dB");
                }
            }
        }
    }
}

#[test]
fn output_planes_become_an_encode_frame_without_a_copy() {
    let (client, device) = cpu();
    let spec = specs()[2];
    let layout = LAYOUTS[0];
    let rgb = rgb_tensor::<CpuRuntime>(&pattern(16, 12, 0), 16, 12, layout, &device);
    let planes: OutputPlanes<CpuRuntime> = from_rgb(&client, &rgb, &spec, FORMATS[0], layout)
        .unwrap_or_else(|error| panic!("from_rgb: {error}"));
    let frame = EncodeFrame::from(&planes);
    assert!(std::ptr::eq(frame.y(), &planes.y), "luma is not borrowed");
    assert!(
        std::ptr::eq(frame.uv(), &planes.uv),
        "chroma is not borrowed"
    );
}

#[test]
fn f64_rgb_passes_through_to_numr() {
    let (client, device) = cpu();
    let spec = specs()[2];
    let frame = Samples::smooth((16, 12), FORMATS[0]).frame::<CpuRuntime>(&device);
    let rgb = to_rgb(&client, &frame, &spec, DType::F64, LAYOUTS[0])
        .unwrap_or_else(|error| panic!("F64: {error}"));
    assert_eq!(rgb.dtype(), DType::F64);
}
