//! `to_rgb` against ffmpeg's swscale with `bilinear+accurate_rnd+full_chroma_int`.
//!
//! swscale gives two references for an odd frame:
//!
//! - The replicated even frame, cropped. Its chroma sits where numr places it,
//!   so only rounding differs.
//! - The odd frame itself. swscale scales chroma by `ceil(W / 2) / W`, not by
//!   1/2, so its chroma drifts up to one luma pixel across the frame. On a
//!   sharp chroma edge that moves a pixel by far more than rounding. Its floors
//!   are therefore lower, and they measure swscale's drift.
//!
//! Each test skips with a notice when ffmpeg or its fixture is missing.

use std::path::PathBuf;
use std::process::Command;

use numr::dtype::DType;
use numr::ops::{ChromaSiting, ColorMatrix, ColorRange, YuvSpec};
use numr::runtime::cpu::CpuRuntime;
use vidr::decode::PixelFormat;
use vidr::frame::to_rgb;

use super::support::{
    FORMATS, LAYOUTS, ODD_SIZES, Samples, cpu, crop_rows, max_difference, psnr, rgb_values, specs,
    to_codes,
};

/// Largest allowed code difference against the replicated even frame: one
/// rounding step of the 8-bit output.
const EVEN_MAX_CODES: u8 = 1;
/// PSNR floor against the replicated even frame. swscale keeps 10-bit full-range
/// samples at a slightly different scale, hence the second floor.
const EVEN_FLOOR_DB: f64 = 65.0;
/// PSNR floor against the replicated even frame for full-range P010.
const EVEN_FLOOR_P010_FULL_DB: f64 = 50.0;
/// Largest allowed code difference against the odd frame on smooth chroma.
const DRIFT_MAX_CODES: u8 = 5;
/// PSNR floor against the odd frame on smooth chroma.
const DRIFT_FLOOR_DB: f64 = 44.0;
/// PSNR floor against the odd frame on a fixture with sharp chroma edges.
const DRIFT_FIXTURE_FLOOR_DB: f64 = 25.0;

/// ffmpeg's rgb24 conversion of raw `samples`, or a notice and `None` when
/// ffmpeg does not run.
fn swscale(samples: &Samples, spec: &YuvSpec, name: &str) -> Option<Vec<u8>> {
    let pix_fmt = match samples.format {
        PixelFormat::Nv12 => "nv12",
        _ => "p010le",
    };
    let matrix = match spec.matrix {
        ColorMatrix::Bt709 => "bt709",
        _ => "smpte170m",
    };
    let range = match spec.range {
        ColorRange::Limited => "tv",
        _ => "pc",
    };
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("frame");
    std::fs::create_dir_all(&directory)
        .unwrap_or_else(|error| panic!("creating {}: {error}", directory.display()));
    let path = directory.join(format!("{name}.raw"));
    std::fs::write(&path, samples.raw())
        .unwrap_or_else(|error| panic!("writing {}: {error}", path.display()));
    let filter = format!(
        "scale=in_color_matrix={matrix}:in_range={range}:in_chroma_loc=left:\
         flags=bilinear+accurate_rnd+full_chroma_int,format=rgb24"
    );
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s"])
        .arg(format!("{}x{}", samples.width, samples.height))
        .arg("-i")
        .arg(&path)
        .args(["-vf", filter.as_str()])
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        .output();
    let output = match output {
        Ok(output) => output,
        Err(error) => {
            eprintln!("skipping the swscale check of {name}: ffmpeg does not run: {error}");
            return None;
        }
    };
    assert!(
        output.status.success(),
        "{name}: ffmpeg: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        output.stdout.len(),
        3 * samples.width * samples.height,
        "{name}: ffmpeg output size"
    );
    Some(output.stdout)
}

/// 8-bit codes of `to_rgb`, channels last, in each layout. Both layouts must
/// give the same codes.
fn our_codes(samples: &Samples, spec: &YuvSpec, label: &str) -> Vec<u8> {
    let (client, device) = cpu();
    let frame = samples.frame::<CpuRuntime>(&device);
    let mut codes = Vec::new();
    for layout in LAYOUTS {
        let rgb = to_rgb(&client, &frame, spec, DType::F32, layout)
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        let values = to_codes(&rgb_values(&rgb, layout));
        if codes.is_empty() {
            codes = values;
        } else {
            assert_eq!(codes, values, "{label}: layouts differ");
        }
    }
    codes
}

fn name(prefix: &str, samples: &Samples, spec: &YuvSpec) -> String {
    format!(
        "{prefix}_{}x{}_{}_{:?}_{:?}",
        samples.width, samples.height, samples.format, spec.matrix, spec.range
    )
}

fn check_against_even(samples: &Samples, spec: &YuvSpec, label: &str) -> bool {
    let even = samples.replicated();
    let Some(reference) = swscale(&even, spec, &name(label, samples, spec)) else {
        return false;
    };
    let reference = crop_rows(&reference, even.width, samples.width, samples.height);
    let ours = our_codes(samples, spec, label);
    let floor = match (samples.format, spec.range) {
        (PixelFormat::P010, ColorRange::Full) => EVEN_FLOOR_P010_FULL_DB,
        _ => EVEN_FLOOR_DB,
    };
    let db = psnr(&ours, &reference, 255.0);
    assert!(
        db > floor,
        "{label} {} {spec:?}: PSNR {db:.2} dB",
        samples.format
    );
    let codes = max_difference(&ours, &reference);
    assert!(
        codes <= EVEN_MAX_CODES,
        "{label} {} {spec:?}: {codes} codes",
        samples.format
    );
    true
}

#[test]
fn odd_frames_match_swscale_of_the_replicated_even_frame() {
    for size in ODD_SIZES {
        for format in FORMATS {
            let samples = Samples::smooth(size, format);
            for spec in specs() {
                if !check_against_even(&samples, &spec, "smooth_even") {
                    return;
                }
            }
        }
    }
}

#[test]
fn odd_frames_stay_near_swscale_of_the_odd_frame() {
    for size in ODD_SIZES {
        for format in FORMATS {
            let samples = Samples::smooth(size, format);
            for spec in specs() {
                let file = name("smooth_odd", &samples, &spec);
                let Some(reference) = swscale(&samples, &spec, &file) else {
                    return;
                };
                let ours = our_codes(&samples, &spec, &file);
                let db = psnr(&ours, &reference, 255.0);
                assert!(db > DRIFT_FLOOR_DB, "{file}: PSNR {db:.2} dB");
                let codes = max_difference(&ours, &reference);
                assert!(codes <= DRIFT_MAX_CODES, "{file}: {codes} codes");
            }
        }
    }
}

/// `tests/fixtures/gen_nvdec.py` writes these streams as BT.709 limited.
fn fixture(stream: &str) -> Option<Vec<u8>> {
    let directory = std::env::var_os("VIDR_NVDEC_FIXTURES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/nvdec"),
        PathBuf::from,
    );
    let path = directory.join(format!("{stream}.yuv"));
    match std::fs::read(&path) {
        Ok(bytes) => Some(bytes),
        Err(error) => {
            eprintln!(
                "skipping: {} does not read: {error}. Run `python3 \
                 crates/vidr/tests/fixtures/gen_nvdec.py`, or set VIDR_NVDEC_FIXTURES.",
                path.display()
            );
            None
        }
    }
}

fn check_fixture(stream: &str) {
    let Some(bytes) = fixture(stream) else {
        return;
    };
    let (width, height) = (321_usize, 181_usize);
    let frame_bytes = width * height + height.div_ceil(2) * width.div_ceil(2) * 2;
    let spec = YuvSpec::new(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left);
    assert_eq!(bytes.len() % frame_bytes, 0, "{stream}: partial frame");
    for (index, frame) in bytes.chunks_exact(frame_bytes).enumerate() {
        let samples = Samples::from_nv12(frame, width, height);
        let label = format!("{stream}_{index}");
        if !check_against_even(&samples, &spec, &label) {
            return;
        }
        let file = format!("{label}_odd");
        let Some(reference) = swscale(&samples, &spec, &file) else {
            return;
        };
        let db = psnr(&our_codes(&samples, &spec, &file), &reference, 255.0);
        assert!(db > DRIFT_FIXTURE_FLOOR_DB, "{file}: PSNR {db:.2} dB");
    }
}

#[test]
fn av1_odd_fixture_matches_swscale() {
    check_fixture("av1_odd");
}

#[test]
fn vp9_odd_fixture_matches_swscale() {
    check_fixture("vp9_odd");
}
