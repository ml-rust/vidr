//! A flat patch across decode, model, and encode, read back as raw YUV.
//!
//! A conversion that decodes with one matrix or range and encodes with another moves
//! the codes. The stream tags then say how a player reads them. A run that uses the
//! same wrong spec both ways keeps a flat patch's codes, so this test does not catch
//! that case.

use std::path::Path;
use std::process::Command;

use vidr_remaster::ResolutionMode;

use crate::harness::{Case, probe};

/// The patch: a saturated red, RGB 192, 32, 32.
const PATCH: &str = "color=c=0xC02020:size=320x240:rate=25,format=yuv420p";
/// Largest difference between a source and an output plane's mean code, 8-bit.
///
/// A mismatch moves this patch further. A range mismatch moves V by 10 codes. A matrix
/// mismatch moves Y by 12 codes.
const OUTPUT_TOLERANCE: f64 = 6.0;
/// Largest difference between the generated source and the codes of the formula.
const SOURCE_TOLERANCE: f64 = 2.0;
/// The mean of a square of this side, at the centre of each plane.
const SQUARE: usize = 16;

/// The mean Y, U, and V codes at the centre of the first frame of `path`.
fn centre_codes(path: &Path, width: usize, height: usize) -> [f64; 3] {
    // No `-pix_fmt`, so ffmpeg converts neither matrix nor range.
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-frames:v", "1", "-f", "rawvideo", "-"])
        .output()
        .expect("ffmpeg runs");
    assert!(
        output.status.success(),
        "{}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
    let bytes = output.stdout;
    let expected = width * height + 2 * chroma_width * chroma_height;
    assert_eq!(bytes.len(), expected, "one 8-bit 4:2:0 frame");
    let mean = |start: usize, plane_width: usize, plane_height: usize| {
        let (top, left) = ((plane_height - SQUARE) / 2, (plane_width - SQUARE) / 2);
        let mut sum = 0_u64;
        for row in top..top + SQUARE {
            let first = start + row * plane_width + left;
            sum += bytes[first..first + SQUARE]
                .iter()
                .map(|&code| u64::from(code))
                .sum::<u64>();
        }
        sum as f64 / (SQUARE * SQUARE) as f64
    };
    let chroma_u = width * height;
    let chroma_v = chroma_u + chroma_width * chroma_height;
    [
        mean(0, width, height),
        mean(chroma_u, chroma_width, chroma_height),
        mean(chroma_v, chroma_width, chroma_height),
    ]
}

fn assert_within(found: [f64; 3], expected: [f64; 3], tolerance: f64, what: &str) {
    for (plane, (found, expected)) in ["Y", "U", "V"].iter().zip(found.iter().zip(&expected)) {
        assert!(
            (found - expected).abs() <= tolerance,
            "{what}: {plane} is {found}, expected {expected} within {tolerance}"
        );
    }
}

/// One test: generate with `filter` and `tags`, run, and compare.
struct Patch<'a> {
    name: &'a str,
    filter: &'a str,
    tags: &'a [&'a str],
    /// The codes the BT.601 or BT.709 formula gives for the patch.
    codes: [f64; 3],
    matrix_code: u8,
    full_range: bool,
    matrix_name: &'a str,
    range_name: &'a str,
}

fn assert_patch_survives(patch: &Patch<'_>) {
    let Some(mut case) = Case::start(patch.name) else {
        return;
    };
    let mut args = vec![
        "-f",
        "lavfi",
        "-i",
        PATCH,
        "-frames:v",
        "3",
        "-vf",
        patch.filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-bf",
        "0",
        "-crf",
        "10",
    ];
    args.extend_from_slice(patch.tags);
    let source = case.generate("clip.mp4", &args);
    let read = probe(&source);
    assert_eq!(read.video()["color_space"], patch.matrix_name, "the source");
    assert_eq!(read.video()["color_range"], patch.range_name, "the source");
    let source_codes = centre_codes(&source, 320, 240);
    assert_within(source_codes, patch.codes, SOURCE_TOLERANCE, "the source");

    let run = case.run(&source, &[]);
    let output = case.default_output(&source);
    let written = case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    assert_eq!(written.size(), (640, 480));
    assert_eq!(
        written.video()["color_space"],
        patch.matrix_name,
        "the output"
    );
    assert_eq!(
        written.video()["color_range"],
        patch.range_name,
        "the output"
    );
    let report = run.report();
    for section in ["input_video", "output_video"] {
        assert_eq!(report[section]["matrix_coefficients"], patch.matrix_code);
        assert_eq!(report[section]["full_range"], patch.full_range);
    }
    let output_codes = centre_codes(&output, 640, 480);
    assert_within(output_codes, source_codes, OUTPUT_TOLERANCE, "the output");
}

#[test]
fn a_bt601_limited_patch_keeps_its_codes() {
    assert_patch_survives(&Patch {
        name: "colour_bt601_limited",
        filter: "scale=out_color_matrix=bt601:out_range=tv,format=yuv420p",
        tags: &["-colorspace", "smpte170m", "-color_range", "tv"],
        codes: [85.0, 104.0, 198.0],
        matrix_code: 6,
        full_range: false,
        matrix_name: "smpte170m",
        range_name: "tv",
    });
}

#[test]
fn a_bt709_full_range_patch_keeps_its_codes() {
    assert_patch_survives(&Patch {
        name: "colour_bt709_full",
        filter: "scale=out_color_matrix=bt709:out_range=pc,format=yuv420p",
        tags: &["-colorspace", "bt709", "-color_range", "pc"],
        codes: [66.0, 109.0, 207.0],
        matrix_code: 1,
        full_range: true,
        matrix_name: "bt709",
        range_name: "pc",
    });
}
