//! Reuse across runs: a change of encoder invalidates a recorded upscale.

#[cfg(all(feature = "openh264", feature = "cpu-av1", feature = "dav1d"))]
use crate::harness::Case;

#[cfg(all(feature = "openh264", feature = "cpu-av1", feature = "dav1d"))]
#[test]
fn a_change_of_encoder_stops_the_reuse_and_the_same_encoder_reuses() {
    let Some(mut case) = Case::start("reuse_encoder") else {
        return;
    };
    if case.first_encoder(&["libx264"]).is_none() {
        return;
    }
    let baseline = [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-profile:v",
        "baseline",
    ];
    let source = case.source("clip.mp4", 3, &baseline);
    let explicit = case.enhanced("explicit.mp4", 3, &baseline);
    let candidate = case.path("clip.fast.2x.mp4");
    let first = case.run(
        &source,
        &[
            "--model",
            "fast",
            "--codec",
            "h264",
            "--resolution",
            "2x",
            "--from-upscaled",
            explicit.to_str().expect("a UTF-8 path"),
            "-o",
            candidate.to_str().expect("a UTF-8 path"),
        ],
    );
    assert_eq!(first.report()["status"], "complete", "{:?}", first.report);

    // `--codec av1` runs rav1r, and the record names OpenH264.
    let changed = case.path("changed.mp4");
    let refused = case.run(
        &source,
        &[
            "--model",
            "fast",
            "--codec",
            "av1",
            "-o",
            changed.to_str().expect("a UTF-8 path"),
        ],
    );
    let error = refused.error();
    assert!(error.contains("Recorded encoder differs"), "{error}");
    assert!(error.contains("--from-upscaled"), "{error}");
    assert!(error.contains("--force-ai"), "{error}");
    assert!(!changed.exists());

    // The same encoder reuses the candidate with no model.
    let output = case.path("same.mp4");
    let reused = case.run(
        &source,
        &[
            "--model",
            "fast",
            "--codec",
            "h264",
            "-o",
            output.to_str().expect("a UTF-8 path"),
        ],
    );
    assert_eq!(reused.report()["status"], "complete", "{:?}", reused.report);
    assert_eq!(
        reused.report()["reused_upscaled"],
        candidate.display().to_string()
    );
}
