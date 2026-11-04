//! What a CPU build refuses by name, each with a report and no output.

#[cfg(feature = "openh264")]
use crate::harness::Case;

/// `Case` with an H.264 Baseline source, when ffmpeg holds libx264.
#[cfg(feature = "openh264")]
fn baseline_case(name: &str) -> Option<(Case, std::path::PathBuf)> {
    let case = Case::start(name)?;
    case.first_encoder(&["libx264"])?;
    let codec = [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-profile:v",
        "baseline",
    ];
    let source = case.source("clip.mp4", 2, &codec);
    Some((case, source))
}

/// Asserts a blocked run: the report states the error, and only `kept` remains.
#[cfg(feature = "openh264")]
fn assert_blocked(run: &crate::harness::Run, case: &Case, kept: usize) {
    assert_eq!(run.report()["status"], "blocked");
    assert_eq!(run.report()["error"], run.error());
    let names = std::fs::read_dir(case.path("")).expect("the work directory");
    assert_eq!(names.count(), kept);
}

#[cfg(feature = "openh264")]
#[test]
fn a_source_with_b_frames_is_refused_by_name() {
    let Some(mut case) = Case::start("h264_b_frames") else {
        return;
    };
    if case.first_encoder(&["libx264"]).is_none() {
        return;
    }
    let codec = [
        "-c:v", "libx264", "-preset", "veryfast", "-bf", "2", "-g", "5",
    ];
    let source = case.source("clip.mp4", 6, &codec);
    let run = case.run(&source, &["--codec", "h264", "--decoder", "cpu"]);
    let error = run.error();
    assert!(error.contains("OpenH264"), "{error}");
    assert!(error.contains("B-frames"), "{error}");
    assert!(error.contains("reference decode"), "{error}");
    assert_blocked(&run, &case, 1);
}

#[cfg(feature = "openh264")]
#[test]
fn hevc_cpu_encode_is_refused_and_names_av1() {
    let Some((mut case, source)) = baseline_case("hevc_cpu_encode") else {
        return;
    };
    let run = case.run(&source, &["--codec", "hevc", "--encoder", "cpu"]);
    let error = run.error();
    assert!(error.contains("--codec av1"), "{error}");
    assert_blocked(&run, &case, 1);
}

#[cfg(feature = "openh264")]
#[test]
fn hardware_on_the_cpu_runtime_is_a_refused_pair() {
    let Some((mut case, source)) = baseline_case("refused_pair") else {
        return;
    };
    let run = case.run(&source, &["--decoder", "nvdec"]);
    let error = run.error();
    assert!(error.contains("--runtime cpu"), "{error}");
    assert!(error.contains("--decoder nvdec"), "{error}");
    assert!(error.contains("--runtime cuda"), "{error}");
    assert_blocked(&run, &case, 1);
}

#[cfg(all(feature = "openh264", not(feature = "cpu-av1")))]
#[test]
fn an_encoder_not_compiled_names_its_feature() {
    let Some((mut case, source)) = baseline_case("no_rav1r") else {
        return;
    };
    let run = case.run(&source, &["--codec", "av1"]);
    assert!(run.error().contains("cpu-av1"), "{}", run.error());
    assert_blocked(&run, &case, 1);
}
