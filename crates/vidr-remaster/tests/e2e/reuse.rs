//! Reuse of an earlier upscale, and the flags around it.

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use vidr_remaster::{ResolutionMode, StandardResolution, record_path};

use crate::harness::Case;

const BACKEND: &str = "NVENC resize of existing enhancement";

/// Runs the model once and returns the source and `clip.fast.2x.mp4`, exactly twice
/// the source's size, which reuse looks for.
fn enhanced(case: &mut Case) -> (PathBuf, PathBuf) {
    let source = case.clip("clip.mp4", 6);
    let first = case.path("clip.fast.2x.mp4");
    let run = case.run(
        &source,
        &[
            "--resolution",
            "2x",
            "-o",
            first.to_str().expect("a UTF-8 path"),
        ],
    );
    case.assert_complete(&run, &source, &first, ResolutionMode::Double);
    assert_eq!(run.report()["processing"]["ai_executed"], true);
    (source, first)
}

#[test]
fn a_second_run_reuses_the_first_output_with_no_ai_pass() {
    let Some(mut case) = Case::start("reuse_second_run") else {
        return;
    };
    let (source, first) = enhanced(&mut case);
    // A model root that does not exist fails any model load.
    let no_models = case.aside("no-models");
    let run = case.run_with_models(&no_models, &source, &[]);
    let output = case.default_output(&source);
    case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    let report = run.report();
    assert_eq!(report["reused_upscaled"], first.display().to_string());
    assert_eq!(report["backend"], BACKEND);
    assert_eq!(report["processing"]["ai_executed"], false);
    assert_eq!(
        report["processing"]["reused_input"],
        first.display().to_string()
    );
}

#[test]
fn force_ai_runs_the_model_again() {
    let Some(mut case) = Case::start("reuse_force_ai") else {
        return;
    };
    let (source, _first) = enhanced(&mut case);
    let output = case.path("forced.mp4");
    let run = case.run(
        &source,
        &["--force-ai", "-o", output.to_str().expect("a UTF-8 path")],
    );
    case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    assert_eq!(run.report()["processing"]["ai_executed"], true);
    assert!(run.report().get("reused_upscaled").is_none());
}

#[test]
fn from_upscaled_resizes_without_a_model() {
    let Some(mut case) = Case::start("reuse_from_upscaled") else {
        return;
    };
    let (source, first) = enhanced(&mut case);
    // An explicit input needs no sidecar and no candidate name.
    let renamed = case.path("elsewhere.mp4");
    fs::rename(&first, &renamed).expect("renaming the upscale");
    fs::remove_file(record_path(&first)).expect("removing the sidecar");
    let output = case.path("resized.mp4");
    let no_models = case.aside("no-models");
    let run = case.run_with_models(
        &no_models,
        &source,
        &[
            "--from-upscaled",
            renamed.to_str().expect("a UTF-8 path"),
            "--resolution",
            "720",
            "-o",
            output.to_str().expect("a UTF-8 path"),
        ],
    );
    let mode = ResolutionMode::Fixed(StandardResolution::P720);
    let probe = case.assert_complete(&run, &source, &output, mode);
    assert_eq!(probe.size(), (960, 720));
    assert_eq!(run.report()["backend"], BACKEND);
    assert_eq!(run.report()["processing"]["ai_executed"], false);
}

#[test]
fn a_provenance_mismatch_is_refused() {
    let Some(mut case) = Case::start("reuse_mismatch") else {
        return;
    };
    let (source, first) = enhanced(&mut case);
    let sidecar = record_path(&first);
    let record = fs::read_to_string(&sidecar).expect("the sidecar");
    let changed = record.replace("\"model\": \"fast\"", "\"model\": \"realesrgan\"");
    assert_ne!(changed, record);
    fs::write(&sidecar, changed).expect("writing the sidecar");

    let run = case.run(&source, &[]);
    assert_eq!(run.exit, ExitCode::FAILURE);
    assert_eq!(run.report()["status"], "blocked");
    let error = run.error();
    assert!(error.contains("cannot be checked"), "{error}");
    assert!(error.contains("Recorded model differs"), "{error}");
    assert!(error.contains("--force-ai"), "{error}");
    assert!(!case.default_output(&source).exists());
}
