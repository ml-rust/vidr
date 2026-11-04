//! `remaster_in` with no GPU: the report on a stop and on a failure, and the
//! refusals that come before any work.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use vidr::pipeline::PumpControl;
use vidr_remaster::{
    Progress, RemasterError, Request, RequestError, RunOutcome, missing_features, remaster_in,
};

fn only_report(state: &Path) -> Value {
    let runs: Vec<PathBuf> = fs::read_dir(state.join("vidr-remaster"))
        .expect("run directory")
        .map(|entry| entry.expect("entry").path())
        .collect();
    assert_eq!(runs.len(), 1, "{runs:?}");
    let text = fs::read_to_string(runs[0].join("report.json")).expect("report");
    serde_json::from_str(&text).expect("JSON")
}

fn source_in(directory: &Path) -> PathBuf {
    let source = directory.join("clip.mp4");
    fs::write(&source, b"video").expect("write");
    source
}

fn entries(directory: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(directory)
        .expect("read_dir")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    names
}

#[test]
fn a_stop_before_the_run_gives_a_named_outcome_and_a_report() {
    let work = tempfile::tempdir().expect("tempdir");
    let state = tempfile::tempdir().expect("tempdir");
    let source = source_in(work.path());
    let control = PumpControl::new();
    control.request_stop();
    let mut events = Vec::new();
    let outcome = remaster_in(
        Request::new(&source),
        |event| events.push(event),
        &control,
        state.path(),
    )
    .expect("outcome");
    let RunOutcome::Stopped { report: Some(path) } = outcome else {
        panic!("not stopped: {outcome:?}");
    };
    assert_eq!(events, [Progress::Report { path: path.clone() }]);
    let report = only_report(state.path());
    assert_eq!(report["status"], "interrupted");
    assert!(report.get("error").is_none());
    assert_eq!(report["model"], "purevision");
    assert_eq!(report["codec"], "hevc");
    assert_eq!(report["quality"], 24);
    assert_eq!(report["resolution"], "auto");
    assert_eq!(report["decoder_request"], "auto");
    assert_eq!(report["encoder_request"], "auto");
    assert_eq!(report["runtime_request"], "auto");
    assert!(report["licence_notice"].as_str().is_some());
    assert_eq!(entries(work.path()), ["clip.mp4"]);
}

#[test]
fn a_refused_request_writes_no_report() {
    let work = tempfile::tempdir().expect("tempdir");
    let state = tempfile::tempdir().expect("tempdir");
    let control = PumpControl::new();
    let error = remaster_in(
        Request::new(work.path().join("absent.mp4")),
        |_| {},
        &control,
        state.path(),
    )
    .expect_err("refused");
    assert!(matches!(
        error,
        RemasterError::Request(RequestError::SourceMissing { .. })
    ));
    assert!(entries(state.path()).is_empty());
}

#[test]
fn a_report_directory_that_cannot_exist_is_refused_before_any_work() {
    let work = tempfile::tempdir().expect("tempdir");
    let source = source_in(work.path());
    let blocked = work.path().join("state");
    fs::write(&blocked, b"").expect("write");
    let control = PumpControl::new();
    let error =
        remaster_in(Request::new(&source), |_| {}, &control, &blocked).expect_err("refused");
    assert!(matches!(error, RemasterError::Report(_)), "{error}");
}

/// The named error of a build with no whole path, and the report that records it.
#[test]
fn a_build_with_no_whole_path_names_the_features_and_reports_the_failure() {
    if missing_features().is_empty() {
        return;
    }
    let work = tempfile::tempdir().expect("tempdir");
    let state = tempfile::tempdir().expect("tempdir");
    let source = source_in(work.path());
    let control = PumpControl::new();
    let mut events = Vec::new();
    let error = remaster_in(
        Request::new(&source),
        |event| events.push(event),
        &control,
        state.path(),
    )
    .expect_err("refused");
    let RemasterError::MissingFeatures { missing } = &error else {
        panic!("another error: {error}");
    };
    assert!(!missing.is_empty());
    let message = error.to_string();
    for feature in missing {
        assert!(message.contains(*feature), "{message}");
    }
    assert!(message.contains("--features cuda"), "{message}");
    assert!(matches!(events.as_slice(), [Progress::Report { .. }]));
    let report = only_report(state.path());
    assert_eq!(report["status"], "blocked");
    assert_eq!(report["error"], message);
    assert_eq!(entries(work.path()), ["clip.mp4"]);
}
