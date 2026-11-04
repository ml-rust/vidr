//! A stop request during the encode pass.

use std::ffi::OsStr;

use vidr::pipeline::PumpControl;
use vidr_remaster::{ModelId, Progress, Request, RunOutcome, remaster_in};

use crate::harness::{Case, read_report};

/// Long enough that the watcher thread sees frames before the pump finishes.
const FRAMES: u32 = 600;
/// The frame count at which the callback asks the run to stop.
const STOP_AT: u64 = 2;

#[test]
fn a_stop_request_mid_run_leaves_no_temporary_file_and_no_output() {
    let Some(mut case) = Case::start("stop") else {
        return;
    };
    let source = case.clip("clip.mp4", FRAMES);
    let state = case.next_state();
    let mut request = Request::new(&source);
    request.model = ModelId::Fast;
    request.models_dir = Some(case.models().to_path_buf());

    let control = PumpControl::new();
    let outcome = remaster_in(
        request,
        |event| {
            if let Progress::Frames { count, .. } = event
                && count >= STOP_AT
            {
                control.request_stop();
            }
        },
        &control,
        &state,
    )
    .expect("the outcome");

    assert!(
        matches!(outcome, RunOutcome::Stopped { report: Some(_) }),
        "{outcome:?}"
    );
    let frames = control.progress().frames_out;
    assert!((STOP_AT..u64::from(FRAMES)).contains(&frames), "{frames}");
    assert_eq!(
        read_report(&state).expect("the report")["status"],
        "interrupted"
    );
    let name = source.file_name().and_then(OsStr::to_str).expect("a name");
    assert_eq!(case.listing(), [name]);
}
