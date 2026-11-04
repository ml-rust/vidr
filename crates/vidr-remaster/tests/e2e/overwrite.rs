//! An existing output: it stays without `--overwrite`, and goes with it.

use std::fs;
use std::process::ExitCode;

use vidr_remaster::ResolutionMode;

use crate::harness::Case;

const EXISTING: &[u8] = b"an existing file";

#[test]
fn an_existing_output_stays_without_overwrite_and_goes_with_it() {
    let Some(mut case) = Case::start("overwrite") else {
        return;
    };
    let source = case.clip("clip.mp4", 6);
    let output = case.default_output(&source);
    fs::write(&output, EXISTING).expect("writing the existing file");

    let refused = case.run(&source, &[]);
    assert_eq!(refused.exit, ExitCode::FAILURE);
    assert!(refused.report.is_none(), "validation writes no report");
    assert_eq!(fs::read(&output).expect("the file"), EXISTING);

    let run = case.run(&source, &["--overwrite"]);
    let probe = case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    assert_eq!(probe.size(), (640, 480));
    assert_ne!(fs::read(&output).expect("the file"), EXISTING);
}
