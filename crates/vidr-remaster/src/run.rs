//! One whole run: validation, the report, the GPU path, and the named outcome.
//!
//! The GPU path lives in `run_gpu`, behind `cuda` and a container feature. A
//! build without them still validates, reports, and names what is missing.

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use vidr::Dimensions;
use vidr::pipeline::PumpControl;

use crate::error::RemasterError;
use crate::publish::{MissingSidecar, Published};
use crate::report::{ReportStatus, RunReport, state_home_from_env};
use crate::request::{Request, Validated};
use crate::reuse::Skipped;

/// What a run reports while it works. The text of each event is its `Display`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Progress {
    /// The pre-scan of the source starts.
    Analyzing {
        /// The source.
        source: PathBuf,
    },
    /// A scan reaches a round number of frames.
    Scanned {
        /// Frames so far.
        frames: u64,
    },
    /// `find_upscaled` passes over a candidate.
    Skipped(Skipped),
    /// An enhanced video replaces the model.
    Reusing {
        /// The enhanced video.
        path: PathBuf,
    },
    /// The model is on the device.
    ModelLoaded {
        /// The model's directory name in the catalog.
        name: &'static str,
        /// The device's name.
        device: String,
    },
    /// The sizes and frame count of the encode pass.
    Sizes {
        /// The source's upright size.
        input: Dimensions,
        /// The output's upright size.
        output: Dimensions,
        /// Frames to encode.
        frames: u64,
        /// Whether an enhanced video replaces the model.
        reused: bool,
    },
    /// The final path of the output.
    Output {
        /// The output.
        path: PathBuf,
    },
    /// Frames the encoder took so far.
    Frames {
        /// Frames so far.
        count: u64,
        /// Frames to encode.
        total: u64,
        /// Time since the encode pass started.
        elapsed: Duration,
    },
    /// The output check starts.
    Checking,
    /// The output is final.
    Saved {
        /// The output.
        path: PathBuf,
    },
    /// The report is written.
    Report {
        /// The report.
        path: PathBuf,
    },
    /// The report does not write. The run's own result stays as it is.
    ReportFailed {
        /// The error text, which names the report path and the cause.
        message: String,
    },
}

impl fmt::Display for Progress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Analyzing { source } => write!(f, "Analyzing: {}", source.display()),
            Self::Scanned { frames } => write!(f, "Scanned {frames} frames"),
            Self::Skipped(skipped) => fmt::Display::fmt(skipped, f),
            Self::Reusing { path } => write!(f, "Reusing AI-enhanced video: {}", path.display()),
            Self::ModelLoaded { name, device } => write!(f, "{name} | {device} | CUDA FP16"),
            Self::Sizes {
                input,
                output,
                frames,
                reused,
            } => {
                let step = if *reused {
                    "existing enhancement"
                } else {
                    "AI enhancement"
                };
                write!(
                    f,
                    "{input} \u{2192} {step} \u{2192} {output} | {} frames",
                    grouped(*frames)
                )
            }
            Self::Output { path } => write!(f, "Output: {}", path.display()),
            Self::Frames {
                count,
                total,
                elapsed,
            } => f.write_str(&progress_line(*count, *total, *elapsed)),
            Self::Checking => f.write_str("Checking the output..."),
            Self::Saved { path } => write!(f, "Saved: {}", path.display()),
            Self::Report { path } => write!(f, "Run report: {}", path.display()),
            Self::ReportFailed { message } => f.write_str(message),
        }
    }
}

/// `1234567` as `1,234,567`.
fn grouped(value: u64) -> String {
    let digits = value.to_string();
    let mut text = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            text.push(',');
        }
        text.push(digit);
    }
    text
}

/// The PoC's progress line: frames, percent, rate, and remaining time.
pub fn progress_line(count: u64, total: u64, elapsed: Duration) -> String {
    let seconds = elapsed.as_secs_f64();
    let fps = if seconds > 0.0 {
        count as f64 / seconds
    } else {
        0.0
    };
    let remaining = if fps > 0.0 {
        total.saturating_sub(count) as f64 / fps
    } else {
        0.0
    };
    let percent = if total > 0 {
        count as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    // A float to integer `as` saturates, so a huge estimate stays in range.
    let remaining = remaining.round() as u64;
    format!(
        "{}/{} frames ({percent:.1}%) | {fps:.1} fps | ETA {:02}:{:02}:{:02}",
        grouped(count),
        grouped(total),
        remaining / 3600,
        remaining / 60 % 60,
        remaining % 60
    )
}

/// How a run that returns no error ends.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RunOutcome {
    /// The output and its sidecar are written.
    Complete {
        /// The output.
        output: PathBuf,
        /// The run report, or `None` when it does not write. A
        /// [`Progress::ReportFailed`] event then names the path and the cause.
        report: Option<PathBuf>,
    },
    /// The output is final and complete, and its provenance sidecar did not land.
    OutputWithoutSidecar {
        /// Both paths and the cause.
        missing: MissingSidecar,
        /// The run report, or `None` when it does not write.
        report: Option<PathBuf>,
    },
    /// A stop request ended the run. No output is published, and the temporary
    /// output is gone.
    Stopped {
        /// The run report, or `None` when it does not write.
        report: Option<PathBuf>,
    },
}

/// The features a build leaves out for a whole run: empty in a complete build.
pub fn missing_features() -> Vec<&'static str> {
    let mut missing = Vec::new();
    if !cfg!(feature = "cuda") {
        missing.push("cuda");
    }
    if !cfg!(any(feature = "mp4", feature = "mkv")) {
        missing.push("mp4 or mkv");
    }
    missing
}

/// Remasters one video, with the report under `$XDG_STATE_HOME`, else
/// `~/.local/state`.
///
/// The steps run in this order: validation, the copy plan, a full NVDEC pre-scan,
/// the output sizes, reuse of an earlier upscale, the encode pass into a temporary
/// directory beside the output, the output check, the provenance sidecar in that
/// directory, the publish, the sidecar's rename, and the report.
///
/// `progress` runs on the calling thread and on one watcher thread, never both at
/// once. [`PumpControl::request_stop`] ends the run at the next step or frame.
///
/// # Errors
///
/// A [`RemasterError`] for any refusal or failure before the publish. A validation
/// refusal writes no report. Every later error writes one, and leaves no partial
/// output. After the publish, no error returns. A sidecar that does not land gives
/// [`RunOutcome::OutputWithoutSidecar`], and a report that does not write keeps the
/// outcome.
pub fn remaster(
    request: Request,
    mut progress: impl FnMut(Progress) + Send,
    control: &PumpControl,
) -> Result<RunOutcome, RemasterError> {
    let validated = request.validate()?;
    let state = state_home_from_env()?;
    run(&request, &validated, &mut progress, control, &state)
}

/// [`remaster`] with the report under `state_home`, whatever the environment says.
///
/// # Errors
///
/// As [`remaster`].
pub fn remaster_in(
    request: Request,
    mut progress: impl FnMut(Progress) + Send,
    control: &PumpControl,
    state_home: &Path,
) -> Result<RunOutcome, RemasterError> {
    let validated = request.validate()?;
    run(&request, &validated, &mut progress, control, state_home)
}

fn run<P: FnMut(Progress) + Send>(
    request: &Request,
    validated: &Validated,
    progress: &mut P,
    control: &PumpControl,
    state_home: &Path,
) -> Result<RunOutcome, RemasterError> {
    let mut report = RunReport::start(state_home, request, validated)?;
    let result = if control.stop_requested() {
        Err(RemasterError::Stopped)
    } else {
        drive(request, validated, progress, control, &mut report)
    };
    finish_run(result, &validated.destination, &mut report, progress)
}

/// Writes the report for `result`, and returns the run's outcome.
///
/// A report error never changes the outcome. After the publish it is a warning,
/// since the output is final.
fn finish_run<P: FnMut(Progress)>(
    result: Result<Published, RemasterError>,
    output: &Path,
    report: &mut RunReport,
    progress: &mut P,
) -> Result<RunOutcome, RemasterError> {
    let (status, message) = match &result {
        Ok(Published::WithoutSidecar(missing)) => (
            ReportStatus::OutputWithoutSidecar,
            Some(missing.to_string()),
        ),
        Ok(_) => (ReportStatus::Complete, None),
        Err(error) if error.is_stop() => (ReportStatus::Interrupted, None),
        Err(error) => (ReportStatus::Blocked, Some(error.to_string())),
    };
    let written = report.finish(status, message.as_deref());
    match &written {
        Ok(path) => progress(Progress::Report { path: path.clone() }),
        Err(error) if result.is_ok() => progress(Progress::ReportFailed {
            message: format!("Warning: {error} The output is final and stays."),
        }),
        Err(error) => progress(Progress::ReportFailed {
            message: error.to_string(),
        }),
    }
    let report = written.ok();
    match result {
        Ok(Published::WithoutSidecar(missing)) => {
            Ok(RunOutcome::OutputWithoutSidecar { missing, report })
        }
        Ok(_) => Ok(RunOutcome::Complete {
            output: output.to_path_buf(),
            report,
        }),
        Err(error) if error.is_stop() => Ok(RunOutcome::Stopped { report }),
        Err(error) => Err(error),
    }
}

#[cfg(all(feature = "cuda", any(feature = "mp4", feature = "mkv")))]
fn drive<P: FnMut(Progress) + Send>(
    request: &Request,
    validated: &Validated,
    progress: &mut P,
    control: &PumpControl,
    report: &mut RunReport,
) -> Result<Published, RemasterError> {
    crate::run_gpu::execute(request, validated, progress, control, report)
}

#[cfg(not(all(feature = "cuda", any(feature = "mp4", feature = "mkv"))))]
fn drive<P: FnMut(Progress) + Send>(
    _request: &Request,
    _validated: &Validated,
    _progress: &mut P,
    _control: &PumpControl,
    _report: &mut RunReport,
) -> Result<Published, RemasterError> {
    Err(RemasterError::MissingFeatures {
        missing: missing_features(),
    })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::Value;

    use super::*;

    const OUTPUT: &str = "/videos/clip.fast.restored.mp4";

    fn started(state: &Path) -> RunReport {
        let request = Request::new("/videos/clip.mp4");
        let validated = Validated {
            source: PathBuf::from("/videos/clip.mp4"),
            destination: PathBuf::from(OUTPUT),
        };
        RunReport::start(state, &request, &validated).expect("start")
    }

    /// Removes the run directory, so the report write fails.
    fn break_report(report: &RunReport) {
        fs::remove_dir_all(report.path().parent().expect("parent")).expect("remove");
    }

    fn missing() -> MissingSidecar {
        MissingSidecar {
            output: PathBuf::from(OUTPUT),
            sidecar: PathBuf::from("/videos/clip.fast.restored.mp4.upscale.json"),
            reason: "disk gone".to_owned(),
        }
    }

    fn finish(
        result: Result<Published, RemasterError>,
        report: &mut RunReport,
    ) -> (Result<RunOutcome, RemasterError>, Vec<Progress>) {
        let mut events = Vec::new();
        let outcome = finish_run(result, Path::new(OUTPUT), report, &mut |event: Progress| {
            events.push(event);
        });
        (outcome, events)
    }

    #[test]
    fn a_report_error_after_the_publish_keeps_the_run_complete_with_a_warning() {
        let state = tempfile::tempdir().expect("tempdir");
        let mut report = started(state.path());
        break_report(&report);
        let path = report.path().display().to_string();
        let (outcome, events) = finish(Ok(Published::Complete), &mut report);
        assert_eq!(
            outcome.expect("outcome"),
            RunOutcome::Complete {
                output: PathBuf::from(OUTPUT),
                report: None,
            }
        );
        let [Progress::ReportFailed { message }] = events.as_slice() else {
            panic!("other events: {events:?}");
        };
        assert!(message.starts_with("Warning: "), "{message}");
        assert!(message.contains(&path), "{message}");
        assert!(message.contains("os error"), "{message}");
    }

    #[test]
    fn a_complete_run_returns_its_report_path() {
        let state = tempfile::tempdir().expect("tempdir");
        let mut report = started(state.path());
        let path = report.path().to_path_buf();
        let (outcome, events) = finish(Ok(Published::Complete), &mut report);
        assert_eq!(
            outcome.expect("outcome"),
            RunOutcome::Complete {
                output: PathBuf::from(OUTPUT),
                report: Some(path.clone()),
            }
        );
        assert_eq!(events, [Progress::Report { path }]);
    }

    #[test]
    fn a_sidecar_that_does_not_land_is_reported_by_name_with_both_paths() {
        let state = tempfile::tempdir().expect("tempdir");
        let mut report = started(state.path());
        let path = report.path().to_path_buf();
        let (outcome, _) = finish(Ok(Published::WithoutSidecar(missing())), &mut report);
        assert_eq!(
            outcome.expect("outcome"),
            RunOutcome::OutputWithoutSidecar {
                missing: missing(),
                report: Some(path.clone()),
            }
        );
        let text = fs::read_to_string(path).expect("read");
        let value: Value = serde_json::from_str(&text).expect("JSON");
        assert_eq!(value["status"], "output_without_sidecar");
        let error = value["error"].as_str().expect("error");
        assert!(error.contains(OUTPUT), "{error}");
        assert!(error.contains(".upscale.json"), "{error}");
    }

    #[test]
    fn a_report_error_before_the_publish_keeps_the_failure() {
        let state = tempfile::tempdir().expect("tempdir");
        let mut report = started(state.path());
        break_report(&report);
        let (outcome, events) = finish(Err(RemasterError::NoGpu), &mut report);
        assert!(matches!(outcome, Err(RemasterError::NoGpu)), "{outcome:?}");
        let [Progress::ReportFailed { message }] = events.as_slice() else {
            panic!("other events: {events:?}");
        };
        assert!(!message.starts_with("Warning: "), "{message}");
    }

    #[test]
    fn a_report_error_after_a_stop_keeps_the_stop() {
        let state = tempfile::tempdir().expect("tempdir");
        let mut report = started(state.path());
        break_report(&report);
        let (outcome, _) = finish(Err(RemasterError::Stopped), &mut report);
        assert_eq!(
            outcome.expect("outcome"),
            RunOutcome::Stopped { report: None }
        );
    }

    #[test]
    fn numbers_group_in_threes() {
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(999), "999");
        assert_eq!(grouped(1000), "1,000");
        assert_eq!(grouped(1_234_567), "1,234,567");
    }

    #[test]
    fn the_progress_line_has_the_poc_form() {
        let line = progress_line(1000, 2000, Duration::from_secs(100));
        assert_eq!(line, "1,000/2,000 frames (50.0%) | 10.0 fps | ETA 00:01:40");
    }

    #[test]
    fn the_progress_line_handles_no_time_and_no_frames() {
        assert_eq!(
            progress_line(0, 0, Duration::ZERO),
            "0/0 frames (0.0%) | 0.0 fps | ETA 00:00:00"
        );
    }

    #[test]
    fn long_estimates_keep_counting_hours() {
        let line = progress_line(1, 3602, Duration::from_secs(1));
        assert!(line.ends_with("ETA 01:00:01"), "{line}");
    }

    #[test]
    fn events_print_the_poc_text() {
        let path = PathBuf::from("/videos/clip.mp4");
        assert_eq!(
            Progress::Analyzing {
                source: path.clone()
            }
            .to_string(),
            "Analyzing: /videos/clip.mp4"
        );
        assert_eq!(
            Progress::Reusing { path: path.clone() }.to_string(),
            "Reusing AI-enhanced video: /videos/clip.mp4"
        );
        assert_eq!(
            Progress::Output { path: path.clone() }.to_string(),
            "Output: /videos/clip.mp4"
        );
        assert_eq!(
            Progress::Saved { path: path.clone() }.to_string(),
            "Saved: /videos/clip.mp4"
        );
        assert_eq!(
            Progress::Report { path }.to_string(),
            "Run report: /videos/clip.mp4"
        );
        assert_eq!(
            Progress::Scanned { frames: 10000 }.to_string(),
            "Scanned 10000 frames"
        );
    }

    #[test]
    fn the_sizes_line_names_the_step() {
        let sizes = |reused| Progress::Sizes {
            input: Dimensions::new(406, 720).expect("size"),
            output: Dimensions::new(480, 852).expect("size"),
            frames: 12345,
            reused,
        };
        assert_eq!(
            sizes(false).to_string(),
            "406x720 \u{2192} AI enhancement \u{2192} 480x852 | 12,345 frames"
        );
        assert_eq!(
            sizes(true).to_string(),
            "406x720 \u{2192} existing enhancement \u{2192} 480x852 | 12,345 frames"
        );
    }

    #[test]
    fn missing_features_follow_the_build() {
        let missing = missing_features();
        assert_eq!(missing.contains(&"cuda"), !cfg!(feature = "cuda"));
        assert_eq!(
            missing.contains(&"mp4 or mkv"),
            !cfg!(any(feature = "mp4", feature = "mkv"))
        );
    }
}
