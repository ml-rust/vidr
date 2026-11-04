//! What every CPU run shares: the skip rules, source generation, one run through
//! `parse_request` and `remaster_in`, and the checks from outside.

// Each feature set compiles only some of the tests.
#![allow(dead_code)]

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use vidr::media::{Input, OutputKind, plan_copy, read_stream_facts, read_track_facts};
use vidr::pipeline::PumpControl;
use vidr::{FrameTiming, check_output, check_tracks};
use vidr_remaster::cli::parse_request;
use vidr_remaster::plan::output_sizes;
use vidr_remaster::{Progress, RemasterError, ResolutionMode, RunOutcome, remaster_in};

/// The moving test picture of every source, at the size `--from-upscaled` resizes.
const PICTURE: &str = "testsrc2=size=320x240:rate=25";
/// The picture of an enhanced video, twice the source's size.
const LARGE_PICTURE: &str = "testsrc2=size=640x480:rate=25";
/// The picture of a source that a model run decodes, as small as a macroblock grid allows.
const TINY_PICTURE: &str = "testsrc2=size=64x48:rate=25";

/// One test's directories.
pub struct Case {
    work: PathBuf,
    state: PathBuf,
    runs: u32,
}

/// One finished run.
pub struct Run {
    /// What `remaster_in` returned.
    pub result: Result<RunOutcome, RemasterError>,
    /// Each progress event.
    pub events: Vec<Progress>,
    /// The run's `report.json`. A validation refusal writes none.
    pub report: Option<Value>,
}

impl Run {
    /// The report, which a run past validation always writes.
    pub fn report(&self) -> &Value {
        self.report.as_ref().expect("the run report")
    }

    /// The report's `frame_path` section.
    pub fn frame_path(&self) -> &Value {
        &self.report()["frame_path"]
    }

    /// The error's text.
    pub fn error(&self) -> String {
        match &self.result {
            Ok(outcome) => panic!("the run completes: {outcome:?}"),
            Err(error) => error.to_string(),
        }
    }

    /// Whether stdout gets the frame path line.
    pub fn states_host_copies(&self) -> bool {
        self.events.iter().any(|event| {
            matches!(event, Progress::FramePath { .. }) && event.to_string().contains("host cop")
        })
    }
}

/// The encoder names of this ffmpeg, or `None` after a printed notice.
fn ffmpeg_encoders() -> Option<String> {
    let listed = Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .output();
    match listed {
        Ok(output) if output.status.success() => {
            Some(String::from_utf8_lossy(&output.stdout).into_owned())
        }
        Ok(output) => {
            eprintln!(
                "skipping: ffmpeg lists no encoders: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            None
        }
        Err(error) => {
            eprintln!("skipping: ffmpeg does not run: {error}");
            None
        }
    }
}

impl Case {
    /// A case with an empty work directory, or `None` after a printed notice.
    pub fn start(name: &str) -> Option<Self> {
        ffmpeg_encoders()?;
        let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join("remaster_e2e_cpu")
            .join(name);
        if root.exists() {
            fs::remove_dir_all(&root).expect("clearing an earlier run");
        }
        let work = root.join("work");
        fs::create_dir_all(&work).expect("creating the test directory");
        // A run resolves its paths, so the paths here match the report's.
        let root = fs::canonicalize(&root).expect("resolving the test directory");
        Some(Self {
            work: root.join("work"),
            state: root.join("state"),
            runs: 0,
        })
    }

    /// `name` in the work directory.
    pub fn path(&self, name: &str) -> PathBuf {
        self.work.join(name)
    }

    /// The first of `encoders` that this ffmpeg holds, or `None` after a notice.
    pub fn first_encoder<'e>(&self, encoders: &[&'e str]) -> Option<&'e str> {
        let listed = ffmpeg_encoders()?;
        let found = encoders.iter().copied().find(|encoder| {
            listed
                .lines()
                .any(|line| line.split_whitespace().nth(1) == Some(*encoder))
        });
        if found.is_none() {
            eprintln!("skipping: this ffmpeg has none of {encoders:?}.");
        }
        found
    }

    /// Writes `frames` frames of the source picture to `name`, encoded by `codec`.
    pub fn source(&self, name: &str, frames: u32, codec: &[&str]) -> PathBuf {
        self.video(PICTURE, name, frames, codec)
    }

    /// Writes `frames` frames of the tiny picture to `name`, encoded by `codec`.
    pub fn tiny_source(&self, name: &str, frames: u32, codec: &[&str]) -> PathBuf {
        self.video(TINY_PICTURE, name, frames, codec)
    }

    /// Writes `frames` frames of the enhanced picture to `name`, encoded by `codec`.
    pub fn enhanced(&self, name: &str, frames: u32, codec: &[&str]) -> PathBuf {
        self.video(LARGE_PICTURE, name, frames, codec)
    }

    fn video(&self, picture: &str, name: &str, frames: u32, codec: &[&str]) -> PathBuf {
        let path = self.path(name);
        let generated = Command::new("ffmpeg")
            .args(["-y", "-v", "error", "-f", "lavfi", "-i", picture])
            .args(["-frames:v", &frames.to_string(), "-pix_fmt", "yuv420p"])
            .args(codec)
            .arg(&path)
            .output()
            .expect("ffmpeg runs");
        assert!(
            generated.status.success(),
            "ffmpeg cannot write {name}: {}",
            String::from_utf8_lossy(&generated.stderr)
        );
        path
    }

    /// Runs the command on `source` with `flags` on the CPU runtime, on its own state
    /// directory.
    pub fn run(&mut self, source: &Path, flags: &[&str]) -> Run {
        self.run_on("cpu", source, flags)
    }

    /// Runs the command on `source` with `flags` on `runtime`, on its own state
    /// directory.
    pub fn run_on(&mut self, runtime: &str, source: &Path, flags: &[&str]) -> Run {
        self.runs += 1;
        let state = self.state.join(self.runs.to_string());
        let mut args: Vec<OsString> = vec!["vidr-remaster".into(), source.into()];
        args.extend(["--runtime", runtime].iter().map(OsString::from));
        args.extend(flags.iter().map(OsString::from));
        let request = parse_request(args).expect("the flags parse");
        let control = PumpControl::new();
        let mut events = Vec::new();
        let result = remaster_in(request, |event| events.push(event), &control, &state);
        Run {
            result,
            events,
            report: read_report(&state),
        }
    }
}

fn read_report(state: &Path) -> Option<Value> {
    let entries = fs::read_dir(state.join("vidr-remaster")).ok()?;
    let runs: Vec<PathBuf> = entries.map(|entry| entry.expect("entry").path()).collect();
    assert_eq!(runs.len(), 1, "{runs:?}");
    let text = fs::read_to_string(runs[0].join("report.json")).expect("the report");
    Some(serde_json::from_str(&text).expect("JSON"))
}

/// Asserts a complete run. `check_output` and `check_tracks` pass, and the output
/// keeps the source's frame count and each pts on the source's clock. Returns the
/// report.
pub fn assert_complete(run: &Run, source: &Path, output: &Path, mode: ResolutionMode) -> Value {
    let outcome = run.result.as_ref().expect("the run completes");
    assert!(
        matches!(outcome, RunOutcome::Complete { .. }),
        "{outcome:?}"
    );
    assert_eq!(run.report()["status"], "complete", "{:?}", run.report);
    let mut input = Input::open(source).expect("the source opens");
    let kind = OutputKind::from_path(output).expect("the output kind");
    let expected = plan_copy(&mut input, kind)
        .expect("the copy plan")
        .expected_tracks();
    let stated = read_stream_facts(&input).expect("the source's facts");
    let sizes = output_sizes(&stated.geometry, mode).expect("the output size");
    let written = Input::open(output).expect("the output opens");
    let facts = read_stream_facts(&written).expect("the output's facts");
    let tracks = read_track_facts(&written).expect("the output's tracks");
    check_output(&stated, &facts, sizes.display).expect("check_output");
    check_tracks(&expected, &tracks).expect("check_tracks");
    assert_eq!(facts.frame_count(), stated.frame_count(), "frame count");
    assert_eq!(seconds(&stated), seconds(&facts), "frame times");
    run.report().clone()
}

/// Each frame's start after the first, in whole milliseconds.
fn seconds(stream: &vidr::StreamFacts) -> Vec<i64> {
    let base = stream.time_base;
    let den = i64::try_from(base.den()).expect("a small time base");
    let first = stream.frames.first().map_or(0, |frame| frame.pts);
    stream
        .frames
        .iter()
        .map(|frame: &FrameTiming| (frame.pts - first) * base.num() * 1000 / den)
        .collect()
}

/// Asserts the frame path of a run on the CPU runtime with host copies on both sides.
pub fn assert_cpu_path(run: &Run, decoder: &str, encoder: &str, frames: u64) {
    let path = run.frame_path();
    assert_eq!(path["runtime"], "cpu");
    assert_eq!(path["decoder"]["name"], decoder);
    assert_eq!(path["decoder"]["route"], "host only");
    assert_eq!(path["encoder"]["name"], encoder);
    assert_eq!(path["encoder"]["route"], "host only");
    assert_eq!(path["host_copied_frames"], 2 * frames);
    assert!(path["host_copied_bytes"].as_u64().expect("bytes") > 0);
    assert!(path["scan_host_copied_frames"].as_u64().expect("frames") >= 2 * frames);
    assert!(path["why"]["decoder"].as_str().is_some());
    assert!(path["why"]["encoder"].as_str().is_some());
    assert!(run.states_host_copies(), "{:?}", run.events);
}
