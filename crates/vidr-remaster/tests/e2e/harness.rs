//! What every end-to-end test shares: the skip rules, source generation, one run
//! through `main_with_args`, and the checks from outside.

use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Output};
use std::sync::{Mutex, MutexGuard, PoisonError};

use numr::runtime::cuda::is_cuda_available;
use serde_json::Value;
use vidr::decode::nvdec::nvdec_available;
use vidr::encode::nvenc::nvenc_available;
use vidr::media::{Input, OutputKind, plan_copy, read_stream_facts, read_track_facts};
use vidr::{StreamFacts, check_output, check_tracks};
use vidr_remaster::plan::output_sizes;
use vidr_remaster::{ModelId, ResolutionMode, check_record, main_with_args, resolve_model_dir};

/// The moving test picture of most sources.
pub const PICTURE: &str = "testsrc2=size=320x240:rate=25";
/// The audio of a source with a copied track.
pub const TONE: &str = "sine=frequency=440:sample_rate=48000";
/// Largest difference between the first video times of a source and an output, in
/// seconds: `DURATION_TOLERANCE`, which covers a millisecond clock on either side.
pub const START_TOLERANCE: f64 = 0.002;

/// `cargo test` runs these tests on threads of one process, and nextest runs each in
/// its own. This lock makes the first form serial, since a run reads the process's
/// `XDG_STATE_HOME`. The nextest `nvenc` group bounds the second form.
static SERIAL: Mutex<()> = Mutex::new(());

/// One test's directories and model. It holds the lock for the whole test.
pub struct Case {
    _serial: MutexGuard<'static, ()>,
    root: PathBuf,
    work: PathBuf,
    models: PathBuf,
    model: ModelId,
    states: u32,
}

/// One finished run.
pub struct Run {
    /// What `main_with_args` returned.
    pub exit: ExitCode,
    /// The run's `report.json`. A validation refusal writes none.
    pub report: Option<Value>,
}

impl Run {
    /// The report, which a run past validation always writes here.
    pub fn report(&self) -> &Value {
        self.report.as_ref().expect("the run report")
    }

    /// The report's `error` text.
    pub fn error(&self) -> &str {
        self.report()["error"]
            .as_str()
            .expect("the report's error text")
    }
}

impl Case {
    /// A case for the `fast` model, or `None` after a printed notice.
    pub fn start(name: &str) -> Option<Self> {
        Self::with_model(name, ModelId::Fast)
    }

    /// A case for `model`, or `None` after a printed notice.
    pub fn with_model(name: &str, model: ModelId) -> Option<Self> {
        let serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        if !gpu_ready() || !tools_ready() {
            return None;
        }
        let Some(root) = env::var_os("BOOSTR_MODELS_DIR") else {
            eprintln!("skipping: BOOSTR_MODELS_DIR is not set.");
            return None;
        };
        let models = PathBuf::from(root);
        if let Err(error) = resolve_model_dir(&models, model) {
            eprintln!("skipping: {error}");
            return None;
        }
        let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join("remaster_e2e")
            .join(name);
        if root.exists() {
            fs::remove_dir_all(&root).expect("clearing an earlier run");
        }
        let work = root.join("work");
        fs::create_dir_all(&work).expect("creating the test directory");
        // A run resolves its paths, so the paths here match the report's.
        let root = fs::canonicalize(&root).expect("resolving the test directory");
        let work = fs::canonicalize(&work).expect("resolving the test directory");
        Some(Self {
            _serial: serial,
            root,
            work,
            models,
            model,
            states: 0,
        })
    }

    /// `name` in the directory that holds the media and nothing else.
    pub fn path(&self, name: &str) -> PathBuf {
        self.work.join(name)
    }

    /// `name` outside the media directory, for files a listing leaves out.
    pub fn aside(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    /// The model root.
    pub fn models(&self) -> &Path {
        &self.models
    }

    /// `true` when ffmpeg holds `encoder`. Otherwise prints a notice.
    pub fn has_encoder(&self, encoder: &str) -> bool {
        let listed = run_tool(Command::new("ffmpeg").args(["-hide_banner", "-encoders"]));
        let text = String::from_utf8_lossy(&listed.stdout);
        let found = text
            .lines()
            .any(|line| line.split_whitespace().nth(1) == Some(encoder));
        if !found {
            eprintln!("skipping: this ffmpeg has no {encoder} encoder.");
        }
        found
    }

    /// Runs ffmpeg with `args`, which end in `name` in the media directory, and
    /// returns that path. A failure here is a broken test, so it panics.
    pub fn generate(&self, name: &str, args: &[&str]) -> PathBuf {
        let path = self.path(name);
        run_ffmpeg(args, &path);
        path
    }

    /// Ten frames of the test picture, H.264 with B-frames, and AAC.
    pub fn clip_with_audio(&self, name: &str) -> PathBuf {
        self.generate(
            name,
            &[
                "-f",
                "lavfi",
                "-i",
                PICTURE,
                "-f",
                "lavfi",
                "-i",
                TONE,
                "-frames:v",
                "10",
                "-t",
                "0.4",
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
            ],
        )
    }

    /// `frames` frames of the test picture as H.264 with no B-frames, video only.
    pub fn clip(&self, name: &str, frames: u32) -> PathBuf {
        let frames = frames.to_string();
        self.generate(
            name,
            &[
                "-f",
                "lavfi",
                "-i",
                PICTURE,
                "-frames:v",
                &frames,
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-bf",
                "0",
                "-pix_fmt",
                "yuv420p",
            ],
        )
    }

    /// A state home no earlier run of this case used.
    pub fn next_state(&mut self) -> PathBuf {
        self.states += 1;
        self.root.join(format!("state-{}", self.states))
    }

    /// Runs the command on `source`, with this case's model and model root.
    pub fn run(&mut self, source: &Path, flags: &[&str]) -> Run {
        let models = self.models.clone();
        self.run_with_models(&models, source, flags)
    }

    /// Runs the command with `models` as the model root.
    pub fn run_with_models(&mut self, models: &Path, source: &Path, flags: &[&str]) -> Run {
        let state = self.next_state();
        // SAFETY: `SERIAL` makes this test the only one in its process, and no other
        // thread of this test runs at this moment.
        unsafe { env::set_var("XDG_STATE_HOME", &state) };
        let mut args: Vec<OsString> = vec![
            "vidr-remaster".into(),
            source.into(),
            "--model".into(),
            self.model.as_str().into(),
            "--models-dir".into(),
            models.into(),
        ];
        args.extend(flags.iter().map(OsString::from));
        let exit = main_with_args(args);
        Run {
            exit,
            report: read_report(&state),
        }
    }

    /// The names in the media directory, sorted, hidden ones included.
    pub fn listing(&self) -> Vec<String> {
        listing(&self.work)
    }

    /// The default output of `source` for this case's model.
    pub fn default_output(&self, source: &Path) -> PathBuf {
        vidr_remaster::output_path(source, self.model, None).expect("the default output")
    }

    /// Asserts a complete run: exit 0, status `complete`, and no temporary directory.
    /// The sidecar checks, and ffmpeg decodes with no message. `check_output`,
    /// `check_tracks`, and ffprobe's frame count pass. Returns ffprobe's view.
    pub fn assert_complete(
        &self,
        run: &Run,
        source: &Path,
        output: &Path,
        mode: ResolutionMode,
    ) -> Probe {
        assert_eq!(run.exit, ExitCode::SUCCESS, "{:?}", run.report);
        assert_eq!(run.report()["status"], "complete", "{:?}", run.report);
        assert_eq!(
            run.report()["output"],
            output.display().to_string(),
            "the report names the output"
        );
        self.assert_no_temporary();
        check_record(source, output, self.model, None).expect("the sidecar checks");
        assert_decodes_cleanly(output);
        let stated = assert_output_checks(source, output, mode);
        let probe = probe(output);
        assert_eq!(
            probe.video()["nb_read_frames"],
            stated.frame_count().to_string(),
            "ffprobe's frame count"
        );
        probe
    }

    /// Asserts a refusal: exit 1, status `blocked`, `needle` in the error, and only
    /// `kept` in the media directory, so the failed run leaves no file.
    pub fn assert_refused(&self, run: &Run, needle: &str, kept: &[&Path]) {
        assert_eq!(run.exit, ExitCode::FAILURE, "{:?}", run.report);
        assert_eq!(run.report()["status"], "blocked", "{:?}", run.report);
        let error = run.error();
        assert!(error.contains(needle), "{needle} is not in: {error}");
        let mut expected: Vec<String> = kept
            .iter()
            .map(|path| {
                path.file_name()
                    .expect("a file name")
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        expected.sort();
        assert_eq!(self.listing(), expected);
    }

    fn assert_no_temporary(&self) {
        let hidden: Vec<String> = self
            .listing()
            .into_iter()
            .filter(|name| name.starts_with('.'))
            .collect();
        assert!(hidden.is_empty(), "{hidden:?}");
    }
}

fn gpu_ready() -> bool {
    if let Err(error) = nvdec_available() {
        eprintln!("skipping: {error}");
        return false;
    }
    if let Err(error) = nvenc_available() {
        eprintln!("skipping: {error}");
        return false;
    }
    if !is_cuda_available() {
        eprintln!("skipping: no CUDA device 0. Run this test on a machine with a GPU.");
        return false;
    }
    true
}

fn tools_ready() -> bool {
    for tool in ["ffmpeg", "ffprobe"] {
        match Command::new(tool).arg("-version").output() {
            Ok(output) if output.status.success() => {}
            Ok(output) => {
                eprintln!("skipping: {tool} -version exits {}.", output.status);
                return false;
            }
            Err(error) => {
                eprintln!("skipping: {tool} does not run: {error}. Install ffmpeg.");
                return false;
            }
        }
    }
    true
}

fn run_tool(command: &mut Command) -> Output {
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("{command:?} does not run: {error}"));
    assert!(
        output.status.success(),
        "{command:?} exits {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

/// Runs `ffmpeg -y -v error <args> <output>`.
pub fn run_ffmpeg(args: &[&str], output: &Path) {
    run_tool(
        Command::new("ffmpeg")
            .args(["-y", "-v", "error"])
            .args(args)
            .arg(output),
    );
}

fn listing(directory: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(directory)
        .expect("listing a directory")
        .map(|entry| {
            entry
                .expect("a directory entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    names
}

/// The one report under `state`, or `None` when the run wrote none.
pub fn read_report(state: &Path) -> Option<Value> {
    let runs = state.join("vidr-remaster");
    if !runs.is_dir() {
        return None;
    }
    let names = listing(&runs);
    assert_eq!(names.len(), 1, "one run directory: {names:?}");
    let text = fs::read_to_string(runs.join(&names[0]).join("report.json")).expect("the report");
    Some(serde_json::from_str(&text).expect("the report is JSON"))
}

/// ffprobe's view of a file, with every frame and packet counted.
pub struct Probe(Value);

impl Probe {
    /// The streams of `kind`, such as `audio`.
    pub fn streams(&self, kind: &str) -> Vec<&Value> {
        self.0["streams"]
            .as_array()
            .expect("the streams")
            .iter()
            .filter(|stream| stream["codec_type"] == kind)
            .collect()
    }

    /// The first video stream.
    pub fn video(&self) -> &Value {
        self.streams("video")
            .first()
            .copied()
            .expect("a video stream")
    }

    /// The first video stream's start, in seconds.
    pub fn video_start(&self) -> f64 {
        self.video()["start_time"]
            .as_str()
            .and_then(|text| text.parse().ok())
            .expect("the start time")
    }

    /// The container's duration, in seconds.
    pub fn duration(&self) -> f64 {
        self.0["format"]["duration"]
            .as_str()
            .and_then(|text| text.parse().ok())
            .expect("the duration")
    }

    /// The first video stream's rotation side data, 0 when it has none.
    pub fn rotation(&self) -> i64 {
        self.video()["side_data_list"]
            .as_array()
            .into_iter()
            .flatten()
            .find_map(|side_data| side_data["rotation"].as_i64())
            .unwrap_or(0)
    }

    /// The first video stream's width and height.
    pub fn size(&self) -> (i64, i64) {
        let side = |name: &str| self.video()[name].as_i64().expect("a stream side");
        (side("width"), side("height"))
    }
}

/// Reads `path` through ffprobe.
pub fn probe(path: &Path) -> Probe {
    let output = run_tool(
        Command::new("ffprobe")
            .args(["-v", "error", "-count_frames", "-count_packets"])
            .args(["-show_format", "-show_streams", "-print_format", "json"])
            .arg(path),
    );
    Probe(serde_json::from_slice(&output.stdout).expect("ffprobe's JSON"))
}

/// The pts and duration of each video packet in `path`, in ticks, in pts order.
/// ffprobe states no duration for a block with none.
pub fn video_packets(path: &Path) -> Vec<(i64, Option<i64>)> {
    let output = run_tool(
        Command::new("ffprobe")
            .args(["-v", "error", "-select_streams", "v:0"])
            .args(["-show_entries", "packet=pts,duration"])
            .args(["-print_format", "json"])
            .arg(path),
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("ffprobe's JSON");
    let mut packets: Vec<(i64, Option<i64>)> = value["packets"]
        .as_array()
        .expect("the packets")
        .iter()
        .map(|packet| {
            (
                packet["pts"].as_i64().expect("a pts"),
                packet["duration"].as_i64(),
            )
        })
        .collect();
    packets.sort_unstable();
    packets
}

/// Asserts that ffmpeg decodes every stream of `path` with no message.
pub fn assert_decodes_cleanly(path: &Path) {
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-f", "null", "-"])
        .output()
        .expect("ffmpeg runs");
    let messages = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{}: {messages}", path.display());
    assert!(messages.is_empty(), "{}: {messages}", path.display());
}

/// Runs `check_output` and `check_tracks` on `output`, from the same facts a run
/// reads, and returns the source's facts.
pub fn assert_output_checks(source: &Path, output: &Path, mode: ResolutionMode) -> StreamFacts {
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
    stated
}

/// Asserts that the first video time of `output` matches `source` within
/// [`START_TOLERANCE`], so copied tracks keep their sync.
pub fn assert_start_kept(source: &Path, output: &Probe) {
    let source_start = probe(source).video_start();
    let difference = (output.video_start() - source_start).abs();
    assert!(
        difference <= START_TOLERANCE,
        "the output starts at {}, and the source at {source_start}",
        output.video_start()
    );
}
