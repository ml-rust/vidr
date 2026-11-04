//! One whole run: NVDEC, the `fast` model, NVENC, and the output check, through
//! `main_with_args`.
//!
//! Run with `cargo nextest run -p vidr-remaster --features cuda,mp4,model-srvgg
//! --test smoke`. `$BOOSTR_MODELS_DIR/realesr-general-x4v3` holds the model. A
//! missing model, GPU, or ffmpeg skips with a printed notice.

#![cfg(all(feature = "cuda", feature = "mp4", feature = "model-srvgg"))]

use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use numr::runtime::cuda::is_cuda_available;
use vidr::decode::nvdec::nvdec_available;
use vidr::encode::nvenc::nvenc_available;
use vidr::media::{Input, read_stream_facts, read_track_facts};
use vidr_remaster::{ModelId, main_with_args, record_path, resolve_model_dir};

const FRAMES: usize = 10;

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

fn models_root() -> Option<PathBuf> {
    let Some(root) = env::var_os("BOOSTR_MODELS_DIR") else {
        eprintln!("skipping: BOOSTR_MODELS_DIR is not set.");
        return None;
    };
    let root = PathBuf::from(root);
    if let Err(error) = resolve_model_dir(&root, ModelId::Fast) {
        eprintln!("skipping: {error}");
        return None;
    }
    Some(root)
}

/// A clip of `FRAMES` H.264 frames at 25 fps, and AAC audio to copy.
fn make_clip(path: &Path) -> bool {
    let generated = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-f", "lavfi", "-i"])
        .arg("testsrc2=size=128x96:rate=25")
        .args(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"])
        .args(["-frames:v", "10", "-t", "0.4", "-map", "0:v", "-map", "1:a"])
        .args(["-c:v", "libx264", "-preset", "veryfast", "-bf", "0"])
        .args(["-pix_fmt", "yuv420p", "-c:a", "aac"])
        .arg(path)
        .output();
    match generated {
        Ok(output) if output.status.success() => true,
        Ok(output) => {
            eprintln!(
                "skipping: ffmpeg cannot generate the clip: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            false
        }
        Err(error) => {
            eprintln!("skipping: ffmpeg does not run: {error}");
            false
        }
    }
}

#[test]
fn the_fast_model_writes_the_output_and_the_sidecar() {
    if !gpu_ready() {
        return;
    }
    let Some(models) = models_root() else {
        return;
    };
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("remaster_smoke");
    if directory.exists() {
        fs::remove_dir_all(&directory).expect("clearing an earlier run");
    }
    fs::create_dir_all(&directory).expect("creating the test directory");
    let source = directory.join("clip.mp4");
    if !make_clip(&source) {
        return;
    }
    let state = directory.join("state");
    // SAFETY: this file holds one test, and nothing else in its process reads
    // the environment on another thread at this moment.
    unsafe { env::set_var("XDG_STATE_HOME", &state) };

    let args: Vec<OsString> = vec![
        "vidr-remaster".into(),
        source.clone().into(),
        "--model".into(),
        "fast".into(),
        "--models-dir".into(),
        models.into(),
    ];
    assert_eq!(main_with_args(args), ExitCode::SUCCESS);

    let output = directory.join("clip.fast.restored.mp4");
    let input = Input::open(&output).expect("the output opens");
    let facts = read_stream_facts(&input).expect("video facts");
    assert_eq!(facts.frame_count(), FRAMES);
    assert_eq!(facts.geometry.display_dimensions().to_string(), "640x480");
    let tracks = read_track_facts(&input).expect("track facts");
    assert_eq!(tracks.audio.len(), 1, "{tracks:?}");

    let sidecar = fs::read_to_string(record_path(&output)).expect("the sidecar");
    assert!(sidecar.contains("\"model\": \"fast\""), "{sidecar}");

    let runs: Vec<PathBuf> = fs::read_dir(state.join("vidr-remaster"))
        .expect("the report directory")
        .map(|entry| entry.expect("entry").path())
        .collect();
    assert_eq!(runs.len(), 1, "{runs:?}");
    let report = fs::read_to_string(runs[0].join("report.json")).expect("the report");
    assert!(report.contains("\"status\": \"complete\""), "{report}");

    let leftovers: Vec<String> = fs::read_dir(&directory)
        .expect("the test directory")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter(|name| name.starts_with('.'))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}
