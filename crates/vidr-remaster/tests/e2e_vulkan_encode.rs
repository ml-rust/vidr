//! Whole runs on the WebGPU runtime with Vulkan Video encode: the full hardware path of WebGPU.
//!
//! Run with `cargo nextest run -p vidr-remaster --features
//! mp4,vulkan-video,model-srvgg,cpu-av1,dav1d --test e2e_vulkan_encode`. No Vulkan video device, a
//! missing ffmpeg, a missing ffmpeg encoder, or missing weights skips with a printed notice.

#![cfg(all(feature = "vulkan-video", feature = "transform", feature = "mp4"))]

#[path = "e2e_cpu/harness.rs"]
mod harness;

use std::path::{Path, PathBuf};
use std::process::Command;

use vidr_remaster::ResolutionMode;

use harness::{Case, Run, assert_complete};

const FRAMES: u32 = 5;
/// libx264 High with B-frames.
const HIGH: [&str; 6] = ["-c:v", "libx264", "-preset", "fast", "-bf", "2"];

fn start(name: &str) -> Option<Case> {
    if let Err(error) = vidr::vulkan::vulkan_video_available() {
        eprintln!("skipping: {error}");
        return None;
    }
    if let Err(error) = vidr::vulkan::vulkan_encode_available() {
        eprintln!("skipping: {error}");
        return None;
    }
    let case = Case::start(name)?;
    case.first_encoder(&["libx264"])?;
    Some(case)
}

/// Asserts that a run's frames never reach the host.
fn assert_device_path(run: &Run, encoder: &str) {
    let path = run.frame_path();
    assert_eq!(path["runtime"], "wgpu");
    assert_eq!(path["decoder"]["name"], "vulkan");
    assert_eq!(path["decoder"]["route"], "device");
    assert_eq!(path["encoder"]["name"], encoder);
    assert_eq!(path["encoder"]["route"], "device");
    assert_eq!(path["output_check_decoder"], "vulkan");
    assert_eq!(path["host_copied_frames"], 0);
    assert_eq!(path["host_copied_bytes"], 0);
    assert!(!run.states_host_copies(), "{:?}", run.events);
}

/// Asserts that ffmpeg decodes `output` silently and ffprobe reads its size and packet count.
fn assert_readable(output: &Path, codec: &str, size: (u32, u32), name: &str) {
    let decoded = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(output)
        .args(["-f", "null", "-"])
        .output()
        .expect("ffmpeg runs");
    assert!(
        decoded.status.success() && decoded.stderr.is_empty(),
        "{name}: ffmpeg decode: {}",
        String::from_utf8_lossy(&decoded.stderr)
    );
    let probed = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-count_packets"])
        .args([
            "-show_entries",
            "stream=codec_name,width,height,nb_read_packets",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(output)
        .output()
        .expect("ffprobe runs");
    let text = String::from_utf8_lossy(&probed.stdout);
    assert!(probed.status.success(), "{name}: ffprobe failed");
    assert!(
        text.contains(&format!("codec_name={codec}")),
        "{name}: {text}"
    );
    assert!(
        text.contains(&format!("width={}", size.0)),
        "{name}: {text}"
    );
    assert!(
        text.contains(&format!("height={}", size.1)),
        "{name}: {text}"
    );
    assert!(
        text.contains(&format!("nb_read_packets={FRAMES}")),
        "{name}: {text}"
    );
}

/// `--from-upscaled` on Vulkan decode, the numr resize, and Vulkan encode.
fn from_upscaled(case: &mut Case, codec: &str, encoder: &str) -> (Run, PathBuf) {
    let source = case.source("clip.mp4", FRAMES, &HIGH);
    let enhanced = case.enhanced("up.mp4", FRAMES, &HIGH);
    let output = case.path("out.mp4");
    let run = case.run_on(
        "wgpu",
        &source,
        &[
            "--codec",
            codec,
            "--decoder",
            "vulkan",
            "--encoder",
            encoder,
            "--from-upscaled",
            enhanced.to_str().expect("a UTF-8 path"),
            "-o",
            output.to_str().expect("a UTF-8 path"),
        ],
    );
    assert_complete(&run, &source, &output, ResolutionMode::Auto);
    (run, output)
}

#[test]
fn h264_from_upscaled_stays_on_the_device() {
    let Some(mut case) = start("vulkan_encode_h264") else {
        return;
    };
    let (run, output) = from_upscaled(&mut case, "h264", "vulkan");
    assert_device_path(&run, "vulkan");
    assert_readable(&output, "h264", (640, 480), "h264");
    let sidecar = std::fs::read_to_string(output.with_extension("mp4.upscale.json"))
        .expect("the provenance sidecar");
    assert!(sidecar.contains("\"encoder\""), "{sidecar}");
    assert!(sidecar.contains("vulkan"), "{sidecar}");
}

#[test]
fn hevc_from_upscaled_stays_on_the_device() {
    let Some(mut case) = start("vulkan_encode_hevc") else {
        return;
    };
    let (run, output) = from_upscaled(&mut case, "hevc", "vulkan");
    assert_device_path(&run, "vulkan");
    assert_readable(&output, "hevc", (640, 480), "hevc");
}

#[test]
fn auto_on_webgpu_picks_vulkan_encode() {
    let Some(mut case) = start("vulkan_encode_auto") else {
        return;
    };
    let (run, output) = from_upscaled(&mut case, "hevc", "auto");
    assert_device_path(&run, "vulkan");
    let path = run.frame_path();
    let why = path["why"]["encoder"].as_str().expect("a reason");
    assert!(why.contains("Vulkan Video"), "{why}");
    assert_readable(&output, "hevc", (640, 480), "auto");
}

/// A driver with no AV1 encode refuses at open, and `auto` takes rav1r with that reason. A driver
/// that lists AV1 stays on Vulkan Video.
#[cfg(all(feature = "cpu-av1", feature = "dav1d"))]
#[test]
fn auto_falls_to_the_cpu_when_the_driver_lists_no_av1_encode() {
    let Some(mut case) = start("vulkan_encode_av1_auto") else {
        return;
    };
    let (run, output) = from_upscaled(&mut case, "av1", "auto");
    let path = run.frame_path();
    let why = path["why"]["encoder"].as_str().expect("a reason");
    if path["encoder"]["name"] == "rav1r" {
        assert_eq!(path["encoder"]["route"], "host download");
        assert!(why.contains("refused"), "{why}");
        assert!(why.contains("AV1"), "{why}");
    } else {
        assert_device_path(&run, "vulkan");
    }
    assert_readable(&output, "av1", (640, 480), "av1 auto");
}

/// The `fast` model on the full hardware path, when `BOOSTR_MODELS_DIR` holds it.
#[cfg(feature = "model-srvgg")]
#[test]
fn the_fast_model_runs_between_vulkan_decode_and_encode() {
    use vidr_remaster::{ModelId, resolve_model_dir};

    let Some(root) = std::env::var_os("BOOSTR_MODELS_DIR") else {
        eprintln!("skipping: BOOSTR_MODELS_DIR is unset");
        return;
    };
    let models = PathBuf::from(root);
    if let Err(error) = resolve_model_dir(&models, ModelId::Fast) {
        eprintln!("skipping: {error}");
        return;
    }
    let Some(mut case) = start("vulkan_encode_model") else {
        return;
    };
    let source = case.tiny_source("clip.mp4", FRAMES, &HIGH);
    let output = case.path("out.mp4");
    let run = case.run_on(
        "wgpu",
        &source,
        &[
            "--codec",
            "h264",
            "--decoder",
            "vulkan",
            "--encoder",
            "vulkan",
            "--model",
            "fast",
            "--models-dir",
            models.to_str().expect("a UTF-8 path"),
            "--force-ai",
            "-o",
            output.to_str().expect("a UTF-8 path"),
        ],
    );
    let report = assert_complete(&run, &source, &output, ResolutionMode::Auto);
    assert_device_path(&run, "vulkan");
    assert_eq!(report["processing"]["ai_executed"], true);
    assert_eq!(report["backend"], "boostr on numr WebGPU, F32");
}
