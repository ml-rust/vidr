//! One complete CPU run per decoder and encoder pair.

#[cfg(any(feature = "openh264", all(feature = "dav1d", feature = "cpu-av1")))]
use vidr_remaster::ResolutionMode;

#[cfg(any(feature = "openh264", all(feature = "dav1d", feature = "cpu-av1")))]
use crate::harness::{Case, assert_complete, assert_cpu_path};

/// rav1r is slow in a debug build, so its runs stay short.
#[cfg(all(feature = "dav1d", feature = "cpu-av1"))]
const AV1_FRAMES: u32 = 3;
#[cfg(feature = "openh264")]
const H264_FRAMES: u32 = 5;

/// ffmpeg's flags for its fastest setting of `encoder`.
#[cfg(all(feature = "dav1d", feature = "cpu-av1"))]
fn av1_args(encoder: &str) -> Vec<&str> {
    let mut args = vec!["-c:v", encoder];
    match encoder {
        "libaom-av1" => args.extend(["-cpu-used", "8", "-row-mt", "0"]),
        "libsvtav1" => args.extend(["-preset", "12"]),
        _ => args.extend(["-speed", "10"]),
    }
    args
}

#[cfg(all(feature = "dav1d", feature = "cpu-av1"))]
#[test]
fn dav1d_to_rav1r_keeps_frames_and_timing_and_states_the_path() {
    let Some(mut case) = Case::start("av1_to_av1") else {
        return;
    };
    let Some(encoder) = case.first_encoder(&["libaom-av1", "libsvtav1", "librav1e"]) else {
        return;
    };
    let codec = av1_args(encoder);
    let source = case.source("clip.mp4", AV1_FRAMES, &codec);
    let enhanced = case.enhanced("up.mp4", AV1_FRAMES, &codec);
    let output = case.path("out.mp4");
    let run = case.run(
        &source,
        &[
            "--codec",
            "av1",
            "--decoder",
            "cpu",
            "--encoder",
            "cpu",
            "--from-upscaled",
            enhanced.to_str().expect("a UTF-8 path"),
            "-o",
            output.to_str().expect("a UTF-8 path"),
        ],
    );
    let report = assert_complete(&run, &source, &output, ResolutionMode::Auto);
    assert_cpu_path(&run, "dav1d", "rav1r", u64::from(AV1_FRAMES));
    assert_eq!(report["processing"]["ai_executed"], false);
    assert_eq!(report["backend"], "rav1r resize of existing enhancement");
    let notices = run.frame_path()["notices"].as_array().expect("notices");
    assert!(
        notices
            .iter()
            .any(|notice| notice.as_str().is_some_and(|text| text.contains("rav1e"))),
        "{notices:?}"
    );
    let sidecar =
        std::fs::read_to_string(vidr_remaster::record_path(&output)).expect("the sidecar");
    assert!(sidecar.contains("\"decoder\": \"dav1d\""), "{sidecar}");
    assert!(sidecar.contains("\"encoder\": \"rav1r\""), "{sidecar}");
    vidr_remaster::check_record_for(
        &source,
        &output,
        vidr_remaster::ModelId::default(),
        Some("rav1r"),
        None,
    )
    .expect("the sidecar checks");
}

/// OpenH264 decodes and encodes, and an AAC track copies.
#[cfg(feature = "openh264")]
#[test]
fn openh264_both_ways_copies_audio_and_keeps_timing() {
    let Some(mut case) = Case::start("h264_to_h264") else {
        return;
    };
    if case.first_encoder(&["libx264"]).is_none() {
        return;
    }
    let baseline = [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-profile:v",
        "baseline",
    ];
    let plain = case.source("plain.mp4", H264_FRAMES, &baseline);
    let source = case.path("clip.mp4");
    let muxed = std::process::Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i"])
        .arg(&plain)
        .args(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"])
        .args([
            "-t", "0.2", "-c:v", "copy", "-c:a", "aac", "-map", "0:v", "-map", "1:a",
        ])
        .arg(&source)
        .output()
        .expect("ffmpeg runs");
    assert!(muxed.status.success(), "{muxed:?}");
    let enhanced = case.enhanced("up.mp4", H264_FRAMES, &baseline);
    let output = case.path("out.mp4");
    let run = case.run(
        &source,
        &[
            "--codec",
            "h264",
            "--from-upscaled",
            enhanced.to_str().expect("a UTF-8 path"),
            "-o",
            output.to_str().expect("a UTF-8 path"),
        ],
    );
    let report = assert_complete(&run, &source, &output, ResolutionMode::Auto);
    assert_cpu_path(&run, "openh264", "openh264", u64::from(H264_FRAMES));
    let audio = report["output_video"]["audio_streams"][0]["packets"]
        .as_u64()
        .expect("audio packets");
    assert!(audio > 0);
}

#[cfg(all(
    feature = "cpu-vp9",
    feature = "dav1d",
    feature = "cpu-av1",
    feature = "mkv"
))]
#[test]
fn rusty_vp9_to_rav1r_writes_matroska() {
    let Some(mut case) = Case::start("vp9_to_av1") else {
        return;
    };
    if case.first_encoder(&["libvpx-vp9"]).is_none() {
        return;
    }
    let codec = [
        "-c:v",
        "libvpx-vp9",
        "-deadline",
        "realtime",
        "-cpu-used",
        "8",
    ];
    let source = case.source("clip.webm", AV1_FRAMES, &codec);
    let enhanced = case.enhanced("up.webm", AV1_FRAMES, &codec);
    let output = case.path("out.mkv");
    let run = case.run(
        &source,
        &[
            "--codec",
            "av1",
            "--from-upscaled",
            enhanced.to_str().expect("a UTF-8 path"),
            "-o",
            output.to_str().expect("a UTF-8 path"),
        ],
    );
    assert_complete(&run, &source, &output, ResolutionMode::Auto);
    assert_cpu_path(&run, "rusty_vp9", "rav1r", u64::from(AV1_FRAMES));
    assert_eq!(run.frame_path()["output_check_decoder"], "dav1d");
}

#[cfg(all(feature = "cpu-hevc", feature = "dav1d", feature = "cpu-av1"))]
#[test]
fn hevc_decoder_to_rav1r() {
    let Some(mut case) = Case::start("hevc_to_av1") else {
        return;
    };
    if case.first_encoder(&["libx265"]).is_none() {
        return;
    }
    let codec = [
        "-c:v",
        "libx265",
        "-preset",
        "ultrafast",
        "-tag:v",
        "hvc1",
        "-x265-params",
        "log-level=error:bframes=0",
    ];
    let source = case.source("clip.mp4", AV1_FRAMES, &codec);
    let enhanced = case.enhanced("up.mp4", AV1_FRAMES, &codec);
    let output = case.path("out.mp4");
    let run = case.run(
        &source,
        &[
            "--codec",
            "av1",
            "--from-upscaled",
            enhanced.to_str().expect("a UTF-8 path"),
            "-o",
            output.to_str().expect("a UTF-8 path"),
        ],
    );
    assert_complete(&run, &source, &output, ResolutionMode::Auto);
    assert_cpu_path(&run, "hevc", "rav1r", u64::from(AV1_FRAMES));
}

/// The `fast` model on the CPU, when `BOOSTR_MODELS_DIR` holds it.
#[cfg(all(feature = "openh264", feature = "model-srvgg"))]
#[test]
fn the_fast_model_runs_on_the_cpu_runtime() {
    use std::path::PathBuf;

    use vidr_remaster::{ModelId, resolve_model_dir};

    let Some(root) = std::env::var_os("BOOSTR_MODELS_DIR") else {
        eprintln!("skipping: BOOSTR_MODELS_DIR is not set.");
        return;
    };
    let models = PathBuf::from(root);
    if let Err(error) = resolve_model_dir(&models, ModelId::Fast) {
        eprintln!("skipping: {error}");
        return;
    }
    let Some(mut case) = Case::start("model_on_cpu") else {
        return;
    };
    if case.first_encoder(&["libx264"]).is_none() {
        return;
    }
    let baseline = [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-profile:v",
        "baseline",
    ];
    let source = case.tiny_source("clip.mp4", 2, &baseline);
    let output = case.path("out.mp4");
    let run = case.run(
        &source,
        &[
            "--codec",
            "h264",
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
    assert_cpu_path(&run, "openh264", "openh264", 2);
    assert_eq!(report["processing"]["ai_executed"], true);
    assert_eq!(report["backend"], "boostr on numr CPU, F32");
    assert!(
        run.events
            .iter()
            .any(|event| event.to_string().contains("CPU F32")),
        "{:?}",
        run.events
    );
}
