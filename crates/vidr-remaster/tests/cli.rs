//! The command line: each flag, each refusal, and the exit codes of the binary.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Output};

use vidr_remaster::cli::{
    Command as Parsed, EXIT_FAILURE, flag_help, parse_command, parse_request, run_command,
};
use vidr_remaster::{
    Codec, DecoderChoice, EncoderChoice, ModelId, Request, ResolutionMode, RuntimeChoice,
    StandardResolution, main_with_args, missing_features,
};

fn args(list: &[&str]) -> Vec<OsString> {
    std::iter::once("vidr-remaster")
        .chain(list.iter().copied())
        .map(OsString::from)
        .collect()
}

fn parse(list: &[&str]) -> Result<Request, clap::Error> {
    parse_request(args(list))
}

fn binary(state: &Path, list: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_vidr-remaster"))
        .args(list)
        .env("XDG_STATE_HOME", state)
        .output()
        .expect("the binary runs")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn runs(state: &Path) -> Vec<PathBuf> {
    match fs::read_dir(state.join("vidr-remaster")) {
        Ok(entries) => entries.map(|entry| entry.expect("entry").path()).collect(),
        Err(_) => Vec::new(),
    }
}

#[test]
fn the_source_alone_takes_every_default() {
    let request = parse(&["clip.mp4"]).expect("parse");
    assert_eq!(request, Request::new("clip.mp4"));
}

#[test]
fn each_flag_reaches_the_request() {
    let request = parse(&[
        "clip.mp4",
        "--model",
        "fast",
        "--codec",
        "h264",
        "--quality",
        "30",
        "--resolution",
        "720",
        "--overwrite",
        "--models-dir",
        "/models",
        "-o",
        "/out/clip.mkv",
        "--force-ai",
    ])
    .expect("parse");
    assert_eq!(request.source, Path::new("clip.mp4"));
    assert_eq!(request.model, ModelId::Fast);
    assert_eq!(request.codec, Codec::H264);
    assert_eq!(request.quality, 30);
    assert_eq!(
        request.resolution,
        ResolutionMode::Fixed(StandardResolution::P720)
    );
    assert!(request.overwrite);
    assert!(request.force_ai);
    assert_eq!(request.models_dir.as_deref(), Some(Path::new("/models")));
    assert_eq!(request.output.as_deref(), Some(Path::new("/out/clip.mkv")));
    assert_eq!(request.from_upscaled, None);
}

#[test]
fn the_execution_flags_reach_the_request() {
    let request = parse(&[
        "clip.mp4",
        "--codec",
        "av1",
        "--runtime",
        "wgpu",
        "--decoder",
        "cpu",
        "--encoder",
        "cpu",
        "--openh264-library",
        "/opt/libopenh264.so",
    ])
    .expect("parse");
    assert_eq!(request.codec, Codec::Av1);
    assert_eq!(request.runtime, RuntimeChoice::Wgpu);
    assert_eq!(request.decoder, DecoderChoice::Cpu);
    assert_eq!(request.encoder, EncoderChoice::Cpu);
    assert_eq!(
        request.openh264_library.as_deref(),
        Some(Path::new("/opt/libopenh264.so"))
    );
    let hardware = parse(&["clip.mp4", "--decoder", "nvdec", "--encoder", "nvenc"]).expect("parse");
    assert_eq!(hardware.decoder, DecoderChoice::Nvdec);
    assert_eq!(hardware.encoder, EncoderChoice::Nvenc);
}

#[test]
fn notices_asks_for_no_run_and_takes_no_source() {
    assert_eq!(
        parse_command(args(&["--notices"])).expect("parse"),
        Parsed::Notices
    );
    assert!(parse_command(args(&["clip.mp4", "--notices"])).is_err());
    assert!(parse(&["--notices"]).is_err());
    assert_eq!(run_command(args(&["--notices"])), 0);
}

#[test]
fn the_binary_prints_the_notices_and_the_help_names_rav1r_and_its_licences() {
    let dir = tempfile::tempdir().expect("tempdir");
    let notices = binary(dir.path(), &["--notices"]);
    assert_eq!(notices.status.code(), Some(0));
    let text = String::from_utf8_lossy(&notices.stdout).into_owned();
    assert!(text.contains("Apache-2.0"), "{text}");
    let help = binary(dir.path(), &["--help"]);
    assert_eq!(help.status.code(), Some(0));
    let text = String::from_utf8_lossy(&help.stdout).into_owned();
    assert!(text.contains("fork of rav1e"), "{text}");
    assert!(text.contains("BSD-2-Clause"), "{text}");
    assert!(
        text.contains("Alliance for Open Media Patent License 1.0"),
        "{text}"
    );
}

#[test]
fn the_long_output_flag_and_from_upscaled_parse() {
    let request = parse(&[
        "clip.mp4",
        "--output",
        "out.mp4",
        "--from-upscaled",
        "up.mkv",
    ])
    .expect("parse");
    assert_eq!(request.output.as_deref(), Some(Path::new("out.mp4")));
    assert_eq!(request.from_upscaled.as_deref(), Some(Path::new("up.mkv")));
}

#[test]
fn every_resolution_mode_parses() {
    for mode in [
        "auto", "original", "2x", "480", "720", "1080", "1440", "2160", "4320",
    ] {
        let request = parse(&["clip.mp4", "--resolution", mode]).expect("parse");
        assert_eq!(request.resolution.to_string(), mode);
    }
}

#[test]
fn unknown_values_are_refused_and_name_the_choices() {
    let cases = [
        (["--model", "tensorrt"], "fast, realesrgan, or purevision"),
        (["--codec", "vp9"], "hevc, h264, or av1"),
        (["--runtime", "metal"], "auto, cuda, wgpu, or cpu"),
        (["--decoder", "dxva"], "auto, nvdec, vulkan, or cpu"),
        (["--encoder", "vaapi"], "auto, nvenc, vulkan, or cpu"),
        (["--resolution", "999"], "auto, original, 2x"),
    ];
    for (flag, choices) in cases {
        let error = parse(&["clip.mp4", flag[0], flag[1]]).expect_err("refused");
        assert!(error.use_stderr());
        assert!(error.to_string().contains(choices), "{error}");
    }
}

#[test]
fn the_backend_flag_is_gone() {
    assert!(parse(&["clip.mp4", "--backend", "pytorch"]).is_err());
}

#[test]
fn a_source_is_required() {
    assert!(parse(&[]).is_err());
}

#[test]
fn a_negative_quality_is_refused_by_the_parser() {
    assert!(parse(&["clip.mp4", "--quality", "-1"]).is_err());
}

#[test]
fn from_upscaled_and_force_ai_are_exclusive() {
    let error =
        parse(&["clip.mp4", "--from-upscaled", "up.mp4", "--force-ai"]).expect_err("refused");
    assert!(error.use_stderr());
    assert_eq!(
        error.kind(),
        clap::error::ErrorKind::ArgumentConflict,
        "{error}"
    );
}

#[test]
fn each_help_line_starts_with_a_verb_and_shows_the_default() {
    let flags = flag_help();
    assert_eq!(flags.len(), 15, "{flags:?}");
    let verbs = [
        "Read", "Select", "Choose", "Set", "Replace", "Resize", "Run", "Print",
    ];
    for (flag, line) in flags {
        assert!(line.len() < 70, "{flag}: {line}");
        assert!(!line.contains('\n'), "{flag}: {line}");
        let first = line.split(' ').next().unwrap_or_default();
        assert!(verbs.contains(&first), "{flag}: {line}");
        if flag != "input" {
            assert!(line.contains("(default: "), "{flag}: {line}");
        }
    }
}

#[test]
fn help_and_version_exit_zero() {
    assert_eq!(run_command(args(&["--help"])), 0);
    assert_eq!(run_command(args(&["--version"])), 0);
    assert_eq!(main_with_args(args(&["--help"])), ExitCode::SUCCESS);
}

#[test]
fn a_bad_flag_exits_one() {
    assert_eq!(run_command(args(&["--nonsense"])), EXIT_FAILURE);
    assert_eq!(main_with_args(args(&["--nonsense"])), ExitCode::FAILURE);
}

#[test]
fn a_missing_source_exits_one_with_no_report() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join("absent.mp4");
    let output = binary(dir.path(), &[source.to_str().expect("utf-8")]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).starts_with("Input video does not exist:"),
        "{}",
        stderr(&output)
    );
    assert!(runs(dir.path()).is_empty());
}

#[test]
fn an_existing_output_exits_one_and_names_overwrite() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join("clip.mp4");
    fs::write(&source, b"video").expect("write");
    let target = dir.path().join("out.mp4");
    fs::write(&target, b"old").expect("write");
    let output = binary(
        dir.path(),
        &[
            source.to_str().expect("utf-8"),
            "-o",
            target.to_str().expect("utf-8"),
        ],
    );
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("--overwrite"),
        "{}",
        stderr(&output)
    );
    assert_eq!(fs::read(&target).expect("read"), b"old");
}

#[test]
fn quality_zero_and_an_unsupported_container_exit_one() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join("clip.mp4");
    fs::write(&source, b"video").expect("write");
    let source = source.to_str().expect("utf-8");
    let zero = binary(dir.path(), &[source, "--quality", "0"]);
    assert_eq!(zero.status.code(), Some(1));
    assert!(
        stderr(&zero).contains("quality 0 is refused"),
        "{}",
        stderr(&zero)
    );
    let avi = dir.path().join("out.avi");
    let container = binary(dir.path(), &[source, "-o", avi.to_str().expect("utf-8")]);
    assert_eq!(container.status.code(), Some(1));
    assert!(
        stderr(&container).contains("Unsupported output container .avi"),
        "{}",
        stderr(&container)
    );
}

/// A build with no whole path reports the features, in the text and in the report.
#[test]
fn the_binary_names_the_missing_features_and_writes_a_report() {
    if missing_features().is_empty() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join("clip.mp4");
    fs::write(&source, b"video").expect("write");
    let output = binary(dir.path(), &[source.to_str().expect("utf-8")]);
    assert_eq!(output.status.code(), Some(1));
    let text = stderr(&output);
    assert!(text.contains("--features cuda"), "{text}");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(stdout.contains("Run report: "), "{stdout}");
    let found = runs(dir.path());
    assert_eq!(found.len(), 1, "{found:?}");
    let report = fs::read_to_string(found[0].join("report.json")).expect("report");
    assert!(report.contains("\"status\": \"blocked\""), "{report}");
}
