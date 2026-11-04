//! Every input and output container pair, with copied AAC, and attachments.

use std::fs;
use std::path::{Path, PathBuf};

use vidr_remaster::ResolutionMode;

use crate::harness::{Case, PICTURE, TONE, assert_start_kept, run_ffmpeg};

/// The largest error `plan_rounded` states for a 1 ms clock, in nanoseconds.
const MAX_ROUNDING_NS: u64 = 500_000;

fn attachment(case: &Case) -> PathBuf {
    let path = case.aside("note.txt");
    fs::write(&path, b"attached bytes\n").expect("writing the attachment");
    path
}

/// An MKV source with AAC and one attachment. The files it starts from sit outside
/// the media directory.
fn clip_with_attachment(case: &Case, name: &str) -> PathBuf {
    let plain = case.aside("plain.mkv");
    run_ffmpeg(
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
        &plain,
    );
    let note = attachment(case);
    let plain = plain.to_str().expect("a UTF-8 path");
    let note = note.to_str().expect("a UTF-8 path");
    case.generate(
        name,
        &[
            "-i",
            plain,
            "-c",
            "copy",
            "-attach",
            note,
            "-metadata:s:t",
            "mimetype=text/plain",
        ],
    )
}

#[test]
fn mp4_to_mp4_copies_the_audio_and_states_no_rounding() {
    let Some(mut case) = Case::start("mp4_to_mp4") else {
        return;
    };
    let source = case.clip_with_audio("clip.mp4");
    let run = case.run(&source, &[]);
    let output = case.default_output(&source);
    let probe = case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    assert_eq!(probe.video()["codec_name"], "hevc");
    assert_eq!(probe.size(), (640, 480));
    assert_eq!(probe.streams("audio")[0]["codec_name"], "aac");
    assert!(run.report()["max_rounding_error_ns"].is_null());
    assert_start_kept(&source, &probe);
}

#[test]
fn mp4_to_mkv_rounds_the_audio_and_the_report_states_the_error() {
    let Some(mut case) = Case::start("mp4_to_mkv") else {
        return;
    };
    let source = case.clip_with_audio("clip.mp4");
    let output = case.path("out.mkv");
    let run = case.run(&source, &["-o", output.to_str().expect("a UTF-8 path")]);
    let probe = case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    assert_eq!(probe.streams("audio")[0]["codec_name"], "aac");
    // A 1024-sample AAC packet at 48 kHz lasts 21.333... ms, so a 1 ms clock rounds.
    let error = run.report()["max_rounding_error_ns"]
        .as_u64()
        .expect("the rounding error");
    assert!((1..=MAX_ROUNDING_NS).contains(&error), "{error}");
    assert_start_kept(&source, &probe);
}

#[test]
fn mkv_to_mp4_copies_the_audio() {
    let Some(mut case) = Case::start("mkv_to_mp4") else {
        return;
    };
    let source = case.clip_with_audio("clip.mkv");
    let output = case.path("out.mp4");
    let run = case.run(&source, &["-o", output.to_str().expect("a UTF-8 path")]);
    let probe = case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    assert_eq!(probe.streams("audio")[0]["codec_name"], "aac");
    assert!(run.report()["max_rounding_error_ns"].is_null());
    assert_start_kept(&source, &probe);
}

#[test]
fn mkv_to_mkv_keeps_an_attachment_byte_for_byte() {
    let Some(mut case) = Case::start("mkv_to_mkv") else {
        return;
    };
    let source = clip_with_attachment(&case, "clip.mkv");
    let output = case.path("out.mkv");
    let run = case.run(&source, &["-o", output.to_str().expect("a UTF-8 path")]);
    let probe = case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    assert!(run.report()["max_rounding_error_ns"].is_null());
    let attachments = probe.streams("attachment");
    assert_eq!(attachments.len(), 1, "{attachments:?}");
    assert_eq!(attachments[0]["tags"]["filename"], "note.txt");
    assert_eq!(attachments[0]["tags"]["mimetype"], "text/plain");
    let dumped = case.aside("dumped.txt");
    run_ffmpeg(
        &[
            "-dump_attachment:t:0",
            dumped.to_str().expect("a UTF-8 path"),
            "-i",
            output.to_str().expect("a UTF-8 path"),
            "-f",
            "null",
        ],
        Path::new("-"),
    );
    assert_eq!(fs::read(dumped).expect("the dump"), b"attached bytes\n");
}

#[test]
fn mkv_to_mp4_with_an_attachment_is_refused_and_names_mkv() {
    let Some(mut case) = Case::start("mkv_attachment_to_mp4") else {
        return;
    };
    let source = clip_with_attachment(&case, "clip.mkv");
    let output = case.path("out.mp4");
    let run = case.run(&source, &["-o", output.to_str().expect("a UTF-8 path")]);
    case.assert_refused(&run, "MKV", &[&source]);
    assert!(run.error().contains("note.txt"), "{}", run.error());
}
