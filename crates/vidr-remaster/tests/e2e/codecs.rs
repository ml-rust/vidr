//! The output codecs and the bit depth rule.

use vidr_remaster::ResolutionMode;

use crate::harness::{Case, PICTURE};

/// Six 10-bit HEVC frames in MP4, or `None` when ffmpeg has no libx265.
fn ten_bit_source(case: &Case) -> Option<std::path::PathBuf> {
    if !case.has_encoder("libx265") {
        return None;
    }
    Some(case.generate(
        "ten.mp4",
        &[
            "-f",
            "lavfi",
            "-i",
            PICTURE,
            "-frames:v",
            "6",
            "-c:v",
            "libx265",
            "-preset",
            "ultrafast",
            "-tag:v",
            "hvc1",
            "-x265-params",
            "log-level=error:bframes=0",
            "-pix_fmt",
            "yuv420p10le",
        ],
    ))
}

#[test]
fn h264_output_is_h264() {
    let Some(mut case) = Case::start("h264_output") else {
        return;
    };
    let source = case.clip("clip.mp4", 6);
    let run = case.run(&source, &["--codec", "h264"]);
    let output = case.default_output(&source);
    let probe = case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    assert_eq!(probe.video()["codec_name"], "h264");
    assert_eq!(probe.video()["pix_fmt"], "yuv420p");
    assert_eq!(run.report()["codec"], "h264");
}

#[test]
fn hevc_output_is_hevc() {
    let Some(mut case) = Case::start("hevc_output") else {
        return;
    };
    let source = case.clip("clip.mp4", 6);
    let run = case.run(&source, &["--codec", "hevc"]);
    let output = case.default_output(&source);
    let probe = case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    assert_eq!(probe.video()["codec_name"], "hevc");
    assert_eq!(probe.video()["pix_fmt"], "yuv420p");
    assert_eq!(run.report()["codec"], "hevc");
}

#[test]
fn a_ten_bit_source_gives_p010_hevc_at_ten_bits() {
    let Some(mut case) = Case::start("ten_bit_hevc") else {
        return;
    };
    let Some(source) = ten_bit_source(&case) else {
        return;
    };
    let run = case.run(&source, &[]);
    let output = case.default_output(&source);
    let probe = case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    assert_eq!(probe.video()["codec_name"], "hevc");
    assert_eq!(probe.video()["pix_fmt"], "yuv420p10le");
    assert_eq!(probe.video()["profile"], "Main 10");
    assert_eq!(run.report()["input_video"]["pixel_format"], "P010");
    assert_eq!(run.report()["output_video"]["pixel_format"], "P010");
}

#[test]
fn a_ten_bit_source_with_h264_is_refused_and_names_hevc() {
    let Some(mut case) = Case::start("ten_bit_h264") else {
        return;
    };
    let Some(source) = ten_bit_source(&case) else {
        return;
    };
    let run = case.run(&source, &["--codec", "h264"]);
    // The error text is `Pass --codec hevc.`
    case.assert_refused(&run, "--codec hevc", &[&source]);
}
