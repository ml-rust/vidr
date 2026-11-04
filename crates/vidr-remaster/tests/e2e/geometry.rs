//! Odd sizes, rotation, and sample aspect ratio.

use std::path::Path;

use vidr_remaster::ResolutionMode;

use crate::harness::{Case, PICTURE, probe, run_ffmpeg};

/// An odd size. `testsrc` keeps it, while `testsrc2` rounds it down to even.
const ODD_PICTURE: &str = "testsrc=size=321x181:rate=25,format=yuv420p";
/// 321x181 under the 480-pixel minimum: 480 * 321 / 181, to the nearest even.
const ODD_OUTPUT: (i64, i64) = (852, 480);

fn assert_odd_run(case: &mut Case, source: &Path) {
    assert_eq!(probe(source).size(), (321, 181), "the source's size");
    let run = case.run(source, &[]);
    let output = case.default_output(source);
    let probe = case.assert_complete(&run, source, &output, ResolutionMode::Auto);
    assert_eq!(probe.size(), ODD_OUTPUT);
    assert_eq!(run.report()["input_video"]["coded_width"], 321);
    assert_eq!(run.report()["input_video"]["coded_height"], 181);
}

#[test]
fn an_odd_av1_source_runs() {
    let Some(mut case) = Case::start("odd_av1") else {
        return;
    };
    if !case.has_encoder("libaom-av1") {
        return;
    }
    let source = case.generate(
        "odd.mp4",
        &[
            "-f",
            "lavfi",
            "-i",
            ODD_PICTURE,
            "-frames:v",
            "6",
            "-c:v",
            "libaom-av1",
            "-cpu-used",
            "8",
            "-crf",
            "40",
            "-g",
            "250",
        ],
    );
    assert_odd_run(&mut case, &source);
}

#[test]
fn an_odd_vp9_source_runs() {
    let Some(mut case) = Case::start("odd_vp9") else {
        return;
    };
    if !case.has_encoder("libvpx-vp9") {
        return;
    }
    let source = case.generate(
        "odd.webm",
        &[
            "-f",
            "lavfi",
            "-i",
            ODD_PICTURE,
            "-frames:v",
            "6",
            "-c:v",
            "libvpx-vp9",
            "-cpu-used",
            "8",
            "-deadline",
            "realtime",
        ],
    );
    assert_odd_run(&mut case, &source);
}

#[test]
fn a_rotated_source_keeps_its_rotation_and_the_display_size_follows_the_rule() {
    let Some(mut case) = Case::start("rotation") else {
        return;
    };
    let plain = case.aside("plain.mp4");
    run_ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            PICTURE,
            "-frames:v",
            "6",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
        ],
        &plain,
    );
    let plain = plain.to_str().expect("a UTF-8 path");
    let source = case.generate(
        "clip.mp4",
        &["-display_rotation", "90", "-i", plain, "-c", "copy"],
    );
    let source_rotation = probe(&source).rotation();
    assert_eq!(source_rotation.abs(), 90, "the source's rotation");

    let run = case.run(&source, &[]);
    let output = case.default_output(&source);
    let probe = case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    assert_eq!(probe.rotation(), source_rotation);
    // The upright source is 240x320. Its shorter edge rises to 480, so the upright
    // output is 480x640, and the encoder takes 640x480.
    assert_eq!(probe.size(), (640, 480));
    assert_eq!(run.report()["output_video"]["width"], 480);
    assert_eq!(run.report()["output_video"]["height"], 640);
}

#[test]
fn an_anamorphic_source_keeps_its_sample_and_display_ratios() {
    let Some(mut case) = Case::start("anamorphic") else {
        return;
    };
    let source = case.generate(
        "clip.mp4",
        &[
            "-f",
            "lavfi",
            "-i",
            PICTURE,
            "-frames:v",
            "6",
            "-vf",
            "setsar=4/3",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
        ],
    );
    assert_eq!(probe(&source).video()["sample_aspect_ratio"], "4:3");
    let run = case.run(&source, &[]);
    let output = case.default_output(&source);
    let probe = case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    assert_eq!(probe.size(), (640, 480));
    assert_eq!(probe.video()["sample_aspect_ratio"], "4:3");
    assert_eq!(probe.video()["display_aspect_ratio"], "16:9");
}
