//! What a run refuses, and that each refusal leaves no file.

use std::fs;

use crate::harness::{Case, PICTURE, probe, run_ffmpeg};

/// Plain H.264 outside the media directory, with no colour signal in its SPS.
fn plain_source(case: &Case) -> String {
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
    plain.to_str().expect("a UTF-8 path").to_owned()
}

#[test]
fn hdr_tagged_in_the_container_alone_is_refused() {
    let Some(mut case) = Case::start("hdr_container") else {
        return;
    };
    let plain = plain_source(&case);
    // Stream copy leaves the SPS alone, so only the `colr` box states PQ.
    let source = case.generate(
        "clip.mp4",
        &[
            "-i",
            &plain,
            "-c",
            "copy",
            "-color_trc",
            "smpte2084",
            "-color_primaries",
            "bt2020",
            "-colorspace",
            "bt2020nc",
        ],
    );
    assert_eq!(probe(&source).video()["color_transfer"], "smpte2084");
    let run = case.run(&source, &[]);
    case.assert_refused(&run, "HDR video is unsupported", &[&source]);
    assert!(
        run.error().contains("TransferCharacteristics 16"),
        "{}",
        run.error()
    );
}

#[test]
fn interlaced_content_is_refused() {
    let Some(mut case) = Case::start("interlaced") else {
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
            "tinterlace=interleave_top",
            "-flags",
            "+ildct+ilme",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-bf",
            "0",
            "-x264-params",
            "tff=1",
            "-pix_fmt",
            "yuv420p",
        ],
    );
    assert_eq!(probe(&source).video()["field_order"], "tt");
    let run = case.run(&source, &[]);
    case.assert_refused(&run, "interlaced or MBAFF-coded", &[&source]);
    assert!(run.error().contains("deinterlace"), "{}", run.error());
}

#[test]
fn fragmented_mp4_is_refused() {
    let Some(mut case) = Case::start("fragmented") else {
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
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "frag_keyframe+empty_moov",
        ],
    );
    let run = case.run(&source, &[]);
    case.assert_refused(&run, "fragmented MP4", &[&source]);
    assert!(run.error().contains("Defragment"), "{}", run.error());
}

#[test]
fn a_mid_stream_size_change_is_refused() {
    let Some(mut case) = Case::start("size_change") else {
        return;
    };
    let mut joined = Vec::new();
    for (name, size) in [("a.h264", "320x240"), ("b.h264", "352x288")] {
        let part = case.aside(name);
        let picture = format!("testsrc2=size={size}:rate=25");
        run_ffmpeg(
            &[
                "-f",
                "lavfi",
                "-i",
                &picture,
                "-frames:v",
                "5",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-bf",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-f",
                "h264",
            ],
            &part,
        );
        joined.extend(fs::read(&part).expect("reading a part"));
    }
    let raw = case.aside("joined.h264");
    fs::write(&raw, joined).expect("writing the joined stream");
    let raw = raw.to_str().expect("a UTF-8 path");
    let source = case.generate(
        "clip.mkv",
        &["-r", "25", "-f", "h264", "-i", raw, "-c", "copy"],
    );
    let run = case.run(&source, &[]);
    let output = case.default_output(&source);
    assert!(!output.exists());
    case.assert_refused(&run, "Frame dimensions change", &[&source]);
    assert!(run.error().contains("352x288"), "{}", run.error());
    assert!(run.error().contains("320x240"), "{}", run.error());
}

#[test]
fn ass_subtitles_into_mp4_are_refused_and_name_mkv() {
    let Some(mut case) = Case::start("ass_to_mp4") else {
        return;
    };
    let plain = plain_source(&case);
    let subtitles = case.aside("subtitles.ass");
    fs::write(
        &subtitles,
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 320\nPlayResY: 240\n\n\
         [V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, \
         OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, \
         Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n\
         Style: Default,Arial,16,&Hffffff,&Hffffff,&H0,&H0,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,\
         1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, \
         Effect, Text\nDialogue: 0,0:00:00.00,0:00:00.20,Default,,0,0,0,,hello\n",
    )
    .expect("writing the subtitles");
    let subtitles = subtitles.to_str().expect("a UTF-8 path");
    let source = case.generate(
        "clip.mkv",
        &[
            "-i", &plain, "-i", subtitles, "-map", "0", "-map", "1", "-c:v", "copy", "-c:s", "ass",
        ],
    );
    assert_eq!(probe(&source).streams("subtitle")[0]["codec_name"], "ass");
    let output = case.path("out.mp4");
    let run = case.run(&source, &["-o", output.to_str().expect("a UTF-8 path")]);
    case.assert_refused(&run, "MKV", &[&source]);
    assert!(run.error().contains("S_TEXT/ASS"), "{}", run.error());
}
