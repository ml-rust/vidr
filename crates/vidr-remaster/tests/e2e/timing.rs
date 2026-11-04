//! Variable frame rate, the last frame's duration, and edit lists.

use vidr_remaster::ResolutionMode;

use crate::harness::{Case, PICTURE, probe, run_ffmpeg, video_packets};

/// Eight frames on a 1 ms clock. Gaps of 40, 80, and 120 ms, and a last frame of
/// 40 ms that follows a gap of 80 ms, so no constant rate fits.
fn variable_rate_source(case: &Case) -> std::path::PathBuf {
    let source = case.generate(
        "clip.mp4",
        &[
            "-f",
            "lavfi",
            "-i",
            PICTURE,
            "-frames:v",
            "8",
            "-vf",
            "setpts='if(lt(N,4),N*0.04,0.16+(N-4)*0.1)/TB'",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-video_track_timescale",
            "1000",
        ],
    );
    let packets = video_packets(&source);
    let last = packets.last().copied().expect("a packet");
    let gap = last.0 - packets[packets.len() - 2].0;
    assert_ne!(
        last.1,
        Some(gap),
        "the source's last duration differs from its gap"
    );
    source
}

#[test]
fn variable_rate_mp4_keeps_every_pts_and_the_last_duration_exactly() {
    let Some(mut case) = Case::start("vfr_mp4") else {
        return;
    };
    let source = variable_rate_source(&case);
    let run = case.run(&source, &[]);
    let output = case.default_output(&source);
    case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    // MP4 stores each time in the source's own ticks, so the tolerance is zero.
    assert_eq!(video_packets(&output), video_packets(&source));
}

#[test]
fn variable_rate_mkv_keeps_every_pts_within_a_millisecond_tick() {
    let Some(mut case) = Case::start("vfr_mkv") else {
        return;
    };
    let source = variable_rate_source(&case);
    let output = case.path("out.mkv");
    let run = case.run(&source, &["-o", output.to_str().expect("a UTF-8 path")]);
    let written = case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    let source_pts: Vec<i64> = video_packets(&source).iter().map(|pair| pair.0).collect();
    let output_pts: Vec<i64> = video_packets(&output).iter().map(|pair| pair.0).collect();
    // Both clocks count whole milliseconds, so the half-tick rounding gives equality.
    assert_eq!(output_pts, source_pts);
    // The total holds the last frame's duration. Matroska keeps it within half a tick.
    let difference = (written.duration() - probe(&source).duration()).abs();
    assert!(
        difference <= 0.001,
        "the durations differ by {difference} s"
    );
}

#[test]
fn a_start_offset_survives() {
    let Some(mut case) = Case::start("start_offset") else {
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
            "8",
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
    // ffmpeg writes the offset as an empty edit of 0.5 s.
    let source = case.generate(
        "clip.mp4",
        &["-itsoffset", "0.5", "-i", plain, "-c", "copy"],
    );
    let run = case.run(&source, &[]);
    let output = case.default_output(&source);
    let probe = case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    // The output's empty edit is within half a movie tick. The movie clock of this
    // source is finer than this bound.
    let difference = (probe.video_start() - 0.5).abs();
    assert!(
        difference <= 0.001,
        "the output starts at {}",
        probe.video_start()
    );
    assert_eq!(run.report()["processing"]["dropped_frames"], 0);
}

#[test]
fn a_head_trim_drops_those_frames_before_the_model() {
    let Some(mut case) = Case::start("head_trim") else {
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
            "10",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-bf",
            "0",
            "-g",
            "250",
            "-pix_fmt",
            "yuv420p",
            "-video_track_timescale",
            "12800",
        ],
        &plain,
    );
    let plain = plain.to_str().expect("a UTF-8 path");
    // Stream copy from a seek keeps the packets from the keyframe. It writes an edit
    // that starts at frame 5: `elst` 2560 ticks from media time 2560.
    let source = case.generate("clip.mp4", &["-ss", "0.2", "-i", plain, "-c", "copy"]);
    let before = probe(&source);
    assert_eq!(before.video()["nb_read_packets"], "10");
    assert_eq!(before.video()["nb_read_frames"], "5");

    let run = case.run(&source, &[]);
    let output = case.default_output(&source);
    let written = case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    assert_eq!(written.video()["nb_read_packets"], "5");
    assert_eq!(written.video()["nb_read_frames"], "5");
    // The pump drops a frame before the transform, so only 5 of 10 reach the model.
    let processing = &run.report()["processing"];
    assert_eq!(processing["decoded_frames"], 10);
    assert_eq!(processing["dropped_frames"], 5);
    assert_eq!(processing["encoded_frames"], 5);
    assert_eq!(run.report()["input_video"]["frames"], 5);
    assert_eq!(run.report()["input_video"]["decoded_frames"], 10);
}
