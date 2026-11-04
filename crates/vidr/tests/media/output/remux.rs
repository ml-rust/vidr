//! A remux of each codec into each output kind reads back with the same
//! streams in ffprobe and passes vidr's own output checks.

use vidr::check::{check_output, check_tracks};
use vidr::media::{Input, OutputKind, read_stream_facts, read_track_facts};

use super::probe::{Probe, assert_near, probe};
use super::support::{Remuxed, remux};

/// 1 ms and one rounding step of ffprobe's nanosecond view.
const ONE_TICK_NS: i128 = 1_000_001;
/// `DURATION_TOLERANCE`, which also covers a reader's own rounding of a copied
/// track's `CodecDelay`.
const TWO_TICKS_NS: i128 = 2_000_000;

const FIXTURES: [&str; 4] = [
    "remux_h264.mp4",
    "remux_h264.mkv",
    "remux_hevc.mp4",
    "remux_hevc.mkv",
];
const KINDS: [OutputKind; 2] = [OutputKind::Mp4, OutputKind::Mkv];

/// The tag of a stream title: `name` in MP4, `title` in Matroska.
fn title(probe: &Probe, index: usize) -> Option<String> {
    probe
        .stream_tag(index, "title")
        .or_else(|| probe.stream_tag(index, "name"))
        .map(str::to_owned)
}

fn assert_video(name: &str, source: &Probe, output: &Probe) {
    let from = source.streams("video")[0];
    let to = output.streams("video");
    assert_eq!(to.len(), 1, "{name}: video streams");
    let to = to[0];
    for field in [
        "codec_name",
        "width",
        "height",
        "sample_aspect_ratio",
        "color_range",
        "color_space",
        "color_transfer",
        "color_primaries",
    ] {
        assert_eq!(
            output.stream(to, field),
            source.stream(from, field),
            "{name}: video {field}"
        );
    }
    assert_eq!(output.rotation(to), source.rotation(from), "{name}");
    assert_near(
        &format!("{name}: video pts"),
        &source.sorted_pts(from),
        &output.sorted_pts(to),
        ONE_TICK_NS,
    );
}

fn assert_audio(name: &str, source: &Probe, output: &Probe) {
    let from = source.streams("audio");
    let to = output.streams("audio");
    assert_eq!(to.len(), from.len(), "{name}: audio streams");
    for (&from, &to) in from.iter().zip(&to) {
        for field in ["codec_name", "sample_rate", "channels"] {
            assert_eq!(
                output.stream(to, field),
                source.stream(from, field),
                "{name}: audio {field}"
            );
        }
        assert_eq!(output.stream_tag(to, "language"), Some("eng"), "{name}");
        assert_eq!(title(output, to).as_deref(), Some("Tone"), "{name}");
        // Relative to the first packet: Matroska to MP4 writes `CodecDelay` as an
        // edit, so ffprobe's absolute times differ by the priming samples.
        let relative = |pts: Vec<i128>| -> Vec<i128> {
            let first = pts[0];
            pts.iter().map(|value| value - first).collect()
        };
        assert_near(
            &format!("{name}: audio pts"),
            &relative(source.sorted_pts(from)),
            &relative(output.sorted_pts(to)),
            TWO_TICKS_NS,
        );
    }
}

/// The cue times of a text stream. MP4 fills the gaps with two-byte samples.
fn cues(probe: &Probe, index: usize) -> Vec<i128> {
    let empty = if probe.stream(index, "codec_name") == Some("mov_text") {
        2
    } else {
        0
    };
    probe
        .packets(index)
        .into_iter()
        .filter(|&(_, size)| size > empty)
        .map(|(pts, _)| pts)
        .collect()
}

fn assert_subtitles(name: &str, kind: OutputKind, source: &Probe, output: &Probe) {
    let from = source.streams("subtitle");
    let to = output.streams("subtitle");
    assert_eq!(to.len(), from.len(), "{name}: subtitle streams");
    let codec = match kind {
        OutputKind::Mp4 => "mov_text",
        OutputKind::Mkv => "subrip",
    };
    for (&from, &to) in from.iter().zip(&to) {
        assert_eq!(output.stream(to, "codec_name"), Some(codec), "{name}");
        assert_eq!(output.stream_tag(to, "language"), Some("eng"), "{name}");
        assert_eq!(title(output, to).as_deref(), Some("Cues"), "{name}");
        assert_near(
            &format!("{name}: cues"),
            &cues(source, from),
            &cues(output, to),
            TWO_TICKS_NS,
        );
    }
}

fn assert_file_facts(name: &str, source: &Probe, output: &Probe) {
    let from = source.chapters();
    let to = output.chapters();
    let starts = |list: &[(i128, String)]| list.iter().map(|chapter| chapter.0).collect::<Vec<_>>();
    assert_eq!(from.len(), 2, "{name}: the fixture holds its chapters");
    assert_near(
        &format!("{name}: chapters"),
        &starts(&from),
        &starts(&to),
        ONE_TICK_NS,
    );
    let titles = |list: &[(i128, String)]| {
        list.iter()
            .map(|chapter| chapter.1.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(titles(&to), titles(&from), "{name}: chapter titles");
    for key in ["title", "comment"] {
        assert_eq!(output.format_tag(key), source.format_tag(key), "{name}");
        assert!(source.format_tag(key).is_some(), "{name}: fixture {key}");
    }
}

/// vidr's own checks agree with ffprobe.
fn assert_checks(remuxed: &Remuxed) {
    let name = &remuxed.fixture.name;
    let source = Input::open(&remuxed.fixture.path).unwrap();
    let output = Input::open(&remuxed.output).unwrap();
    let source_facts = read_stream_facts(&source).unwrap();
    let output_facts = read_stream_facts(&output).unwrap();
    let target = source_facts.geometry.display_dimensions();
    check_output(&source_facts, &output_facts, target)
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    let tracks = read_track_facts(&output).unwrap();
    assert_eq!(tracks.audio.len(), 1, "{name}");
    assert_eq!(tracks.subtitles.len(), 1, "{name}");
    assert_eq!(tracks.chapters, 2, "{name}");
    check_tracks(&remuxed.expected, &tracks).unwrap_or_else(|error| panic!("{name}: {error}"));
}

#[test]
fn every_stream_copies_for_each_codec_and_output_kind() {
    for name in FIXTURES {
        for kind in KINDS {
            let Some(remuxed) = remux("copies", name, kind) else {
                continue;
            };
            let what = format!("{name} to {kind}");
            let Some(output) = probe(&remuxed.output) else {
                return;
            };
            let source = probe(&remuxed.fixture.path).unwrap();
            assert_video(&what, &source, &output);
            assert_audio(&what, &source, &output);
            assert_subtitles(&what, kind, &source, &output);
            assert_file_facts(&what, &source, &output);
            assert_eq!(output.decode_log(), "", "{what}: ffmpeg logs errors");
            assert_checks(&remuxed);
            assert!(remuxed.dropped.is_empty(), "{what}");
        }
    }
}

#[test]
fn the_start_offset_is_an_empty_edit_in_mp4_and_a_pts_shift_in_matroska() {
    let Some(remuxed) = remux("offset_mp4", "remux_h264.mp4", OutputKind::Mp4) else {
        return;
    };
    let input = Input::open(&remuxed.output).unwrap();
    let track = input.video_track().unwrap();
    assert_eq!(track.edit_list.entries[0].media_time, -1);
    // The output keeps the source's movie timescale. The fixture's offset is
    // half a second.
    assert_eq!(
        track.edit_list.entries[0].segment_duration * 2,
        u64::from(track.edit_list.movie_timescale)
    );

    let Some(remuxed) = remux("offset_mkv", "remux_h264.mp4", OutputKind::Mkv) else {
        return;
    };
    let Some(output) = probe(&remuxed.output) else {
        return;
    };
    let video = output.streams("video")[0];
    assert_eq!(output.sorted_pts(video)[0], 500_000_000);
}

#[test]
fn mp4_output_starts_with_moov() {
    let Some(remuxed) = remux("faststart", "remux_h264.mkv", OutputKind::Mp4) else {
        return;
    };
    let bytes = std::fs::read(&remuxed.output).unwrap();
    let ftyp_end = usize::try_from(u32::from_be_bytes(bytes[..4].try_into().unwrap())).unwrap();
    assert_eq!(&bytes[ftyp_end + 4..ftyp_end + 8], b"moov");
}

#[test]
fn hevc_output_is_hvc1() {
    let Some(remuxed) = remux("hvc1", "remux_hevc.mkv", OutputKind::Mp4) else {
        return;
    };
    let input = Input::open(&remuxed.output).unwrap();
    let codec = input.video_track().unwrap().codec().unwrap();
    assert_eq!(codec.to_string(), "hvc1");
}

#[test]
fn matroska_to_matroska_copies_audio_blocks_exactly() {
    let Some(remuxed) = remux("exact", "remux_h264.mkv", OutputKind::Mkv) else {
        return;
    };
    assert_eq!(remuxed.max_rounding_error_ns, None);
    let Some(output) = probe(&remuxed.output) else {
        return;
    };
    let source = probe(&remuxed.fixture.path).unwrap();
    let from = source.streams("audio")[0];
    let to = output.streams("audio")[0];
    assert_eq!(output.packets(to), source.packets(from));
}
