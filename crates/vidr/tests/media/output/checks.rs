//! The facts readers and `check_tracks` over real files.

use vidr::check::{TrackFacts, TrackPacketCount, check_tracks};
use vidr::media::{Input, OutputKind, plan_copy, read_stream_facts, read_track_facts};
use vidr::{Deviation, Rational, TrackClass};

use crate::media::fixtures::{Fixture, open};

fn meta_pts(fixture: &Fixture) -> Vec<i64> {
    fixture
        .meta("edited_pts")
        .split(',')
        .map(|pts| pts.parse().unwrap())
        .collect()
}

#[test]
fn stream_facts_apply_the_edit_list() {
    let Some((fixture, input)) = open("edit_offset.mp4") else {
        return;
    };
    let facts = read_stream_facts(&input).unwrap();
    let expected = meta_pts(&fixture);
    assert_eq!(facts.frame_count(), expected.len());
    // ffprobe adds the 0.5 s offset, and `StreamFacts` keeps pts relative.
    let origin = expected[0] - facts.frames[0].pts;
    for (frame, pts) in facts.frames.iter().zip(expected) {
        assert_eq!(frame.pts + origin, pts);
    }
    assert_eq!(facts.time_base, Rational::new(1, 12800).unwrap());
    assert_eq!(
        i64::from(facts.geometry.coded.width()),
        fixture.number("width")
    );
}

#[test]
fn stream_facts_read_matroska_geometry() {
    let Some((_, input)) = open("anamorphic.mkv") else {
        return;
    };
    let facts = read_stream_facts(&input).unwrap();
    assert_eq!(
        facts.geometry.sample_aspect_ratio,
        vidr::SampleAspectRatio::new(8, 9)
    );
    assert_eq!(facts.time_base, Rational::new(1, 1000).unwrap());
}

#[test]
fn track_facts_count_one_set_of_chapters_and_no_chapter_track() {
    for name in ["remux_h264.mp4", "remux_h264.mkv"] {
        let Some((_, input)) = open(name) else {
            continue;
        };
        let facts = read_track_facts(&input).unwrap();
        assert_eq!(facts.chapters, 2, "{name}");
        assert_eq!(facts.audio.len(), 1, "{name}");
        assert_eq!(facts.subtitles.len(), 1, "{name}");
        assert_eq!(facts.subtitles[0].packets, 2, "{name}: two cues");
    }
}

#[test]
fn a_plan_expects_what_the_source_holds_short_of_dropped_tracks() {
    let Some((_, mut input)) = open("second_default.mp4") else {
        return;
    };
    let plan = plan_copy(&mut input, OutputKind::Mp4).unwrap();
    assert_eq!(plan.expected_tracks(), TrackFacts::default());
    let facts = read_track_facts(&Input::open(input.path()).unwrap()).unwrap();
    assert_eq!(check_tracks(&plan.expected_tracks(), &facts), Ok(()));
}

#[test]
fn a_file_with_a_track_missing_fails_the_check() {
    let Some((_, input)) = open("plain.mp4") else {
        return;
    };
    let output = read_track_facts(&input).unwrap();
    let expected = TrackFacts {
        audio: vec![TrackPacketCount {
            track_id: 2,
            packets: 94,
        }],
        ..TrackFacts::default()
    };
    let error = check_tracks(&expected, &output).unwrap_err();
    assert_eq!(
        error.deviations(),
        [Deviation::TrackCount {
            class: TrackClass::Audio,
            source: 1,
            output: 0,
        }]
        .as_slice()
    );
}
