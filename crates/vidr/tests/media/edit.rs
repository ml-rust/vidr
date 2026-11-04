//! Edit lists against the pts ffprobe reports with the edit list applied.

use vidr::media::{Input, Track};
use vidr::{EditTimeline, FrameTiming, Rational};

use super::fixtures::{Fixture, open, video};

/// The track's raw frames in presentation order, in track ticks.
fn raw_frames(input: &Input, track: &Track) -> Vec<FrameTiming> {
    let packets: Vec<containr::Result<containr::Packet>> = match input {
        #[cfg(feature = "mp4")]
        Input::Mp4 { demuxer, .. } => demuxer.track_packets(track.id).unwrap().collect(),
        #[cfg(feature = "mkv")]
        Input::Mkv { demuxer, .. } => demuxer.track_packets(track.id).unwrap().collect(),
        _ => panic!("vidr has a container this test does not name"),
    };
    let mut frames: Vec<_> = packets
        .into_iter()
        .map(|packet| {
            let packet = packet.unwrap();
            let duration = packet
                .duration
                .map_or(0, |ticks| i64::try_from(ticks).unwrap());
            FrameTiming::new(packet.pts, packet.dts, duration)
        })
        .collect();
    frames.sort_by_key(|frame| frame.pts);
    frames
}

fn time_base(fixture: &Fixture) -> Rational {
    let (num, den) = fixture.meta("time_base").split_once('/').unwrap();
    Rational::new(num.parse().unwrap(), den.parse().unwrap()).unwrap()
}

fn edited_pts(fixture: &Fixture) -> Vec<i64> {
    fixture
        .meta("edited_pts")
        .split(',')
        .map(|pts| pts.parse().unwrap())
        .collect()
}

/// The timeline applied to the raw frames, plus its start offset, gives the
/// pts ffprobe reports.
fn assert_matches_ffprobe(fixture: &Fixture, input: &Input, track: &Track) -> EditTimeline {
    let name = &fixture.name;
    let base = time_base(fixture);
    let track_base =
        Rational::new(i64::from(track.timebase.num), u64::from(track.timebase.den)).unwrap();
    assert_eq!(track_base, base, "{name}: time base");

    let timeline = input.edit_timeline(track).unwrap();
    let presented = timeline.present(&raw_frames(input, track)).unwrap();
    let expected = edited_pts(fixture);
    assert_eq!(presented.len(), expected.len(), "{name}: frame count");
    for (frame, expected) in presented.iter().zip(expected) {
        let shift = Rational::from_ticks(expected - frame.pts, base).unwrap();
        assert_eq!(shift, timeline.start_offset(), "{name}: pts {expected}");
    }
    timeline
}

#[test]
fn an_empty_edit_gives_the_start_offset_ffprobe_reports() {
    let Some((fixture, input)) = open("edit_offset.mp4") else {
        return;
    };
    let track = video(&fixture, &input);
    assert_eq!(track.edit_list.entries.len(), 2);
    let timeline = assert_matches_ffprobe(&fixture, &input, &track);
    assert_eq!(timeline.start_offset(), Rational::new(1, 2).unwrap());
    // The recipe's B-frames delay the first pts, and the media edit cuts it.
    assert_eq!(
        timeline.media_start(),
        track.edit_list.entries[1].media_time
    );
    assert!(timeline.media_start() > 0);
}

#[test]
fn a_media_edit_moves_the_first_pts_to_zero() {
    let Some((fixture, input)) = open("plain.mp4") else {
        return;
    };
    let track = video(&fixture, &input);
    let timeline = assert_matches_ffprobe(&fixture, &input, &track);
    assert_eq!(timeline.start_offset(), Rational::ZERO);
    assert!(timeline.media_start() > 0);
}

#[test]
fn matroska_has_no_edit_list() {
    let Some((fixture, input)) = open("plain.mkv") else {
        return;
    };
    let track = video(&fixture, &input);
    let timeline = assert_matches_ffprobe(&fixture, &input, &track);
    assert_eq!(timeline, EditTimeline::IDENTITY);
}
