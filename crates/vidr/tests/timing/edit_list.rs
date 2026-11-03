use vidr::{
    DurationHints, EditEntry, EditListError, EditTimeline, FrameTiming, Rational, StartOffset,
    TrackTiming,
};

use super::r;

/// Frames every 512 ticks of a 1/12800 s media clock (25 fps).
fn frames(count: i64) -> Vec<FrameTiming> {
    (0..count)
        .map(|index| FrameTiming::new(index * 512, Some(index * 512 - 1024), 512))
        .collect()
}

#[test]
fn empty_edit_gives_a_start_offset() {
    // 500 ticks of a 1/1000 s movie clock: the track starts 0.5 s in.
    let entries = [EditEntry::empty(500), EditEntry::media(0, 0)];
    let timeline = EditTimeline::resolve(&entries, r(1, 1000), r(1, 12800)).unwrap();
    assert_eq!(timeline.start_offset(), r(1, 2));
    assert_eq!(timeline.media_start(), 0);
    assert_eq!(timeline.media_end(), None);
    let presented = timeline.present(&frames(4)).unwrap();
    assert_eq!(presented, frames(4));
}

#[test]
fn media_edit_trims_the_start_and_end() {
    // Skip the first 1024 media ticks (B-frame pre-roll), then play 80 ms.
    let entries = [EditEntry::media(80, 1024)];
    let timeline = EditTimeline::resolve(&entries, r(1, 1000), r(1, 12800)).unwrap();
    assert_eq!(timeline.media_end(), Some(1024 + 1024));
    let presented = timeline.present(&frames(6)).unwrap();
    let pts: Vec<_> = presented.iter().map(|frame| frame.pts).collect();
    assert_eq!(pts, [0, 512]);
    assert_eq!(presented[0].dts, Some(-1024));
    let track = TrackTiming::new(
        r(1, 12800),
        StartOffset {
            edit: timeline.start_offset(),
            container_start: Rational::ZERO,
        },
        presented,
        DurationHints::default(),
    )
    .unwrap();
    assert_eq!(track.total_duration(), r(2, 25));
}

#[test]
fn trim_inside_a_frame_keeps_that_frame_on_screen() {
    let entries = [EditEntry::empty(100), EditEntry::media(0, 700)];
    let timeline = EditTimeline::resolve(&entries, r(1, 1000), r(1, 12800)).unwrap();
    let presented = timeline.present(&frames(3)).unwrap();
    assert_eq!(
        presented[0],
        FrameTiming::new(0, Some(512 - 1024 - 700), 512 - 188)
    );
    assert_eq!(presented[1].pts, 1024 - 700);
    assert_eq!(timeline.start_offset(), r(1, 10));
}

#[test]
fn rate_other_than_one_is_refused() {
    let slow = EditEntry {
        segment_duration: 1000,
        media_time: 0,
        rate: r(1, 2),
    };
    assert_eq!(
        EditTimeline::resolve(&[EditEntry::empty(10), slow], r(1, 1000), r(1, 12800)),
        Err(EditListError::UnsupportedRate {
            index: 1,
            rate: r(1, 2),
        })
    );
    let dwell = EditEntry {
        rate: Rational::ZERO,
        ..slow
    };
    assert!(matches!(
        EditTimeline::resolve(&[dwell], r(1, 1000), r(1, 12800)),
        Err(EditListError::UnsupportedRate { index: 0, .. })
    ));
}

#[test]
fn second_media_edit_is_refused() {
    let entries = [EditEntry::media(40, 0), EditEntry::media(40, 1024)];
    assert_eq!(
        EditTimeline::resolve(&entries, r(1, 1000), r(1, 12800)),
        Err(EditListError::SegmentAfterMedia {
            index: 1,
            media_index: 0,
        })
    );
}

#[test]
fn unordered_frames_are_refused() {
    let mut list = frames(3);
    list.swap(1, 2);
    assert!(matches!(
        EditTimeline::IDENTITY.present(&list),
        Err(EditListError::Timing(_))
    ));
}
