use vidr::{
    DurationHints, FrameTiming, LastDurationSource, Rational, Rounding, StartOffset, TimingError,
    TrackTiming,
};

use super::r;

/// An NTSC track in a 90 kHz clock: one frame every 3003 ticks.
fn ntsc(count: i64, last_duration: i64, hints: DurationHints) -> Result<TrackTiming, TimingError> {
    let frames = (0..count)
        .map(|index| {
            let duration = if index + 1 == count {
                last_duration
            } else {
                3003
            };
            FrameTiming::new(1000 + index * 3003, None, duration)
        })
        .collect();
    TrackTiming::new(r(1, 90000), StartOffset::default(), frames, hints)
}

#[test]
fn chain_uses_frame_duration_first() {
    let hints = DurationHints {
        packet_duration: Some(1),
        average_frame_rate: Some(r(1, 1)),
        base_frame_rate: None,
    };
    let track = ntsc(3, 3003, hints).unwrap();
    assert_eq!(track.last_duration_source(), LastDurationSource::Frame);
    assert_eq!(track.total_duration(), r(3 * 3003, 90000));
}

#[test]
fn chain_falls_back_to_packet_duration() {
    let hints = DurationHints {
        packet_duration: Some(3003),
        average_frame_rate: Some(r(1, 1)),
        base_frame_rate: None,
    };
    let track = ntsc(3, 0, hints).unwrap();
    assert_eq!(track.last_duration_source(), LastDurationSource::Packet);
    assert_eq!(track.last_duration(), r(3003, 90000));
}

#[test]
fn chain_falls_back_to_average_rate() {
    let hints = DurationHints {
        packet_duration: None,
        average_frame_rate: Some(r(30000, 1001)),
        base_frame_rate: Some(r(60, 1)),
    };
    let track = ntsc(3, 0, hints).unwrap();
    assert_eq!(
        track.last_duration_source(),
        LastDurationSource::FrameRate(r(30000, 1001))
    );
    assert_eq!(track.last_duration_ticks(Rounding::Exact), Ok(3003));
}

#[test]
fn chain_falls_back_to_base_rate() {
    let hints = DurationHints {
        packet_duration: Some(-5),
        average_frame_rate: Some(r(-1, 1)),
        base_frame_rate: Some(r(60, 1)),
    };
    let track = ntsc(3, 0, hints).unwrap();
    assert_eq!(
        track.last_duration_source(),
        LastDurationSource::FrameRate(r(60, 1))
    );
    assert_eq!(track.last_duration(), r(1, 60));
}

#[test]
fn chain_falls_back_to_previous_gap() {
    let track = ntsc(3, 0, DurationHints::default()).unwrap();
    assert_eq!(
        track.last_duration_source(),
        LastDurationSource::PreviousGap
    );
    assert_eq!(track.total_duration(), r(3 * 3003, 90000));
}

#[test]
fn chain_error_names_every_missing_input() {
    let hints = DurationHints {
        packet_duration: Some(0),
        average_frame_rate: Some(Rational::ZERO),
        base_frame_rate: None,
    };
    assert_eq!(
        ntsc(1, -1, hints),
        Err(TimingError::NoLastFrameDuration {
            frame_duration: -1,
            packet_duration: Some(0),
            average_frame_rate: Some(Rational::ZERO),
            base_frame_rate: None,
            frame_count: 1,
        })
    );
}

#[test]
fn pts_must_strictly_increase() {
    let frames = vec![
        FrameTiming::new(0, None, 1),
        FrameTiming::new(5, None, 1),
        FrameTiming::new(3, None, 1),
    ];
    assert_eq!(
        TrackTiming::new(
            r(1, 30),
            StartOffset::default(),
            frames,
            DurationHints::default()
        ),
        Err(TimingError::NonIncreasingPts {
            index: 2,
            previous: 5,
            pts: 3,
        })
    );
}

#[test]
fn total_duration_ignores_the_first_pts_offset() {
    let track = ntsc(2, 3003, DurationHints::default()).unwrap();
    assert_eq!(track.first_pts(), 1000);
    assert_eq!(track.last_pts(), 4003);
    assert_eq!(track.total_duration(), r(6006, 90000));
}

#[test]
fn start_offset_gives_the_poc_video_offset() {
    let start = StartOffset {
        edit: r(1, 1),
        container_start: r(1, 90),
    };
    let frames = vec![FrameTiming::new(1000, Some(0), 3003)];
    let track = TrackTiming::new(r(1, 90000), start, frames, DurationHints::default()).unwrap();
    // 1 s + 1000/90000 s - 1/90 s = 1 s.
    assert_eq!(track.video_offset(), Ok(Rational::ONE));
}
