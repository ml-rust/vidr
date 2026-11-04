//! Order, pts, durations, and the edit window through the whole pump.

use vidr::pipeline::{PipelineError, PumpReport, PumpTiming};
use vidr::time::{EditEntry, EditTimeline};
use vidr::{
    Dimensions, DurationHints, EditListError, FrameTiming, Geometry, Rational, Rotation,
    SampleAspectRatio, StreamFacts, TimingError, check_output,
};

use super::fakes::{Probe, Script, millis, pump, steady};

fn script(packets: Vec<(i64, i64)>) -> Script {
    Script {
        packets,
        ..Script::default()
    }
}

/// The encoder's `(pts, duration)` list from a run that must pass.
fn encoded(script: Script, timing: PumpTiming) -> Vec<(i64, i64)> {
    let (result, probe) = pump(script, timing);
    result.unwrap();
    probe.encoded()
}

/// A decoder that returns two frames at one pts cannot keep the output's timing, so
/// the run ends in a named error. NVDEC's parser has no rule against this.
#[test]
fn two_frames_at_one_pts_are_a_named_error() {
    let script = Script {
        decoder_returns_two: true,
        ..script(steady(3))
    };
    let (result, probe) = pump(script, millis());
    assert_eq!(
        result.unwrap_err(),
        PipelineError::from(TimingError::NonIncreasingPts {
            index: 1,
            previous: 0,
            pts: 0,
        })
    );
    assert_eq!(Probe::read(&probe.sink_finished), 0);
}

#[test]
fn order_and_pts_survive_decoder_reordering() {
    // Decode order of a stream with B-frames.
    let packets = vec![
        (0, 1001),
        (3003, 1001),
        (1001, 1001),
        (2002, 1001),
        (6006, 1001),
        (4004, 1001),
        (5005, 1001),
    ];
    let script = Script {
        packets,
        reorder: 2,
        ..Script::default()
    };
    let (result, probe) = pump(script, millis());
    let expected: Vec<i64> = (0..7).map(|index| index * 1001).collect();
    assert_eq!(
        result,
        Ok(PumpReport {
            frames_in: 7,
            frames_dropped: 0,
            frames_out: 7,
            packets: 7,
            host_copied_frames: 0,
            host_copied_bytes: 0,
            output: 7,
        })
    );
    assert_eq!(probe.written_pts(), expected);
    assert_eq!(probe.applied_pts(), expected);
    assert_eq!(
        probe.encoded(),
        expected
            .iter()
            .map(|&pts| (pts, 1001))
            .collect::<Vec<(i64, i64)>>()
    );
    assert_eq!(Probe::read(&probe.decoder_flushed), 1);
    assert_eq!(Probe::read(&probe.encoder_flushed), 1);
    assert_eq!(Probe::read(&probe.sink_started), 1);
    assert_eq!(Probe::read(&probe.sink_finished), 1);
    assert_eq!(Probe::read(&probe.drops), 5);
}

#[test]
fn each_duration_is_the_next_pts_minus_its_own() {
    // Every packet states 7 ticks, and the gaps win but for the last frame.
    let packets = vec![(5, 7), (15, 7), (40, 7), (41, 7)];
    assert_eq!(
        encoded(script(packets), millis()),
        vec![(5, 10), (15, 25), (40, 1), (41, 7)]
    );
}

#[test]
fn a_single_frame_keeps_its_own_duration() {
    assert_eq!(encoded(script(vec![(9, 33)]), millis()), vec![(9, 33)]);
}

#[test]
fn the_last_duration_takes_the_chain_in_order() {
    let rate = Rational::new(50, 1).unwrap();
    let with_rate = PumpTiming {
        hints: DurationHints {
            average_frame_rate: Some(rate),
            ..DurationHints::default()
        },
        ..millis()
    };
    // 1. The last frame's own packet, though another packet decodes last.
    let own = Script {
        packets: vec![(0, 3), (30, 9), (20, 4)],
        reorder: 1,
        ..Script::default()
    };
    assert_eq!(encoded(own, with_rate).last(), Some(&(30, 9)));
    // 2. The last packet in decode order, when the last frame's own has none.
    let packet = Script {
        packets: vec![(0, 3), (30, 0), (20, 4)],
        reorder: 1,
        ..Script::default()
    };
    assert_eq!(encoded(packet, with_rate).last(), Some(&(30, 4)));
    // 3. One frame period, when no packet has a duration: 1/50 s is 20 ticks.
    let period = script(vec![(0, 0), (10, 0)]);
    assert_eq!(encoded(period, with_rate).last(), Some(&(10, 20)));
    // 4. The gap between the last two pts.
    let gap = script(vec![(0, 0), (10, 0)]);
    assert_eq!(encoded(gap, millis()).last(), Some(&(10, 10)));
}

#[test]
fn a_rate_duration_rounds_half_even_to_a_tick() {
    let timing = PumpTiming {
        hints: DurationHints {
            average_frame_rate: Some(Rational::new(30000, 1001).unwrap()),
            ..DurationHints::default()
        },
        ..PumpTiming::new(Rational::new(1, 24000).unwrap())
    };
    // 1001/30000 s is 800.8 ticks.
    assert_eq!(
        encoded(script(vec![(0, 0)]), timing).last(),
        Some(&(0, 801))
    );
}

#[test]
fn a_last_frame_with_no_duration_source_is_an_error() {
    let (result, probe) = pump(script(vec![(0, 0)]), millis());
    assert_eq!(
        result,
        Err(PipelineError::from(TimingError::NoLastFrameDuration {
            frame_duration: 0,
            packet_duration: None,
            average_frame_rate: None,
            base_frame_rate: None,
            frame_count: 1,
        }))
    );
    assert_eq!(Probe::read(&probe.sink_finished), 0);
}

#[test]
fn a_last_duration_below_a_tick_is_an_error() {
    let rate = Rational::new(1000, 1).unwrap();
    let time_base = Rational::new(1, 1).unwrap();
    let timing = PumpTiming {
        hints: DurationHints {
            average_frame_rate: Some(rate),
            ..DurationHints::default()
        },
        ..PumpTiming::new(time_base)
    };
    let (result, _) = pump(script(vec![(4, 0)]), timing);
    assert_eq!(
        result,
        Err(PipelineError::LastDurationBelowTick {
            pts: 4,
            duration: Rational::new(1, 1000).unwrap(),
            time_base,
        })
    );
}

#[test]
fn no_edit_list_keeps_stored_pts() {
    let packets = vec![(-20, 10), (-10, 10), (0, 10)];
    assert_eq!(
        encoded(script(packets), millis()),
        vec![(-20, 10), (-10, 10), (0, 10)]
    );
}

fn timeline(segment: i64, media_time: i64) -> EditTimeline {
    let base = Rational::new(1, 1000).unwrap();
    EditTimeline::resolve(&[EditEntry::media(segment, media_time)], base, base).unwrap()
}

fn windowed(window: EditTimeline) -> PumpTiming {
    PumpTiming {
        timeline: Some(window),
        ..millis()
    }
}

#[test]
fn frames_outside_the_edit_window_cost_no_work() {
    // The window plays 25..65, so 20 is on screen at its start and 70 on is out.
    let window = timeline(40, 25);
    let (result, probe) = pump(script(steady(10)), windowed(window));
    let report = result.unwrap();
    assert_eq!(report.frames_in, 10);
    assert_eq!(report.frames_dropped, 5);
    assert_eq!(report.frames_out, 5);
    assert_eq!(report.packets, 5);
    assert_eq!(probe.applied_pts(), vec![20, 30, 40, 50, 60]);
    // The frame at 60 lasts until the window ends at 65, and 20 starts at 25.
    assert_eq!(
        probe.encoded(),
        vec![(0, 5), (5, 10), (15, 10), (25, 10), (35, 5)]
    );
}

#[test]
fn the_window_matches_present_and_passes_check_output() {
    let base = Rational::new(1, 1000).unwrap();
    let window = timeline(40, 25);
    let frames: Vec<FrameTiming> = steady(10)
        .into_iter()
        .map(|(pts, duration)| FrameTiming::new(pts, None, duration))
        .collect();
    let presented = window.present(&frames).unwrap();
    let encoded = encoded(script(steady(10)), windowed(window));
    let pts: Vec<i64> = encoded.iter().map(|pair| pair.0).collect();
    let expected: Vec<i64> = presented.iter().map(|frame| frame.pts).collect();
    assert_eq!(pts, expected);

    let size = Dimensions::new(2, 2).unwrap();
    let geometry = Geometry::new(size, Rotation::Upright, SampleAspectRatio::SQUARE);
    let source = StreamFacts::new(geometry, base, presented, DurationHints::default());
    let output = StreamFacts::new(
        geometry,
        base,
        encoded
            .iter()
            .map(|&(pts, duration)| FrameTiming::new(pts, None, duration))
            .collect(),
        DurationHints::default(),
    );
    assert_eq!(check_output(&source, &output, size), Ok(()));
}

#[test]
fn a_window_with_no_frame_is_an_error() {
    let window = timeline(5, 1000);
    let (result, probe) = pump(script(steady(3)), windowed(window));
    assert_eq!(
        result,
        Err(PipelineError::from(EditListError::NoPresentedFrames {
            media_start: 1000,
            media_end: Some(1005),
            frames: 3,
        }))
    );
    assert_eq!(probe.applied_pts(), Vec::<i64>::new());
    assert_eq!(Probe::read(&probe.sink_finished), 0);
}
