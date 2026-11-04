//! Errors, panics, stop requests, and refused runs: each returns its named
//! result and joins every thread.

use std::sync::Arc;

use vidr::pipeline::{PipelineError, PumpControl};
use vidr::{Error, TimingError};

use super::fakes::{Fault, Gate, Point, Probe, Script, launch, millis, pump, steady};

/// Every fake dropped, so every stage ended before `run` returned.
fn assert_joined(probe: &Probe, name: &str) {
    assert_eq!(
        Probe::read(&probe.drops),
        5,
        "{name}: a fake outlives `run`"
    );
}

fn assert_not_final(probe: &Probe, name: &str) {
    assert_eq!(
        Probe::read(&probe.sink_finished),
        0,
        "{name}: the sink finished"
    );
}

fn faulty(point: Point, fault: Fault) -> Script {
    Script {
        packets: steady(20),
        fault: Some((point, fault)),
        ..Script::default()
    }
}

#[test]
fn an_error_at_each_stage_returns_that_error_and_joins() {
    for point in Point::ALL {
        let name = format!("{point:?}");
        let (result, probe) = pump(faulty(point, Fault::Error), millis());
        assert_eq!(result, Err(point.error()), "{name}");
        assert_joined(&probe, &name);
        assert_not_final(&probe, &name);
    }
}

#[test]
fn a_panic_at_each_stage_returns_a_named_error_and_joins() {
    for point in Point::ALL {
        let name = format!("{point:?}");
        let (result, probe) = pump(faulty(point, Fault::Panic), millis());
        assert_eq!(
            result,
            Err(PipelineError::Panic {
                stage: point.stage(),
                message: format!("injected panic at {point:?}"),
            }),
            "{name}"
        );
        assert_joined(&probe, &name);
        assert_not_final(&probe, &name);
    }
}

#[test]
fn a_panic_with_no_text_is_named_too() {
    let (result, probe) = pump(faulty(Point::Transform, Fault::PanicWithNoText), millis());
    let Err(PipelineError::Panic { message, .. }) = result else {
        panic!("{result:?}");
    };
    assert_eq!(message, "a payload that is not text");
    assert_joined(&probe, "no text");
}

#[test]
fn a_downstream_error_stops_upstream_with_no_flush() {
    let (result, probe) = pump(faulty(Point::SinkWrite, Fault::Error), millis());
    assert_eq!(result, Err(Point::SinkWrite.error()));
    assert_eq!(Probe::read(&probe.encoder_flushed), 0);
    // Back-pressure keeps T1 short of the end of the stream, so it never flushes.
    assert_eq!(Probe::read(&probe.decoder_flushed), 0);
}

#[test]
fn an_upstream_error_never_flushes_the_encoder() {
    for point in [Point::Source, Point::DecoderSend, Point::Transform] {
        let (result, probe) = pump(faulty(point, Fault::Error), millis());
        assert_eq!(result, Err(point.error()), "{point:?}");
        assert_eq!(Probe::read(&probe.encoder_flushed), 0, "{point:?}");
    }
}

#[test]
fn the_first_error_sets_the_stop_flag() {
    let control = Arc::new(PumpControl::new());
    let (_, result) = launch(faulty(Point::Transform, Fault::Error), millis(), &control);
    assert_eq!(result.wait(), Err(Point::Transform.error()));
    assert!(control.stop_requested());
}

#[test]
fn a_stop_before_the_run_ends_it_with_no_work() {
    let control = Arc::new(PumpControl::new());
    control.request_stop();
    let script = Script {
        packets: steady(20),
        ..Script::default()
    };
    let (probe, result) = launch(script, millis(), &control);
    let result = result.wait();
    assert_eq!(
        result,
        Err(PipelineError::Stopped {
            frames_in: 0,
            packets: 0,
        })
    );
    assert!(result.is_err_and(|error| error.is_stop()));
    assert_eq!(Probe::read(&probe.reads), 0);
    assert_joined(&probe, "stop before");
}

#[test]
fn a_stop_during_the_run_flushes_nothing_as_final() {
    let control = Arc::new(PumpControl::new());
    let script = Script {
        packets: steady(1000),
        stop_at_write: Some((Arc::clone(&control), 3)),
        ..Script::default()
    };
    let (probe, result) = launch(script, millis(), &control);
    let result = result.wait();
    let Err(PipelineError::Stopped { frames_in, packets }) = result else {
        panic!("{result:?}");
    };
    // The write in progress completes, and T3 sees the flag after it.
    assert_eq!(packets, 4);
    assert!(frames_in < 1000, "{frames_in} frames read after the stop");
    assert_eq!(Probe::read(&probe.decoder_flushed), 0);
    assert_eq!(Probe::read(&probe.encoder_flushed), 0);
    assert_not_final(&probe, "stop during");
    assert_joined(&probe, "stop during");
}

#[test]
fn a_stop_while_every_stage_is_blocked_ends_the_run() {
    let control = Arc::new(PumpControl::new());
    let gate = Arc::new(Gate::default());
    let script = Script {
        packets: steady(1000),
        sink_gate: Some(Arc::clone(&gate)),
        ..Script::default()
    };
    let (probe, result) = launch(script, millis(), &control);
    super::pressure::wait_for(|| Probe::read(&probe.emitted) >= 7, "the pipeline to fill");
    control.request_stop();
    gate.open();
    assert!(result.wait().is_err_and(|error| error.is_stop()));
    assert_not_final(&probe, "stop blocked");
    assert_joined(&probe, "stop blocked");
}

#[test]
fn an_empty_source_is_a_named_error() {
    let (result, probe) = pump(Script::default(), millis());
    assert_eq!(result, Err(PipelineError::NoFrames { packets_read: 0 }));
    assert_eq!(Probe::read(&probe.sink_started), 0);
    assert_eq!(Probe::read(&probe.encoder_flushed), 0);
    assert_not_final(&probe, "empty");
    assert_joined(&probe, "empty");
}

#[test]
fn a_decoder_that_returns_nothing_is_a_named_error() {
    let script = Script {
        packets: steady(3),
        decoder_returns_nothing: true,
        ..Script::default()
    };
    let (result, probe) = pump(script, millis());
    assert_eq!(result, Err(PipelineError::NoFrames { packets_read: 3 }));
    assert_eq!(Probe::read(&probe.decoder_flushed), 1);
    assert_joined(&probe, "nothing");
}

#[test]
fn a_count_mismatch_is_an_error_and_never_finishes() {
    let script = Script {
        packets: steady(5),
        encoder_loses: Some(2),
        ..Script::default()
    };
    let (result, probe) = pump(script, millis());
    assert_eq!(
        result,
        Err(PipelineError::CountMismatch {
            frames_in: 5,
            frames_dropped: 0,
            frames_out: 5,
            packets: 4,
        })
    );
    assert_not_final(&probe, "mismatch");
    assert_joined(&probe, "mismatch");
}

#[test]
fn a_packet_with_no_config_record_is_an_error() {
    let script = Script {
        packets: steady(2),
        no_config_record: true,
        ..Script::default()
    };
    let (result, probe) = pump(script, millis());
    assert_eq!(result, Err(PipelineError::NoConfigRecord { pts: 0 }));
    assert_eq!(Probe::read(&probe.sink_started), 0);
    assert_joined(&probe, "no record");
}

#[test]
fn frames_out_of_pts_order_are_a_timing_error() {
    let script = Script {
        packets: vec![(0, 10), (10, 10), (10, 10)],
        ..Script::default()
    };
    let (result, probe) = pump(script, millis());
    assert_eq!(
        result,
        Err(PipelineError::from(TimingError::NonIncreasingPts {
            index: 2,
            previous: 10,
            pts: 10,
        }))
    );
    assert_not_final(&probe, "order");
    assert_joined(&probe, "order");
}

#[test]
fn the_crate_error_holds_a_pump_error_and_its_message_names_the_next_action() {
    let stopped = PipelineError::Stopped {
        frames_in: 12,
        packets: 9,
    };
    let message = stopped.to_string();
    assert!(message.contains("12 decoded frame(s)"), "{message}");
    assert!(message.contains("Remove the output file"), "{message}");
    assert!(!message.contains(';'), "{message}");
    assert_eq!(Error::from(stopped.clone()).to_string(), message);
}
