//! Back-pressure: a blocked sink stops the stages above it, at a stated bound.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use vidr::pipeline::{MAX_IN_FLIGHT_FRAMES, PumpControl};

use super::fakes::{Gate, Probe, Script, launch, millis, pump, steady};

/// Polls `reached` for a state the pump reaches on its own, up to a limit.
pub(super) fn wait_for(reached: impl Fn() -> bool, what: &str) {
    let start = Instant::now();
    while !reached() {
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "the pump hangs: waiting for {what}"
        );
        thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn a_blocked_sink_holds_the_bounded_number_of_frames() {
    let control = Arc::new(PumpControl::new());
    let gate = Arc::new(Gate::default());
    let packets = 100;
    let script = Script {
        packets: steady(packets),
        sink_gate: Some(Arc::clone(&gate)),
        ..Script::default()
    };
    let (probe, result) = launch(script, millis(), &control);

    let bound = MAX_IN_FLIGHT_FRAMES as u64;
    wait_for(
        || Probe::read(&probe.emitted) >= bound,
        "the pipeline to fill",
    );
    // Any further progress happens in this window, and the asserts then see it.
    thread::sleep(Duration::from_millis(200));
    assert_eq!(Probe::read(&probe.emitted), bound, "T1 keeps decoding");
    assert_eq!(Probe::read(&probe.reads), bound, "T1 keeps parsing");
    // Frame 0 waits in the sink, 1 in the channel, and T2 blocks on 2.
    assert_eq!(Probe::read(&probe.applied), 3, "T2 keeps queuing work");
    assert_eq!(Probe::read(&probe.sent), 1, "T3 takes a second frame");
    assert_eq!(Probe::read(&probe.written), 0);

    gate.open();
    let report = result.wait().unwrap();
    assert_eq!(report.packets, packets as u64);
    assert!(Probe::read(&probe.high_water) <= bound);
}

#[test]
fn a_slow_sink_never_passes_the_bound() {
    let packets = 60;
    let script = Script {
        packets: steady(packets),
        sink_delay: Duration::from_millis(2),
        ..Script::default()
    };
    let (result, probe) = pump(script, millis());
    assert_eq!(result.unwrap().packets, packets as u64);
    let high_water = Probe::read(&probe.high_water);
    assert!(
        high_water <= MAX_IN_FLIGHT_FRAMES as u64,
        "{high_water} frames in flight"
    );
}

#[test]
fn a_decoder_queue_drains_through_the_same_bound() {
    let packets = 40;
    let script = Script {
        packets: steady(packets),
        // The whole stream reaches the decoder's queue at the flush.
        reorder: packets as usize,
        sink_delay: Duration::from_millis(1),
        ..Script::default()
    };
    let (result, probe) = pump(script, millis());
    assert_eq!(result.unwrap().packets, packets as u64);
    assert!(Probe::read(&probe.high_water) <= MAX_IN_FLIGHT_FRAMES as u64);
}
