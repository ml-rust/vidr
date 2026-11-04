//! MP4 to Matroska rounds to a 1 ms `TimestampScale`, and says so.

use vidr::check::check_output;
use vidr::media::{Input, OutputKind, read_stream_facts};

use super::probe::{assert_near, probe};
use super::support::remux;
use crate::media::fixtures::open;
use vidr::media::plan_copy;

/// Half a 1 ms tick, in nanoseconds.
const HALF_A_TICK_NS: u64 = 500_000;

#[test]
fn aac_at_48_khz_reports_an_error_of_at_most_half_a_tick() {
    let Some((_, mut input)) = open("remux_h264.mp4") else {
        return;
    };
    let plan = plan_copy(&mut input, OutputKind::Mkv).unwrap();
    let error = plan.max_rounding_error_ns().unwrap();
    assert!(error > 0, "1024 ticks at 48 kHz fall between milliseconds");
    assert!(error <= HALF_A_TICK_NS, "{error} ns");
}

#[test]
fn an_exact_output_reports_no_rounding() {
    for (name, kind) in [
        ("remux_h264.mp4", OutputKind::Mp4),
        ("remux_h264.mkv", OutputKind::Mkv),
        ("remux_h264.mkv", OutputKind::Mp4),
    ] {
        if let Some((_, mut input)) = open(name) {
            let plan = plan_copy(&mut input, kind).unwrap();
            assert_eq!(plan.max_rounding_error_ns(), None, "{name} to {kind}");
        }
    }
}

#[test]
fn a_clock_with_no_whole_milliseconds_passes_check_output_in_matroska() {
    let Some(remuxed) = remux("ntsc", "remux_ntsc.mp4", OutputKind::Mkv) else {
        return;
    };
    let source = Input::open(&remuxed.fixture.path).unwrap();
    let output = Input::open(&remuxed.output).unwrap();
    let source_facts = read_stream_facts(&source).unwrap();
    let output_facts = read_stream_facts(&output).unwrap();
    assert_ne!(source_facts.time_base, output_facts.time_base);
    let target = source_facts.geometry.display_dimensions();
    check_output(&source_facts, &output_facts, target).unwrap_or_else(|error| panic!("{error}"));

    let Some(probed) = probe(&remuxed.output) else {
        return;
    };
    let from = probe(&remuxed.fixture.path).unwrap();
    assert_near(
        "video pts",
        &from.sorted_pts(from.streams("video")[0]),
        &probed.sorted_pts(probed.streams("video")[0]),
        1_000_001,
    );
    assert_eq!(probed.decode_log(), "");
}
