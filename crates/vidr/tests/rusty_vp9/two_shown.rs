//! A superframe with two shown frames through the pump. `RustyVp9Decoder` returns the
//! last shown frame of a chunk, as libvpx does. So the frame count stays one per
//! packet, and each frame keeps its packet's timing.

use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use vidr::Rational;
use vidr::decode::rusty_vp9::RustyVp9Decoder;
use vidr::pipeline::PumpTiming;

use crate::decode_fixtures::cpu::{check_bytes, decode_all, samples};
use crate::decode_fixtures::load;
use crate::pipeline::fakes::{Probe, pump_decoder};

/// Ticks of one packet, in the fixture's 1/25 s time base: one per frame it holds.
const PACKET_TICKS: i64 = 2;

#[test]
fn each_superframe_gives_its_last_shown_frame() {
    let name = "vp9_two_shown";
    let Some(stream) = load(name) else {
        return;
    };
    let mut decoder = RustyVp9Decoder::<CpuRuntime>::new(&CpuDevice::new());
    let frames = decode_all(&mut decoder, samples(&stream), name);
    assert_eq!(frames.len(), stream.samples.len(), "{name}: frame count");
    let pts: Vec<i64> = frames.iter().map(|frame| frame.pts()).collect();
    assert_eq!(pts, stream.pts, "{name}: pts");
    check_bytes(&stream, &frames);
}

#[test]
fn the_pump_keeps_frame_count_and_timing() {
    let name = "vp9_two_shown";
    let Some(stream) = load(name) else {
        return;
    };
    let (height, width) = (
        stream.number("height") as usize,
        stream.number("width") as usize,
    );
    let packets: Vec<(Vec<u8>, i64, i64)> = samples(&stream)
        .into_iter()
        .map(|(bytes, pts)| (bytes, pts, PACKET_TICKS))
        .collect();
    let count = packets.len() as u64;
    let timing = PumpTiming::new(Rational::new(1, 25).unwrap());
    let decoder = RustyVp9Decoder::<CpuRuntime>::new(&CpuDevice::new());
    let (result, probe) = pump_decoder(packets, decoder, [height, width], timing);
    let report = result.unwrap();
    assert_eq!(report.frames_in, count, "{name}: frames in");
    assert_eq!(report.frames_out, count, "{name}: frames out");
    assert_eq!(report.packets, count, "{name}: packets");
    let encoded = probe.encoded();
    let expected: Vec<(i64, i64)> = stream.pts.iter().map(|&pts| (pts, PACKET_TICKS)).collect();
    assert_eq!(encoded, expected, "{name}: pts and durations");
    assert_eq!(Probe::read(&probe.sink_finished), 1);
}
