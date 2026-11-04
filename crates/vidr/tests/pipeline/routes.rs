//! Host-copy counts from each decoder and encoder route, through the whole pump.

use vidr::FrameRoute;
use vidr::pipeline::PumpReport;
use vidr::time::{EditEntry, EditTimeline};

use super::fakes::{Script, millis, pump, pump_boxed, steady};

const FRAMES: u64 = 5;
/// One fake frame: U8 luma `[2, 2]` and chroma `[1, 2]`.
const FRAME_BYTES: u64 = 6;
/// One doubled frame: U8 luma `[4, 4]` and chroma `[2, 4]`.
const DOUBLED_BYTES: u64 = 24;

const HOST_ROUTES: [FrameRoute; 3] = [
    FrameRoute::HostUpload,
    FrameRoute::HostDownload,
    FrameRoute::HostOnly,
];

fn routed(decoder: FrameRoute, encoder: FrameRoute) -> Script {
    Script {
        packets: steady(FRAMES as i64),
        decoder_route: Some(decoder),
        encoder_route: Some(encoder),
        ..Script::default()
    }
}

fn counts(report: &PumpReport<u64>) -> (u64, u64) {
    (report.host_copied_frames, report.host_copied_bytes)
}

#[test]
fn device_routes_copy_nothing() {
    let (result, _) = pump(routed(FrameRoute::Device, FrameRoute::Device), millis());
    assert_eq!(counts(&result.unwrap()), (0, 0));
}

/// Each host route on the decoder alone counts one copy per frame.
#[test]
fn a_host_decoder_counts_each_frame_once() {
    for route in HOST_ROUTES {
        let (result, _) = pump(routed(route, FrameRoute::Device), millis());
        assert_eq!(
            counts(&result.unwrap()),
            (FRAMES, FRAMES * FRAME_BYTES),
            "{route}"
        );
    }
}

/// Each host route on the encoder alone counts one copy per frame.
#[test]
fn a_host_encoder_counts_each_frame_once() {
    for route in HOST_ROUTES {
        let (result, _) = pump(routed(FrameRoute::Device, route), millis());
        assert_eq!(
            counts(&result.unwrap()),
            (FRAMES, FRAMES * FRAME_BYTES),
            "{route}"
        );
    }
}

#[test]
fn a_host_decoder_and_a_host_encoder_each_count() {
    let script = routed(FrameRoute::HostUpload, FrameRoute::HostDownload);
    let (result, _) = pump(script, millis());
    assert_eq!(
        counts(&result.unwrap()),
        (2 * FRAMES, 2 * FRAMES * FRAME_BYTES)
    );
}

/// The decoder's copies count the decoded size, and the encoder's count the
/// transform's output size.
#[test]
fn each_side_counts_its_own_frame_size() {
    let doubled = |decoder, encoder| Script {
        transform_doubles: true,
        ..routed(decoder, encoder)
    };
    let (result, _) = pump(
        doubled(FrameRoute::HostUpload, FrameRoute::Device),
        millis(),
    );
    assert_eq!(counts(&result.unwrap()), (FRAMES, FRAMES * FRAME_BYTES));
    let (result, _) = pump(
        doubled(FrameRoute::Device, FrameRoute::HostDownload),
        millis(),
    );
    assert_eq!(counts(&result.unwrap()), (FRAMES, FRAMES * DOUBLED_BYTES));
    let (result, _) = pump(
        doubled(FrameRoute::HostUpload, FrameRoute::HostDownload),
        millis(),
    );
    assert_eq!(
        counts(&result.unwrap()),
        (2 * FRAMES, FRAMES * (FRAME_BYTES + DOUBLED_BYTES))
    );
}

/// A frame the edit window drops took its decoder copy, and takes no encoder copy.
#[test]
fn a_dropped_frame_counts_only_its_decoder_copy() {
    let base = vidr::Rational::new(1, 1000).unwrap();
    let window = EditTimeline::resolve(&[EditEntry::media(40, 25)], base, base).unwrap();
    let timing = vidr::pipeline::PumpTiming {
        timeline: Some(window),
        ..millis()
    };
    let script = Script {
        packets: steady(10),
        decoder_route: Some(FrameRoute::HostUpload),
        encoder_route: Some(FrameRoute::HostDownload),
        ..Script::default()
    };
    let (result, _) = pump(script, timing);
    let report = result.unwrap();
    assert_eq!(report.frames_in, 10);
    assert_eq!(report.frames_out, 5);
    assert_eq!(counts(&report), (15, 15 * FRAME_BYTES));
}

/// The pump takes boxed trait objects, and the boxes forward the route.
#[test]
fn boxed_codecs_run_and_forward_their_routes() {
    let script = routed(FrameRoute::HostUpload, FrameRoute::HostDownload);
    let report = pump_boxed(script, millis()).unwrap();
    assert_eq!(report.frames_out, FRAMES);
    assert_eq!(report.packets, FRAMES);
    assert_eq!(counts(&report), (2 * FRAMES, 2 * FRAMES * FRAME_BYTES));
}
