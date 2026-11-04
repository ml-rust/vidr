//! Packet timing, keyframes, the config record, refusals, and call order.

use numr::runtime::cpu::CpuRuntime;
use vidr::bitstream::av1::{Av1Config, av1_sequence_facts};
use vidr::decode::ChromaLocation;
use vidr::encode::rav1r::Rav1rEncoder;
use vidr::encode::{
    ColorFacts, EncodeCodec, EncodeConfig, EncodeFrame, EncodedPacket, Encoder, PixelFormat,
    RateControl, Tuning,
};
use vidr::{
    Dimensions, DurationHints, EncodeError, FrameRoute, FrameTiming, Geometry, Rotation,
    SampleAspectRatio, StreamFacts, check_output,
};

use super::support::{CLIPS, Clip, DURATION, FRAMES, base, encode, pts_of, rate};

const BASE_CLIP: Clip = Clip::new(96, 64, PixelFormat::Nv12);

/// Pts and durations as sent, dts equal to pts, strictly increasing.
fn check_timing(packets: &[EncodedPacket], frames: usize, name: &str) {
    assert_eq!(packets.len(), frames, "{name}: packet count");
    assert!(packets[0].keyframe, "{name}: packet 0 is not a keyframe");
    for (index, packet) in packets.iter().enumerate() {
        assert_eq!(packet.pts, pts_of(index), "{name}: packet {index} pts");
        assert_eq!(packet.dts, packet.pts, "{name}: packet {index} dts");
        assert_eq!(packet.duration, DURATION, "{name}: packet {index} duration");
        assert!(!packet.data.is_empty(), "{name}: packet {index} is empty");
    }
}

/// The frame count, size, aspect ratio, pts, and total duration pass the same
/// check a run's output takes.
fn check_as_output(clip: &Clip, packets: &[EncodedPacket], name: &str) {
    let size = Dimensions::new(clip.width as u32, clip.height as u32)
        .unwrap_or_else(|error| panic!("{name}: size: {error}"));
    let geometry = Geometry::new(size, Rotation::default(), SampleAspectRatio::SQUARE);
    let hints = DurationHints {
        packet_duration: Some(DURATION),
        average_frame_rate: Some(rate()),
        base_frame_rate: Some(rate()),
    };
    let facts = |frames: Vec<FrameTiming>| StreamFacts::new(geometry, base(), frames, hints);
    let source: Vec<FrameTiming> = (0..packets.len())
        .map(|index| FrameTiming::new(pts_of(index), None, DURATION))
        .collect();
    let output: Vec<FrameTiming> = packets
        .iter()
        .map(|packet| FrameTiming::new(packet.pts, Some(packet.dts), packet.duration))
        .collect();
    check_output(&facts(source), &facts(output), size)
        .unwrap_or_else(|error| panic!("{name}: {error}"));
}

#[test]
fn packets_keep_their_timing_at_each_depth_and_size() {
    for clip in &CLIPS {
        let name = format!("{clip:?}");
        let (packets, _) = encode(clip, clip.config(), FRAMES, &name);
        check_timing(&packets, FRAMES, &name);
        check_as_output(clip, &packets, &name);
    }
}

/// rav1r's reorder hides frames inside temporal units, so packets stay in display
/// order with dts equal to pts, one per frame.
#[test]
fn reorder_keeps_display_order_and_the_frame_count() {
    let name = "rav1r_reorder";
    let mut config = BASE_CLIP.config();
    config.b_frames = 3;
    let (packets, _) = encode(&BASE_CLIP, config, FRAMES, name);
    check_timing(&packets, FRAMES, name);
    check_as_output(&BASE_CLIP, &packets, name);
}

#[test]
fn a_low_latency_tuning_keeps_the_frame_count() {
    let name = "rav1r_low_latency";
    let mut config = BASE_CLIP.config();
    config.tuning = Tuning::UltraLowLatency;
    let (packets, _) = encode(&BASE_CLIP, config, FRAMES, name);
    check_timing(&packets, FRAMES, name);
}

#[test]
fn bitrate_mode_keeps_the_frame_count() {
    let name = "rav1r_bitrate";
    let mut config = BASE_CLIP.config();
    config.rate_control = RateControl::Bitrate {
        average: 300_000,
        maximum: None,
    };
    let (packets, _) = encode(&BASE_CLIP, config, FRAMES, name);
    check_timing(&packets, FRAMES, name);
}

fn key_positions(packets: &[EncodedPacket]) -> Vec<usize> {
    packets
        .iter()
        .enumerate()
        .filter_map(|(index, packet)| packet.keyframe.then_some(index))
        .collect()
}

#[test]
fn keyframes_follow_the_gop_length() {
    let name = "rav1r_gop";
    let mut config = BASE_CLIP.config();
    config.gop_length = 4;
    let (packets, _) = encode(&BASE_CLIP, config, 9, name);
    assert_eq!(key_positions(&packets), [0, 4, 8], "{name}: keyframes");
}

/// With reorder, rav1r places no keyframe in the last 5 frames of a stream.
#[test]
fn reorder_keyframes_follow_the_gop_length() {
    let name = "rav1r_reorder_gop";
    let mut config = BASE_CLIP.config();
    config.b_frames = 3;
    config.gop_length = 6;
    let (packets, _) = encode(&BASE_CLIP, config, FRAMES, name);
    assert_eq!(key_positions(&packets), [0, 6], "{name}: keyframes");
    assert_eq!(packets.len(), FRAMES, "{name}: packet count");
}

/// A scene cut changes no frame's type and no frame count, in either mode.
#[test]
fn a_scene_cut_adds_no_keyframe() {
    let clip = Clip {
        cut: Some(4),
        ..BASE_CLIP
    };
    for b_frames in [0, 3] {
        let name = format!("rav1r_cut_b{b_frames}");
        let mut config = clip.config();
        config.b_frames = b_frames;
        let (packets, _) = encode(&clip, config, FRAMES, &name);
        assert_eq!(key_positions(&packets), [0], "{name}: keyframes");
        assert_eq!(packets.len(), FRAMES, "{name}: packet count");
    }
}

fn tagged_config(clip: &Clip) -> EncodeConfig {
    let mut config = clip.config();
    config.color = ColorFacts {
        matrix_coefficients: 1,
        transfer_characteristics: 1,
        colour_primaries: 1,
        full_range: true,
        chroma_location: Some(ChromaLocation::Left),
    };
    config
}

/// `av1C` parses, and its fields and sequence header state the config.
#[test]
fn config_record_describes_the_stream() {
    for clip in &CLIPS[..2] {
        let name = format!("{clip:?}");
        let (_, record) = encode(clip, tagged_config(clip), 2, &name);
        let parsed = Av1Config::parse(&record).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(parsed.seq_profile, 0, "{name}: profile");
        assert_eq!(
            parsed.high_bitdepth,
            clip.format == PixelFormat::P010,
            "{name}: high_bitdepth"
        );
        assert!(!parsed.twelve_bit, "{name}: twelve_bit");
        assert!(!parsed.monochrome, "{name}: monochrome");
        assert!(parsed.chroma_subsampling_x, "{name}: subsampling_x");
        assert!(parsed.chroma_subsampling_y, "{name}: subsampling_y");
        assert_eq!(parsed.chroma_sample_position, 1, "{name}: chroma position");
        let header = parsed
            .sequence_header()
            .unwrap_or_else(|error| panic!("{name}: {error}"))
            .unwrap_or_else(|| panic!("{name}: av1C holds no sequence header"));
        let facts =
            av1_sequence_facts(header.payload).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(facts.bit_depth, clip.format.bit_depth(), "{name}: depth");
        assert_eq!(facts.color_primaries, 1, "{name}: primaries");
        assert_eq!(facts.transfer_characteristics, 1, "{name}: transfer");
        assert_eq!(facts.matrix_coefficients, 1, "{name}: matrix");
        assert!(facts.full_range, "{name}: range");
        assert_eq!(facts.chroma_sample_position, 1, "{name}: chroma position");
    }
}

fn refused_field(config: EncodeConfig) -> Option<&'static str> {
    match Rav1rEncoder::<CpuRuntime>::new(config) {
        Err(EncodeError::InvalidConfig { field, .. }) => Some(field),
        _ => None,
    }
}

/// Each config the adapter cannot honour names its field in `new`.
#[test]
fn refusals_name_the_field() {
    let with = |change: fn(&mut EncodeConfig)| {
        let mut config = BASE_CLIP.config();
        change(&mut config);
        config
    };
    type Case<'a> = (fn(&mut EncodeConfig), &'a str);
    let cases: [Case<'_>; 8] = [
        (|config| config.codec = EncodeCodec::Hevc, "codec"),
        (|config| config.width = 15, "width, height"),
        (|config| config.b_frames = 1, "b_frames"),
        (|config| config.b_frames = 2, "b_frames"),
        (|config| config.tuning = Tuning::Lossless, "tuning"),
        (
            |config| {
                config.rate_control = RateControl::ConstQp {
                    intra: 40,
                    inter_p: 40,
                    inter_b: 40,
                };
            },
            "rate_control",
        ),
        (
            |config| {
                config.rate_control = RateControl::Bitrate {
                    average: 500_000,
                    maximum: Some(900_000),
                };
            },
            "rate_control",
        ),
        (
            |config| config.color.colour_primaries = 3,
            "color.colour_primaries",
        ),
    ];
    for (change, field) in cases {
        assert_eq!(refused_field(with(change)), Some(field));
    }
}

fn open(clip: &Clip) -> Rav1rEncoder<CpuRuntime> {
    Rav1rEncoder::new(clip.config()).unwrap_or_else(|error| panic!("open: {error}"))
}

#[test]
fn the_route_declares_the_host_copy() {
    assert_eq!(open(&BASE_CLIP).route(), FrameRoute::HostOnly);
}

#[test]
fn calls_out_of_order_are_refused() {
    let mut encoder = open(&BASE_CLIP);
    let (y, uv) = BASE_CLIP.planes(0);
    let frame = EncodeFrame::new(&y, &uv);
    assert_eq!(
        encoder.send(&frame, 5, 0),
        Err(EncodeError::NonPositiveDuration {
            pts: 5,
            duration: 0
        })
    );
    assert_eq!(encoder.send(&frame, 5, DURATION), Ok(()));
    assert_eq!(
        encoder.send(&frame, 5, DURATION),
        Err(EncodeError::PtsNotIncreasing {
            pts: 5,
            previous: 5
        })
    );
    assert_eq!(encoder.flush(), Ok(()));
    assert_eq!(encoder.flush(), Ok(()));
    assert_eq!(
        encoder.send(&frame, 6, DURATION),
        Err(EncodeError::AfterFlush { pts: 6 })
    );
}

/// A frame of another size or depth is refused, and the encoder stays usable.
#[test]
fn a_frame_unlike_the_config_is_refused() {
    let mut encoder = open(&BASE_CLIP);
    let (small_y, small_uv) = Clip::new(48, 32, PixelFormat::Nv12).planes(0);
    assert!(matches!(
        encoder.send(&EncodeFrame::new(&small_y, &small_uv), 0, DURATION),
        Err(EncodeError::SurfaceMismatch { .. })
    ));
    let (deep_y, deep_uv) = Clip::new(96, 64, PixelFormat::P010).planes(0);
    assert!(matches!(
        encoder.send(&EncodeFrame::new(&deep_y, &deep_uv), 0, DURATION),
        Err(EncodeError::SurfaceMismatch { .. })
    ));
    let (y, uv) = BASE_CLIP.planes(0);
    assert_eq!(
        encoder.send(&EncodeFrame::new(&y, &uv), 0, DURATION),
        Ok(())
    );
    assert_eq!(encoder.flush(), Ok(()));
    assert!(encoder.receive().is_some(), "no packet for the frame");
    assert!(encoder.receive().is_none(), "a second packet for one frame");
}
