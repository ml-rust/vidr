//! A fake encoder over a real stream's samples, and a remux through it.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use vidr::check::TrackFacts;
use vidr::color::{color_plan, merge_container};
use vidr::decode::{
    ColorFacts, DecodeCodec, MATRIX_UNSPECIFIED, PRIMARIES_UNSPECIFIED, TRANSFER_UNSPECIFIED,
};
use vidr::encode::{EncodeCodec, EncodedPacket};
use vidr::media::{DroppedTrack, Input, OutputKind, OutputMux, VideoOutput, plan_copy};
use vidr::{
    DurationHints, FrameTiming, Rational, Rounding, SampleAspectRatio, StartOffset, TrackTiming,
};

use crate::media::fixtures::{Fixture, open};

/// A path under cargo's test directory. `tag` keeps tests in separate files,
/// since nextest runs them in parallel.
pub(crate) fn scratch_path(tag: &str, stem: &str, extension: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("output_{tag}_{stem}.{extension}"))
}

/// A file a muxer writes and reads back.
pub(crate) fn create(path: &Path) -> File {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .unwrap_or_else(|error| panic!("creating {}: {error}", path.display()))
}

/// What an encoder hands the output for one stream.
pub(crate) struct FakeEncoder {
    pub(crate) packets: Vec<EncodedPacket>,
    pub(crate) video: VideoOutput,
}

/// The demuxed samples of track `track_id`, in decode order.
fn samples(input: &mut Input, track_id: u32) -> Vec<(containr::Packet, Vec<u8>)> {
    match input {
        Input::Mp4 { demuxer, .. } => {
            let packets: Vec<_> = demuxer
                .track_packets(track_id)
                .unwrap()
                .map(Result::unwrap)
                .collect();
            packets
                .into_iter()
                .map(|packet| {
                    let data = demuxer.read_packet(&packet).unwrap();
                    (packet, data)
                })
                .collect()
        }
        Input::Mkv { demuxer, .. } => {
            let packets: Vec<_> = demuxer
                .track_packets(track_id)
                .unwrap()
                .map(Result::unwrap)
                .collect();
            packets
                .into_iter()
                .map(|packet| {
                    let data = demuxer.read_packet(&packet).unwrap();
                    (packet, data)
                })
                .collect()
        }
        _ => panic!("vidr has a container this test does not name"),
    }
}

/// The encoder's view of the source's first video track: the samples in decode
/// order, the presented pts, and decode times that precede the first pts by the
/// B-frame delay, as NVENC writes them.
pub(crate) fn fake_encoder(fixture: &Fixture, input: &mut Input) -> FakeEncoder {
    let track = input.video_track().unwrap().clone();
    let time_base =
        Rational::new(i64::from(track.timebase.num), u64::from(track.timebase.den)).unwrap();
    let samples = samples(input, track.id);
    let timeline = input.edit_timeline(&track).unwrap();

    let mut frames: Vec<FrameTiming> = samples
        .iter()
        .map(|(packet, _)| {
            let duration = packet
                .duration
                .map_or(0, |ticks| i64::try_from(ticks).unwrap());
            FrameTiming::new(packet.pts, packet.dts, duration)
        })
        .collect();
    frames.sort_by_key(|frame| frame.pts);
    let presented = timeline.present(&frames).unwrap();
    assert_eq!(presented.len(), samples.len(), "{}", fixture.name);
    let timing = TrackTiming::new(
        time_base,
        StartOffset {
            edit: timeline.start_offset(),
            container_start: Rational::ZERO,
        },
        presented.clone(),
        DurationHints::default(),
    )
    .unwrap();

    // Each frame lasts until the next one presents.
    let mut durations: HashMap<i64, i64> = presented
        .windows(2)
        .map(|pair| (pair[0].pts, pair[1].pts - pair[0].pts))
        .collect();
    durations.insert(
        timing.last_pts(),
        timing.last_duration_ticks(Rounding::HalfEven).unwrap(),
    );

    let shift = timeline.media_start();
    let delay = presented
        .iter()
        .zip(&samples)
        .map(|(frame, (packet, _))| frame.pts - (packet.pts - shift))
        .max()
        .unwrap();
    let packets = presented
        .iter()
        .zip(samples)
        .map(|(frame, (packet, data))| {
            let pts = packet.pts - shift;
            EncodedPacket {
                data,
                pts,
                dts: frame.pts - delay,
                duration: durations[&pts],
                keyframe: packet.sync,
            }
        })
        .collect();

    let config = input.decode_config(&track).unwrap();
    let codec = match config.codec {
        DecodeCodec::H264 => EncodeCodec::H264,
        DecodeCodec::Hevc => EncodeCodec::Hevc,
        DecodeCodec::Av1 => EncodeCodec::Av1,
        other => panic!("{}: no encoder writes {other}", fixture.name),
    };
    let width = u32::try_from(fixture.number("width")).unwrap();
    let height = u32::try_from(fixture.number("height")).unwrap();
    let mut video = VideoOutput::new(codec, config.config, width, height, time_base);
    video.sample_aspect_ratio = input
        .sample_aspect_ratio(&track)
        .unwrap_or(SampleAspectRatio::SQUARE);
    video.rotation = input.rotation(&track).unwrap();
    video.language = track.language;
    video.start_offset = timing.video_offset().unwrap();
    let unspecified = ColorFacts {
        matrix_coefficients: MATRIX_UNSPECIFIED,
        transfer_characteristics: TRANSFER_UNSPECIFIED,
        colour_primaries: PRIMARIES_UNSPECIFIED,
        full_range: false,
        chroma_location: None,
    };
    let merged = merge_container(unspecified, input.colour(&track));
    video.colour = color_plan(&merged).unwrap().tags;
    FakeEncoder { packets, video }
}

/// A finished remux.
pub(crate) struct Remuxed {
    pub(crate) fixture: Fixture,
    pub(crate) output: PathBuf,
    pub(crate) expected: TrackFacts,
    pub(crate) dropped: Vec<DroppedTrack>,
    pub(crate) max_rounding_error_ns: Option<u64>,
}

/// Remuxes fixture `name` into `kind` through a fake encoder. `None` when the
/// fixture is missing.
pub(crate) fn remux(tag: &str, name: &str, kind: OutputKind) -> Option<Remuxed> {
    let (fixture, mut input) = open(name)?;
    let plan =
        plan_copy(&mut input, kind).unwrap_or_else(|error| panic!("{name} to {kind}: {error}"));
    let expected = plan.expected_tracks();
    let dropped = plan.dropped().to_vec();
    let max_rounding_error_ns = plan.max_rounding_error_ns();
    let encoder = fake_encoder(&fixture, &mut input);

    let output = scratch_path(tag, name, kind.extension());
    let mut mux = OutputMux::start(create(&output), plan, encoder.video)
        .unwrap_or_else(|error| panic!("{name} to {kind}: {error}"));
    for packet in &encoder.packets {
        mux.write_video(packet)
            .unwrap_or_else(|error| panic!("{name} to {kind}: {error}"));
    }
    mux.finish()
        .unwrap_or_else(|error| panic!("{name} to {kind}: {error}"));
    Some(Remuxed {
        fixture,
        output,
        expected,
        dropped,
        max_rounding_error_ns,
    })
}
