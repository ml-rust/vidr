//! A long stream stays within the configured buffer bound, and every packet of
//! the copied track still arrives.

use containr::mp4::{MuxOptions, Muxer as Mp4Muxer, PacketIn, TrackSpec};
use containr::{EditList, TrackKind};
use vidr::Rational;
use vidr::encode::{EncodeCodec, EncodedPacket};
use vidr::media::{Input, OutputKind, OutputMux, VideoOutput, plan_copy, read_track_facts};

use super::support::{create, scratch_path};
use crate::media::fixtures::open;

const VIDEO_TICKS_PER_SECOND: u64 = 12_800;
const VIDEO_FRAME: i64 = 512;
const VIDEO_FRAMES: i64 = 10_000;
const VIDEO_PACKET: usize = 4096;
const AUDIO_PACKETS: i64 = 18_750;
const AUDIO_FRAME: i64 = 1024;
const AUDIO_PACKET: usize = 256;
/// `containr::mp4::MAX_CHUNK_BYTES`, the least an MP4 muxer takes.
const MAX_BUFFER: usize = 1 << 20;

/// An MP4 source with a long audio track. Each sample holds filler bytes, and
/// no reader decodes them. The sample entries come from a real fixture.
fn long_source(tag: &str) -> Option<std::path::PathBuf> {
    let (_, input) = open("remux_h264.mp4")?;
    let tracks = input.tracks();
    let spec_of = |kind: TrackKind| {
        let track = tracks.iter().find(|track| track.kind == kind).unwrap();
        let mut spec = TrackSpec::copy_from(track).unwrap();
        spec.edit_list = EditList::default();
        spec.metadata.clear();
        spec.references.clear();
        spec
    };
    let path = scratch_path(tag, "long_source", "mp4");
    let mut muxer = Mp4Muxer::new(create(&path), MuxOptions::default()).unwrap();
    let video = muxer.add_track(spec_of(TrackKind::Video)).unwrap();
    let audio = muxer.add_track(spec_of(TrackKind::Audio)).unwrap();
    muxer
        .write_packet(
            video,
            &PacketIn {
                data: &[0; 4],
                pts: 0,
                dts: 0,
                duration: Some(512),
                sync: true,
            },
        )
        .unwrap();
    let data = vec![0u8; AUDIO_PACKET];
    for packet in 0..AUDIO_PACKETS {
        let dts = packet * AUDIO_FRAME;
        muxer
            .write_packet(
                audio,
                &PacketIn {
                    data: &data,
                    pts: dts,
                    dts,
                    duration: Some(1024),
                    sync: true,
                },
            )
            .unwrap();
    }
    muxer.finish().unwrap();
    Some(path)
}

fn run(kind: OutputKind) {
    let tag = format!("memory_{}", kind.extension());
    let Some(source) = long_source(&tag) else {
        return;
    };
    let mut input = Input::open(&source).unwrap();
    let plan = plan_copy(&mut input, kind)
        .unwrap()
        .with_max_buffer(MAX_BUFFER);
    let video = VideoOutput::new(
        EncodeCodec::H264,
        vec![1, 2, 3],
        64,
        48,
        Rational::new(1, VIDEO_TICKS_PER_SECOND).unwrap(),
    );
    let output = scratch_path(&tag, "long_output", kind.extension());
    let mut mux = OutputMux::start(create(&output), plan, video).unwrap();
    let bound = mux.buffer_bound() + VIDEO_PACKET + AUDIO_PACKET;
    assert!(bound < VIDEO_PACKET * usize::try_from(VIDEO_FRAMES).unwrap());
    let mut peak = 0;
    for frame in 0..VIDEO_FRAMES {
        let pts = frame * VIDEO_FRAME;
        mux.write_video(&EncodedPacket {
            data: vec![0; VIDEO_PACKET],
            pts,
            dts: pts,
            duration: VIDEO_FRAME,
            keyframe: frame == 0,
        })
        .unwrap();
        let held = mux.buffered_bytes();
        assert!(held <= bound, "frame {frame}: {held} bytes above {bound}");
        peak = peak.max(held);
    }
    assert!(peak > 0, "the muxer holds packets before it writes them");
    mux.finish().unwrap();

    let written = Input::open(&output).unwrap();
    let facts = read_track_facts(&written).unwrap();
    assert_eq!(facts.audio.len(), 1);
    assert_eq!(facts.audio[0].packets, AUDIO_PACKETS as u64);
    std::fs::remove_file(&output).unwrap();
    std::fs::remove_file(&source).unwrap();
}

#[test]
fn a_long_stream_stays_within_the_bound_in_mp4() {
    run(OutputKind::Mp4);
}

#[test]
fn a_long_stream_stays_within_the_bound_in_mkv() {
    run(OutputKind::Mkv);
}
