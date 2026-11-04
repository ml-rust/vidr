//! B-frames: packets in decode order, with strictly increasing dts.

use vidr::encode::nvenc::NvencEncoder;
use vidr::encode::{EncodeCodec, PixelFormat};

use super::support::{self, DURATION, FIRST_PTS};

const FRAMES: usize = 16;
const B_FRAMES: u32 = 2;
const FLOOR_DB: f64 = 45.0;

fn run(codec: EncodeCodec, name: &str) {
    let Some(client) = support::gpu() else {
        return;
    };
    let format = PixelFormat::Nv12;
    let mut config = support::config(codec, format);
    config.b_frames = B_FRAMES;
    let mut encoder =
        NvencEncoder::new(&client, config).unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let encoded = support::encode_gradient(&client, &mut encoder, format, FRAMES, 0, name);
    support::check_packets(&encoded, name);

    let packets = &encoded.packets;
    assert!(
        packets.windows(2).any(|pair| pair[0].pts > pair[1].pts),
        "{name}: pts never reorders with b_frames {B_FRAMES}"
    );
    assert_eq!(
        packets[0].dts,
        FIRST_PTS - i64::from(B_FRAMES) * DURATION,
        "{name}: first dts"
    );
    assert_eq!(packets[0].pts, FIRST_PTS, "{name}: first pts");

    support::check_nvdec(&client, codec, format, &encoded, FLOOR_DB, name);
    let path = support::write_annexb(codec, &encoded, name);
    if let Some(facts) = support::ffmpeg_facts(&path, name) {
        assert_eq!(
            facts.get("nb_read_frames").map(String::as_str),
            Some(FRAMES.to_string().as_str()),
            "{name}: ffmpeg frames"
        );
    }
}

#[test]
fn h264_b_frames_reorder_with_increasing_dts() {
    run(EncodeCodec::H264, "nvenc_h264_bframes");
}

#[test]
fn hevc_b_frames_reorder_with_increasing_dts() {
    run(EncodeCodec::Hevc, "nvenc_hevc_bframes");
}

/// Drop with pictures the encoder still holds must end the stream and release
/// every resource without a flush.
#[test]
fn drop_without_flush_releases_held_pictures() {
    let name = "nvenc_drop";
    let Some(client) = support::gpu() else {
        return;
    };
    let format = PixelFormat::Nv12;
    let mut config = support::config(EncodeCodec::H264, format);
    config.b_frames = B_FRAMES;
    let mut encoder =
        NvencEncoder::new(&client, config).unwrap_or_else(|error| panic!("{name}: open: {error}"));
    for t in 0..2 {
        let planes = support::frame_planes(&client, format, t, 0);
        let frame = vidr::encode::EncodeFrame::new(&planes.y, &planes.uv);
        encoder
            .send(&frame, t as i64 * DURATION, DURATION)
            .unwrap_or_else(|error| panic!("{name}: send: {error}"));
    }
    drop(encoder);
}
