//! Rotation, sample aspect ratio, and the decoder's config against ffprobe.

use vidr::decode::DecodeCodec;
use vidr::media::Input;
use vidr::{GeometryError, MediaError, Rational, Rotation};

use super::fixtures::{Fixture, open, video};

fn rotation_of(fixture: &Fixture, input: &Input) -> Result<Rotation, MediaError> {
    input.rotation(&video(fixture, input))
}

#[test]
fn right_angle_rotations_match_ffprobe() {
    let cases = [
        ("plain.mp4", Rotation::Upright),
        ("plain.mkv", Rotation::Upright),
        ("rotate_90.mp4", Rotation::Clockwise270),
        ("rotate_90.mkv", Rotation::Clockwise270),
        ("rotate_180.mp4", Rotation::Clockwise180),
        ("rotate_180.mkv", Rotation::Clockwise180),
        ("rotate_270.mp4", Rotation::Clockwise90),
        ("rotate_270.mkv", Rotation::Clockwise90),
    ];
    for (name, expected) in cases {
        let Some((fixture, input)) = open(name) else {
            continue;
        };
        let rotation = rotation_of(&fixture, &input);
        assert_eq!(rotation, Ok(expected), "{name}");
        // ffprobe's side-data rotation is counterclockwise.
        assert_eq!(
            i64::from(expected.counterclockwise_degrees()),
            fixture.number("rotation").rem_euclid(360),
            "{name}"
        );
    }
}

#[test]
fn other_angles_are_refused_though_ffprobe_reads_them() {
    if let Some((fixture, input)) = open("rotate_45.mp4") {
        assert_eq!(fixture.number("rotation"), 45);
        assert!(matches!(
            rotation_of(&fixture, &input),
            Err(MediaError::Geometry {
                error: GeometryError::UnsupportedDisplayMatrix { .. },
                ..
            })
        ));
    }
    if let Some((fixture, input)) = open("rotate_45.mkv") {
        assert_eq!(fixture.number("rotation"), 45);
        assert!(matches!(
            rotation_of(&fixture, &input),
            Err(MediaError::Geometry {
                error: GeometryError::UnsupportedRotation { degrees: -45 },
                ..
            })
        ));
    }
}

#[test]
fn a_mirror_is_refused() {
    if let Some((fixture, input)) = open("mirror.mp4") {
        assert!(matches!(
            rotation_of(&fixture, &input),
            Err(MediaError::Geometry {
                error: GeometryError::UnsupportedDisplayMatrix { .. },
                ..
            })
        ));
    }
    if let Some((fixture, input)) = open("mirror.mkv") {
        assert!(matches!(
            rotation_of(&fixture, &input),
            Err(MediaError::Projection { .. })
        ));
    }
}

/// Parses ffprobe's `W:H`.
fn ratio(text: &str) -> Rational {
    let (num, den) = text.split_once(':').unwrap();
    Rational::new(num.parse().unwrap(), den.parse().unwrap()).unwrap()
}

#[test]
fn a_stated_sample_aspect_ratio_matches_ffprobe() {
    for name in ["anamorphic.mp4", "anamorphic.mkv", "plain.mp4"] {
        let Some((fixture, input)) = open(name) else {
            continue;
        };
        let track = video(&fixture, &input);
        let sar = input.sample_aspect_ratio(&track).unwrap();
        assert_eq!(sar.ratio(), ratio(fixture.meta("sar")), "{name}");
    }
}

#[test]
fn a_matroska_track_with_no_display_size_states_no_ratio() {
    // ffprobe reports 1:1 here, from the SPS.
    let Some((fixture, input)) = open("plain.mkv") else {
        return;
    };
    let track = video(&fixture, &input);
    assert_eq!(fixture.meta("sar"), "1:1");
    assert_eq!(input.sample_aspect_ratio(&track), None);
}

fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in bytes {
        a = (a + u32::from(byte)) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

#[test]
fn decode_config_matches_ffprobes_codec_and_extradata() {
    let cases = [
        ("plain.mp4", DecodeCodec::H264),
        ("plain.mov", DecodeCodec::H264),
        ("plain.mkv", DecodeCodec::H264),
        ("hevc.mp4", DecodeCodec::Hevc),
        ("hevc.mkv", DecodeCodec::Hevc),
        ("av1.mp4", DecodeCodec::Av1),
        ("av1.mkv", DecodeCodec::Av1),
        ("vp9.mp4", DecodeCodec::Vp9),
        ("vp9.webm", DecodeCodec::Vp9),
    ];
    for (name, codec) in cases {
        let Some((fixture, input)) = open(name) else {
            continue;
        };
        let track = video(&fixture, &input);
        let config = input
            .decode_config(&track)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(config.codec, codec, "{name}");
        let ffprobe_name = match codec {
            DecodeCodec::H264 => "h264",
            DecodeCodec::Hevc => "hevc",
            DecodeCodec::Av1 => "av1",
            DecodeCodec::Vp9 => "vp9",
            _ => panic!("{name}: a codec this test does not name"),
        };
        assert_eq!(fixture.meta("codec"), ffprobe_name, "{name}");
        // ffprobe reports no extradata for VP9.
        if let Some(hash) = fixture.meta_opt("extradata_adler32") {
            assert_eq!(format!("{:08x}", adler32(&config.config)), hash, "{name}");
        } else {
            assert_eq!(codec, DecodeCodec::Vp9, "{name}");
        }
    }
}

#[test]
fn an_audio_track_has_no_decode_config() {
    for name in ["audio_only.mp4", "audio_only.mkv"] {
        let Some((_, input)) = open(name) else {
            continue;
        };
        assert!(
            matches!(
                input.decode_config(&input.tracks()[0]),
                Err(MediaError::UnsupportedCodec { .. })
            ),
            "{name}"
        );
    }
}
