//! The format sniff and the track choice.

use vidr::MediaError;
use vidr::media::Input;

use super::fixtures::{container, load, open, scratch_file, video, width};

fn open_error(path: &std::path::Path) -> MediaError {
    match Input::open(path) {
        Ok(input) => panic!("{} opened as {}", path.display(), container(&input)),
        Err(error) => error,
    }
}

#[test]
fn the_bytes_pick_the_container_and_the_extension_does_not() {
    for (name, expected) in [
        ("mp4_bytes.mkv", "mp4"),
        ("mkv_bytes.mp4", "mkv"),
        ("plain.mp4", "mp4"),
        ("plain.mov", "mp4"),
        ("plain.mkv", "mkv"),
        ("vp9.webm", "mkv"),
    ] {
        if let Some((_, input)) = open(name) {
            assert_eq!(container(&input), expected, "{name}");
        }
    }
}

#[test]
fn an_unknown_format_names_its_first_bytes_and_the_containers() {
    let path = scratch_file("media_unknown.bin", b"RIFF\x24\0\0\0AVI LIST\0\0\0\0");
    let error = open_error(&path);
    assert!(matches!(
        &error,
        MediaError::UnknownFormat { first_bytes, .. } if first_bytes.starts_with(b"RIFF")
    ));
    let message = error.to_string();
    assert!(message.contains("52 49 46 46"), "{message}");
    assert!(message.contains("MP4/MOV"), "{message}");
    assert!(message.contains("Matroska/WebM"), "{message}");
    assert!(message.contains("media_unknown.bin"), "{message}");
    assert!(!message.contains(';'), "{message}");
}

#[test]
fn an_empty_file_is_an_unknown_format() {
    let path = scratch_file("media_empty.bin", b"");
    assert_eq!(
        open_error(&path),
        MediaError::UnknownFormat {
            path,
            first_bytes: Vec::new(),
        }
    );
}

#[test]
fn a_missing_file_is_a_read_error_that_names_it() {
    let path = std::path::Path::new("no such directory/no such file.mp4");
    let error = open_error(path);
    assert!(matches!(error, MediaError::Read { .. }), "{error}");
    assert!(error.to_string().contains("no such file.mp4"), "{error}");
}

#[cfg(not(feature = "mkv"))]
#[test]
fn matroska_bytes_without_the_feature_name_the_feature() {
    let path = scratch_file("media_ebml.bin", &[0x1A, 0x45, 0xDF, 0xA3, 0x80]);
    assert!(matches!(
        open_error(&path),
        MediaError::ContainerNotEnabled { feature: "mkv", .. }
    ));
}

#[cfg(not(feature = "mp4"))]
#[test]
fn mp4_bytes_without_the_feature_name_the_feature() {
    let path = scratch_file("media_ftyp.bin", b"\0\0\0\x08ftyp");
    assert!(matches!(
        open_error(&path),
        MediaError::ContainerNotEnabled { feature: "mp4", .. }
    ));
}

#[test]
fn a_damaged_file_surfaces_containrs_refusal() {
    // A box type the sniff takes, and a size past the input end.
    let path = scratch_file("media_damaged.bin", b"\0\0\xFF\xFFmoov");
    if cfg!(feature = "mp4") {
        let error = open_error(&path);
        assert!(matches!(error, MediaError::Demux { .. }), "{error}");
        assert!(error.to_string().contains("media_damaged.bin"), "{error}");
    }
}

#[test]
fn fragmented_mp4_surfaces_containrs_refusal() {
    let Some(fixture) = load("fragmented.mp4") else {
        return;
    };
    let error = open_error(&fixture.path);
    assert!(matches!(error, MediaError::Fragmented { .. }), "{error}");
    let message = error.to_string();
    assert!(message.contains("fragmented MP4"), "{message}");
    assert!(message.contains("fragmented.mp4"), "{message}");
}

#[test]
fn the_first_video_track_wins_over_a_later_default_track() {
    for name in ["second_default.mp4", "second_default.mkv"] {
        let Some((fixture, input)) = open(name) else {
            continue;
        };
        // The recipe holds: ffprobe sees two streams and flags the second.
        assert_eq!(fixture.number("video_streams"), 2, "{name}");
        assert_eq!(fixture.number("default_index"), 1, "{name}");
        let track = video(&fixture, &input);
        assert_eq!(
            width(&track) as i64,
            fixture.number("first_width"),
            "{name}"
        );
        assert_eq!(width(&track) as i64, fixture.number("width"), "{name}");
        assert_eq!(input.tracks()[0].id, track.id, "{name}");
    }
}

#[test]
fn an_attached_picture_is_not_the_video_track() {
    for name in ["attached_picture.mp4", "attached_picture.mkv"] {
        let Some((fixture, input)) = open(name) else {
            continue;
        };
        assert_eq!(fixture.number("attached_pictures"), 1, "{name}");
        let track = video(&fixture, &input);
        assert_eq!(width(&track) as i64, fixture.number("width"), "{name}");
    }
}

#[test]
fn no_video_track_is_an_error_that_names_the_input() {
    for name in ["audio_only.mp4", "audio_only.mkv"] {
        let Some((fixture, input)) = open(name) else {
            continue;
        };
        let error = input.video_track().map(|_| ()).unwrap_err();
        assert!(
            matches!(&error, MediaError::NoVideoTrack { path, tracks: 1 } if *path == fixture.path),
            "{name}: {error}"
        );
        assert!(error.to_string().contains(name), "{error}");
    }
}

#[test]
fn media_errors_convert_into_the_crate_error() {
    let path = std::path::Path::new("no such directory/no such file.mp4");
    let Err(error) = Input::open(path) else {
        panic!("{} opened", path.display());
    };
    assert!(matches!(error, MediaError::Read { .. }));
    let wrapped = vidr::Error::from(error.clone());
    assert_eq!(wrapped.to_string(), error.to_string());
}
