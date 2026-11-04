//! Each refusal comes from `plan_copy`, before GPU work, and names the stream
//! and the container that holds it.

use std::path::Path;

use containr::matroska::{DisplayUnit, MatroskaVideo};
use containr::mkv::{MkvOptions, Muxer, PacketIn, TrackSpec};
use containr::{Timebase, TrackKind};
use vidr::MediaError;
use vidr::media::{Input, OutputKind, plan_copy};

use super::probe::probe;
use super::support::{create, remux, scratch_path};
use crate::media::fixtures::open;

fn refusal(name: &str, kind: OutputKind) -> Option<MediaError> {
    let (_, mut input) = open(name)?;
    match plan_copy(&mut input, kind) {
        Ok(_) => panic!("{name} to {kind} is planned"),
        Err(error) => Some(error),
    }
}

/// Checks a `StreamRefused` that names `stream` and the output that holds it.
fn assert_refused(error: &MediaError, stream: &str, holder: OutputKind) {
    let MediaError::StreamRefused {
        stream: named,
        use_instead,
        ..
    } = error
    else {
        panic!("{error}");
    };
    assert!(named.contains(stream), "{error}");
    assert_eq!(*use_instead, Some(holder), "{error}");
    let message = error.to_string();
    assert!(message.contains(&holder.to_string()), "{message}");
    assert!(!message.contains(';'), "{message}");
}

#[test]
fn attachments_into_mp4_name_mkv_and_mkv_holds_them() {
    let Some(error) = refusal("attached_picture.mkv", OutputKind::Mp4) else {
        return;
    };
    assert!(
        matches!(error, MediaError::Attachments { count: 1, .. }),
        "{error}"
    );
    let message = error.to_string();
    assert!(message.contains("MKV"), "{message}");
    assert!(message.contains("attached_picture.mkv"), "{message}");
    assert!(!message.contains(';'), "{message}");

    let remuxed = remux("attachment", "attached_picture.mkv", OutputKind::Mkv).unwrap();
    let Some(output) = probe(&remuxed.output) else {
        return;
    };
    let source = probe(&remuxed.fixture.path).unwrap();
    let filename = |probe: &super::probe::Probe| {
        (0..4)
            .find_map(|index| probe.stream_tag(index, "filename"))
            .map(str::to_owned)
    };
    assert!(filename(&source).is_some(), "the fixture holds a file");
    assert_eq!(filename(&output), filename(&source));
}

#[test]
fn ass_into_mp4_names_mkv_and_copies_into_mkv() {
    let Some(error) = refusal("refuse_ass.mkv", OutputKind::Mp4) else {
        return;
    };
    assert_refused(&error, "S_TEXT/ASS", OutputKind::Mkv);
    let remuxed = remux("ass", "refuse_ass.mkv", OutputKind::Mkv).unwrap();
    let Some(output) = probe(&remuxed.output) else {
        return;
    };
    let subtitle = output.streams("subtitle")[0];
    assert_eq!(output.stream(subtitle, "codec_name"), Some("ass"));
}

#[test]
fn flac_into_mp4_names_mkv() {
    if let Some(error) = refusal("refuse_flac.mkv", OutputKind::Mp4) {
        assert_refused(&error, "A_FLAC", OutputKind::Mkv);
    }
}

#[test]
fn ac3_into_mkv_names_mp4() {
    if let Some(error) = refusal("refuse_ac3.mp4", OutputKind::Mkv) {
        assert_refused(&error, "ac-3", OutputKind::Mp4);
    }
}

/// A Matroska file with a video track and an image subtitle track. containr
/// writes it, since ffmpeg has no text-to-image subtitle path.
fn image_subtitle_file(path: &Path) {
    let timebase = Timebase::from_nanoseconds(1_000_000);
    let mut muxer = Muxer::new(create(path), MkvOptions::default()).unwrap();
    let mut video = TrackSpec::new(TrackKind::Video, "V_MPEG4/ISO/AVC", timebase);
    video.matroska.codec_private = Some(vec![1, 2, 3]);
    video.matroska.video = Some(MatroskaVideo {
        pixel_width: 64,
        pixel_height: 48,
        display_width: None,
        display_height: None,
        display_unit: DisplayUnit::Pixels,
        crop: [0; 4],
        interlaced: 0,
        projection: None,
        colour: None,
        extra: Vec::new(),
    });
    let video = muxer.add_track(video).unwrap();
    let image = TrackSpec::new(TrackKind::Subtitle, "S_HDMV/PGS", timebase);
    let image = muxer.add_track(image).unwrap();
    for (track, pts) in [(video, 0), (image, 0), (video, 40), (image, 40)] {
        let packet = PacketIn {
            data: &[1, 2, 3],
            pts,
            duration: Some(40),
            sync: true,
            discardable: false,
            discard_padding: None,
        };
        muxer.write_packet(track, &packet).unwrap();
    }
    muxer.finish().unwrap();
}

#[test]
fn an_image_subtitle_into_mp4_names_mkv_and_plans_into_mkv() {
    let path = scratch_path("image", "subtitle", "mkv");
    image_subtitle_file(&path);
    let mut input = Input::open(&path).unwrap();
    let error = plan_copy(&mut input, OutputKind::Mp4).unwrap_err();
    assert_refused(&error, "S_HDMV/PGS", OutputKind::Mkv);
    let plan = plan_copy(&mut input, OutputKind::Mkv).unwrap();
    assert_eq!(plan.expected_tracks().subtitles[0].packets, 2);
}

#[test]
fn a_vp9_source_needs_no_refusal_since_the_encoder_replaces_its_video() {
    for name in ["vp9.mp4", "vp9.webm"] {
        for kind in [OutputKind::Mp4, OutputKind::Mkv] {
            if let Some((_, mut input)) = open(name) {
                let plan = plan_copy(&mut input, kind)
                    .unwrap_or_else(|error| panic!("{name} to {kind}: {error}"));
                assert!(plan.dropped().is_empty(), "{name}");
            }
        }
    }
}

#[test]
fn only_mp4_and_mkv_are_output_extensions() {
    assert_eq!(
        OutputKind::from_path(Path::new("out.MP4")),
        Ok(OutputKind::Mp4)
    );
    assert_eq!(
        OutputKind::from_path(Path::new("out.mkv")),
        Ok(OutputKind::Mkv)
    );
    for name in ["out.webm", "out.mov", "out.avi", "out"] {
        let error = OutputKind::from_path(Path::new(name)).unwrap_err();
        assert!(
            matches!(error, MediaError::OutputExtension { .. }),
            "{error}"
        );
        let message = error.to_string();
        assert!(message.contains(name), "{message}");
        assert!(message.contains(".mp4"), "{message}");
        assert!(message.contains(".mkv"), "{message}");
    }
}

#[test]
fn an_inexact_chapter_is_refused_for_mkv_and_kept_for_mp4() {
    let Some(error) = refusal("refuse_chapter.mp4", OutputKind::Mkv) else {
        return;
    };
    assert_refused(&error, "chapters", OutputKind::Mp4);
    let remuxed = remux("chapter", "refuse_chapter.mp4", OutputKind::Mp4).unwrap();
    assert_eq!(remuxed.expected.chapters, 2);
}

#[test]
fn extra_video_data_and_timecode_tracks_drop_and_are_listed() {
    if let Some((_, mut input)) = open("second_default.mp4") {
        let plan = plan_copy(&mut input, OutputKind::Mkv).unwrap();
        let dropped = plan.dropped();
        assert_eq!(dropped.len(), 1);
        assert_eq!(dropped[0].kind, TrackKind::Video);
        assert_eq!(dropped[0].codec, "avc1");
    }
    if let Some((_, mut input)) = open("remux_tmcd.mov") {
        let plan = plan_copy(&mut input, OutputKind::Mp4).unwrap();
        let dropped = plan.dropped();
        assert_eq!(dropped.len(), 1);
        assert_eq!(dropped[0].kind, TrackKind::Data);
        assert_eq!(dropped[0].codec, "tmcd");
        assert_eq!(plan.expected_tracks().audio.len(), 1);
    }
}
