//! The facts vidr reads from a containr track: rotation, sample aspect ratio,
//! the decoder's codec and config record, and the facts the output checks
//! compare.

use containr::matroska::{DisplayUnit, MatroskaTrack, MatroskaVideo, Projection};
use containr::{Packet, Track, TrackKind};

use super::input::Input;
use crate::check::{StreamFacts, TrackFacts, TrackPacketCount};
use crate::decode::DecodeCodec;
use crate::error::MediaError;
use crate::geometry::{Dimensions, Geometry, Rotation, SampleAspectRatio};
use crate::time::{DurationHints, FrameTiming, Rational};

/// What a decoder takes for one track, such as
/// `NvdecDecoder::new(client, config.codec, &config.config)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeConfig {
    /// The codec.
    pub codec: DecodeCodec,
    /// The raw `avcC`, `hvcC`, `av1C` or `vpcC` body. Empty only for VP9,
    /// whose config record no decoder needs.
    pub config: Vec<u8>,
}

/// Reads the facts [`check_output`](crate::check::check_output) compares from
/// the first video track of `input`.
///
/// Frames come from the demuxed packets, in presentation order, with the track's
/// edit list applied. A track with no edit list keeps its pts as stored, so an
/// order mismatch reaches `check_output`. With an edit list, a frame's duration
/// stops at the edit's end, as [`EditTimeline::present`] states.
///
/// [`EditTimeline::present`]: crate::time::EditTimeline::present
///
/// # Errors
///
/// [`MediaError::NoVideoTrack`] for no video, and any error
/// [`Input::rotation`] or [`Input::edit_timeline`] gives.
/// [`MediaError::EditList`] when pts under an edit list do not increase.
pub fn read_stream_facts(input: &Input) -> Result<StreamFacts, MediaError> {
    let track = input.video_track()?;
    let (width, height) = coded_size(track);
    let coded = Dimensions::new(width, height).map_err(|error| MediaError::Geometry {
        track_id: track.id,
        error,
    })?;
    let geometry = Geometry::new(
        coded,
        input.rotation(track)?,
        input.sample_aspect_ratio(track).unwrap_or_default(),
    );
    let time_base = Rational::new(i64::from(track.timebase.num), u64::from(track.timebase.den))
        .map_err(|error| MediaError::Time {
            what: "the video track's time base",
            error,
        })?;
    let mut frames = Vec::new();
    let mut last_duration = None;
    for packet in input.track_packets(track.id)? {
        let packet = packet.map_err(|error| input.demux_error(&error))?;
        let duration = packet
            .duration
            .map_or(0, |ticks| i64::try_from(ticks).unwrap_or(i64::MAX));
        last_duration = Some(duration);
        frames.push(FrameTiming::new(packet.pts, packet.dts, duration));
    }
    frames.sort_by_key(|frame: &FrameTiming| frame.pts);
    if !track.edit_list.is_empty() && !frames.is_empty() {
        let timeline = input.edit_timeline(track)?;
        frames = timeline
            .present(&frames)
            .map_err(|error| MediaError::EditList {
                track_id: track.id,
                error,
            })?;
    }
    let hints = DurationHints {
        packet_duration: last_duration,
        ..DurationHints::default()
    };
    Ok(StreamFacts::new(geometry, time_base, frames, hints))
}

/// Reads the audio and subtitle tracks and the chapters of `input`, for
/// [`check_tracks`](crate::check::check_tracks). A chapter track counts as
/// chapters, and MP4's two lists of one set of chapters count once.
///
/// A text subtitle's count leaves out empty samples. MP4 fills the gaps between
/// cues with them, and Matroska has no such blocks.
///
/// # Errors
///
/// [`MediaError::Demux`] for a failed read of the packet tables.
pub fn read_track_facts(input: &Input) -> Result<TrackFacts, MediaError> {
    let mut facts = TrackFacts {
        chapters: chapter_count(input),
        ..TrackFacts::default()
    };
    for track in input.tracks() {
        if track.chapter_track {
            continue;
        }
        let list = match track.kind {
            TrackKind::Audio => &mut facts.audio,
            TrackKind::Subtitle => &mut facts.subtitles,
            TrackKind::Video | TrackKind::Data | TrackKind::Other => continue,
        };
        let mut packets = 0;
        for packet in input.track_packets(track.id)? {
            let packet = packet.map_err(|error| input.demux_error(&error))?;
            if holds_content(track, &packet) {
                packets += 1;
            }
        }
        list.push(TrackPacketCount {
            track_id: track.id,
            packets,
        });
    }
    Ok(facts)
}

fn chapter_count(input: &Input) -> usize {
    match input {
        #[cfg(feature = "mp4")]
        Input::Mp4 { .. } => containr::mp4::StreamCopy::chapters(input.chapters()).len(),
        #[cfg(feature = "mkv")]
        Input::Mkv { .. } => input.chapters().len(),
    }
}

/// True when `packet` carries content a copy must keep. An empty `tx3g` sample
/// and an empty Matroska text block hold none, and each container writes its
/// gaps its own way.
pub(super) fn holds_content(track: &Track, packet: &Packet) -> bool {
    if track.kind != TrackKind::Subtitle {
        return true;
    }
    if track.matroska.is_some() {
        return packet.size > 0;
    }
    let tx3g = track.codec().is_some_and(|fourcc| &fourcc.0 == b"tx3g");
    !tx3g || packet.size > TX3G_EMPTY_SAMPLE
}

/// The size of an empty `tx3g` sample: its 16-bit text length alone.
const TX3G_EMPTY_SAMPLE: u32 = 2;

/// A track's picture size, after a Matroska crop. A track that states none
/// gives 0, which [`Dimensions::new`] refuses.
fn coded_size(track: &Track) -> (u32, u32) {
    let (width, height) = match &track.matroska {
        Some(mkv) => mkv.video.as_ref().map_or((0, 0), |video| {
            let [top, bottom, left, right] = video.crop;
            (
                video.pixel_width.saturating_sub(left).saturating_sub(right),
                video
                    .pixel_height
                    .saturating_sub(top)
                    .saturating_sub(bottom),
            )
        }),
        None => track.video().map_or((0, 0), |video| {
            (u64::from(video.width), u64::from(video.height))
        }),
    };
    (
        u32::try_from(width).unwrap_or(0),
        u32::try_from(height).unwrap_or(0),
    )
}

pub(super) fn rotation(track: &Track) -> Result<Rotation, MediaError> {
    let Some(mkv) = &track.matroska else {
        return Rotation::from_display_matrix(track.matrix).map_err(|error| MediaError::Geometry {
            track_id: track.id,
            error,
        });
    };
    match mkv
        .video
        .as_ref()
        .and_then(|video| video.projection.as_ref())
    {
        Some(projection) => pose_rotation(track.id, projection),
        None => Ok(Rotation::Upright),
    }
}

/// FFmpeg writes a display rotation as a roll on a rectangular projection,
/// and a mirror as a yaw of 180. A roll on a spherical projection turns the
/// view, not the picture. The angle is a stored float and no timestamp.
fn pose_rotation(track_id: u32, projection: &Projection) -> Result<Rotation, MediaError> {
    let roll = projection.pose_roll;
    let flat = projection.pose_yaw == 0.0
        && projection.pose_pitch == 0.0
        && (projection.projection_type == 0 || roll == 0.0);
    if !flat || roll.fract() != 0.0 {
        return Err(MediaError::Projection {
            track_id,
            pose: format!(
                "type {}, yaw {}, pitch {}, roll {roll}",
                projection.projection_type, projection.pose_yaw, projection.pose_pitch
            ),
        });
    }
    // The roll is counterclockwise and `Rotation` is clockwise. `as` saturates
    // on a huge value, and `from_degrees` takes any i64.
    Rotation::from_degrees((-roll) as i64).map_err(|error| MediaError::Geometry { track_id, error })
}

pub(super) fn sample_aspect_ratio(track: &Track) -> Option<SampleAspectRatio> {
    match &track.matroska {
        Some(mkv) => matroska_aspect(mkv.video.as_ref()?),
        None => {
            let aspect = track.video()?.pixel_aspect?;
            SampleAspectRatio::from_fraction(
                u64::from(aspect.h_spacing),
                u64::from(aspect.v_spacing),
            )
        }
    }
}

/// FFmpeg's reading of a Matroska display size:
/// `cropped_height * display_width : cropped_width * display_height`. The
/// ratio applies whichever of the four defined units the size has.
fn matroska_aspect(video: &MatroskaVideo) -> Option<SampleAspectRatio> {
    let defined = matches!(
        video.display_unit,
        DisplayUnit::Pixels
            | DisplayUnit::Centimeters
            | DisplayUnit::Inches
            | DisplayUnit::AspectRatio
    );
    if !defined {
        return None;
    }
    let [top, bottom, left, right] = video.crop;
    let width = video.pixel_width.checked_sub(left)?.checked_sub(right)?;
    let height = video.pixel_height.checked_sub(top)?.checked_sub(bottom)?;
    SampleAspectRatio::from_fraction(
        height.checked_mul(video.display_width?)?,
        width.checked_mul(video.display_height?)?,
    )
}

pub(super) fn decode_config(track: &Track) -> Result<DecodeConfig, MediaError> {
    let (codec, config) = match &track.matroska {
        Some(mkv) => matroska_config(track.id, mkv)?,
        None => mp4_config(track)?,
    };
    // An empty config selects raw input in the decoder, and these samples are
    // not raw.
    if config.is_empty() && codec != DecodeCodec::Vp9 {
        return Err(MediaError::MissingCodecConfig {
            track_id: track.id,
            codec,
        });
    }
    Ok(DecodeConfig { codec, config })
}

fn matroska_config(
    track_id: u32,
    mkv: &MatroskaTrack,
) -> Result<(DecodeCodec, Vec<u8>), MediaError> {
    if !mkv.content_encodings.is_empty() {
        return Err(MediaError::ContentEncoding {
            track_id,
            count: mkv.content_encodings.len(),
        });
    }
    let codec = match mkv.codec_id.as_str() {
        "V_MPEG4/ISO/AVC" => DecodeCodec::H264,
        "V_MPEGH/ISO/HEVC" => DecodeCodec::Hevc,
        "V_AV1" => DecodeCodec::Av1,
        "V_VP9" => DecodeCodec::Vp9,
        other => {
            return Err(MediaError::UnsupportedCodec {
                track_id,
                codec: other.to_owned(),
            });
        }
    };
    Ok((codec, mkv.codec_private.clone().unwrap_or_default()))
}

fn mp4_config(track: &Track) -> Result<(DecodeCodec, Vec<u8>), MediaError> {
    if track.sample_entries.len() > 1 {
        return Err(MediaError::SampleDescriptions {
            track_id: track.id,
            count: track.sample_entries.len(),
        });
    }
    let unsupported = |codec: String| MediaError::UnsupportedCodec {
        track_id: track.id,
        codec,
    };
    let fourcc = track
        .codec()
        .ok_or_else(|| unsupported("no sample entry".to_owned()))?;
    let codec = match &fourcc.0 {
        b"avc1" | b"avc3" => DecodeCodec::H264,
        b"hvc1" | b"hev1" => DecodeCodec::Hevc,
        b"av01" => DecodeCodec::Av1,
        b"vp09" => DecodeCodec::Vp9,
        _ => return Err(unsupported(fourcc.to_string())),
    };
    let config = track
        .codec_config()
        .map(|record| record.bytes().to_vec())
        .unwrap_or_default();
    Ok((codec, config))
}

#[cfg(test)]
mod tests {
    use containr::matroska::ContentEncoding;

    use super::*;
    use crate::error::GeometryError;

    fn projection(projection_type: u64, yaw: f64, pitch: f64, roll: f64) -> Projection {
        Projection {
            projection_type,
            private: None,
            pose_yaw: yaw,
            pose_pitch: pitch,
            pose_roll: roll,
        }
    }

    fn video(display: Option<(u64, u64)>, unit: DisplayUnit, crop: [u64; 4]) -> MatroskaVideo {
        MatroskaVideo {
            pixel_width: 720,
            pixel_height: 480,
            display_width: display.map(|size| size.0),
            display_height: display.map(|size| size.1),
            display_unit: unit,
            crop,
            interlaced: 0,
            projection: None,
            colour: None,
            extra: Vec::new(),
        }
    }

    fn matroska(codec_id: &str, private: Option<&[u8]>) -> MatroskaTrack {
        MatroskaTrack {
            uid: None,
            track_type: 1,
            codec_id: codec_id.to_owned(),
            codec_private: private.map(<[u8]>::to_vec),
            codec_name: None,
            name: None,
            language: "eng".to_owned(),
            language_bcp47: None,
            default_duration: None,
            codec_delay: 0,
            seek_pre_roll: 0,
            enabled: true,
            default: true,
            forced: false,
            lacing: false,
            content_encodings: Vec::new(),
            video: None,
            audio: None,
            extra: Vec::new(),
        }
    }

    #[test]
    fn roll_is_counterclockwise() {
        let cases = [
            (0.0, Rotation::Upright),
            (90.0, Rotation::Clockwise270),
            (-90.0, Rotation::Clockwise90),
            (180.0, Rotation::Clockwise180),
            (-180.0, Rotation::Clockwise180),
            (270.0, Rotation::Clockwise90),
        ];
        for (roll, expected) in cases {
            assert_eq!(
                pose_rotation(1, &projection(0, 0.0, 0.0, roll)),
                Ok(expected),
                "roll {roll}"
            );
        }
    }

    #[test]
    fn a_whole_roll_off_a_right_angle_is_refused_as_a_rotation() {
        assert_eq!(
            pose_rotation(1, &projection(0, 0.0, 0.0, 45.0)),
            Err(MediaError::Geometry {
                track_id: 1,
                error: GeometryError::UnsupportedRotation { degrees: -45 },
            })
        );
    }

    #[test]
    fn huge_rolls_do_not_overflow() {
        for roll in [1e300, -1e300] {
            assert!(matches!(
                pose_rotation(1, &projection(0, 0.0, 0.0, roll)),
                Err(MediaError::Geometry { .. })
            ));
        }
    }

    #[test]
    fn other_poses_are_refused_by_name() {
        let refused = [
            projection(0, 0.0, 0.0, 89.5),
            projection(0, 0.0, 0.0, f64::NAN),
            projection(0, 0.0, 0.0, f64::INFINITY),
            projection(0, 180.0, 0.0, -90.0),
            projection(0, 0.0, 10.0, 0.0),
            projection(1, 0.0, 0.0, 90.0),
        ];
        for pose in refused {
            assert!(
                matches!(
                    pose_rotation(9, &pose),
                    Err(MediaError::Projection { track_id: 9, .. })
                ),
                "{pose:?}"
            );
        }
    }

    #[test]
    fn a_spherical_projection_with_no_pose_is_upright() {
        assert_eq!(
            pose_rotation(1, &projection(1, 0.0, 0.0, 0.0)),
            Ok(Rotation::Upright)
        );
    }

    #[test]
    fn display_size_gives_the_ratio() {
        let aspect = matroska_aspect(&video(Some((640, 480)), DisplayUnit::Pixels, [0; 4]));
        assert_eq!(aspect, Some(SampleAspectRatio::new(8, 9)));
        let aspect = matroska_aspect(&video(Some((4, 3)), DisplayUnit::AspectRatio, [0; 4]));
        assert_eq!(aspect, Some(SampleAspectRatio::new(8, 9)));
    }

    #[test]
    fn crop_shrinks_the_pixel_size() {
        // top, bottom, left, right. The picture is 640 x 480 after the crop.
        let aspect = matroska_aspect(&video(
            Some((640, 480)),
            DisplayUnit::Pixels,
            [0, 0, 40, 40],
        ));
        assert_eq!(aspect, Some(SampleAspectRatio::SQUARE));
    }

    #[test]
    fn no_ratio_for_a_missing_or_unusable_size() {
        let cases = [
            video(None, DisplayUnit::Pixels, [0; 4]),
            video(Some((640, 480)), DisplayUnit::Unknown, [0; 4]),
            video(Some((640, 480)), DisplayUnit::Other(9), [0; 4]),
            video(Some((0, 480)), DisplayUnit::Pixels, [0; 4]),
            video(Some((640, 0)), DisplayUnit::Pixels, [0; 4]),
            video(Some((640, 480)), DisplayUnit::Pixels, [0, 0, 800, 0]),
            video(Some((u64::MAX, 480)), DisplayUnit::Pixels, [0; 4]),
        ];
        for case in &cases {
            assert_eq!(matroska_aspect(case), None, "{case:?}");
        }
    }

    #[test]
    fn matroska_codec_ids_map_to_codecs() {
        let cases = [
            ("V_MPEG4/ISO/AVC", DecodeCodec::H264),
            ("V_MPEGH/ISO/HEVC", DecodeCodec::Hevc),
            ("V_AV1", DecodeCodec::Av1),
            ("V_VP9", DecodeCodec::Vp9),
        ];
        for (id, codec) in cases {
            assert_eq!(
                matroska_config(1, &matroska(id, Some(&[1, 2]))),
                Ok((codec, vec![1, 2]))
            );
        }
    }

    #[test]
    fn matroska_refuses_other_codecs_and_content_encodings() {
        assert_eq!(
            matroska_config(3, &matroska("V_MPEG2", None)),
            Err(MediaError::UnsupportedCodec {
                track_id: 3,
                codec: "V_MPEG2".to_owned(),
            })
        );
        let mut encoded = matroska("V_MPEG4/ISO/AVC", Some(&[1]));
        encoded.content_encodings.push(ContentEncoding {
            order: 0,
            scope: 1,
            encoding_type: 0,
            compression_algorithm: Some(3),
            compression_settings: None,
            encryption_algorithm: None,
            encryption_settings: None,
        });
        assert_eq!(
            matroska_config(3, &encoded),
            Err(MediaError::ContentEncoding {
                track_id: 3,
                count: 1,
            })
        );
    }
}
