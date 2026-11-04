//! What an output copies from its source, decided before any GPU work.
//!
//! [`plan_copy`] runs every `StreamCopy` plan, then a dry run of the whole mux
//! with no audio bytes, so every refusal containr can state comes before the
//! first frame decodes. The plan holds packet timing per copied track and no
//! packet bytes, and it reads the source through its own file handle.

use std::ffi::OsStr;
use std::fmt;
use std::path::Path;

use containr::{Chapter, Container, MetadataItem, Packet, Track, TrackKind};

use super::facts::holds_content;
use super::input::Input;
#[cfg(feature = "mkv")]
use super::output_mkv::{self, MkvCopies};
#[cfg(feature = "mp4")]
use super::output_mp4::{self, Mp4Copies};
use crate::check::{TrackFacts, TrackPacketCount};
use crate::error::MediaError;

/// The container an output file holds.
///
/// vidr writes the two containers the PoC takes. WebM is not an output: its
/// codec rules refuse H.264 and HEVC, the codecs most outputs use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OutputKind {
    /// MP4, with `moov` before `mdat`.
    Mp4,
    /// Matroska.
    Mkv,
}

impl OutputKind {
    /// Picks the kind from the extension of `path`, in any case.
    ///
    /// # Errors
    ///
    /// [`MediaError::OutputExtension`] for any extension other than `.mp4` and
    /// `.mkv`, and [`MediaError::OutputNotEnabled`] when the kind's feature is
    /// off.
    pub fn from_path(path: &Path) -> Result<Self, MediaError> {
        let extension = path
            .extension()
            .and_then(OsStr::to_str)
            .map(str::to_ascii_lowercase);
        let kind = match extension.as_deref() {
            Some("mp4") => Self::Mp4,
            Some("mkv") => Self::Mkv,
            _ => {
                return Err(MediaError::OutputExtension {
                    path: path.to_path_buf(),
                });
            }
        };
        kind.check_enabled()?;
        Ok(kind)
    }

    /// The file extension, with no dot.
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Mkv => "mkv",
        }
    }

    fn check_enabled(self) -> Result<(), MediaError> {
        let (enabled, feature) = match self {
            Self::Mp4 => (cfg!(feature = "mp4"), "mp4"),
            Self::Mkv => (cfg!(feature = "mkv"), "mkv"),
        };
        if enabled {
            Ok(())
        } else {
            Err(MediaError::OutputNotEnabled {
                kind: self,
                feature,
            })
        }
    }
}

impl fmt::Display for OutputKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Mp4 => "MP4",
            Self::Mkv => "MKV",
        })
    }
}

/// A source track the output leaves out, as the PoC's `-map` does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DroppedTrack {
    /// The source track id.
    pub track_id: u32,
    /// What the track carries.
    pub kind: TrackKind,
    /// The sample entry type or `CodecID`.
    pub codec: String,
}

/// One copied track's source, in output order.
#[derive(Clone, Debug)]
pub(super) struct CopiedStream {
    pub(super) track_id: u32,
    pub(super) kind: TrackKind,
    /// The stream's name in messages.
    pub(super) label: String,
    /// The packets the output must hold for it.
    pub(super) packets: u64,
}

/// The copy plan of one output kind.
#[derive(Debug)]
pub(super) enum Copies {
    #[cfg(feature = "mp4")]
    Mp4(Mp4Copies),
    #[cfg(feature = "mkv")]
    Mkv(MkvCopies),
}

/// What one output kind builds from the source.
pub(super) struct Built {
    pub(super) copies: Copies,
    pub(super) streams: Vec<CopiedStream>,
    pub(super) metadata: Vec<MetadataItem>,
    pub(super) chapters: Vec<Chapter>,
}

/// The file-level facts a writer starts from.
pub(super) struct MovieFacts {
    pub(super) metadata: Vec<MetadataItem>,
    pub(super) chapters: Vec<Chapter>,
    pub(super) max_buffer: Option<usize>,
}

/// What an output copies, checked against the output container.
#[derive(Debug)]
pub struct CopyPlan {
    pub(super) copies: Copies,
    pub(super) streams: Vec<CopiedStream>,
    pub(super) source: Input,
    pub(super) metadata: Vec<MetadataItem>,
    pub(super) chapters: Vec<Chapter>,
    pub(super) max_buffer: Option<usize>,
    dropped: Vec<DroppedTrack>,
}

impl CopyPlan {
    /// The output kind.
    pub fn kind(&self) -> OutputKind {
        match self.copies {
            #[cfg(feature = "mp4")]
            Copies::Mp4(_) => OutputKind::Mp4,
            #[cfg(feature = "mkv")]
            Copies::Mkv(_) => OutputKind::Mkv,
        }
    }

    /// The source tracks the output leaves out: extra video tracks, data
    /// tracks, and `tmcd`. A chapter track is not listed, since the chapters
    /// copy.
    pub fn dropped(&self) -> &[DroppedTrack] {
        &self.dropped
    }

    /// The largest error of any rounded time in a copied track, in
    /// nanoseconds, rounded up.
    ///
    /// `Some` for MP4 to Matroska, whose copied tracks round to a 1 ms
    /// `TimestampScale`, so the value is at most 500 000. `None` when no copied
    /// track rounds. Encoded video pts round apart from this value:
    /// [`VideoOutput`](super::VideoOutput) states their tolerance.
    pub fn max_rounding_error_ns(&self) -> Option<u64> {
        match &self.copies {
            #[cfg(feature = "mp4")]
            Copies::Mp4(_) => None,
            #[cfg(feature = "mkv")]
            Copies::Mkv(copies) => copies.max_rounding_error_ns,
        }
    }

    /// The tracks and chapters the output must hold, for
    /// [`check_tracks`](crate::check::check_tracks). Dropped tracks are not
    /// in it.
    pub fn expected_tracks(&self) -> TrackFacts {
        let counts = |kind: TrackKind| {
            self.streams
                .iter()
                .filter(|stream| stream.kind == kind)
                .map(|stream| TrackPacketCount {
                    track_id: stream.track_id,
                    packets: stream.packets,
                })
                .collect()
        };
        TrackFacts {
            audio: counts(TrackKind::Audio),
            subtitles: counts(TrackKind::Subtitle),
            chapters: self.chapters.len(),
        }
    }

    /// Sets the bound on packet bytes the muxer holds, in place of containr's
    /// default of 64 MiB. MP4 output needs at least
    /// `containr::mp4::MAX_CHUNK_BYTES`. `OutputMux::start` refuses less.
    #[must_use]
    pub fn with_max_buffer(mut self, bytes: usize) -> Self {
        self.max_buffer = Some(bytes);
        self
    }
}

/// Plans the copy of every stream of `input` but its main video track.
///
/// Audio, text subtitles, chapters, global metadata, and each copied track's
/// metadata copy. Nothing transcodes. Subtitles convert only where containr
/// maps them: `S_TEXT/UTF8` and `tx3g`. MP4 to MKV rounds each copied track to
/// a 1 ms `TimestampScale`, and [`CopyPlan::max_rounding_error_ns`] reports the
/// error. Chapters stay exact: a chapter time with no whole nanosecond is
/// refused and names MP4. Matroska to Matroska copies exactly.
///
/// The main video's own metadata does not copy, as in the PoC, whose video
/// comes from the encoder. The plan opens `input`'s path again, so the caller
/// keeps `input` for decoding.
///
/// # Errors
///
/// - [`MediaError::NoVideoTrack`] for a source with no video.
/// - [`MediaError::OutputNotEnabled`] when the kind's feature is off.
/// - [`MediaError::Attachments`] for Matroska attachments into MP4.
/// - [`MediaError::StreamRefused`] for a stream, metadata item, or chapter the
///   output cannot hold, with the output kind that can.
pub fn plan_copy(input: &mut Input, kind: OutputKind) -> Result<CopyPlan, MediaError> {
    kind.check_enabled()?;
    let main = input.video_track()?.clone();
    refuse_attachments(input, kind)?;
    let (copied, dropped) = split_tracks(input.tracks(), main.id);
    let built = match kind {
        OutputKind::Mp4 => build_mp4(input, &copied, main.id)?,
        OutputKind::Mkv => build_mkv(input, &copied, &main)?,
    };
    let source = Input::open(input.path())?;
    Ok(CopyPlan {
        copies: built.copies,
        streams: built.streams,
        source,
        metadata: built.metadata,
        chapters: built.chapters,
        max_buffer: None,
        dropped,
    })
}

#[cfg(feature = "mp4")]
fn build_mp4(input: &mut Input, copied: &[Track], main: u32) -> Result<Built, MediaError> {
    output_mp4::build(input, copied, main)
}

#[cfg(not(feature = "mp4"))]
fn build_mp4(_input: &mut Input, _copied: &[Track], _main: u32) -> Result<Built, MediaError> {
    Err(MediaError::OutputNotEnabled {
        kind: OutputKind::Mp4,
        feature: "mp4",
    })
}

#[cfg(feature = "mkv")]
fn build_mkv(input: &mut Input, copied: &[Track], main: &Track) -> Result<Built, MediaError> {
    output_mkv::build(input, copied, main)
}

#[cfg(not(feature = "mkv"))]
fn build_mkv(_input: &mut Input, _copied: &[Track], _main: &Track) -> Result<Built, MediaError> {
    Err(MediaError::OutputNotEnabled {
        kind: OutputKind::Mkv,
        feature: "mkv",
    })
}

fn refuse_attachments(input: &Input, kind: OutputKind) -> Result<(), MediaError> {
    if kind != OutputKind::Mp4 {
        return Ok(());
    }
    let names = input.attachment_names();
    match names.first() {
        Some(first) => Err(MediaError::Attachments {
            path: input.path().to_path_buf(),
            count: names.len(),
            first: (*first).to_owned(),
        }),
        None => Ok(()),
    }
}

/// Splits the tracks into those that copy and those the output drops. The main
/// video goes to the encoder, and a chapter track becomes chapters.
fn split_tracks(tracks: &[Track], main: u32) -> (Vec<Track>, Vec<DroppedTrack>) {
    let mut copied = Vec::new();
    let mut dropped = Vec::new();
    for track in tracks {
        if track.id == main || track.chapter_track {
            continue;
        }
        match track.kind {
            TrackKind::Audio | TrackKind::Subtitle => copied.push(track.clone()),
            TrackKind::Video | TrackKind::Data | TrackKind::Other => dropped.push(DroppedTrack {
                track_id: track.id,
                kind: track.kind,
                codec: codec_name(track),
            }),
        }
    }
    (copied, dropped)
}

fn codec_name(track: &Track) -> String {
    track
        .codec_id()
        .map(str::to_owned)
        .or_else(|| track.codec().map(|fourcc| fourcc.to_string()))
        .unwrap_or_else(|| "unknown".to_owned())
}

/// A stream's name in messages, such as ``audio track 2 (codec `A_FLAC`)``.
fn stream_label(track: &Track) -> String {
    let kind = match track.kind {
        TrackKind::Video => "video",
        TrackKind::Audio => "audio",
        TrackKind::Subtitle => "subtitle",
        TrackKind::Data => "data",
        TrackKind::Other => "other",
    };
    format!("{kind} track {} (codec `{}`)", track.id, codec_name(track))
}

/// Runs `plan_one` over each copied track. Each track's packet list lives for
/// its own plan and no longer.
pub(super) fn plan_tracks<P>(
    input: &Input,
    copied: &[Track],
    kind: OutputKind,
    mut plan_one: impl FnMut(&Track, &[Packet]) -> containr::Result<P>,
) -> Result<(Vec<P>, Vec<CopiedStream>), MediaError> {
    let mut plans = Vec::with_capacity(copied.len());
    let mut streams = Vec::with_capacity(copied.len());
    for track in copied {
        let label = stream_label(track);
        let packets = input
            .track_packets(track.id)?
            .collect::<containr::Result<Vec<Packet>>>()
            .map_err(|error| input.demux_error(&error))?;
        let plan = plan_one(track, &packets).map_err(|error| refused(kind, &label, &error))?;
        plans.push(plan);
        let held = packets
            .iter()
            .filter(|packet| holds_content(track, packet))
            .count();
        streams.push(CopiedStream {
            track_id: track.id,
            kind: track.kind,
            label,
            packets: u64::try_from(held).unwrap_or(u64::MAX),
        });
    }
    Ok((plans, streams))
}

/// A containr refusal of `stream`. containr's `Unsupported` names the container
/// that holds it.
pub(super) fn refused(output: OutputKind, stream: &str, error: &containr::Error) -> MediaError {
    let use_instead = match error {
        containr::Error::Unsupported { use_instead, .. } => match use_instead {
            Container::Mp4 => Some(OutputKind::Mp4),
            Container::Matroska | Container::WebM => Some(OutputKind::Mkv),
            _ => None,
        },
        _ => None,
    };
    MediaError::StreamRefused {
        output,
        stream: stream.to_owned(),
        use_instead,
        message: error.to_string(),
    }
}

/// A refusal of the dry run's last step. `MissingDuration` names its copied
/// track, which the dry run numbers from 1. Any other error names `fallback`.
pub(super) fn finish_refused(
    output: OutputKind,
    streams: &[CopiedStream],
    error: &containr::Error,
    fallback: &str,
) -> MediaError {
    let stream = match error {
        containr::Error::MissingDuration { track_id, .. } => usize::try_from(*track_id)
            .ok()
            .and_then(|id| streams.get(id.saturating_sub(1)))
            .map_or(fallback, |stream| stream.label.as_str()),
        _ => fallback,
    };
    refused(output, stream, error)
}

/// A failed write to the output.
pub(super) fn mux_error(output: OutputKind, error: &containr::Error) -> MediaError {
    MediaError::Mux {
        output,
        message: error.to_string(),
    }
}
