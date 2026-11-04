//! Errors in media input and output.

use std::error::Error as StdError;
use std::fmt;
use std::path::PathBuf;

use super::{EditListError, GeometryError, TimeError};
use crate::decode::DecodeCodec;
use crate::media::OutputKind;

/// An error in opening a media file, reading a fact from it, or writing one.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MediaError {
    /// The file does not open or read.
    Read {
        /// The input.
        path: PathBuf,
        /// The system's message.
        reason: String,
    },
    /// The first bytes match no container vidr reads.
    UnknownFormat {
        /// The input.
        path: PathBuf,
        /// The bytes the sniff read. Empty for an empty file.
        first_bytes: Vec<u8>,
    },
    /// The bytes are a container this build has no feature for.
    ContainerNotEnabled {
        /// The input.
        path: PathBuf,
        /// The container the bytes match.
        container: &'static str,
        /// The Cargo feature that reads it.
        feature: &'static str,
    },
    /// containr refuses the file.
    Demux {
        /// The input.
        path: PathBuf,
        /// containr's message.
        message: String,
    },
    /// containr refuses the file as fragmented MP4.
    Fragmented {
        /// The input.
        path: PathBuf,
        /// containr's message.
        message: String,
    },
    /// The file has no video track.
    NoVideoTrack {
        /// The input.
        path: PathBuf,
        /// The number of tracks of other kinds.
        tracks: usize,
    },
    /// An edit's duration does not fit vidr's signed ticks.
    EditDuration {
        /// The track id.
        track_id: u32,
        /// The index of the edit.
        index: usize,
        /// The duration, in movie ticks.
        segment_duration: u64,
    },
    /// A track's edit list is outside what vidr applies.
    EditList {
        /// The track id.
        track_id: u32,
        /// The reason.
        error: EditListError,
    },
    /// A track's display matrix or roll is not a right-angle rotation.
    Geometry {
        /// The track id.
        track_id: u32,
        /// The reason.
        error: GeometryError,
    },
    /// A Matroska `Projection` is more than a whole-degree roll of a flat
    /// picture.
    Projection {
        /// The track id.
        track_id: u32,
        /// The projection type and pose, as text.
        pose: String,
    },
    /// A track's codec has no decoder in vidr.
    UnsupportedCodec {
        /// The track id.
        track_id: u32,
        /// The sample entry type or `CodecID`.
        codec: String,
    },
    /// A track lacks the config record its samples need.
    MissingCodecConfig {
        /// The track id.
        track_id: u32,
        /// The codec.
        codec: DecodeCodec,
    },
    /// A Matroska track's packets or config carry a `ContentEncoding`.
    ContentEncoding {
        /// The track id.
        track_id: u32,
        /// The number of encodings.
        count: usize,
    },
    /// An MP4 track holds more than one sample description.
    SampleDescriptions {
        /// The track id.
        track_id: u32,
        /// The number of sample descriptions.
        count: usize,
    },
    /// An output path has an extension vidr does not write.
    OutputExtension {
        /// The output.
        path: PathBuf,
    },
    /// An output kind needs a feature this build does not enable.
    OutputNotEnabled {
        /// The output kind.
        kind: OutputKind,
        /// The Cargo feature that writes it.
        feature: &'static str,
    },
    /// A Matroska source holds attachments, and MP4 has no form for them.
    Attachments {
        /// The source.
        path: PathBuf,
        /// The number of attachments.
        count: usize,
        /// The first attachment's file name.
        first: String,
    },
    /// The output cannot hold a stream, or containr refuses its copy.
    StreamRefused {
        /// The output kind.
        output: OutputKind,
        /// The stream or item, such as ``audio track 2 (codec `A_FLAC`)``.
        stream: String,
        /// The output kind that holds it, when containr names one.
        use_instead: Option<OutputKind>,
        /// containr's message.
        message: String,
    },
    /// containr refuses a write to the output.
    Mux {
        /// The output kind.
        output: OutputKind,
        /// containr's message.
        message: String,
    },
    /// A field of [`VideoOutput`](crate::media::VideoOutput) is out of range.
    VideoOutput {
        /// The field.
        field: &'static str,
        /// The value, as text.
        value: String,
        /// Why the value is refused, and the next action.
        reason: &'static str,
    },
    /// An encoded video packet breaks a rule of the output clock.
    VideoPacket {
        /// The packet's pts.
        pts: i64,
        /// The packet's dts.
        dts: i64,
        /// The packet's duration.
        duration: i64,
        /// What the packet breaks, and the next action.
        reason: &'static str,
    },
    /// An output finished with no video packet.
    NoVideoPackets {
        /// The output kind.
        output: OutputKind,
    },
    /// Exact time arithmetic on the output clock failed.
    Time {
        /// The value that failed.
        what: &'static str,
        /// The reason.
        error: TimeError,
    },
}

impl fmt::Display for MediaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, reason } => write!(
                f,
                "reading {} failed: {reason}. Check that the file exists and is readable.",
                path.display()
            ),
            Self::UnknownFormat { path, first_bytes } => {
                write!(f, "{} starts with bytes [", path.display())?;
                for (index, byte) in first_bytes.iter().enumerate() {
                    let space = if index == 0 { "" } else { " " };
                    write!(f, "{space}{byte:02x}")?;
                }
                write!(
                    f,
                    "], which match no container vidr reads. vidr reads MP4/MOV and \
                     Matroska/WebM. Remux the file into one of these."
                )
            }
            Self::ContainerNotEnabled {
                path,
                container,
                feature,
            } => write!(
                f,
                "{} is {container}, and this build of vidr does not read it. \
                 Enable the `{feature}` feature.",
                path.display()
            ),
            Self::Demux { path, message } | Self::Fragmented { path, message } => {
                write!(f, "{}: {message}", path.display())
            }
            Self::NoVideoTrack { path, tracks } => write!(
                f,
                "{} has {tracks} track(s) and no video track. Choose a file with a video track.",
                path.display()
            ),
            Self::EditDuration {
                track_id,
                index,
                segment_duration,
            } => write!(
                f,
                "track {track_id}: edit {index} lasts {segment_duration} movie ticks, above \
                 the largest signed 64-bit tick. The input is damaged. Remux it without the \
                 edit list."
            ),
            Self::EditList { track_id, error } => write!(f, "track {track_id}: {error}"),
            Self::Geometry { track_id, error } => write!(f, "track {track_id}: {error}"),
            Self::Projection { track_id, pose } => write!(
                f,
                "track {track_id}: Matroska projection with {pose} is not a whole-degree roll \
                 of a flat picture. Remux the source without yaw, pitch, or a spherical \
                 projection."
            ),
            Self::UnsupportedCodec { track_id, codec } => write!(
                f,
                "track {track_id} has codec `{codec}`, which vidr does not decode. vidr decodes \
                 H.264, HEVC, AV1, and VP9. Transcode the track to one of these."
            ),
            Self::MissingCodecConfig { track_id, codec } => write!(
                f,
                "track {track_id} holds {codec} with no config record. The input is damaged. \
                 Remux it with `ffmpeg -i <input> -c copy <output.mp4>`."
            ),
            Self::ContentEncoding { track_id, count } => write!(
                f,
                "track {track_id} carries {count} Matroska ContentEncoding(s), which vidr does \
                 not undo. Remux it with `ffmpeg -i <input> -c copy <output.mkv>`."
            ),
            Self::SampleDescriptions { track_id, count } => write!(
                f,
                "track {track_id} holds {count} sample descriptions, and a decoder takes one \
                 config record. Split the source at the change, or transcode it."
            ),
            Self::OutputExtension { path } => write!(
                f,
                "output {} has an extension vidr does not write. vidr writes `.mp4` and `.mkv`. \
                 Name the output with one of these.",
                path.display()
            ),
            Self::OutputNotEnabled { kind, feature } => write!(
                f,
                "this build of vidr does not write {kind} output. Enable the `{feature}` feature."
            ),
            Self::Attachments { path, count, first } => write!(
                f,
                "{} holds {count} attachment(s), the first named `{first}`, and MP4 has no form \
                 for attachments. Write MKV output (`.mkv`), which holds them.",
                path.display()
            ),
            Self::StreamRefused {
                output,
                stream,
                use_instead,
                message,
            } => {
                write!(f, "{output} output cannot copy {stream}: {message}")?;
                match use_instead {
                    Some(kind) => write!(
                        f,
                        ". Write {kind} output (`.{}`), which holds it.",
                        kind.extension()
                    ),
                    None => Ok(()),
                }
            }
            Self::Mux { output, message } => write!(
                f,
                "writing {output} output failed: {message}. Check the target, then start again \
                 on a new file."
            ),
            Self::VideoOutput {
                field,
                value,
                reason,
            } => write!(f, "`VideoOutput::{field}` is {value}: {reason}"),
            Self::VideoPacket {
                pts,
                dts,
                duration,
                reason,
            } => write!(
                f,
                "video packet with pts {pts}, dts {dts} and duration {duration}: {reason}"
            ),
            Self::NoVideoPackets { output } => write!(
                f,
                "the {output} output finished with no video packet. Call \
                 `OutputMux::write_video` for each encoded packet."
            ),
            Self::Time { what, error } => write!(f, "{what}: {error}"),
        }
    }
}

impl StdError for MediaError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::EditList { error, .. } => Some(error),
            Self::Geometry { error, .. } => Some(error),
            Self::Time { error, .. } => Some(error),
            _ => None,
        }
    }
}
