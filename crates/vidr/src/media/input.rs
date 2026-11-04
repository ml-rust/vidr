//! Opening a file by its bytes, and the facts of its tracks.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[cfg(feature = "mkv")]
use containr::mkv;
#[cfg(feature = "mp4")]
use containr::mp4;
use containr::{ContainerColour, Track, TrackKind};

use super::edit;
use super::facts::{self, DecodeConfig};
use crate::error::MediaError;
use crate::geometry::{Rotation, SampleAspectRatio};
use crate::time::EditTimeline;

/// The bytes the sniff reads: the EBML magic, or a box size and type.
const HEAD_LEN: u64 = 16;
/// The EBML header ID, the first bytes of Matroska and WebM.
const EBML_MAGIC: [u8; 4] = [0x1A, 0x45, 0xDF, 0xA3];
/// The box types an MP4 or MOV file starts with. QuickTime files can omit
/// `ftyp`. A fragment file starts with `styp` or `moof`, and containr
/// refuses it by name.
const MP4_FIRST_BOXES: [&[u8; 4]; 10] = [
    b"ftyp", b"styp", b"moov", b"moof", b"mdat", b"free", b"skip", b"wide", b"pnot", b"sidx",
];

/// The container the first bytes of a file match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Mp4,
    Matroska,
}

fn sniff(head: &[u8]) -> Option<Format> {
    if head.starts_with(&EBML_MAGIC) {
        return Some(Format::Matroska);
    }
    let kind = head.get(4..8)?;
    MP4_FIRST_BOXES
        .iter()
        .any(|known| known.as_slice() == kind)
        .then_some(Format::Mp4)
}

/// An open media file.
///
/// The demuxer is public, so a caller reads packets from it directly.
#[derive(Debug)]
#[non_exhaustive]
pub enum Input {
    /// MP4 or QuickTime MOV.
    #[cfg(feature = "mp4")]
    Mp4 {
        /// The demuxer.
        demuxer: Box<mp4::Demuxer<BufReader<File>>>,
        /// The path the file opened from.
        path: PathBuf,
    },
    /// Matroska or WebM.
    #[cfg(feature = "mkv")]
    Mkv {
        /// The demuxer.
        demuxer: Box<mkv::Demuxer<BufReader<File>>>,
        /// The path the file opened from.
        path: PathBuf,
    },
}

impl Input {
    /// Opens `path`. The first bytes pick the container, never the extension.
    ///
    /// # Errors
    ///
    /// - [`MediaError::Read`] when the file does not open or read.
    /// - [`MediaError::UnknownFormat`] for bytes of no container vidr reads.
    /// - [`MediaError::ContainerNotEnabled`] for a container whose feature is
    ///   off.
    /// - [`MediaError::Fragmented`] for fragmented MP4, which containr refuses.
    /// - [`MediaError::Demux`] for any other refusal from containr.
    pub fn open(path: &Path) -> Result<Self, MediaError> {
        let read_error = |error: std::io::Error| MediaError::Read {
            path: path.to_path_buf(),
            reason: error.to_string(),
        };
        let mut file = File::open(path).map_err(read_error)?;
        let mut head = Vec::new();
        file.by_ref()
            .take(HEAD_LEN)
            .read_to_end(&mut head)
            .map_err(read_error)?;
        file.seek(SeekFrom::Start(0)).map_err(read_error)?;
        match sniff(&head) {
            Some(Format::Mp4) => open_mp4(path, file),
            Some(Format::Matroska) => open_mkv(path, file),
            None => Err(MediaError::UnknownFormat {
                path: path.to_path_buf(),
                first_bytes: head,
            }),
        }
    }

    /// The path the file opened from.
    pub fn path(&self) -> &Path {
        match self {
            #[cfg(feature = "mp4")]
            Self::Mp4 { path, .. } => path,
            #[cfg(feature = "mkv")]
            Self::Mkv { path, .. } => path,
        }
    }

    /// Every track, in file order.
    pub fn tracks(&self) -> &[Track] {
        match self {
            #[cfg(feature = "mp4")]
            Self::Mp4 { demuxer, .. } => demuxer.tracks(),
            #[cfg(feature = "mkv")]
            Self::Mkv { demuxer, .. } => demuxer.tracks(),
        }
    }

    /// The first video track in file order, with the PoC's rule.
    ///
    /// `-map 0:v:0` ignores the default flag, and so does this function. The
    /// PoC also skips an attached picture. containr lists none as a track: MP4
    /// cover art sits in metadata and Matroska pictures in attachments.
    ///
    /// # Errors
    ///
    /// [`MediaError::NoVideoTrack`], which names the input.
    pub fn video_track(&self) -> Result<&Track, MediaError> {
        let tracks = self.tracks();
        tracks
            .iter()
            .find(|track| track.kind == TrackKind::Video)
            .ok_or_else(|| MediaError::NoVideoTrack {
                path: self.path().to_path_buf(),
                tracks: tracks.len(),
            })
    }

    /// The presentation timeline of `track`'s edit list. Matroska has no edit
    /// list, so its tracks give [`EditTimeline::IDENTITY`].
    ///
    /// # Errors
    ///
    /// [`MediaError::EditList`] for a list vidr does not apply, and
    /// [`MediaError::EditDuration`] for a duration above the signed range.
    pub fn edit_timeline(&self, track: &Track) -> Result<EditTimeline, MediaError> {
        edit::edit_timeline(track)
    }

    /// The rotation that shows `track` upright: the `tkhd` display matrix, or
    /// the Matroska `ProjectionPoseRoll`, which is counterclockwise.
    ///
    /// # Errors
    ///
    /// [`MediaError::Geometry`] for a matrix or roll that is not a right-angle
    /// rotation, and [`MediaError::Projection`] for a Matroska projection that
    /// holds more than a whole-degree roll.
    pub fn rotation(&self, track: &Track) -> Result<Rotation, MediaError> {
        facts::rotation(track)
    }

    /// The sample aspect ratio the container states: the `pasp` box, or the
    /// ratio of the Matroska display size to the cropped pixel size.
    ///
    /// `None` when the container states none. The bitstream's ratio then
    /// applies.
    pub fn sample_aspect_ratio(&self, track: &Track) -> Option<SampleAspectRatio> {
        facts::sample_aspect_ratio(track)
    }

    /// The colour tags the container states for `track`. Pass them to
    /// [`merge_container`](crate::color::merge_container).
    pub fn colour(&self, track: &Track) -> Option<ContainerColour> {
        track.colour()
    }

    /// The codec and config record a decoder takes for `track`.
    ///
    /// # Errors
    ///
    /// - [`MediaError::UnsupportedCodec`] for a codec with no decoder in vidr.
    /// - [`MediaError::MissingCodecConfig`] for an H.264, HEVC, or AV1 track
    ///   with no config record.
    /// - [`MediaError::ContentEncoding`] for a Matroska track with a
    ///   `ContentEncoding`.
    /// - [`MediaError::SampleDescriptions`] for an MP4 track with more than one
    ///   sample description.
    pub fn decode_config(&self, track: &Track) -> Result<DecodeConfig, MediaError> {
        facts::decode_config(track)
    }
}

#[cfg(feature = "mp4")]
fn open_mp4(path: &Path, file: File) -> Result<Input, MediaError> {
    let demuxer = mp4::Demuxer::open(BufReader::new(file)).map_err(|e| demux_error(path, &e))?;
    Ok(Input::Mp4 {
        demuxer: Box::new(demuxer),
        path: path.to_path_buf(),
    })
}

#[cfg(not(feature = "mp4"))]
fn open_mp4(path: &Path, _file: File) -> Result<Input, MediaError> {
    Err(MediaError::ContainerNotEnabled {
        path: path.to_path_buf(),
        container: "MP4/MOV",
        feature: "mp4",
    })
}

#[cfg(feature = "mkv")]
fn open_mkv(path: &Path, file: File) -> Result<Input, MediaError> {
    let demuxer = mkv::Demuxer::open(BufReader::new(file)).map_err(|e| demux_error(path, &e))?;
    Ok(Input::Mkv {
        demuxer: Box::new(demuxer),
        path: path.to_path_buf(),
    })
}

#[cfg(not(feature = "mkv"))]
fn open_mkv(path: &Path, _file: File) -> Result<Input, MediaError> {
    Err(MediaError::ContainerNotEnabled {
        path: path.to_path_buf(),
        container: "Matroska/WebM",
        feature: "mkv",
    })
}

fn demux_error(path: &Path, error: &containr::Error) -> MediaError {
    let path = path.to_path_buf();
    let message = error.to_string();
    if matches!(error, containr::Error::Fragmented { .. }) {
        MediaError::Fragmented { path, message }
    } else {
        MediaError::Demux { path, message }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sniff_reads_the_ebml_magic() {
        assert_eq!(
            sniff(&[0x1A, 0x45, 0xDF, 0xA3, 0x01]),
            Some(Format::Matroska)
        );
    }

    #[test]
    fn sniff_reads_each_first_box() {
        for kind in MP4_FIRST_BOXES {
            let mut head = vec![0, 0, 0, 8];
            head.extend_from_slice(kind);
            assert_eq!(sniff(&head), Some(Format::Mp4), "{kind:?}");
        }
    }

    #[test]
    fn sniff_refuses_other_bytes() {
        assert_eq!(sniff(b""), None);
        assert_eq!(sniff(b"\x1A\x45"), None);
        assert_eq!(sniff(b"RIFF\0\0\0\0AVI "), None);
        assert_eq!(sniff(b"\0\0\0\x08fty"), None);
    }
}
