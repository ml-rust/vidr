//! The Matroska output: the copy plan, its dry run, and the writer.
//!
//! The muxer writes blocks in time order across tracks. A track with an empty
//! queue holds the others back until the buffer passes
//! `MkvOptions::max_buffer`, and the earliest block then goes out. One Cluster
//! stays in memory until it closes at `MkvOptions::cluster_bytes`. Attachments
//! read at the finish, one at a time, and the muxer holds them until it writes
//! the tail.

use std::io::{Cursor, Seek, Write};

use containr::mkv::{self, AttachmentIn, MkvOptions, PacketIn};
use containr::{Chapter, ChapterSource, MetadataItem, Packet, Timebase, Track, TrackKind};

use super::clock::MkvClock;
use super::copied::{CopiedTrack, StreamClock, StreamSink, pump};
use super::input::Input;
use super::plan::{
    Built, CopiedStream, Copies, MovieFacts, OutputKind, finish_refused, mux_error, plan_tracks,
    refused,
};
use super::video::VideoOutput;
use crate::encode::EncodedPacket;
use crate::error::MediaError;
use crate::time::Rational;

const KIND: OutputKind = OutputKind::Mkv;
/// The `TimestampScale` an MP4 copy rounds to: 1 ms, the Matroska default.
const ROUNDED_SCALE: u64 = mkv::DEFAULT_TIMESTAMP_SCALE;

/// The copy plan of a Matroska output.
#[derive(Debug)]
pub(super) struct MkvCopies {
    plans: Vec<mkv::StreamCopy>,
    timestamp_scale: u64,
    pub(super) max_rounding_error_ns: Option<u64>,
    title: Option<String>,
    attachments: Vec<mkv::Attachment>,
    /// The source video's `TrackUID`, when the source has one. No copied track
    /// repeats it, so a tag target stays unique.
    video_uid: Option<u64>,
}

/// Plans each copied track, then dry-runs the mux.
///
/// A Matroska track copies exactly. An MP4 track rounds to [`ROUNDED_SCALE`].
pub(super) fn build(
    input: &mut Input,
    copied: &[Track],
    main: &Track,
) -> Result<Built, MediaError> {
    let (plans, streams) = plan_tracks(input, copied, KIND, |track, packets| {
        if track.matroska.is_some() {
            mkv::StreamCopy::plan(track, packets)
        } else {
            mkv::StreamCopy::plan_rounded(track, packets, ROUNDED_SCALE)
        }
    })?;
    let timestamp_scale = mkv::StreamCopy::timestamp_scale(&plans);
    let max_rounding_error_ns = plans
        .iter()
        .filter_map(mkv::StreamCopy::rounding)
        .map(|rounding| rounding.max_error_ns)
        .max();
    let metadata = mkv::tags_from_mp4(input.metadata())
        .map_err(|error| refused(KIND, "the global metadata", &error))?;
    let chapters = chapters_of(input);
    let copies = MkvCopies {
        plans,
        timestamp_scale,
        max_rounding_error_ns,
        title: input.title().map(str::to_owned),
        attachments: input.attachments().to_vec(),
        video_uid: main.matroska.as_ref().and_then(|mkv| mkv.uid),
    };
    dry_run(input, &copies, &streams, &metadata, &chapters)?;
    Ok(Built {
        copies: Copies::Mkv(copies),
        streams,
        metadata,
        chapters,
    })
}

/// The chapters to write. An MP4 file can list one set twice. `chpl` counts in
/// 100 ns, so its list always converts exactly. A chapter track's list can fall
/// between nanoseconds, and `Muxer::set_chapters` refuses that.
fn chapters_of(input: &Input) -> Vec<Chapter> {
    let chapters = input.chapters();
    let chpl: Vec<Chapter> = chapters
        .iter()
        .filter(|chapter| chapter.source == ChapterSource::Chpl)
        .cloned()
        .collect();
    if chpl.is_empty() {
        chapters.to_vec()
    } else {
        chpl
    }
}

fn options(copies: &MkvCopies, max_buffer: Option<usize>) -> MkvOptions {
    let defaults = MkvOptions::default();
    let max_buffer = max_buffer.unwrap_or(defaults.max_buffer);
    MkvOptions {
        timestamp_scale: copies.timestamp_scale,
        cluster_bytes: defaults.cluster_bytes.min(max_buffer),
        max_buffer,
        title: copies.title.clone(),
        ..defaults
    }
}

fn apply_file_facts<W: Write + Seek>(
    muxer: &mut mkv::Muxer<W>,
    metadata: Vec<MetadataItem>,
    chapters: Vec<Chapter>,
) -> Result<(), MediaError> {
    muxer
        .set_metadata(metadata)
        .map_err(|error| refused(KIND, "the global metadata", &error))?;
    muxer
        .set_chapters(chapters)
        .map_err(|error| refused(KIND, "the chapters", &error))
}

fn attachment_in<'a>(file: &'a mkv::Attachment, data: &'a [u8]) -> AttachmentIn<'a> {
    AttachmentIn {
        name: &file.name,
        mime: &file.mime,
        description: file.description.as_deref(),
        uid: Some(file.uid).filter(|&uid| uid != 0),
        data,
    }
}

/// Runs the whole mux on memory: audio packets with no bytes, text packets
/// with their bytes, whose framing can refuse, and empty attachments that keep
/// their UIDs for the tag targets. Nothing in the real mux can then refuse a
/// copied stream, short of a failed write.
fn dry_run(
    input: &mut Input,
    copies: &MkvCopies,
    streams: &[CopiedStream],
    metadata: &[MetadataItem],
    chapters: &[Chapter],
) -> Result<(), MediaError> {
    let mut muxer = mkv::Muxer::new(Cursor::new(Vec::new()), options(copies, None))
        .map_err(|e| mux_error(KIND, &e))?;
    apply_file_facts(&mut muxer, metadata.to_vec(), chapters.to_vec())?;
    for file in &copies.attachments {
        muxer
            .attach(&attachment_in(file, &[]))
            .map_err(|error| refused(KIND, &file.name, &error))?;
    }
    let mut ids = Vec::with_capacity(streams.len());
    for (plan, stream) in copies.plans.iter().zip(streams) {
        let id = muxer
            .add_track(plan.spec.clone())
            .map_err(|error| refused(KIND, &stream.label, &error))?;
        ids.push(id);
    }
    let mut buffer = Vec::new();
    for ((plan, stream), id) in copies.plans.iter().zip(streams).zip(ids) {
        for packet in input.track_packets(stream.track_id)? {
            let packet = packet.map_err(|error| input.demux_error(&error))?;
            buffer.clear();
            if stream.kind == TrackKind::Subtitle {
                input.read_packet_into(&packet, &mut buffer)?;
            }
            let written = plan
                .packet(&packet, &buffer)
                .and_then(|packet_in| match packet_in {
                    Some(packet_in) => muxer.write_packet(id, &packet_in),
                    None => Ok(()),
                });
            written.map_err(|error| refused(KIND, &stream.label, &error))?;
        }
    }
    muxer
        .finish()
        .map_err(|error| finish_refused(KIND, streams, &error, "a copied track or tag"))?;
    Ok(())
}

/// One copied track's plan and output id.
#[derive(Debug)]
struct MkvCopy {
    copy: mkv::StreamCopy,
    id: mkv::TrackId,
}

impl StreamClock for MkvCopy {
    fn decode_time(&self, packet: &Packet) -> Result<Option<Rational>, MediaError> {
        let timing = usize::try_from(packet.index)
            .ok()
            .and_then(|index| self.copy.timing().get(index));
        // A packet past the plan writes first, so `StreamCopy::packet` names it.
        let Some(timing) = timing else {
            return Ok(Some(Rational::ZERO));
        };
        let Some(timing) = timing else {
            return Ok(None);
        };
        tick_time(self.copy.spec.timebase, timing.pts).map(Some)
    }
}

/// `ticks` of `timebase`, in seconds.
fn tick_time(timebase: Timebase, ticks: i64) -> Result<Rational, MediaError> {
    let time_error = |error| MediaError::Time {
        what: "the time of a copied packet",
        error,
    };
    let tick =
        Rational::new(i64::from(timebase.num), u64::from(timebase.den)).map_err(time_error)?;
    Rational::from_ticks(ticks, tick).map_err(time_error)
}

impl<W: Write + Seek> StreamSink<mkv::Muxer<W>> for MkvCopy {
    fn write(
        &self,
        muxer: &mut mkv::Muxer<W>,
        packet: &Packet,
        data: &[u8],
    ) -> Result<(), MediaError> {
        let written = self
            .copy
            .packet(packet, data)
            .and_then(|packet_in| match packet_in {
                Some(packet_in) => muxer.write_packet(self.id, &packet_in),
                None => Ok(()),
            });
        written.map_err(|e| mux_error(KIND, &e))
    }
}

/// The running Matroska mux.
#[derive(Debug)]
pub(super) struct MkvWriter<W> {
    muxer: mkv::Muxer<W>,
    video: mkv::TrackId,
    clock: MkvClock,
    copies: Vec<CopiedTrack<MkvCopy>>,
    attachments: Vec<mkv::Attachment>,
    buffer_bound: usize,
}

impl<W: Write + Seek> MkvWriter<W> {
    /// Writes the file start and adds every track, the video first.
    pub(super) fn start(
        writer: W,
        copies: MkvCopies,
        streams: &[CopiedStream],
        source: &Input,
        facts: MovieFacts,
        video: &VideoOutput,
    ) -> Result<Self, MediaError> {
        let options = options(&copies, facts.max_buffer);
        let buffer_bound = options.max_buffer.saturating_add(options.cluster_bytes);
        let mut muxer = mkv::Muxer::new(writer, options).map_err(|e| mux_error(KIND, &e))?;
        let scale = u32::try_from(copies.timestamp_scale).map_err(|_| MediaError::Mux {
            output: KIND,
            message: format!(
                "the `TimestampScale` {} does not fit 32 bits",
                copies.timestamp_scale
            ),
        })?;
        let spec = video.mkv_spec(Timebase::from_nanoseconds(scale), copies.video_uid)?;
        let video_id = muxer.add_track(spec).map_err(|e| mux_error(KIND, &e))?;
        apply_file_facts(&mut muxer, facts.metadata, facts.chapters)?;
        let mut copied = Vec::with_capacity(copies.plans.len());
        for (copy, stream) in copies.plans.into_iter().zip(streams) {
            let id = muxer
                .add_track(copy.spec.clone())
                .map_err(|e| mux_error(KIND, &e))?;
            let packets = source.track_packets(stream.track_id)?;
            copied.push(CopiedTrack::new(packets, MkvCopy { copy, id }));
        }
        Ok(Self {
            muxer,
            video: video_id,
            clock: MkvClock::new(video, copies.timestamp_scale)?,
            copies: copied,
            attachments: copies.attachments,
            buffer_bound,
        })
    }

    /// Writes `packet`, and copies the streams up to its time.
    pub(super) fn write_video(
        &mut self,
        packet: &EncodedPacket,
        source: &mut Input,
        buffer: &mut Vec<u8>,
    ) -> Result<(), MediaError> {
        let (pts, duration) = self.clock.place(packet)?;
        self.muxer
            .write_packet(
                self.video,
                &PacketIn {
                    data: &packet.data,
                    pts,
                    duration: Some(duration),
                    sync: packet.keyframe,
                    discardable: false,
                    discard_padding: None,
                },
            )
            .map_err(|e| mux_error(KIND, &e))?;
        let up_to = Rational::from_ticks(pts, self.clock.output_base()).map_err(|error| {
            MediaError::Time {
                what: "the output time of a video packet",
                error,
            }
        })?;
        self.copy_pending(up_to, source, buffer)
    }

    /// Copies every packet at or before `up_to` seconds.
    pub(super) fn copy_pending(
        &mut self,
        up_to: Rational,
        source: &mut Input,
        buffer: &mut Vec<u8>,
    ) -> Result<(), MediaError> {
        pump(
            &mut self.copies,
            &mut self.muxer,
            source,
            buffer,
            Some(up_to),
        )
    }

    /// The packet bytes in memory.
    pub(super) fn buffered_bytes(&self) -> usize {
        self.muxer.buffered_bytes()
    }

    /// The bound on [`MkvWriter::buffered_bytes`], short of one block.
    pub(super) const fn buffer_bound(&self) -> usize {
        self.buffer_bound
    }

    /// Writes every copied packet left, the attachments, and the file's end.
    pub(super) fn finish(
        mut self,
        source: &mut Input,
        buffer: &mut Vec<u8>,
    ) -> Result<W, MediaError> {
        if !self.clock.started() {
            return Err(MediaError::NoVideoPackets { output: KIND });
        }
        pump(&mut self.copies, &mut self.muxer, source, buffer, None)?;
        for file in &self.attachments {
            let data = source.read_attachment(file)?;
            self.muxer
                .attach(&attachment_in(file, &data))
                .map_err(|e| mux_error(KIND, &e))?;
        }
        self.muxer.finish().map_err(|e| mux_error(KIND, &e))
    }
}
