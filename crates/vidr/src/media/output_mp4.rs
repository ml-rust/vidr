//! The MP4 output: the copy plan, its dry run, and the writer.
//!
//! The muxer writes each track's chunks in decode order across tracks, within
//! `MuxOptions::max_buffer`. A sparse subtitle track holds the others back
//! until the buffer reaches that bound, and the earliest chunk then goes out.
//! The writer holds its newest video packet back by one, since only the last
//! packet states a duration: earlier `stts` entries come from dts deltas.

use std::io::{Cursor, Read, Seek, Write};

use containr::mp4::{self, MuxOptions, PacketIn};
use containr::{
    Chapter, MetadataItem, MetadataKey, MetadataScope, Packet, TagScope, TagTarget, Track,
    TrackKind,
};

use super::clock::{Mp4Clock, Mp4Placed};
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

const KIND: OutputKind = OutputKind::Mp4;
/// A Matroska title's item in an MP4 file.
const TITLE: &[u8] = b"TITLE";
/// The `TargetTypeValue` of a whole movie.
const MOVIE_TARGET: u64 = 50;

/// The copy plan of an MP4 output.
#[derive(Debug)]
pub(super) struct Mp4Copies {
    plans: Vec<mp4::StreamCopy>,
    movie_timescale: u32,
}

/// Plans each copied track, then dry-runs the mux.
pub(super) fn build(input: &mut Input, copied: &[Track], main: u32) -> Result<Built, MediaError> {
    let movie_timescale = input.movie_timescale();
    let (mut plans, streams) = plan_tracks(input, copied, KIND, |track, packets| {
        mp4::StreamCopy::plan(track, packets, movie_timescale)
    })?;
    remap_references(&mut plans, &streams, main);
    keep_track_names(&mut plans, copied);
    let metadata = movie_metadata(input);
    let chapters = mp4::StreamCopy::chapters(input.chapters());
    let copies = Mp4Copies {
        plans,
        movie_timescale,
    };
    dry_run(input, &copies, &streams, &metadata, &chapters)?;
    Ok(Built {
        copies: Copies::Mp4(copies),
        streams,
        metadata,
        chapters,
    })
}

/// `tref` entries name output tracks. The video is track 1, and the copies
/// follow it. A reference to a dropped track goes with that track.
fn remap_references(plans: &mut [mp4::StreamCopy], streams: &[CopiedStream], main: u32) {
    let output_id = |source: u32| {
        if source == main {
            return Some(1);
        }
        let position = streams
            .iter()
            .position(|stream| stream.track_id == source)?;
        u32::try_from(position + 2).ok()
    };
    for plan in plans {
        plan.spec.references.retain_mut(|reference| {
            reference.track_ids = reference
                .track_ids
                .iter()
                .filter_map(|&id| output_id(id))
                .collect();
            !reference.track_ids.is_empty()
        });
    }
}

/// A Matroska track `Name` has no field in the spec, so it copies as the
/// `udta/name` text box FFmpeg writes.
fn keep_track_names(plans: &mut [mp4::StreamCopy], copied: &[Track]) {
    for (plan, track) in plans.iter_mut().zip(copied) {
        let name = track.matroska.as_ref().and_then(|mkv| mkv.name.as_ref());
        if let Some(name) = name {
            plan.spec.metadata.push(MetadataItem {
                scope: MetadataScope::Udta,
                key: MetadataKey::FourCc(containr::FourCc(*b"name")),
                value: name.as_bytes().to_vec(),
                data: Vec::new(),
            });
        }
    }
}

/// The global metadata, with a Matroska `Title` as a tag when no tag states it.
fn movie_metadata(input: &Input) -> Vec<MetadataItem> {
    let mut items = input.metadata().to_vec();
    let tagged = items.iter().any(|item| {
        item.name()
            .is_some_and(|name| name.eq_ignore_ascii_case("TITLE"))
    });
    if let (Some(title), false) = (input.title(), tagged) {
        items.push(MetadataItem {
            scope: MetadataScope::Tag(TagScope {
                target_type_value: MOVIE_TARGET,
                target: TagTarget::Segment,
                binary: false,
            }),
            key: MetadataKey::Named(TITLE.to_vec()),
            value: title.as_bytes().to_vec(),
            data: Vec::new(),
        });
    }
    items
}

fn options(movie_timescale: u32, max_buffer: Option<usize>, faststart: bool) -> MuxOptions {
    let defaults = MuxOptions::default();
    MuxOptions {
        faststart,
        movie_timescale,
        max_buffer: max_buffer.unwrap_or(defaults.max_buffer),
        ..defaults
    }
}

fn apply_movie_facts<W: Write + Seek>(
    muxer: &mut mp4::Muxer<W>,
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

/// Runs the whole mux on memory: audio packets with no bytes, text packets
/// with their bytes, whose framing can refuse. Nothing in the real mux can then
/// refuse a copied stream, short of a failed write.
fn dry_run(
    input: &mut Input,
    copies: &Mp4Copies,
    streams: &[CopiedStream],
    metadata: &[MetadataItem],
    chapters: &[Chapter],
) -> Result<(), MediaError> {
    let dry_options = options(copies.movie_timescale, None, false);
    let mut muxer =
        mp4::Muxer::new(Cursor::new(Vec::new()), dry_options).map_err(|e| mux_error(KIND, &e))?;
    apply_movie_facts(&mut muxer, metadata.to_vec(), chapters.to_vec())?;
    let mut ids = Vec::with_capacity(streams.len());
    for (plan, stream) in copies.plans.iter().zip(streams) {
        let mut spec = plan.spec.clone();
        // The dry run holds no video track, so output ids do not hold here.
        spec.references.clear();
        let id = muxer
            .add_track(spec)
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
            plan.packet(&packet, &buffer)
                .and_then(|packet_in| muxer.write_packet(id, &packet_in))
                .map_err(|error| refused(KIND, &stream.label, &error))?;
        }
    }
    muxer
        .finish()
        .map_err(|error| finish_refused(KIND, streams, &error, "a copied track"))?;
    Ok(())
}

/// One copied track's plan and output id.
#[derive(Debug)]
struct Mp4Copy {
    copy: mp4::StreamCopy,
    id: mp4::TrackId,
}

impl StreamClock for Mp4Copy {
    fn decode_time(&self, packet: &Packet) -> Result<Option<Rational>, MediaError> {
        let timing = usize::try_from(packet.index)
            .ok()
            .and_then(|index| self.copy.timing().get(index));
        // A packet past the plan writes first, so `StreamCopy::packet` names it.
        let Some(timing) = timing else {
            return Ok(Some(Rational::ZERO));
        };
        Rational::new(timing.dts, u64::from(self.copy.spec.timescale))
            .map(Some)
            .map_err(|error| MediaError::Time {
                what: "the decode time of a copied packet",
                error,
            })
    }
}

impl<W: Write + Seek> StreamSink<mp4::Muxer<W>> for Mp4Copy {
    fn write(
        &self,
        muxer: &mut mp4::Muxer<W>,
        packet: &Packet,
        data: &[u8],
    ) -> Result<(), MediaError> {
        self.copy
            .packet(packet, data)
            .and_then(|packet_in| muxer.write_packet(self.id, &packet_in))
            .map_err(|error| mux_error(KIND, &error))
    }
}

/// The newest video packet, waiting for the next one.
#[derive(Debug)]
struct Held {
    data: Vec<u8>,
    placed: Mp4Placed,
    sync: bool,
}

/// The running MP4 mux.
#[derive(Debug)]
pub(super) struct Mp4Writer<W> {
    muxer: mp4::Muxer<W>,
    video: mp4::TrackId,
    clock: Mp4Clock,
    copies: Vec<CopiedTrack<Mp4Copy>>,
    held: Option<Held>,
    max_buffer: usize,
}

impl<W: Write + Seek> Mp4Writer<W> {
    /// Writes the file start and adds every track, the video first.
    pub(super) fn start(
        writer: W,
        copies: Mp4Copies,
        streams: &[CopiedStream],
        source: &Input,
        facts: MovieFacts,
        video: &VideoOutput,
    ) -> Result<Self, MediaError> {
        let options = options(copies.movie_timescale, facts.max_buffer, true);
        let mut muxer = mp4::Muxer::new(writer, options).map_err(|e| mux_error(KIND, &e))?;
        let clock = Mp4Clock::new(video, copies.movie_timescale)?;
        let video_id = muxer
            .add_track(video.mp4_spec(clock.timescale())?)
            .map_err(|e| mux_error(KIND, &e))?;
        apply_movie_facts(&mut muxer, facts.metadata, facts.chapters)?;
        let mut copied = Vec::with_capacity(copies.plans.len());
        for (copy, stream) in copies.plans.into_iter().zip(streams) {
            let id = muxer
                .add_track(copy.spec.clone())
                .map_err(|e| mux_error(KIND, &e))?;
            let packets = source.track_packets(stream.track_id)?;
            copied.push(CopiedTrack::new(packets, Mp4Copy { copy, id }));
        }
        Ok(Self {
            muxer,
            video: video_id,
            clock,
            copies: copied,
            held: None,
            max_buffer: options.max_buffer,
        })
    }

    /// Takes `packet`, writes the packet before it, and copies the streams up
    /// to that packet's decode time.
    pub(super) fn write_video(
        &mut self,
        packet: &EncodedPacket,
        source: &mut Input,
        buffer: &mut Vec<u8>,
    ) -> Result<(), MediaError> {
        let placed = self.clock.place(packet)?;
        let mut up_to = None;
        let mut data = Vec::new();
        if let Some(held) = self.held.take() {
            self.write_held(&held, None)?;
            up_to = Some(self.track_time(held.placed.dts)?);
            data = held.data;
        }
        data.clear();
        data.extend_from_slice(&packet.data);
        self.held = Some(Held {
            data,
            placed,
            sync: packet.keyframe,
        });
        match up_to {
            Some(up_to) => self.copy_pending(up_to, source, buffer),
            None => Ok(()),
        }
    }

    fn track_time(&self, ticks: i64) -> Result<Rational, MediaError> {
        Rational::new(ticks, u64::from(self.clock.timescale())).map_err(|error| MediaError::Time {
            what: "the decode time of a video packet",
            error,
        })
    }

    fn write_held(&mut self, held: &Held, duration: Option<u64>) -> Result<(), MediaError> {
        self.muxer
            .write_packet(
                self.video,
                &PacketIn {
                    data: &held.data,
                    pts: held.placed.pts,
                    dts: held.placed.dts,
                    duration,
                    sync: held.sync,
                },
            )
            .map_err(|e| mux_error(KIND, &e))
    }

    /// Copies every packet that decodes at or before `up_to` seconds.
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

    /// The packet bytes in memory: the muxer's, and the held packet's.
    pub(super) fn buffered_bytes(&self) -> usize {
        let held = self.held.as_ref().map_or(0, |held| held.data.len());
        self.muxer.buffered_bytes().saturating_add(held)
    }

    /// The bound on [`Mp4Writer::buffered_bytes`], short of one packet.
    pub(super) const fn buffer_bound(&self) -> usize {
        self.max_buffer
    }
}

impl<W: Read + Write + Seek> Mp4Writer<W> {
    /// Writes the last video packet with its duration, every copied packet
    /// left, the edit list, and `moov`.
    pub(super) fn finish(
        mut self,
        source: &mut Input,
        buffer: &mut Vec<u8>,
    ) -> Result<W, MediaError> {
        let Some(held) = self.held.take() else {
            return Err(MediaError::NoVideoPackets { output: KIND });
        };
        self.write_held(&held, Some(held.placed.duration))?;
        pump(&mut self.copies, &mut self.muxer, source, buffer, None)?;
        if let Some(edits) = self.clock.edit_list() {
            self.muxer
                .set_edit_list(self.video, &edits)
                .map_err(|e| mux_error(KIND, &e))?;
        }
        self.muxer.finish().map_err(|e| mux_error(KIND, &e))
    }
}
