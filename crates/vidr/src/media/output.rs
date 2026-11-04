//! The output file: an encoded video track and every copied stream, written in
//! one pass.
//!
//! The encoder's first packet supplies the config record, so the output starts
//! after it, every track goes in at the start, and no intermediate video file
//! exists.

use std::io::{Read, Seek, Write};

use super::input::Input;
#[cfg(feature = "mkv")]
use super::output_mkv::MkvWriter;
#[cfg(feature = "mp4")]
use super::output_mp4::Mp4Writer;
use super::plan::{Copies, CopyPlan, MovieFacts};
use super::video::VideoOutput;
use crate::encode::EncodedPacket;
use crate::error::MediaError;
use crate::time::Rational;

#[derive(Debug)]
enum Writer<W> {
    #[cfg(feature = "mp4")]
    Mp4(Box<Mp4Writer<W>>),
    #[cfg(feature = "mkv")]
    Mkv(Box<MkvWriter<W>>),
}

/// A running output.
///
/// Copied packets demux one at a time, by decode time, alongside the video, so
/// memory holds no whole track. `OutputMux::buffered_bytes` states what the
/// output holds. Copied tracks keep their timeline.
#[derive(Debug)]
pub struct OutputMux<W> {
    writer: Writer<W>,
    source: Input,
    buffer: Vec<u8>,
}

impl<W: Write + Seek + Read> OutputMux<W> {
    /// Writes the start of the file on `writer`, which must start empty.
    ///
    /// # Errors
    ///
    /// [`MediaError::VideoOutput`] for a field out of range, and
    /// [`MediaError::Mux`] for a failed write or a buffer bound containr
    /// refuses.
    pub fn start(writer: W, plan: CopyPlan, video: VideoOutput) -> Result<Self, MediaError> {
        video.check()?;
        let facts = MovieFacts {
            metadata: plan.metadata,
            chapters: plan.chapters,
            max_buffer: plan.max_buffer,
        };
        let streams = plan.streams;
        let source = plan.source;
        let writer = match plan.copies {
            #[cfg(feature = "mp4")]
            Copies::Mp4(copies) => Writer::Mp4(Box::new(Mp4Writer::start(
                writer, copies, &streams, &source, facts, &video,
            )?)),
            #[cfg(feature = "mkv")]
            Copies::Mkv(copies) => Writer::Mkv(Box::new(MkvWriter::start(
                writer, copies, &streams, &source, facts, &video,
            )?)),
        };
        Ok(Self {
            writer,
            source,
            buffer: Vec::new(),
        })
    }

    /// Takes the next encoded packet, in decode order, and copies the other
    /// streams up to its time. The first packet's pts must be the earliest.
    ///
    /// # Errors
    ///
    /// [`MediaError::VideoPacket`] for a packet that breaks the output clock,
    /// [`MediaError::Mux`] for a failed write, and [`MediaError::Demux`] for a
    /// failed read of a copied packet.
    pub fn write_video(&mut self, packet: &EncodedPacket) -> Result<(), MediaError> {
        match &mut self.writer {
            #[cfg(feature = "mp4")]
            Writer::Mp4(writer) => writer.write_video(packet, &mut self.source, &mut self.buffer),
            #[cfg(feature = "mkv")]
            Writer::Mkv(writer) => writer.write_video(packet, &mut self.source, &mut self.buffer),
        }
    }

    /// Copies every packet at or before `up_to` seconds on the output's decode
    /// clock. [`OutputMux::write_video`] calls this itself. Call it directly to
    /// keep copied streams moving while an encoder holds its packets back.
    ///
    /// # Errors
    ///
    /// As [`OutputMux::write_video`].
    pub fn copy_pending(&mut self, up_to: Rational) -> Result<(), MediaError> {
        match &mut self.writer {
            #[cfg(feature = "mp4")]
            Writer::Mp4(writer) => writer.copy_pending(up_to, &mut self.source, &mut self.buffer),
            #[cfg(feature = "mkv")]
            Writer::Mkv(writer) => writer.copy_pending(up_to, &mut self.source, &mut self.buffer),
        }
    }

    /// The packet bytes the output holds in memory now. Attachments, which
    /// Matroska holds from the finish, do not count.
    pub fn buffered_bytes(&self) -> usize {
        match &self.writer {
            #[cfg(feature = "mp4")]
            Writer::Mp4(writer) => writer.buffered_bytes(),
            #[cfg(feature = "mkv")]
            Writer::Mkv(writer) => writer.buffered_bytes(),
        }
    }

    /// The bound on [`OutputMux::buffered_bytes`], short of one packet: the
    /// plan's `with_max_buffer` value, plus one Cluster for Matroska.
    pub fn buffer_bound(&self) -> usize {
        match &self.writer {
            #[cfg(feature = "mp4")]
            Writer::Mp4(writer) => writer.buffer_bound(),
            #[cfg(feature = "mkv")]
            Writer::Mkv(writer) => writer.buffer_bound(),
        }
    }

    /// Writes everything left and returns the writer.
    ///
    /// # Errors
    ///
    /// [`MediaError::NoVideoPackets`] when no packet came, and as
    /// [`OutputMux::write_video`].
    pub fn finish(mut self) -> Result<W, MediaError> {
        match self.writer {
            #[cfg(feature = "mp4")]
            Writer::Mp4(writer) => writer.finish(&mut self.source, &mut self.buffer),
            #[cfg(feature = "mkv")]
            Writer::Mkv(writer) => writer.finish(&mut self.source, &mut self.buffer),
        }
    }
}
