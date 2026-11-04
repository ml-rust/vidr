//! The pump's source and sink over [`Input`] and [`OutputMux`].

use std::io::{Read, Seek, Write};

use super::input::{Input, TrackPacketIter};
use super::output::OutputMux;
use super::plan::{CopyPlan, OutputKind};
use super::video::VideoOutput;
use crate::encode::EncodedPacket;
use crate::error::{MediaError, PipelineError};
use crate::pipeline::{PacketSink, PacketSource, PumpTiming, SourcePacket};
use crate::time::Rational;

/// The packets of an input's first video track, in decode order.
#[derive(Debug)]
pub struct InputSource {
    input: Input,
    packets: TrackPacketIter,
    timing: PumpTiming,
}

impl InputSource {
    /// Reads the first video track of `input`.
    ///
    /// # Errors
    ///
    /// [`MediaError::NoVideoTrack`] for no video, [`MediaError::Time`] for a
    /// time base of zero, and any error [`Input::edit_timeline`] gives.
    pub fn new(input: Input) -> Result<Self, MediaError> {
        let track = input.video_track()?;
        let time_base = Rational::new(i64::from(track.timebase.num), u64::from(track.timebase.den))
            .map_err(|error| MediaError::Time {
                what: "the video track's time base",
                error,
            })?;
        let timeline = if track.edit_list.is_empty() {
            None
        } else {
            Some(input.edit_timeline(track)?)
        };
        let packets = input.track_packets(track.id)?;
        Ok(Self {
            input,
            packets,
            timing: PumpTiming {
                timeline,
                ..PumpTiming::new(time_base)
            },
        })
    }

    /// The track's time base and edit list, with no frame-rate hint. Set
    /// [`PumpTiming::hints`] for a track whose packets carry no duration.
    pub fn timing(&self) -> PumpTiming {
        self.timing
    }
}

impl PacketSource for InputSource {
    fn next(&mut self, data: &mut Vec<u8>) -> Result<Option<SourcePacket>, PipelineError> {
        let Some(packet) = self.packets.next() else {
            return Ok(None);
        };
        let packet = packet.map_err(|error| self.input.demux_error(&error))?;
        self.input.read_packet_into(&packet, data)?;
        Ok(Some(SourcePacket {
            pts: packet.pts,
            duration: packet
                .duration
                .map_or(0, |ticks| i64::try_from(ticks).unwrap_or(i64::MAX)),
        }))
    }
}

/// What a [`MuxSink`] holds before the encoder's first packet.
#[derive(Debug)]
struct Waiting<W> {
    writer: W,
    plan: CopyPlan,
    video: VideoOutput,
}

/// An [`OutputMux`] that starts when the encoder's first packet supplies the
/// config record. [`VideoOutput::config_record`] takes that record.
///
/// A run that ends in an error or a stop request drops the writer. The file is
/// not final, so the caller removes it.
#[derive(Debug)]
pub struct MuxSink<W> {
    kind: OutputKind,
    waiting: Option<Waiting<W>>,
    mux: Option<OutputMux<W>>,
}

impl<W> MuxSink<W> {
    /// A sink that writes `plan` and `video` to `writer`, which must start empty.
    pub fn new(writer: W, plan: CopyPlan, video: VideoOutput) -> Self {
        Self {
            kind: plan.kind(),
            waiting: Some(Waiting {
                writer,
                plan,
                video,
            }),
            mux: None,
        }
    }
}

impl<W: Write + Seek + Read + Send> PacketSink for MuxSink<W> {
    type Output = W;

    fn start(&mut self, config_record: &[u8]) -> Result<(), PipelineError> {
        if let Some(mut waiting) = self.waiting.take() {
            waiting.video.config_record = config_record.to_vec();
            self.mux = Some(OutputMux::start(
                waiting.writer,
                waiting.plan,
                waiting.video,
            )?);
        }
        Ok(())
    }

    fn write(&mut self, packet: EncodedPacket) -> Result<(), PipelineError> {
        match &mut self.mux {
            Some(mux) => Ok(mux.write_video(&packet)?),
            None => Err(PipelineError::failed(
                format!("writing the packet at pts {} to `MuxSink`", packet.pts),
                "the sink has no config record yet. Call `PacketSink::start` first.",
            )),
        }
    }

    fn finish(self) -> Result<W, PipelineError> {
        match self.mux {
            Some(mux) => Ok(mux.finish()?),
            None => Err(MediaError::NoVideoPackets { output: self.kind }.into()),
        }
    }
}
