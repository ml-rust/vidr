//! What a caller plugs into the pump.

use numr::runtime::Runtime;

use super::PipelineError;
use crate::decode::DecodedFrame;
use crate::encode::EncodedPacket;
use crate::frame::OutputPlanes;
use crate::time::{DurationHints, EditTimeline, Rational};

/// The timing facts of one packet. [`PacketSource::next`] writes its bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SourcePacket {
    /// Presentation timestamp, in ticks of [`PumpTiming::time_base`].
    pub pts: i64,
    /// Duration in ticks. Zero or less means the container carries none.
    pub duration: i64,
}

/// Packets of one video track in decode order. It runs on T1.
pub trait PacketSource: Send {
    /// Replaces `data` with the next packet's bytes and returns its timing, or
    /// returns `None` at the end of the stream. The pump reuses one buffer, so
    /// a source allocates no memory per packet.
    ///
    /// # Errors
    ///
    /// Any [`PipelineError`]. The first error ends the run.
    fn next(&mut self, data: &mut Vec<u8>) -> Result<Option<SourcePacket>, PipelineError>;
}

/// The work on one frame: colour conversion, a model, and a resize. It runs on
/// T2, and takes one frame and returns one frame.
pub trait FrameTransform<R: Runtime>: Send {
    /// Returns the planes to encode. The pump already dropped frames outside
    /// the edit window, and it keeps the pts and duration apart from the frame.
    ///
    /// # Errors
    ///
    /// Any [`PipelineError`]. The first error ends the run.
    fn apply(&mut self, frame: DecodedFrame<R>) -> Result<OutputPlanes<R>, PipelineError>;
}

/// The output of a run. It runs on T3.
pub trait PacketSink: Send {
    /// What a finished output returns, such as the writer of a mux.
    type Output: Send;

    /// Takes the encoder's config record, once, before the first packet.
    ///
    /// # Errors
    ///
    /// Any [`PipelineError`]. The first error ends the run.
    fn start(&mut self, config_record: &[u8]) -> Result<(), PipelineError>;

    /// Takes the next packet in decode order.
    ///
    /// # Errors
    ///
    /// Any [`PipelineError`]. The first error ends the run.
    fn write(&mut self, packet: EncodedPacket) -> Result<(), PipelineError>;

    /// Makes the output final. The pump calls this after a complete run's last
    /// packet. It never calls this after an error or a stop request.
    ///
    /// # Errors
    ///
    /// Any [`PipelineError`].
    fn finish(self) -> Result<Self::Output, PipelineError>;
}

/// The timing of the source track.
///
/// The pump passes each pts and duration to the encoder in this time base. The
/// encoder's config must name the same one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PumpTiming {
    /// Seconds per tick of [`SourcePacket`] timestamps.
    pub time_base: Rational,
    /// The track's edit list. `None` for a track with no edit list, whose pts
    /// pass as stored, as `read_stream_facts` reads them.
    pub timeline: Option<EditTimeline>,
    /// The last-frame duration chain's stream-level inputs. The last packet's
    /// duration replaces `packet_duration` when it is above zero.
    pub hints: DurationHints,
}

impl PumpTiming {
    /// Timing with no edit list and no hints.
    pub fn new(time_base: Rational) -> Self {
        Self {
            time_base,
            timeline: None,
            hints: DurationHints::default(),
        }
    }
}
