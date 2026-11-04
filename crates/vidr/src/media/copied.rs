//! Copied tracks at run time: packets demux one at a time, and the earliest
//! one goes next.
//!
//! The source is read lazily, so a long track holds one packet's bytes here at
//! a time. The muxer's `max_buffer` bounds what it holds behind that.

use containr::Packet;

use super::input::{Input, TrackPacketIter};
use crate::error::MediaError;
use crate::time::Rational;

/// When a copied packet goes out.
pub(super) trait StreamClock {
    /// The packet's decode time in the output, in seconds. `None` when the
    /// packet writes nothing.
    fn decode_time(&self, packet: &Packet) -> Result<Option<Rational>, MediaError>;
}

/// How a copied packet goes out through a muxer of type `M`.
pub(super) trait StreamSink<M>: StreamClock {
    /// Writes one packet.
    fn write(&self, muxer: &mut M, packet: &Packet, data: &[u8]) -> Result<(), MediaError>;
}

/// One copied track: its unread packets, and the next one.
#[derive(Debug)]
pub(super) struct CopiedTrack<C> {
    packets: TrackPacketIter,
    head: Option<Packet>,
    stream: C,
}

impl<C> CopiedTrack<C> {
    pub(super) fn new(packets: TrackPacketIter, stream: C) -> Self {
        Self {
            packets,
            head: None,
            stream,
        }
    }
}

impl<C: StreamClock> CopiedTrack<C> {
    /// The decode time of the next packet that writes something.
    fn head_time(&mut self, source: &Input) -> Result<Option<Rational>, MediaError> {
        loop {
            if self.head.is_none() {
                self.head = self
                    .packets
                    .next()
                    .transpose()
                    .map_err(|error| source.demux_error(&error))?;
            }
            let Some(packet) = self.head else {
                return Ok(None);
            };
            match self.stream.decode_time(&packet)? {
                Some(time) => return Ok(Some(time)),
                None => self.head = None,
            }
        }
    }
}

/// Writes copied packets in decode time order, up to and including `up_to`.
/// `None` writes every packet left.
pub(super) fn pump<M, C: StreamSink<M>>(
    tracks: &mut [CopiedTrack<C>],
    muxer: &mut M,
    source: &mut Input,
    buffer: &mut Vec<u8>,
    up_to: Option<Rational>,
) -> Result<(), MediaError> {
    loop {
        let mut earliest: Option<(usize, Rational)> = None;
        for (slot, track) in tracks.iter_mut().enumerate() {
            let Some(time) = track.head_time(source)? else {
                continue;
            };
            if earliest.is_none_or(|(_, best)| time < best) {
                earliest = Some((slot, time));
            }
        }
        let Some((slot, time)) = earliest else {
            return Ok(());
        };
        if up_to.is_some_and(|limit| time > limit) {
            return Ok(());
        }
        let Some(track) = tracks.get_mut(slot) else {
            return Ok(());
        };
        let Some(packet) = track.head.take() else {
            return Ok(());
        };
        source.read_packet_into(&packet, buffer)?;
        track.stream.write(muxer, &packet, buffer)?;
    }
}
