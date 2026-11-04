//! The stop flag and progress counters a caller shares with a run.
//!
//! This file needs no `pipeline` feature, so a caller holds a control in any
//! build, and a build with no pump can still report a stop.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// A stop flag and progress counters. Create one per run.
///
/// The counters only increase. The first error of a run also sets the flag.
#[derive(Debug, Default)]
pub struct PumpControl {
    stop: AtomicBool,
    pub(super) packets_read: AtomicU64,
    pub(super) frames_in: AtomicU64,
    pub(super) frames_dropped: AtomicU64,
    pub(super) frames_out: AtomicU64,
    pub(super) packets: AtomicU64,
}

/// A reading of the counters of a [`PumpControl`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PumpProgress {
    /// Packets the source supplied.
    pub packets_read: u64,
    /// Frames the decoder returned.
    pub frames_in: u64,
    /// Frames the edit window dropped.
    pub frames_dropped: u64,
    /// Frames the encoder took.
    pub frames_out: u64,
    /// Packets the sink took.
    pub packets: u64,
}

/// The result of a complete run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PumpReport<O> {
    /// Frames the decoder returned.
    pub frames_in: u64,
    /// Frames the edit window dropped.
    pub frames_dropped: u64,
    /// Frames the encoder took.
    pub frames_out: u64,
    /// Packets the sink took.
    pub packets: u64,
    /// What `PacketSink::finish` returned.
    pub output: O,
}

impl PumpControl {
    /// A control with no stop request and zero counters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Asks the run to stop. A stage sees the flag between frames, and a run
    /// then returns `PipelineError::Stopped`.
    /// A flag set before the run starts stops it at once.
    ///
    /// This call only stores an atomic flag, so a signal handler can make it.
    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Whether a stop request or a first error set the flag.
    pub fn stop_requested(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// The counters now.
    pub fn progress(&self) -> PumpProgress {
        PumpProgress {
            packets_read: self.packets_read.load(Ordering::Relaxed),
            frames_in: self.frames_in.load(Ordering::Relaxed),
            frames_dropped: self.frames_dropped.load(Ordering::Relaxed),
            frames_out: self.frames_out.load(Ordering::Relaxed),
            packets: self.packets.load(Ordering::Relaxed),
        }
    }
}

/// Adds one to a counter.
#[cfg(feature = "pipeline")]
pub(super) fn count(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_control_has_no_stop_and_zero_counters() {
        let control = PumpControl::new();
        assert!(!control.stop_requested());
        assert_eq!(control.progress(), PumpProgress::default());
    }

    #[test]
    fn a_stop_request_stays_set() {
        let control = PumpControl::new();
        control.request_stop();
        assert!(control.stop_requested());
        control.request_stop();
        assert!(control.stop_requested());
    }
}
