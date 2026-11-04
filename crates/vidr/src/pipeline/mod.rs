//! The threaded pump: demux and decode, transform, then encode and mux, with
//! frames in device memory throughout.
//!
//! The pump is the `pipeline` feature. It needs no GPU, and runs over the
//! `Decoder` and `Encoder` traits on any numr backend. `PumpControl` is in every
//! build, so an application takes a stop request whatever its features.

mod control;
#[cfg(feature = "pipeline")]
mod present;
#[cfg(feature = "pipeline")]
mod pump;
#[cfg(feature = "pipeline")]
mod traits;

#[cfg(feature = "pipeline")]
pub use crate::error::{PipelineError, PumpStage};
pub use control::{PumpControl, PumpProgress, PumpReport};
#[cfg(feature = "pipeline")]
pub use pump::{MAX_IN_FLIGHT_FRAMES, run};
#[cfg(feature = "pipeline")]
pub use traits::{FrameTransform, PacketSink, PacketSource, PumpTiming, SourcePacket};
