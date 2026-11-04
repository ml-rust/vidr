//! The threaded pump: demux and decode, transform, then encode and mux, with
//! frames in device memory throughout. See [`run`].
//!
//! The pump needs no GPU. It runs over the [`Decoder`](crate::decode::Decoder)
//! and [`Encoder`](crate::encode::Encoder) traits on any numr backend.

mod control;
mod present;
mod pump;
mod traits;

pub use crate::error::{PipelineError, PumpStage};
pub use control::{PumpControl, PumpProgress, PumpReport};
pub use pump::{MAX_IN_FLIGHT_FRAMES, run};
pub use traits::{FrameTransform, PacketSink, PacketSource, PumpTiming, SourcePacket};
