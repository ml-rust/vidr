//! Media input and output over containr: which container a file holds, which
//! video track to read, the facts vidr needs from that track, and the output
//! that replaces the video and copies every other stream.
//!
//! Each container is a Cargo feature: `mp4` reads MP4 and MOV and writes MP4,
//! and `mkv` reads Matroska and WebM and writes Matroska. The parsing and
//! writing live in containr.

mod clock;
mod copied;
mod edit;
mod facts;
mod input;
mod output;
#[cfg(feature = "mkv")]
mod output_mkv;
#[cfg(feature = "mp4")]
mod output_mp4;
mod plan;
#[cfg(feature = "pipeline")]
mod pump;
mod video;

pub use containr::{ContainerColour, Language, Track, TrackKind};
pub use facts::{DecodeConfig, read_stream_facts, read_track_facts};
pub use input::Input;
pub use output::OutputMux;
pub use plan::{CopyPlan, DroppedTrack, OutputKind, plan_copy};
#[cfg(feature = "pipeline")]
pub use pump::{InputSource, MuxSink};
pub use video::VideoOutput;
