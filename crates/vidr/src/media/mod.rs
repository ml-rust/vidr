//! Media input over containr: which container a file holds, which video track
//! to read, and the facts vidr needs from that track.
//!
//! Each container is a Cargo feature: `mp4` reads MP4 and MOV, and `mkv` reads
//! Matroska and WebM. The parsing lives in containr.

mod edit;
mod facts;
mod input;

pub use containr::{ContainerColour, Track};
pub use facts::DecodeConfig;
pub use input::Input;
