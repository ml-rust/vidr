//! Whole runs with no GPU, on the CPU runtime and the CPU codecs, checked from outside
//! through the vidr output checks.
//!
//! Run with `cargo nextest run -p vidr-remaster --features
//! mp4,mkv,dav1d,cpu-vp9,cpu-hevc,openh264,cpu-av1,model-srvgg --test e2e_cpu`. A
//! missing ffmpeg, ffmpeg encoder, or model skips with a printed notice. A test whose
//! codec feature is off does not compile.
//!
//! ffmpeg generates each source under `CARGO_TARGET_TMPDIR`.

#![cfg(all(feature = "transform", feature = "mp4"))]

#[path = "e2e_cpu/harness.rs"]
mod harness;
#[path = "e2e_cpu/refusals.rs"]
mod refusals;
#[path = "e2e_cpu/reuse.rs"]
mod reuse;
#[path = "e2e_cpu/runs.rs"]
mod runs;
