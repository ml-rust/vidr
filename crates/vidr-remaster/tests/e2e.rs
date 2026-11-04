//! Whole runs on a real GPU, checked from outside through ffprobe, ffmpeg, and the
//! vidr output checks.
//!
//! Run with `cargo nextest run -p vidr-remaster --features
//! cuda,mp4,mkv,model-srvgg,model-rrdb --test e2e`. `BOOSTR_MODELS_DIR` names the
//! model root. A missing GPU, model, ffmpeg, or ffprobe skips with a printed notice.
//!
//! ffmpeg generates each source under `CARGO_TARGET_TMPDIR`. Every run goes through
//! `main_with_args`, except the stop test, which needs `remaster_in`.

#![cfg(all(
    feature = "cuda",
    feature = "mp4",
    feature = "mkv",
    feature = "model-srvgg",
    feature = "model-rrdb"
))]

#[path = "e2e/codecs.rs"]
mod codecs;
#[path = "e2e/colour.rs"]
mod colour;
#[path = "e2e/containers.rs"]
mod containers;
#[path = "e2e/geometry.rs"]
mod geometry;
#[path = "e2e/harness.rs"]
mod harness;
#[path = "e2e/models.rs"]
mod models;
#[path = "e2e/overwrite.rs"]
mod overwrite;
#[path = "e2e/refusals.rs"]
mod refusals;
#[path = "e2e/reuse.rs"]
mod reuse;
#[path = "e2e/stop.rs"]
mod stop;
#[path = "e2e/timing.rs"]
mod timing;
