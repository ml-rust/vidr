//! Media input and colour policy against ffprobe.
//!
//! `tests/fixtures/gen_media.py` writes the fixtures. A test whose fixture is
//! missing, or whose container feature is off, prints a notice and passes
//! without checks.

mod colour;
mod edit;
mod facts;
pub(crate) mod fixtures;
mod input;
