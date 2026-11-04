//! Motion data and its derivation (8.5.3.2): merge mode, motion vector prediction,
//! temporal prediction, and per-picture storage. This stage reads no bitstream.

pub(crate) mod amvp;
pub(crate) mod col;
pub(crate) mod context;
pub(crate) mod derive;
pub(crate) mod field;
pub(crate) mod merge;
pub(crate) mod scale;
pub(crate) mod temporal;
pub(crate) mod types;

#[cfg(test)]
mod test_support;
