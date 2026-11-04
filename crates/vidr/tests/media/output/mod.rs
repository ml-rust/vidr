//! The output layer: every copied stream, each refusal, rounding, and the
//! memory bound, with a real stream's samples in place of an encoder. No test
//! here needs a GPU.

mod checks;
mod memory;
mod probe;
mod refusals;
mod remux;
mod rounding;
pub(crate) mod support;
