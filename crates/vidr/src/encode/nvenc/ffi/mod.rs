//! Hand-written bindings to NVENC API 13.1 in `libnvidia-encode`.

#[macro_use]
mod zeroed;

mod config_types;
mod guid;
mod layout;
mod library;
mod result;
mod types;

pub(crate) use config_types::*;
pub(crate) use guid::*;
pub(crate) use library::{Nvenc, nvenc};
pub(crate) use result::{ERR_INCOMPATIBLE_CLIENT_KEY, ERR_OUT_OF_MEMORY, NEED_MORE_INPUT, check};
pub(crate) use types::*;
