//! Hand-written bindings to the NVDEC decoder and parser in `libnvcuvid`.

mod layout;
mod library;
mod result;
mod types;

pub(crate) use library::{Cuvid, cuvid};
pub(crate) use result::check;
pub(crate) use types::*;
