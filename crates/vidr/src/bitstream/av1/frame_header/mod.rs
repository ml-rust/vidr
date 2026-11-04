//! The AV1 frame header: `uncompressed_header()` (AV1 specification 5.9), with the saved state
//! of the eight reference names that its syntax reads. No samples and no CDFs.
//!
//! | Module | Holds |
//! | --- | --- |
//! | `types` | The frame header, its parts, and the constants of section 3 |
//! | `ref_table` | `RefValid`, `RefOrderHint`, and everything section 7.20 saves |
//! | `set_frame_refs` | The set frame refs process of section 7.8 |
//! | `parse` | `uncompressed_header()` and `show_existing_frame` |
//! | `size` | Frame size, superres, render size, and sizes from references |
//! | `tile_info` | `tile_info()` |
//! | `quant` | Quantization, segmentation, and the delta parameters |
//! | `filters` | Loop filter, CDEF, loop restoration, and the transform mode |
//! | `skip_mode` | `skip_mode_params()` |
//! | `global_motion` | `global_motion_params()` with the previous parameters |
//! | `film_grain` | `film_grain_params()` with its load |

mod film_grain;
mod filters;
mod global_motion;
mod parse;
mod quant;
mod ref_table;
mod set_frame_refs;
mod size;
mod skip_mode;
mod tile_info;
mod types;

#[cfg(test)]
mod tests;

pub(crate) use parse::parse_frame_header;
pub(crate) use ref_table::RefTable;
pub(crate) use types::*;
