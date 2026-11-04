//! HEVC parsing and block-level steps for Main and Main 10 at 4:2:0, written
//! from ITU-T H.265 V11 (01/2026). Section numbers in comments refer to it.
//!
//! The parse stage reads CABAC data and writes plain records for one CTU. The
//! reconstruct stage reads those records and no bitstream state. A later unit
//! can run the two stages on separate threads.

pub(crate) mod cabac;
pub(crate) mod error;
pub(crate) mod layout;
pub(crate) mod params;
pub(crate) mod picture;
pub(crate) mod qp;
pub(crate) mod residual;
pub(crate) mod scaling;
pub(crate) mod scan;
pub(crate) mod syntax;
