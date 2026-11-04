//! HEVC decoding for Main and Main 10 at 4:2:0, written from ITU-T H.265 V11
//! (01/2026). Section numbers in comments refer to it.
//!
//! The parse stage reads CABAC data and writes plain records for one CTU. The
//! reconstruct stage reads those records and no bitstream state: it derives motion,
//! predicts, and adds residuals. A later unit can run the two stages on separate
//! threads. The in-loop filters run per picture after both stages. Finished pictures
//! enter a decoded picture buffer, which holds references and returns output order.

pub(crate) mod cabac;
pub(crate) mod dpb;
pub(crate) mod error;
pub(crate) mod filter;
pub(crate) mod hash;
pub(crate) mod layout;
pub(crate) mod motion;
pub(crate) mod params;
pub(crate) mod picture;
pub(crate) mod picture_decoder;
pub(crate) mod qp;
pub(crate) mod recon;
pub(crate) mod residual;
pub(crate) mod scaling;
pub(crate) mod scan;
pub(crate) mod stream;
#[cfg(test)]
mod stream_tests;
pub(crate) mod syntax;
