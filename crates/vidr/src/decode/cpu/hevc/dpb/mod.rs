//! Reference picture management (8.3) and the output order decoded picture buffer
//! (C.5.2). The sample-free parts live in [`crate::decode::hevc_order`].

pub(crate) mod buffer;
pub(crate) mod ref_lists;
pub(crate) mod stored;

pub(crate) use crate::decode::hevc_order::{poc, rps};
