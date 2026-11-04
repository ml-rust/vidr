//! HEVC picture order and reference marking with no samples, shared by the CPU decoder and
//! Vulkan Video.
//!
//! | Module | Holds |
//! | --- | --- |
//! | `poc` | `NoRaslOutputFlag` (8.1.3) and picture order count (8.3.1) |
//! | `rps` | The POC lists of a reference picture set (8-5) |
//! | `buffer` | Reference marking (8.3.2) and bumping (C.5.2), generic over the stored picture |

#![forbid(unsafe_code)]

pub(crate) mod buffer;
pub(crate) mod poc;
pub(crate) mod rps;
