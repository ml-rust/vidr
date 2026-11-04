//! AV1: `av1C`, OBU framing, the sequence and frame headers, and the sample filters between MP4
//! or Matroska and the decoder or encoder.

mod av1c;
#[cfg_attr(
    not(feature = "vulkan-video"),
    allow(dead_code, reason = "read by the Vulkan decoder alone")
)]
pub(crate) mod frame_header;
mod from_encoder;
pub mod obu;
#[cfg_attr(
    not(feature = "vulkan-video"),
    allow(
        dead_code,
        reason = "a build with `av1` alone reads the sequence facts only"
    )
)]
pub(crate) mod seq_header;
mod sequence;
#[cfg_attr(
    not(feature = "vulkan-video"),
    allow(dead_code, reason = "read by the Vulkan decoder alone")
)]
pub(crate) mod temporal_unit;
#[cfg(test)]
pub(crate) mod test_stream;
#[cfg_attr(
    not(feature = "vulkan-video"),
    allow(dead_code, reason = "read by the Vulkan decoder alone")
)]
pub(crate) mod tile_group;
mod to_decoder;

pub use av1c::Av1Config;
pub use from_encoder::Av1FromEncoder;
pub use obu::{Obu, ObuExtension, ObuHeader, ObuType};
pub use sequence::{Av1SequenceFacts, av1_sequence_facts};
pub use to_decoder::Av1ToDecoder;
