//! The decoder interface every backend implements: NVDEC now, and the
//! Vulkan Video and CPU paths later.

use numr::runtime::Runtime;

use super::frame::DecodedFrame;
use crate::error::DecodeError;

/// A video decoder that takes packets in decode order and returns frames in
/// display order.
///
/// The cycle is `send` for each packet, `receive` until it returns `None`,
/// and `flush` once after the last packet, then `receive` until `None`.
/// Frames already decoded stay queued until `receive` takes them, so a
/// caller that sends without receiving holds every frame in device memory.
pub trait Decoder<R: Runtime> {
    /// Passes one packet in decode order: one container sample, or one
    /// Annex-B access unit or AV1 temporal unit when the decoder takes raw
    /// input. `pts` is the packet's presentation timestamp in any time base.
    /// The frame that the packet carries returns it unchanged.
    ///
    /// # Errors
    ///
    /// A [`DecodeError`] that names the failed step, from the bitstream
    /// filter, the parser, or the hardware decoder. A call after
    /// [`Decoder::flush`] returns [`DecodeError::AfterFlush`].
    fn send(&mut self, packet: &[u8], pts: i64) -> Result<(), DecodeError>;

    /// Signals the end of the stream and decodes every picture the decoder
    /// still holds. `send` refuses packets after this call.
    ///
    /// # Errors
    ///
    /// As [`Decoder::send`].
    fn flush(&mut self) -> Result<(), DecodeError>;

    /// The next frame in display order, or `None` when no frame is ready.
    fn receive(&mut self) -> Option<DecodedFrame<R>>;
}
