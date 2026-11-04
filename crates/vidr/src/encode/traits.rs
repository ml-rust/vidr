//! The encoder interface every backend implements: NVENC and the CPU encoder now,
//! and Vulkan Video later.

use numr::runtime::Runtime;

use super::frame::{EncodeFrame, EncodedPacket};
use crate::error::EncodeError;
use crate::route::FrameRoute;

/// A video encoder that takes frames in display order and returns packets in
/// decode order.
///
/// The cycle is `send` for each frame, `receive` until it returns `None`, and
/// `flush` once after the last frame, then `receive` until `None`. A backend
/// that reorders frames for B-frames holds them, so `receive` can return `None`
/// after a `send` while the stream is not finished.
pub trait Encoder<R: Runtime> {
    /// Passes one frame in display order. `pts` and `duration` are in the
    /// config's time base. `pts` must increase with each frame, and `duration`
    /// must be above zero.
    ///
    /// # Errors
    ///
    /// An [`EncodeError`] that names the refused input or the failed step. A
    /// call after [`Encoder::flush`] returns [`EncodeError::AfterFlush`].
    fn send(
        &mut self,
        frame: &EncodeFrame<'_, R>,
        pts: i64,
        duration: i64,
    ) -> Result<(), EncodeError>;

    /// Signals the end of the stream and encodes every frame the encoder still
    /// holds. `send` refuses frames after this call.
    ///
    /// # Errors
    ///
    /// As [`Encoder::send`].
    fn flush(&mut self) -> Result<(), EncodeError>;

    /// The next packet in decode order, or `None` when no packet is ready.
    fn receive(&mut self) -> Option<EncodedPacket>;

    /// The `avcC`, `hvcC`, or `av1C` box payload, or `None` before the first
    /// packet exists.
    fn config_record(&self) -> Option<&[u8]>;

    /// How this encoder moves each frame. The pump counts a host copy from it.
    fn route(&self) -> FrameRoute;
}

impl<R: Runtime, T: Encoder<R> + ?Sized> Encoder<R> for Box<T> {
    fn send(
        &mut self,
        frame: &EncodeFrame<'_, R>,
        pts: i64,
        duration: i64,
    ) -> Result<(), EncodeError> {
        (**self).send(frame, pts, duration)
    }

    fn flush(&mut self) -> Result<(), EncodeError> {
        (**self).flush()
    }

    fn receive(&mut self) -> Option<EncodedPacket> {
        (**self).receive()
    }

    fn config_record(&self) -> Option<&[u8]> {
        (**self).config_record()
    }

    fn route(&self) -> FrameRoute {
        (**self).route()
    }
}
