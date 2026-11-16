//! The decoder wrapper that crops each frame.

use std::collections::VecDeque;

use numr::runtime::Runtime;

use super::rect::CropRect;
use super::window::crop_frame;
use crate::decode::frame::DecodedFrame;
use crate::decode::traits::Decoder;
use crate::error::DecodeError;
use crate::route::FrameRoute;

/// A decoder whose frames come out cropped by `crop`.
///
/// A Matroska track can state a crop, which `DecodeConfig::crop` carries and
/// no decoder applies. Wrap the decoder in this type when it holds one, before
/// passing it to the pump.
///
/// Each frame keeps the layout [`DecodedFrame`] documents at the cropped
/// display size, with dense planes in the same
/// [`PlaneForm`](crate::decode::PlaneForm). The crop runs on the frame's
/// device, so the wrapped decoder's [`FrameRoute`] stays the route.
///
/// A frame that the crop leaves empty gives
/// [`DecodeError::CropOutsideFrame`]. `receive` cannot return an error, so it
/// returns `None` and keeps the error. The next `send` or `flush` returns it.
/// `flush` crops every remaining frame, so an error at the end of the stream
/// also reaches the caller.
pub struct Cropped<R: Runtime, D> {
    inner: D,
    crop: CropRect,
    /// Frames `flush` cropped, in display order.
    ready: VecDeque<DecodedFrame<R>>,
    error: Option<DecodeError>,
}

impl<R: Runtime, D: Decoder<R>> Cropped<R, D> {
    /// Wraps `inner`, whose frames this crops by `crop`.
    pub fn new(inner: D, crop: CropRect) -> Self {
        Self {
            inner,
            crop,
            ready: VecDeque::new(),
            error: None,
        }
    }

    /// The crop.
    pub fn crop(&self) -> CropRect {
        self.crop
    }

    /// The wrapped decoder.
    pub fn into_inner(self) -> D {
        self.inner
    }

    fn stored_error(&self) -> Result<(), DecodeError> {
        match &self.error {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }
}

impl<R: Runtime, D: Decoder<R>> Decoder<R> for Cropped<R, D> {
    fn send(&mut self, packet: &[u8], pts: i64) -> Result<(), DecodeError> {
        self.stored_error()?;
        self.inner.send(packet, pts)
    }

    fn flush(&mut self) -> Result<(), DecodeError> {
        self.stored_error()?;
        self.inner.flush()?;
        while let Some(frame) = self.inner.receive() {
            match crop_frame(frame, self.crop) {
                Ok(frame) => self.ready.push_back(frame),
                Err(error) => {
                    self.error = Some(error.clone());
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    fn receive(&mut self) -> Option<DecodedFrame<R>> {
        if let Some(frame) = self.ready.pop_front() {
            return Some(frame);
        }
        if self.error.is_some() {
            return None;
        }
        match crop_frame(self.inner.receive()?, self.crop) {
            Ok(frame) => Some(frame),
            Err(error) => {
                self.error = Some(error);
                None
            }
        }
    }

    fn route(&self) -> FrameRoute {
        self.inner.route()
    }
}

#[cfg(test)]
mod tests {
    use numr::runtime::cpu::CpuRuntime;

    use super::*;
    use crate::decode::crop::test_frames::{frame, rect};
    use crate::decode::{PixelFormat, PlaneForm};
    use crate::geometry::Dimensions;

    /// A decoder that returns queued frames, for the wrapper's error path.
    struct Queue(VecDeque<DecodedFrame<CpuRuntime>>);

    impl Decoder<CpuRuntime> for Queue {
        fn send(&mut self, _packet: &[u8], _pts: i64) -> Result<(), DecodeError> {
            Ok(())
        }

        fn flush(&mut self) -> Result<(), DecodeError> {
            Ok(())
        }

        fn receive(&mut self) -> Option<DecodedFrame<CpuRuntime>> {
            self.0.pop_front()
        }

        fn route(&self) -> FrameRoute {
            FrameRoute::Device
        }
    }

    fn queued(count: usize) -> Queue {
        Queue(
            (0..count)
                .map(|_| frame(PixelFormat::Nv12, PlaneForm::Samples).0)
                .collect(),
        )
    }

    #[test]
    fn cropped_frames_come_out_of_receive_and_flush() {
        let mut decoder = Cropped::new(queued(2), rect(1, 0, 1, 0));
        let first = decoder.receive().unwrap();
        assert_eq!(first.display(), Dimensions::new(10, 6).unwrap());
        decoder.flush().unwrap();
        assert_eq!(decoder.receive().unwrap().display(), first.display());
        assert!(decoder.receive().is_none());
        assert_eq!(decoder.route(), FrameRoute::Device);
    }

    #[test]
    fn a_receive_error_comes_back_from_the_next_send() {
        let crop = rect(0, 0, 20, 0);
        let mut decoder = Cropped::new(queued(1), crop);
        assert!(decoder.receive().is_none());
        let error = decoder.send(&[1], 0).unwrap_err();
        assert!(
            matches!(error, DecodeError::CropOutsideFrame { width: 11, .. }),
            "{error}"
        );
        assert_eq!(decoder.flush().unwrap_err(), error);
    }

    #[test]
    fn a_flush_error_reaches_the_caller() {
        let mut decoder = Cropped::new(queued(1), rect(0, 7, 0, 0));
        assert!(matches!(
            decoder.flush(),
            Err(DecodeError::CropOutsideFrame { height: 7, .. })
        ));
        assert!(decoder.receive().is_none());
    }
}
