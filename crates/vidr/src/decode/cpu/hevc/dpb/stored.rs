//! How the decoded picture buffer holds a picture. The buffer reads only a picture's POC,
//! so it can hold a picture that is still decoding.

use std::sync::Arc;

use crate::decode::cpu::dsp::sample::Sample;

use super::buffer::Frame;

/// What [`StoredPicture::reclaim`] finds.
pub(crate) enum Reclaim<P: StoredPicture> {
    /// The buffer has no other holder.
    Buffer(Box<Frame<P::Sample>>),
    /// Another holder keeps the picture. A later call can succeed.
    Busy(P),
    /// No buffer comes back from this picture.
    Lost,
}

/// A picture in the decoded picture buffer.
pub(crate) trait StoredPicture: Clone {
    /// The sample type of the picture.
    type Sample: Sample;

    /// True when the buffer retries [`StoredPicture::reclaim`] on a [`Reclaim::Busy`]
    /// picture at each new picture.
    const WAITS_TO_RECLAIM: bool;

    /// A picture that is finished now.
    fn finished(frame: Frame<Self::Sample>) -> Self;

    /// `PicOrderCntVal`.
    fn poc(&self) -> i32;

    /// The picture's buffer, for a new picture of the same size.
    fn reclaim(self) -> Reclaim<Self>;
}

impl<S: Sample> StoredPicture for Arc<Frame<S>> {
    type Sample = S;

    const WAITS_TO_RECLAIM: bool = false;

    fn finished(frame: Frame<S>) -> Self {
        Self::new(frame)
    }

    fn poc(&self) -> i32 {
        self.poc
    }

    fn reclaim(self) -> Reclaim<Self> {
        match Self::try_unwrap(self) {
            Ok(frame) => Reclaim::Buffer(Box::new(frame)),
            Err(_) => Reclaim::Lost,
        }
    }
}
