//! The decoded picture buffer with samples: generated pictures (8.3.3) and buffer reuse, on
//! the sample-free [`RefBuffer`] of marking and output.
//!
//! Pictures live behind [`Arc`], so a reference list or an output holds a picture
//! with no copy of its samples. The buffer is generic over [`StoredPicture`], so the
//! threaded path keeps handles to pictures that later finish.

use std::sync::Arc;

use crate::decode::cpu::dsp::sample::Sample;
use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::motion::col::ColMotion;
use crate::decode::cpu::hevc::picture::{DisplayWindow, Picture};
#[cfg(test)]
use crate::decode::hevc_order::buffer::Marking;
use crate::decode::hevc_order::buffer::RefBuffer;
pub(crate) use crate::decode::hevc_order::buffer::{DpbLimits, RefEntry, RefPicSets};

#[cfg(test)]
use super::rps::LongTermPoc;
use super::rps::RpsPocs;
use super::stored::{Reclaim, StoredPicture};

/// Finished buffers kept for reuse.
const MAX_FREE_FRAMES: usize = 4;

/// A finished picture with what later pictures read from it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Frame<S> {
    /// The samples after the in-loop filters.
    pub(crate) picture: Picture<S>,
    /// The motion that 8.5.3.2.9 reads.
    pub(crate) col: ColMotion,
    /// `PicOrderCntVal`.
    pub(crate) poc: i32,
    /// How many pictures started decoding before this one. A generated picture
    /// never leaves as output, so its value holds no meaning.
    pub(crate) decode_order: u64,
}

/// The picture a generated frame copies its size from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FrameFormat {
    /// `pic_width_in_luma_samples`.
    pub(crate) width: usize,
    /// `pic_height_in_luma_samples`.
    pub(crate) height: usize,
    /// `BitDepthY` and `BitDepthC`.
    pub(crate) bit_depth: u32,
    /// The conformance window in luma samples.
    pub(crate) window: DisplayWindow,
}

/// The decoded picture buffer of one sample type.
#[derive(Clone, Debug)]
pub(crate) struct Dpb<S, P = Arc<Frame<S>>> {
    core: RefBuffer<P>,
    /// Buffers with no remaining reader.
    free: Vec<Frame<S>>,
    /// Removed pictures that other holders still keep.
    waiting: Vec<P>,
}

impl<S: Sample, P: StoredPicture<Sample = S>> Dpb<S, P> {
    /// An empty buffer.
    pub(crate) const fn new() -> Self {
        Self {
            core: RefBuffer::new(),
            free: Vec::new(),
            waiting: Vec::new(),
        }
    }

    /// Keeps the buffer of each picture that left [`Self::core`].
    fn reclaim_removed(&mut self) {
        for picture in self.core.take_removed() {
            recycle(&mut self.free, &mut self.waiting, picture);
        }
    }

    /// Runs the reference picture set process of 8.3.2, and returns the sets of the
    /// current picture. `irap_no_rasl` marks every picture unused first.
    pub(crate) fn apply_rps(
        &mut self,
        rps: &RpsPocs,
        max_poc_lsb: u32,
        irap_no_rasl: bool,
    ) -> RefPicSets<P> {
        self.core.apply_rps(rps, max_poc_lsb, irap_no_rasl)
    }

    /// C.5.2.2: output and removal before the current picture decodes.
    /// `no_output_of_prior_pics` is `NoOutputOfPriorPicsFlag`, read only with
    /// `irap_no_rasl`.
    pub(crate) fn before_picture(
        &mut self,
        irap_no_rasl: bool,
        no_output_of_prior_pics: bool,
        limits: &DpbLimits,
    ) {
        self.core
            .before_picture(irap_no_rasl, no_output_of_prior_pics, limits);
        self.reclaim_removed();
    }

    /// Ends a sequence for a change of sample type. The other sample type's pictures
    /// leave the buffer, as they do at an IRAP picture with `NoRaslOutputFlag` 1.
    pub(crate) fn close(&mut self, discard: bool) {
        self.core.empty(discard);
        self.reclaim_removed();
    }

    /// Outputs every waiting picture, at the end of the stream.
    pub(crate) fn flush(&mut self) {
        self.core.flush();
        self.reclaim_removed();
    }

    /// C.5.2.3 for the finished current picture, with C.3.4 before it. A picture with
    /// `PicOutputFlag` 0 stores as "not needed for output".
    pub(crate) fn insert(&mut self, frame: Frame<S>, output: bool, limits: &DpbLimits) {
        self.insert_picture(P::finished(frame), output, limits);
    }

    /// [`Self::insert`] for a picture the buffer holds in its stored form.
    pub(crate) fn insert_picture(&mut self, picture: P, output: bool, limits: &DpbLimits) {
        self.core.insert_picture(picture, output, limits);
        self.reclaim_removed();
    }

    /// 8.3.3: a generated picture for each missing entry of the "Foll" lists.
    ///
    /// # Errors
    ///
    /// [`HevcError::Allocation`] when a picture does not fit in memory.
    pub(crate) fn generate_unavailable(
        &mut self,
        sets: &RefPicSets<P>,
        format: &FrameFormat,
    ) -> Result<(), HevcError> {
        for &(poc, long_term) in &sets.missing_foll {
            let mut frame = self.take_frame(format)?;
            frame.poc = poc;
            let value = 1i32 << (format.bit_depth - 1);
            let sample = S::clipped(value, (1 << format.bit_depth) - 1);
            for c_idx in 0..3 {
                frame.picture.plane_mut(c_idx).samples_mut().fill(sample);
            }
            // 8.3.3.2: CuPredMode is MODE_INTRA everywhere.
            frame
                .col
                .prepare(format.width as u32, format.height as u32)?;
            self.core.insert_generated(P::finished(frame), long_term);
        }
        Ok(())
    }

    /// A buffer for a new picture: a reused one of the same size, or a new one. The
    /// samples of a reused buffer are stale, and the slices overwrite every CTB.
    ///
    /// # Errors
    ///
    /// - [`HevcError::Allocation`] when a picture does not fit in memory.
    /// - [`HevcError::DisplayWindow`] for a conformance window that does not fit.
    pub(crate) fn take_frame(&mut self, format: &FrameFormat) -> Result<Frame<S>, HevcError> {
        for picture in std::mem::take(&mut self.waiting) {
            recycle(&mut self.free, &mut self.waiting, picture);
        }
        let reusable = self.free.iter().position(|frame| {
            frame.picture.luma().width() == format.width
                && frame.picture.luma().height() == format.height
        });
        let mut frame = match reusable {
            Some(index) => self.free.swap_remove(index),
            None => {
                self.free.clear();
                Frame {
                    picture: Picture::new(format.width, format.height)?,
                    col: ColMotion::new(),
                    poc: 0,
                    decode_order: 0,
                }
            }
        };
        frame.picture.set_window(format.window)?;
        Ok(frame)
    }

    /// The oldest output picture.
    pub(crate) fn pop_output(&mut self) -> Option<P> {
        self.core.pop_output()
    }
}

/// Keeps the buffer of a picture nothing else holds. A picture that another holder keeps
/// waits for a later call, when its type retries.
fn recycle<S, P: StoredPicture<Sample = S>>(
    free: &mut Vec<Frame<S>>,
    waiting: &mut Vec<P>,
    picture: P,
) {
    match picture.reclaim() {
        Reclaim::Buffer(frame) if free.len() < MAX_FREE_FRAMES => free.push(*frame),
        Reclaim::Busy(picture) if P::WAITS_TO_RECLAIM => waiting.push(picture),
        _ => {}
    }
}

#[cfg(test)]
#[path = "buffer_tests.rs"]
mod tests;
