//! The decoded picture buffer: reference marking (8.3.2), unavailable pictures
//! (8.3.3), and the output order process (C.5.2).
//!
//! Pictures live behind [`Arc`], so a reference list or an output holds a picture
//! with no copy of its samples.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::bitstream::hevc_sps_full::SeqParameterSet;
use crate::decode::cpu::dsp::sample::Sample;
use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::motion::col::ColMotion;
use crate::decode::cpu::hevc::picture::{DisplayWindow, Picture};
use crate::decode::cpu::hevc::syntax::records::push_record;

use super::rps::{LongTermPoc, RpsPocs};

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
}

/// The reference marking of a picture (8.3.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Marking {
    Unused,
    ShortTerm,
    LongTerm,
}

/// One picture storage buffer.
#[derive(Clone, Debug)]
struct Entry<S> {
    frame: Arc<Frame<S>>,
    marking: Marking,
    /// "Needed for output".
    needed_for_output: bool,
    /// `PicLatencyCount`.
    latency: u32,
    /// True while the current RPS holds this picture.
    in_rps: bool,
}

/// One entry of `RefPicSetStCurrBefore`, `RefPicSetStCurrAfter`, or `RefPicSetLtCurr`.
#[derive(Clone, Debug)]
pub(crate) struct RefEntry<S> {
    /// `None` for "no reference picture".
    pub(crate) frame: Option<Arc<Frame<S>>>,
    /// The POC the entry names.
    pub(crate) poc: i32,
    /// True for an entry of `RefPicSetLtCurr`.
    pub(crate) long_term: bool,
}

/// The current picture's sets that reference lists read.
#[derive(Clone, Debug)]
pub(crate) struct RefPicSets<S> {
    /// `RefPicSetStCurrBefore`.
    pub(crate) before: Vec<RefEntry<S>>,
    /// `RefPicSetStCurrAfter`.
    pub(crate) after: Vec<RefEntry<S>>,
    /// `RefPicSetLtCurr`.
    pub(crate) long_term: Vec<RefEntry<S>>,
    /// Entries of `RefPicSetStFoll` and `RefPicSetLtFoll` that hold "no reference
    /// picture": the POC, and true for a long-term entry.
    missing_foll: Vec<(i32, bool)>,
}

impl<S> RefPicSets<S> {
    /// Sets with no entries.
    pub(crate) const fn empty() -> Self {
        Self {
            before: Vec::new(),
            after: Vec::new(),
            long_term: Vec::new(),
            missing_foll: Vec::new(),
        }
    }
}

/// The limits of C.5.2 for `HighestTid` equal to the highest sub-layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DpbLimits {
    /// `sps_max_num_reorder_pics[ HighestTid ]`.
    pub(crate) max_num_reorder: usize,
    /// `SpsMaxLatencyPictures[ HighestTid ]`, or `None` when
    /// `sps_max_latency_increase_plus1[ HighestTid ]` is 0.
    pub(crate) max_latency: Option<u32>,
    /// `sps_max_dec_pic_buffering_minus1[ HighestTid ] + 1`.
    pub(crate) max_dec_buffering: usize,
}

impl DpbLimits {
    /// Reads the limits of the highest sub-layer. (7-9) gives `SpsMaxLatencyPictures`.
    pub(crate) fn from_sps(sps: &SeqParameterSet) -> Self {
        let top = usize::from(sps.sps_max_sub_layers_minus1).min(6);
        let reorder = sps.sps_max_num_reorder_pics[top];
        let increase_plus1 = sps.sps_max_latency_increase_plus1[top];
        Self {
            max_num_reorder: reorder as usize,
            max_latency: (increase_plus1 != 0).then(|| reorder.saturating_add(increase_plus1) - 1),
            max_dec_buffering: sps.sps_max_dec_pic_buffering_minus1[top] as usize + 1,
        }
    }
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
pub(crate) struct Dpb<S> {
    entries: Vec<Entry<S>>,
    /// Pictures the bumping process output, oldest first.
    output: VecDeque<Arc<Frame<S>>>,
    /// Buffers with no remaining reader.
    free: Vec<Frame<S>>,
}

impl<S: Sample> Dpb<S> {
    /// An empty buffer.
    pub(crate) const fn new() -> Self {
        Self {
            entries: Vec::new(),
            output: VecDeque::new(),
            free: Vec::new(),
        }
    }

    /// Runs the reference picture set process of 8.3.2, and returns the sets of the
    /// current picture. `irap_no_rasl` marks every picture unused first.
    ///
    /// # Errors
    ///
    /// [`HevcError::Allocation`] when a list does not fit in memory.
    pub(crate) fn apply_rps(
        &mut self,
        rps: &RpsPocs,
        max_poc_lsb: u32,
        irap_no_rasl: bool,
    ) -> Result<RefPicSets<S>, HevcError> {
        for entry in &mut self.entries {
            entry.in_rps = false;
            if irap_no_rasl {
                entry.marking = Marking::Unused;
            }
        }
        let mut sets = RefPicSets::empty();
        // Steps 1 and 2: long-term entries first, because they change the marking the
        // short-term search reads.
        for item in rps.lt_curr.as_slice() {
            let frame = self.take_long_term(item, max_poc_lsb);
            let entry = RefEntry {
                frame,
                poc: item.poc,
                long_term: true,
            };
            push_record(&mut sets.long_term, entry, "reference picture set")?;
        }
        for item in rps.lt_foll.as_slice() {
            if self.take_long_term(item, max_poc_lsb).is_none() {
                push_record(
                    &mut sets.missing_foll,
                    (item.poc, true),
                    "reference picture set",
                )?;
            }
        }
        for &poc in rps.st_curr_before.as_slice() {
            let frame = self.take_short_term(poc);
            let entry = RefEntry {
                frame,
                poc,
                long_term: false,
            };
            push_record(&mut sets.before, entry, "reference picture set")?;
        }
        for &poc in rps.st_curr_after.as_slice() {
            let frame = self.take_short_term(poc);
            let entry = RefEntry {
                frame,
                poc,
                long_term: false,
            };
            push_record(&mut sets.after, entry, "reference picture set")?;
        }
        for &poc in rps.st_foll.as_slice() {
            if self.take_short_term(poc).is_none() {
                push_record(
                    &mut sets.missing_foll,
                    (poc, false),
                    "reference picture set",
                )?;
            }
        }
        // Step 4.
        for entry in &mut self.entries {
            if !entry.in_rps {
                entry.marking = Marking::Unused;
            }
        }
        Ok(sets)
    }

    /// (8-6) for one entry, and step 2 of 8.3.2.
    fn take_long_term(&mut self, item: &LongTermPoc, max_poc_lsb: u32) -> Option<Arc<Frame<S>>> {
        let mask = i64::from(max_poc_lsb) - 1;
        let found = self.entries.iter_mut().find(|entry| {
            let poc = i64::from(entry.frame.poc);
            let matches = if item.msb_present {
                poc == i64::from(item.poc)
            } else {
                poc & mask == i64::from(item.poc)
            };
            entry.marking != Marking::Unused && matches
        })?;
        found.marking = Marking::LongTerm;
        found.in_rps = true;
        Some(Arc::clone(&found.frame))
    }

    /// (8-7) for one entry.
    fn take_short_term(&mut self, poc: i32) -> Option<Arc<Frame<S>>> {
        let found = self
            .entries
            .iter_mut()
            .find(|entry| entry.marking == Marking::ShortTerm && entry.frame.poc == poc)?;
        found.in_rps = true;
        Some(Arc::clone(&found.frame))
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
        if irap_no_rasl {
            self.empty(no_output_of_prior_pics);
            return;
        }
        self.remove_finished();
        while self.must_bump(limits) || self.entries.len() >= limits.max_dec_buffering {
            if !self.bump() {
                break;
            }
        }
    }

    /// Empties the buffer, with or without output of the waiting pictures.
    fn empty(&mut self, discard: bool) {
        if !discard {
            while self.bump() {}
        }
        for entry in self.entries.drain(..) {
            recycle(&mut self.free, entry.frame);
        }
    }

    /// Ends a sequence for a change of sample type. The other sample type's pictures
    /// leave the buffer, as they do at an IRAP picture with `NoRaslOutputFlag` 1.
    pub(crate) fn close(&mut self, discard: bool) {
        self.empty(discard);
    }

    /// Outputs every waiting picture, at the end of the stream.
    pub(crate) fn flush(&mut self) {
        while self.bump() {}
    }

    /// C.5.2.3 for the finished current picture, with C.3.4 before it. A picture with
    /// `PicOutputFlag` 0 stores as "not needed for output".
    pub(crate) fn insert(&mut self, frame: Frame<S>, output: bool, limits: &DpbLimits) {
        let poc = frame.poc;
        if output {
            for entry in &mut self.entries {
                if entry.needed_for_output && entry.frame.poc > poc {
                    entry.latency = entry.latency.saturating_add(1);
                }
            }
        }
        self.entries.push(Entry {
            frame: Arc::new(frame),
            marking: Marking::ShortTerm,
            needed_for_output: output,
            latency: 0,
            in_rps: false,
        });
        while self.must_bump(limits) {
            if !self.bump() {
                break;
            }
        }
    }

    /// The first two conditions of C.5.2.2 and C.5.2.3.
    fn must_bump(&self, limits: &DpbLimits) -> bool {
        let waiting = self
            .entries
            .iter()
            .filter(|entry| entry.needed_for_output)
            .count();
        if waiting > limits.max_num_reorder {
            return true;
        }
        limits.max_latency.is_some_and(|max_latency| {
            self.entries
                .iter()
                .any(|entry| entry.needed_for_output && entry.latency >= max_latency)
        })
    }

    /// C.5.2.4: outputs the waiting picture with the smallest POC. Returns false when
    /// no picture waits.
    fn bump(&mut self) -> bool {
        let first = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.needed_for_output)
            .min_by_key(|(_, entry)| entry.frame.poc)
            .map(|(index, _)| index);
        let Some(index) = first else {
            return false;
        };
        if let Some(entry) = self.entries.get_mut(index) {
            entry.needed_for_output = false;
            self.output.push_back(Arc::clone(&entry.frame));
        }
        self.remove_finished();
        true
    }

    /// Removes each picture that is "not needed for output" and "unused for
    /// reference".
    fn remove_finished(&mut self) {
        let mut index = 0;
        while let Some(entry) = self.entries.get(index) {
            if !entry.needed_for_output && entry.marking == Marking::Unused {
                let removed = self.entries.swap_remove(index);
                recycle(&mut self.free, removed.frame);
            } else {
                index += 1;
            }
        }
    }

    /// 8.3.3: a generated picture for each missing entry of the "Foll" lists.
    ///
    /// # Errors
    ///
    /// [`HevcError::Allocation`] when a picture does not fit in memory.
    pub(crate) fn generate_unavailable(
        &mut self,
        sets: &RefPicSets<S>,
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
            let entry = Entry {
                frame: Arc::new(frame),
                marking: if long_term {
                    Marking::LongTerm
                } else {
                    Marking::ShortTerm
                },
                needed_for_output: false,
                latency: 0,
                in_rps: true,
            };
            push_record(&mut self.entries, entry, "decoded picture buffer")?;
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
                }
            }
        };
        frame.picture.set_window(format.window)?;
        Ok(frame)
    }

    /// The oldest output picture.
    pub(crate) fn pop_output(&mut self) -> Option<Arc<Frame<S>>> {
        self.output.pop_front()
    }
}

/// Keeps the buffer of a picture nothing else holds.
fn recycle<S>(free: &mut Vec<Frame<S>>, frame: Arc<Frame<S>>) {
    if free.len() < MAX_FREE_FRAMES
        && let Ok(frame) = Arc::try_unwrap(frame)
    {
        free.push(frame);
    }
}

#[cfg(test)]
#[path = "buffer_tests.rs"]
mod tests;
