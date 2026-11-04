//! Reference marking (8.3.2) and the output order process (C.5.2), over any stored picture.
//!
//! The buffer reads only a picture's POC. A picture stored here can be a frame of samples
//! or a slot of a hardware session. The buffer keeps each picture that leaves it in
//! [`RefBuffer::take_removed`], so its owner frees what the picture holds.

use std::collections::VecDeque;

use crate::bitstream::hevc_sps_full::SeqParameterSet;

use super::rps::{LongTermPoc, RpsPocs};

/// A picture in the buffer.
pub(crate) trait BufferedPicture: Clone {
    /// `PicOrderCntVal`.
    fn poc(&self) -> i32;
}

/// The reference marking of a picture (8.3.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Marking {
    Unused,
    ShortTerm,
    LongTerm,
}

/// One picture storage buffer.
#[derive(Clone, Debug)]
pub(crate) struct Entry<P> {
    pub(crate) frame: P,
    pub(crate) marking: Marking,
    /// "Needed for output".
    pub(crate) needed_for_output: bool,
    /// `PicLatencyCount`.
    latency: u32,
    /// True while the current RPS holds this picture.
    in_rps: bool,
}

/// One entry of `RefPicSetStCurrBefore`, `RefPicSetStCurrAfter`, or `RefPicSetLtCurr`.
#[derive(Clone, Debug)]
pub(crate) struct RefEntry<P> {
    /// `None` for "no reference picture".
    pub(crate) frame: Option<P>,
    /// The POC the entry names.
    pub(crate) poc: i32,
    /// True for an entry of `RefPicSetLtCurr`.
    pub(crate) long_term: bool,
}

/// The current picture's sets that reference lists read.
#[derive(Clone, Debug)]
pub(crate) struct RefPicSets<P> {
    /// `RefPicSetStCurrBefore`.
    pub(crate) before: Vec<RefEntry<P>>,
    /// `RefPicSetStCurrAfter`.
    pub(crate) after: Vec<RefEntry<P>>,
    /// `RefPicSetLtCurr`.
    pub(crate) long_term: Vec<RefEntry<P>>,
    /// Entries of `RefPicSetStFoll` and `RefPicSetLtFoll` that hold "no reference
    /// picture": the POC, and true for a long-term entry.
    pub(crate) missing_foll: Vec<(i32, bool)>,
}

impl<P> RefPicSets<P> {
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

/// The marking and output state of the decoded picture buffer.
#[derive(Clone, Debug)]
pub(crate) struct RefBuffer<P> {
    pub(crate) entries: Vec<Entry<P>>,
    /// Pictures the bumping process output, oldest first.
    output: VecDeque<P>,
    /// Pictures that left the buffer since the last [`Self::take_removed`].
    removed: Vec<P>,
}

impl<P: BufferedPicture> RefBuffer<P> {
    /// An empty buffer.
    pub(crate) const fn new() -> Self {
        Self {
            entries: Vec::new(),
            output: VecDeque::new(),
            removed: Vec::new(),
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
            sets.long_term.push(RefEntry {
                frame,
                poc: item.poc,
                long_term: true,
            });
        }
        for item in rps.lt_foll.as_slice() {
            if self.take_long_term(item, max_poc_lsb).is_none() {
                sets.missing_foll.push((item.poc, true));
            }
        }
        for &poc in rps.st_curr_before.as_slice() {
            let frame = self.take_short_term(poc);
            sets.before.push(RefEntry {
                frame,
                poc,
                long_term: false,
            });
        }
        for &poc in rps.st_curr_after.as_slice() {
            let frame = self.take_short_term(poc);
            sets.after.push(RefEntry {
                frame,
                poc,
                long_term: false,
            });
        }
        for &poc in rps.st_foll.as_slice() {
            if self.take_short_term(poc).is_none() {
                sets.missing_foll.push((poc, false));
            }
        }
        // Step 4.
        for entry in &mut self.entries {
            if !entry.in_rps {
                entry.marking = Marking::Unused;
            }
        }
        sets
    }

    /// (8-6) for one entry, and step 2 of 8.3.2.
    fn take_long_term(&mut self, item: &LongTermPoc, max_poc_lsb: u32) -> Option<P> {
        let mask = i64::from(max_poc_lsb) - 1;
        let found = self.entries.iter_mut().find(|entry| {
            let poc = i64::from(entry.frame.poc());
            let matches = if item.msb_present {
                poc == i64::from(item.poc)
            } else {
                poc & mask == i64::from(item.poc)
            };
            entry.marking != Marking::Unused && matches
        })?;
        found.marking = Marking::LongTerm;
        found.in_rps = true;
        Some(found.frame.clone())
    }

    /// (8-7) for one entry.
    fn take_short_term(&mut self, poc: i32) -> Option<P> {
        let found = self
            .entries
            .iter_mut()
            .find(|entry| entry.marking == Marking::ShortTerm && entry.frame.poc() == poc)?;
        found.in_rps = true;
        Some(found.frame.clone())
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
    pub(crate) fn empty(&mut self, discard: bool) {
        if !discard {
            while self.bump() {}
        }
        self.removed
            .extend(self.entries.drain(..).map(|entry| entry.frame));
    }

    /// Outputs every waiting picture, at the end of the stream.
    pub(crate) fn flush(&mut self) {
        while self.bump() {}
    }

    /// C.5.2.3 for the finished current picture, with C.3.4 before it. A picture with
    /// `PicOutputFlag` 0 stores as "not needed for output".
    pub(crate) fn insert_picture(&mut self, picture: P, output: bool, limits: &DpbLimits) {
        let poc = picture.poc();
        if output {
            for entry in &mut self.entries {
                if entry.needed_for_output && entry.frame.poc() > poc {
                    entry.latency = entry.latency.saturating_add(1);
                }
            }
        }
        self.entries.push(Entry {
            frame: picture,
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

    /// Stores a picture of 8.3.3, which never leaves as output.
    pub(crate) fn insert_generated(&mut self, picture: P, long_term: bool) {
        self.entries.push(Entry {
            frame: picture,
            marking: if long_term {
                Marking::LongTerm
            } else {
                Marking::ShortTerm
            },
            needed_for_output: false,
            latency: 0,
            in_rps: true,
        });
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
            .min_by_key(|(_, entry)| entry.frame.poc())
            .map(|(index, _)| index);
        let Some(index) = first else {
            return false;
        };
        if let Some(entry) = self.entries.get_mut(index) {
            entry.needed_for_output = false;
            self.output.push_back(entry.frame.clone());
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
                self.removed.push(removed.frame);
            } else {
                index += 1;
            }
        }
    }

    /// The oldest output picture.
    pub(crate) fn pop_output(&mut self) -> Option<P> {
        self.output.pop_front()
    }

    /// Every picture that left the buffer since the last call.
    pub(crate) fn take_removed(&mut self) -> Vec<P> {
        std::mem::take(&mut self.removed)
    }
}
