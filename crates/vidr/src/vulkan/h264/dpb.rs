//! Reference marking for frames, ITU-T H.264 8.2.5, and the DPB slot of each picture.

/// `memory_management_control_operation` with its syntax.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mmco {
    /// Operation 1.
    UnmarkShort { difference_of_pic_nums_minus1: u32 },
    /// Operation 2.
    UnmarkLong { long_term_pic_num: u32 },
    /// Operation 3.
    ShortToLong {
        difference_of_pic_nums_minus1: u32,
        long_term_frame_idx: u32,
    },
    /// Operation 4.
    MaxLongTermIndex { max_long_term_frame_idx_plus1: u32 },
    /// Operation 5.
    UnmarkAll,
    /// Operation 6.
    CurrentToLong { long_term_frame_idx: u32 },
}

/// How a picture marks references.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Marking {
    /// `nal_ref_idc` 0.
    NonReference,
    /// An IDR picture, with `long_term_reference_flag`.
    Idr { long_term: bool },
    /// `adaptive_ref_pic_marking_mode_flag` 0.
    SlidingWindow,
    /// `adaptive_ref_pic_marking_mode_flag` 1.
    Adaptive(Vec<Mmco>),
}

impl Marking {
    /// Whether the marking holds operation 5.
    pub(super) fn has_mmco5(&self) -> bool {
        matches!(self, Self::Adaptive(operations) if operations.contains(&Mmco::UnmarkAll))
    }
}

/// One frame held as a reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RefFrame {
    pub(super) slot: u32,
    /// The picture's number in decode order, shared with the output queue.
    pub(super) id: u64,
    pub(super) frame_num: u32,
    /// `LongTermFrameIdx`, for a long-term reference.
    pub(super) long_term: Option<u32>,
    pub(super) poc: [i32; 2],
}

/// The picture just decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CurrentPicture {
    pub(super) slot: u32,
    pub(super) id: u64,
    pub(super) frame_num: u32,
    pub(super) poc: [i32; 2],
}

/// What marking leaves.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct MarkOutcome {
    /// Slots that hold no reference now.
    pub(super) freed: Vec<u32>,
    /// The picture's order count after operation 5, or `None` with no operation 5.
    pub(super) mmco5_poc: Option<[i32; 2]>,
}

/// The reference frame `current` becomes once `marking` runs, by 8.2.1, 8.2.5.1, and
/// 7.4.3. An IDR picture or operation 6 makes it long-term. Operation 5 sets `frame_num`
/// to 0 and rebases the order count. A decode states this for its setup slot, so the slot
/// holds what later pictures name. The picture's own decode keeps the values in its header.
pub(super) fn marked_frame(current: CurrentPicture, marking: &Marking) -> RefFrame {
    let long_term = match marking {
        Marking::Idr { long_term: true } => Some(0),
        Marking::Adaptive(operations) => {
            operations
                .iter()
                .rev()
                .find_map(|operation| match operation {
                    Mmco::CurrentToLong {
                        long_term_frame_idx,
                    } => Some(*long_term_frame_idx),
                    _ => None,
                })
        }
        _ => None,
    };
    let (frame_num, poc) = if marking.has_mmco5() {
        let smallest = current.poc[0].min(current.poc[1]);
        (
            0,
            [
                current.poc[0].wrapping_sub(smallest),
                current.poc[1].wrapping_sub(smallest),
            ],
        )
    } else {
        (current.frame_num, current.poc)
    };
    RefFrame {
        slot: current.slot,
        id: current.id,
        frame_num,
        long_term,
        poc,
    }
}

/// `FrameNumWrap` of 8.2.4.1 for a frame, against the current `frame_num`.
const fn frame_num_wrap(frame_num: u32, current: u32, max_frame_num: u32) -> i64 {
    if frame_num > current {
        frame_num as i64 - max_frame_num as i64
    } else {
        frame_num as i64
    }
}

/// The reference state of one sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Dpb {
    refs: Vec<RefFrame>,
    slots: u32,
    max_references: u32,
    max_frame_num: u32,
    prev_ref_frame_num: u32,
    started: bool,
}

impl Dpb {
    pub(super) fn new(slots: u32, max_references: u32, max_frame_num: u32) -> Self {
        Self {
            refs: Vec::with_capacity(slots as usize),
            slots,
            max_references,
            max_frame_num,
            prev_ref_frame_num: 0,
            started: false,
        }
    }

    /// Every reference.
    pub(super) fn references(&self) -> &[RefFrame] {
        &self.refs
    }

    /// Whether a picture decoded yet.
    pub(super) const fn started(&self) -> bool {
        self.started
    }

    /// Whether the frame with this id is a reference.
    pub(super) fn holds(&self, id: u64) -> bool {
        self.refs.iter().any(|frame| frame.id == id)
    }

    /// The lowest slot with no reference.
    pub(super) fn free_slot(&self) -> Option<u32> {
        (0..self.slots).find(|&slot| !self.refs.iter().any(|frame| frame.slot == slot))
    }

    /// Whether `frame_num` skips frame numbers, per 7.4.3.
    pub(super) const fn has_gap(&self, frame_num: u32) -> bool {
        self.started
            && frame_num != self.prev_ref_frame_num
            && frame_num != (self.prev_ref_frame_num + 1) % self.max_frame_num
    }

    /// Applies a picture's marking after its decode.
    ///
    /// # Errors
    ///
    /// The reason when the references no longer fit `Max(max_num_ref_frames, 1)`.
    pub(super) fn mark(
        &mut self,
        current: CurrentPicture,
        marking: &Marking,
    ) -> Result<MarkOutcome, String> {
        let mut outcome = MarkOutcome::default();
        match marking {
            Marking::NonReference => {}
            Marking::Idr { .. } => self.remove_where(&mut outcome, |_| true),
            Marking::SlidingWindow => self.sliding_window(current.frame_num, &mut outcome),
            Marking::Adaptive(operations) => {
                for operation in operations {
                    self.apply(*operation, current.frame_num, &mut outcome);
                }
            }
        }
        let frame = marked_frame(current, marking);
        if marking.has_mmco5() {
            outcome.mmco5_poc = Some(frame.poc);
        }
        if *marking != Marking::NonReference {
            self.prev_ref_frame_num = frame.frame_num;
            self.refs.push(frame);
        }
        self.started = true;
        if self.refs.len() > self.max_references as usize {
            return Err(format!(
                "marking leaves {} references, and the SPS allows {}",
                self.refs.len(),
                self.max_references
            ));
        }
        Ok(outcome)
    }

    fn remove_where(&mut self, outcome: &mut MarkOutcome, gone: impl Fn(&RefFrame) -> bool) {
        let freed = &mut outcome.freed;
        self.refs.retain(|frame| {
            if gone(frame) {
                freed.push(frame.slot);
                false
            } else {
                true
            }
        });
    }

    /// 8.2.5.3.
    fn sliding_window(&mut self, current: u32, outcome: &mut MarkOutcome) {
        if self.refs.len() < self.max_references as usize {
            return;
        }
        let oldest = self
            .refs
            .iter()
            .filter(|frame| frame.long_term.is_none())
            .min_by_key(|frame| frame_num_wrap(frame.frame_num, current, self.max_frame_num))
            .map(|frame| frame.id);
        if let Some(id) = oldest {
            self.remove_where(outcome, |frame| frame.id == id);
        }
    }

    /// 8.2.5.4. An operation that names an absent picture does nothing, so a stream that
    /// starts at a recovery point decodes. [`marked_frame`] gives the current picture's own
    /// result of operations 5 and 6.
    fn apply(&mut self, operation: Mmco, current: u32, outcome: &mut MarkOutcome) {
        let max = self.max_frame_num;
        let pic_num_x = |difference: u32| i64::from(current) - (i64::from(difference) + 1);
        let wrap = move |frame_num: u32| frame_num_wrap(frame_num, current, max);
        match operation {
            Mmco::UnmarkShort {
                difference_of_pic_nums_minus1,
            } => {
                let target = pic_num_x(difference_of_pic_nums_minus1);
                self.remove_where(outcome, |frame| {
                    frame.long_term.is_none() && wrap(frame.frame_num) == target
                });
            }
            Mmco::UnmarkLong { long_term_pic_num } => {
                self.remove_where(outcome, |frame| frame.long_term == Some(long_term_pic_num));
            }
            Mmco::ShortToLong {
                difference_of_pic_nums_minus1,
                long_term_frame_idx,
            } => {
                let target = pic_num_x(difference_of_pic_nums_minus1);
                let is_target =
                    |frame: &RefFrame| frame.long_term.is_none() && wrap(frame.frame_num) == target;
                if !self.refs.iter().any(is_target) {
                    return;
                }
                self.remove_where(outcome, |frame| {
                    frame.long_term == Some(long_term_frame_idx)
                });
                if let Some(frame) = self.refs.iter_mut().find(|frame| is_target(frame)) {
                    frame.long_term = Some(long_term_frame_idx);
                }
            }
            Mmco::MaxLongTermIndex {
                max_long_term_frame_idx_plus1,
            } => {
                self.remove_where(outcome, |frame| {
                    frame
                        .long_term
                        .is_some_and(|idx| idx >= max_long_term_frame_idx_plus1)
                });
            }
            Mmco::UnmarkAll => self.remove_where(outcome, |_| true),
            Mmco::CurrentToLong {
                long_term_frame_idx,
            } => {
                self.remove_where(outcome, |frame| {
                    frame.long_term == Some(long_term_frame_idx)
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn picture(id: u64, slot: u32, frame_num: u32) -> CurrentPicture {
        CurrentPicture {
            slot,
            id,
            frame_num,
            poc: [id as i32 * 2, id as i32 * 2],
        }
    }

    /// Short-term frame numbers, in the order they sit in the DPB.
    fn short_terms(dpb: &Dpb) -> Vec<u32> {
        dpb.references()
            .iter()
            .filter(|frame| frame.long_term.is_none())
            .map(|frame| frame.frame_num)
            .collect()
    }

    #[test]
    fn the_sliding_window_drops_the_smallest_wrap() {
        let mut dpb = Dpb::new(4, 3, 16);
        for (id, frame_num) in [(0, 14), (1, 15), (2, 0)] {
            dpb.mark(picture(id, id as u32, frame_num), &Marking::SlidingWindow)
                .expect("fits");
        }
        // At frame_num 1, 14 and 15 wrap to -2 and -1, so 14 is the oldest.
        let outcome = dpb
            .mark(picture(3, 3, 1), &Marking::SlidingWindow)
            .expect("fits");
        assert_eq!(outcome.freed, vec![0]);
        assert_eq!(short_terms(&dpb), vec![15, 0, 1]);
    }

    #[test]
    fn a_non_reference_picture_marks_nothing() {
        let mut dpb = Dpb::new(2, 1, 16);
        dpb.mark(picture(0, 0, 0), &Marking::Idr { long_term: false })
            .expect("fits");
        let outcome = dpb
            .mark(picture(1, 1, 1), &Marking::NonReference)
            .expect("fits");
        assert_eq!(outcome, MarkOutcome::default());
        assert_eq!(short_terms(&dpb), vec![0]);
        assert!(!dpb.has_gap(1), "the next reference repeats frame_num 1");
    }

    #[test]
    fn idr_clears_everything_and_can_start_long_term() {
        let mut dpb = Dpb::new(3, 2, 16);
        dpb.mark(picture(0, 0, 0), &Marking::Idr { long_term: false })
            .expect("fits");
        dpb.mark(picture(1, 1, 1), &Marking::SlidingWindow)
            .expect("fits");
        let outcome = dpb
            .mark(picture(2, 2, 0), &Marking::Idr { long_term: true })
            .expect("fits");
        assert_eq!(outcome.freed, vec![0, 1]);
        assert_eq!(dpb.references().len(), 1);
        assert_eq!(dpb.references()[0].long_term, Some(0));
    }

    /// Hand-worked: frames 3, 4, 5 are short-term at current frame_num 6, so PicNum is
    /// the frame number and picNumX = 6 - (difference + 1).
    fn three_shorts() -> Dpb {
        let mut dpb = Dpb::new(5, 4, 16);
        for (id, frame_num) in [(0, 3), (1, 4), (2, 5)] {
            dpb.mark(picture(id, id as u32, frame_num), &Marking::SlidingWindow)
                .expect("fits");
        }
        dpb
    }

    #[test]
    fn mmco1_unmarks_by_pic_num_difference() {
        let mut dpb = three_shorts();
        let marking = Marking::Adaptive(vec![Mmco::UnmarkShort {
            difference_of_pic_nums_minus1: 1,
        }]);
        let outcome = dpb.mark(picture(3, 3, 6), &marking).expect("fits");
        assert_eq!(outcome.freed, vec![1], "picNumX 4 is the frame in slot 1");
        assert_eq!(short_terms(&dpb), vec![3, 5, 6]);
    }

    #[test]
    fn mmco1_finds_a_frame_num_past_the_wrap() {
        let mut dpb = Dpb::new(3, 2, 16);
        dpb.mark(picture(0, 0, 15), &Marking::SlidingWindow)
            .expect("fits");
        // At frame_num 1, frame 15 has PicNum -1, and difference 1 gives 1 - 2 = -1.
        let marking = Marking::Adaptive(vec![Mmco::UnmarkShort {
            difference_of_pic_nums_minus1: 1,
        }]);
        let outcome = dpb.mark(picture(1, 1, 1), &marking).expect("fits");
        assert_eq!(outcome.freed, vec![0]);
    }

    #[test]
    fn mmco3_and_mmco2_move_a_frame_through_long_term() {
        let mut dpb = three_shorts();
        let marking = Marking::Adaptive(vec![Mmco::ShortToLong {
            difference_of_pic_nums_minus1: 2,
            long_term_frame_idx: 1,
        }]);
        dpb.mark(picture(3, 3, 6), &marking).expect("fits");
        assert_eq!(
            dpb.references()[0].long_term,
            Some(1),
            "frame 3 is long-term"
        );
        assert_eq!(short_terms(&dpb), vec![4, 5, 6]);

        // A second frame on index 1 frees the first holder.
        let marking = Marking::Adaptive(vec![Mmco::ShortToLong {
            difference_of_pic_nums_minus1: 0,
            long_term_frame_idx: 1,
        }]);
        let outcome = dpb.mark(picture(4, 4, 7), &marking).expect("fits");
        assert_eq!(outcome.freed, vec![0], "index 1 had the frame in slot 0");

        let marking = Marking::Adaptive(vec![Mmco::UnmarkLong {
            long_term_pic_num: 1,
        }]);
        let outcome = dpb.mark(picture(5, 0, 8), &marking).expect("fits");
        assert_eq!(outcome.freed.len(), 1);
        assert!(
            dpb.references()
                .iter()
                .all(|frame| frame.long_term.is_none())
        );
    }

    #[test]
    fn mmco4_drops_each_index_past_the_new_maximum() {
        let mut dpb = three_shorts();
        let marking = Marking::Adaptive(vec![
            Mmco::ShortToLong {
                difference_of_pic_nums_minus1: 0,
                long_term_frame_idx: 0,
            },
            Mmco::ShortToLong {
                difference_of_pic_nums_minus1: 1,
                long_term_frame_idx: 2,
            },
            Mmco::MaxLongTermIndex {
                max_long_term_frame_idx_plus1: 1,
            },
        ]);
        let outcome = dpb.mark(picture(3, 3, 6), &marking).expect("fits");
        assert_eq!(outcome.freed, vec![1], "index 2 held the frame in slot 1");
        // No long-term index stays allowed, so index 0 leaves too.
        let marking = Marking::Adaptive(vec![Mmco::MaxLongTermIndex {
            max_long_term_frame_idx_plus1: 0,
        }]);
        let outcome = dpb.mark(picture(4, 1, 7), &marking).expect("fits");
        assert_eq!(outcome.freed, vec![2], "index 0 held the frame in slot 2");
    }

    #[test]
    fn mmco5_clears_all_and_rebases_the_order_count() {
        let mut dpb = three_shorts();
        let marking = Marking::Adaptive(vec![Mmco::UnmarkAll]);
        let current = CurrentPicture {
            slot: 3,
            id: 3,
            frame_num: 6,
            poc: [24, 23],
        };
        let outcome = dpb.mark(current, &marking).expect("fits");
        assert_eq!(outcome.freed.len(), 3);
        assert_eq!(outcome.mmco5_poc, Some([1, 0]));
        assert_eq!(dpb.references()[0].frame_num, 0);
        assert_eq!(dpb.references()[0].poc, [1, 0]);
        assert_eq!(dpb.prev_ref_frame_num, 0);
        assert!(!dpb.has_gap(1));
    }

    #[test]
    fn mmco6_makes_the_current_picture_long_term() {
        let mut dpb = three_shorts();
        let marking = Marking::Adaptive(vec![Mmco::CurrentToLong {
            long_term_frame_idx: 2,
        }]);
        dpb.mark(picture(3, 3, 6), &marking).expect("fits");
        assert_eq!(dpb.references()[3].long_term, Some(2));
    }

    /// What a setup slot states, and what marking then holds, for each marking.
    #[test]
    fn the_marked_frame_equals_what_marking_leaves() {
        let current = CurrentPicture {
            slot: 3,
            id: 3,
            frame_num: 6,
            poc: [24, 23],
        };
        let cases = [
            (Marking::SlidingWindow, None, 6, [24, 23]),
            (Marking::Idr { long_term: false }, None, 6, [24, 23]),
            (Marking::Idr { long_term: true }, Some(0), 6, [24, 23]),
            (
                Marking::Adaptive(vec![Mmco::CurrentToLong {
                    long_term_frame_idx: 2,
                }]),
                Some(2),
                6,
                [24, 23],
            ),
            (Marking::Adaptive(vec![Mmco::UnmarkAll]), None, 0, [1, 0]),
            (
                Marking::Adaptive(vec![
                    Mmco::UnmarkAll,
                    Mmco::CurrentToLong {
                        long_term_frame_idx: 1,
                    },
                ]),
                Some(1),
                0,
                [1, 0],
            ),
        ];
        for (marking, long_term, frame_num, poc) in cases {
            let frame = marked_frame(current, &marking);
            assert_eq!(frame.long_term, long_term, "{marking:?}");
            assert_eq!(
                (frame.frame_num, frame.poc),
                (frame_num, poc),
                "{marking:?}"
            );
            let mut dpb = three_shorts();
            dpb.mark(current, &marking).expect("fits");
            assert_eq!(dpb.references().last(), Some(&frame), "{marking:?}");
        }
    }

    #[test]
    fn a_non_reference_picture_stays_short_term_with_its_header_values() {
        let current = picture(1, 1, 5);
        let frame = marked_frame(current, &Marking::NonReference);
        assert_eq!(frame.long_term, None);
        assert_eq!((frame.frame_num, frame.poc), (5, current.poc));
    }

    #[test]
    fn an_absent_target_does_nothing_and_too_many_references_is_an_error() {
        let mut dpb = Dpb::new(3, 2, 16);
        dpb.mark(picture(0, 0, 4), &Marking::SlidingWindow)
            .expect("fits");
        let marking = Marking::Adaptive(vec![Mmco::UnmarkShort {
            difference_of_pic_nums_minus1: 9,
        }]);
        let outcome = dpb.mark(picture(1, 1, 5), &marking).expect("fits");
        assert!(outcome.freed.is_empty());
        let error = dpb
            .mark(picture(2, 2, 6), &marking)
            .expect_err("three references");
        assert!(error.contains("3 references"), "{error}");
    }

    #[test]
    fn gaps_follow_the_previous_reference_frame_number() {
        let mut dpb = Dpb::new(3, 2, 16);
        assert!(!dpb.has_gap(9), "the first picture has no previous number");
        dpb.mark(picture(0, 0, 15), &Marking::SlidingWindow)
            .expect("fits");
        assert!(!dpb.has_gap(15));
        assert!(!dpb.has_gap(0), "15 + 1 wraps to 0");
        assert!(dpb.has_gap(1));
        assert!(dpb.has_gap(14));
    }

    #[test]
    fn a_free_slot_skips_every_reference() {
        let mut dpb = Dpb::new(3, 2, 16);
        assert_eq!(dpb.free_slot(), Some(0));
        dpb.mark(picture(0, 0, 0), &Marking::SlidingWindow)
            .expect("fits");
        assert_eq!(dpb.free_slot(), Some(1));
        assert!(dpb.holds(0));
        dpb.mark(picture(1, 2, 1), &Marking::SlidingWindow)
            .expect("fits");
        assert_eq!(dpb.free_slot(), Some(1));
        assert!(!dpb.holds(7));
    }
}
