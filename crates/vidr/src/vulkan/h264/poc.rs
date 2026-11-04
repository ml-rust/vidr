//! Picture order count for frames, ITU-T H.264 8.2.1. A test states each input.

/// `pic_order_cnt_type` with what its SPS holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum PocConfig {
    /// Type 0, with `MaxPicOrderCntLsb`.
    Type0 { max_lsb: u32 },
    /// Type 1.
    Type1 {
        offset_for_non_ref_pic: i32,
        offset_for_top_to_bottom_field: i32,
        offsets_for_ref_frame: Vec<i32>,
    },
    /// Type 2: output order equals decode order.
    Type2,
}

/// The slice header fields the count reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PocSlice {
    pub(super) idr: bool,
    pub(super) reference: bool,
    pub(super) frame_num: u32,
    pub(super) lsb: u32,
    pub(super) delta_bottom: i32,
    pub(super) delta: [i32; 2],
}

/// The state 8.2.1 carries from one picture to the next.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PocState {
    prev_msb: i64,
    prev_lsb: i64,
    prev_frame_num_offset: i64,
    prev_frame_num: u32,
    /// The picture in decode: its msb and frame number offset.
    current_msb: i64,
    current_frame_num_offset: i64,
}

fn to_i32(value: i64) -> i32 {
    value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

impl PocState {
    /// `[TopFieldOrderCnt, BottomFieldOrderCnt]` of a frame. Call [`Self::finish`] after
    /// its reference marking.
    pub(super) fn compute(
        &mut self,
        config: &PocConfig,
        max_frame_num: u32,
        slice: &PocSlice,
    ) -> [i32; 2] {
        match config {
            PocConfig::Type0 { max_lsb } => self.type0(*max_lsb, slice),
            PocConfig::Type1 {
                offset_for_non_ref_pic,
                offset_for_top_to_bottom_field,
                offsets_for_ref_frame,
            } => {
                let offset = self.frame_num_offset(max_frame_num, slice);
                let mut expected = expected_type1(offset, slice, offsets_for_ref_frame);
                if !slice.reference {
                    expected += i64::from(*offset_for_non_ref_pic);
                }
                let top = expected + i64::from(slice.delta[0]);
                let bottom =
                    top + i64::from(*offset_for_top_to_bottom_field) + i64::from(slice.delta[1]);
                [to_i32(top), to_i32(bottom)]
            }
            PocConfig::Type2 => {
                let offset = self.frame_num_offset(max_frame_num, slice);
                let value = if slice.idr {
                    0
                } else {
                    let doubled = 2 * (offset + i64::from(slice.frame_num));
                    if slice.reference {
                        doubled
                    } else {
                        doubled - 1
                    }
                };
                [to_i32(value), to_i32(value)]
            }
        }
    }

    fn type0(&mut self, max_lsb: u32, slice: &PocSlice) -> [i32; 2] {
        if slice.idr {
            self.prev_msb = 0;
            self.prev_lsb = 0;
        }
        let (lsb, max) = (i64::from(slice.lsb), i64::from(max_lsb));
        let msb = if lsb < self.prev_lsb && self.prev_lsb - lsb >= max / 2 {
            self.prev_msb + max
        } else if lsb > self.prev_lsb && lsb - self.prev_lsb > max / 2 {
            self.prev_msb - max
        } else {
            self.prev_msb
        };
        self.current_msb = msb;
        let top = msb + lsb;
        [to_i32(top), to_i32(top + i64::from(slice.delta_bottom))]
    }

    fn frame_num_offset(&mut self, max_frame_num: u32, slice: &PocSlice) -> i64 {
        let offset = if slice.idr {
            0
        } else if self.prev_frame_num > slice.frame_num {
            self.prev_frame_num_offset + i64::from(max_frame_num)
        } else {
            self.prev_frame_num_offset
        };
        self.current_frame_num_offset = offset;
        offset
    }

    /// Takes the state from the picture just decoded. `mmco5_top` holds its
    /// `TopFieldOrderCnt` after the reset, for a picture with operation 5.
    pub(super) fn finish(&mut self, slice: &PocSlice, mmco5_top: Option<i32>) {
        if let Some(top) = mmco5_top {
            self.prev_msb = 0;
            self.prev_lsb = i64::from(top);
            self.prev_frame_num_offset = 0;
            self.prev_frame_num = 0;
            return;
        }
        if slice.reference {
            self.prev_msb = self.current_msb;
            self.prev_lsb = i64::from(slice.lsb);
        }
        self.prev_frame_num_offset = self.current_frame_num_offset;
        self.prev_frame_num = slice.frame_num;
    }
}

/// `expectedPicOrderCnt` of 8.2.1.2, before the offset for a non-reference picture.
fn expected_type1(frame_num_offset: i64, slice: &PocSlice, offsets: &[i32]) -> i64 {
    let cycle = offsets.len() as i64;
    let mut abs_frame_num = if cycle != 0 {
        frame_num_offset + i64::from(slice.frame_num)
    } else {
        0
    };
    if !slice.reference && abs_frame_num > 0 {
        abs_frame_num -= 1;
    }
    if abs_frame_num == 0 {
        return 0;
    }
    let delta_per_cycle: i64 = offsets.iter().copied().map(i64::from).sum();
    let cycles = (abs_frame_num - 1) / cycle;
    let in_cycle = ((abs_frame_num - 1) % cycle) as usize;
    let partial: i64 = offsets
        .iter()
        .take(in_cycle + 1)
        .copied()
        .map(i64::from)
        .sum();
    cycles * delta_per_cycle + partial
}

#[cfg(test)]
mod tests {
    use super::*;

    const TYPE0: PocConfig = PocConfig::Type0 { max_lsb: 16 };

    fn step0(state: &mut PocState, slice: PocSlice) -> [i32; 2] {
        let poc = state.compute(&TYPE0, 16, &slice);
        state.finish(&slice, None);
        poc
    }

    fn reference(lsb: u32) -> PocSlice {
        PocSlice {
            reference: true,
            lsb,
            ..PocSlice::default()
        }
    }

    #[test]
    fn type0_wraps_the_lsb_in_both_directions() {
        let mut state = PocState::default();
        let idr = PocSlice {
            idr: true,
            ..reference(0)
        };
        assert_eq!(step0(&mut state, idr), [0, 0]);
        assert_eq!(step0(&mut state, reference(6)), [6, 6]);
        // 12 is past 6 by no more than half of 16: no wrap.
        assert_eq!(step0(&mut state, reference(12)), [12, 12]);
        // 2 is below 12 by at least half: msb gains 16.
        assert_eq!(step0(&mut state, reference(2)), [18, 18]);
        // 15 is above 2 by more than half: msb loses 16 for a preceding picture.
        let earlier = PocSlice {
            reference: false,
            lsb: 15,
            ..PocSlice::default()
        };
        assert_eq!(step0(&mut state, earlier), [15, 15]);
        // A non-reference picture leaves the state, so 3 still counts from 2 at msb 16.
        assert_eq!(step0(&mut state, reference(3)), [19, 19]);
    }

    #[test]
    fn type0_adds_the_bottom_delta_and_idr_resets() {
        let mut state = PocState::default();
        // 14 is above the start of 0 by more than half, so msb goes to -16.
        assert_eq!(step0(&mut state, reference(14)), [-2, -2]);
        // 4 is below 14 by at least half, so msb returns to 0.
        let delta = PocSlice {
            delta_bottom: -1,
            ..reference(4)
        };
        assert_eq!(step0(&mut state, delta), [4, 3]);
        let idr = PocSlice {
            idr: true,
            delta_bottom: 1,
            ..reference(0)
        };
        assert_eq!(step0(&mut state, idr), [0, 1]);
    }

    #[test]
    fn type0_after_mmco5_counts_from_the_reset_top() {
        let mut state = PocState::default();
        let slice = reference(10);
        let _ = state.compute(&TYPE0, 16, &slice);
        state.finish(&slice, Some(0));
        assert_eq!(step0(&mut state, reference(4)), [4, 4]);
    }

    fn type1() -> PocConfig {
        PocConfig::Type1 {
            offset_for_non_ref_pic: -2,
            offset_for_top_to_bottom_field: 1,
            offsets_for_ref_frame: vec![2, 4],
        }
    }

    fn step1(state: &mut PocState, slice: PocSlice) -> [i32; 2] {
        let poc = state.compute(&type1(), 16, &slice);
        state.finish(&slice, None);
        poc
    }

    /// With offsets 2 and 4: abs 1 gives 2, abs 2 gives 6, abs 3 gives 8, abs 4 gives 12.
    #[test]
    fn type1_walks_the_cycle_by_hand() {
        let mut state = PocState::default();
        let frame = |frame_num, reference| PocSlice {
            frame_num,
            reference,
            ..PocSlice::default()
        };
        let idr = PocSlice {
            idr: true,
            ..frame(0, true)
        };
        assert_eq!(step1(&mut state, idr), [0, 1]);
        assert_eq!(step1(&mut state, frame(1, true)), [2, 3]);
        assert_eq!(step1(&mut state, frame(2, true)), [6, 7]);
        // A non-reference picture: abs 3 - 1 = 2 gives 6, and the offset gives 4.
        assert_eq!(step1(&mut state, frame(3, false)), [4, 5]);
        assert_eq!(step1(&mut state, frame(3, true)), [8, 9]);
        let delta = PocSlice {
            delta: [1, -1],
            ..frame(4, true)
        };
        assert_eq!(step1(&mut state, delta), [13, 13]);
    }

    #[test]
    fn type2_offsets_at_the_frame_num_wrap() {
        let mut state = PocState::default();
        let config = PocConfig::Type2;
        let mut step = |frame_num, reference| {
            let slice = PocSlice {
                frame_num,
                reference,
                ..PocSlice::default()
            };
            let poc = state.compute(&config, 16, &slice);
            state.finish(&slice, None);
            poc[0]
        };
        assert_eq!(step(14, true), 28);
        assert_eq!(step(15, false), 29);
        assert_eq!(step(15, true), 30);
        // frame_num wraps to 0, so FrameNumOffset gains 16.
        assert_eq!(step(0, true), 32);
        assert_eq!(step(1, false), 33);
    }

    #[test]
    fn type2_resets_at_idr_and_after_mmco5() {
        let mut state = PocState::default();
        let config = PocConfig::Type2;
        let slice = PocSlice {
            frame_num: 9,
            reference: true,
            ..PocSlice::default()
        };
        assert_eq!(state.compute(&config, 16, &slice), [18, 18]);
        state.finish(&slice, Some(0));
        let next = PocSlice {
            frame_num: 1,
            reference: true,
            ..PocSlice::default()
        };
        assert_eq!(state.compute(&config, 16, &next), [2, 2]);
        let idr = PocSlice {
            idr: true,
            reference: true,
            ..PocSlice::default()
        };
        assert_eq!(state.compute(&config, 16, &idr), [0, 0]);
    }
}
