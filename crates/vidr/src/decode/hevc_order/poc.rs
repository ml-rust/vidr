//! `NoRaslOutputFlag` (8.1.3), picture order count (8.3.1), and the NAL unit type
//! classes they read.

/// True for `nal_unit_type` from `BLA_W_LP` to `RSV_IRAP_VCL23`.
pub(crate) const fn is_irap(nal_unit_type: u8) -> bool {
    matches!(nal_unit_type, 16..=23)
}

/// True for `BLA_W_LP`, `BLA_W_RADL`, and `BLA_N_LP`.
pub(crate) const fn is_bla(nal_unit_type: u8) -> bool {
    matches!(nal_unit_type, 16..=18)
}

/// True for `IDR_W_RADL` and `IDR_N_LP`.
pub(crate) const fn is_idr(nal_unit_type: u8) -> bool {
    matches!(nal_unit_type, 19 | 20)
}

/// True for `CRA_NUT`.
pub(crate) const fn is_cra(nal_unit_type: u8) -> bool {
    nal_unit_type == 21
}

/// True for `RASL_N` and `RASL_R`.
const fn is_rasl(nal_unit_type: u8) -> bool {
    matches!(nal_unit_type, 8 | 9)
}

/// True for `RADL_N` and `RADL_R`.
const fn is_radl(nal_unit_type: u8) -> bool {
    matches!(nal_unit_type, 6 | 7)
}

/// True for a sub-layer non-reference picture: the even types below 16.
const fn is_slnr(nal_unit_type: u8) -> bool {
    nal_unit_type < 16 && nal_unit_type & 1 == 0
}

/// What the sequence state decides for one picture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PictureStart {
    /// A RASL picture of an IRAP picture with `NoRaslOutputFlag` 1. It is not
    /// decoded, because its references do not exist (8.3.3.1).
    Skip,
    /// A picture to decode.
    Decode {
        /// `PicOrderCntVal`.
        poc: i32,
        /// True for an IRAP picture with `NoRaslOutputFlag` 1.
        irap_no_rasl: bool,
    },
}

/// The picture facts [`SequenceState::begin_picture`] reads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PictureFacts {
    /// `nal_unit_type`.
    pub(crate) nal_unit_type: u8,
    /// `TemporalId`.
    pub(crate) temporal_id: u8,
    /// `slice_pic_order_cnt_lsb`.
    pub(crate) poc_lsb: u16,
    /// `MaxPicOrderCntLsb`.
    pub(crate) max_poc_lsb: u32,
}

/// The state that carries between pictures of one bitstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SequenceState {
    /// True before the first picture, and after an end of sequence NAL unit.
    starts_sequence: bool,
    /// `NoRaslOutputFlag` of the latest IRAP picture.
    irap_no_rasl: bool,
    /// `slice_pic_order_cnt_lsb` of `prevTid0Pic`.
    prev_lsb: i32,
    /// `PicOrderCntMsb` of `prevTid0Pic`.
    prev_msb: i32,
}

impl SequenceState {
    /// The state at the start of a bitstream.
    pub(crate) const fn new() -> Self {
        Self {
            starts_sequence: true,
            irap_no_rasl: true,
            prev_lsb: 0,
            prev_msb: 0,
        }
    }

    /// Records an end of sequence NAL unit: the next picture restarts POC.
    pub(crate) fn end_of_sequence(&mut self) {
        self.starts_sequence = true;
    }

    /// Runs 8.1.3 and 8.3.1 for the first slice segment of a picture.
    pub(crate) fn begin_picture(&mut self, facts: PictureFacts) -> PictureStart {
        let nal_type = facts.nal_unit_type;
        let irap = is_irap(nal_type);
        let mut irap_no_rasl = false;
        if irap {
            irap_no_rasl = is_idr(nal_type) || is_bla(nal_type) || self.starts_sequence;
            self.irap_no_rasl = irap_no_rasl;
        }
        self.starts_sequence = false;
        if is_rasl(nal_type) && self.irap_no_rasl {
            return PictureStart::Skip;
        }
        let lsb = i32::from(facts.poc_lsb);
        let max_lsb = facts.max_poc_lsb as i32;
        let msb = if irap_no_rasl {
            0
        } else if lsb < self.prev_lsb && self.prev_lsb - lsb >= max_lsb / 2 {
            // (8-1).
            self.prev_msb.wrapping_add(max_lsb)
        } else if lsb > self.prev_lsb && lsb - self.prev_lsb > max_lsb / 2 {
            self.prev_msb.wrapping_sub(max_lsb)
        } else {
            self.prev_msb
        };
        if facts.temporal_id == 0 && !is_rasl(nal_type) && !is_radl(nal_type) && !is_slnr(nal_type)
        {
            self.prev_lsb = lsb;
            self.prev_msb = msb;
        }
        PictureStart::Decode {
            // (8-2).
            poc: msb.wrapping_add(lsb),
            irap_no_rasl,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRAIL_N: u8 = 0;
    const TRAIL_R: u8 = 1;
    const RADL_R: u8 = 7;
    const RASL_N: u8 = 8;
    const RASL_R: u8 = 9;
    const BLA_W_RADL: u8 = 17;
    const IDR_W_RADL: u8 = 19;
    const CRA: u8 = 21;

    fn facts(nal_unit_type: u8, temporal_id: u8, poc_lsb: u16) -> PictureFacts {
        PictureFacts {
            nal_unit_type,
            temporal_id,
            poc_lsb,
            max_poc_lsb: 16,
        }
    }

    fn poc_of(start: PictureStart) -> i32 {
        match start {
            PictureStart::Decode { poc, .. } => poc,
            PictureStart::Skip => panic!("the picture is skipped"),
        }
    }

    #[test]
    fn an_idr_picture_restarts_at_zero() {
        let mut state = SequenceState::new();
        assert_eq!(
            state.begin_picture(facts(IDR_W_RADL, 0, 0)),
            PictureStart::Decode {
                poc: 0,
                irap_no_rasl: true
            }
        );
    }

    /// (8-1): MaxPicOrderCntLsb 16. From lsb 14, an lsb of 2 wraps up, because
    /// 14 - 2 >= 8. From msb 16 lsb 2, an lsb of 14 wraps down, because 14 - 2 > 8.
    #[test]
    fn the_msb_wraps_both_ways() {
        let mut state = SequenceState::new();
        state.begin_picture(facts(IDR_W_RADL, 0, 0));
        assert_eq!(poc_of(state.begin_picture(facts(TRAIL_R, 0, 7))), 7);
        assert_eq!(poc_of(state.begin_picture(facts(TRAIL_R, 0, 14))), 14);
        assert_eq!(poc_of(state.begin_picture(facts(TRAIL_R, 0, 2))), 18);
        assert_eq!(poc_of(state.begin_picture(facts(TRAIL_R, 0, 14))), 14);
    }

    /// A distance of exactly half the range: lsb 8 from lsb 0 stays, because the
    /// upward test is strict. Lsb 0 from lsb 8 wraps, because the downward test is not.
    #[test]
    fn half_the_range_follows_the_two_comparisons() {
        let mut state = SequenceState::new();
        state.begin_picture(facts(IDR_W_RADL, 0, 0));
        assert_eq!(poc_of(state.begin_picture(facts(TRAIL_R, 0, 8))), 8);
        assert_eq!(poc_of(state.begin_picture(facts(TRAIL_R, 0, 0))), 16);
    }

    /// prevTid0Pic skips a higher temporal layer, RASL, RADL, and SLNR pictures.
    #[test]
    fn only_tid0_reference_pictures_set_the_previous_values() {
        let mut state = SequenceState::new();
        state.begin_picture(facts(IDR_W_RADL, 0, 0));
        // None of these moves the base, so lsb 6 keeps msb 0 from lsb 0.
        assert_eq!(poc_of(state.begin_picture(facts(TRAIL_R, 1, 12))), -4);
        assert_eq!(poc_of(state.begin_picture(facts(TRAIL_N, 0, 12))), -4);
        assert_eq!(poc_of(state.begin_picture(facts(RADL_R, 0, 12))), -4);
        assert_eq!(poc_of(state.begin_picture(facts(TRAIL_R, 0, 6))), 6);
    }

    /// A first CRA picture has NoRaslOutputFlag 1, so its RASL pictures are skipped
    /// and leave POC alone. A later CRA picture has 0, and its RASL pictures decode.
    #[test]
    fn rasl_pictures_follow_their_irap_picture() {
        let mut state = SequenceState::new();
        assert_eq!(
            state.begin_picture(facts(CRA, 0, 4)),
            PictureStart::Decode {
                poc: 4,
                irap_no_rasl: true
            }
        );
        assert_eq!(state.begin_picture(facts(RASL_R, 0, 2)), PictureStart::Skip);
        assert_eq!(state.begin_picture(facts(RASL_N, 0, 3)), PictureStart::Skip);
        assert_eq!(poc_of(state.begin_picture(facts(TRAIL_R, 0, 5))), 5);
        assert_eq!(
            state.begin_picture(facts(CRA, 0, 9)),
            PictureStart::Decode {
                poc: 9,
                irap_no_rasl: false
            }
        );
        assert_eq!(poc_of(state.begin_picture(facts(RASL_N, 0, 8))), 8);
    }

    /// A CRA picture after an end of sequence NAL unit starts over, and a BLA picture
    /// always does.
    #[test]
    fn an_end_of_sequence_restarts_a_cra_picture() {
        let mut state = SequenceState::new();
        state.begin_picture(facts(IDR_W_RADL, 0, 0));
        state.begin_picture(facts(TRAIL_R, 0, 15));
        state.end_of_sequence();
        assert_eq!(
            state.begin_picture(facts(CRA, 0, 3)),
            PictureStart::Decode {
                poc: 3,
                irap_no_rasl: true
            }
        );
        assert_eq!(
            state.begin_picture(facts(BLA_W_RADL, 0, 1)),
            PictureStart::Decode {
                poc: 1,
                irap_no_rasl: true
            }
        );
        assert_eq!(state.begin_picture(facts(RASL_N, 0, 0)), PictureStart::Skip);
    }
}
