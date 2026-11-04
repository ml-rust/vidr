//! Each picture's type, DPB slots, and counters. The reference structure holds one previous
//! frame: an IDR picture every `gop_length` frames, and P pictures between them. Two DPB slots take
//! turns, so a picture never writes the slot it reads.

/// The DPB slots of a session.
pub(super) const SLOTS: u32 = 2;
/// The active references of a session.
pub(super) const ACTIVE_REFERENCES: u32 = 1;

/// What one picture is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PicturePlan {
    /// An IDR picture, or a key frame for AV1.
    pub(super) idr: bool,
    /// Pictures since the last IDR picture. An IDR picture holds 0.
    pub(super) since_idr: u32,
    /// IDR pictures before this picture.
    pub(super) idr_count: u32,
    /// The slot this picture writes.
    pub(super) setup_slot: u32,
    /// The slot this picture reads. An IDR picture reads none.
    pub(super) reference_slot: Option<u32>,
}

/// The counters of one stream.
#[derive(Clone, Copy, Debug)]
pub(super) struct GopState {
    gop_length: u32,
    next_since_idr: u32,
    idr_count: u32,
    last_slot: Option<u32>,
}

impl GopState {
    pub(super) const fn new(gop_length: u32) -> Self {
        Self {
            gop_length,
            next_since_idr: 0,
            idr_count: 0,
            last_slot: None,
        }
    }

    /// The next picture in decode order, which is also display order.
    pub(super) fn next(&mut self) -> PicturePlan {
        if self.next_since_idr >= self.gop_length {
            self.next_since_idr = 0;
        }
        let since_idr = self.next_since_idr;
        let idr = since_idr == 0;
        let setup_slot = self.last_slot.map_or(0, |slot| (slot + 1) % SLOTS);
        let reference_slot = if idr { None } else { self.last_slot };
        let plan = PicturePlan {
            idr,
            since_idr,
            idr_count: self.idr_count,
            setup_slot,
            reference_slot,
        };
        self.next_since_idr += 1;
        self.idr_count += u32::from(idr);
        self.last_slot = Some(setup_slot);
        plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyframes_land_on_the_gop_length() {
        let mut gop = GopState::new(3);
        let idr: Vec<bool> = (0..7).map(|_| gop.next().idr).collect();
        assert_eq!(idr, [true, false, false, true, false, false, true]);
    }

    #[test]
    fn a_gop_of_one_holds_only_idr_pictures() {
        let mut gop = GopState::new(1);
        for count in 0..4 {
            let plan = gop.next();
            assert!(plan.idr);
            assert_eq!(plan.since_idr, 0);
            assert_eq!(plan.idr_count, count);
            assert_eq!(plan.reference_slot, None);
        }
    }

    #[test]
    fn two_slots_take_turns_and_a_p_picture_reads_the_last_one() {
        let mut gop = GopState::new(4);
        let plans: Vec<PicturePlan> = (0..6).map(|_| gop.next()).collect();
        let slots: Vec<u32> = plans.iter().map(|plan| plan.setup_slot).collect();
        assert_eq!(slots, [0, 1, 0, 1, 0, 1]);
        assert_eq!(plans[0].reference_slot, None);
        assert_eq!(plans[1].reference_slot, Some(0));
        assert_eq!(plans[2].reference_slot, Some(1));
        assert_eq!(plans[4].reference_slot, None, "a new IDR picture");
        assert_eq!(plans[5].reference_slot, Some(0));
        for plan in &plans {
            assert_ne!(Some(plan.setup_slot), plan.reference_slot);
        }
    }

    #[test]
    fn the_counters_restart_at_each_idr_picture() {
        let mut gop = GopState::new(2);
        let plans: Vec<(u32, u32)> = (0..5)
            .map(|_| {
                let plan = gop.next();
                (plan.since_idr, plan.idr_count)
            })
            .collect();
        assert_eq!(plans, [(0, 0), (1, 1), (0, 1), (1, 2), (0, 2)]);
    }
}
