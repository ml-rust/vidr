//! The reference frame pool of VP9 (8.10) and AV1 (7.20): eight names, each naming one DPB slot.
//! A slot goes free once no name holds it. Each codec states what a name holds.

/// `NUM_REF_FRAMES` of VP9 and of AV1.
pub(super) const NAMES: usize = 8;

/// One slot per name, and one for the frame in decode.
pub(super) const SLOTS: u32 = NAMES as u32 + 1;

/// What one reference name holds.
pub(super) trait PoolFrame: Copy {
    /// The DPB slot of the frame.
    fn slot(&self) -> u32;
}

/// The eight reference names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RefPool<F> {
    names: [Option<F>; NAMES],
}

impl<F: Copy> Default for RefPool<F> {
    fn default() -> Self {
        Self {
            names: [None; NAMES],
        }
    }
}

/// The slots one frame reads, for `N` references per frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FrameRefs<F, const N: usize> {
    /// `referenceNameSlotIndices`: -1 where an intra frame names nothing.
    pub(super) name_slots: [i32; N],
    /// Each distinct slot, once, with its frame. `count` entries hold values.
    pub(super) distinct: [Option<F>; N],
    pub(super) count: usize,
}

/// A reference index of a frame with an empty name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct EmptyName {
    pub(super) name: u8,
}

impl<F: PoolFrame> RefPool<F> {
    /// What `name` holds.
    pub(super) fn get(&self, name: u8) -> Option<F> {
        self.names.get(usize::from(name)).copied().flatten()
    }

    /// The lowest slot no name holds. [`SLOTS`] leaves one free at all times.
    pub(super) fn free_slot(&self) -> Option<u32> {
        (0..SLOTS).find(|&slot| !self.holds(slot))
    }

    fn holds(&self, slot: u32) -> bool {
        self.names
            .iter()
            .flatten()
            .any(|named| named.slot() == slot)
    }

    /// The slots a frame with `ref_frame_idx` reads. An intra frame reads none.
    pub(super) fn read_refs<const N: usize>(
        &self,
        ref_frame_idx: &[u8; N],
        intra: bool,
    ) -> Result<FrameRefs<F, N>, EmptyName> {
        let mut refs = FrameRefs {
            name_slots: [-1; N],
            distinct: [None; N],
            count: 0,
        };
        if intra {
            return Ok(refs);
        }
        for (target, &name) in refs.name_slots.iter_mut().zip(ref_frame_idx) {
            let named = self.get(name).ok_or(EmptyName { name })?;
            *target = named.slot() as i32;
            let known = refs.distinct[..refs.count]
                .iter()
                .flatten()
                .any(|entry| entry.slot() == named.slot());
            if !known {
                refs.distinct[refs.count] = Some(named);
                refs.count += 1;
            }
        }
        Ok(refs)
    }

    /// Empties `name`, so no later frame reads its stale slot. The result is the slot when it goes
    /// free.
    pub(super) fn clear(&mut self, name: u8) -> Option<u32> {
        let old = self.names.get_mut(usize::from(name))?.take()?;
        (!self.holds(old.slot())).then(|| old.slot())
    }

    /// Stores `frame` under each name `refresh_frame_flags` sets, and returns each slot that
    /// goes free. A frame no name takes frees nothing here: its decode already emptied it.
    pub(super) fn refresh(&mut self, refresh_frame_flags: u8, frame: F) -> Vec<u32> {
        let mut replaced = Vec::new();
        for (index, name) in self.names.iter_mut().enumerate() {
            if (refresh_frame_flags >> index) & 1 == 1 {
                if let Some(old) = *name
                    && old.slot() != frame.slot()
                {
                    replaced.push(old.slot());
                }
                *name = Some(frame);
            }
        }
        replaced.sort_unstable();
        replaced.dedup();
        replaced.retain(|&slot| !self.holds(slot));
        replaced
    }
}
