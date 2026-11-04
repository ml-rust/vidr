//! The reference frame pool of VP9 (8.10): eight names, each naming one DPB slot. A slot goes
//! free once no name holds it.

use crate::bitstream::{NUM_REF_FRAMES, UncompressedHeader};
use crate::decode::ColorFacts;
use crate::vulkan::gap::MAX_VP9_REFERENCES as REFS_PER_FRAME;

/// One slot per name, and one for the frame in decode.
pub(super) const SLOTS: u32 = NUM_REF_FRAMES as u32 + 1;

/// What one reference name holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct NamedFrame {
    pub(super) slot: u32,
    pub(super) coded: (u32, u32),
    pub(super) color: ColorFacts,
}

/// The eight reference names.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct RefPool {
    names: [Option<NamedFrame>; NUM_REF_FRAMES],
}

/// The slots one frame reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FrameRefs {
    /// `referenceNameSlotIndices`: -1 where an intra frame names nothing.
    pub(super) name_slots: [i32; REFS_PER_FRAME],
    /// Each distinct slot, once, with its coded size. `count` entries hold values.
    pub(super) distinct: [(u32, (u32, u32)); REFS_PER_FRAME],
    pub(super) count: usize,
}

/// A reference index of a frame with an empty name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct EmptyName {
    pub(super) name: u8,
}

impl RefPool {
    /// What `name` holds.
    pub(super) fn get(&self, name: u8) -> Option<NamedFrame> {
        self.names.get(usize::from(name)).copied().flatten()
    }

    /// The lowest slot no name holds. [`SLOTS`] leaves one free at all times.
    pub(super) fn free_slot(&self) -> Option<u32> {
        (0..SLOTS).find(|&slot| !self.holds(slot))
    }

    fn holds(&self, slot: u32) -> bool {
        self.names.iter().flatten().any(|named| named.slot == slot)
    }

    /// The slots `header` reads. An intra frame reads none.
    pub(super) fn frame_refs(&self, header: &UncompressedHeader) -> Result<FrameRefs, EmptyName> {
        let mut refs = FrameRefs {
            name_slots: [-1; REFS_PER_FRAME],
            distinct: [(0, (0, 0)); REFS_PER_FRAME],
            count: 0,
        };
        if header.frame_is_intra {
            return Ok(refs);
        }
        for (target, &name) in refs.name_slots.iter_mut().zip(&header.ref_frame_idx) {
            let named = self.get(name).ok_or(EmptyName { name })?;
            *target = named.slot as i32;
            let known = refs.distinct[..refs.count]
                .iter()
                .any(|(slot, _)| *slot == named.slot);
            if !known {
                refs.distinct[refs.count] = (named.slot, named.coded);
                refs.count += 1;
            }
        }
        Ok(refs)
    }

    /// Stores `frame` under each name `refresh_frame_flags` sets, and returns each slot that
    /// goes free. A frame no name takes frees nothing here: its decode already emptied it.
    pub(super) fn refresh(&mut self, refresh_frame_flags: u8, frame: NamedFrame) -> Vec<u32> {
        let mut replaced = Vec::new();
        for (index, name) in self.names.iter_mut().enumerate() {
            if (refresh_frame_flags >> index) & 1 == 1 {
                if let Some(old) = *name
                    && old.slot != frame.slot
                {
                    replaced.push(old.slot);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::vp9_frame::VP9_UNSTATED;

    fn frame(slot: u32) -> NamedFrame {
        NamedFrame {
            slot,
            coded: (64 + slot, 48),
            color: VP9_UNSTATED,
        }
    }

    fn inter_header(ref_frame_idx: [u8; 3]) -> UncompressedHeader {
        UncompressedHeader {
            ref_frame_idx,
            ..UncompressedHeader::default()
        }
    }

    #[test]
    fn a_key_frame_fills_every_name_with_one_slot() {
        let mut pool = RefPool::default();
        assert!(pool.refresh(0xFF, frame(0)).is_empty());
        for name in 0..8 {
            assert_eq!(pool.get(name), Some(frame(0)));
        }
        assert_eq!(pool.free_slot(), Some(1));
    }

    #[test]
    fn a_slot_goes_free_when_its_last_name_leaves() {
        let mut pool = RefPool::default();
        pool.refresh(0b0000_0011, frame(0));
        assert!(
            pool.refresh(0b0000_0001, frame(1)).is_empty(),
            "name 1 holds slot 0"
        );
        assert_eq!(pool.refresh(0b0000_0010, frame(2)), [0]);
        assert_eq!(pool.free_slot(), Some(0));
    }

    #[test]
    fn every_distinct_name_still_leaves_a_slot_free() {
        let mut pool = RefPool::default();
        for name in 0..8u32 {
            pool.refresh(1 << name, frame(name));
        }
        assert_eq!(pool.free_slot(), Some(8));
    }

    /// `referenceNameSlotIndices` keeps three entries, and `pReferenceSlots` takes a slot once.
    #[test]
    fn frame_refs_keep_each_name_and_bind_each_slot_once() {
        let mut pool = RefPool::default();
        pool.refresh(0b0000_0101, frame(3));
        pool.refresh(0b0000_0010, frame(5));
        let refs = pool.frame_refs(&inter_header([0, 1, 2])).expect("refs");
        assert_eq!(refs.name_slots, [3, 5, 3]);
        assert_eq!(refs.count, 2);
        assert_eq!(refs.distinct[0], (3, (67, 48)));
        assert_eq!(refs.distinct[1], (5, (69, 48)));
    }

    #[test]
    fn an_intra_frame_names_no_slot() {
        let pool = RefPool::default();
        let header = UncompressedHeader {
            frame_is_intra: true,
            ..UncompressedHeader::default()
        };
        let refs = pool.frame_refs(&header).expect("refs");
        assert_eq!(refs.name_slots, [-1, -1, -1]);
        assert_eq!(refs.count, 0);
    }

    #[test]
    fn an_empty_name_is_named() {
        let pool = RefPool::default();
        assert_eq!(
            pool.frame_refs(&inter_header([4, 0, 0])),
            Err(EmptyName { name: 4 })
        );
    }
}
