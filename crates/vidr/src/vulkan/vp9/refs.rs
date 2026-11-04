//! What a reference name of VP9 (8.10) holds, on the shared pool.

use crate::bitstream::{NUM_REF_FRAMES, UncompressedHeader};
use crate::decode::ColorFacts;
use crate::vulkan::gap::MAX_VP9_REFERENCES as REFS_PER_FRAME;
pub(super) use crate::vulkan::ref_pool::SLOTS;
use crate::vulkan::ref_pool::{self, EmptyName, NAMES, PoolFrame};

const _: () = assert!(NAMES == NUM_REF_FRAMES);

/// What one reference name holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct NamedFrame {
    pub(super) slot: u32,
    pub(super) coded: (u32, u32),
    pub(super) color: ColorFacts,
}

impl PoolFrame for NamedFrame {
    fn slot(&self) -> u32 {
        self.slot
    }
}

/// The eight reference names.
pub(super) type RefPool = ref_pool::RefPool<NamedFrame>;

/// The slots one frame reads.
pub(super) type FrameRefs = ref_pool::FrameRefs<NamedFrame, REFS_PER_FRAME>;

impl RefPool {
    /// The slots `header` reads. An intra frame reads none.
    pub(super) fn frame_refs(&self, header: &UncompressedHeader) -> Result<FrameRefs, EmptyName> {
        self.read_refs(&header.ref_frame_idx, header.frame_is_intra)
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
        let held = refs
            .distinct
            .map(|named| named.map(|named| (named.slot, named.coded)));
        assert_eq!(held[0], Some((3, (67, 48))));
        assert_eq!(held[1], Some((5, (69, 48))));
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
