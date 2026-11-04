//! What a reference name of AV1 (7.20) holds, on the shared pool, and the scale limits of section
//! 6.8.6.

use crate::bitstream::av1::frame_header::{FrameHeader, NUM_REF_FRAMES, REFS_PER_FRAME};
use crate::decode::ColorFacts;
pub(super) use crate::vulkan::ref_pool::SLOTS;
use crate::vulkan::ref_pool::{self, EmptyName, NAMES, PoolFrame};

const _: () = assert!(NAMES == NUM_REF_FRAMES);

/// What `StdVideoDecodeAV1ReferenceInfo` restates of a slot's frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct SlotFacts {
    pub(super) disable_frame_end_update_cdf: bool,
    pub(super) segmentation_enabled: bool,
    pub(super) frame_type: u8,
    pub(super) sign_bias: u8,
    pub(super) order_hint: u8,
    pub(super) saved_order_hints: [u8; NUM_REF_FRAMES],
}

/// What one reference name holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct NamedFrame {
    pub(super) slot: u32,
    /// `UpscaledWidth` and `FrameHeight` of the frame.
    pub(super) coded: (u32, u32),
    pub(super) color: ColorFacts,
    pub(super) facts: SlotFacts,
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

/// A reference whose size the frame cannot scale from: AV1 specification 6.8.6.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ScaleRefused {
    pub(super) name: u8,
    pub(super) reference: (u32, u32),
    pub(super) frame: (u32, u32),
}

impl RefPool {
    /// The slots `header` reads. An intra frame reads none.
    pub(super) fn frame_refs(&self, header: &FrameHeader) -> Result<FrameRefs, EmptyName> {
        self.read_refs(&header.ref_frame_idx, header.frame_is_intra)
    }

    /// Refuses what section 6.8.6 forbids for every reference of an inter frame, whether a block
    /// reads it or not: a reference over twice the frame's size, or under a sixteenth of it.
    pub(super) fn check_scaling(&self, header: &FrameHeader) -> Result<(), ScaleRefused> {
        if header.frame_is_intra {
            return Ok(());
        }
        let frame = (header.frame_width, header.frame_height);
        for &name in &header.ref_frame_idx {
            let Some(named) = self.get(name) else {
                continue;
            };
            let (width, height) = named.coded;
            let fits = 2 * frame.0 >= width
                && 2 * frame.1 >= height
                && frame.0 <= 16 * width
                && frame.1 <= 16 * height;
            if !fits {
                return Err(ScaleRefused {
                    name,
                    reference: named.coded,
                    frame,
                });
            }
        }
        Ok(())
    }
}

/// The first name `header` reads whose frame `lost` holds as lost. A slot cannot stand in for it:
/// a decode alone activates a slot, and libaom's grey picture has no decode behind it.
pub(super) fn lost_reference(header: &FrameHeader, lost: u8) -> Option<u8> {
    if header.frame_is_intra {
        return None;
    }
    header
        .ref_frame_idx
        .iter()
        .copied()
        .find(|name| (lost >> (name & 7)) & 1 == 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{MATRIX_UNSPECIFIED, PRIMARIES_UNSPECIFIED, TRANSFER_UNSPECIFIED};

    fn frame(slot: u32) -> NamedFrame {
        NamedFrame {
            slot,
            coded: (64 + slot, 48),
            color: ColorFacts {
                matrix_coefficients: MATRIX_UNSPECIFIED,
                transfer_characteristics: TRANSFER_UNSPECIFIED,
                colour_primaries: PRIMARIES_UNSPECIFIED,
                full_range: false,
                chroma_location: None,
            },
            facts: SlotFacts {
                order_hint: slot as u8,
                ..SlotFacts::default()
            },
        }
    }

    fn inter_header(ref_frame_idx: [u8; 7]) -> FrameHeader {
        FrameHeader {
            ref_frame_idx,
            frame_width: 64,
            frame_height: 48,
            ..FrameHeader::default()
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

    /// A key frame shown again refreshes every name, so each other slot goes free.
    #[test]
    fn a_refresh_of_every_name_frees_every_other_slot() {
        let mut pool = RefPool::default();
        pool.refresh(0b0000_1111, frame(0));
        pool.refresh(0b1111_0000, frame(1));
        assert_eq!(pool.refresh(0xFF, frame(1)), [0]);
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

    /// `referenceNameSlotIndices` keeps seven entries, and `pReferenceSlots` takes a slot once, as
    /// `VUID-vkCmdDecodeVideoKHR-slotIndex-09263` and `-referenceNameSlotIndices-09262` state.
    #[test]
    fn frame_refs_keep_each_name_and_bind_each_slot_once() {
        let mut pool = RefPool::default();
        pool.refresh(0b0000_0101, frame(3));
        pool.refresh(0b0000_0010, frame(5));
        pool.refresh(0b1111_1000, frame(7));
        let refs = pool
            .frame_refs(&inter_header([0, 1, 2, 3, 4, 5, 6]))
            .expect("refs");
        assert_eq!(refs.name_slots, [3, 5, 3, 7, 7, 7, 7]);
        assert_eq!(refs.count, 3);
        let slots: Vec<u32> = refs.distinct[..refs.count]
            .iter()
            .flatten()
            .map(|named| named.slot)
            .collect();
        assert_eq!(slots, [3, 5, 7]);
        for slot in refs.name_slots {
            assert!(slots.contains(&(slot as u32)), "slot {slot}");
        }
    }

    #[test]
    fn an_intra_frame_names_no_slot() {
        let pool = RefPool::default();
        let header = FrameHeader {
            frame_is_intra: true,
            ..FrameHeader::default()
        };
        let refs = pool.frame_refs(&header).expect("refs");
        assert_eq!(refs.name_slots, [-1; 7]);
        assert_eq!(refs.count, 0);
    }

    #[test]
    fn an_empty_name_is_named() {
        let pool = RefPool::default();
        assert_eq!(
            pool.frame_refs(&inter_header([4, 0, 0, 0, 0, 0, 0])),
            Err(EmptyName { name: 4 })
        );
    }

    /// A cleared name reads as empty, so a frame never decodes against its old slot.
    #[test]
    fn a_cleared_name_gives_no_stale_slot() {
        let mut pool = RefPool::default();
        pool.refresh(0b0000_0011, frame(4));
        pool.refresh(0b1111_1100, frame(5));
        assert_eq!(pool.clear(0), None, "name 1 still holds slot 4");
        assert_eq!(pool.get(0), None);
        assert_eq!(pool.clear(1), Some(4));
        assert_eq!(pool.clear(1), None);
        assert_eq!(
            pool.frame_refs(&inter_header([2, 2, 1, 2, 2, 2, 2])),
            Err(EmptyName { name: 1 })
        );
        assert_eq!(pool.free_slot(), Some(0));
    }

    #[test]
    fn a_frame_that_reads_a_lost_name_is_named() {
        let header = inter_header([2, 1, 3, 3, 3, 3, 3]);
        assert_eq!(lost_reference(&header, 0), None);
        assert_eq!(lost_reference(&header, 0b0001), None);
        assert_eq!(lost_reference(&header, 0b0010), Some(1));
        assert_eq!(lost_reference(&header, 0b1100), Some(2));
        let intra = FrameHeader {
            frame_is_intra: true,
            ..header
        };
        assert_eq!(lost_reference(&intra, 0xff), None);
    }

    #[test]
    fn a_reference_within_the_scale_limits_passes() {
        let mut pool = RefPool::default();
        // Twice the frame's height, the largest allowed.
        pool.refresh(
            0xFF,
            NamedFrame {
                coded: (64, 96),
                ..frame(0)
            },
        );
        assert_eq!(pool.check_scaling(&inter_header([0; 7])), Ok(()));
        // A sixteenth of the frame's width, the smallest allowed.
        pool.refresh(
            0xFF,
            NamedFrame {
                coded: (4, 48),
                ..frame(0)
            },
        );
        assert_eq!(pool.check_scaling(&inter_header([0; 7])), Ok(()));
    }

    #[test]
    fn a_reference_past_the_scale_limits_is_named() {
        for coded in [(129, 48), (3, 48), (64, 97), (64, 2)] {
            let mut pool = RefPool::default();
            pool.refresh(0b10, NamedFrame { coded, ..frame(0) });
            pool.refresh(0b01, frame(1));
            assert_eq!(
                pool.check_scaling(&inter_header([0, 1, 0, 0, 0, 0, 0])),
                Err(ScaleRefused {
                    name: 1,
                    reference: coded,
                    frame: (64, 48),
                })
            );
        }
    }
}
