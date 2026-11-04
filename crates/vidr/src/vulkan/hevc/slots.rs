//! DPB slots of stored pictures, and the current picture's slot lists. It reads marking from
//! the shared [`RefPicSets`], so tests run with no GPU.
//!
//! A picture 8.3.3 generates sits only in a "Foll" list, so no decode binds it, and it takes
//! no slot. Vulkan Video activates a slot only through a decode
//! (`VUID-vkCmdBeginVideoCodingKHR-slotIndex-07239`). So a clear or a copy cannot give a slot
//! to a reference that a picture binds. A missing entry of a "Curr" list is refused by name.

use super::sequence::MAX_CURRENT_REFERENCES;
use crate::decode::hevc_order::buffer::{BufferedPicture, RefEntry, RefPicSets};

/// `STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE`.
const LIST_SIZE: usize = MAX_CURRENT_REFERENCES as usize;
/// "No reference picture" in a `RefPicSet*` list.
const NO_SLOT: u8 = 0xFF;

/// One picture in the buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SlotPicture {
    pub(super) poc: i32,
    /// `None` for a picture of 8.3.3.
    pub(super) slot: Option<u32>,
    /// Keys the frame that waits for output.
    pub(super) id: u64,
}

impl BufferedPicture for SlotPicture {
    fn poc(&self) -> i32 {
        self.poc
    }
}

/// Which slots hold a picture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SlotPool {
    used: Vec<bool>,
}

impl SlotPool {
    pub(super) fn new(slots: u32) -> Self {
        Self {
            used: vec![false; slots as usize],
        }
    }

    /// Takes the lowest free slot.
    pub(super) fn take(&mut self) -> Option<u32> {
        let index = self.used.iter().position(|used| !used)?;
        self.used[index] = true;
        Some(index as u32)
    }

    pub(super) fn release(&mut self, slot: u32) {
        if let Some(used) = self.used.get_mut(slot as usize) {
            *used = false;
        }
    }
}

/// One active reference of a decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ActiveReference {
    pub(super) slot: u32,
    pub(super) poc: i32,
    pub(super) long_term: bool,
}

/// The current picture's `RefPicSet*` lists as slot indices, and what the decode binds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SlotLists {
    pub(super) before: [u8; LIST_SIZE],
    pub(super) after: [u8; LIST_SIZE],
    pub(super) long_term: [u8; LIST_SIZE],
    pub(super) references: Vec<ActiveReference>,
}

/// Why a picture's sets have no slot lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ListError {
    /// A "Curr" entry names a picture with no slot.
    Missing { poc: i32, long_term: bool },
    /// A list holds more entries than `RefPicSet*` does.
    TooLong { entries: usize },
}

fn fill(
    entries: &[RefEntry<SlotPicture>],
    list: &mut [u8; LIST_SIZE],
    references: &mut Vec<ActiveReference>,
) -> Result<(), ListError> {
    if entries.len() > LIST_SIZE {
        return Err(ListError::TooLong {
            entries: entries.len(),
        });
    }
    for (entry, target) in entries.iter().zip(list.iter_mut()) {
        let slot = entry
            .frame
            .and_then(|picture| picture.slot)
            .ok_or(ListError::Missing {
                poc: entry.poc,
                long_term: entry.long_term,
            })?;
        *target = slot as u8;
        references.push(ActiveReference {
            slot,
            poc: entry.frame.map_or(entry.poc, |picture| picture.poc),
            long_term: entry.long_term,
        });
    }
    Ok(())
}

/// The slot lists of `sets`. Two entries can resolve to one picture on a damaged stream, so
/// `references` can name a slot twice. The engine refuses that request by name.
pub(super) fn slot_lists(sets: &RefPicSets<SlotPicture>) -> Result<SlotLists, ListError> {
    let mut lists = SlotLists {
        before: [NO_SLOT; LIST_SIZE],
        after: [NO_SLOT; LIST_SIZE],
        long_term: [NO_SLOT; LIST_SIZE],
        references: Vec::with_capacity(LIST_SIZE),
    };
    fill(&sets.before, &mut lists.before, &mut lists.references)?;
    fill(&sets.after, &mut lists.after, &mut lists.references)?;
    fill(&sets.long_term, &mut lists.long_term, &mut lists.references)?;
    Ok(lists)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::hevc_rps::ShortTermRefPicSet;
    use crate::bitstream::hevc_slice::SliceSegmentHeader;
    use crate::bitstream::hevc_sps_full::tests as sps_tests;
    use crate::decode::hevc_order::buffer::{DpbLimits, RefBuffer};
    use crate::decode::hevc_order::rps::RpsPocs;

    const LIMITS: DpbLimits = DpbLimits {
        max_num_reorder: 2,
        max_latency: None,
        max_dec_buffering: 5,
    };

    fn picture(poc: i32, slot: u32) -> SlotPicture {
        SlotPicture {
            poc,
            slot: Some(slot),
            id: u64::from(slot),
        }
    }

    /// The POC lists of a picture at `poc` whose set names `before` and `after` as used.
    fn rps(poc: i32, before: &[i32], after: &[i32], kept: &[i32]) -> RpsPocs {
        let mut set = ShortTermRefPicSet {
            num_negative_pics: (before.len() + kept.len()) as u8,
            num_positive_pics: after.len() as u8,
            ..ShortTermRefPicSet::default()
        };
        for (index, &delta) in before.iter().enumerate() {
            set.delta_poc_s0[index] = delta;
            set.used_by_curr_pic_s0[index] = true;
        }
        for (index, &delta) in kept.iter().enumerate() {
            set.delta_poc_s0[before.len() + index] = delta;
        }
        for (index, &delta) in after.iter().enumerate() {
            set.delta_poc_s1[index] = delta;
            set.used_by_curr_pic_s1[index] = true;
        }
        let header = SliceSegmentHeader {
            nal_unit_type: 1,
            short_term_ref_pic_set: Some(set),
            ..SliceSegmentHeader::default()
        };
        RpsPocs::derive(&header, &sps_tests::valid(&sps_tests::Spec::default()), poc)
    }

    fn buffer(pocs: &[i32]) -> RefBuffer<SlotPicture> {
        let mut buffer = RefBuffer::new();
        for (slot, &poc) in pocs.iter().enumerate() {
            buffer.insert_picture(picture(poc, slot as u32), false, &LIMITS);
        }
        buffer
    }

    #[test]
    fn each_list_holds_the_slot_of_its_poc_and_ends_in_no_slot() {
        let mut buffer = buffer(&[0, 8, 4]);
        let sets = buffer.apply_rps(&rps(2, &[-2], &[2, 6], &[]), 256, false);
        let lists = slot_lists(&sets).expect("lists");
        assert_eq!(lists.before[..2], [0, NO_SLOT]);
        assert_eq!(lists.after[..3], [2, 1, NO_SLOT]);
        assert_eq!(lists.long_term, [NO_SLOT; LIST_SIZE]);
        assert_eq!(lists.references.len(), 3);
        assert_eq!(
            lists.references[1],
            ActiveReference {
                slot: 2,
                poc: 4,
                long_term: false,
            }
        );
    }

    #[test]
    fn a_picture_out_of_the_set_leaves_when_no_output_waits() {
        let mut buffer = buffer(&[0, 4, 8]);
        // POC 4 falls out of the set, so the bump before the next picture removes it.
        buffer.apply_rps(&rps(9, &[-1, -9], &[], &[]), 256, false);
        buffer.before_picture(false, false, &LIMITS);
        let removed = buffer.take_removed();
        assert_eq!(removed, [picture(4, 1)]);
    }

    #[test]
    fn an_entry_kept_by_foll_stays_with_its_slot_and_is_not_bound() {
        let mut buffer = buffer(&[0, 4]);
        let sets = buffer.apply_rps(&rps(8, &[-4], &[], &[-8]), 256, false);
        buffer.before_picture(false, false, &LIMITS);
        assert!(buffer.take_removed().is_empty());
        let lists = slot_lists(&sets).expect("lists");
        assert_eq!(lists.references.len(), 1);
        assert_eq!(lists.references[0].poc, 4);
    }

    /// A CRA picture with `NoRaslOutputFlag` 1 names a missing picture in a "Foll" list: it
    /// takes a place in the buffer with no slot, and no list binds it.
    #[test]
    fn a_generated_picture_takes_no_slot_and_a_curr_miss_is_named() {
        let mut buffer = RefBuffer::new();
        let sets = buffer.apply_rps(&rps(8, &[], &[], &[-4]), 256, true);
        assert_eq!(sets.missing_foll, [(4, false)]);
        for &(poc, long_term) in &sets.missing_foll {
            let generated = SlotPicture {
                poc,
                slot: None,
                id: u64::MAX,
            };
            buffer.insert_generated(generated, long_term);
        }
        assert_eq!(buffer.entries.len(), 1);

        // A later "Curr" entry on that picture has nothing a decode can bind.
        let sets = buffer.apply_rps(&rps(9, &[-5], &[], &[]), 256, false);
        assert_eq!(
            slot_lists(&sets),
            Err(ListError::Missing {
                poc: 4,
                long_term: false,
            })
        );
    }

    /// The engine's `VulkanDuplicateSlot` refuses what this gives.
    #[test]
    fn two_long_term_entries_on_one_picture_name_one_slot_twice() {
        let entry = RefEntry {
            frame: Some(picture(4, 3)),
            poc: 4,
            long_term: true,
        };
        let mut sets = RefPicSets::empty();
        sets.long_term = vec![entry.clone(), entry];
        let lists = slot_lists(&sets).expect("lists");
        assert_eq!(lists.long_term[..3], [3, 3, NO_SLOT]);
        assert_eq!(lists.references.len(), 2);
        assert_eq!(lists.references[0].slot, lists.references[1].slot);
    }

    #[test]
    fn the_pool_gives_the_lowest_free_slot_and_runs_out() {
        let mut pool = SlotPool::new(2);
        assert_eq!(pool.take(), Some(0));
        assert_eq!(pool.take(), Some(1));
        assert_eq!(pool.take(), None);
        pool.release(0);
        assert_eq!(pool.take(), Some(0));
        pool.release(9);
    }
}
