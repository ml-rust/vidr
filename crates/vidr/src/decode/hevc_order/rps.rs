//! The five lists of picture order counts of a reference picture set (8-5).

use crate::bitstream::hevc_rps::MAX_PICS;
use crate::bitstream::hevc_slice::SliceSegmentHeader;
use crate::bitstream::hevc_sps_full::SeqParameterSet;

/// One long-term entry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LongTermPoc {
    /// `PocLtCurr` or `PocLtFoll`.
    pub(crate) poc: i32,
    /// `CurrDeltaPocMsbPresentFlag` or `FollDeltaPocMsbPresentFlag`.
    pub(crate) msb_present: bool,
}

/// A list that holds at most [`MAX_PICS`] entries. A push past the end drops the
/// entry. The slice header parser holds each list below that size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PocList<T> {
    entries: [T; MAX_PICS],
    count: usize,
}

impl<T: Copy + Default> PocList<T> {
    fn new() -> Self {
        Self {
            entries: [T::default(); MAX_PICS],
            count: 0,
        }
    }

    fn push(&mut self, entry: T) {
        if let Some(slot) = self.entries.get_mut(self.count) {
            *slot = entry;
            self.count += 1;
        }
    }

    /// Appends an entry from a test in another module.
    #[cfg(all(test, feature = "cpu-hevc"))]
    pub(crate) fn push_for_tests(&mut self, entry: T) {
        self.push(entry);
    }

    /// The entries in order.
    pub(crate) fn as_slice(&self) -> &[T] {
        self.entries.get(..self.count).unwrap_or_default()
    }
}

/// `PocStCurrBefore`, `PocStCurrAfter`, `PocStFoll`, `PocLtCurr`, and `PocLtFoll`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RpsPocs {
    pub(crate) st_curr_before: PocList<i32>,
    pub(crate) st_curr_after: PocList<i32>,
    pub(crate) st_foll: PocList<i32>,
    pub(crate) lt_curr: PocList<LongTermPoc>,
    pub(crate) lt_foll: PocList<LongTermPoc>,
}

impl RpsPocs {
    /// The entries of the five lists.
    pub(crate) fn entry_count(&self) -> usize {
        self.st_curr_before.as_slice().len()
            + self.st_curr_after.as_slice().len()
            + self.st_foll.as_slice().len()
            + self.lt_curr.as_slice().len()
            + self.lt_foll.as_slice().len()
    }

    /// (8-5) for a picture with `PicOrderCntVal` `poc`. An IDR picture has no set in
    /// its header, and gets five empty lists.
    pub(crate) fn derive(header: &SliceSegmentHeader, sps: &SeqParameterSet, poc: i32) -> Self {
        let mut lists = Self {
            st_curr_before: PocList::new(),
            st_curr_after: PocList::new(),
            st_foll: PocList::new(),
            lt_curr: PocList::new(),
            lt_foll: PocList::new(),
        };
        let Some(set) = header.current_short_term_set(sps) else {
            return lists;
        };
        let negative = usize::from(set.num_negative_pics).min(MAX_PICS);
        for index in 0..negative {
            let value = poc.wrapping_add(set.delta_poc_s0[index]);
            if set.used_by_curr_pic_s0[index] {
                lists.st_curr_before.push(value);
            } else {
                lists.st_foll.push(value);
            }
        }
        let positive = usize::from(set.num_positive_pics).min(MAX_PICS);
        for index in 0..positive {
            let value = poc.wrapping_add(set.delta_poc_s1[index]);
            if set.used_by_curr_pic_s1[index] {
                lists.st_curr_after.push(value);
            } else {
                lists.st_foll.push(value);
            }
        }
        let max_lsb = 1i64 << (sps.log2_max_pic_order_cnt_lsb_minus4 + 4);
        let long_term = (header.num_long_term_sps + header.num_long_term_pics) as usize;
        for index in 0..long_term.min(header.poc_lsb_lt.len()) {
            let msb_present = header.delta_poc_msb_present_flag[index];
            let mut value = i64::from(header.poc_lsb_lt[index]);
            if msb_present {
                let cycle = header.delta_poc_msb_cycle_lt_sum[index] as i64;
                value +=
                    i64::from(poc) - cycle.wrapping_mul(max_lsb) - (i64::from(poc) & (max_lsb - 1));
            }
            let entry = LongTermPoc {
                poc: value as i32,
                msb_present,
            };
            if header.used_by_curr_pic_lt[index] {
                lists.lt_curr.push(entry);
            } else {
                lists.lt_foll.push(entry);
            }
        }
        lists
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::hevc_rps::ShortTermRefPicSet;
    use crate::bitstream::hevc_sps_full::tests as sps_tests;

    const IDR_W_RADL: u8 = 19;
    const TRAIL_R: u8 = 1;

    fn set() -> ShortTermRefPicSet {
        let mut set = ShortTermRefPicSet {
            num_negative_pics: 3,
            num_positive_pics: 2,
            ..ShortTermRefPicSet::default()
        };
        set.delta_poc_s0[..3].copy_from_slice(&[-1, -2, -4]);
        set.used_by_curr_pic_s0[..3].copy_from_slice(&[true, false, true]);
        set.delta_poc_s1[..2].copy_from_slice(&[1, 3]);
        set.used_by_curr_pic_s1[..2].copy_from_slice(&[false, true]);
        set
    }

    fn header() -> SliceSegmentHeader {
        SliceSegmentHeader {
            nal_unit_type: TRAIL_R,
            short_term_ref_pic_set: Some(set()),
            ..SliceSegmentHeader::default()
        }
    }

    fn sps() -> SeqParameterSet {
        sps_tests::valid(&sps_tests::Spec::default())
    }

    #[test]
    fn an_idr_picture_has_empty_lists() {
        let header = SliceSegmentHeader {
            nal_unit_type: IDR_W_RADL,
            ..SliceSegmentHeader::default()
        };
        let lists = RpsPocs::derive(&header, &sps(), 0);
        assert!(lists.st_curr_before.as_slice().is_empty());
        assert!(lists.st_foll.as_slice().is_empty());
        assert!(lists.lt_curr.as_slice().is_empty());
    }

    /// (8-5) at POC 20.
    #[test]
    fn short_term_entries_split_by_their_used_flags() {
        let lists = RpsPocs::derive(&header(), &sps(), 20);
        assert_eq!(lists.st_curr_before.as_slice(), &[19, 16]);
        assert_eq!(lists.st_curr_after.as_slice(), &[23]);
        // Negative entries come first, then positive ones.
        assert_eq!(lists.st_foll.as_slice(), &[18, 21]);
    }

    /// (8-5): an entry with no MSB keeps its LSB. `Spec::default` gives a
    /// MaxPicOrderCntLsb of 256. An entry with the MSB is at POC
    /// 3 + 40 - 1 * 256 - ( 40 & 255 ). That POC is -253.
    #[test]
    fn long_term_entries_follow_the_msb_flag() {
        let mut header = header();
        header.num_long_term_pics = 2;
        header.poc_lsb_lt[0] = 7;
        header.used_by_curr_pic_lt[0] = true;
        header.poc_lsb_lt[1] = 3;
        header.delta_poc_msb_present_flag[1] = true;
        header.delta_poc_msb_cycle_lt_sum[1] = 1;
        let lists = RpsPocs::derive(&header, &sps(), 40);
        assert_eq!(
            lists.lt_curr.as_slice(),
            &[LongTermPoc {
                poc: 7,
                msb_present: false
            }]
        );
        assert_eq!(
            lists.lt_foll.as_slice(),
            &[LongTermPoc {
                poc: -253,
                msb_present: true
            }]
        );
    }
}
