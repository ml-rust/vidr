//! Reference picture lists (8.3.4), the collocated picture, and
//! `NoBackwardPredFlag` (8.3.5) for one slice.

use std::sync::Arc;

use crate::bitstream::hevc_pred_weight::MAX_REFS;
use crate::bitstream::hevc_slice::{SLICE_TYPE_B, SLICE_TYPE_I, SliceSegmentHeader};
use crate::bitstream::hevc_sps_full::SeqParameterSet;
use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::motion::types::RefLists;
use crate::decode::cpu::hevc::recon::weights::ExplicitWeights;
use crate::decode::cpu::hevc::syntax::records::push_record;

use super::buffer::{Frame, RefEntry, RefPicSets};
use super::stored::StoredPicture;

/// `RefPicList0` and `RefPicList1` of one slice. The pictures stay behind handles, so a
/// slice copies no samples.
#[derive(Clone, Debug)]
pub(crate) struct SliceLists<P> {
    /// The pictures of each list, by reference index.
    pub(crate) frames: [Vec<P>; 2],
    /// The POC and long-term facts of each list.
    pub(crate) lists: RefLists,
    /// `ColPic`. `None` with `slice_temporal_mvp_enabled_flag` 0.
    pub(crate) col: Option<P>,
    /// `NoBackwardPredFlag`.
    pub(crate) no_backward_pred: bool,
    /// `collocated_from_l0_flag`.
    pub(crate) collocated_from_l0: bool,
    /// The weights of 8.5.3.3.4.3. `None` when `weightedPredFlag` is 0.
    pub(crate) weights: Option<Box<ExplicitWeights>>,
}

/// The lists of a slice whose pictures are finished, as the reconstruct stage reads them.
pub(crate) type SliceRefs<S> = SliceLists<Arc<Frame<S>>>;

impl<P: StoredPicture> SliceLists<P> {
    /// The state of an I slice: no lists.
    pub(crate) fn intra() -> Self {
        Self {
            frames: [Vec::new(), Vec::new()],
            lists: RefLists::default(),
            col: None,
            no_backward_pred: true,
            collocated_from_l0: true,
            weights: None,
        }
    }

    /// Runs 8.3.4 and 8.3.5 at the start of a slice. An I slice gets no lists.
    ///
    /// # Errors
    ///
    /// - [`HevcError::MissingReference`] when a list entry holds "no reference
    ///   picture".
    /// - [`HevcError::InvalidSyntax`] for a P or B slice with an empty set, or a list
    ///   entry past the set.
    /// - [`HevcError::Allocation`] when a list does not fit in memory.
    pub(crate) fn build(
        header: &SliceSegmentHeader,
        sets: &RefPicSets<P>,
        current_poc: i32,
        sps: &SeqParameterSet,
    ) -> Result<Self, HevcError> {
        if header.slice_type == SLICE_TYPE_I {
            return Ok(Self::intra());
        }
        let mut refs = Self::intra();
        refs.collocated_from_l0 = header.collocated_from_l0_flag;
        let lt = &sets.long_term;
        let l0_order = [&sets.before, &sets.after, lt];
        refs.fill_list(
            0,
            &l0_order,
            ListRequest {
                active: header.num_ref_idx_l0_active_minus1 as usize + 1,
                total: header.num_pic_total_curr as usize,
                entries: header
                    .ref_pic_list_modification_flag_l0
                    .then_some(&header.list_entry_l0),
            },
        )?;
        if header.slice_type == SLICE_TYPE_B {
            let l1_order = [&sets.after, &sets.before, lt];
            refs.fill_list(
                1,
                &l1_order,
                ListRequest {
                    active: header.num_ref_idx_l1_active_minus1 as usize + 1,
                    total: header.num_pic_total_curr as usize,
                    entries: header
                        .ref_pic_list_modification_flag_l1
                        .then_some(&header.list_entry_l1),
                },
            )?;
        }
        // 8.3.5.
        refs.no_backward_pred = refs
            .frames
            .iter()
            .flatten()
            .all(|frame| frame.poc() <= current_poc);
        if header.slice_temporal_mvp_enabled_flag {
            let col_list =
                usize::from(header.slice_type == SLICE_TYPE_B && !header.collocated_from_l0_flag);
            refs.col = refs.frames[col_list]
                .get(header.collocated_ref_idx as usize)
                .cloned();
        }
        refs.weights = header
            .pred_weight_table
            .as_ref()
            .map(|table| Box::new(ExplicitWeights::new(table, sps)));
        Ok(refs)
    }

    /// (8-8) and (8-9), or (8-10) and (8-11): the sets in `order` repeat until the
    /// temporary list holds `NumRpsCurrTempListX` entries, then the slice selects from
    /// it.
    fn fill_list(
        &mut self,
        list: usize,
        order: &[&Vec<RefEntry<P>>; 3],
        request: ListRequest<'_>,
    ) -> Result<(), HevcError> {
        let temporary_len = request.active.max(request.total);
        let chosen: Vec<&RefEntry<P>> = order
            .iter()
            .flat_map(|set| set.iter())
            .cycle()
            .take(temporary_len)
            .collect();
        if chosen.is_empty() {
            return Err(HevcError::InvalidSyntax {
                element: "NumPicTotalCurr",
                value: 0,
                allowed: "above 0 in a P or B slice",
            });
        }
        let active = request.active.min(MAX_REFS);
        for index in 0..active {
            let position = request
                .entries
                .map_or(index, |entries| usize::from(entries[index]));
            let Some(entry) = chosen.get(position) else {
                return Err(HevcError::InvalidSyntax {
                    element: "list_entry_lX",
                    value: position as i64,
                    allowed: "0 to NumPicTotalCurr - 1",
                });
            };
            let Some(frame) = &entry.frame else {
                return Err(HevcError::MissingReference { poc: entry.poc });
            };
            push_record(
                &mut self.frames[list],
                frame.clone(),
                "reference picture list",
            )?;
            self.lists.poc[list][index] = entry.poc_value();
            self.lists.long_term[list][index] = entry.long_term;
        }
        self.lists.count[list] = active;
        Ok(())
    }
}

impl<P: StoredPicture> RefEntry<P> {
    /// `PicOrderCntVal` of the picture. For a long-term entry with no MSB the entry
    /// names an LSB, and the picture holds the whole value.
    fn poc_value(&self) -> i32 {
        self.frame.as_ref().map_or(self.poc, StoredPicture::poc)
    }
}

/// What one list asks for.
#[derive(Clone, Copy, Debug)]
struct ListRequest<'a> {
    /// `num_ref_idx_lX_active_minus1 + 1`.
    active: usize,
    /// `NumPicTotalCurr`.
    total: usize,
    /// `list_entry_lX` with `ref_pic_list_modification_flag_lX` 1.
    entries: Option<&'a [u8; MAX_REFS]>,
}

#[cfg(test)]
#[path = "ref_lists_tests.rs"]
mod tests;
