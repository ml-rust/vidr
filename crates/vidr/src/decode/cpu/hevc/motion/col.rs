//! The motion a finished picture keeps for later pictures: one unit per 16x16 luma
//! samples (8.5.3.2.8), with its reference facts resolved.

use crate::decode::cpu::hevc::error::HevcError;

use super::types::PbMotion;

/// The side of one unit in luma samples, as a shift.
const COL_LOG2: u32 = 4;

/// One compressed block. A block with no list in use is intra.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ColUnit {
    /// `MvLX`, `RefIdxLX`, and `PredFlagLX` of the block.
    pub(crate) motion: PbMotion,
    /// `PicOrderCnt( refPicListCol[ refIdxCol ] )` of each list.
    pub(crate) ref_poc: [i32; 2],
    /// `LongTermRefPic( ColPic, colPb, refIdxCol, listCol )` of each list.
    pub(crate) long_term: [bool; 2],
}

/// The compressed motion of one picture.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ColMotion {
    /// Units per row.
    stride: usize,
    /// Rows.
    rows: usize,
    units: Vec<ColUnit>,
}

impl ColMotion {
    /// A field of no units.
    pub(crate) const fn new() -> Self {
        Self {
            stride: 0,
            rows: 0,
            units: Vec::new(),
        }
    }

    /// Sets the size for a picture, and marks every unit intra. A buffer of the same
    /// size keeps its memory.
    ///
    /// # Errors
    ///
    /// [`HevcError::Allocation`] for a size that does not fit in memory.
    pub(crate) fn prepare(&mut self, width: u32, height: u32) -> Result<(), HevcError> {
        let stride = (width as usize).div_ceil(1 << COL_LOG2);
        let rows = (height as usize).div_ceil(1 << COL_LOG2);
        let count = stride.checked_mul(rows).ok_or(HevcError::Allocation {
            what: "collocated motion",
            count: usize::MAX,
        })?;
        self.units.clear();
        self.units
            .try_reserve_exact(count)
            .map_err(|_| HevcError::Allocation {
                what: "collocated motion",
                count,
            })?;
        self.units.resize(count, ColUnit::default());
        self.stride = stride;
        self.rows = rows;
        Ok(())
    }

    /// Sets each unit from the luma location of its top-left sample.
    pub(super) fn fill_with(&mut self, unit_at: impl Fn(u32, u32) -> ColUnit) {
        if self.stride == 0 {
            return;
        }
        for (row, cells) in self.units.chunks_exact_mut(self.stride).enumerate() {
            for (column, cell) in cells.iter_mut().enumerate() {
                *cell = unit_at((column as u32) << COL_LOG2, (row as u32) << COL_LOG2);
            }
        }
    }

    /// The unit covering a luma location, or `None` outside the picture.
    pub(crate) fn at(&self, x: u32, y: u32) -> Option<ColUnit> {
        let (column, row) = ((x >> COL_LOG2) as usize, (y >> COL_LOG2) as usize);
        if column >= self.stride || row >= self.rows {
            return None;
        }
        self.units.get(row * self.stride + column).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_field_holds_nothing() {
        assert_eq!(ColMotion::new().at(0, 0), None);
    }

    #[test]
    fn prepare_rounds_the_size_up_and_marks_units_intra() {
        let mut col = ColMotion::new();
        col.prepare(40, 16).unwrap();
        assert_eq!(col.at(39, 15).unwrap().motion, PbMotion::NONE);
        assert_eq!(col.at(48, 0), None);
        assert_eq!(col.at(0, 16), None);
    }

    #[test]
    fn prepare_clears_old_units() {
        let mut col = ColMotion::new();
        col.prepare(16, 16).unwrap();
        col.fill_with(|_, _| ColUnit {
            ref_poc: [5, 6],
            ..ColUnit::default()
        });
        assert_eq!(col.at(0, 0).unwrap().ref_poc, [5, 6]);
        col.prepare(16, 16).unwrap();
        assert_eq!(col.at(0, 0).unwrap().ref_poc, [0, 0]);
    }

    #[test]
    fn an_overflowing_size_is_refused_before_allocation() {
        let mut col = ColMotion::new();
        assert!(matches!(
            col.prepare(u32::MAX, u32::MAX),
            Err(HevcError::Allocation { .. })
        ));
    }
}
