//! The motion of the picture being decoded, in 4x4 luma units, and the reference
//! lists of its slices. Spatial candidates and `bS` read it.

use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::layout::PicLayout;
use crate::decode::cpu::hevc::syntax::maps::filled;
use crate::decode::cpu::hevc::syntax::records::push_record;

use super::col::{ColMotion, ColUnit};
use super::types::{PbMotion, RefLists};

/// The smallest prediction block side is 4 luma samples.
const UNIT_LOG2: u32 = 2;

/// Motion and reference lists of one picture. A coding unit fills its area before
/// any later block reads it, so a new picture clears nothing but the slice tables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MotionField {
    /// Map width in 4x4 units.
    stride: usize,
    units: Vec<PbMotion>,
    ctb_log2: u32,
    width_in_ctbs: usize,
    /// The index in `tables` of the slice that holds each CTB, in raster order.
    ctb_table: Vec<u32>,
    /// One entry per independent slice segment of this picture.
    tables: Vec<RefLists>,
}

impl MotionField {
    /// Allocates the field of one picture size.
    ///
    /// # Errors
    ///
    /// [`HevcError::Allocation`] for a size that does not fit in memory.
    pub(crate) fn new(width: u32, height: u32, layout: &PicLayout) -> Result<Self, HevcError> {
        let stride = (width >> UNIT_LOG2) as usize;
        let rows = (height >> UNIT_LOG2) as usize;
        let count = stride.checked_mul(rows).ok_or(HevcError::Allocation {
            what: "motion units",
            count: usize::MAX,
        })?;
        Ok(Self {
            stride,
            units: filled(count, PbMotion::NONE, "motion units")?,
            ctb_log2: layout.ctb_log2(),
            width_in_ctbs: layout.width_in_ctbs() as usize,
            ctb_table: filled(layout.ctb_count(), 0, "CTB reference tables")?,
            tables: Vec::new(),
        })
    }

    /// Prepares the field for a new picture of the same size.
    pub(crate) fn begin_picture(&mut self) {
        self.tables.clear();
    }

    /// Records the reference lists of a new slice.
    ///
    /// # Errors
    ///
    /// [`HevcError::Allocation`] when the table does not fit in memory.
    pub(crate) fn begin_slice(&mut self, lists: RefLists) -> Result<(), HevcError> {
        push_record(&mut self.tables, lists, "slice reference tables")
    }

    /// Records that CTB `rs` belongs to the latest slice.
    pub(crate) fn set_ctb(&mut self, rs: usize) {
        let table = self.tables.len().saturating_sub(1) as u32;
        if let Some(entry) = self.ctb_table.get_mut(rs) {
            *entry = table;
        }
    }

    /// Writes `motion` over a block, (8-80) to (8-85). A block past the picture
    /// edge writes only the units inside the picture.
    pub(crate) fn fill(&mut self, x: u32, y: u32, width: u32, height: u32, motion: PbMotion) {
        if self.stride == 0 {
            return;
        }
        let left = (x >> UNIT_LOG2) as usize;
        let right = (((x + width) >> UNIT_LOG2) as usize).min(self.stride);
        let top = (y >> UNIT_LOG2) as usize;
        let rows = (height >> UNIT_LOG2) as usize;
        for row in self
            .units
            .chunks_exact_mut(self.stride)
            .skip(top)
            .take(rows)
        {
            if let Some(cells) = row.get_mut(left..right) {
                cells.fill(motion);
            }
        }
    }

    /// The motion of the block covering a luma location.
    pub(crate) fn at(&self, x: u32, y: u32) -> PbMotion {
        let index = (y >> UNIT_LOG2) as usize * self.stride + (x >> UNIT_LOG2) as usize;
        self.units.get(index).copied().unwrap_or_default()
    }

    /// The reference lists of the slice covering a luma location.
    pub(crate) fn refs_at(&self, x: u32, y: u32) -> Option<&RefLists> {
        let rs = (y >> self.ctb_log2) as usize * self.width_in_ctbs + (x >> self.ctb_log2) as usize;
        let table = *self.ctb_table.get(rs)?;
        self.tables.get(table as usize)
    }

    /// `PicOrderCnt` of the two references of `motion` at a luma location. An unused
    /// list gives 0.
    pub(crate) fn ref_pocs(&self, x: u32, y: u32, motion: &PbMotion) -> [i32; 2] {
        let Some(refs) = self.refs_at(x, y) else {
            return [0; 2];
        };
        [
            refs.poc_of(0, motion.ref_idx[0]),
            refs.poc_of(1, motion.ref_idx[1]),
        ]
    }

    /// Writes the 16x16 form of this field (8.5.3.2.8). Each unit keeps the block
    /// covering its top-left sample, with the reference facts that block's slice
    /// held.
    ///
    /// # Errors
    ///
    /// [`HevcError::Allocation`] when `out` does not fit in memory.
    pub(crate) fn compress(
        &self,
        width: u32,
        height: u32,
        out: &mut ColMotion,
    ) -> Result<(), HevcError> {
        out.prepare(width, height)?;
        out.fill_with(|x, y| {
            let motion = self.at(x, y);
            let mut unit = ColUnit {
                motion,
                ..ColUnit::default()
            };
            if let Some(refs) = self.refs_at(x, y) {
                for list in 0..2 {
                    if motion.uses(list) {
                        unit.ref_poc[list] = refs.poc_of(list, motion.ref_idx[list]);
                        unit.long_term[list] = refs.is_long_term(list, motion.ref_idx[list]);
                    }
                }
            }
            unit
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::cpu::hevc::layout::LayoutSizes;
    use crate::decode::cpu::hevc::motion::types::Mv;

    fn layout() -> PicLayout {
        PicLayout::new(&LayoutSizes {
            width: 48,
            height: 32,
            ctb_log2: 4,
            min_tb_log2: 2,
            column_widths: &[3],
            row_heights: &[2],
        })
        .unwrap()
    }

    fn motion(x: i16, reference: i8) -> PbMotion {
        PbMotion {
            mv: [Mv::new(x, 0), Mv::default()],
            ref_idx: [reference, -1],
        }
    }

    #[test]
    fn fill_covers_the_block_and_stops_at_the_edge() {
        let mut field = MotionField::new(48, 32, &layout()).unwrap();
        field.fill(8, 4, 8, 4, motion(3, 0));
        assert_eq!(field.at(8, 4).mv[0].x, 3);
        assert_eq!(field.at(15, 7).mv[0].x, 3);
        assert_eq!(field.at(16, 4), PbMotion::NONE);
        assert_eq!(field.at(8, 8), PbMotion::NONE);
        assert_eq!(field.at(7, 4), PbMotion::NONE);
        field.fill(40, 24, 16, 16, motion(9, 0));
        assert_eq!(field.at(47, 31).mv[0].x, 9);
        // The block past the right edge does not wrap to the next row.
        assert_eq!(field.at(0, 28), PbMotion::NONE);
    }

    #[test]
    fn each_ctb_reads_the_lists_of_its_slice() {
        let mut field = MotionField::new(48, 32, &layout()).unwrap();
        let mut first = RefLists::default();
        first.poc[0][0] = 10;
        let mut second = RefLists::default();
        second.poc[0][0] = 20;
        field.begin_slice(first).unwrap();
        field.set_ctb(0);
        field.begin_slice(second).unwrap();
        field.set_ctb(1);
        assert_eq!(field.refs_at(15, 15).unwrap().poc[0][0], 10);
        assert_eq!(field.refs_at(16, 0).unwrap().poc[0][0], 20);
        assert_eq!(field.ref_pocs(16, 0, &motion(0, 0)), [20, 0]);
    }

    /// (8.5.3.2.8): a unit takes the block covering `( x >> 4 ) << 4`.
    #[test]
    fn compression_keeps_the_top_left_block_of_each_16x16() {
        let mut field = MotionField::new(48, 32, &layout()).unwrap();
        let mut refs = RefLists::default();
        refs.poc[0][0] = 7;
        refs.long_term[0][0] = true;
        field.begin_slice(refs).unwrap();
        for rs in 0..6 {
            field.set_ctb(rs);
        }
        field.fill(0, 0, 4, 4, motion(1, 0));
        field.fill(4, 0, 12, 16, motion(2, 0));
        field.fill(16, 0, 16, 16, motion(3, 0));
        let mut col = ColMotion::new();
        field.compress(48, 32, &mut col).unwrap();
        let unit = col.at(15, 15).unwrap();
        assert_eq!(unit.motion.mv[0].x, 1);
        assert_eq!(unit.ref_poc[0], 7);
        assert!(unit.long_term[0]);
        assert_eq!(col.at(31, 0).unwrap().motion.mv[0].x, 3);
        assert_eq!(col.at(32, 0).unwrap().motion, PbMotion::NONE);
        assert_eq!(col.at(48, 0), None);
    }
}
