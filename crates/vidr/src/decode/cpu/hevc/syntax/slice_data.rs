//! `slice_segment_data()` (7.3.8.1) with CTB address order, and the CABAC
//! initialization, storage, and synchronization of 9.3.1 and 9.3.2.

use crate::decode::cpu::hevc::cabac::context::ContextTable;
use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::layout::PicLayout;
use crate::decode::cpu::hevc::params::CodingParams;
use crate::decode::cpu::hevc::qp::QpState;

use super::maps::ParseMaps;
use super::parser::CtuParser;
use super::records::CtuRecords;

/// The parse state that carries between the slice segments of one picture.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct EntropyCarry {
    /// `TableStateIdxWpp` and `TableMpsValWpp`.
    wavefront: Option<ContextTable>,
    /// `TableStateIdxDs` and `TableMpsValDs`.
    dependent: Option<ContextTable>,
    /// `QpY` of the last coding unit, for a dependent slice segment.
    last_cu_qp_y: i32,
    /// CTBs that slices have reached in this picture.
    pub(crate) decoded_ctbs: usize,
    /// `CtbAddrInTs` after the last slice segment of this picture.
    next_ts: usize,
}

/// One slice segment's data and place in the picture.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SegmentInput<'a> {
    pub(crate) params: &'a CodingParams,
    pub(crate) layout: &'a PicLayout,
    /// The NAL unit without emulation prevention bytes.
    pub(crate) data: &'a [u8],
    /// The byte where the slice data starts.
    pub(crate) start: usize,
    /// `slice_segment_address`.
    pub(crate) first_ctb_rs: u32,
    /// `dependent_slice_segment_flag`.
    pub(crate) dependent: bool,
}

/// Parses one slice segment. `on_ctu` runs after each CTU with the picture maps and
/// that CTU's records. It can swap the records for another set: the next CTU clears them.
///
/// # Errors
///
/// - [`HevcError::InvalidSyntax`] for an address outside the picture, an address that
///   does not follow the slice segment before it (7.4.2.4.5), a slice that runs past
///   the last CTB, an `end_of_subset_one_bit` of 0, or a dependent slice
///   segment with no slice segment before it.
/// - Any error of the syntax inside a CTU, and any error of `on_ctu`.
pub(crate) fn parse_slice_segment<F>(
    input: &SegmentInput<'_>,
    maps: &mut ParseMaps,
    carry: &mut EntropyCarry,
    mut on_ctu: F,
) -> Result<(), HevcError>
where
    F: FnMut(&ParseMaps, &mut CtuRecords) -> Result<(), HevcError>,
{
    let (params, layout) = (input.params, input.layout);
    let mut parser = CtuParser::new(params, layout, maps, input.data, input.start)?;
    let mut ts = first_ts(input)?;
    if ts < carry.next_ts {
        return Err(HevcError::InvalidSyntax {
            element: "slice_segment_address",
            value: i64::from(input.first_ctb_rs),
            allowed: "an address after the last CTB of the slice segment before it",
        });
    }
    if input.dependent {
        // Only a new slice resets qPY_PREV (8.6.1). A dependent segment continues.
        parser.qp.last_cu_qp_y = carry.last_cu_qp_y;
    }
    let mut segment_start = true;
    loop {
        let rs = layout.ts_to_rs(ts).map_or(0, |rs| rs as usize);
        let (x, y) = ctb_location(layout, rs);
        if !parser.maps.ctb_reached(rs) {
            carry.decoded_ctbs += 1;
        }
        parser.begin_ctu(x, y);
        parser.start_contexts(rs, ts, segment_start && input.dependent, carry)?;
        segment_start = false;
        if params.sao_luma || params.sao_chroma {
            parser.sao_syntax(rs);
        }
        parser.parse_coding_tree(x, y)?;
        on_ctu(parser.maps, &mut parser.records)?;
        parser.store_wavefront(rs, carry);
        let end = parser.end_of_slice_segment_flag();
        parser.check_not_exhausted()?;
        ts += 1;
        if end {
            if params.dependent_slice_segments {
                carry.dependent = Some(parser.cabac.contexts.clone());
            }
            carry.last_cu_qp_y = parser.qp.last_cu_qp_y;
            carry.next_ts = ts;
            return Ok(());
        }
        let Some(next_rs) = layout.ts_to_rs(ts) else {
            return Err(HevcError::InvalidSyntax {
                element: "end_of_slice_segment_flag",
                value: 0,
                allowed: "1 at the last CTB of the picture",
            });
        };
        parser.end_subset_if_due(rs, next_rs as usize)?;
    }
}

/// `CtbAddrRsToTs[ slice_segment_address ]`.
fn first_ts(input: &SegmentInput<'_>) -> Result<usize, HevcError> {
    let rs = input.first_ctb_rs as usize;
    match input.layout.rs_to_ts(rs) {
        Some(ts) => Ok(ts as usize),
        None => Err(HevcError::InvalidSyntax {
            element: "slice_segment_address",
            value: i64::from(input.first_ctb_rs),
            allowed: "0 to PicSizeInCtbsY - 1",
        }),
    }
}

/// The luma location of CTB `rs`.
fn ctb_location(layout: &PicLayout, rs: usize) -> (u32, u32) {
    let width = layout.width_in_ctbs() as usize;
    (
        ((rs % width) as u32) << layout.ctb_log2(),
        ((rs / width) as u32) << layout.ctb_log2(),
    )
}

impl CtuParser<'_, '_> {
    /// True when CTB `rs` is the first of its tile's row with wavefronts on.
    fn starts_wavefront_row(&self, rs: usize) -> bool {
        let width = self.layout.width_in_ctbs() as usize;
        self.params.entropy_coding_sync
            && (rs.is_multiple_of(width) || self.layout.tile_id(rs) != self.layout.tile_id(rs - 1))
    }

    /// The context part of 9.3.2.1 at the start of a CTU, with the `qPY_PREV` reset
    /// of 8.6.1 that the same conditions select.
    fn start_contexts(
        &mut self,
        rs: usize,
        ts: usize,
        dependent_start: bool,
        carry: &EntropyCarry,
    ) -> Result<(), HevcError> {
        let first_in_tile = ts == 0 || {
            let previous = self.layout.ts_to_rs(ts - 1).map_or(0, |rs| rs as usize);
            self.layout.tile_id(previous) != self.layout.tile_id(rs)
        };
        let params = self.params;
        if first_in_tile {
            self.cabac.contexts = ContextTable::new(params.init_type, params.slice_qp);
            self.qp = QpState::new(params.slice_qp);
        } else if self.starts_wavefront_row(rs) {
            // ( xNbT, yNbT ) of (9-3).
            let (x, y) = ctb_location(self.layout, rs);
            let size = 1i32 << self.layout.ctb_log2();
            let x_t = x as i32 + size;
            let y_t = y as i32 - size;
            self.cabac.contexts = match &carry.wavefront {
                Some(stored) if self.available((x, y), x_t, y_t) => stored.clone(),
                _ => ContextTable::new(params.init_type, params.slice_qp),
            };
            self.qp = QpState::new(params.slice_qp);
        } else if dependent_start {
            let Some(stored) = &carry.dependent else {
                return Err(HevcError::InvalidSyntax {
                    element: "dependent_slice_segment_flag",
                    value: 1,
                    allowed: "0 when no slice segment of this picture precedes it",
                });
            };
            self.cabac.contexts = stored.clone();
        }
        Ok(())
    }

    /// The storage process for wavefronts (9.3.1), after a CTU's syntax.
    fn store_wavefront(&self, rs: usize, carry: &mut EntropyCarry) {
        if !self.params.entropy_coding_sync {
            return;
        }
        let width = self.layout.width_in_ctbs() as usize;
        let second_in_row =
            rs % width == 1 || (rs > 1 && self.layout.tile_id(rs) != self.layout.tile_id(rs - 2));
        if second_in_row {
            carry.wavefront = Some(self.cabac.contexts.clone());
        }
    }

    /// The condition of 7.3.8.1 on `end_of_subset_one_bit`, the bit itself,
    /// `byte_alignment()`, and the engine restart of 9.3.2.6.
    fn end_subset_if_due(&mut self, rs: usize, next_rs: usize) -> Result<(), HevcError> {
        let tile_ends =
            self.params.tiles_enabled && self.layout.tile_id(next_rs) != self.layout.tile_id(rs);
        if !tile_ends && !self.starts_wavefront_row(next_rs) {
            return Ok(());
        }
        if !self.cabac.terminate() {
            return Err(HevcError::InvalidSyntax {
                element: "end_of_subset_one_bit",
                value: 0,
                allowed: "1",
            });
        }
        // 9.3.4.3.5: the last bit in ivlOffset is alignment_bit_equal_to_one.
        let start = self
            .cabac
            .engine
            .aligned_byte_after("alignment_bit_equal_to_zero")?;
        self.cabac.engine.restart_at(start)
    }
}
