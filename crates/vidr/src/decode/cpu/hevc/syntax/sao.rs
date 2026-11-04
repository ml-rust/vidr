//! `sao()` (7.3.8.3) with the semantics of 7.4.9.3.

use crate::decode::cpu::hevc::cabac::context as ctx;

use super::parser::CtuParser;

/// Offsets per plane in the syntax.
const OFFSET_COUNT: usize = 4;

/// `SaoTypeIdx` with the values each type reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum SaoKind {
    /// `SaoTypeIdx` 0.
    #[default]
    Off,
    /// `SaoTypeIdx` 1, with `sao_band_position`.
    Band(u8),
    /// `SaoTypeIdx` 2, with `SaoEoClass`.
    Edge(u8),
}

/// The SAO values of one plane in one CTB.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SaoPlane {
    pub(crate) kind: SaoKind,
    /// `SaoOffsetVal` (7-72). `log2OffsetScale` is 0 in Main and Main 10.
    pub(crate) offsets: [i16; 5],
}

/// The SAO values of one CTB, by `cIdx`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SaoCtb {
    pub(crate) planes: [SaoPlane; 3],
}

impl CtuParser<'_, '_> {
    /// Parses `sao()` for CTB `rs`, and records the result. The caller checks that
    /// the slice enables SAO.
    pub(super) fn sao_syntax(&mut self, rs: usize) {
        let values = self
            .sao_merge_source(rs)
            .unwrap_or_else(|| self.sao_values());
        self.maps.set_sao(rs, values);
    }

    /// The values of the CTB that `sao_merge_left_flag` or `sao_merge_up_flag`
    /// names, or `None` when both are 0.
    fn sao_merge_source(&mut self, rs: usize) -> Option<SaoCtb> {
        let width = self.layout.width_in_ctbs() as usize;
        let slice_start = self.params.slice_addr_rs as usize;
        let tile = self.layout.tile_id(rs);
        if !rs.is_multiple_of(width) && rs > slice_start && tile == self.layout.tile_id(rs - 1) {
            let merge = self.cabac.decision(ctx::SAO_MERGE);
            if merge {
                return Some(self.maps.sao(rs - 1));
            }
        }
        if rs >= width && rs - width >= slice_start && tile == self.layout.tile_id(rs - width) {
            let merge = self.cabac.decision(ctx::SAO_MERGE);
            if merge {
                return Some(self.maps.sao(rs - width));
            }
        }
        None
    }

    /// The syntax after the merge flags.
    fn sao_values(&mut self) -> SaoCtb {
        let mut values = SaoCtb::default();
        // Luma and Cb read these, and Cr repeats the values of Cb (7.4.9.3).
        let mut type_idx = 0;
        let mut class = 0;
        for (c_idx, plane) in values.planes.iter_mut().enumerate() {
            let enabled = if c_idx == 0 {
                self.params.sao_luma
            } else {
                self.params.sao_chroma
            };
            if !enabled {
                continue;
            }
            if c_idx < 2 {
                type_idx = self.sao_type_idx();
            }
            if type_idx != 0 {
                *plane = self.sao_plane(c_idx, type_idx, &mut class);
            }
        }
        values
    }

    /// `sao_type_idx_luma` or `sao_type_idx_chroma`: TR with `cMax` 2, the first bin
    /// on a context and the second in bypass (Table 9-48).
    fn sao_type_idx(&mut self) -> u32 {
        if !self.cabac.decision(ctx::SAO_TYPE_IDX) {
            return 0;
        }
        if self.cabac.bypass() { 2 } else { 1 }
    }

    /// The offsets and the type's own element for one plane.
    fn sao_plane(&mut self, c_idx: usize, type_idx: u32, class: &mut u8) -> SaoPlane {
        let c_max = (1u32 << (self.params.bit_depth.min(10) - 5)) - 1;
        let mut magnitudes = [0i16; OFFSET_COUNT];
        for magnitude in &mut magnitudes {
            *magnitude = self.bypass_unary(c_max) as i16;
        }
        if type_idx == 1 {
            let mut offsets = [0i16; 5];
            for (offset, &magnitude) in offsets[1..].iter_mut().zip(&magnitudes) {
                let negative = magnitude != 0 && self.cabac.bypass();
                *offset = if negative { -magnitude } else { magnitude };
            }
            let position = self.cabac.bypass_bits(5) as u8;
            return SaoPlane {
                kind: SaoKind::Band(position),
                offsets,
            };
        }
        if c_idx < 2 {
            *class = self.cabac.bypass_bits(2) as u8;
        }
        // 7.4.9.3 infers a sign of 0 for the first two offsets and 1 for the others.
        SaoPlane {
            kind: SaoKind::Edge(*class),
            offsets: [
                0,
                magnitudes[0],
                magnitudes[1],
                -magnitudes[2],
                -magnitudes[3],
            ],
        }
    }
}
