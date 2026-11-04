//! The parse stage's output for one CTU. These plain records hold no bitstream
//! state, so a reconstruct stage can run on another thread.

use crate::decode::cpu::hevc::error::HevcError;

/// `CuPredMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PredMode {
    /// `MODE_INTRA`.
    Intra,
    /// `MODE_INTER`.
    Inter,
    /// `MODE_SKIP`.
    Skip,
}

/// `PartMode` (Table 7-10).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PartMode {
    /// `PART_2Nx2N`.
    TwoNxTwoN,
    /// `PART_2NxN`.
    TwoNxN,
    /// `PART_Nx2N`.
    NxTwoN,
    /// `PART_NxN`.
    NxN,
    /// `PART_2NxnU`.
    TwoNxnU,
    /// `PART_2NxnD`.
    TwoNxnD,
    /// `PART_nLx2N`.
    NLxTwoN,
    /// `PART_nRx2N`.
    NRxTwoN,
}

/// One prediction block in a coding block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PuRect {
    /// Horizontal offset in luma samples.
    pub(crate) x: u32,
    /// Vertical offset in luma samples.
    pub(crate) y: u32,
    /// `nPbW`.
    pub(crate) width: u32,
    /// `nPbH`.
    pub(crate) height: u32,
}

const fn rect(x: u32, y: u32, width: u32, height: u32) -> PuRect {
    PuRect {
        x,
        y,
        width,
        height,
    }
}

impl PartMode {
    /// The prediction blocks of a coding block of `size` samples, in the order of
    /// 7.3.8.5. The count says how many of the entries apply.
    pub(crate) const fn prediction_blocks(self, size: u32) -> ([PuRect; 4], usize) {
        let half = size / 2;
        let quarter = size / 4;
        let rest = size - quarter;
        let unused = rect(0, 0, 0, 0);
        match self {
            Self::TwoNxTwoN => ([rect(0, 0, size, size), unused, unused, unused], 1),
            Self::TwoNxN => (
                [
                    rect(0, 0, size, half),
                    rect(0, half, size, half),
                    unused,
                    unused,
                ],
                2,
            ),
            Self::NxTwoN => (
                [
                    rect(0, 0, half, size),
                    rect(half, 0, half, size),
                    unused,
                    unused,
                ],
                2,
            ),
            Self::TwoNxnU => (
                [
                    rect(0, 0, size, quarter),
                    rect(0, quarter, size, rest),
                    unused,
                    unused,
                ],
                2,
            ),
            Self::TwoNxnD => (
                [
                    rect(0, 0, size, rest),
                    rect(0, rest, size, quarter),
                    unused,
                    unused,
                ],
                2,
            ),
            Self::NLxTwoN => (
                [
                    rect(0, 0, quarter, size),
                    rect(quarter, 0, rest, size),
                    unused,
                    unused,
                ],
                2,
            ),
            Self::NRxTwoN => (
                [
                    rect(0, 0, rest, size),
                    rect(rest, 0, quarter, size),
                    unused,
                    unused,
                ],
                2,
            ),
            Self::NxN => (
                [
                    rect(0, 0, half, half),
                    rect(half, 0, half, half),
                    rect(0, half, half, half),
                    rect(half, half, half, half),
                ],
                4,
            ),
        }
    }
}

/// `inter_pred_idc` (Table 7-11).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InterPredIdc {
    /// `PRED_L0`.
    L0,
    /// `PRED_L1`.
    L1,
    /// `PRED_BI`.
    Bi,
}

/// One coding unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CuRecord {
    /// `x0`.
    pub(crate) x: u32,
    /// `y0`.
    pub(crate) y: u32,
    /// `log2CbSize`.
    pub(crate) log2_size: u32,
    /// `CuPredMode`.
    pub(crate) pred_mode: PredMode,
    /// `PartMode`.
    pub(crate) part_mode: PartMode,
    /// `cu_transquant_bypass_flag`.
    pub(crate) transquant_bypass: bool,
    /// `pcm_flag`.
    pub(crate) pcm: bool,
    /// `QpY` (8-283).
    pub(crate) qp_y: i32,
    /// `IntraPredModeY` of each prediction block. A single block repeats in all
    /// four entries. Blocks with no intra mode hold `INTRA_DC`.
    pub(crate) intra_luma: [u8; 4],
    /// `IntraPredModeC`.
    pub(crate) intra_chroma: u8,
    /// Index of this unit's first entry in [`CtuRecords::pus`].
    pub(crate) first_pu: usize,
    /// Index of this unit's first entry in [`CtuRecords::tus`].
    pub(crate) first_tu: usize,
    /// Index of this unit's first entry in [`CtuRecords::pcm_samples`].
    pub(crate) first_pcm_sample: usize,
}

/// One prediction unit, as parsed. Merge and motion vector derivation read it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PuRecord {
    /// `x0` of `prediction_unit`.
    pub(crate) x: u32,
    /// `y0` of `prediction_unit`.
    pub(crate) y: u32,
    /// `nPbW`.
    pub(crate) width: u32,
    /// `nPbH`.
    pub(crate) height: u32,
    /// `merge_flag`.
    pub(crate) merge_flag: bool,
    /// `merge_idx`.
    pub(crate) merge_idx: u32,
    /// `inter_pred_idc`.
    pub(crate) inter_pred_idc: InterPredIdc,
    /// `ref_idx_l0` and `ref_idx_l1`.
    pub(crate) ref_idx: [u32; 2],
    /// `MvdL0` and `MvdL1`, horizontal then vertical.
    pub(crate) mvd: [[i32; 2]; 2],
    /// `mvp_l0_flag` and `mvp_l1_flag`.
    pub(crate) mvp_flag: [bool; 2],
}

/// One transform unit. Every leaf of a transform tree has one, because intra
/// prediction runs per transform unit even with no residual.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TuRecord {
    /// `x0`.
    pub(crate) x: u32,
    /// `y0`.
    pub(crate) y: u32,
    /// `xBase`.
    pub(crate) x_base: u32,
    /// `yBase`.
    pub(crate) y_base: u32,
    /// `log2TrafoSize`.
    pub(crate) log2_size: u32,
    /// `trafoDepth`.
    pub(crate) depth: u32,
    /// `blkIdx`.
    pub(crate) blk_idx: u32,
    /// `cbf_luma`.
    pub(crate) cbf_luma: bool,
    /// `cbf_cb` of this unit, or of its parent at a `log2TrafoSize` of 2.
    pub(crate) cbf_cb: bool,
    /// `cbf_cr` of this unit, or of its parent at a `log2TrafoSize` of 2.
    pub(crate) cbf_cr: bool,
    /// Index of the coding unit in [`CtuRecords::cus`].
    pub(crate) cu: usize,
    /// Index of this unit's first entry in [`CtuRecords::tbs`].
    pub(crate) first_tb: usize,
    /// Transform blocks with residual in this unit.
    pub(crate) tb_count: usize,
}

/// One transform block with residual.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TbRecord {
    /// Luma location `x0` of `residual_coding`.
    pub(crate) x: u32,
    /// Luma location `y0` of `residual_coding`.
    pub(crate) y: u32,
    /// `log2TrafoSize` of `residual_coding`.
    pub(crate) log2_size: u32,
    /// `cIdx`.
    pub(crate) c_idx: usize,
    /// `transform_skip_flag`.
    pub(crate) transform_skip: bool,
    /// Index of this block's first entry in [`CtuRecords::coeffs`].
    pub(crate) first_coeff: usize,
    /// Levels that are not zero.
    pub(crate) coeff_count: usize,
}

/// One `TransCoeffLevel` that is not zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Coeff {
    /// `xC`.
    pub(crate) x: u8,
    /// `yC`.
    pub(crate) y: u8,
    /// The level, within `CoeffMinY` and `CoeffMaxY`.
    pub(crate) level: i16,
}

/// Everything parsed from one CTU, in decoding order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CtuRecords {
    pub(crate) cus: Vec<CuRecord>,
    pub(crate) pus: Vec<PuRecord>,
    pub(crate) tus: Vec<TuRecord>,
    pub(crate) tbs: Vec<TbRecord>,
    pub(crate) coeffs: Vec<Coeff>,
    /// `pcm_sample_luma` then `pcm_sample_chroma` of each PCM coding unit.
    pub(crate) pcm_samples: Vec<u16>,
}

impl CtuRecords {
    /// Empties every list and keeps their memory for the next CTU.
    pub(crate) fn clear(&mut self) {
        self.cus.clear();
        self.pus.clear();
        self.tus.clear();
        self.tbs.clear();
        self.coeffs.clear();
        self.pcm_samples.clear();
    }
}

/// Appends `item`, and names the list when memory runs out.
#[inline]
pub(crate) fn push_record<T>(
    list: &mut Vec<T>,
    item: T,
    what: &'static str,
) -> Result<(), HevcError> {
    list.try_reserve(1).map_err(|_| HevcError::Allocation {
        what,
        count: list.len().saturating_add(1),
    })?;
    list.push(item);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prediction_blocks_follow_7_3_8_5() {
        let cases = [
            (PartMode::TwoNxTwoN, vec![(0, 0, 16, 16)]),
            (PartMode::TwoNxN, vec![(0, 0, 16, 8), (0, 8, 16, 8)]),
            (PartMode::NxTwoN, vec![(0, 0, 8, 16), (8, 0, 8, 16)]),
            (PartMode::TwoNxnU, vec![(0, 0, 16, 4), (0, 4, 16, 12)]),
            (PartMode::TwoNxnD, vec![(0, 0, 16, 12), (0, 12, 16, 4)]),
            (PartMode::NLxTwoN, vec![(0, 0, 4, 16), (4, 0, 12, 16)]),
            (PartMode::NRxTwoN, vec![(0, 0, 12, 16), (12, 0, 4, 16)]),
            (
                PartMode::NxN,
                vec![(0, 0, 8, 8), (8, 0, 8, 8), (0, 8, 8, 8), (8, 8, 8, 8)],
            ),
        ];
        for (mode, expected) in cases {
            let (blocks, count) = mode.prediction_blocks(16);
            let found: Vec<_> = blocks
                .iter()
                .take(count)
                .map(|block| (block.x, block.y, block.width, block.height))
                .collect();
            assert_eq!(found, expected, "{mode:?}");
        }
    }

    #[test]
    fn clear_keeps_the_memory() {
        let mut records = CtuRecords::default();
        push_record(
            &mut records.coeffs,
            Coeff {
                x: 1,
                y: 2,
                level: 3,
            },
            "coefficients",
        )
        .unwrap();
        let capacity = records.coeffs.capacity();
        records.clear();
        assert!(records.coeffs.is_empty());
        assert_eq!(records.coeffs.capacity(), capacity);
    }
}
