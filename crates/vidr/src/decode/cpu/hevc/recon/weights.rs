//! Explicit weights of one slice from `pred_weight_table()` (8-265) to (8-274).

use crate::bitstream::hevc_pred_weight::{MAX_REFS, PredWeightList, PredWeightTable};
use crate::bitstream::hevc_sps_full::SeqParameterSet;
use crate::decode::cpu::dsp::inter::weight::{BlockWeights, shift1};

/// Values of one reference index, by `cIdx`.
type PerComponent = [i32; 3];

/// `w` and `o` of every reference index in both lists, ready for 8.5.3.3.4.3.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExplicitWeights {
    /// `log2Wd` of luma, then chroma.
    log2_wd: [u32; 2],
    weight: [[PerComponent; MAX_REFS]; 2],
    offset: [[PerComponent; MAX_REFS]; 2],
}

impl ExplicitWeights {
    /// Reads the table. Absent flags give `2 ^ denom` and 0, as 7.4.7.3 infers them.
    pub(crate) fn new(table: &PredWeightTable, sps: &SeqParameterSet) -> Self {
        let bit_depth = sps.bit_depth_y();
        let shift1 = shift1(bit_depth);
        // WpOffsetBdShiftY and WpOffsetBdShiftC. The stream check refuses unequal depth.
        let high_precision = sps.sps_range_extension.high_precision_offsets_enabled_flag;
        let offset_shift = if high_precision { 0 } else { bit_depth - 8 };
        let half_range = sps.wp_offset_half_range_c();
        let mut weights = Self {
            log2_wd: [
                u32::from(table.luma_log2_weight_denom) + shift1,
                table.chroma_log2_weight_denom().max(0) as u32 + shift1,
            ],
            weight: [[[0; 3]; MAX_REFS]; 2],
            offset: [[[0; 3]; MAX_REFS]; 2],
        };
        let lists: [&PredWeightList; 2] = [&table.l0, &table.l1];
        for (list_index, list) in lists.into_iter().enumerate() {
            for index in 0..MAX_REFS {
                weights.weight[list_index][index] = [
                    table.luma_weight(list, index),
                    table.chroma_weight(list, index, 0),
                    table.chroma_weight(list, index, 1),
                ];
                weights.offset[list_index][index] = [
                    list.luma_offset[index] << offset_shift,
                    (table.chroma_offset(list, index, 0, half_range) as i32) << offset_shift,
                    (table.chroma_offset(list, index, 1, half_range) as i32) << offset_shift,
                ];
            }
        }
        weights
    }

    /// The values of one block for `c_idx`. An unused list holds zeros.
    pub(crate) fn block(&self, c_idx: usize, ref_idx: [i8; 2]) -> BlockWeights {
        let component = c_idx.min(2);
        let read = |table: &[[PerComponent; MAX_REFS]; 2], list: usize| {
            table[list]
                .get(ref_idx[list] as usize)
                .map_or(0, |values| values[component])
        };
        BlockWeights {
            log2_wd: self.log2_wd[usize::from(component != 0)],
            weight: [read(&self.weight, 0), read(&self.weight, 1)],
            offset: [read(&self.offset, 0), read(&self.offset, 1)],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::hevc_sps_full::tests as sps_tests;

    fn sps(bit_depth_minus8: u32) -> SeqParameterSet {
        sps_tests::valid(&sps_tests::Spec {
            bit_depth_luma_minus8: bit_depth_minus8,
            bit_depth_chroma_minus8: bit_depth_minus8,
            ..sps_tests::Spec::default()
        })
    }

    fn table() -> PredWeightTable {
        let mut table = PredWeightTable {
            luma_log2_weight_denom: 2,
            delta_chroma_log2_weight_denom: 1,
            ..PredWeightTable::default()
        };
        table.l0.luma_weight_flag[1] = true;
        table.l0.delta_luma_weight[1] = 3;
        table.l0.luma_offset[1] = -5;
        table.l1.chroma_weight_flag[0] = true;
        table.l1.delta_chroma_weight[0] = [2, -2];
        table.l1.delta_chroma_offset[0] = [4, 0];
        table
    }

    /// 8 bits: shift1 6. (8-265): 2 + 6. LumaWeightL0 is 4 + 3, and an absent flag
    /// gives 4 with no offset.
    #[test]
    fn luma_weights_follow_8_265_to_8_269() {
        let weights = ExplicitWeights::new(&table(), &sps(0));
        let block = weights.block(0, [1, 2]);
        assert_eq!(block.log2_wd, 8);
        assert_eq!(block.weight, [7, 4]);
        assert_eq!(block.offset, [-5, 0]);
    }

    /// ChromaLog2WeightDenom 3. ChromaWeightL1 Cb is 8 + 2.
    /// (7-58): half range 128. 128 + 4 - ( ( 128 * 10 ) >> 3 ) = 132 - 160 = -28.
    /// Cr: weight 6, 128 + 0 - ( ( 128 * 6 ) >> 3 ) = 128 - 96 = 32.
    #[test]
    fn chroma_weights_follow_7_58() {
        let weights = ExplicitWeights::new(&table(), &sps(0));
        let cb = weights.block(1, [-1, 0]);
        assert_eq!(cb.log2_wd, 9);
        assert_eq!(cb.weight[1], 10);
        assert_eq!(cb.offset[1], -28);
        let cr = weights.block(2, [-1, 0]);
        assert_eq!(cr.weight[1], 6);
        assert_eq!(cr.offset[1], 32);
    }

    /// 10 bits: shift1 4, and offsets shift by 2.
    #[test]
    fn ten_bit_offsets_shift_up() {
        let weights = ExplicitWeights::new(&table(), &sps(2));
        let block = weights.block(0, [1, -1]);
        assert_eq!(block.log2_wd, 6);
        assert_eq!(block.offset[0], -20);
        // The unused list reads nothing.
        assert_eq!(block.weight[1], 0);
    }
}
