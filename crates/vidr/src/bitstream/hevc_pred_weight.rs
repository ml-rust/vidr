//! HEVC `pred_weight_table()` (ITU-T H.265 7.3.6.3).
//!
//! The syntax omits a flag for a reference picture in another layer or with the
//! POC of the current picture. A layer 0 stream without screen content coding
//! has neither case, so every flag is present here.

use super::bits::BitReader;
use super::hevc_sps_full::SeqParameterSet;
use crate::error::BitstreamError;

/// The most reference pictures in one list.
pub(crate) const MAX_REFS: usize = 15;
/// The largest allowed `sumWeightL0Flags`, or sum of both lists (7.4.7.3).
const MAX_SUM_WEIGHT_FLAGS: u32 = 24;

/// The elements of one reference list, indexed by reference index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PredWeightList {
    /// `luma_weight_l0_flag` or `luma_weight_l1_flag`.
    pub(crate) luma_weight_flag: [bool; MAX_REFS],
    /// `chroma_weight_l0_flag` or `chroma_weight_l1_flag`.
    pub(crate) chroma_weight_flag: [bool; MAX_REFS],
    /// `delta_luma_weight_l0` or `delta_luma_weight_l1`.
    pub(crate) delta_luma_weight: [i16; MAX_REFS],
    /// `luma_offset_l0` or `luma_offset_l1`.
    pub(crate) luma_offset: [i32; MAX_REFS],
    /// `delta_chroma_weight_l0` or `delta_chroma_weight_l1`, for Cb then Cr.
    pub(crate) delta_chroma_weight: [[i16; 2]; MAX_REFS],
    /// `delta_chroma_offset_l0` or `delta_chroma_offset_l1`, for Cb then Cr.
    pub(crate) delta_chroma_offset: [[i32; 2]; MAX_REFS],
}

/// One `pred_weight_table()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PredWeightTable {
    pub(crate) luma_log2_weight_denom: u8,
    pub(crate) delta_chroma_log2_weight_denom: i8,
    pub(crate) l0: PredWeightList,
    /// All zero for a P slice.
    pub(crate) l1: PredWeightList,
}

/// The syntax element names of one list.
struct ListNames {
    luma_flag: &'static str,
    chroma_flag: &'static str,
    delta_luma: &'static str,
    luma_offset: &'static str,
    delta_chroma: &'static str,
    chroma_offset: &'static str,
}

const NAMES_L0: ListNames = ListNames {
    luma_flag: "luma_weight_l0_flag",
    chroma_flag: "chroma_weight_l0_flag",
    delta_luma: "delta_luma_weight_l0",
    luma_offset: "luma_offset_l0",
    delta_chroma: "delta_chroma_weight_l0",
    chroma_offset: "delta_chroma_offset_l0",
};
const NAMES_L1: ListNames = ListNames {
    luma_flag: "luma_weight_l1_flag",
    chroma_flag: "chroma_weight_l1_flag",
    delta_luma: "delta_luma_weight_l1",
    luma_offset: "luma_offset_l1",
    delta_chroma: "delta_chroma_weight_l1",
    chroma_offset: "delta_chroma_offset_l1",
};

impl PredWeightList {
    fn parse(
        reader: &mut BitReader<'_>,
        sps: &SeqParameterSet,
        count: usize,
        names: &ListNames,
    ) -> Result<Self, BitstreamError> {
        let chroma = sps.chroma_array_type() != 0;
        let mut list = Self::default();
        for flag in list.luma_weight_flag.iter_mut().take(count) {
            *flag = reader.bit(names.luma_flag)?;
        }
        if chroma {
            for flag in list.chroma_weight_flag.iter_mut().take(count) {
                *flag = reader.bit(names.chroma_flag)?;
            }
        }
        let half_y = sps.wp_offset_half_range_y();
        let half_c = sps.wp_offset_half_range_c();
        for index in 0..count {
            if list.luma_weight_flag[index] {
                list.delta_luma_weight[index] =
                    reader.se_range(names.delta_luma, -128, 127, "-128 to 127")? as i16;
                list.luma_offset[index] = reader.se_range(
                    names.luma_offset,
                    -half_y,
                    half_y - 1,
                    "-WpOffsetHalfRangeY to WpOffsetHalfRangeY - 1",
                )? as i32;
            }
            if list.chroma_weight_flag[index] {
                for component in 0..2 {
                    list.delta_chroma_weight[index][component] =
                        reader.se_range(names.delta_chroma, -128, 127, "-128 to 127")? as i16;
                    list.delta_chroma_offset[index][component] = reader.se_range(
                        names.chroma_offset,
                        -4 * half_c,
                        4 * half_c - 1,
                        "-4 * WpOffsetHalfRangeC to 4 * WpOffsetHalfRangeC - 1",
                    )? as i32;
                }
            }
        }
        Ok(list)
    }

    /// `sumWeightL0Flags` or `sumWeightL1Flags` (7.4.7.3).
    fn sum_weight_flags(&self) -> u32 {
        let count = |flags: &[bool]| flags.iter().filter(|flag| **flag).count() as u32;
        count(&self.luma_weight_flag) + 2 * count(&self.chroma_weight_flag)
    }
}

impl PredWeightTable {
    /// Reads the table. `num_ref_idx_l1` is `None` for a P slice. Counts are
    /// entries, at most [`MAX_REFS`].
    pub(crate) fn parse(
        reader: &mut BitReader<'_>,
        sps: &SeqParameterSet,
        num_ref_idx_l0: usize,
        num_ref_idx_l1: Option<usize>,
    ) -> Result<Self, BitstreamError> {
        let luma_log2_weight_denom = reader.ue_max("luma_log2_weight_denom", 7, "0 to 7")? as u8;
        let mut delta_chroma_log2_weight_denom = 0;
        if sps.chroma_array_type() != 0 {
            let denom = i64::from(luma_log2_weight_denom);
            delta_chroma_log2_weight_denom = reader.se_range(
                "delta_chroma_log2_weight_denom",
                -denom,
                7 - denom,
                "a value that gives ChromaLog2WeightDenom 0 to 7",
            )? as i8;
        }
        let l0 = PredWeightList::parse(reader, sps, num_ref_idx_l0, &NAMES_L0)?;
        let l1 = match num_ref_idx_l1 {
            Some(count) => PredWeightList::parse(reader, sps, count, &NAMES_L1)?,
            None => PredWeightList::default(),
        };
        let sum = l0.sum_weight_flags() + l1.sum_weight_flags();
        if sum > MAX_SUM_WEIGHT_FLAGS {
            return Err(reader.invalid(
                "sumWeightL0Flags",
                sum.into(),
                "at most 24 over all lists",
            ));
        }
        Ok(Self {
            luma_log2_weight_denom,
            delta_chroma_log2_weight_denom,
            l0,
            l1,
        })
    }

    /// `ChromaLog2WeightDenom`.
    pub(crate) fn chroma_log2_weight_denom(&self) -> i32 {
        i32::from(self.luma_log2_weight_denom) + i32::from(self.delta_chroma_log2_weight_denom)
    }

    /// `LumaWeightL0` or `LumaWeightL1` of `list` at reference `index`.
    pub(crate) fn luma_weight(&self, list: &PredWeightList, index: usize) -> i32 {
        (1 << self.luma_log2_weight_denom) + i32::from(list.delta_luma_weight[index])
    }

    /// `ChromaWeightL0` or `ChromaWeightL1` of `list` at reference `index`, for
    /// `component` 0 (Cb) or 1 (Cr).
    pub(crate) fn chroma_weight(
        &self,
        list: &PredWeightList,
        index: usize,
        component: usize,
    ) -> i32 {
        (1 << self.chroma_log2_weight_denom())
            + i32::from(list.delta_chroma_weight[index][component])
    }

    /// `ChromaOffsetL0` or `ChromaOffsetL1` (7-58). It is 0 without the chroma
    /// weight flag. `half_range` is `WpOffsetHalfRangeC`.
    pub(crate) fn chroma_offset(
        &self,
        list: &PredWeightList,
        index: usize,
        component: usize,
        half_range: i64,
    ) -> i64 {
        if !list.chroma_weight_flag[index] {
            return 0;
        }
        let weight = i64::from(self.chroma_weight(list, index, component));
        let scaled = (half_range * weight) >> self.chroma_log2_weight_denom();
        let value = half_range + i64::from(list.delta_chroma_offset[index][component]) - scaled;
        value.clamp(-half_range, half_range - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::super::hevc_sps_full::tests::{Spec as SpsSpec, valid as valid_sps};
    use super::super::test_bits::BitWriter;
    use super::*;

    fn parse(
        w: &BitWriter,
        sps: &SeqParameterSet,
        l0: usize,
        l1: Option<usize>,
    ) -> Result<PredWeightTable, BitstreamError> {
        let bytes = w.bytes();
        PredWeightTable::parse(&mut BitReader::new(&bytes, "test"), sps, l0, l1)
    }

    #[test]
    fn reads_weights_and_derives_the_values() {
        let sps = valid_sps(&SpsSpec::default());
        let mut w = BitWriter::new();
        // Denominators 5 and 4.
        w.ue(5).se(-1);
        // Luma flags 1, 0. Chroma flags 0, 1.
        w.flag(true).flag(false).flag(false).flag(true);
        // Reference 0: luma. Reference 1: chroma for Cb, then Cr.
        w.se(-3).se(9);
        w.se(2).se(-40).se(-2).se(40);
        // L1: one reference with luma only.
        w.flag(true).flag(false).se(127).se(-128);
        let table = parse(&w, &sps, 2, Some(1)).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(table.luma_log2_weight_denom, 5);
        assert_eq!(table.delta_chroma_log2_weight_denom, -1);
        assert_eq!(table.chroma_log2_weight_denom(), 4);
        assert_eq!(table.l0.luma_weight_flag[..2], [true, false]);
        assert_eq!(table.l0.chroma_weight_flag[..2], [false, true]);
        assert_eq!(table.l0.delta_luma_weight[0], -3);
        assert_eq!(table.l0.luma_offset[0], 9);
        assert_eq!(table.l0.delta_chroma_weight[1], [2, -2]);
        assert_eq!(table.l0.delta_chroma_offset[1], [-40, 40]);
        assert_eq!(table.l1.delta_luma_weight[0], 127);
        assert_eq!(table.l1.luma_offset[0], -128);
        assert_eq!(table.luma_weight(&table.l0, 0), 29);
        assert_eq!(table.luma_weight(&table.l0, 1), 32);
        assert_eq!(table.chroma_weight(&table.l0, 1, 0), 18);
        // 128 - 40 - ( ( 128 * 18 ) >> 4 ) = -56.
        assert_eq!(table.chroma_offset(&table.l0, 1, 0, 128), -56);
        // 128 + 40 - ( ( 128 * 14 ) >> 4 ) = 56.
        assert_eq!(table.chroma_offset(&table.l0, 1, 1, 128), 56);
        // No chroma flag at reference 0.
        assert_eq!(table.chroma_offset(&table.l0, 0, 0, 128), 0);
    }

    #[test]
    fn chroma_offset_clips_to_the_half_range() {
        let table = PredWeightTable {
            luma_log2_weight_denom: 0,
            ..PredWeightTable::default()
        };
        let mut list = PredWeightList::default();
        list.chroma_weight_flag[0] = true;
        list.delta_chroma_offset[0][0] = 511;
        assert_eq!(table.chroma_offset(&list, 0, 0, 128), 127);
        list.delta_chroma_offset[0][0] = -512;
        assert_eq!(table.chroma_offset(&list, 0, 0, 128), -128);
    }

    #[test]
    fn monochrome_reads_no_chroma_syntax() {
        let sps = valid_sps(&SpsSpec {
            chroma_format_idc: 0,
            ..SpsSpec::default()
        });
        let mut w = BitWriter::new();
        w.ue(2).flag(true).se(1).se(2);
        let table = parse(&w, &sps, 1, None).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(table.delta_chroma_log2_weight_denom, 0);
        assert_eq!(table.l0.luma_offset[0], 2);
    }

    #[test]
    fn high_precision_offsets_widen_the_ranges() {
        let plain = valid_sps(&SpsSpec {
            bit_depth_luma_minus8: 2,
            ..SpsSpec::default()
        });
        let wide = valid_sps(&SpsSpec {
            bit_depth_luma_minus8: 2,
            range_extension: Some([false, false, false, false, false, false, true, false, false]),
            ..SpsSpec::default()
        });
        let mut w = BitWriter::new();
        w.ue(0).se(0).flag(true).flag(false).se(0).se(511);
        assert!(matches!(
            parse(&w, &plain, 1, None),
            Err(BitstreamError::InvalidSignedField {
                field: "luma_offset_l0",
                value: 511,
                ..
            })
        ));
        let table = parse(&w, &wide, 1, None).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(table.l0.luma_offset[0], 511);
    }

    #[test]
    fn out_of_range_values_are_refused() {
        let sps = valid_sps(&SpsSpec::default());
        let mut w = BitWriter::new();
        w.ue(8);
        assert!(matches!(
            parse(&w, &sps, 1, None),
            Err(BitstreamError::InvalidField {
                field: "luma_log2_weight_denom",
                value: 8,
                ..
            })
        ));
        type Build = fn(&mut BitWriter);
        let cases: [(&str, i64, Build); 6] = [
            ("delta_chroma_log2_weight_denom", 3, |w| {
                w.ue(5).se(3);
            }),
            ("delta_chroma_log2_weight_denom", -6, |w| {
                w.ue(5).se(-6);
            }),
            ("delta_luma_weight_l0", 128, |w| {
                w.ue(0).se(0).flag(true).flag(false).se(128);
            }),
            ("luma_offset_l0", -129, |w| {
                w.ue(0).se(0).flag(true).flag(false).se(0).se(-129);
            }),
            ("delta_chroma_weight_l0", -129, |w| {
                w.ue(0).se(0).flag(false).flag(true).se(-129);
            }),
            ("delta_chroma_offset_l0", 512, |w| {
                w.ue(0).se(0).flag(false).flag(true).se(0).se(512);
            }),
        ];
        for (name, expected, build) in cases {
            let mut w = BitWriter::new();
            build(&mut w);
            let result = parse(&w, &sps, 1, None);
            assert!(
                matches!(
                    &result,
                    Err(BitstreamError::InvalidSignedField { field, value, .. })
                        if *field == name && *value == expected
                ),
                "{name}: {result:?}"
            );
        }
    }

    /// Nine references with chroma and luma weights give a sum of 27.
    #[test]
    fn too_many_weight_flags_are_refused() {
        let sps = valid_sps(&SpsSpec::default());
        let mut w = BitWriter::new();
        w.ue(0).se(0);
        for _ in 0..18 {
            w.flag(true);
        }
        for _ in 0..9 {
            w.se(0).se(0);
            w.se(0).se(0).se(0).se(0);
        }
        assert!(matches!(
            parse(&w, &sps, 9, None),
            Err(BitstreamError::InvalidField {
                field: "sumWeightL0Flags",
                value: 27,
                ..
            })
        ));
    }

    #[test]
    fn truncation_at_every_byte_is_an_error() {
        let sps = valid_sps(&SpsSpec::default());
        let mut w = BitWriter::new();
        w.ue(5).se(-1).flag(true).flag(true);
        w.se(-3).se(9).se(2).se(-40).se(-2).se(40);
        let bytes = w.bytes();
        for end in 0..bytes.len() - 1 {
            let result =
                PredWeightTable::parse(&mut BitReader::new(&bytes[..end], "test"), &sps, 1, None);
            assert!(result.is_err(), "cut at byte {end}");
        }
    }
}
