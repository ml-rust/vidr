//! HEVC `st_ref_pic_set()` (ITU-T H.265 7.3.7), with the lists 7.4.8 derives.

use super::bits::BitReader;
use crate::error::BitstreamError;

/// The most pictures one set holds. `MaxDpbSize` is at most 16 (A.4.2).
pub(crate) const MAX_PICS: usize = 16;
/// The largest `delta_poc_s0_minus1`, `delta_poc_s1_minus1`, and
/// `abs_delta_rps_minus1`.
const MAX_DELTA_MINUS1: u32 = 32_767;
const DELTA_RANGE: &str = "0 to 32767";

/// One `st_ref_pic_set()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ShortTermRefPicSet {
    pub(crate) inter_ref_pic_set_prediction_flag: bool,
    /// 0 unless the set is predicted and sits in a slice header.
    pub(crate) delta_idx_minus1: u32,
    pub(crate) delta_rps_sign: bool,
    pub(crate) abs_delta_rps_minus1: u32,
    /// `used_by_curr_pic_flag[ j ]` at bit `j`.
    pub(crate) used_by_curr_pic_flag: u32,
    /// `use_delta_flag[ j ]` at bit `j`, 1 where the syntax omits it.
    pub(crate) use_delta_flag: u32,
    /// `NumDeltaPocs[ RefRpsIdx ] + 1`: the flag pairs read. 0 for an explicit
    /// set.
    pub(crate) num_prediction_flags: u8,
    /// `NumNegativePics`, equal to `num_negative_pics` for an explicit set.
    pub(crate) num_negative_pics: u8,
    /// `NumPositivePics`, equal to `num_positive_pics` for an explicit set.
    pub(crate) num_positive_pics: u8,
    /// Syntax elements of an explicit set. 0 for a predicted set.
    pub(crate) delta_poc_s0_minus1: [u16; MAX_PICS],
    pub(crate) delta_poc_s1_minus1: [u16; MAX_PICS],
    /// `DeltaPocS0`.
    pub(crate) delta_poc_s0: [i32; MAX_PICS],
    /// `DeltaPocS1`.
    pub(crate) delta_poc_s1: [i32; MAX_PICS],
    /// `UsedByCurrPicS0`.
    pub(crate) used_by_curr_pic_s0: [bool; MAX_PICS],
    /// `UsedByCurrPicS1`.
    pub(crate) used_by_curr_pic_s1: [bool; MAX_PICS],
}

/// One derived list of (7-61) or (7-62), filled in the syntax order.
struct PocList {
    poc: [i32; MAX_PICS],
    used: [bool; MAX_PICS],
    count: usize,
    field: &'static str,
}

impl PocList {
    const fn new(field: &'static str) -> Self {
        Self {
            poc: [0; MAX_PICS],
            used: [false; MAX_PICS],
            count: 0,
            field,
        }
    }

    fn push(&mut self, reader: &BitReader<'_>, poc: i32, used: bool) -> Result<(), BitstreamError> {
        let (Some(poc_slot), Some(used_slot)) =
            (self.poc.get_mut(self.count), self.used.get_mut(self.count))
        else {
            return Err(reader.invalid(self.field, self.count as u64 + 1, "at most 16"));
        };
        *poc_slot = poc;
        *used_slot = used;
        self.count += 1;
        Ok(())
    }
}

impl ShortTermRefPicSet {
    /// Reads `st_ref_pic_set( previous.len() )`. `previous` holds the sets of
    /// lower index: every set of the SPS for a set in a slice header.
    pub(crate) fn parse(
        reader: &mut BitReader<'_>,
        previous: &[Self],
        in_slice_header: bool,
    ) -> Result<Self, BitstreamError> {
        let predicted = !previous.is_empty() && reader.bit("inter_ref_pic_set_prediction_flag")?;
        if predicted {
            Self::parse_predicted(reader, previous, in_slice_header)
        } else {
            Self::parse_explicit(reader)
        }
    }

    fn parse_explicit(reader: &mut BitReader<'_>) -> Result<Self, BitstreamError> {
        let negative = reader.ue_max("num_negative_pics", MAX_PICS as u32, "0 to 16")?;
        let positive = reader.ue_max(
            "num_positive_pics",
            MAX_PICS as u32 - negative,
            "0 to 16 minus num_negative_pics",
        )?;
        let mut set = Self {
            num_negative_pics: negative as u8,
            num_positive_pics: positive as u8,
            ..Self::default()
        };
        let mut poc = 0i32;
        for index in 0..negative as usize {
            let delta = reader.ue_max("delta_poc_s0_minus1", MAX_DELTA_MINUS1, DELTA_RANGE)?;
            poc -= delta as i32 + 1;
            set.delta_poc_s0_minus1[index] = delta as u16;
            set.delta_poc_s0[index] = poc;
            set.used_by_curr_pic_s0[index] = reader.bit("used_by_curr_pic_s0_flag")?;
        }
        poc = 0;
        for index in 0..positive as usize {
            let delta = reader.ue_max("delta_poc_s1_minus1", MAX_DELTA_MINUS1, DELTA_RANGE)?;
            poc += delta as i32 + 1;
            set.delta_poc_s1_minus1[index] = delta as u16;
            set.delta_poc_s1[index] = poc;
            set.used_by_curr_pic_s1[index] = reader.bit("used_by_curr_pic_s1_flag")?;
        }
        Ok(set)
    }

    fn parse_predicted(
        reader: &mut BitReader<'_>,
        previous: &[Self],
        in_slice_header: bool,
    ) -> Result<Self, BitstreamError> {
        let delta_idx_minus1 = if in_slice_header {
            reader.ue_max(
                "delta_idx_minus1",
                previous.len() as u32 - 1,
                "0 to stRpsIdx - 1",
            )?
        } else {
            0
        };
        let source = previous
            .get(previous.len() - 1 - delta_idx_minus1 as usize)
            .ok_or_else(|| {
                reader.invalid(
                    "delta_idx_minus1",
                    u64::from(delta_idx_minus1),
                    "0 to stRpsIdx - 1",
                )
            })?;
        let delta_rps_sign = reader.bit("delta_rps_sign")?;
        let abs_delta_rps_minus1 =
            reader.ue_max("abs_delta_rps_minus1", MAX_DELTA_MINUS1, DELTA_RANGE)?;
        let magnitude = abs_delta_rps_minus1 as i32 + 1;
        let delta_rps = if delta_rps_sign {
            -magnitude
        } else {
            magnitude
        };

        let source_negative = usize::from(source.num_negative_pics);
        let source_positive = usize::from(source.num_positive_pics);
        let source_total = source_negative + source_positive;
        let mut used_flags = 0u32;
        let mut use_delta_flags = 0u32;
        for j in 0..=source_total {
            let used = reader.bit("used_by_curr_pic_flag")?;
            if used {
                used_flags |= 1 << j;
            }
            if used || reader.bit("use_delta_flag")? {
                use_delta_flags |= 1 << j;
            }
        }
        let used = |j: usize| (used_flags >> j) & 1 == 1;
        let take = |j: usize| (use_delta_flags >> j) & 1 == 1;

        // (7-61).
        let mut negative = PocList::new("NumNegativePics");
        for j in (0..source_positive).rev() {
            let poc = source.delta_poc_s1[j] + delta_rps;
            if poc < 0 && take(source_negative + j) {
                negative.push(reader, poc, used(source_negative + j))?;
            }
        }
        if delta_rps < 0 && take(source_total) {
            negative.push(reader, delta_rps, used(source_total))?;
        }
        for j in 0..source_negative {
            let poc = source.delta_poc_s0[j] + delta_rps;
            if poc < 0 && take(j) {
                negative.push(reader, poc, used(j))?;
            }
        }

        // (7-62).
        let mut positive = PocList::new("NumPositivePics");
        for j in (0..source_negative).rev() {
            let poc = source.delta_poc_s0[j] + delta_rps;
            if poc > 0 && take(j) {
                positive.push(reader, poc, used(j))?;
            }
        }
        if delta_rps > 0 && take(source_total) {
            positive.push(reader, delta_rps, used(source_total))?;
        }
        for j in 0..source_positive {
            let poc = source.delta_poc_s1[j] + delta_rps;
            if poc > 0 && take(source_negative + j) {
                positive.push(reader, poc, used(source_negative + j))?;
            }
        }
        if negative.count + positive.count > MAX_PICS {
            return Err(reader.invalid(
                "NumDeltaPocs",
                (negative.count + positive.count) as u64,
                "at most 16",
            ));
        }

        Ok(Self {
            inter_ref_pic_set_prediction_flag: true,
            delta_idx_minus1,
            delta_rps_sign,
            abs_delta_rps_minus1,
            used_by_curr_pic_flag: used_flags,
            use_delta_flag: use_delta_flags,
            num_prediction_flags: source_total as u8 + 1,
            num_negative_pics: negative.count as u8,
            num_positive_pics: positive.count as u8,
            delta_poc_s0: negative.poc,
            delta_poc_s1: positive.poc,
            used_by_curr_pic_s0: negative.used,
            used_by_curr_pic_s1: positive.used,
            ..Self::default()
        })
    }

    /// Checks the limits 7.4.8 sets on an explicit set from
    /// `sps_max_dec_pic_buffering_minus1[ sps_max_sub_layers_minus1 ]`.
    pub(crate) fn check_dpb_limit(
        &self,
        structure: &'static str,
        max_dec_pic_buffering_minus1: u32,
    ) -> Result<(), BitstreamError> {
        if self.inter_ref_pic_set_prediction_flag {
            return Ok(());
        }
        let negative = u32::from(self.num_negative_pics);
        let positive = u32::from(self.num_positive_pics);
        if negative > max_dec_pic_buffering_minus1 {
            return Err(BitstreamError::InvalidField {
                structure,
                field: "num_negative_pics",
                value: u64::from(negative),
                allowed: "0 to sps_max_dec_pic_buffering_minus1 of the highest sub-layer",
            });
        }
        if positive > max_dec_pic_buffering_minus1 - negative {
            return Err(BitstreamError::InvalidField {
                structure,
                field: "num_positive_pics",
                value: u64::from(positive),
                allowed: "0 to sps_max_dec_pic_buffering_minus1 of the highest sub-layer \
                          minus num_negative_pics",
            });
        }
        Ok(())
    }

    /// `NumDeltaPocs`.
    pub(crate) fn num_delta_pocs(&self) -> u32 {
        u32::from(self.num_negative_pics) + u32::from(self.num_positive_pics)
    }

    /// The entries of (7-57) that this set adds to `NumPicTotalCurr`.
    pub(crate) fn num_used_by_curr_pic(&self) -> u32 {
        let count = |flags: &[bool], len: u8| {
            flags
                .iter()
                .take(usize::from(len))
                .filter(|used| **used)
                .count() as u32
        };
        count(&self.used_by_curr_pic_s0, self.num_negative_pics)
            + count(&self.used_by_curr_pic_s1, self.num_positive_pics)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_bits::BitWriter;
    use super::*;
    use crate::error::BitstreamError;

    fn parse(writer: &BitWriter, previous: &[ShortTermRefPicSet], in_slice: bool) -> ParseResult {
        let bytes = writer.bytes();
        ShortTermRefPicSet::parse(&mut BitReader::new(&bytes, "test"), previous, in_slice)
    }

    type ParseResult = Result<ShortTermRefPicSet, BitstreamError>;

    /// DeltaPocS0 -1 and -3, DeltaPocS1 +2, with the middle picture unused.
    fn source_set() -> ShortTermRefPicSet {
        let mut writer = BitWriter::new();
        writer.ue(2).ue(1);
        writer.ue(0).flag(true).ue(1).flag(false);
        writer.ue(1).flag(true);
        parse(&writer, &[], false).unwrap_or_else(|error| panic!("{error}"))
    }

    #[test]
    fn explicit_set_derives_the_delta_lists() {
        let set = source_set();
        assert_eq!(set.num_negative_pics, 2);
        assert_eq!(set.num_positive_pics, 1);
        assert_eq!(set.delta_poc_s0_minus1[..2], [0, 1]);
        assert_eq!(set.delta_poc_s0[..2], [-1, -3]);
        assert_eq!(set.delta_poc_s1[..1], [2]);
        assert_eq!(set.used_by_curr_pic_s0[..2], [true, false]);
        assert_eq!(set.num_delta_pocs(), 3);
        assert_eq!(set.num_used_by_curr_pic(), 2);
    }

    /// The source has `NumDeltaPocs` 3, so four flag pairs follow.
    #[test]
    fn predicted_set_follows_7_61_and_7_62() {
        let source = source_set();
        let mut writer = BitWriter::new();
        // Predicted, deltaRps -1.
        writer.flag(true).flag(true).ue(0);
        // j 0 and 1 kept and used. j 2 unused but kept. j 3 dropped.
        writer
            .flag(true)
            .flag(true)
            .flag(false)
            .flag(true)
            .flag(false)
            .flag(false);
        let set = parse(&writer, &[source], false).unwrap_or_else(|error| panic!("{error}"));
        assert!(set.inter_ref_pic_set_prediction_flag);
        assert_eq!(set.used_by_curr_pic_flag, 0b0011);
        assert_eq!(set.use_delta_flag, 0b0111);
        assert_eq!(set.num_prediction_flags, 4);
        // S1 entry +2 stays positive at +1. S0 entries move to -2 and -4.
        assert_eq!(set.num_negative_pics, 2);
        assert_eq!(set.delta_poc_s0[..2], [-2, -4]);
        assert_eq!(set.used_by_curr_pic_s0[..2], [true, true]);
        assert_eq!(set.num_positive_pics, 1);
        assert_eq!(set.delta_poc_s1[..1], [1]);
        assert_eq!(set.used_by_curr_pic_s1[..1], [false]);
    }

    /// A source entry whose new delta is 0 belongs to no list. Its flags do
    /// not count, whatever they hold.
    #[test]
    fn predicted_entry_at_delta_zero_is_dropped() {
        let source = source_set();
        let mut writer = BitWriter::new();
        // deltaRps +1 moves S0 entry -1 to 0.
        writer.flag(true).flag(false).ue(0);
        writer.flag(true).flag(true).flag(true).flag(true);
        let set = parse(&writer, &[source], false).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(set.num_negative_pics, 1);
        assert_eq!(set.delta_poc_s0[..1], [-2]);
        // New entries: +1 from deltaRps and +3 from S1.
        assert_eq!(set.num_positive_pics, 2);
        assert_eq!(set.delta_poc_s1[..2], [1, 3]);
        assert_eq!(set.num_delta_pocs(), 3);
    }

    #[test]
    fn slice_header_set_picks_its_source_with_delta_idx() {
        let empty = ShortTermRefPicSet::default();
        let source = source_set();
        let mut writer = BitWriter::new();
        // delta_idx_minus1 1 of two sets: the first set, `source`.
        writer.flag(true).ue(1).flag(true).ue(0);
        for _ in 0..4 {
            writer.flag(true);
        }
        let set = parse(&writer, &[source, empty], true).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(set.delta_idx_minus1, 1);
        assert_eq!(set.num_prediction_flags, 4);
        // delta_idx_minus1 2 has no source.
        let mut writer = BitWriter::new();
        writer.flag(true).ue(2);
        assert!(matches!(
            parse(&writer, &[source, empty], true),
            Err(BitstreamError::InvalidField {
                field: "delta_idx_minus1",
                value: 2,
                ..
            })
        ));
    }

    #[test]
    fn out_of_range_values_are_refused() {
        type Case<'a> = (&'a str, u64, fn(&mut BitWriter));
        let cases: [Case<'_>; 4] = [
            ("num_negative_pics", 17, |w| {
                w.ue(17);
            }),
            ("num_positive_pics", 2, |w| {
                w.ue(15).ue(2);
            }),
            ("delta_poc_s0_minus1", 32_768, |w| {
                w.ue(1).ue(0).ue(32_768);
            }),
            ("delta_poc_s1_minus1", 32_768, |w| {
                w.ue(0).ue(1).ue(32_768);
            }),
        ];
        for (name, expected, build) in cases {
            let mut writer = BitWriter::new();
            build(&mut writer);
            assert!(
                matches!(
                    parse(&writer, &[], false),
                    Err(BitstreamError::InvalidField { field, value, .. })
                        if field == name && value == expected
                ),
                "{name}"
            );
        }
        let mut writer = BitWriter::new();
        writer.flag(true).flag(false).ue(32_768);
        assert!(matches!(
            parse(&writer, &[source_set()], false),
            Err(BitstreamError::InvalidField {
                field: "abs_delta_rps_minus1",
                ..
            })
        ));
    }

    /// Sixteen source entries plus deltaRps make seventeen, one past capacity.
    #[test]
    fn predicted_overflow_is_an_error_never_a_panic() {
        let mut writer = BitWriter::new();
        writer.ue(16).ue(0);
        for _ in 0..16 {
            writer.ue(0).flag(true);
        }
        let source = parse(&writer, &[], false).unwrap_or_else(|error| panic!("{error}"));
        let mut writer = BitWriter::new();
        // deltaRps -1 keeps every entry negative.
        writer.flag(true).flag(true).ue(0);
        for _ in 0..17 {
            writer.flag(true);
        }
        assert!(matches!(
            parse(&writer, &[source], false),
            Err(BitstreamError::InvalidField {
                field: "NumNegativePics",
                value: 17,
                ..
            })
        ));
    }

    #[test]
    fn dpb_limit_applies_to_explicit_sets_only() {
        let set = source_set();
        assert_eq!(set.check_dpb_limit("test", 3), Ok(()));
        assert!(matches!(
            set.check_dpb_limit("test", 1),
            Err(BitstreamError::InvalidField {
                field: "num_negative_pics",
                ..
            })
        ));
        assert!(matches!(
            set.check_dpb_limit("test", 2),
            Err(BitstreamError::InvalidField {
                field: "num_positive_pics",
                ..
            })
        ));
        let predicted = ShortTermRefPicSet {
            inter_ref_pic_set_prediction_flag: true,
            ..set
        };
        assert_eq!(predicted.check_dpb_limit("test", 0), Ok(()));
    }

    #[test]
    fn truncation_at_every_byte_is_an_error() {
        let mut writer = BitWriter::new();
        writer.ue(16).ue(0);
        for _ in 0..16 {
            writer.ue(300).flag(true);
        }
        let bytes = writer.bytes();
        for end in 0..bytes.len() - 1 {
            let result =
                ShortTermRefPicSet::parse(&mut BitReader::new(&bytes[..end], "test"), &[], false);
            assert!(result.is_err(), "cut at byte {end}");
        }
    }
}
