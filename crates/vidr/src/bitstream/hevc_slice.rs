//! HEVC `slice_segment_header()` (ITU-T H.265 7.3.6) for `nuh_layer_id` 0.
//!
//! A slice segment header names its PPS. [`slice_pic_parameter_set_id`] reads
//! that name, so that the caller can find the active PPS and SPS before
//! [`SliceSegmentHeader::parse`]. A dependent slice segment copies every absent
//! element from the independent slice segment of its slice (7.4.7.1).

use super::bits::BitReader;
use super::hevc_pps::PicParameterSet;
use super::hevc_pred_weight::{MAX_REFS, PredWeightTable};
use super::hevc_rps::ShortTermRefPicSet;
use super::hevc_sps_full::SeqParameterSet;
use super::rbsp::{escaped_offset, unescape};
use crate::error::BitstreamError;

const STRUCTURE: &str = "HEVC slice segment header";
/// The most long-term entries: `sps_max_dec_pic_buffering_minus1` is at most 15.
const MAX_LONG_TERM: usize = 16;
/// `slice_type` of a B slice.
pub(crate) const SLICE_TYPE_B: u8 = 0;
/// `slice_type` of a P slice.
pub(crate) const SLICE_TYPE_P: u8 = 1;
/// `slice_type` of an I slice.
pub(crate) const SLICE_TYPE_I: u8 = 2;
/// Bytes after the NAL header that always hold `slice_pic_parameter_set_id`,
/// emulation prevention bytes included. The element ends within 15 bits.
const PPS_ID_PREFIX: usize = 8;

/// One `slice_segment_header()`, with absent elements as 7.4.7 infers them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SliceSegmentHeader {
    /// From the NAL unit header.
    pub(crate) nal_unit_type: u8,
    /// `TemporalId`, from the NAL unit header.
    pub(crate) temporal_id: u8,
    pub(crate) first_slice_segment_in_pic_flag: bool,
    pub(crate) no_output_of_prior_pics_flag: bool,
    pub(crate) slice_pic_parameter_set_id: u32,
    pub(crate) dependent_slice_segment_flag: bool,
    pub(crate) slice_segment_address: u32,
    /// `SliceAddrRs`.
    pub(crate) slice_addr_rs: u32,
    /// `slice_reserved_flag[ i ]` at bit `num_extra_slice_header_bits - 1 - i`.
    pub(crate) slice_reserved_flag: u8,
    pub(crate) slice_type: u8,
    pub(crate) pic_output_flag: bool,
    pub(crate) colour_plane_id: u8,
    pub(crate) slice_pic_order_cnt_lsb: u16,
    pub(crate) short_term_ref_pic_set_sps_flag: bool,
    pub(crate) short_term_ref_pic_set_idx: u32,
    /// The set in this header. `None` for an IDR picture, and with
    /// `short_term_ref_pic_set_sps_flag`.
    pub(crate) short_term_ref_pic_set: Option<ShortTermRefPicSet>,
    /// Bits the set in this header takes.
    pub(crate) short_term_ref_pic_set_size: u32,
    pub(crate) num_long_term_sps: u32,
    pub(crate) num_long_term_pics: u32,
    /// `lt_idx_sps[ i ]` below `num_long_term_sps`.
    pub(crate) lt_idx_sps: [u8; MAX_LONG_TERM],
    /// `PocLsbLt`: `poc_lsb_lt[ i ]`, or the SPS candidate of `lt_idx_sps[ i ]`.
    pub(crate) poc_lsb_lt: [u16; MAX_LONG_TERM],
    /// `UsedByCurrPicLt`: `used_by_curr_pic_lt_flag[ i ]`, or the SPS candidate's.
    pub(crate) used_by_curr_pic_lt: [bool; MAX_LONG_TERM],
    pub(crate) delta_poc_msb_present_flag: [bool; MAX_LONG_TERM],
    pub(crate) delta_poc_msb_cycle_lt: [u32; MAX_LONG_TERM],
    /// `DeltaPocMsbCycleLt` (7-52).
    pub(crate) delta_poc_msb_cycle_lt_sum: [u64; MAX_LONG_TERM],
    pub(crate) slice_temporal_mvp_enabled_flag: bool,
    pub(crate) slice_sao_luma_flag: bool,
    pub(crate) slice_sao_chroma_flag: bool,
    pub(crate) num_ref_idx_active_override_flag: bool,
    pub(crate) num_ref_idx_l0_active_minus1: u32,
    pub(crate) num_ref_idx_l1_active_minus1: u32,
    /// `NumPicTotalCurr` (7-57).
    pub(crate) num_pic_total_curr: u32,
    pub(crate) ref_pic_list_modification_flag_l0: bool,
    pub(crate) ref_pic_list_modification_flag_l1: bool,
    pub(crate) list_entry_l0: [u8; MAX_REFS],
    pub(crate) list_entry_l1: [u8; MAX_REFS],
    pub(crate) mvd_l1_zero_flag: bool,
    pub(crate) cabac_init_flag: bool,
    pub(crate) collocated_from_l0_flag: bool,
    pub(crate) collocated_ref_idx: u32,
    /// `None` when the syntax holds no table.
    pub(crate) pred_weight_table: Option<Box<PredWeightTable>>,
    pub(crate) five_minus_max_num_merge_cand: u32,
    pub(crate) slice_qp_delta: i32,
    pub(crate) slice_cb_qp_offset: i32,
    pub(crate) slice_cr_qp_offset: i32,
    pub(crate) cu_chroma_qp_offset_enabled_flag: bool,
    pub(crate) deblocking_filter_override_flag: bool,
    pub(crate) slice_deblocking_filter_disabled_flag: bool,
    pub(crate) slice_beta_offset_div2: i32,
    pub(crate) slice_tc_offset_div2: i32,
    pub(crate) slice_loop_filter_across_slices_enabled_flag: bool,
    pub(crate) offset_len_minus1: u32,
    /// One entry per `num_entry_point_offsets`.
    pub(crate) entry_point_offset_minus1: Vec<u32>,
    pub(crate) slice_segment_header_extension_length: u32,
    /// Offset of the slice data in the NAL unit without emulation prevention
    /// bytes, with the 2-byte NAL header counted.
    pub(crate) slice_data_rbsp_offset: usize,
    /// Offset of the slice data in the NAL unit as passed in, with emulation
    /// prevention bytes counted. Entry point offsets count from here (7.4.7.1).
    pub(crate) slice_data_nal_offset: usize,
}

fn invalid(field: &'static str, value: u64, allowed: &'static str) -> BitstreamError {
    BitstreamError::InvalidField {
        structure: STRUCTURE,
        field,
        value,
        allowed,
    }
}

fn invalid_signed(field: &'static str, value: i64, allowed: &'static str) -> BitstreamError {
    BitstreamError::InvalidSignedField {
        structure: STRUCTURE,
        field,
        value,
        allowed,
    }
}

/// `Ceil( Log2( value ) )`, and 0 for a value of 0 or 1.
const fn ceil_log2(value: u32) -> u32 {
    u32::BITS - value.saturating_sub(1).leading_zeros()
}

/// True for `nal_unit_type` values from `BLA_W_LP` to `RSV_IRAP_VCL23`.
const fn is_irap(nal_unit_type: u8) -> bool {
    matches!(nal_unit_type, 16..=23)
}

/// True for `IDR_W_RADL` and `IDR_N_LP`.
const fn is_idr(nal_unit_type: u8) -> bool {
    matches!(nal_unit_type, 19 | 20)
}

/// Reads the NAL unit header of `rbsp`: `nal_unit_type` and `TemporalId`.
fn read_nal_header(rbsp: &[u8]) -> Result<(u8, u8), BitstreamError> {
    let mut reader = BitReader::new(rbsp, STRUCTURE);
    if reader.bit("forbidden_zero_bit")? {
        return Err(reader.invalid("forbidden_zero_bit", 1, "0"));
    }
    let nal_unit_type = reader.u8(6, "nal_unit_type")?;
    // Types 10 to 15 and 22 to 31 are reserved, and a decoder drops them.
    if !matches!(nal_unit_type, 0..=9 | 16..=21) {
        return Err(reader.invalid(
            "nal_unit_type",
            nal_unit_type.into(),
            "a VCL type: 0 to 9 or 16 to 21",
        ));
    }
    let layer = reader.u8(6, "nuh_layer_id")?;
    if layer != 0 {
        return Err(reader.invalid("nuh_layer_id", layer.into(), "0"));
    }
    let temporal_plus1 = reader.u8(3, "nuh_temporal_id_plus1")?;
    if temporal_plus1 == 0 {
        return Err(reader.invalid("nuh_temporal_id_plus1", 0, "1 to 7"));
    }
    Ok((nal_unit_type, temporal_plus1 - 1))
}

/// Reads `slice_pic_parameter_set_id` of a slice segment NAL unit.
pub(crate) fn slice_pic_parameter_set_id(nal: &[u8]) -> Result<u32, BitstreamError> {
    let prefix = unescape(nal.get(..2 + PPS_ID_PREFIX).unwrap_or(nal));
    let (nal_unit_type, _) = read_nal_header(&prefix)?;
    let mut reader = BitReader::new(prefix.get(2..).unwrap_or_default(), STRUCTURE);
    reader.bit("first_slice_segment_in_pic_flag")?;
    if is_irap(nal_unit_type) {
        reader.bit("no_output_of_prior_pics_flag")?;
    }
    reader.ue_max("slice_pic_parameter_set_id", 63, "0 to 63")
}

impl SliceSegmentHeader {
    /// Reads a slice segment NAL unit, 2-byte header included, under the active
    /// `pps` and `sps`. `independent` is the header of the slice this segment
    /// belongs to. A dependent segment needs it, and others ignore it.
    pub(crate) fn parse(
        nal: &[u8],
        pps: &PicParameterSet,
        sps: &SeqParameterSet,
        independent: Option<&Self>,
    ) -> Result<Self, BitstreamError> {
        pps.check_with_sps(sps)?;
        let rbsp = unescape(nal);
        let (nal_unit_type, temporal_id) = read_nal_header(&rbsp)?;
        if temporal_id > sps.sps_max_sub_layers_minus1 {
            return Err(invalid(
                "nuh_temporal_id_plus1",
                u64::from(temporal_id) + 1,
                "at most sps_max_sub_layers_minus1 + 1",
            ));
        }
        let mut reader = BitReader::new(rbsp.get(2..).unwrap_or_default(), STRUCTURE);
        let reader = &mut reader;
        let first_slice_segment_in_pic_flag = reader.bit("first_slice_segment_in_pic_flag")?;
        let no_output_of_prior_pics_flag =
            is_irap(nal_unit_type) && reader.bit("no_output_of_prior_pics_flag")?;
        let slice_pic_parameter_set_id =
            reader.ue_max("slice_pic_parameter_set_id", 63, "0 to 63")?;
        if slice_pic_parameter_set_id != pps.pps_pic_parameter_set_id {
            return Err(invalid(
                "slice_pic_parameter_set_id",
                slice_pic_parameter_set_id.into(),
                "the pps_pic_parameter_set_id of the PPS passed in",
            ));
        }
        let mut dependent_slice_segment_flag = false;
        let mut slice_segment_address = 0;
        if !first_slice_segment_in_pic_flag {
            if pps.dependent_slice_segments_enabled_flag {
                dependent_slice_segment_flag = reader.bit("dependent_slice_segment_flag")?;
            }
            let size = sps.pic_size_in_ctbs_y();
            slice_segment_address = reader.bits(ceil_log2(size), "slice_segment_address")? as u32;
            if slice_segment_address >= size {
                return Err(invalid(
                    "slice_segment_address",
                    slice_segment_address.into(),
                    "0 to PicSizeInCtbsY - 1",
                ));
            }
        }

        let mut header = if dependent_slice_segment_flag {
            let source = independent.ok_or_else(|| {
                invalid(
                    "dependent_slice_segment_flag",
                    1,
                    "0 when no slice segment of this slice precedes it",
                )
            })?;
            if source.slice_pic_parameter_set_id != slice_pic_parameter_set_id {
                return Err(invalid(
                    "slice_pic_parameter_set_id",
                    slice_pic_parameter_set_id.into(),
                    "the value of the independent slice segment of this slice",
                ));
            }
            source.clone()
        } else {
            let mut header = Self {
                nal_unit_type,
                temporal_id,
                slice_addr_rs: slice_segment_address,
                ..Self::default()
            };
            header.read_independent(reader, pps, sps)?;
            header
        };
        header.first_slice_segment_in_pic_flag = first_slice_segment_in_pic_flag;
        header.no_output_of_prior_pics_flag = no_output_of_prior_pics_flag;
        header.slice_pic_parameter_set_id = slice_pic_parameter_set_id;
        header.dependent_slice_segment_flag = dependent_slice_segment_flag;
        header.slice_segment_address = slice_segment_address;

        header.read_entry_points(reader, pps, sps)?;
        header.slice_segment_header_extension_length = 0;
        if pps.slice_segment_header_extension_present_flag {
            header.slice_segment_header_extension_length =
                reader.ue_max("slice_segment_header_extension_length", 256, "0 to 256")?;
            reader.skip(
                u64::from(header.slice_segment_header_extension_length) * 8,
                "slice_segment_header_extension_data_byte",
            )?;
        }
        read_byte_alignment(reader)?;
        header.slice_data_rbsp_offset = 2 + (reader.position() / 8) as usize;
        header.slice_data_nal_offset = escaped_offset(nal, header.slice_data_rbsp_offset);
        Ok(header)
    }

    /// Reads the elements a dependent slice segment omits (7.3.6.1).
    fn read_independent(
        &mut self,
        reader: &mut BitReader<'_>,
        pps: &PicParameterSet,
        sps: &SeqParameterSet,
    ) -> Result<(), BitstreamError> {
        if pps.num_extra_slice_header_bits > 0 {
            self.slice_reserved_flag = reader.bits(
                u32::from(pps.num_extra_slice_header_bits),
                "slice_reserved_flag",
            )? as u8;
        }
        self.slice_type = reader.ue_max("slice_type", 2, "0 to 2")? as u8;
        let intra_only = is_irap(self.nal_unit_type)
            || sps.sps_max_dec_pic_buffering_minus1[usize::from(self.temporal_id)] == 0;
        if intra_only && self.slice_type != SLICE_TYPE_I {
            return Err(invalid(
                "slice_type",
                self.slice_type.into(),
                "2 in an IRAP picture, and with sps_max_dec_pic_buffering_minus1 0",
            ));
        }
        self.pic_output_flag = !pps.output_flag_present_flag || reader.bit("pic_output_flag")?;
        if sps.separate_colour_plane_flag {
            self.colour_plane_id = reader.u8(2, "colour_plane_id")?;
            if self.colour_plane_id > 2 {
                return Err(invalid(
                    "colour_plane_id",
                    self.colour_plane_id.into(),
                    "0 to 2",
                ));
            }
        }
        if !is_idr(self.nal_unit_type) {
            self.read_reference_sets(reader, sps)?;
        }
        if sps.sample_adaptive_offset_enabled_flag {
            self.slice_sao_luma_flag = reader.bit("slice_sao_luma_flag")?;
            if sps.chroma_array_type() != 0 {
                self.slice_sao_chroma_flag = reader.bit("slice_sao_chroma_flag")?;
            }
        }
        self.collocated_from_l0_flag = true;
        if self.slice_type != SLICE_TYPE_I {
            self.read_inter_elements(reader, pps, sps)?;
        }
        self.read_qp_and_filter(reader, pps, sps)
    }

    /// Reads the POC, short-term set, long-term entries, and temporal MVP flag.
    fn read_reference_sets(
        &mut self,
        reader: &mut BitReader<'_>,
        sps: &SeqParameterSet,
    ) -> Result<(), BitstreamError> {
        let lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 + 4;
        self.slice_pic_order_cnt_lsb = reader.bits(lsb_bits, "slice_pic_order_cnt_lsb")? as u16;
        self.short_term_ref_pic_set_sps_flag = reader.bit("short_term_ref_pic_set_sps_flag")?;
        let sets = sps.st_ref_pic_set.len() as u32;
        let top = usize::from(sps.sps_max_sub_layers_minus1);
        let set = if self.short_term_ref_pic_set_sps_flag {
            if sets == 0 {
                return Err(invalid(
                    "short_term_ref_pic_set_sps_flag",
                    1,
                    "0 when num_short_term_ref_pic_sets is 0",
                ));
            }
            if sets > 1 {
                self.short_term_ref_pic_set_idx =
                    reader.bits(ceil_log2(sets), "short_term_ref_pic_set_idx")? as u32;
            }
            *sps.st_ref_pic_set
                .get(self.short_term_ref_pic_set_idx as usize)
                .ok_or_else(|| {
                    invalid(
                        "short_term_ref_pic_set_idx",
                        self.short_term_ref_pic_set_idx.into(),
                        "0 to num_short_term_ref_pic_sets - 1",
                    )
                })?
        } else {
            let start = reader.position();
            let set = ShortTermRefPicSet::parse(reader, &sps.st_ref_pic_set, true)?;
            set.check_dpb_limit(STRUCTURE, sps.sps_max_dec_pic_buffering_minus1[top])?;
            self.short_term_ref_pic_set_size = (reader.position() - start) as u32;
            self.short_term_ref_pic_set = Some(set);
            set
        };
        if sps.long_term_ref_pics_present_flag {
            self.read_long_term(reader, sps, &set)?;
        }
        if sps.sps_temporal_mvp_enabled_flag {
            self.slice_temporal_mvp_enabled_flag = reader.bit("slice_temporal_mvp_enabled_flag")?;
        }
        // (7-57), without the term for `pps_curr_pic_ref_enabled_flag`.
        let total_lt = (self.num_long_term_sps + self.num_long_term_pics) as usize;
        let used_lt = self
            .used_by_curr_pic_lt
            .iter()
            .take(total_lt)
            .filter(|used| **used)
            .count() as u32;
        self.num_pic_total_curr = set.num_used_by_curr_pic() + used_lt;
        // 8.3.2: a BLA or CRA picture uses no reference pictures.
        if is_irap(self.nal_unit_type) && self.num_pic_total_curr != 0 {
            return Err(invalid(
                "NumPicTotalCurr",
                self.num_pic_total_curr.into(),
                "0 in a BLA or CRA picture",
            ));
        }
        Ok(())
    }

    fn read_long_term(
        &mut self,
        reader: &mut BitReader<'_>,
        sps: &SeqParameterSet,
        set: &ShortTermRefPicSet,
    ) -> Result<(), BitstreamError> {
        let candidates = sps.num_long_term_ref_pics_sps;
        if candidates > 0 {
            self.num_long_term_sps = reader.ue_max(
                "num_long_term_sps",
                candidates,
                "0 to num_long_term_ref_pics_sps",
            )?;
        }
        self.num_long_term_pics = reader.ue("num_long_term_pics")?;
        // 7.4.7.1, for layer 0 without a second copy of the current picture.
        let limit = i64::from(sps.sps_max_dec_pic_buffering_minus1[usize::from(self.temporal_id)])
            - i64::from(set.num_delta_pocs())
            - i64::from(self.num_long_term_sps);
        if i64::from(self.num_long_term_pics) > limit {
            return Err(invalid(
                "num_long_term_pics",
                self.num_long_term_pics.into(),
                "at most sps_max_dec_pic_buffering_minus1[ TemporalId ] minus NumNegativePics, \
                 NumPositivePics, and num_long_term_sps",
            ));
        }
        let total = (self.num_long_term_sps + self.num_long_term_pics) as usize;
        let index_bits = ceil_log2(candidates);
        let lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 + 4;
        let cycle_limit = 1u32 << (28 - sps.log2_max_pic_order_cnt_lsb_minus4);
        let from_sps = self.num_long_term_sps as usize;
        for index in 0..total {
            if index < from_sps {
                let candidate = if candidates > 1 {
                    reader.bits(index_bits, "lt_idx_sps")? as usize
                } else {
                    0
                };
                if candidate >= candidates as usize {
                    return Err(invalid(
                        "lt_idx_sps",
                        candidate as u64,
                        "0 to num_long_term_ref_pics_sps - 1",
                    ));
                }
                self.lt_idx_sps[index] = candidate as u8;
                self.poc_lsb_lt[index] = sps.lt_ref_pic_poc_lsb_sps[candidate];
                self.used_by_curr_pic_lt[index] = sps.used_by_curr_pic_lt_sps_flag[candidate];
            } else {
                self.poc_lsb_lt[index] = reader.bits(lsb_bits, "poc_lsb_lt")? as u16;
                self.used_by_curr_pic_lt[index] = reader.bit("used_by_curr_pic_lt_flag")?;
            }
            self.delta_poc_msb_present_flag[index] = reader.bit("delta_poc_msb_present_flag")?;
            if self.delta_poc_msb_present_flag[index] {
                self.delta_poc_msb_cycle_lt[index] = reader.ue_max(
                    "delta_poc_msb_cycle_lt",
                    cycle_limit,
                    "0 to 2 ^ ( 28 - log2_max_pic_order_cnt_lsb_minus4 )",
                )?;
            }
            // (7-52).
            let previous = if index == 0 || index == from_sps {
                0
            } else {
                self.delta_poc_msb_cycle_lt_sum[index - 1]
            };
            self.delta_poc_msb_cycle_lt_sum[index] =
                previous + u64::from(self.delta_poc_msb_cycle_lt[index]);
        }
        Ok(())
    }

    /// Reads from `num_ref_idx_active_override_flag` to
    /// `five_minus_max_num_merge_cand`.
    fn read_inter_elements(
        &mut self,
        reader: &mut BitReader<'_>,
        pps: &PicParameterSet,
        sps: &SeqParameterSet,
    ) -> Result<(), BitstreamError> {
        // 8.3.2: a P or B slice needs a reference picture.
        if self.num_pic_total_curr == 0 {
            return Err(invalid(
                "NumPicTotalCurr",
                0,
                "above 0 in a picture with a P or B slice",
            ));
        }
        let is_b = self.slice_type == SLICE_TYPE_B;
        self.num_ref_idx_active_override_flag = reader.bit("num_ref_idx_active_override_flag")?;
        self.num_ref_idx_l0_active_minus1 = pps.num_ref_idx_l0_default_active_minus1;
        self.num_ref_idx_l1_active_minus1 = pps.num_ref_idx_l1_default_active_minus1;
        if self.num_ref_idx_active_override_flag {
            self.num_ref_idx_l0_active_minus1 =
                reader.ue_max("num_ref_idx_l0_active_minus1", 14, "0 to 14")?;
            if is_b {
                self.num_ref_idx_l1_active_minus1 =
                    reader.ue_max("num_ref_idx_l1_active_minus1", 14, "0 to 14")?;
            }
        }
        if pps.lists_modification_present_flag && self.num_pic_total_curr > 1 {
            self.read_list_modification(reader, is_b)?;
        }
        if is_b {
            self.mvd_l1_zero_flag = reader.bit("mvd_l1_zero_flag")?;
        }
        if pps.cabac_init_present_flag {
            self.cabac_init_flag = reader.bit("cabac_init_flag")?;
        }
        if self.slice_temporal_mvp_enabled_flag {
            if is_b {
                self.collocated_from_l0_flag = reader.bit("collocated_from_l0_flag")?;
            }
            let last = if self.collocated_from_l0_flag {
                self.num_ref_idx_l0_active_minus1
            } else {
                self.num_ref_idx_l1_active_minus1
            };
            if last > 0 {
                self.collocated_ref_idx = reader.ue_max(
                    "collocated_ref_idx",
                    last,
                    "0 to the num_ref_idx_active_minus1 of the collocated list",
                )?;
            }
        }
        let weighted = (pps.weighted_pred_flag && self.slice_type == SLICE_TYPE_P)
            || (pps.weighted_bipred_flag && is_b);
        if weighted {
            let l0 = self.num_ref_idx_l0_active_minus1 as usize + 1;
            let l1 = is_b.then_some(self.num_ref_idx_l1_active_minus1 as usize + 1);
            self.pred_weight_table = Some(Box::new(PredWeightTable::parse(reader, sps, l0, l1)?));
        }
        self.five_minus_max_num_merge_cand =
            reader.ue_max("five_minus_max_num_merge_cand", 4, "0 to 4")?;
        Ok(())
    }

    /// Reads `ref_pic_lists_modification()` (7.3.6.2).
    fn read_list_modification(
        &mut self,
        reader: &mut BitReader<'_>,
        is_b: bool,
    ) -> Result<(), BitstreamError> {
        let total = self.num_pic_total_curr;
        self.ref_pic_list_modification_flag_l0 = reader.bit("ref_pic_list_modification_flag_l0")?;
        if self.ref_pic_list_modification_flag_l0 {
            read_list_entries(
                reader,
                &mut self.list_entry_l0,
                self.num_ref_idx_l0_active_minus1,
                total,
                "list_entry_l0",
            )?;
        }
        if is_b {
            self.ref_pic_list_modification_flag_l1 =
                reader.bit("ref_pic_list_modification_flag_l1")?;
            if self.ref_pic_list_modification_flag_l1 {
                read_list_entries(
                    reader,
                    &mut self.list_entry_l1,
                    self.num_ref_idx_l1_active_minus1,
                    total,
                    "list_entry_l1",
                )?;
            }
        }
        Ok(())
    }

    /// Reads from `slice_qp_delta` to
    /// `slice_loop_filter_across_slices_enabled_flag`.
    fn read_qp_and_filter(
        &mut self,
        reader: &mut BitReader<'_>,
        pps: &PicParameterSet,
        sps: &SeqParameterSet,
    ) -> Result<(), BitstreamError> {
        let qp_delta = reader.se("slice_qp_delta")?;
        // SliceQpY (7-54) stays in -QpBdOffsetY to 51.
        let slice_qp = 26 + i64::from(pps.init_qp_minus26) + qp_delta;
        if !(-i64::from(sps.qp_bd_offset_y())..=51).contains(&slice_qp) {
            return Err(invalid_signed(
                "slice_qp_delta",
                qp_delta,
                "a value that gives SliceQpY from -QpBdOffsetY to 51",
            ));
        }
        self.slice_qp_delta = qp_delta as i32;
        if pps.pps_slice_chroma_qp_offsets_present_flag {
            self.slice_cb_qp_offset =
                read_chroma_offset(reader, "slice_cb_qp_offset", pps.pps_cb_qp_offset)?;
            self.slice_cr_qp_offset =
                read_chroma_offset(reader, "slice_cr_qp_offset", pps.pps_cr_qp_offset)?;
        }
        if pps.pps_range_extension.chroma_qp_offset_list_enabled_flag {
            self.cu_chroma_qp_offset_enabled_flag =
                reader.bit("cu_chroma_qp_offset_enabled_flag")?;
        }
        if pps.deblocking_filter_override_enabled_flag {
            self.deblocking_filter_override_flag = reader.bit("deblocking_filter_override_flag")?;
        }
        self.slice_deblocking_filter_disabled_flag = pps.pps_deblocking_filter_disabled_flag;
        self.slice_beta_offset_div2 = pps.pps_beta_offset_div2;
        self.slice_tc_offset_div2 = pps.pps_tc_offset_div2;
        if self.deblocking_filter_override_flag {
            self.slice_deblocking_filter_disabled_flag =
                reader.bit("slice_deblocking_filter_disabled_flag")?;
            if !self.slice_deblocking_filter_disabled_flag {
                self.slice_beta_offset_div2 =
                    reader.se_range("slice_beta_offset_div2", -6, 6, "-6 to 6")? as i32;
                self.slice_tc_offset_div2 =
                    reader.se_range("slice_tc_offset_div2", -6, 6, "-6 to 6")? as i32;
            }
        }
        self.slice_loop_filter_across_slices_enabled_flag =
            pps.pps_loop_filter_across_slices_enabled_flag;
        if pps.pps_loop_filter_across_slices_enabled_flag
            && (self.slice_sao_luma_flag
                || self.slice_sao_chroma_flag
                || !self.slice_deblocking_filter_disabled_flag)
        {
            self.slice_loop_filter_across_slices_enabled_flag =
                reader.bit("slice_loop_filter_across_slices_enabled_flag")?;
        }
        Ok(())
    }

    /// Reads `num_entry_point_offsets` and the offsets.
    fn read_entry_points(
        &mut self,
        reader: &mut BitReader<'_>,
        pps: &PicParameterSet,
        sps: &SeqParameterSet,
    ) -> Result<(), BitstreamError> {
        self.offset_len_minus1 = 0;
        self.entry_point_offset_minus1 = Vec::new();
        if !pps.tiles_enabled_flag && !pps.entropy_coding_sync_enabled_flag {
            return Ok(());
        }
        let columns = pps.num_tile_columns_minus1 + 1;
        let rows = pps.num_tile_rows_minus1 + 1;
        let height = sps.pic_height_in_ctbs_y();
        // 7.4.7.1.
        let (limit, allowed) = match (pps.tiles_enabled_flag, pps.entropy_coding_sync_enabled_flag)
        {
            (false, _) => (height - 1, "0 to PicHeightInCtbsY - 1"),
            (true, false) => (columns * rows - 1, "0 to the tile count minus 1"),
            (true, true) => (
                columns * height - 1,
                "0 to the tile column count * PicHeightInCtbsY - 1",
            ),
        };
        let count = reader.ue_max("num_entry_point_offsets", limit, allowed)?;
        if count == 0 {
            return Ok(());
        }
        self.offset_len_minus1 = reader.ue_max("offset_len_minus1", 31, "0 to 31")?;
        let bits = self.offset_len_minus1 + 1;
        // The count is bounded above, and the data must hold every offset.
        if u64::from(count) * u64::from(bits) > reader.bits_left() {
            return Err(reader.truncated("entry_point_offset_minus1"));
        }
        let mut offsets = Vec::with_capacity(count as usize);
        for _ in 0..count {
            offsets.push(reader.bits(bits, "entry_point_offset_minus1")? as u32);
        }
        self.entry_point_offset_minus1 = offsets;
        Ok(())
    }

    /// The short-term set of this picture: the set in this header, or the SPS
    /// set it names. `None` for an IDR picture.
    pub(crate) fn current_short_term_set<'a>(
        &'a self,
        sps: &'a SeqParameterSet,
    ) -> Option<&'a ShortTermRefPicSet> {
        if is_idr(self.nal_unit_type) {
            None
        } else if self.short_term_ref_pic_set_sps_flag {
            sps.st_ref_pic_set
                .get(self.short_term_ref_pic_set_idx as usize)
        } else {
            self.short_term_ref_pic_set.as_ref()
        }
    }

    /// `MaxNumMergeCand` (7-53).
    pub(crate) const fn max_num_merge_cand(&self) -> u32 {
        5 - self.five_minus_max_num_merge_cand
    }
}

/// Reads the entries of one list, `0..=last`, each below `total`.
fn read_list_entries(
    reader: &mut BitReader<'_>,
    entries: &mut [u8; MAX_REFS],
    last: u32,
    total: u32,
    name: &'static str,
) -> Result<(), BitstreamError> {
    let bits = ceil_log2(total);
    for entry in entries.iter_mut().take(last as usize + 1) {
        let value = reader.bits(bits, name)? as u32;
        if value >= total {
            return Err(invalid(name, value.into(), "0 to NumPicTotalCurr - 1"));
        }
        *entry = value as u8;
    }
    Ok(())
}

/// Reads a slice chroma QP offset. It and its sum with the PPS offset stay in
/// -12 to 12.
fn read_chroma_offset(
    reader: &mut BitReader<'_>,
    name: &'static str,
    pps_offset: i32,
) -> Result<i32, BitstreamError> {
    let value = reader.se_range(name, -12, 12, "-12 to 12")?;
    if !(-12..=12).contains(&(value + i64::from(pps_offset))) {
        return Err(invalid_signed(
            name,
            value,
            "a value that keeps the sum with the PPS offset in -12 to 12",
        ));
    }
    Ok(value as i32)
}

/// Reads `byte_alignment()` (7.3.2.12).
fn read_byte_alignment(reader: &mut BitReader<'_>) -> Result<(), BitstreamError> {
    if !reader.bit("alignment_bit_equal_to_one")? {
        return Err(reader.invalid("alignment_bit_equal_to_one", 0, "1"));
    }
    while !reader.position().is_multiple_of(8) {
        if reader.bit("alignment_bit_equal_to_zero")? {
            return Err(reader.invalid("alignment_bit_equal_to_zero", 1, "0"));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "hevc_slice_tests.rs"]
mod tests;
