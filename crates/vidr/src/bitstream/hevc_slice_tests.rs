//! Hand-built slice segment headers under hand-built parameter sets.

use super::*;
use crate::bitstream::hevc_pps::tests::{RangeSpec, Spec as PpsSpec, valid as valid_pps};
use crate::bitstream::hevc_sps_full::tests::{Spec as SpsSpec, valid as valid_sps};
use crate::bitstream::test_bits::{BitWriter, hevc_nal};

/// The short-term set a test writes.
#[derive(Clone, Copy)]
enum Rps {
    /// `short_term_ref_pic_set_sps_flag` 1 and the set index.
    Sps(u64),
    /// A set in the header: negative and positive counts, all used.
    Explicit(u32, u32),
}

/// The values a test varies. The defaults make a valid `TRAIL_R` P slice under
/// the default SPS and PPS.
#[derive(Clone)]
struct Spec {
    nal_type: u8,
    first: bool,
    no_output: bool,
    pps_id: u32,
    dependent: bool,
    address: u64,
    reserved: u64,
    slice_type: u32,
    pic_output: bool,
    colour_plane: u64,
    poc: u64,
    rps: Rps,
    /// SPS index and `delta_poc_msb_cycle_lt`.
    long_term_sps: Vec<(u64, Option<u32>)>,
    /// `poc_lsb_lt`, `used_by_curr_pic_lt_flag`, and `delta_poc_msb_cycle_lt`.
    long_term_pics: Vec<(u64, bool, Option<u32>)>,
    tmvp: bool,
    sao: (bool, bool),
    ref_override: Option<(u32, u32)>,
    list_entries: (Option<Vec<u64>>, Option<Vec<u64>>),
    mvd_l1_zero: bool,
    cabac_init: bool,
    from_l0: bool,
    collocated: u32,
    weights: Option<fn(&mut BitWriter)>,
    five_minus: u32,
    qp_delta: i64,
    chroma_offsets: (i64, i64),
    cu_chroma: bool,
    /// `deblocking_filter_override_flag` when enabled: disabled, beta, and tc.
    deblock: Option<(bool, i64, i64)>,
    loop_filter: bool,
    entries: Vec<u64>,
    /// Written directly, so a test can pass a count that disagrees.
    entry_count: Option<u32>,
    offset_len_minus1: u32,
    extension: Vec<u8>,
    extension_length: Option<u32>,
    data: Vec<u8>,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            nal_type: 1,
            first: true,
            no_output: false,
            pps_id: 0,
            dependent: false,
            address: 0,
            reserved: 0,
            slice_type: 1,
            pic_output: true,
            colour_plane: 0,
            poc: 9,
            rps: Rps::Sps(0),
            long_term_sps: Vec::new(),
            long_term_pics: Vec::new(),
            tmvp: true,
            sao: (true, true),
            ref_override: None,
            list_entries: (None, None),
            mvd_l1_zero: false,
            cabac_init: false,
            from_l0: true,
            collocated: 0,
            weights: None,
            five_minus: 1,
            qp_delta: 0,
            chroma_offsets: (0, 0),
            cu_chroma: false,
            deblock: None,
            loop_filter: true,
            entries: Vec::new(),
            entry_count: None,
            offset_len_minus1: 7,
            extension: Vec::new(),
            extension_length: None,
            data: vec![0x80; 4],
        }
    }
}

fn write_rps(w: &mut BitWriter, spec: &Spec, sps: &SeqParameterSet) {
    match spec.rps {
        Rps::Sps(index) => {
            w.flag(true);
            if sps.st_ref_pic_set.len() > 1 {
                w.u(ceil_log2(sps.st_ref_pic_set.len() as u32), index);
            }
        }
        Rps::Explicit(negative, positive) => {
            w.flag(false);
            if !sps.st_ref_pic_set.is_empty() {
                w.flag(false);
            }
            w.ue(negative).ue(positive);
            for _ in 0..negative + positive {
                w.ue(0).flag(true);
            }
        }
    }
}

fn write_long_term(w: &mut BitWriter, spec: &Spec, sps: &SeqParameterSet) {
    if sps.num_long_term_ref_pics_sps > 0 {
        w.ue(spec.long_term_sps.len() as u32);
    }
    w.ue(spec.long_term_pics.len() as u32);
    for (index, msb) in &spec.long_term_sps {
        if sps.num_long_term_ref_pics_sps > 1 {
            w.u(ceil_log2(sps.num_long_term_ref_pics_sps), *index);
        }
        w.flag(msb.is_some());
        if let Some(cycle) = msb {
            w.ue(*cycle);
        }
    }
    for (lsb, used, msb) in &spec.long_term_pics {
        w.u(sps.log2_max_pic_order_cnt_lsb_minus4 + 4, *lsb)
            .flag(*used)
            .flag(msb.is_some());
        if let Some(cycle) = msb {
            w.ue(*cycle);
        }
    }
}

fn write_inter(w: &mut BitWriter, spec: &Spec, sps: &SeqParameterSet, pps: &PicParameterSet) {
    let is_b = spec.slice_type == 0;
    w.flag(spec.ref_override.is_some());
    if let Some((l0, l1)) = spec.ref_override {
        w.ue(l0);
        if is_b {
            w.ue(l1);
        }
    }
    let used_set = match spec.rps {
        Rps::Sps(index) => sps
            .st_ref_pic_set
            .get(index as usize)
            .map_or(0, ShortTermRefPicSet::num_used_by_curr_pic),
        Rps::Explicit(negative, positive) => negative + positive,
    };
    let used_sps = spec
        .long_term_sps
        .iter()
        .filter(|(index, _)| {
            sps.used_by_curr_pic_lt_sps_flag
                .get(*index as usize)
                .is_some_and(|used| *used)
        })
        .count() as u32;
    let used_pics = spec
        .long_term_pics
        .iter()
        .filter(|(_, used, _)| *used)
        .count() as u32;
    let total = used_set + used_sps + used_pics;
    if pps.lists_modification_present_flag && total > 1 {
        let bits = ceil_log2(total);
        let mut write = |entries: &Option<Vec<u64>>| {
            w.flag(entries.is_some());
            for entry in entries.iter().flatten() {
                w.u(bits, *entry);
            }
        };
        write(&spec.list_entries.0);
        if is_b {
            write(&spec.list_entries.1);
        }
    }
    if is_b {
        w.flag(spec.mvd_l1_zero);
    }
    if pps.cabac_init_present_flag {
        w.flag(spec.cabac_init);
    }
    if spec.tmvp && sps.sps_temporal_mvp_enabled_flag {
        if is_b {
            w.flag(spec.from_l0);
        }
        let (l0, l1) = spec.ref_override.unwrap_or((
            pps.num_ref_idx_l0_default_active_minus1,
            pps.num_ref_idx_l1_default_active_minus1,
        ));
        let last = if spec.from_l0 || !is_b { l0 } else { l1 };
        if last > 0 {
            w.ue(spec.collocated);
        }
    }
    if let Some(weights) = spec.weights {
        weights(w);
    }
    w.ue(spec.five_minus);
}

fn write_independent(w: &mut BitWriter, spec: &Spec, sps: &SeqParameterSet, pps: &PicParameterSet) {
    if pps.num_extra_slice_header_bits > 0 {
        w.u(u32::from(pps.num_extra_slice_header_bits), spec.reserved);
    }
    w.ue(spec.slice_type);
    if pps.output_flag_present_flag {
        w.flag(spec.pic_output);
    }
    if sps.separate_colour_plane_flag {
        w.u(2, spec.colour_plane);
    }
    if !matches!(spec.nal_type, 19 | 20) {
        w.u(sps.log2_max_pic_order_cnt_lsb_minus4 + 4, spec.poc);
        write_rps(w, spec, sps);
        if sps.long_term_ref_pics_present_flag {
            write_long_term(w, spec, sps);
        }
        if sps.sps_temporal_mvp_enabled_flag {
            w.flag(spec.tmvp);
        }
    }
    let mut sao_any = false;
    if sps.sample_adaptive_offset_enabled_flag {
        w.flag(spec.sao.0);
        sao_any = spec.sao.0;
        if sps.chroma_array_type() != 0 {
            w.flag(spec.sao.1);
            sao_any |= spec.sao.1;
        }
    }
    if spec.slice_type != 2 {
        write_inter(w, spec, sps, pps);
    }
    w.se(spec.qp_delta);
    if pps.pps_slice_chroma_qp_offsets_present_flag {
        w.se(spec.chroma_offsets.0).se(spec.chroma_offsets.1);
    }
    if pps.pps_range_extension.chroma_qp_offset_list_enabled_flag {
        w.flag(spec.cu_chroma);
    }
    if pps.deblocking_filter_override_enabled_flag {
        w.flag(spec.deblock.is_some());
    }
    let mut disabled = pps.pps_deblocking_filter_disabled_flag;
    if let Some((off, beta, tc)) = spec.deblock {
        disabled = off;
        w.flag(off);
        if !off {
            w.se(beta).se(tc);
        }
    }
    if pps.pps_loop_filter_across_slices_enabled_flag && (sao_any || !disabled) {
        w.flag(spec.loop_filter);
    }
}

fn build(spec: &Spec, sps: &SeqParameterSet, pps: &PicParameterSet) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.flag(spec.first);
    if (16..=23).contains(&spec.nal_type) {
        w.flag(spec.no_output);
    }
    w.ue(spec.pps_id);
    if !spec.first {
        if pps.dependent_slice_segments_enabled_flag {
            w.flag(spec.dependent);
        }
        w.u(ceil_log2(sps.pic_size_in_ctbs_y()), spec.address);
    }
    if !spec.dependent {
        write_independent(&mut w, spec, sps, pps);
    }
    if pps.tiles_enabled_flag || pps.entropy_coding_sync_enabled_flag {
        let count = spec.entry_count.unwrap_or(spec.entries.len() as u32);
        w.ue(count);
        if count > 0 {
            w.ue(spec.offset_len_minus1);
            for offset in &spec.entries {
                w.u(spec.offset_len_minus1.min(31) + 1, *offset);
            }
        }
    }
    if pps.slice_segment_header_extension_present_flag {
        w.ue(spec.extension_length.unwrap_or(spec.extension.len() as u32));
        for byte in &spec.extension {
            w.u(8, u64::from(*byte));
        }
    }
    w.trailing();
    let mut rbsp = w.bytes();
    rbsp.extend_from_slice(&spec.data);
    hevc_nal(spec.nal_type, &rbsp)
}

fn parse(
    spec: &Spec,
    sps: &SeqParameterSet,
    pps: &PicParameterSet,
) -> Result<SliceSegmentHeader, BitstreamError> {
    SliceSegmentHeader::parse(&build(spec, sps, pps), pps, sps, None)
}

fn good(spec: &Spec, sps: &SeqParameterSet, pps: &PicParameterSet) -> SliceSegmentHeader {
    parse(spec, sps, pps).unwrap_or_else(|error| panic!("{error}"))
}

fn default_sets() -> (SeqParameterSet, PicParameterSet) {
    (
        valid_sps(&SpsSpec::default()),
        valid_pps(&PpsSpec::default()),
    )
}

#[test]
fn a_plain_p_slice_infers_absent_elements() {
    let (sps, pps) = default_sets();
    let header = good(&Spec::default(), &sps, &pps);
    assert_eq!(header.nal_unit_type, 1);
    assert_eq!(header.temporal_id, 0);
    assert!(header.first_slice_segment_in_pic_flag);
    assert!(!header.dependent_slice_segment_flag);
    assert_eq!(header.slice_type, SLICE_TYPE_P);
    assert!(header.pic_output_flag);
    assert_eq!(header.slice_pic_order_cnt_lsb, 9);
    assert!(header.short_term_ref_pic_set_sps_flag);
    assert_eq!(header.num_pic_total_curr, 3);
    assert_eq!(
        header
            .current_short_term_set(&sps)
            .map(ShortTermRefPicSet::num_delta_pocs),
        Some(3)
    );
    assert!(header.slice_temporal_mvp_enabled_flag);
    assert!(header.slice_sao_luma_flag && header.slice_sao_chroma_flag);
    assert_eq!(header.num_ref_idx_l0_active_minus1, 0);
    assert!(header.collocated_from_l0_flag);
    assert_eq!(header.max_num_merge_cand(), 4);
    assert!(header.pred_weight_table.is_none());
    assert!(!header.slice_deblocking_filter_disabled_flag);
    assert!(header.slice_loop_filter_across_slices_enabled_flag);
    assert!(header.entry_point_offset_minus1.is_empty());
}

#[test]
fn an_idr_slice_has_no_reference_sets() {
    let (sps, pps) = default_sets();
    let header = good(
        &Spec {
            nal_type: 19,
            no_output: true,
            slice_type: 2,
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    assert!(header.no_output_of_prior_pics_flag);
    assert_eq!(header.slice_type, SLICE_TYPE_I);
    assert_eq!(header.slice_pic_order_cnt_lsb, 0);
    assert!(header.current_short_term_set(&sps).is_none());
    assert!(!header.slice_temporal_mvp_enabled_flag);
    assert_eq!(header.num_pic_total_curr, 0);
}

#[test]
fn a_set_in_the_header_counts_its_bits() {
    let (sps, pps) = default_sets();
    let header = good(
        &Spec {
            rps: Rps::Explicit(1, 1),
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    assert!(!header.short_term_ref_pic_set_sps_flag);
    let set = header
        .current_short_term_set(&sps)
        .unwrap_or_else(|| panic!("the set is missing"));
    assert_eq!(set.delta_poc_s0[0], -1);
    assert_eq!(set.delta_poc_s1[0], 1);
    assert_eq!(header.num_pic_total_curr, 2);
    // Prediction flag 1, two 3-bit counts of 1 (`010`), and two 2-bit entries.
    assert_eq!(header.short_term_ref_pic_set_size, 1 + 3 + 3 + 2 + 2);
}

#[test]
fn the_set_index_takes_ceil_log2_bits() {
    let sps = valid_sps(&SpsSpec {
        sets: vec![(0, 0), (1, 0), (2, 0)],
        ..SpsSpec::default()
    });
    let pps = valid_pps(&PpsSpec::default());
    let header = good(
        &Spec {
            rps: Rps::Sps(2),
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    assert_eq!(header.short_term_ref_pic_set_idx, 2);
    assert_eq!(header.num_pic_total_curr, 2);
    assert!(matches!(
        parse(
            &Spec {
                rps: Rps::Sps(3),
                ..Spec::default()
            },
            &sps,
            &pps
        ),
        Err(BitstreamError::InvalidField {
            field: "short_term_ref_pic_set_idx",
            value: 3,
            ..
        })
    ));
}

#[test]
fn long_term_entries_come_from_the_sps_and_the_header() {
    let sps = valid_sps(&SpsSpec {
        dec_pic_buffering: [8, 8],
        long_term: vec![(17, true), (200, false), (5, true)],
        ..SpsSpec::default()
    });
    let pps = valid_pps(&PpsSpec::default());
    let header = good(
        &Spec {
            long_term_sps: vec![(2, Some(3)), (1, Some(4))],
            long_term_pics: vec![(99, true, Some(7)), (11, false, None)],
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    assert_eq!(header.num_long_term_sps, 2);
    assert_eq!(header.num_long_term_pics, 2);
    assert_eq!(header.lt_idx_sps[..2], [2, 1]);
    assert_eq!(header.poc_lsb_lt[..4], [5, 200, 99, 11]);
    assert_eq!(header.used_by_curr_pic_lt[..4], [true, false, true, false]);
    assert_eq!(header.delta_poc_msb_cycle_lt[..4], [3, 4, 7, 0]);
    // (7-52): the sum restarts at `num_long_term_sps`.
    assert_eq!(header.delta_poc_msb_cycle_lt_sum[..4], [3, 7, 7, 7]);
    // Three short-term pictures and two long-term pictures.
    assert_eq!(header.num_pic_total_curr, 5);
}

#[test]
fn list_modification_reads_entries_of_ceil_log2_bits() {
    let (sps, _) = default_sets();
    let pps = valid_pps(&PpsSpec {
        lists_modification: true,
        l0_default: 1,
        l1_default: 0,
        ..PpsSpec::default()
    });
    let header = good(
        &Spec {
            slice_type: 0,
            list_entries: (Some(vec![2, 1]), Some(vec![1])),
            collocated: 1,
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    assert!(header.ref_pic_list_modification_flag_l0);
    assert_eq!(header.list_entry_l0[..2], [2, 1]);
    assert!(header.ref_pic_list_modification_flag_l1);
    assert_eq!(header.list_entry_l1[0], 1);
    assert_eq!(header.collocated_ref_idx, 1);
}

#[test]
fn inter_elements_are_read_in_order() {
    let (sps, _) = default_sets();
    let pps = valid_pps(&PpsSpec {
        cabac_init: true,
        weighted: (false, true),
        ..PpsSpec::default()
    });
    let header = good(
        &Spec {
            slice_type: 0,
            ref_override: Some((2, 1)),
            mvd_l1_zero: true,
            cabac_init: true,
            from_l0: false,
            collocated: 1,
            weights: Some(|w| {
                // Denominators, then no weights in three plus two references.
                w.ue(3).se(1);
                for _ in 0..10 {
                    w.flag(false);
                }
            }),
            five_minus: 4,
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    assert!(header.num_ref_idx_active_override_flag);
    assert_eq!(header.num_ref_idx_l0_active_minus1, 2);
    assert_eq!(header.num_ref_idx_l1_active_minus1, 1);
    assert!(header.mvd_l1_zero_flag && header.cabac_init_flag);
    assert!(!header.collocated_from_l0_flag);
    assert_eq!(header.collocated_ref_idx, 1);
    assert_eq!(
        header
            .pred_weight_table
            .as_ref()
            .map(|table| table.chroma_log2_weight_denom()),
        Some(4)
    );
    assert_eq!(header.max_num_merge_cand(), 1);
    // A P slice under the same PPS holds no table: weighted_pred_flag is 0.
    assert!(
        good(&Spec::default(), &sps, &pps)
            .pred_weight_table
            .is_none()
    );
}

#[test]
fn qp_offsets_and_deblocking_inherit_from_the_pps() {
    let (sps, _) = default_sets();
    let pps = valid_pps(&PpsSpec {
        slice_chroma_offsets: true,
        deblock: Some((true, true, 0, 0)),
        loop_filter_across_slices: true,
        ..PpsSpec::default()
    });
    // No override: the disabled flag comes from the PPS. SAO keeps the flag
    // `slice_loop_filter_across_slices_enabled_flag` present.
    let header = good(
        &Spec {
            chroma_offsets: (5, -5),
            qp_delta: -3,
            loop_filter: false,
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    assert_eq!(header.slice_qp_delta, -3);
    assert_eq!(
        (header.slice_cb_qp_offset, header.slice_cr_qp_offset),
        (5, -5)
    );
    assert!(header.slice_deblocking_filter_disabled_flag);
    assert!(!header.slice_loop_filter_across_slices_enabled_flag);

    // Without SAO and with deblocking disabled, the flag is absent.
    let header = good(
        &Spec {
            sao: (false, false),
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    assert!(header.slice_loop_filter_across_slices_enabled_flag);

    // An override enables the filter with its own offsets.
    let header = good(
        &Spec {
            deblock: Some((false, 2, -2)),
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    assert!(header.deblocking_filter_override_flag);
    assert!(!header.slice_deblocking_filter_disabled_flag);
    assert_eq!(
        (header.slice_beta_offset_div2, header.slice_tc_offset_div2),
        (2, -2)
    );

    // PPS offsets apply without an override.
    let pps = valid_pps(&PpsSpec {
        deblock: Some((false, false, 3, -4)),
        ..PpsSpec::default()
    });
    let header = good(&Spec::default(), &sps, &pps);
    assert_eq!(
        (header.slice_beta_offset_div2, header.slice_tc_offset_div2),
        (3, -4)
    );
}

#[test]
fn cu_chroma_flag_follows_the_pps_range_extension() {
    let (sps, _) = default_sets();
    let pps = valid_pps(&PpsSpec {
        range_extension: Some(RangeSpec {
            chroma_list: Some((0, vec![(1, 1)])),
            ..RangeSpec::default()
        }),
        ..PpsSpec::default()
    });
    let header = good(
        &Spec {
            cu_chroma: true,
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    assert!(header.cu_chroma_qp_offset_enabled_flag);
}

#[test]
fn entry_points_and_extension_data_are_read() {
    let (sps, _) = default_sets();
    let pps = valid_pps(&PpsSpec {
        tiles: Some((1, 0, None)),
        wavefront: true,
        header_extension: true,
        ..PpsSpec::default()
    });
    let header = good(
        &Spec {
            entries: vec![0, u64::from(u32::MAX), 7],
            offset_len_minus1: 31,
            extension: vec![0xaa, 0xbb],
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    assert_eq!(header.offset_len_minus1, 31);
    assert_eq!(header.entry_point_offset_minus1, vec![0, u32::MAX, 7]);
    assert_eq!(header.slice_segment_header_extension_length, 2);
    // Tiles and wavefront: 2 columns * 4 CTB rows - 1 is the largest count.
    let full = good(
        &Spec {
            entries: vec![1; 7],
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    assert_eq!(full.entry_point_offset_minus1.len(), 7);
}

#[test]
fn the_data_offset_counts_emulation_prevention_bytes() {
    let (sps, _) = default_sets();
    let pps = valid_pps(&PpsSpec {
        header_extension: true,
        ..PpsSpec::default()
    });
    let spec = Spec {
        extension: vec![0, 0, 0, 0, 0, 0],
        data: vec![0x5a, 0xa5],
        ..Spec::default()
    };
    let nal = build(&spec, &sps, &pps);
    let header =
        SliceSegmentHeader::parse(&nal, &pps, &sps, None).unwrap_or_else(|error| panic!("{error}"));
    assert!(header.slice_data_nal_offset > header.slice_data_rbsp_offset);
    assert_eq!(nal[header.slice_data_nal_offset], 0x5a);
    assert_eq!(unescape(&nal)[header.slice_data_rbsp_offset], 0x5a);
}

#[test]
fn a_dependent_segment_copies_from_its_slice() {
    let (sps, _) = default_sets();
    let pps = valid_pps(&PpsSpec {
        dependent_slices: true,
        wavefront: true,
        ..PpsSpec::default()
    });
    let independent = good(
        &Spec {
            qp_delta: 4,
            entries: vec![9],
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    let spec = Spec {
        first: false,
        dependent: true,
        address: 14,
        entries: vec![3, 4],
        ..Spec::default()
    };
    let nal = build(&spec, &sps, &pps);
    let dependent = SliceSegmentHeader::parse(&nal, &pps, &sps, Some(&independent))
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(dependent.dependent_slice_segment_flag);
    assert!(!dependent.first_slice_segment_in_pic_flag);
    assert_eq!(dependent.slice_segment_address, 14);
    assert_eq!(dependent.slice_addr_rs, 0);
    assert_eq!(dependent.slice_qp_delta, 4);
    assert_eq!(dependent.slice_type, SLICE_TYPE_P);
    assert_eq!(dependent.slice_pic_order_cnt_lsb, 9);
    assert_eq!(dependent.entry_point_offset_minus1, vec![3, 4]);
    // A chain copies through a dependent segment too.
    let again = SliceSegmentHeader::parse(&nal, &pps, &sps, Some(&dependent))
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(again.slice_qp_delta, 4);
    // An independent later segment sets its own address.
    let later = good(
        &Spec {
            first: false,
            address: 20,
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    assert_eq!(later.slice_addr_rs, 20);
    // No slice to copy from.
    assert!(matches!(
        SliceSegmentHeader::parse(&nal, &pps, &sps, None),
        Err(BitstreamError::InvalidField {
            field: "dependent_slice_segment_flag",
            ..
        })
    ));
    // A slice under another PPS.
    let other = SliceSegmentHeader {
        slice_pic_parameter_set_id: 1,
        ..independent
    };
    assert!(matches!(
        SliceSegmentHeader::parse(&nal, &pps, &sps, Some(&other)),
        Err(BitstreamError::InvalidField {
            field: "slice_pic_parameter_set_id",
            ..
        })
    ));
}

#[test]
fn extra_bits_output_flag_and_colour_plane_are_read() {
    let sps = valid_sps(&SpsSpec {
        chroma_format_idc: 3,
        separate_colour_plane: true,
        ..SpsSpec::default()
    });
    let pps = valid_pps(&PpsSpec {
        extra_bits: 7,
        output_flag: true,
        ..PpsSpec::default()
    });
    let header = good(
        &Spec {
            reserved: 0b101_0011,
            pic_output: false,
            colour_plane: 2,
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    assert_eq!(header.slice_reserved_flag, 0b101_0011);
    assert!(!header.pic_output_flag);
    assert_eq!(header.colour_plane_id, 2);
    assert!(header.slice_sao_luma_flag && !header.slice_sao_chroma_flag);
    let plane = parse(
        &Spec {
            colour_plane: 3,
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    assert!(matches!(
        plane,
        Err(BitstreamError::InvalidField {
            field: "colour_plane_id",
            value: 3,
            ..
        })
    ));
}

#[test]
fn out_of_range_values_are_refused_under_the_default_sets() {
    type Change = fn(&mut Spec);
    let (sps, pps) = default_sets();
    let cases: [(&str, u64, Change); 12] = [
        ("slice_pic_parameter_set_id", 64, |s| s.pps_id = 64),
        ("slice_pic_parameter_set_id", 1, |s| s.pps_id = 1),
        ("slice_segment_address", 28, |s| {
            s.first = false;
            s.address = 28;
        }),
        ("slice_type", 3, |s| s.slice_type = 3),
        ("slice_type", 1, |s| s.nal_type = 19),
        ("NumPicTotalCurr", 0, |s| s.rps = Rps::Explicit(0, 0)),
        ("NumPicTotalCurr", 3, |s| {
            s.nal_type = 21;
            s.slice_type = 2;
        }),
        ("num_negative_pics", 6, |s| s.rps = Rps::Explicit(6, 0)),
        ("num_positive_pics", 5, |s| s.rps = Rps::Explicit(1, 5)),
        ("num_ref_idx_l0_active_minus1", 15, |s| {
            s.ref_override = Some((15, 0));
        }),
        ("collocated_ref_idx", 2, |s| {
            s.ref_override = Some((1, 0));
            s.collocated = 2;
        }),
        ("five_minus_max_num_merge_cand", 5, |s| s.five_minus = 5),
    ];
    for (name, expected, change) in cases {
        let mut spec = Spec::default();
        change(&mut spec);
        let result = parse(&spec, &sps, &pps);
        assert!(
            matches!(
                &result,
                Err(BitstreamError::InvalidField { field, value, .. })
                    if *field == name && *value == expected
            ),
            "{name} {expected}: {result:?}"
        );
    }
    let l1 = parse(
        &Spec {
            slice_type: 0,
            ref_override: Some((0, 15)),
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    assert!(matches!(
        l1,
        Err(BitstreamError::InvalidField {
            field: "num_ref_idx_l1_active_minus1",
            value: 15,
            ..
        })
    ));
    let signed: [(&str, i64, Change); 4] = [
        ("slice_qp_delta", 26, |s| s.qp_delta = 26),
        ("slice_qp_delta", -27, |s| s.qp_delta = -27),
        ("slice_beta_offset_div2", 7, |_| {}),
        ("slice_tc_offset_div2", -7, |_| {}),
    ];
    let override_pps = valid_pps(&PpsSpec {
        deblock: Some((true, false, 0, 0)),
        ..PpsSpec::default()
    });
    for (name, expected, change) in signed {
        let mut spec = Spec::default();
        change(&mut spec);
        match name {
            "slice_beta_offset_div2" => spec.deblock = Some((false, 7, 0)),
            "slice_tc_offset_div2" => spec.deblock = Some((false, 0, -7)),
            _ => {}
        }
        let result = parse(&spec, &sps, &override_pps);
        assert!(
            matches!(
                &result,
                Err(BitstreamError::InvalidSignedField { field, value, .. })
                    if *field == name && *value == expected
            ),
            "{name} {expected}: {result:?}"
        );
    }
}

#[test]
fn set_flag_with_no_sps_sets_is_refused() {
    let sps = valid_sps(&SpsSpec {
        sets: Vec::new(),
        ..SpsSpec::default()
    });
    let pps = valid_pps(&PpsSpec::default());
    assert!(matches!(
        parse(&Spec::default(), &sps, &pps),
        Err(BitstreamError::InvalidField {
            field: "short_term_ref_pic_set_sps_flag",
            ..
        })
    ));
}

#[test]
fn long_term_limits_are_refused() {
    let sps = valid_sps(&SpsSpec {
        long_term: vec![(1, true), (2, true)],
        ..SpsSpec::default()
    });
    let pps = valid_pps(&PpsSpec::default());
    // Three candidates asked of two.
    let result = parse(
        &Spec {
            long_term_sps: vec![(0, None); 3],
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    assert!(matches!(
        result,
        Err(BitstreamError::InvalidField {
            field: "num_long_term_sps",
            value: 3,
            ..
        })
    ));
    // Three short-term and two candidates leave no room in a DPB of 6.
    let result = parse(
        &Spec {
            long_term_sps: vec![(0, None), (1, None)],
            long_term_pics: vec![(1, true, None)],
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    assert!(matches!(
        result,
        Err(BitstreamError::InvalidField {
            field: "num_long_term_pics",
            value: 1,
            ..
        })
    ));
    // An index past the candidates.
    let three = valid_sps(&SpsSpec {
        dec_pic_buffering: [8, 8],
        long_term: vec![(1, true), (2, true), (3, true)],
        ..SpsSpec::default()
    });
    let result = parse(
        &Spec {
            long_term_sps: vec![(3, None)],
            ..Spec::default()
        },
        &three,
        &pps,
    );
    assert!(matches!(
        result,
        Err(BitstreamError::InvalidField {
            field: "lt_idx_sps",
            value: 3,
            ..
        })
    ));
    // A cycle past the limit: 2 ^ 24 with an 8-bit POC.
    let result = parse(
        &Spec {
            long_term_sps: vec![(0, Some((1 << 24) + 1))],
            ..Spec::default()
        },
        &three,
        &pps,
    );
    assert!(matches!(
        result,
        Err(BitstreamError::InvalidField {
            field: "delta_poc_msb_cycle_lt",
            ..
        })
    ));
}

#[test]
fn list_entry_past_the_total_is_refused() {
    let (sps, _) = default_sets();
    let pps = valid_pps(&PpsSpec {
        lists_modification: true,
        ..PpsSpec::default()
    });
    let result = parse(
        &Spec {
            list_entries: (Some(vec![3]), None),
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    assert!(matches!(
        result,
        Err(BitstreamError::InvalidField {
            field: "list_entry_l0",
            value: 3,
            ..
        })
    ));
}

#[test]
fn chroma_offsets_are_refused_alone_and_in_sum() {
    let (sps, _) = default_sets();
    let pps = valid_pps(&PpsSpec {
        slice_chroma_offsets: true,
        cb_offset: 6,
        ..PpsSpec::default()
    });
    let result = parse(
        &Spec {
            chroma_offsets: (7, 0),
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    assert!(matches!(
        result,
        Err(BitstreamError::InvalidSignedField {
            field: "slice_cb_qp_offset",
            value: 7,
            ..
        })
    ));
    let result = parse(
        &Spec {
            chroma_offsets: (0, 13),
            ..Spec::default()
        },
        &sps,
        &pps,
    );
    assert!(matches!(
        result,
        Err(BitstreamError::InvalidSignedField {
            field: "slice_cr_qp_offset",
            value: 13,
            ..
        })
    ));
}

#[test]
fn entry_point_and_extension_limits_are_refused() {
    let (sps, _) = default_sets();
    let wavefront = valid_pps(&PpsSpec {
        wavefront: true,
        ..PpsSpec::default()
    });
    let result = parse(
        &Spec {
            entry_count: Some(4),
            ..Spec::default()
        },
        &sps,
        &wavefront,
    );
    assert!(matches!(
        result,
        Err(BitstreamError::InvalidField {
            field: "num_entry_point_offsets",
            value: 4,
            ..
        })
    ));
    let tiles = valid_pps(&PpsSpec {
        tiles: Some((1, 0, None)),
        ..PpsSpec::default()
    });
    let result = parse(
        &Spec {
            entry_count: Some(2),
            ..Spec::default()
        },
        &sps,
        &tiles,
    );
    assert!(matches!(
        result,
        Err(BitstreamError::InvalidField {
            field: "num_entry_point_offsets",
            value: 2,
            ..
        })
    ));
    let result = parse(
        &Spec {
            entry_count: Some(1),
            offset_len_minus1: 32,
            ..Spec::default()
        },
        &sps,
        &wavefront,
    );
    assert!(matches!(
        result,
        Err(BitstreamError::InvalidField {
            field: "offset_len_minus1",
            value: 32,
            ..
        })
    ));
    let extension = valid_pps(&PpsSpec {
        header_extension: true,
        ..PpsSpec::default()
    });
    let result = parse(
        &Spec {
            extension_length: Some(257),
            ..Spec::default()
        },
        &sps,
        &extension,
    );
    assert!(matches!(
        result,
        Err(BitstreamError::InvalidField {
            field: "slice_segment_header_extension_length",
            value: 257,
            ..
        })
    ));
}

#[test]
fn entry_counts_that_exceed_the_data_are_a_truncation_before_any_allocation() {
    let (sps, _) = default_sets();
    let wavefront = valid_pps(&PpsSpec {
        wavefront: true,
        ..PpsSpec::default()
    });
    let result = parse(
        &Spec {
            entry_count: Some(3),
            offset_len_minus1: 31,
            data: Vec::new(),
            ..Spec::default()
        },
        &sps,
        &wavefront,
    );
    assert!(matches!(
        result,
        Err(BitstreamError::Truncated {
            field: "entry_point_offset_minus1",
            ..
        })
    ));
}

#[test]
fn byte_alignment_bits_are_checked() {
    let mut reader = BitReader::new(&[0b1000_0000], "test");
    assert_eq!(read_byte_alignment(&mut reader), Ok(()));
    assert_eq!(reader.position(), 8);
    let mut reader = BitReader::new(&[0b0000_0000], "test");
    assert!(matches!(
        read_byte_alignment(&mut reader),
        Err(BitstreamError::InvalidField {
            field: "alignment_bit_equal_to_one",
            ..
        })
    ));
    let mut reader = BitReader::new(&[0b1100_0000], "test");
    assert!(matches!(
        read_byte_alignment(&mut reader),
        Err(BitstreamError::InvalidField {
            field: "alignment_bit_equal_to_zero",
            ..
        })
    ));
}

#[test]
fn nal_header_values_are_refused() {
    let (sps, pps) = default_sets();
    let nal = build(&Spec::default(), &sps, &pps);
    let check = |mutate: fn(&mut Vec<u8>), name: &str, expected: u64| {
        let mut bad = nal.clone();
        mutate(&mut bad);
        let result = SliceSegmentHeader::parse(&bad, &pps, &sps, None);
        assert!(
            matches!(
                &result,
                Err(BitstreamError::InvalidField { field, value, .. })
                    if *field == name && *value == expected
            ),
            "{name}: {result:?}"
        );
    };
    check(|n| n[0] |= 0x80, "forbidden_zero_bit", 1);
    check(|n| n[0] = 10 << 1, "nal_unit_type", 10);
    check(|n| n[0] = 22 << 1, "nal_unit_type", 22);
    check(|n| n[0] = 33 << 1, "nal_unit_type", 33);
    check(|n| n[1] |= 0x08, "nuh_layer_id", 1);
    check(|n| n[1] &= 0xf8, "nuh_temporal_id_plus1", 0);
    check(|n| n[1] = (n[1] & 0xf8) | 2, "nuh_temporal_id_plus1", 2);
}

#[test]
fn pps_and_sps_are_checked_against_each_other() {
    let (sps, _) = default_sets();
    let pps = valid_pps(&PpsSpec {
        sps_id: 3,
        ..PpsSpec::default()
    });
    assert!(matches!(
        parse(&Spec::default(), &sps, &pps),
        Err(BitstreamError::InvalidField {
            field: "pps_seq_parameter_set_id",
            value: 3,
            ..
        })
    ));
}

#[test]
fn the_pps_id_is_read_ahead_of_the_parse() {
    let (sps, _) = default_sets();
    let pps = valid_pps(&PpsSpec {
        pps_id: 63,
        ..PpsSpec::default()
    });
    let spec = Spec {
        pps_id: 63,
        ..Spec::default()
    };
    let nal = build(&spec, &sps, &pps);
    assert_eq!(slice_pic_parameter_set_id(&nal), Ok(63));
    let idr = build(
        &Spec {
            nal_type: 19,
            slice_type: 2,
            ..spec
        },
        &sps,
        &pps,
    );
    assert_eq!(slice_pic_parameter_set_id(&idr), Ok(63));
    assert!(slice_pic_parameter_set_id(&[]).is_err());
    assert!(slice_pic_parameter_set_id(&nal[..2]).is_err());
    assert!(matches!(
        slice_pic_parameter_set_id(&[33 << 1, 1, 0]),
        Err(BitstreamError::InvalidField {
            field: "nal_unit_type",
            ..
        })
    ));
}

#[test]
fn truncation_at_every_byte_never_panics() {
    let sps = valid_sps(&SpsSpec {
        dec_pic_buffering: [8, 8],
        long_term: vec![(17, true), (200, false), (5, true)],
        ..SpsSpec::default()
    });
    let pps = valid_pps(&PpsSpec {
        dependent_slices: true,
        output_flag: true,
        extra_bits: 2,
        cabac_init: true,
        weighted: (true, true),
        slice_chroma_offsets: true,
        tiles: Some((1, 0, None)),
        wavefront: true,
        deblock: Some((true, false, 1, 1)),
        lists_modification: true,
        header_extension: true,
        ..PpsSpec::default()
    });
    let spec = Spec {
        first: false,
        address: 5,
        slice_type: 0,
        rps: Rps::Explicit(2, 1),
        long_term_sps: vec![(2, Some(3))],
        long_term_pics: vec![(9, true, Some(2))],
        ref_override: Some((1, 1)),
        list_entries: (Some(vec![2, 1]), Some(vec![0, 3])),
        from_l0: false,
        collocated: 1,
        weights: Some(|w| {
            w.ue(3).se(0);
            w.flag(true).flag(false).flag(false).flag(true);
            w.se(1).se(2);
            w.se(1).se(2).se(3).se(4);
            w.flag(true).flag(false).flag(false).flag(false);
            w.se(5).se(6);
        }),
        chroma_offsets: (1, -1),
        deblock: Some((false, 1, 1)),
        entries: vec![1, 2],
        extension: vec![1, 2, 3],
        data: Vec::new(),
        ..Spec::default()
    };
    let nal = build(&spec, &sps, &pps);
    assert!(SliceSegmentHeader::parse(&nal, &pps, &sps, None).is_ok());
    for end in 0..nal.len() {
        let result = SliceSegmentHeader::parse(&nal[..end], &pps, &sps, None);
        // The last byte can hold nothing but the alignment bits.
        if end + 1 < nal.len() {
            assert!(result.is_err(), "cut at byte {end}");
        }
    }
}
