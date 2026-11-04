//! The slice data loop on hand-written bins: address order, SAO syntax, subsets,
//! context storage, and `qPY_PREV` resets.

use crate::decode::cpu::hevc::cabac::context::{self as ctx, ContextTable};
use crate::decode::cpu::hevc::cabac::test_encoder::CabacEncoder;
use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::layout::{LayoutSizes, PicLayout};
use crate::decode::cpu::hevc::params::CodingParams;

use super::maps::ParseMaps;
use super::records::CtuRecords;
use super::sao::{SaoCtb, SaoKind, SaoPlane};
use super::slice_data::{EntropyCarry, SegmentInput, parse_slice_segment};

/// Bins with their own encoder, so a test can restart at each subset.
struct Writer {
    encoder: CabacEncoder,
    contexts: ContextTable,
}

impl Writer {
    fn new(params: &CodingParams) -> Self {
        Self {
            encoder: CabacEncoder::new(),
            contexts: ContextTable::new(params.init_type, params.slice_qp),
        }
    }

    fn ctx(&mut self, index: usize, bin: bool) {
        self.encoder
            .encode_decision(self.contexts.state_mut(index), bin);
    }

    /// One 16x16 intra coding unit with DC prediction and no residual. The CTB and
    /// the minimum coding block have the same size.
    fn empty_ctu(&mut self) {
        // part_mode 2Nx2N, prev_intra_luma_pred_flag, mpm_idx 1 selects DC with both
        // candidates at DC, intra_chroma_pred_mode 4, no chroma or luma flags.
        self.ctx(ctx::PART_MODE, true);
        self.ctx(ctx::PREV_INTRA_LUMA_PRED, true);
        self.encoder.encode_bypass(true);
        self.encoder.encode_bypass(false);
        self.ctx(ctx::INTRA_CHROMA_PRED_MODE, false);
        self.ctx(ctx::CBF_CHROMA, false);
        self.ctx(ctx::CBF_CHROMA, false);
        self.ctx(ctx::CBF_LUMA + 1, false);
    }

    /// Ends a subset: `end_of_slice_segment_flag` 0, `end_of_subset_one_bit` 1, the
    /// flush, alignment zeros, and a new engine.
    fn end_subset(&mut self) {
        self.encoder.encode_terminate(false);
        self.encoder.encode_terminate(true);
        self.encoder.align_zero();
        self.encoder.restart();
    }

    fn finish(mut self) -> Vec<u8> {
        self.encoder.encode_terminate(true);
        self.encoder.finish()
    }
}

/// A picture of `columns` by `rows` 16x16 CTBs with intra coding units of one size.
fn params(columns: u32, rows: u32) -> CodingParams {
    CodingParams {
        width: columns * 16,
        height: rows * 16,
        ctb_log2: 4,
        min_cb_log2: 4,
        ..CodingParams::for_tests()
    }
}

fn layout(params: &CodingParams, tile_columns: &[u32]) -> PicLayout {
    PicLayout::new(&LayoutSizes {
        width: params.width,
        height: params.height,
        ctb_log2: 4,
        min_tb_log2: 2,
        column_widths: tile_columns,
        row_heights: &[params.height / 16],
    })
    .unwrap()
}

struct Parsed {
    /// The first coding unit's location in each CTU.
    ctus: Vec<(u32, u32)>,
    maps: ParseMaps,
    carry: EntropyCarry,
}

fn parse(params: &CodingParams, layout: &PicLayout, data: &[u8]) -> Result<Parsed, HevcError> {
    let mut maps = ParseMaps::new(params, layout)?;
    let mut carry = EntropyCarry::default();
    let mut ctus = Vec::new();
    let input = SegmentInput {
        params,
        layout,
        data,
        start: 0,
        first_ctb_rs: 0,
        dependent: false,
    };
    parse_slice_segment(&input, &mut maps, &mut carry, |_, records: &CtuRecords| {
        let cu = records.cus[0];
        ctus.push((cu.x, cu.y));
        Ok(())
    })?;
    Ok(Parsed { ctus, maps, carry })
}

#[test]
fn ctus_run_in_raster_order_until_the_flag() {
    let params = params(2, 2);
    let layout = layout(&params, &[2]);
    let mut writer = Writer::new(&params);
    for _ in 0..3 {
        writer.empty_ctu();
        writer.encoder.encode_terminate(false);
    }
    writer.empty_ctu();
    let parsed = parse(&params, &layout, &writer.finish()).unwrap();
    assert_eq!(parsed.ctus, vec![(0, 0), (16, 0), (0, 16), (16, 16)]);
    assert_eq!(parsed.carry.decoded_ctbs, 4);
}

#[test]
fn a_slice_past_the_last_ctb_is_refused() {
    let params = params(1, 1);
    let layout = layout(&params, &[1]);
    let mut writer = Writer::new(&params);
    writer.empty_ctu();
    writer.encoder.encode_terminate(false);
    assert!(matches!(
        parse(&params, &layout, &writer.finish()),
        Err(HevcError::InvalidSyntax {
            element: "end_of_slice_segment_flag",
            ..
        })
    ));
}

/// With two tile columns of one CTB, the second CTU is the first of its tile: a new
/// subset and fresh contexts.
#[test]
fn a_new_tile_restarts_the_engine_and_the_contexts() {
    let params = CodingParams {
        tiles_enabled: true,
        ..params(2, 1)
    };
    let layout = layout(&params, &[1, 1]);
    let mut writer = Writer::new(&params);
    writer.empty_ctu();
    writer.end_subset();
    writer.contexts = ContextTable::new(params.init_type, params.slice_qp);
    writer.empty_ctu();
    let parsed = parse(&params, &layout, &writer.finish()).unwrap();
    assert_eq!(parsed.ctus, vec![(0, 0), (16, 0)]);
}

/// A 2 by 2 picture with wavefronts: CTB 2 starts a subset and takes the contexts
/// stored after CTB 1.
#[test]
fn a_wavefront_row_takes_the_contexts_of_the_second_ctb() {
    let params = CodingParams {
        entropy_coding_sync: true,
        ..params(2, 2)
    };
    let layout = layout(&params, &[2]);
    let mut writer = Writer::new(&params);
    writer.empty_ctu();
    writer.encoder.encode_terminate(false);
    writer.empty_ctu();
    let stored = writer.contexts.clone();
    writer.end_subset();
    writer.contexts = stored;
    writer.empty_ctu();
    writer.encoder.encode_terminate(false);
    writer.empty_ctu();
    let parsed = parse(&params, &layout, &writer.finish()).unwrap();
    assert_eq!(parsed.ctus.len(), 4);
}

/// One CTB per row: T lies outside the picture, so each row initializes again.
#[test]
fn a_one_ctb_wide_wavefront_initializes_each_row() {
    let params = CodingParams {
        entropy_coding_sync: true,
        ..params(1, 2)
    };
    let layout = layout(&params, &[1]);
    let mut writer = Writer::new(&params);
    writer.empty_ctu();
    writer.end_subset();
    writer.contexts = ContextTable::new(params.init_type, params.slice_qp);
    writer.empty_ctu();
    let parsed = parse(&params, &layout, &writer.finish()).unwrap();
    assert_eq!(parsed.ctus, vec![(0, 0), (0, 16)]);
}

#[test]
fn a_zero_end_of_subset_bit_is_refused() {
    let params = CodingParams {
        entropy_coding_sync: true,
        ..params(1, 2)
    };
    let layout = layout(&params, &[1]);
    let mut writer = Writer::new(&params);
    writer.empty_ctu();
    writer.encoder.encode_terminate(false);
    writer.encoder.encode_terminate(false);
    assert!(matches!(
        parse(&params, &layout, &writer.finish()),
        Err(HevcError::InvalidSyntax {
            element: "end_of_subset_one_bit",
            ..
        })
    ));
}

/// A dependent slice segment takes the contexts at the end of the segment before it,
/// and its coding units see the slice's QP.
#[test]
fn a_dependent_segment_continues_the_contexts() {
    let params = CodingParams {
        dependent_slice_segments: true,
        ..params(2, 1)
    };
    let layout = layout(&params, &[2]);
    let mut first = Writer::new(&params);
    first.empty_ctu();
    let stored = first.contexts.clone();
    let first_data = first.finish();
    let mut second = Writer::new(&params);
    second.contexts = stored;
    second.empty_ctu();
    let second_data = second.finish();
    let mut maps = ParseMaps::new(&params, &layout).unwrap();
    let mut carry = EntropyCarry::default();
    let mut input = SegmentInput {
        params: &params,
        layout: &layout,
        data: &first_data,
        start: 0,
        first_ctb_rs: 0,
        dependent: false,
    };
    parse_slice_segment(&input, &mut maps, &mut carry, |_, _| Ok(())).unwrap();
    input.data = &second_data;
    input.first_ctb_rs = 1;
    input.dependent = true;
    parse_slice_segment(&input, &mut maps, &mut carry, |_, _| Ok(())).unwrap();
    assert_eq!(carry.decoded_ctbs, 2);
    assert_eq!(maps.qp_y(16, 0), params.slice_qp);
}

#[test]
fn a_dependent_segment_with_nothing_before_it_is_refused() {
    let params = CodingParams {
        dependent_slice_segments: true,
        ..params(2, 1)
    };
    let layout = layout(&params, &[2]);
    let mut writer = Writer::new(&params);
    writer.empty_ctu();
    let data = writer.finish();
    let mut maps = ParseMaps::new(&params, &layout).unwrap();
    let mut carry = EntropyCarry::default();
    let input = SegmentInput {
        params: &params,
        layout: &layout,
        data: &data,
        start: 0,
        first_ctb_rs: 1,
        dependent: true,
    };
    assert!(matches!(
        parse_slice_segment(&input, &mut maps, &mut carry, |_, _| Ok(())),
        Err(HevcError::InvalidSyntax {
            element: "dependent_slice_segment_flag",
            ..
        })
    ));
}

#[test]
fn an_address_outside_the_picture_is_refused() {
    let params = params(1, 1);
    let layout = layout(&params, &[1]);
    let mut maps = ParseMaps::new(&params, &layout).unwrap();
    let mut carry = EntropyCarry::default();
    let input = SegmentInput {
        params: &params,
        layout: &layout,
        data: &[0, 0, 0, 0],
        start: 0,
        first_ctb_rs: 1,
        dependent: false,
    };
    assert!(matches!(
        parse_slice_segment(&input, &mut maps, &mut carry, |_, _| Ok(())),
        Err(HevcError::InvalidSyntax {
            element: "slice_segment_address",
            value: 1,
            ..
        })
    ));
}

/// Band offsets on luma, edge offsets on chroma, then a merge from the left.
#[test]
fn sao_syntax_reads_each_type_and_merges() {
    let params = CodingParams {
        sao_luma: true,
        sao_chroma: true,
        ..params(2, 1)
    };
    let layout = layout(&params, &[2]);
    let mut writer = Writer::new(&params);
    // CTB 0 has no merge flag. Luma: type 1 is bins 1, 0.
    writer.ctx(ctx::SAO_TYPE_IDX, true);
    writer.encoder.encode_bypass(false);
    // Offsets 2, 0, 7, 1 as TR with cMax 7 at 8 bits: 110, 0, 1111111, 10.
    for (value, count) in [(0b110, 3), (0, 1), (0b111_1111, 7), (0b10, 2)] {
        writer.encoder.encode_bypass_bits(value, count);
    }
    // Signs of the three offsets above 0: -, +, -. Then band position 30.
    writer.encoder.encode_bypass_bits(0b101, 3);
    writer.encoder.encode_bypass_bits(30, 5);
    // Chroma: type 2 is bins 1, 1. Cb offsets 1, 1, 1, 1, then class 2.
    writer.ctx(ctx::SAO_TYPE_IDX, true);
    writer.encoder.encode_bypass(true);
    for _ in 0..4 {
        writer.encoder.encode_bypass_bits(0b10, 2);
    }
    writer.encoder.encode_bypass_bits(2, 2);
    // Cr offsets 0, 3, 0, 0, with the same class.
    for (value, count) in [(0, 1), (0b1110, 4), (0, 1), (0, 1)] {
        writer.encoder.encode_bypass_bits(value, count);
    }
    writer.empty_ctu();
    writer.encoder.encode_terminate(false);
    // CTB 1 merges from the left.
    writer.ctx(ctx::SAO_MERGE, true);
    writer.empty_ctu();
    let parsed = parse(&params, &layout, &writer.finish()).unwrap();
    let expected = SaoCtb {
        planes: [
            SaoPlane {
                kind: SaoKind::Band(30),
                offsets: [0, -2, 0, 7, -1],
            },
            SaoPlane {
                kind: SaoKind::Edge(2),
                offsets: [0, 1, 1, -1, -1],
            },
            SaoPlane {
                kind: SaoKind::Edge(2),
                offsets: [0, 0, 3, 0, 0],
            },
        ],
    };
    assert_eq!(parsed.maps.sao(0), expected);
    assert_eq!(parsed.maps.sao(1), expected);
}

/// At 10 bits, `cMax` is 31. A disabled plane stays off.
#[test]
fn sao_offsets_reach_31_at_10_bits() {
    let params = CodingParams {
        sao_luma: true,
        bit_depth: 10,
        ..params(1, 1)
    };
    let layout = layout(&params, &[1]);
    let mut writer = Writer::new(&params);
    writer.ctx(ctx::SAO_TYPE_IDX, true);
    writer.encoder.encode_bypass(true);
    // Offsets 31, 0, 0, 0: 31 bins of 1 with no closing bin.
    writer.encoder.encode_bypass_bits(u32::MAX >> 1, 31);
    writer.encoder.encode_bypass_bits(0, 3);
    writer.encoder.encode_bypass_bits(1, 2);
    writer.empty_ctu();
    let parsed = parse(&params, &layout, &writer.finish()).unwrap();
    assert_eq!(
        parsed.maps.sao(0).planes,
        [
            SaoPlane {
                kind: SaoKind::Edge(1),
                offsets: [0, 31, 0, 0, 0],
            },
            SaoPlane::default(),
            SaoPlane::default(),
        ]
    );
}

/// The second row merges up. A one-CTB-wide picture has no left neighbour.
#[test]
fn sao_merge_up_reads_the_row_above() {
    let params = CodingParams {
        sao_luma: true,
        ..params(1, 2)
    };
    let layout = layout(&params, &[1]);
    let mut writer = Writer::new(&params);
    writer.ctx(ctx::SAO_TYPE_IDX, true);
    writer.encoder.encode_bypass(false);
    for _ in 0..4 {
        writer.encoder.encode_bypass_bits(0, 1);
    }
    writer.encoder.encode_bypass_bits(4, 5);
    writer.empty_ctu();
    writer.encoder.encode_terminate(false);
    writer.ctx(ctx::SAO_MERGE, true);
    writer.empty_ctu();
    let parsed = parse(&params, &layout, &writer.finish()).unwrap();
    assert_eq!(parsed.maps.sao(1).planes[0].kind, SaoKind::Band(4));
}
