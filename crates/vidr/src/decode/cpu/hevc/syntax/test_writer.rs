//! A bin writer and a parse harness for the syntax tests.
//!
//! A test states each bin and its context by hand, from the syntax tables and
//! Table 9-48. The harness then checks that the parser spent the bins through the
//! same contexts: the final context tables must match, and a closing bypass
//! pattern must read back.

use crate::decode::cpu::hevc::cabac::context::ContextTable;
use crate::decode::cpu::hevc::cabac::test_encoder::CabacEncoder;
use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::layout::{LayoutSizes, PicLayout};
use crate::decode::cpu::hevc::params::CodingParams;

use super::maps::ParseMaps;
use super::parser::CtuParser;
use super::records::CtuRecords;

/// The bypass bins after the last syntax element of a test.
const SIGNATURE: u32 = 0xa5c3_0f96;

/// Writes bins of one slice segment.
pub(super) struct SyntaxWriter {
    encoder: CabacEncoder,
    contexts: ContextTable,
}

impl SyntaxWriter {
    pub(super) fn new(params: &CodingParams) -> Self {
        Self {
            encoder: CabacEncoder::new(),
            contexts: ContextTable::new(params.init_type, params.slice_qp),
        }
    }

    /// One bin on context `index`.
    pub(super) fn ctx(&mut self, index: usize, bin: bool) {
        self.encoder
            .encode_decision(self.contexts.state_mut(index), bin);
    }

    /// One bypass bin.
    pub(super) fn bypass(&mut self, bin: bool) {
        self.encoder.encode_bypass(bin);
    }

    /// A fixed-length value in bypass bins.
    pub(super) fn bits(&mut self, value: u32, count: u32) {
        self.encoder.encode_bypass_bits(value, count);
    }

    /// One bin before termination.
    pub(super) fn terminate(&mut self, bin: bool) {
        self.encoder.encode_terminate(bin);
    }

    /// Raw bits, for PCM samples.
    pub(super) fn raw(&mut self, value: u32, count: u32) {
        self.encoder.write_raw_bits(value, count);
    }

    /// `pcm_alignment_zero_bit` up to a byte boundary.
    pub(super) fn align(&mut self) {
        self.encoder.align_zero();
    }

    /// The engine restart after PCM samples.
    pub(super) fn restart(&mut self) {
        self.encoder.restart();
    }

    /// An EGk value in bypass bins (9-13).
    pub(super) fn exp_golomb(&mut self, value: u32, order: u32) {
        let mut abs_v = value;
        let mut k = order;
        while abs_v >= (1 << k) {
            self.bypass(true);
            abs_v -= 1 << k;
            k += 1;
        }
        self.bypass(false);
        self.bits(abs_v, k);
    }

    /// The bytes and final context table, after the closing pattern.
    pub(super) fn finish(mut self) -> (Vec<u8>, ContextTable) {
        self.bits(SIGNATURE, 32);
        self.encoder.encode_terminate(true);
        (self.encoder.finish(), self.contexts)
    }

    /// The bytes as far as written, with no closing pattern.
    pub(super) fn into_bytes(mut self) -> Vec<u8> {
        self.encoder.encode_terminate(true);
        self.encoder.finish()
    }
}

/// The tables a parser reads, for one parameter set.
pub(super) struct Harness {
    pub(super) params: CodingParams,
    layout: PicLayout,
    maps: ParseMaps,
}

impl Harness {
    /// A picture with one tile.
    pub(super) fn new(params: CodingParams) -> Self {
        let ctb = 1u32 << params.ctb_log2;
        let layout = PicLayout::new(&LayoutSizes {
            width: params.width,
            height: params.height,
            ctb_log2: params.ctb_log2,
            min_tb_log2: params.min_tb_log2,
            column_widths: &[params.width.div_ceil(ctb)],
            row_heights: &[params.height.div_ceil(ctb)],
        })
        .unwrap();
        let maps = ParseMaps::new(&params, &layout).unwrap();
        Self {
            params,
            layout,
            maps,
        }
    }

    /// Parses the CTUs at `ctus` from `data`, with no closing checks.
    pub(super) fn try_parse(
        &mut self,
        data: &[u8],
        ctus: &[(u32, u32)],
    ) -> Result<Vec<CtuRecords>, HevcError> {
        let mut parser = CtuParser::new(&self.params, &self.layout, &mut self.maps, data, 0)?;
        let mut records = Vec::new();
        for &(x, y) in ctus {
            parser.begin_ctu(x, y);
            parser.parse_coding_tree(x, y)?;
            records.push(parser.records.clone());
        }
        Ok(records)
    }

    /// Parses `writer`'s CTUs, and checks the closing pattern and contexts.
    pub(super) fn run(&mut self, writer: SyntaxWriter, ctus: &[(u32, u32)]) -> Vec<CtuRecords> {
        let (data, contexts) = writer.finish();
        let mut parser =
            CtuParser::new(&self.params, &self.layout, &mut self.maps, &data, 0).unwrap();
        let mut records = Vec::new();
        for &(x, y) in ctus {
            parser.begin_ctu(x, y);
            parser.parse_coding_tree(x, y).unwrap();
            records.push(parser.records.clone());
        }
        assert_eq!(parser.cabac.bypass_bits(32), SIGNATURE);
        assert_eq!(parser.cabac.contexts, contexts);
        assert!(!parser.cabac.exhausted());
        records
    }

    /// A parser over `data`, for a test that calls one syntax function.
    pub(super) fn parser<'a>(&'a mut self, data: &'a [u8]) -> CtuParser<'a, 'a> {
        CtuParser::new(&self.params, &self.layout, &mut self.maps, data, 0).unwrap()
    }
}

/// Checks the closing pattern and contexts of a parser built by
/// [`Harness::parser`].
pub(super) fn check_end(parser: &mut CtuParser<'_, '_>, contexts: &ContextTable) {
    assert_eq!(parser.cabac.bypass_bits(32), SIGNATURE);
    assert_eq!(&parser.cabac.contexts, contexts);
    assert!(!parser.cabac.exhausted());
}
