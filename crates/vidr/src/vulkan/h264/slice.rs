//! Slice headers through `h264-reader`, picture boundaries (ITU-T H.264 7.4.1.2.4), and
//! a picture's slices in the form Vulkan Video takes.

use h264_reader::Context;
use h264_reader::nal::NalHeader;
use h264_reader::nal::slice::{
    DecRefPicMarking, MemoryManagementControlOperation, PicOrderCountLsb, SliceFamily, SliceHeader,
};
use h264_reader::rbsp::BitReader;

use super::dpb::{Marking, Mmco};
use super::params::DECODER;
use super::poc::PocSlice;
use crate::bitstream::rbsp::unescape;
use crate::error::DecodeError;

/// A slice header's largest form fits here: weights for 32 references and each marking.
const HEADER_BYTES: usize = 4096;
/// Vulkan takes each slice with a start code in front.
const START_CODE: [u8; 3] = [0, 0, 1];

/// The fields of one slice that this decoder reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SliceFacts {
    pub(super) sps_id: u8,
    pub(super) pps_id: u8,
    pub(super) nal_ref_idc: u8,
    pub(super) idr: bool,
    pub(super) intra: bool,
    pub(super) frame_num: u32,
    pub(super) idr_pic_id: u32,
    pub(super) lsb: u32,
    pub(super) delta_bottom: i32,
    pub(super) delta: [i32; 2],
    pub(super) redundant_pic_cnt: u32,
    pub(super) marking: Marking,
    /// `no_output_of_prior_pics_flag`, or `false` for a slice of a non-IDR picture.
    pub(super) no_output_of_prior_pics: bool,
}

impl SliceFacts {
    /// The fields picture order count reads.
    pub(super) const fn poc_slice(&self) -> PocSlice {
        PocSlice {
            idr: self.idr,
            reference: self.nal_ref_idc != 0,
            frame_num: self.frame_num,
            lsb: self.lsb,
            delta_bottom: self.delta_bottom,
            delta: self.delta,
        }
    }

    /// Whether this slice starts a new primary picture after `previous`, by 7.4.1.2.4.
    pub(super) fn starts_picture_after(&self, previous: &Self) -> bool {
        self.frame_num != previous.frame_num
            || self.pps_id != previous.pps_id
            || (self.nal_ref_idc == 0) != (previous.nal_ref_idc == 0)
            || self.lsb != previous.lsb
            || self.delta_bottom != previous.delta_bottom
            || self.delta != previous.delta
            || self.idr != previous.idr
            || (self.idr && self.idr_pic_id != previous.idr_pic_id)
    }
}

/// Parses one slice NAL unit's header, against the parameter sets in `context`.
pub(super) fn parse_slice(context: &Context, nal: &[u8]) -> Result<SliceFacts, DecodeError> {
    let damaged = |reason: String| DecodeError::H264Header {
        decoder: DECODER,
        unit: "slice",
        reason,
    };
    let Some((&first, rest)) = nal.split_first() else {
        return Err(damaged("the NAL unit is empty".to_owned()));
    };
    let header = NalHeader::new(first).map_err(|error| damaged(format!("{error:?}")))?;
    let nal_ref_idc = header.nal_ref_idc();
    let rbsp = unescape(&rest[..rest.len().min(HEADER_BYTES)]);
    let mut reader = BitReader::new(&rbsp[..]);
    let (slice, _, pps) = SliceHeader::from_bits(context, &mut reader, header, None)
        .map_err(|error| damaged(format!("{error:?}")))?;
    let (lsb, delta_bottom, delta) = match slice.pic_order_cnt_lsb {
        Some(PicOrderCountLsb::Frame(lsb)) => (lsb, 0, [0, 0]),
        Some(PicOrderCountLsb::FieldsAbsolute {
            pic_order_cnt_lsb,
            delta_pic_order_cnt_bottom,
        }) => (pic_order_cnt_lsb, delta_pic_order_cnt_bottom, [0, 0]),
        Some(PicOrderCountLsb::FieldsDelta(delta)) => (0, 0, delta),
        None => (0, 0, [0, 0]),
    };
    Ok(SliceFacts {
        sps_id: pps.seq_parameter_set_id.id(),
        pps_id: pps.pic_parameter_set_id.id(),
        nal_ref_idc,
        idr: slice.idr_pic_id.is_some(),
        intra: matches!(slice.slice_type.family, SliceFamily::I | SliceFamily::SI),
        frame_num: u32::from(slice.frame_num),
        idr_pic_id: slice.idr_pic_id.unwrap_or(0),
        lsb,
        delta_bottom,
        delta,
        redundant_pic_cnt: slice.redundant_pic_cnt.unwrap_or(0),
        marking: marking(nal_ref_idc, slice.dec_ref_pic_marking.as_ref()),
        no_output_of_prior_pics: matches!(
            slice.dec_ref_pic_marking,
            Some(DecRefPicMarking::Idr {
                no_output_of_prior_pics_flag: true,
                ..
            })
        ),
    })
}

fn marking(nal_ref_idc: u8, syntax: Option<&DecRefPicMarking>) -> Marking {
    if nal_ref_idc == 0 {
        return Marking::NonReference;
    }
    match syntax {
        Some(DecRefPicMarking::Idr {
            long_term_reference_flag,
            ..
        }) => Marking::Idr {
            long_term: *long_term_reference_flag,
        },
        Some(DecRefPicMarking::Adaptive(operations)) => {
            Marking::Adaptive(operations.iter().map(mmco).collect())
        }
        Some(DecRefPicMarking::SlidingWindow) | None => Marking::SlidingWindow,
    }
}

const fn mmco(operation: &MemoryManagementControlOperation) -> Mmco {
    use MemoryManagementControlOperation as Syntax;
    match *operation {
        Syntax::ShortTermUnusedForRef {
            difference_of_pic_nums_minus1,
        } => Mmco::UnmarkShort {
            difference_of_pic_nums_minus1,
        },
        Syntax::LongTermUnusedForRef { long_term_pic_num } => {
            Mmco::UnmarkLong { long_term_pic_num }
        }
        Syntax::ShortTermUsedForLongTerm {
            difference_of_pic_nums_minus1,
            long_term_frame_idx,
        } => Mmco::ShortToLong {
            difference_of_pic_nums_minus1,
            long_term_frame_idx,
        },
        Syntax::MaxUsedLongTermFrameRef {
            max_long_term_frame_idx_plus1,
        } => Mmco::MaxLongTermIndex {
            max_long_term_frame_idx_plus1,
        },
        Syntax::AllRefPicturesUnused => Mmco::UnmarkAll,
        Syntax::CurrentUsedForLongTerm {
            long_term_frame_idx,
        } => Mmco::CurrentToLong {
            long_term_frame_idx,
        },
    }
}

/// One primary coded picture being gathered. It borrows the packet that holds its slices,
/// and one packet holds one access unit.
#[derive(Clone, Debug)]
pub(super) struct Picture<'u> {
    /// The first slice, whose fields every slice of a picture repeats.
    pub(super) facts: SliceFacts,
    /// Whether every slice is intra.
    pub(super) intra: bool,
    pub(super) pts: i64,
    /// The bitstream in pieces: each slice behind a start code.
    pub(super) parts: Vec<&'u [u8]>,
    /// Each slice's start in the bitstream.
    pub(super) slice_offsets: Vec<u32>,
    bytes: u32,
}

impl<'u> Picture<'u> {
    pub(super) fn new(facts: SliceFacts, pts: i64) -> Self {
        Self {
            intra: facts.intra,
            facts,
            pts,
            parts: Vec::new(),
            slice_offsets: Vec::new(),
            bytes: 0,
        }
    }

    /// Adds a slice's NAL unit. `intra` states whether the slice is intra.
    pub(super) fn push_slice(&mut self, intra: bool, nal: &'u [u8]) {
        self.intra &= intra;
        self.slice_offsets.push(self.bytes);
        self.bytes += (START_CODE.len() + nal.len()) as u32;
        self.parts.push(&START_CODE);
        self.parts.push(nal);
    }
}

#[cfg(test)]
mod tests {
    use super::super::params::{parse_pps, parse_sps};
    use super::super::test_nal::{NalWriter, TestSps, pps_nal, sps_nal};
    use super::*;

    fn context() -> Context {
        let mut context = Context::new();
        context.put_seq_param_set(parse_sps(&sps_nal(&TestSps::default())).expect("parses"));
        let pps = parse_pps(&context, &pps_nal(0, 0, 0)).expect("parses");
        context.put_pic_param_set(pps);
        context
    }

    /// An IDR I slice: first_mb 0, type 7, PPS 0, frame_num 0 in 4 bits, idr_pic_id, lsb.
    fn idr_slice(idr_pic_id: u32, long_term: bool) -> Vec<u8> {
        idr_slice_with_output_flag(idr_pic_id, false, long_term)
    }

    /// The same, with `no_output_of_prior_pics_flag`.
    fn idr_slice_with_output_flag(idr_pic_id: u32, no_output: bool, long_term: bool) -> Vec<u8> {
        let mut nal = NalWriter::new(0x65);
        nal.ue(0).ue(7).ue(0).bits(0, 4).ue(idr_pic_id).bits(0, 6);
        nal.flag(no_output).flag(long_term);
        nal.se(0).ue(1);
        // Slice data follows the header, as the parser requires.
        nal.bits(0xA5A5, 16);
        nal.finish()
    }

    /// A non-IDR P slice with the given marking syntax.
    fn p_slice(frame_num: u32, lsb: u32, mmco: &[(u32, u32)]) -> Vec<u8> {
        let mut nal = NalWriter::new(0x41);
        nal.ue(0).ue(5).ue(0).bits(frame_num, 4).bits(lsb, 6);
        nal.flag(false).flag(false);
        if mmco.is_empty() {
            nal.flag(false);
        } else {
            nal.flag(true);
            for &(operation, value) in mmco {
                nal.ue(operation).ue(value);
            }
            nal.ue(0);
        }
        nal.se(0).ue(1);
        // Slice data follows the header, as the parser requires.
        nal.bits(0xA5A5, 16);
        nal.finish()
    }

    #[test]
    fn an_idr_slice_reads_its_fields() {
        let facts = parse_slice(&context(), &idr_slice(3, true)).expect("parses");
        assert!(facts.idr && facts.intra);
        assert_eq!(facts.nal_ref_idc, 3);
        assert_eq!(facts.idr_pic_id, 3);
        assert_eq!(facts.marking, Marking::Idr { long_term: true });
        assert!(facts.poc_slice().reference);
    }

    #[test]
    fn an_idr_slice_reads_its_output_flag() {
        let facts = parse_slice(&context(), &idr_slice(0, false)).expect("parses");
        assert!(!facts.no_output_of_prior_pics);
        let flagged = idr_slice_with_output_flag(0, true, false);
        let facts = parse_slice(&context(), &flagged).expect("parses");
        assert!(facts.no_output_of_prior_pics);
        assert_eq!(facts.marking, Marking::Idr { long_term: false });
        let facts = parse_slice(&context(), &p_slice(5, 10, &[])).expect("parses");
        assert!(!facts.no_output_of_prior_pics);
    }

    #[test]
    fn a_p_slice_reads_its_marking() {
        let facts = parse_slice(&context(), &p_slice(5, 10, &[])).expect("parses");
        assert!(!facts.idr && !facts.intra);
        assert_eq!((facts.frame_num, facts.lsb), (5, 10));
        assert_eq!(facts.marking, Marking::SlidingWindow);

        let facts = parse_slice(&context(), &p_slice(5, 10, &[(1, 2), (6, 1)])).expect("parses");
        assert_eq!(
            facts.marking,
            Marking::Adaptive(vec![
                Mmco::UnmarkShort {
                    difference_of_pic_nums_minus1: 2,
                },
                Mmco::CurrentToLong {
                    long_term_frame_idx: 1,
                },
            ])
        );
    }

    #[test]
    fn a_non_reference_slice_marks_nothing() {
        assert_eq!(
            marking(0, Some(&DecRefPicMarking::SlidingWindow)),
            Marking::NonReference
        );
    }

    #[test]
    fn a_damaged_slice_is_named() {
        assert!(matches!(
            parse_slice(&context(), &[]),
            Err(DecodeError::H264Header { unit: "slice", .. })
        ));
        let empty = Context::new();
        assert!(matches!(
            parse_slice(&empty, &idr_slice(0, false)),
            Err(DecodeError::H264Header { unit: "slice", .. })
        ));
    }

    #[test]
    fn each_field_of_7_4_1_2_4_starts_a_picture() {
        let base = parse_slice(&context(), &idr_slice(1, false)).expect("parses");
        assert!(!base.starts_picture_after(&base));
        let changed = |change: fn(&mut SliceFacts)| {
            let mut next = base.clone();
            change(&mut next);
            next.starts_picture_after(&base)
        };
        assert!(changed(|next| next.frame_num = 1));
        assert!(changed(|next| next.pps_id = 1));
        assert!(changed(|next| next.nal_ref_idc = 0));
        assert!(changed(|next| next.lsb = 2));
        assert!(changed(|next| next.delta_bottom = 2));
        assert!(changed(|next| next.delta = [1, 0]));
        assert!(changed(|next| next.idr = false));
        assert!(changed(|next| next.idr_pic_id = 9));
        // nal_ref_idc 3 against 2 is the same zero-ness.
        assert!(!changed(|next| next.nal_ref_idc = 2));
    }

    #[test]
    fn slices_gather_behind_start_codes() {
        let facts = parse_slice(&context(), &idr_slice(0, false)).expect("parses");
        let intra = facts.intra;
        let mut picture = Picture::new(facts, 40);
        picture.push_slice(intra, &[0x65, 1, 2]);
        picture.push_slice(false, &[0x41, 3]);
        assert_eq!(picture.slice_offsets, [0, 6]);
        assert_eq!(
            picture.parts.concat(),
            [0, 0, 1, 0x65, 1, 2, 0, 0, 1, 0x41, 3]
        );
        assert!(!picture.intra);
    }
}
