//! One picture at a time, at one sample type: reference picture sets, the parse and
//! reconstruct stages for each slice segment, the in-loop filters, the hash check,
//! and the decoded picture buffer.

use std::sync::Arc;

use crate::bitstream::hevc_pps::PicParameterSet;
use crate::bitstream::hevc_slice::SliceSegmentHeader;
use crate::bitstream::hevc_sps_full::SeqParameterSet;
use crate::decode::cpu::dsp::sample::Sample;

use super::dpb::buffer::{Dpb, DpbLimits, Frame, FrameFormat, RefPicSets};
use super::dpb::poc::{is_bla, is_cra};
use super::dpb::ref_lists::SliceRefs;
use super::dpb::rps::RpsPocs;
use super::error::HevcError;
use super::filter::deblock::{DeblockContext, deblock_picture};
use super::filter::edges::{EdgeMap, StrengthInput};
use super::filter::sao::{SaoContext, sao_is_active, sao_picture};
use super::hash::picture_hash::{PictureHash, check_picture_hash};
use super::layout::PicLayout;
use super::motion::field::MotionField;
use super::params::{CodingParams, check_stream, picture_layout};
use super::picture::{DisplayWindow, Picture};
use super::recon::ctu::{ReconContext, Reconstructor};
use super::recon::inter::InterContext;
use super::scaling::ScalingFactors;
use super::syntax::maps::ParseMaps;
use super::syntax::slice_data::{EntropyCarry, SegmentInput, parse_slice_segment};

/// The tables of one SPS and PPS pair. They stay while the pair stays.
#[derive(Debug)]
struct PictureTables {
    sps: SeqParameterSet,
    pps: PicParameterSet,
    layout: PicLayout,
    maps: ParseMaps,
    edges: EdgeMap,
    field: MotionField,
    factors: Option<ScalingFactors>,
}

impl PictureTables {
    fn new(
        sps: &SeqParameterSet,
        pps: &PicParameterSet,
        header: &SliceSegmentHeader,
    ) -> Result<Self, HevcError> {
        check_stream(sps, pps)?;
        let params = CodingParams::from_headers(sps, pps, header)?;
        let layout = picture_layout(sps, pps)?;
        let maps = ParseMaps::new(&params, &layout)?;
        let field = MotionField::new(params.width, params.height, &layout)?;
        Ok(Self {
            sps: sps.clone(),
            pps: pps.clone(),
            layout,
            maps,
            edges: EdgeMap::new(params.width, params.height)?,
            field,
            factors: ScalingFactors::select(sps, pps),
        })
    }
}

/// The picture being decoded.
#[derive(Debug)]
struct ActivePicture<S> {
    /// The buffer that becomes a decoded picture.
    frame: Frame<S>,
    carry: EntropyCarry,
    /// The first slice's values. In-loop filters read fields that hold per picture.
    params: CodingParams,
    /// `PicOutputFlag`.
    output: bool,
    hash: Option<PictureHash>,
    /// The reference picture set of this picture.
    sets: RefPicSets<S>,
    /// The lists of the current slice.
    refs: SliceRefs<S>,
    limits: DpbLimits,
}

/// What [`PictureDecoder::begin`] reads from the sequence state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PictureBegin {
    /// `PicOrderCntVal`.
    pub(crate) poc: i32,
    /// True for an IRAP picture with `NoRaslOutputFlag` 1.
    pub(crate) irap_no_rasl: bool,
}

/// Decodes pictures of one sample type. Allocations reuse between pictures of one
/// size.
#[derive(Debug)]
pub(crate) struct PictureDecoder<S> {
    tables: Option<PictureTables>,
    reconstructor: Reconstructor,
    active: Option<ActivePicture<S>>,
    /// The working copy for SAO.
    deblocked: Option<Picture<S>>,
    dpb: Dpb<S>,
}

impl<S: Sample> PictureDecoder<S> {
    /// A decoder with no tables yet.
    pub(crate) const fn new() -> Self {
        Self {
            tables: None,
            reconstructor: Reconstructor::new(),
            active: None,
            deblocked: None,
            dpb: Dpb::new(),
        }
    }

    /// True while a picture has slices and no filters yet.
    pub(crate) const fn is_active(&self) -> bool {
        self.active.is_some()
    }

    /// Starts a picture from its first slice segment: the reference picture set
    /// (8.3.2), output and removal (C.5.2.2), and unavailable pictures (8.3.3). Call
    /// [`Self::finish`] on the picture before it first.
    ///
    /// # Errors
    ///
    /// - Any error of `check_stream`, and any error of `CodingParams::from_headers`.
    /// - [`HevcError::Allocation`] when a buffer does not fit in memory.
    /// - [`HevcError::DisplayWindow`] for a conformance window that does not fit.
    pub(crate) fn begin(
        &mut self,
        sps: &SeqParameterSet,
        pps: &PicParameterSet,
        header: &SliceSegmentHeader,
        begin: PictureBegin,
    ) -> Result<(), HevcError> {
        let reuse = self
            .tables
            .as_ref()
            .is_some_and(|tables| tables.sps == *sps && tables.pps == *pps);
        if !reuse {
            self.tables = Some(PictureTables::new(sps, pps, header)?);
        }
        if let Some(tables) = &mut self.tables {
            tables.maps.reset();
            tables.edges.reset();
            tables.field.begin_picture();
        }
        let params = CodingParams::from_headers(sps, pps, header)?;
        let max_poc_lsb = 1u32 << (sps.log2_max_pic_order_cnt_lsb_minus4 + 4);
        let rps = RpsPocs::derive(header, sps, begin.poc);
        let sets = self.dpb.apply_rps(&rps, max_poc_lsb, begin.irap_no_rasl)?;
        let nal_unit_type = header.nal_unit_type;
        // C.5.2.2: a CRA picture sets NoOutputOfPriorPicsFlag to 1.
        let no_output = is_cra(nal_unit_type) || header.no_output_of_prior_pics_flag;
        let limits = DpbLimits::from_sps(sps);
        self.dpb
            .before_picture(begin.irap_no_rasl, no_output, &limits);
        let format = FrameFormat {
            width: params.width as usize,
            height: params.height as usize,
            bit_depth: params.bit_depth,
            window: conformance_window(sps),
        };
        if is_bla(nal_unit_type) || (is_cra(nal_unit_type) && begin.irap_no_rasl) {
            self.dpb.generate_unavailable(&sets, &format)?;
        }
        let mut frame = self.dpb.take_frame(&format)?;
        frame.poc = begin.poc;
        self.active = Some(ActivePicture {
            frame,
            carry: EntropyCarry::default(),
            params,
            output: header.pic_output_flag,
            hash: None,
            sets,
            refs: SliceRefs::intra(),
            limits,
        });
        Ok(())
    }

    /// Decodes one slice segment of the active picture. A call with no active
    /// picture does nothing. `data` is the NAL unit without emulation prevention
    /// bytes.
    ///
    /// # Errors
    ///
    /// - [`HevcError::MissingReference`] for a list entry the buffer does not hold.
    /// - Any error of the slice data.
    pub(crate) fn decode_slice(
        &mut self,
        header: &SliceSegmentHeader,
        data: &[u8],
    ) -> Result<(), HevcError> {
        let (Some(tables), Some(active)) = (&mut self.tables, &mut self.active) else {
            return Ok(());
        };
        let params = CodingParams::from_headers(&tables.sps, &tables.pps, header)?;
        if !header.dependent_slice_segment_flag {
            let poc = active.frame.poc;
            active.refs = SliceRefs::build(header, &active.sets, poc, &tables.sps)?;
            tables.field.begin_slice(active.refs.lists)?;
        }
        let input = SegmentInput {
            params: &params,
            layout: &tables.layout,
            data,
            start: header.slice_data_rbsp_offset,
            first_ctb_rs: header.slice_segment_address,
            dependent: header.dependent_slice_segment_flag,
        };
        let layout = &tables.layout;
        let factors = tables.factors.as_ref();
        let edges = &mut tables.edges;
        let field = &mut tables.field;
        let reconstructor = &mut self.reconstructor;
        let refs = &active.refs;
        let poc = active.frame.poc;
        let picture = &mut active.frame.picture;
        parse_slice_segment(
            &input,
            &mut tables.maps,
            &mut active.carry,
            |maps, records| {
                let context = ReconContext {
                    params: &params,
                    layout,
                    maps,
                    factors,
                };
                let mut inter = InterContext {
                    refs,
                    field: &mut *field,
                    poc,
                };
                reconstructor.reconstruct_ctu(&context, &mut inter, records, picture);
                edges.mark_ctu(
                    &StrengthInput {
                        params: &params,
                        layout,
                        maps,
                        field: &*field,
                    },
                    records,
                );
                Ok(())
            },
        )
    }

    /// Records the hash the picture has to match.
    pub(crate) fn set_hash(&mut self, hash: PictureHash) {
        if let Some(active) = &mut self.active {
            active.hash = Some(hash);
        }
    }

    /// Ends the active picture: the in-loop filters (8.7), the hash check when
    /// `check_hash` is true and a hash arrived, then the buffer steps of C.5.2.3.
    ///
    /// # Errors
    ///
    /// - [`HevcError::IncompletePicture`] when slices do not cover the picture.
    /// - [`HevcError::PictureHashMismatch`] when a plane differs from its hash.
    /// - [`HevcError::Allocation`] when a buffer does not fit in memory.
    pub(crate) fn finish(&mut self, check_hash: bool) -> Result<(), HevcError> {
        let (Some(tables), Some(mut active)) = (&self.tables, self.active.take()) else {
            return Ok(());
        };
        let expected = tables.layout.ctb_count();
        if active.carry.decoded_ctbs != expected {
            return Err(HevcError::IncompletePicture {
                decoded: active.carry.decoded_ctbs,
                expected,
            });
        }
        deblock_picture(
            &mut active.frame.picture,
            &DeblockContext {
                params: &active.params,
                layout: &tables.layout,
                maps: &tables.maps,
                edges: &tables.edges,
            },
        );
        let sao = SaoContext {
            params: &active.params,
            layout: &tables.layout,
            maps: &tables.maps,
        };
        if sao_is_active(&sao) {
            let (width, height) = (
                active.frame.picture.luma().width(),
                active.frame.picture.luma().height(),
            );
            let same_size = self.deblocked.as_ref().is_some_and(|picture| {
                picture.luma().width() == width && picture.luma().height() == height
            });
            if !same_size {
                self.deblocked = Some(Picture::new(width, height)?);
            }
            if let Some(deblocked) = &mut self.deblocked {
                sao_picture(&mut active.frame.picture, deblocked, &sao);
            }
        }
        if check_hash && let Some(hash) = &active.hash {
            check_picture_hash(&active.frame.picture, hash)?;
        }
        let (width, height) = (active.params.width, active.params.height);
        tables
            .field
            .compress(width, height, &mut active.frame.col)?;
        self.dpb.insert(active.frame, active.output, &active.limits);
        Ok(())
    }

    /// Drops the active picture after an error.
    pub(crate) fn abandon(&mut self) {
        self.active = None;
    }

    /// Outputs every waiting picture, at the end of the stream.
    pub(crate) fn flush_output(&mut self) {
        self.dpb.flush();
    }

    /// Empties the buffer when the stream moves to the other sample type. With
    /// `discard`, waiting pictures leave with no output.
    pub(crate) fn close_sequence(&mut self, discard: bool) {
        self.dpb.close(discard);
    }

    /// The oldest output picture.
    pub(crate) fn pop_output(&mut self) -> Option<Arc<Frame<S>>> {
        self.dpb.pop_output()
    }
}

/// The conformance window of (7.4.3.2.1) in luma samples. Offsets count in units of
/// `SubWidthC` and `SubHeightC`. An oversized window saturates, and
/// [`Picture::set_window`] refuses it.
fn conformance_window(sps: &SeqParameterSet) -> DisplayWindow {
    let unit_x = sps.sub_width_c() as usize;
    let unit_y = sps.sub_height_c() as usize;
    let left = sps.conf_win_left_offset as usize * unit_x;
    let right = sps.conf_win_right_offset as usize * unit_x;
    let top = sps.conf_win_top_offset as usize * unit_y;
    let bottom = sps.conf_win_bottom_offset as usize * unit_y;
    DisplayWindow {
        x: left,
        y: top,
        width: (sps.pic_width_in_luma_samples as usize).saturating_sub(left + right),
        height: (sps.pic_height_in_luma_samples as usize).saturating_sub(top + bottom),
    }
}
