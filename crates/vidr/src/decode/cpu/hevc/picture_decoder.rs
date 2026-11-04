//! One picture at a time, at one sample type: the parse and reconstruct stages for
//! each slice segment, then the in-loop filters and the hash check.

use crate::bitstream::hevc_pps::PicParameterSet;
use crate::bitstream::hevc_slice::SliceSegmentHeader;
use crate::bitstream::hevc_sps_full::SeqParameterSet;
use crate::decode::cpu::dsp::sample::Sample;

use super::error::HevcError;
use super::filter::deblock::{DeblockContext, deblock_picture};
use super::filter::edges::EdgeMap;
use super::filter::sao::{SaoContext, sao_is_active, sao_picture};
use super::hash::picture_hash::{PictureHash, check_picture_hash};
use super::layout::PicLayout;
use super::params::{CodingParams, check_stream, picture_layout};
use super::picture::{DisplayWindow, Picture};
use super::recon::ctu::{ReconContext, Reconstructor};
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
        Ok(Self {
            sps: sps.clone(),
            pps: pps.clone(),
            layout,
            maps,
            edges: EdgeMap::new(params.width, params.height)?,
            factors: ScalingFactors::select(sps, pps),
        })
    }
}

/// The picture being decoded.
#[derive(Debug)]
struct ActivePicture<S> {
    picture: Picture<S>,
    carry: EntropyCarry,
    /// The first slice's values. In-loop filters read fields that hold per picture.
    params: CodingParams,
    /// `pic_output_flag`.
    output: bool,
    hash: Option<PictureHash>,
}

/// A finished picture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FinishedPicture<S> {
    pub(crate) picture: Picture<S>,
    /// `pic_output_flag`.
    pub(crate) output: bool,
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
}

impl<S: Sample> PictureDecoder<S> {
    /// A decoder with no tables yet.
    pub(crate) const fn new() -> Self {
        Self {
            tables: None,
            reconstructor: Reconstructor::new(),
            active: None,
            deblocked: None,
        }
    }

    /// True while a picture has slices and no filters yet.
    pub(crate) const fn is_active(&self) -> bool {
        self.active.is_some()
    }

    /// Starts a picture from its first slice segment. Call [`Self::finish`] on the
    /// picture before it first.
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
        }
        let params = CodingParams::from_headers(sps, pps, header)?;
        let mut picture = Picture::new(params.width as usize, params.height as usize)?;
        picture.set_window(conformance_window(sps))?;
        self.active = Some(ActivePicture {
            picture,
            carry: EntropyCarry::default(),
            params,
            output: header.pic_output_flag,
            hash: None,
        });
        Ok(())
    }

    /// Decodes one slice segment of the active picture. A call with no active
    /// picture does nothing. `data` is the NAL unit without emulation prevention
    /// bytes.
    ///
    /// # Errors
    ///
    /// Any error of the slice data.
    pub(crate) fn decode_slice(
        &mut self,
        header: &SliceSegmentHeader,
        data: &[u8],
    ) -> Result<(), HevcError> {
        let (Some(tables), Some(active)) = (&mut self.tables, &mut self.active) else {
            return Ok(());
        };
        let params = CodingParams::from_headers(&tables.sps, &tables.pps, header)?;
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
        let reconstructor = &mut self.reconstructor;
        let picture = &mut active.picture;
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
                reconstructor.reconstruct_ctu(&context, records, picture);
                edges.mark_ctu(&params, layout, maps, records);
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

    /// Ends the active picture: the in-loop filters (8.7), then the hash check when
    /// `check_hash` is true and a hash arrived.
    ///
    /// # Errors
    ///
    /// - [`HevcError::IncompletePicture`] when slices do not cover the picture.
    /// - [`HevcError::PictureHashMismatch`] when a plane differs from its hash.
    /// - [`HevcError::Allocation`] when a buffer does not fit in memory.
    pub(crate) fn finish(
        &mut self,
        check_hash: bool,
    ) -> Result<Option<FinishedPicture<S>>, HevcError> {
        let (Some(tables), Some(mut active)) = (&self.tables, self.active.take()) else {
            return Ok(None);
        };
        let expected = tables.layout.ctb_count();
        if active.carry.decoded_ctbs != expected {
            return Err(HevcError::IncompletePicture {
                decoded: active.carry.decoded_ctbs,
                expected,
            });
        }
        deblock_picture(
            &mut active.picture,
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
            let same_size = self.deblocked.as_ref().is_some_and(|picture| {
                picture.luma().width() == active.picture.luma().width()
                    && picture.luma().height() == active.picture.luma().height()
            });
            if !same_size {
                let (width, height) = (
                    active.picture.luma().width(),
                    active.picture.luma().height(),
                );
                self.deblocked = Some(Picture::new(width, height)?);
            }
            if let Some(deblocked) = &mut self.deblocked {
                sao_picture(&mut active.picture, deblocked, &sao);
            }
        }
        if check_hash && let Some(hash) = &active.hash {
            check_picture_hash(&active.picture, hash)?;
        }
        Ok(Some(FinishedPicture {
            picture: active.picture,
            output: active.output,
        }))
    }

    /// Drops the active picture after an error.
    pub(crate) fn abandon(&mut self) {
        self.active = None;
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
