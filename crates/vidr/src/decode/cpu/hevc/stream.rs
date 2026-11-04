//! The NAL unit walk of the crate-internal decoder core: parameter sets, slice
//! segments, access unit boundaries, hash SEI messages, picture order, and output.
//!
//! This core decodes Main and Main 10 pictures of every slice type, and returns them
//! in output order (C.5.2). `decode::hevc::HevcDecoder` adapts it to the `Decoder` trait.

use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;

use crate::bitstream::hevc_pps::PicParameterSet;
use crate::bitstream::hevc_slice::{SliceSegmentHeader, slice_pic_parameter_set_id};
use crate::bitstream::hevc_sps_full::SeqParameterSet;
use crate::bitstream::rbsp::unescape;
use crate::bitstream::{HevcNalType, VuiSignal};

use super::dpb::buffer::{DpbLimits, Frame};
use super::dpb::poc::{PictureFacts, PictureStart, SequenceState, is_cra};
use super::error::{HevcError, SequenceFormat};
use super::hash::picture_hash::find_picture_hash;
use super::picture_decoder::{PictureBegin, PictureDecoder};
use super::stage::{PictureStage, UnitSite};

/// `sps_seq_parameter_set_id` values.
const SPS_IDS: usize = 16;
/// `pps_pic_parameter_set_id` values.
const PPS_IDS: usize = 64;

/// The decoder options.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DecodeOptions {
    /// Compares each picture with its decoded picture hash SEI message, and returns
    /// [`HevcError::PictureHashMismatch`] when they differ. Off by default.
    pub(crate) check_picture_hash: bool,
}

/// A picture in output order, of either sample type. The decoder keeps the same
/// [`Frame`] while it serves as a reference, so an output holds no copy of its samples.
/// [`Frame::picture`] carries the conformance window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum DecodedPicture<A = Arc<Frame<u8>>, B = Arc<Frame<u16>>> {
    /// `BitDepth` 8.
    Eight(A),
    /// `BitDepth` 9 or 10.
    Ten(B),
}

/// The facts of a picture at its first slice segment, for a caller that labels each
/// output picture with what it knew when the picture went in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StartedPicture {
    /// The value [`Frame::decode_order`] holds.
    pub(crate) decode_order: u64,
    /// `PicOutputFlag`. A picture with 0 never leaves as output.
    pub(crate) output: bool,
    /// True for an IRAP picture with `NoRaslOutputFlag` 1. Each earlier picture has
    /// already left as output or left with no output, as C.5.2.2 states.
    pub(crate) irap_no_rasl: bool,
    /// The colour fields of the active SPS's VUI.
    pub(crate) signal: VuiSignal,
    /// `sps_max_dec_pic_buffering_minus1[ HighestTid ] + 1`.
    pub(crate) dpb_size: usize,
}

/// Takes NAL units in decoding order. [`Self::next_output`] returns the pictures in
/// output order.
pub(crate) struct StreamDecoder<E = PictureDecoder<u8>, T = PictureDecoder<u16>>
where
    E: PictureStage<u8>,
    T: PictureStage<u16>,
{
    options: DecodeOptions,
    sps: Vec<Option<SeqParameterSet>>,
    pps: Vec<Option<PicParameterSet>>,
    /// The independent slice segment header of the current slice.
    independent: Option<SliceSegmentHeader>,
    eight: E,
    ten: T,
    sequence: SequenceState,
    /// True through the slice segments of a skipped RASL picture.
    skip_picture: bool,
    output: VecDeque<DecodedPicture<E::Output, T::Output>>,
    /// Pictures that have started decoding.
    pictures_started: u64,
    /// The picture the latest NAL unit started.
    started: Option<StartedPicture>,
    /// The format of the latest picture that started decoding.
    sequence_format: Option<SequenceFormat>,
    /// `slice_pic_parameter_set_id` of the current picture.
    picture_pps: Option<u32>,
    /// The pts of the packet in progress.
    pts: i64,
}

impl<E, T> fmt::Debug for StreamDecoder<E, T>
where
    E: PictureStage<u8>,
    T: PictureStage<u16>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamDecoder")
            .field("pictures_started", &self.pictures_started)
            .finish_non_exhaustive()
    }
}

impl StreamDecoder {
    /// A decoder with no parameter sets, on the calling thread.
    pub(crate) fn new(options: DecodeOptions) -> Self {
        Self::with_stages(options, PictureDecoder::new(), PictureDecoder::new())
    }
}

impl<E, T> StreamDecoder<E, T>
where
    E: PictureStage<u8>,
    T: PictureStage<u16>,
{
    /// A decoder with no parameter sets, over the given picture stages.
    pub(crate) fn with_stages(options: DecodeOptions, eight: E, ten: T) -> Self {
        Self {
            options,
            sps: vec![None; SPS_IDS],
            pps: vec![None; PPS_IDS],
            independent: None,
            eight,
            ten,
            sequence: SequenceState::new(),
            skip_picture: false,
            output: VecDeque::new(),
            pictures_started: 0,
            started: None,
            sequence_format: None,
            picture_pps: None,
            pts: 0,
        }
    }

    /// The picture stages, for a test that changes how they run.
    #[cfg(test)]
    pub(crate) const fn stages_mut(&mut self) -> (&mut E, &mut T) {
        (&mut self.eight, &mut self.ten)
    }

    /// Names the packet that the next NAL units come from.
    pub(crate) const fn set_pts(&mut self, pts: i64) {
        self.pts = pts;
    }

    /// Decodes one NAL unit, 2-byte header included and start code excluded. Units of
    /// other layers and reserved types are dropped. A picture with an error is dropped.
    ///
    /// # Errors
    ///
    /// - [`HevcError::Header`] for a parameter set or slice header that fails its
    ///   checks.
    /// - [`HevcError::MissingParameterSet`] for a slice before its parameter sets.
    /// - [`HevcError::MissingReference`] for a list entry the buffer does not hold.
    /// - [`HevcError::SequenceFormatChange`] for a new size or bit depth at a picture
    ///   that does not start a coded video sequence.
    /// - [`HevcError::InvalidSyntax`] for a slice segment that names another PPS than
    ///   the first slice segment of its picture.
    /// - [`HevcError::Unsupported`] for a tool outside Main and Main 10 at 4:2:0.
    /// - [`HevcError::PictureHashMismatch`] with the hash check on.
    /// - Any error of the slice data.
    pub(crate) fn decode_nal(&mut self, nal: &[u8]) -> Result<(), HevcError> {
        let (Some(&first), Some(&second)) = (nal.first(), nal.get(1)) else {
            return Ok(());
        };
        // nuh_layer_id: the low bit of the first byte and the top five of the second.
        if first & 1 != 0 || second >> 3 != 0 {
            return Ok(());
        }
        let site = UnitSite::Nal {
            bytes: nal.len(),
            pts: self.pts,
        };
        self.eight.enter_unit(site);
        self.ten.enter_unit(site);
        let result = self.dispatch(nal, HevcNalType::from_header(first));
        if result.is_err() {
            self.eight.abandon();
            self.ten.abandon();
        }
        self.collect_output();
        result
    }

    /// The next picture in output order, when one is ready.
    pub(crate) fn next_output(&mut self) -> Option<DecodedPicture<E::Output, T::Output>> {
        self.output.pop_front()
    }

    /// The picture that [`Self::decode_nal`] started since the last call, if any.
    pub(crate) fn take_started(&mut self) -> Option<StartedPicture> {
        self.started.take()
    }

    /// Moves output pictures from the sample type decoders into the queue.
    fn collect_output(&mut self) {
        while let Some(frame) = self.eight.pop_output() {
            self.output.push_back(DecodedPicture::Eight(frame));
        }
        while let Some(frame) = self.ten.pop_output() {
            self.output.push_back(DecodedPicture::Ten(frame));
        }
    }

    fn dispatch(&mut self, nal: &[u8], nal_type: HevcNalType) -> Result<(), HevcError> {
        match nal_type {
            HevcNalType::SPS => {
                self.end_picture()?;
                let sps = SeqParameterSet::parse(nal)?;
                let id = sps.sps_seq_parameter_set_id as usize;
                if let Some(slot) = self.sps.get_mut(id) {
                    *slot = Some(sps);
                }
                Ok(())
            }
            HevcNalType::PPS => {
                self.end_picture()?;
                let pps = PicParameterSet::parse(nal)?;
                let id = pps.pps_pic_parameter_set_id as usize;
                if let Some(slot) = self.pps.get_mut(id) {
                    *slot = Some(pps);
                }
                Ok(())
            }
            // 7.4.2.4.4: each of these after a picture's last slice starts a new
            // access unit.
            HevcNalType::VPS | HevcNalType::AUD | HevcNalType::SEI_PREFIX => self.end_picture(),
            // 8.1.3: the next picture restarts POC and NoRaslOutputFlag.
            HevcNalType::END_OF_SEQUENCE | HevcNalType::END_OF_BITSTREAM => {
                self.end_picture()?;
                self.sequence.end_of_sequence();
                Ok(())
            }
            HevcNalType::SEI_SUFFIX => {
                if self.options.check_picture_hash
                    && let Some(hash) = find_picture_hash(&unescape(nal))?
                {
                    self.eight.set_hash(hash);
                    self.ten.set_hash(hash);
                }
                Ok(())
            }
            _ => match nal_type.value() {
                0..=9 | 16..=21 => self.decode_slice(nal),
                _ => Ok(()),
            },
        }
    }

    /// Ends the current picture at a unit boundary.
    fn end_picture(&mut self) -> Result<(), HevcError> {
        let check = self.options.check_picture_hash;
        finish_active(&mut self.eight, &mut self.ten, check)
    }

    /// Ends the stream: finishes the current picture and outputs every waiting
    /// picture. The next picture starts a new bitstream.
    ///
    /// # Errors
    ///
    /// Any error of [`PictureStage::finish`]. The waiting pictures still leave.
    pub(crate) fn flush(&mut self) -> Result<(), HevcError> {
        self.eight.enter_unit(UnitSite::End);
        self.ten.enter_unit(UnitSite::End);
        let result = self.end_picture();
        self.eight.flush_output();
        self.ten.flush_output();
        self.collect_output();
        self.sequence = SequenceState::new();
        self.sequence_format = None;
        self.picture_pps = None;
        result
    }

    /// One slice segment NAL unit.
    fn decode_slice(&mut self, nal: &[u8]) -> Result<(), HevcError> {
        let pps_id = slice_pic_parameter_set_id(nal)?;
        let pps = self
            .pps
            .get(pps_id as usize)
            .and_then(Option::as_ref)
            .ok_or(HevcError::MissingParameterSet {
                kind: "PPS",
                id: pps_id,
            })?;
        let sps_id = pps.pps_seq_parameter_set_id;
        let sps = self
            .sps
            .get(sps_id as usize)
            .and_then(Option::as_ref)
            .ok_or(HevcError::MissingParameterSet {
                kind: "SPS",
                id: sps_id,
            })?;
        let header = SliceSegmentHeader::parse(nal, pps, sps, self.independent.as_ref())?;
        if header.first_slice_segment_in_pic_flag {
            let check = self.options.check_picture_hash;
            finish_active(&mut self.eight, &mut self.ten, check)?;
            self.independent = None;
            self.picture_pps = Some(header.slice_pic_parameter_set_id);
            let start = self.sequence.begin_picture(PictureFacts {
                nal_unit_type: header.nal_unit_type,
                temporal_id: header.temporal_id,
                poc_lsb: header.slice_pic_order_cnt_lsb,
                max_poc_lsb: 1 << (sps.log2_max_pic_order_cnt_lsb_minus4 + 4),
            });
            self.skip_picture = start == PictureStart::Skip;
            if let PictureStart::Decode { poc, irap_no_rasl } = start {
                let format = SequenceFormat {
                    width: sps.pic_width_in_luma_samples,
                    height: sps.pic_height_in_luma_samples,
                    bit_depth: sps.bit_depth_y(),
                };
                check_sequence_format(self.sequence_format, format, irap_no_rasl)?;
                self.sequence_format = Some(format);
                let decode_order = self.pictures_started;
                let begin = PictureBegin {
                    poc,
                    irap_no_rasl,
                    decode_order,
                };
                let discard = is_cra(header.nal_unit_type) || header.no_output_of_prior_pics_flag;
                if sps.bit_depth_y() == 8 {
                    self.ten.close_sequence(discard);
                    self.eight.begin(sps, pps, &header, begin)?;
                } else {
                    self.eight.close_sequence(discard);
                    self.ten.begin(sps, pps, &header, begin)?;
                }
                self.pictures_started += 1;
                self.started = Some(StartedPicture {
                    decode_order,
                    output: header.pic_output_flag,
                    irap_no_rasl,
                    signal: sps.vui.map_or_else(VuiSignal::default, |vui| vui.signal),
                    dpb_size: DpbLimits::from_sps(sps).max_dec_buffering,
                });
            }
        } else if !self.skip_picture && !self.eight.is_active() && !self.ten.is_active() {
            return Err(HevcError::InvalidSyntax {
                element: "first_slice_segment_in_pic_flag",
                value: 0,
                allowed: "1 in the first slice segment of a picture",
            });
        } else if self.picture_pps != Some(header.slice_pic_parameter_set_id) {
            // 7.4.7.1: every slice segment header of a picture holds one value.
            return Err(HevcError::InvalidSyntax {
                element: "slice_pic_parameter_set_id",
                value: i64::from(header.slice_pic_parameter_set_id),
                allowed: "the value of the first slice segment of this picture",
            });
        }
        if !header.dependent_slice_segment_flag {
            self.independent = Some(header.clone());
        }
        if self.skip_picture {
            return Ok(());
        }
        let data = unescape(nal);
        if self.eight.is_active() {
            self.eight.decode_slice(&header, data)
        } else {
            self.ten.decode_slice(&header, data)
        }
    }
}

/// 7.4.2.4.2: an SPS stays active for its whole coded video sequence, and an IRAP
/// picture with `NoRaslOutputFlag` 1 starts each one. A later picture of another
/// format has references that its layout does not fit.
fn check_sequence_format(
    previous: Option<SequenceFormat>,
    current: SequenceFormat,
    irap_no_rasl: bool,
) -> Result<(), HevcError> {
    match previous {
        Some(previous) if !irap_no_rasl && previous != current => {
            Err(HevcError::SequenceFormatChange { previous, current })
        }
        _ => Ok(()),
    }
}

/// Ends the picture of whichever sample type has one.
fn finish_active<E: PictureStage<u8>, T: PictureStage<u16>>(
    eight: &mut E,
    ten: &mut T,
    check_hash: bool,
) -> Result<(), HevcError> {
    if eight.is_active() {
        return eight.finish(check_hash);
    }
    ten.finish(check_hash)
}
