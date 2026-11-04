//! The NAL unit walk of the crate-internal decoder core: parameter sets, slice
//! segments, access unit boundaries, and hash SEI messages.
//!
//! This core decodes intra pictures only, in decoding order. It holds no decoded
//! picture buffer. A later unit adds inter prediction, then a `Decoder` adapter.

use crate::bitstream::HevcNalType;
use crate::bitstream::hevc_pps::PicParameterSet;
use crate::bitstream::hevc_slice::{SLICE_TYPE_I, SliceSegmentHeader, slice_pic_parameter_set_id};
use crate::bitstream::hevc_sps_full::SeqParameterSet;
use crate::bitstream::rbsp::unescape;

use super::error::HevcError;
use super::hash::picture_hash::find_picture_hash;
use super::picture_decoder::{FinishedPicture, PictureDecoder};

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

/// A finished picture of either sample type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum DecodedPicture {
    /// `BitDepth` 8.
    Eight(FinishedPicture<u8>),
    /// `BitDepth` 9 or 10.
    Ten(FinishedPicture<u16>),
}

/// Takes NAL units in decoding order, and returns each picture when it ends.
#[derive(Debug)]
pub(crate) struct StreamDecoder {
    options: DecodeOptions,
    sps: Vec<Option<SeqParameterSet>>,
    pps: Vec<Option<PicParameterSet>>,
    /// The independent slice segment header of the current slice.
    independent: Option<SliceSegmentHeader>,
    eight: PictureDecoder<u8>,
    ten: PictureDecoder<u16>,
}

impl StreamDecoder {
    /// A decoder with no parameter sets.
    pub(crate) fn new(options: DecodeOptions) -> Self {
        Self {
            options,
            sps: vec![None; SPS_IDS],
            pps: vec![None; PPS_IDS],
            independent: None,
            eight: PictureDecoder::new(),
            ten: PictureDecoder::new(),
        }
    }

    /// Decodes one NAL unit, 2-byte header included and start code excluded. Returns
    /// the picture that this unit ends, when it ends one. Units of other layers and
    /// reserved types are dropped. A picture with an error is dropped.
    ///
    /// # Errors
    ///
    /// - [`HevcError::Header`] for a parameter set or slice header that fails its
    ///   checks.
    /// - [`HevcError::MissingParameterSet`] for a slice before its parameter sets.
    /// - [`HevcError::Unsupported`] for a P or B slice, and for a tool outside Main
    ///   and Main 10 at 4:2:0.
    /// - [`HevcError::PictureHashMismatch`] with the hash check on.
    /// - Any error of the slice data.
    pub(crate) fn decode_nal(&mut self, nal: &[u8]) -> Result<Option<DecodedPicture>, HevcError> {
        let (Some(&first), Some(&second)) = (nal.first(), nal.get(1)) else {
            return Ok(None);
        };
        // nuh_layer_id: the low bit of the first byte and the top five of the second.
        if first & 1 != 0 || second >> 3 != 0 {
            return Ok(None);
        }
        let result = self.dispatch(nal, HevcNalType::from_header(first));
        if result.is_err() {
            self.eight.abandon();
            self.ten.abandon();
        }
        result
    }

    fn dispatch(
        &mut self,
        nal: &[u8],
        nal_type: HevcNalType,
    ) -> Result<Option<DecodedPicture>, HevcError> {
        match nal_type {
            HevcNalType::SPS => {
                let finished = self.flush()?;
                let sps = SeqParameterSet::parse(nal)?;
                let id = sps.sps_seq_parameter_set_id as usize;
                if let Some(slot) = self.sps.get_mut(id) {
                    *slot = Some(sps);
                }
                Ok(finished)
            }
            HevcNalType::PPS => {
                let finished = self.flush()?;
                let pps = PicParameterSet::parse(nal)?;
                let id = pps.pps_pic_parameter_set_id as usize;
                if let Some(slot) = self.pps.get_mut(id) {
                    *slot = Some(pps);
                }
                Ok(finished)
            }
            // 7.4.2.4.4: each of these after a picture's last slice starts a new
            // access unit, or closes the stream.
            HevcNalType::VPS
            | HevcNalType::AUD
            | HevcNalType::SEI_PREFIX
            | HevcNalType::END_OF_SEQUENCE
            | HevcNalType::END_OF_BITSTREAM => self.flush(),
            HevcNalType::SEI_SUFFIX => {
                if self.options.check_picture_hash
                    && let Some(hash) = find_picture_hash(&unescape(nal))?
                {
                    self.eight.set_hash(hash);
                    self.ten.set_hash(hash);
                }
                Ok(None)
            }
            _ => match nal_type.value() {
                0..=9 | 16..=21 => self.decode_slice(nal),
                _ => Ok(None),
            },
        }
    }

    /// Ends the current picture, at the end of a stream or a unit boundary.
    ///
    /// # Errors
    ///
    /// Any error of [`PictureDecoder::finish`].
    pub(crate) fn flush(&mut self) -> Result<Option<DecodedPicture>, HevcError> {
        let check = self.options.check_picture_hash;
        finish_active(&mut self.eight, &mut self.ten, check)
    }

    /// One slice segment NAL unit.
    fn decode_slice(&mut self, nal: &[u8]) -> Result<Option<DecodedPicture>, HevcError> {
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
        if header.slice_type != SLICE_TYPE_I {
            return Err(HevcError::Unsupported {
                field: "slice_type",
                value: header.slice_type.into(),
                supported: "2, an I slice, because this decoder has no inter prediction yet",
            });
        }
        let mut finished = None;
        if header.first_slice_segment_in_pic_flag {
            let check = self.options.check_picture_hash;
            finished = finish_active(&mut self.eight, &mut self.ten, check)?;
            self.independent = None;
            if sps.bit_depth_y() == 8 {
                self.eight.begin(sps, pps, &header)?;
            } else {
                self.ten.begin(sps, pps, &header)?;
            }
        } else if !self.eight.is_active() && !self.ten.is_active() {
            return Err(HevcError::InvalidSyntax {
                element: "first_slice_segment_in_pic_flag",
                value: 0,
                allowed: "1 in the first slice segment of a picture",
            });
        }
        if !header.dependent_slice_segment_flag {
            self.independent = Some(header.clone());
        }
        let data = unescape(nal);
        if self.eight.is_active() {
            self.eight.decode_slice(&header, &data)?;
        } else {
            self.ten.decode_slice(&header, &data)?;
        }
        Ok(finished)
    }
}

/// Ends the picture of whichever sample type has one.
fn finish_active(
    eight: &mut PictureDecoder<u8>,
    ten: &mut PictureDecoder<u16>,
    check_hash: bool,
) -> Result<Option<DecodedPicture>, HevcError> {
    if eight.is_active() {
        return Ok(eight.finish(check_hash)?.map(DecodedPicture::Eight));
    }
    Ok(ten.finish(check_hash)?.map(DecodedPicture::Ten))
}
