//! The state the parser callbacks share: the hardware decoder, the frames
//! ready in display order, and the surfaces still mapped.

use std::collections::{HashMap, VecDeque};
use std::ffi::c_int;

use cudarc::driver::CudaEvent;
use numr::runtime::cuda::{CudaClient, CudaRuntime};

use super::ffi::{self, CuVideoCtxLock, CuVideoDecoder, CuVideoFormat, Cuvid, CuvidPicParams};
use super::sequence::Sequence;
use super::vp9_color;
use crate::bitstream::Vp9Color;
use crate::decode::{ChromaLocation, ColorFacts, DecodeCodec, DecodedFrame};
use crate::error::DecodeError;

/// A hardware decoder and the sequence it was created for.
pub(super) struct Active {
    pub(super) handle: CuVideoDecoder,
    pub(super) sequence: Sequence,
}

/// An output surface mapped by `cuvidMapVideoFrame64`, and the event
/// recorded on numr's stream after the copies out of it.
pub(super) struct Mapped {
    pub(super) decoder: CuVideoDecoder,
    pub(super) pointer: u64,
    pub(super) copied: CudaEvent,
}

/// The parser's user data.
///
/// The parser calls back only from inside `cuvidParseVideoData`, on the
/// thread that calls it. `NvdecDecoder` reaches the session through the same
/// raw pointer it gives the parser, and never while a parse runs, so at most
/// one `&mut Session` exists at a time.
pub(super) struct Session {
    pub(super) cuvid: &'static Cuvid,
    pub(super) client: CudaClient,
    pub(super) codec: DecodeCodec,
    pub(super) ctx_lock: CuVideoCtxLock,
    pub(super) chroma_location: Option<ChromaLocation>,
    /// The colour of the latest VP9 key frame or intra-only header, or
    /// `None` before one is parsed.
    pub(super) vp9_color: Option<Vp9Color>,
    pub(super) active: Option<Active>,
    /// The colour each picture index had when it was decoded. A picture can wait
    /// for display past a colour change, so display reads its colour here.
    pub(super) picture_colors: HashMap<c_int, ColorFacts>,
    pub(super) frames: VecDeque<DecodedFrame<CudaRuntime>>,
    pub(super) mapped: VecDeque<Mapped>,
    /// The first error a callback hits. Every later callback returns 0, and
    /// `send` or `flush` returns this error.
    pub(super) error: Option<DecodeError>,
}

impl Session {
    /// The sequence callback: checks the stream against the GPU, keeps the
    /// decoder when it still fits, and otherwise replaces it. Returns the
    /// decode surface count, which overrides the parser's.
    pub(super) fn on_sequence(&mut self, format: &CuVideoFormat) -> Result<c_int, DecodeError> {
        let sequence = Sequence::read(self.codec, format, self.chroma_location, self.vp9_color)?;
        let surfaces = c_int::try_from(sequence.decode_surfaces).unwrap_or(c_int::MAX);
        if let Some(active) = &mut self.active
            && active.sequence.same_decoder(&sequence)
        {
            active.sequence.color = sequence.color;
            active.sequence.display = sequence.display;
            return Ok(surfaces);
        }

        let mut caps = sequence.caps_query();
        // SAFETY: `caps` is a valid, writable `CUVIDDECODECAPS` with its
        // input fields set. The caller made numr's context current.
        let code = unsafe { (self.cuvid.get_decoder_caps)(&mut caps) };
        ffi::check("cuvidGetDecoderCaps", code, || sequence.describe())?;
        sequence.check_caps(&caps)?;

        self.release_decoder()?;
        let mut info = sequence.create_info(self.ctx_lock);
        let mut handle: CuVideoDecoder = std::ptr::null_mut();
        // SAFETY: `info` is a fully initialised `CUVIDDECODECREATEINFO` with
        // zeroed reserved fields, `ctx_lock` belongs to numr's context, and
        // `handle` is writable. The caller made numr's context current.
        let code = unsafe { (self.cuvid.create_decoder)(&mut handle, &mut info) };
        ffi::check("cuvidCreateDecoder", code, || sequence.describe())?;
        self.active = Some(Active { handle, sequence });
        Ok(surfaces)
    }

    /// Keeps the colour of a VP9 key frame header. A key frame of the same
    /// size starts no new sequence, so the active sequence takes it too.
    pub(super) fn set_vp9_color(&mut self, color: Vp9Color) {
        self.vp9_color = Some(color);
        if let Some(active) = &mut self.active {
            vp9_color::apply(&mut active.sequence.color, Some(color));
        }
    }

    /// The decode callback: submits one picture to the hardware decoder.
    pub(super) fn on_decode(&mut self, picture: *mut CuvidPicParams) -> Result<c_int, DecodeError> {
        let active = self
            .active
            .as_ref()
            .ok_or(DecodeError::NoSequence { callback: "decode" })?;
        // SAFETY: the parser owns `picture` for the duration of this
        // callback and filled it for `active.handle`, the decoder its last
        // sequence callback created.
        let index = unsafe { (*picture).curr_pic_idx };
        self.picture_colors.insert(index, active.sequence.color);
        // SAFETY: as above.
        let code = unsafe { (self.cuvid.decode_picture)(active.handle, picture) };
        ffi::check("cuvidDecodePicture", code, || active.sequence.describe())?;
        Ok(1)
    }

    /// Unmaps every mapped surface, then destroys the decoder. Frames
    /// already queued keep their own copies.
    pub(super) fn release_decoder(&mut self) -> Result<(), DecodeError> {
        let unmapped = self.unmap_all();
        self.picture_colors.clear();
        let destroyed = match self.active.take() {
            Some(active) => {
                // SAFETY: `active.handle` came from `cuvidCreateDecoder`, no
                // surface of it is mapped any more, and it is destroyed once
                // because `take` cleared it.
                let code = unsafe { (self.cuvid.destroy_decoder)(active.handle) };
                ffi::check("cuvidDestroyDecoder", code, || active.sequence.describe())
            }
            None => Ok(()),
        };
        unmapped.and(destroyed)
    }
}
