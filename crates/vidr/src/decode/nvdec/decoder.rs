//! [`NvdecDecoder`]: NVDEC behind the backend-neutral [`Decoder`] trait.

use std::collections::{HashMap, VecDeque};
use std::ffi::{c_ulong, c_void};
use std::ptr::NonNull;

use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::runtime::{Device, RuntimeClient};

use super::callbacks;
use super::ffi::{
    self, CuVideoParser, CuvidParserParams, CuvidSourceDataPacket, PKT_ENDOFSTREAM, PKT_TIMESTAMP,
    cuvid,
};
use super::input::Input;
use super::sequence::cuvid_codec;
use super::session::Session;
use crate::decode::{DecodeCodec, DecodedFrame, Decoder};
use crate::error::DecodeError;
use crate::route::FrameRoute;

/// Hardware decode on NVDEC, into numr CUDA tensors.
///
/// The decoder runs on numr's CUDA context and compute stream: it creates
/// no context and no stream. Frames stay in device memory from the decode
/// engine to the returned tensors. See [`DecodedFrame`] for the layout.
///
/// # Input
///
/// `config` is the track's `avcC`, `hvcC`, or `av1C` payload. With it,
/// `send` takes container samples and converts them with
/// [`ToAnnexB`](crate::bitstream::ToAnnexB) or
/// [`Av1ToDecoder`](crate::bitstream::av1::Av1ToDecoder). With an empty
/// `config`, `send` takes Annex-B access units or AV1 temporal units. VP9
/// ignores `config` and takes frames as stored.
///
/// # Codecs
///
/// H.264, HEVC, AV1, and VP9, 4:2:0 at 8 or 10 bits, as far as the GPU's
/// `cuvidGetDecoderCaps` reports support. The first sequence header checks
/// the stream, and an error names what the GPU or vidr refuses.
///
/// # Chroma location
///
/// NVDEC reports no chroma sample location. For AV1 the decoder reads it
/// from `av1C`. For H.264 and HEVC it reads the VUI of the config's first
/// SPS at open, and of the last SPS in each later packet, so a new
/// sequence takes effect with its first frame. See
/// [`ColorFacts::chroma_location`](crate::decode::ColorFacts::chroma_location).
///
/// # Threads
///
/// The decoder is `Send`. Every call makes numr's context current on the
/// calling thread first.
pub struct NvdecDecoder {
    /// Owned: created by `Box::leak` in `new`, freed in `drop`.
    session: NonNull<Session>,
    parser: CuVideoParser,
    input: Input,
    scratch: Vec<u8>,
    flushed: bool,
}

// SAFETY: the decoder owns its session, parser, and context lock, and no
// other handle reaches them. NVDEC objects are not bound to a thread: each
// call makes numr's context current on the calling thread, and the context
// lock serialises NVDEC's own use of the context. `&mut self` on every
// method keeps calls from two threads apart.
unsafe impl Send for NvdecDecoder {}

/// Checks that the NVDEC library loads and has every function vidr calls.
///
/// # Errors
///
/// [`DecodeError::LibraryMissing`] or [`DecodeError::SymbolMissing`].
pub fn nvdec_available() -> Result<(), DecodeError> {
    cuvid().map(|_| ())
}

fn bind(client: &CudaClient) -> Result<(), DecodeError> {
    client
        .context()
        .bind_to_thread()
        .map_err(|error| DecodeError::Cuda {
            operation: "binding numr's CUDA context to the calling thread",
            reason: error.to_string(),
        })
}

impl NvdecDecoder {
    /// Opens a decoder for `codec` on `client`'s device.
    ///
    /// `config` is the track's config record, or empty for raw input. See
    /// the type documentation.
    ///
    /// # Errors
    ///
    /// [`DecodeError::LibraryMissing`] when the NVIDIA driver is not
    /// installed, [`DecodeError::Bitstream`] for a config record that does
    /// not parse, and [`DecodeError::Driver`] or [`DecodeError::Cuda`] when
    /// the context lock or the parser cannot be created.
    pub fn new(
        client: &CudaClient,
        codec: DecodeCodec,
        config: &[u8],
    ) -> Result<Self, DecodeError> {
        let cuvid = cuvid()?;
        let (input, chroma_location) = Input::new(codec, config)?;
        bind(client)?;

        let mut ctx_lock = std::ptr::null_mut();
        let context = client.context().cu_ctx().cast::<c_void>();
        // SAFETY: `context` is numr's live context, retained by `client`,
        // and `ctx_lock` is writable.
        let code = unsafe { (cuvid.ctx_lock_create)(&mut ctx_lock, context) };
        ffi::check("cuvidCtxLockCreate", code, || {
            format!("numr's CUDA context on {}", client.device().name())
        })?;

        let session = Box::new(Session {
            cuvid,
            client: client.clone(),
            codec,
            ctx_lock,
            chroma_location,
            vp9_color: None,
            active: None,
            picture_colors: HashMap::new(),
            frames: VecDeque::new(),
            mapped: VecDeque::new(),
            error: None,
        });
        // From here `Drop` frees the session and the context lock.
        let mut decoder = Self {
            session: NonNull::from(Box::leak(session)),
            parser: std::ptr::null_mut(),
            input,
            scratch: Vec::new(),
            flushed: false,
        };

        let mut params = CuvidParserParams::zeroed();
        params.codec_type = cuvid_codec(codec);
        // The sequence callback returns the real count.
        params.max_num_decode_surfaces = 1;
        params.max_display_delay = 0;
        params.user_data = decoder.session.as_ptr().cast::<c_void>();
        params.sequence_callback = Some(callbacks::on_sequence);
        params.decode_picture = Some(callbacks::on_decode);
        params.display_picture = Some(callbacks::on_display);
        let mut parser = std::ptr::null_mut();
        // SAFETY: `params` is fully initialised with zeroed reserved fields.
        // `user_data` points to the session, which outlives the parser: drop
        // destroys the parser before it frees the session.
        let code = unsafe { (cuvid.create_video_parser)(&mut parser, &mut params) };
        ffi::check("cuvidCreateVideoParser", code, || codec.to_string())?;
        decoder.parser = parser;
        Ok(decoder)
    }

    /// The codec the decoder opened.
    pub fn codec(&self) -> DecodeCodec {
        // SAFETY: no parse runs, so nothing else borrows the session.
        unsafe { self.session.as_ref() }.codec
    }

    /// Frames decoded and not yet received.
    pub fn queued(&self) -> usize {
        // SAFETY: as in `codec`.
        unsafe { self.session.as_ref() }.frames.len()
    }

    /// Passes one packet in decode order. See [`Decoder::send`].
    ///
    /// # Errors
    ///
    /// As [`Decoder::send`]. After an error in a parser callback, the
    /// decoder returns the same error on every `send` and `flush`.
    pub fn send(&mut self, packet: &[u8], pts: i64) -> Result<(), DecodeError> {
        if self.flushed {
            return Err(DecodeError::AfterFlush { pts });
        }
        if packet.is_empty() {
            return Err(DecodeError::EmptyPacket { pts });
        }
        self.stored_error()?;
        let (session, parser) = (self.session, self.parser);
        let codec = self.codec();
        let data = self.input.filter(packet, &mut self.scratch)?;
        if let Some(location) = self.input.sps_location(codec, packet) {
            // SAFETY: no parse runs, and `&mut self` excludes other access.
            unsafe { self.session.as_mut() }.chroma_location = location;
        }
        if let Some(color) = self.input.vp9_color(codec, packet) {
            // SAFETY: as above.
            unsafe { self.session.as_mut() }.set_vp9_color(color);
        }
        let mut source = CuvidSourceDataPacket {
            flags: PKT_TIMESTAMP,
            payload_size: data.len() as c_ulong,
            payload: data.as_ptr(),
            timestamp: pts,
        };
        // SAFETY: `session` and `parser` belong to this decoder, and
        // `source` points into `data`, which lives until the call returns.
        unsafe { parse(session, parser, &mut source) }.map_err(|error| match error {
            DecodeError::Driver {
                call, code, name, ..
            } => DecodeError::Driver {
                call,
                code,
                name,
                input: format!("the {}-byte packet at pts {pts}", data.len()),
            },
            other => other,
        })
    }

    /// Ends the stream. See [`Decoder::flush`].
    ///
    /// # Errors
    ///
    /// As [`NvdecDecoder::send`].
    pub fn flush(&mut self) -> Result<(), DecodeError> {
        if self.flushed {
            return Ok(());
        }
        self.flushed = true;
        self.stored_error()?;
        let mut source = CuvidSourceDataPacket {
            flags: PKT_ENDOFSTREAM,
            payload_size: 0,
            payload: std::ptr::null(),
            timestamp: 0,
        };
        // SAFETY: as in `send`. The end-of-stream packet carries no payload.
        unsafe { parse(self.session, self.parser, &mut source) }
    }

    /// The next frame in display order. See [`Decoder::receive`].
    pub fn receive(&mut self) -> Option<DecodedFrame<CudaRuntime>> {
        // SAFETY: no parse runs, and `&mut self` excludes other access.
        unsafe { self.session.as_mut() }.frames.pop_front()
    }

    fn stored_error(&self) -> Result<(), DecodeError> {
        // SAFETY: as in `codec`.
        match &unsafe { self.session.as_ref() }.error {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }
}

/// Feeds one packet to the parser, which runs the callbacks, then unmaps
/// the surfaces whose copies have completed.
///
/// # Safety
///
/// `session` and `parser` belong to one live decoder, no reference to the
/// session exists, and `source` points to valid payload bytes.
unsafe fn parse(
    session: NonNull<Session>,
    parser: CuVideoParser,
    source: &mut CuvidSourceDataPacket,
) -> Result<(), DecodeError> {
    let cuvid = {
        // SAFETY: no other reference exists. This one ends with the block.
        let state = unsafe { session.as_ref() };
        bind(&state.client)?;
        state.cuvid
    };
    // SAFETY: `parser` is live, `source` is valid, and the callbacks reach
    // the session through the pointer the parser holds while no reference
    // to it exists here.
    let code = unsafe { (cuvid.parse_video_data)(parser, source) };
    // SAFETY: the parse has returned, so this is the only reference.
    let state = unsafe { &mut *session.as_ptr() };
    if let Some(error) = &state.error {
        return Err(error.clone());
    }
    ffi::check("cuvidParseVideoData", code, || {
        format!("a {}-byte packet", source.payload_size)
    })?;
    let unmapped = state.unmap_completed();
    if let Err(error) = &unmapped {
        state.error = Some(error.clone());
    }
    unmapped
}

impl Decoder<CudaRuntime> for NvdecDecoder {
    fn send(&mut self, packet: &[u8], pts: i64) -> Result<(), DecodeError> {
        Self::send(self, packet, pts)
    }

    fn flush(&mut self) -> Result<(), DecodeError> {
        Self::flush(self)
    }

    fn receive(&mut self) -> Option<DecodedFrame<CudaRuntime>> {
        Self::receive(self)
    }

    fn route(&self) -> FrameRoute {
        FrameRoute::Device
    }
}

impl Drop for NvdecDecoder {
    /// Unmaps every surface, destroys the decoder, then the parser, then
    /// the context lock, and frees the session. Driver errors here have no
    /// caller to reach, so drop ignores them and releases the rest.
    fn drop(&mut self) {
        let (cuvid, ctx_lock) = {
            // SAFETY: no parse runs during drop. The reference ends with
            // the block, before the parser can call back.
            let state = unsafe { self.session.as_mut() };
            let _ = bind(&state.client);
            let _ = state.release_decoder();
            (state.cuvid, state.ctx_lock)
        };
        if !self.parser.is_null() {
            // SAFETY: `parser` came from `cuvidCreateVideoParser` and is
            // destroyed once. The session it calls back into is still live.
            let _ = unsafe { (cuvid.destroy_video_parser)(self.parser) };
        }
        if !ctx_lock.is_null() {
            // SAFETY: `ctx_lock` came from `cuvidCtxLockCreate`, and the
            // decoder and parser that used it are destroyed.
            let _ = unsafe { (cuvid.ctx_lock_destroy)(ctx_lock) };
        }
        // SAFETY: `session` came from `Box::leak` in `new`, and nothing
        // reaches it after the parser is destroyed.
        drop(unsafe { Box::from_raw(self.session.as_ptr()) });
    }
}
