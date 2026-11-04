//! [`NvencEncoder`]: NVENC behind the backend-neutral [`Encoder`] trait.

use std::collections::VecDeque;

use numr::runtime::cuda::{CudaClient, CudaRuntime};

use super::ffi::{
    NEED_MORE_INPUT, NvEncInputPtr, NvEncPicParams, PIC_FLAG_EOS, PIC_PARAMS_VER, PIC_STRUCT_FRAME,
    nvenc,
};
use super::lock::{discard_output, read_output};
use super::params;
use super::session::Session;
use super::surface::{Slot, unmap};
use super::surface_layout::{PlanePitches, SurfaceLayout};
use crate::encode::dts::DtsClock;
use crate::encode::sample_filter::SampleFilter;
use crate::encode::{EncodeCodec, EncodeConfig, EncodeFrame, EncodedPacket, Encoder};
use crate::error::EncodeError;
use crate::route::FrameRoute;
use crate::time::Rational;

/// A picture the encoder holds: its slot and its mapped surface.
struct Pending {
    slot: usize,
    mapped: NvEncInputPtr,
}

/// Hardware encode on NVENC, from numr CUDA tensors.
///
/// The encoder runs on numr's CUDA context and compute stream: it creates no
/// context and no stream. Samples stay in device memory from the frame to the
/// encode engine. Only the compressed bitstream reaches the host.
///
/// # Calls
///
/// NVENC runs in synchronous mode. `send` waits for numr's stream to reach this
/// frame's copy, and `send` and `flush` wait for each picture the encoder
/// finishes, so an error returns from the call that hits it and `receive` never
/// blocks. After a driver error, every later `send` and `flush` returns the same
/// error. A refused frame leaves the encoder usable.
///
/// # Output
///
/// Packets are MP4 and Matroska samples in decode order. See [`EncodedPacket`].
/// [`NvencEncoder::config_record`] returns the `avcC`, `hvcC`, or `av1C` payload
/// once the first packet exists.
///
/// # Threads
///
/// The encoder is `Send`. Every call makes numr's context current on the calling
/// thread first.
pub struct NvencEncoder {
    config: EncodeConfig,
    layout: SurfaceLayout,
    slots: Vec<Slot>,
    free: Vec<usize>,
    pending: VecDeque<Pending>,
    clock: DtsClock,
    filter: SampleFilter,
    config_record: Option<Vec<u8>>,
    packets: VecDeque<EncodedPacket>,
    frame_index: u32,
    last_pts: Option<i64>,
    flushed: bool,
    error: Option<EncodeError>,
    /// Dropped after `Drop::drop` releases the slots, so the encoder goes last.
    session: Session,
}

// SAFETY: the encoder owns its session, surfaces, and buffers, and no other
// handle reaches them. NVENC objects are not bound to a thread: each call makes
// numr's context current on the calling thread. `&mut self` on every method that
// touches them keeps calls from two threads apart.
unsafe impl Send for NvencEncoder {}

/// Checks that the NVENC library loads, has every function vidr calls, and
/// serves the pinned API version.
///
/// # Errors
///
/// [`EncodeError::LibraryMissing`], [`EncodeError::SymbolMissing`], or
/// [`EncodeError::DriverTooOld`].
pub fn nvenc_available() -> Result<(), EncodeError> {
    nvenc().map(|_| ())
}

impl NvencEncoder {
    /// Opens an encoder on `client`'s device.
    ///
    /// # Errors
    ///
    /// - [`EncodeError::OddSize`] or [`EncodeError::InvalidConfig`] for a config
    ///   no GPU can take.
    /// - [`EncodeError::LibraryMissing`], [`EncodeError::SymbolMissing`], or
    ///   [`EncodeError::DriverTooOld`] for the driver.
    /// - [`EncodeError::CodecUnsupported`], [`EncodeError::BitDepthUnsupported`],
    ///   [`EncodeError::BFramesUnsupported`], or [`EncodeError::SizeOutOfRange`]
    ///   for what this GPU refuses.
    /// - [`EncodeError::Nvenc`], [`EncodeError::Cuda`], or
    ///   [`EncodeError::Allocation`] when a driver call fails.
    pub fn new(client: &CudaClient, config: EncodeConfig) -> Result<Self, EncodeError> {
        config.validate()?;
        let session = Session::open(nvenc()?, client)?;
        session.check_support(&config)?;
        session.initialize(&config)?;
        let mut encoder = Self {
            layout: SurfaceLayout::new(&config),
            slots: Vec::new(),
            free: Vec::new(),
            pending: VecDeque::new(),
            clock: DtsClock::new(config.b_frames),
            filter: SampleFilter::new(config.codec),
            config_record: None,
            packets: VecDeque::new(),
            frame_index: 0,
            last_pts: None,
            flushed: false,
            error: None,
            session,
            config,
        };
        // The encoder holds at most `b_frames + 1` pictures in one reorder group.
        for _ in 0..=encoder.config.b_frames {
            let slot = encoder.create_slot()?;
            encoder.free.push(slot);
        }
        Ok(encoder)
    }

    /// The codec the encoder opened.
    pub const fn codec(&self) -> EncodeCodec {
        self.config.codec
    }

    /// The unit of the packets' timestamps, as the config gave it.
    pub const fn time_base(&self) -> Rational {
        self.config.time_base
    }

    /// The config the encoder opened with.
    pub const fn config(&self) -> &EncodeConfig {
        &self.config
    }

    /// Passes one frame in display order. See [`Encoder::send`].
    ///
    /// # Errors
    ///
    /// - [`EncodeError::AfterFlush`], [`EncodeError::NonPositiveDuration`], or
    ///   [`EncodeError::PtsNotIncreasing`] for the call.
    /// - [`EncodeError::SurfaceMismatch`] or [`EncodeError::PlaneStrides`] for
    ///   the planes.
    /// - The errors of [`NvencEncoder::flush`] for the encode itself.
    pub fn send(
        &mut self,
        frame: &EncodeFrame<'_, CudaRuntime>,
        pts: i64,
        duration: i64,
    ) -> Result<(), EncodeError> {
        if self.flushed {
            return Err(EncodeError::AfterFlush { pts });
        }
        self.stored_error()?;
        if duration <= 0 {
            return Err(EncodeError::NonPositiveDuration { pts, duration });
        }
        if let Some(previous) = self.last_pts
            && pts <= previous
        {
            return Err(EncodeError::PtsNotIncreasing { pts, previous });
        }
        let pitches = self.layout.check_frame(frame)?;
        let result = self.submit(frame, pitches, pts, duration);
        self.keep_error(result)
    }

    /// Ends the stream. See [`Encoder::flush`].
    ///
    /// # Errors
    ///
    /// [`EncodeError::Nvenc`], [`EncodeError::Cuda`], or
    /// [`EncodeError::Bitstream`] from the step that fails, and
    /// [`EncodeError::DtsAfterPts`], [`EncodeError::UnexpectedOutput`], or
    /// [`EncodeError::TimestampOverflow`] from the timestamps.
    pub fn flush(&mut self) -> Result<(), EncodeError> {
        if self.flushed {
            return Ok(());
        }
        self.flushed = true;
        self.stored_error()?;
        let result = self.end_of_stream();
        self.keep_error(result)
    }

    /// The next packet in decode order. See [`Encoder::receive`].
    pub fn receive(&mut self) -> Option<EncodedPacket> {
        self.packets.pop_front()
    }

    /// The `avcC`, `hvcC`, or `av1C` box payload, or `None` before the first
    /// packet exists. See [`Encoder::config_record`].
    pub fn config_record(&self) -> Option<&[u8]> {
        self.config_record.as_deref()
    }

    fn stored_error(&self) -> Result<(), EncodeError> {
        match &self.error {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    fn keep_error(&mut self, result: Result<(), EncodeError>) -> Result<(), EncodeError> {
        if let Err(error) = &result {
            self.error = Some(error.clone());
        }
        result
    }

    fn create_slot(&mut self) -> Result<usize, EncodeError> {
        self.session.bind()?;
        let slot = Slot::create(&self.session, &self.layout)?;
        self.slots.push(slot);
        Ok(self.slots.len() - 1)
    }

    fn submit(
        &mut self,
        frame: &EncodeFrame<'_, CudaRuntime>,
        pitches: PlanePitches,
        pts: i64,
        duration: i64,
    ) -> Result<(), EncodeError> {
        self.session.bind()?;
        // A driver that holds more pictures than one reorder group gets another
        // slot instead of an error.
        let index = match self.free.pop() {
            Some(index) => index,
            None => self.create_slot()?,
        };
        let slot = &self.slots[index];
        if let Err(error) = slot.copy_frame(&self.session.client, &self.layout, frame, pitches) {
            self.free.push(index);
            return Err(error);
        }
        let mapped = match slot.map(&self.session) {
            Ok(mapped) => mapped,
            Err(error) => {
                self.free.push(index);
                return Err(error);
            }
        };
        self.pending.push_back(Pending {
            slot: index,
            mapped,
        });

        let mut picture = NvEncPicParams::zeroed();
        picture.version = PIC_PARAMS_VER;
        picture.input_width = self.layout.width;
        picture.input_height = self.layout.height;
        picture.input_pitch = self.layout.pitch_u32();
        picture.frame_idx = self.frame_index;
        // Opaque to NVENC. The bits come back unchanged in the lock, so a
        // negative pts survives.
        picture.input_time_stamp = pts as u64;
        picture.input_duration = duration as u64;
        picture.input_buffer = mapped;
        picture.output_bitstream = slot.bitstream;
        picture.buffer_fmt = params::buffer_format(self.layout.format);
        picture.picture_struct = PIC_STRUCT_FRAME;
        self.frame_index = self.frame_index.wrapping_add(1);
        self.last_pts = Some(pts);
        self.clock.push(pts, duration);
        self.encode(&mut picture, || format!("the frame at pts {pts}"))
    }

    fn end_of_stream(&mut self) -> Result<(), EncodeError> {
        self.session.bind()?;
        let mut picture = NvEncPicParams::zeroed();
        picture.version = PIC_PARAMS_VER;
        picture.encode_pic_flags = PIC_FLAG_EOS;
        self.encode(&mut picture, || "the end of the stream".to_owned())
    }

    /// Runs `nvEncEncodePicture`, then takes every finished picture.
    fn encode(
        &mut self,
        picture: &mut NvEncPicParams,
        input: impl FnOnce() -> String,
    ) -> Result<(), EncodeError> {
        // SAFETY: the session is live, `picture` is fully initialised with zeroed
        // reserved fields, and any surface and buffer it names belong to this
        // session and are in `pending`. The surface's copy has completed, and
        // the surface stays mapped until the lock of its picture returns. numr's
        // context is current.
        let code = unsafe { (self.session.nvenc.encode_picture)(self.session.handle, picture) };
        if code == NEED_MORE_INPUT {
            return Ok(());
        }
        if let Err(error) = self.session.check("nvEncEncodePicture", code, input) {
            // A refused frame never reached the encoder, so its output buffer
            // holds no picture. Drop must not lock that buffer.
            if !picture.input_buffer.is_null()
                && let Some(refused) = self.pending.pop_back()
            {
                let _ = unmap(&self.session, refused.mapped);
                self.free.push(refused.slot);
            }
            return Err(error);
        }
        self.take_finished()
    }

    /// Locks each pending picture in submission order, which is decode order,
    /// and queues its packet. `nvEncLockBitstream` waits for the picture.
    fn take_finished(&mut self) -> Result<(), EncodeError> {
        while let Some(pending) = self.pending.pop_front() {
            let packet = self.take_output(pending.slot);
            let unmapped = unmap(&self.session, pending.mapped);
            self.free.push(pending.slot);
            let packet = packet?;
            unmapped?;
            self.packets.push_back(packet);
        }
        Ok(())
    }

    fn take_output(&mut self, slot: usize) -> Result<EncodedPacket, EncodeError> {
        let bitstream = self.slots[slot].bitstream;
        let filter = &mut self.filter;
        let (sample, facts) =
            read_output(&self.session, bitstream, |output| filter.sample(output))?;
        let data = sample?;
        if self.config_record.is_none() {
            self.config_record = self.filter.config_record()?;
        }
        Ok(EncodedPacket {
            data,
            pts: facts.pts,
            dts: self.clock.next(facts.pts)?,
            duration: facts.duration,
            keyframe: facts.keyframe,
        })
    }
}

impl Encoder<CudaRuntime> for NvencEncoder {
    fn send(
        &mut self,
        frame: &EncodeFrame<'_, CudaRuntime>,
        pts: i64,
        duration: i64,
    ) -> Result<(), EncodeError> {
        Self::send(self, frame, pts, duration)
    }

    fn flush(&mut self) -> Result<(), EncodeError> {
        Self::flush(self)
    }

    fn receive(&mut self) -> Option<EncodedPacket> {
        Self::receive(self)
    }

    fn config_record(&self) -> Option<&[u8]> {
        Self::config_record(self)
    }

    fn route(&self) -> FrameRoute {
        FrameRoute::Device
    }
}

impl Drop for NvencEncoder {
    /// Ends the stream, then unmaps each surface, unregisters it, destroys its
    /// buffer, and frees it. The session's own drop then destroys the encoder.
    /// Driver errors here have no caller to reach, so drop ignores them and
    /// releases the rest.
    fn drop(&mut self) {
        let _ = self.session.bind();
        if !self.flushed {
            let mut picture = NvEncPicParams::zeroed();
            picture.version = PIC_PARAMS_VER;
            picture.encode_pic_flags = PIC_FLAG_EOS;
            // SAFETY: as in `encode`. The end of the stream names no surface.
            let _ =
                unsafe { (self.session.nvenc.encode_picture)(self.session.handle, &mut picture) };
        }
        while let Some(pending) = self.pending.pop_front() {
            // The wait means the encoder is done with the picture, so the unmap
            // and the free below cannot pull a surface from under it.
            discard_output(&self.session, self.slots[pending.slot].bitstream);
            let _ = unmap(&self.session, pending.mapped);
        }
        for slot in &mut self.slots {
            slot.release(&self.session);
        }
    }
}
