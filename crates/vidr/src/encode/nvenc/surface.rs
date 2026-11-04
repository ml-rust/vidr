//! Input surfaces and output buffers.
//!
//! NVENC reads NV12 and P010 from one allocation: Y rows, then UV rows, at one
//! pitch. numr's planes are two tensors, so each frame copies device to device
//! into an encoder-owned surface. The host never touches a sample.
//!
//! The surface is a plain `cuMemAlloc` block, not a numr tensor. numr allocates
//! with `cuMemAllocAsync`, and `nvEncRegisterResource` refuses that memory with
//! `NV_ENC_ERR_RESOURCE_REGISTER_FAILED` on driver 615.71.09. The same call
//! accepts `cuMemAlloc` memory on numr's context.
//!
//! Ordering: the two `cuMemcpy2DAsync` calls run on numr's stream, after every
//! kernel already queued there, so they copy finished planes. An event follows
//! them. NVENC reads the surface on its own engine and knows nothing of numr's
//! stream, so `send` waits on that event on the host before
//! `nvEncEncodePicture`. The wait covers this frame's work and no later work,
//! which another thread's kernels cannot extend, as a stream wait can.

use cudarc::driver::sys::{self, CUDA_MEMCPY2D, CUmemorytype, CUresult};
use numr::runtime::cuda::{CudaClient, CudaRuntime};

use super::ffi::{
    BUFFER_USAGE_INPUT_IMAGE, CREATE_BITSTREAM_BUFFER_VER, MAP_INPUT_RESOURCE_VER,
    NvEncCreateBitstreamBuffer, NvEncInputPtr, NvEncMapInputResource, NvEncOutputPtr,
    NvEncRegisterResource, NvEncRegisteredPtr, REGISTER_RESOURCE_VER, RESOURCE_CUDA_DEVICE_PTR,
};
use super::params;
use super::session::Session;
use super::surface_layout::{PlanePitches, SurfaceLayout};
use crate::encode::EncodeFrame;
use crate::error::EncodeError;

/// One input surface and its output buffer.
#[derive(Debug)]
pub(super) struct Slot {
    memory: u64,
    registered: NvEncRegisteredPtr,
    pub(super) bitstream: NvEncOutputPtr,
}

impl Slot {
    /// Allocates and registers a surface, and creates a bitstream buffer.
    pub(super) fn create(session: &Session, layout: &SurfaceLayout) -> Result<Self, EncodeError> {
        let mut slot = Self {
            memory: 0,
            registered: std::ptr::null_mut(),
            bitstream: std::ptr::null_mut(),
        };
        match slot.fill(session, layout) {
            Ok(()) => Ok(slot),
            Err(error) => {
                slot.release(session);
                Err(error)
            }
        }
    }

    fn fill(&mut self, session: &Session, layout: &SurfaceLayout) -> Result<(), EncodeError> {
        let bytes = layout.size_in_bytes();
        // SAFETY: numr's context is current, and `memory` is writable.
        let result = unsafe { sys::cuMemAlloc_v2(&mut self.memory, bytes) };
        if result != CUresult::CUDA_SUCCESS {
            self.memory = 0;
            return Err(EncodeError::Allocation {
                bytes,
                reason: format!("{result:?}"),
            });
        }

        let describe = || {
            format!(
                "a {}x{} {} surface at pitch {}",
                layout.width, layout.height, layout.format, layout.pitch
            )
        };
        let mut register = NvEncRegisterResource::zeroed();
        register.version = REGISTER_RESOURCE_VER;
        register.resource_type = RESOURCE_CUDA_DEVICE_PTR;
        register.width = layout.width;
        register.height = layout.height;
        register.pitch = layout.pitch_u32();
        register.resource_to_register = self.memory as *mut std::ffi::c_void;
        register.buffer_format = params::buffer_format(layout.format);
        register.buffer_usage = BUFFER_USAGE_INPUT_IMAGE;
        // SAFETY: the session is live, `register` is fully initialised with
        // zeroed reserved fields, and the memory holds `height` luma rows and
        // the chroma rows after them at `pitch`. `release` unregisters it before
        // freeing it.
        let code = unsafe { (session.nvenc.register_resource)(session.handle, &mut register) };
        session.check("nvEncRegisterResource", code, describe)?;
        self.registered = register.registered_resource;

        let mut create = NvEncCreateBitstreamBuffer::zeroed();
        create.version = CREATE_BITSTREAM_BUFFER_VER;
        // SAFETY: the session is live, and `create` is fully initialised with
        // zeroed reserved fields.
        let code = unsafe { (session.nvenc.create_bitstream_buffer)(session.handle, &mut create) };
        session.check("nvEncCreateBitstreamBuffer", code, describe)?;
        self.bitstream = create.bitstream_buffer;
        Ok(())
    }

    /// Unregisters the surface, frees it, and destroys the bitstream buffer. No
    /// picture uses the slot any more, and numr's context must be current. A
    /// driver error here has no caller to reach, so this ignores it.
    pub(super) fn release(&mut self, session: &Session) {
        if !self.registered.is_null() {
            // SAFETY: `registered` came from `nvEncRegisterResource` on this
            // session, is not mapped, and is cleared below.
            let _ = unsafe { (session.nvenc.unregister_resource)(session.handle, self.registered) };
            self.registered = std::ptr::null_mut();
        }
        if !self.bitstream.is_null() {
            // SAFETY: `bitstream` came from `nvEncCreateBitstreamBuffer` on this
            // session, is not locked, and is cleared below.
            let _ =
                unsafe { (session.nvenc.destroy_bitstream_buffer)(session.handle, self.bitstream) };
            self.bitstream = std::ptr::null_mut();
        }
        if self.memory != 0 {
            // SAFETY: `memory` came from `cuMemAlloc_v2`, NVENC no longer holds
            // it, every copy into it has completed, and it is cleared below.
            let _ = unsafe { sys::cuMemFree_v2(self.memory) };
            self.memory = 0;
        }
    }

    /// Copies a frame into the surface on numr's stream, and waits on the host
    /// until the copy and every earlier operation on the stream complete.
    pub(super) fn copy_frame(
        &self,
        client: &CudaClient,
        layout: &SurfaceLayout,
        frame: &EncodeFrame<'_, CudaRuntime>,
        pitches: PlanePitches,
    ) -> Result<(), EncodeError> {
        let bytes = layout.format.bytes_per_sample();
        let copies = [
            PlaneCopy {
                source: frame.y().ptr(),
                source_pitch: pitches.luma,
                destination: self.memory,
                destination_row: 0,
                destination_pitch: layout.pitch,
                width_bytes: layout.luma[1] * bytes,
                rows: layout.luma[0],
            },
            PlaneCopy {
                source: frame.uv().ptr(),
                source_pitch: pitches.chroma,
                destination: self.memory,
                destination_row: layout.luma[0],
                destination_pitch: layout.pitch,
                width_bytes: layout.chroma[1] * bytes,
                rows: layout.chroma[0],
            },
        ];

        let stream = client.stream();
        let permit = stream.enqueue_permit().map_err(|error| EncodeError::Cuda {
            operation: "binding numr's CUDA context for the frame copy",
            reason: error.to_string(),
        })?;
        let raw_stream = stream.raw().cu_stream();
        let copied = copies
            .iter()
            .try_for_each(|copy| enqueue_copy(raw_stream, copy))
            .and_then(|()| {
                stream
                    .raw()
                    .record_event(None)
                    .map_err(|error| EncodeError::Cuda {
                        operation: "cuEventRecord after the frame copy",
                        reason: error.to_string(),
                    })
            });
        drop(permit);
        match copied {
            Ok(event) => event.synchronize().map_err(|error| EncodeError::Cuda {
                operation: "cuEventSynchronize on the frame copy",
                reason: error.to_string(),
            }),
            Err(error) => {
                // The frame's planes must outlive any copy already enqueued.
                // Waiting for the stream drains it. The copy error is the cause,
                // so a wait error after it is not reported.
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// Maps the surface for one picture.
    pub(super) fn map(&self, session: &Session) -> Result<NvEncInputPtr, EncodeError> {
        let mut map = NvEncMapInputResource::zeroed();
        map.version = MAP_INPUT_RESOURCE_VER;
        map.registered_resource = self.registered;
        // SAFETY: the session is live, `registered` belongs to it and is not
        // mapped, and `map` is fully initialised with zeroed reserved fields.
        let code = unsafe { (session.nvenc.map_input_resource)(session.handle, &mut map) };
        session.check("nvEncMapInputResource", code, || {
            format!("surface {:#x}", self.memory)
        })?;
        Ok(map.mapped_resource)
    }
}

/// Unmaps a surface after the encoder has finished with it.
pub(super) fn unmap(session: &Session, mapped: NvEncInputPtr) -> Result<(), EncodeError> {
    // SAFETY: `mapped` came from `nvEncMapInputResource` on this session, and
    // the bitstream of its picture has been locked, so the encoder is done.
    let code = unsafe { (session.nvenc.unmap_input_resource)(session.handle, mapped) };
    session.check("nvEncUnmapInputResource", code, || {
        format!("mapped surface {mapped:?}")
    })
}

/// One plane copy into a surface.
struct PlaneCopy {
    source: u64,
    source_pitch: usize,
    destination: u64,
    /// First row of the plane in the surface.
    destination_row: usize,
    destination_pitch: usize,
    width_bytes: usize,
    rows: usize,
}

fn enqueue_copy(stream: sys::CUstream, copy: &PlaneCopy) -> Result<(), EncodeError> {
    let parameters = CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null(),
        srcDevice: copy.source,
        srcArray: std::ptr::null_mut(),
        srcPitch: copy.source_pitch,
        dstXInBytes: 0,
        dstY: copy.destination_row,
        dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(),
        dstDevice: copy.destination,
        dstArray: std::ptr::null_mut(),
        dstPitch: copy.destination_pitch,
        WidthInBytes: copy.width_bytes,
        Height: copy.rows,
    };
    // SAFETY: `check_frame` proved the source holds `rows` rows of `width_bytes`
    // at `source_pitch`. The surface holds `destination_row + rows` rows of
    // `destination_pitch`, which covers `width_bytes`. The caller holds numr's
    // enqueue permit and keeps the source alive until the copy completes.
    let result = unsafe { sys::cuMemcpy2DAsync_v2(&parameters, stream) };
    if result == CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(EncodeError::Cuda {
            operation: "cuMemcpy2DAsync into an NVENC surface",
            reason: format!(
                "{result:?} for {} rows of {} bytes to row {} at pitch {}",
                copy.rows, copy.width_bytes, copy.destination_row, copy.destination_pitch
            ),
        })
    }
}
