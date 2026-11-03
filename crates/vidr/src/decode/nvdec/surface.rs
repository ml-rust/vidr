//! The display callback: map the output surface, copy it device to device
//! into numr tensors on numr's stream, and unmap once the copy completes.
//!
//! Ordering on numr's stream: `cuvidMapVideoFrame64` runs NVDEC's
//! post-processing on the stream passed as `output_stream`, which is numr's.
//! The two `cuMemcpy2DAsync` calls follow on the same stream, then an event.
//! The surface stays mapped until that event completes, so no copy reads an
//! unmapped surface. The host waits on an event only when every output
//! surface is mapped, or when the decoder closes or changes sequence.
//! Kernels numr launches on the planes later run after the copies, on the
//! same stream.

use std::ffi::c_int;

use cudarc::driver::sys::{self, CUDA_MEMCPY2D, CUmemorytype, CUresult};
use numr::dtype::DType;
use numr::runtime::RuntimeClient;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

use super::ffi::{self, CuVideoDecoder, CuvidParserDispInfo, CuvidProcParams};
use super::sequence::OUTPUT_SURFACES;
use super::session::{Mapped, Session};
use crate::decode::{DecodedFrame, PixelFormat, frame::chroma_shape};
use crate::error::DecodeError;

/// One pitched-to-dense plane copy.
struct PlaneCopy {
    /// First row of the plane in the mapped surface.
    source_row: usize,
    destination: u64,
    width_bytes: usize,
    rows: usize,
}

impl Session {
    /// The display callback: copies picture `info.picture_index` into a new
    /// frame and queues it.
    pub(super) fn on_display(&mut self, info: &CuvidParserDispInfo) -> Result<c_int, DecodeError> {
        self.unmap_completed()?;
        while self.mapped.len() >= OUTPUT_SURFACES as usize {
            self.unmap_oldest()?;
        }
        let active = self.active.as_ref().ok_or(DecodeError::NoSequence {
            callback: "display",
        })?;
        let sequence = active.sequence;
        let decoder = active.handle;

        let (width, height) = (sequence.display.width(), sequence.display.height());
        let luma_shape = [height as usize, width as usize];
        let chroma = chroma_shape(width, height);
        let (dtype, bytes) = match sequence.format {
            PixelFormat::Nv12 => (DType::U8, 1),
            PixelFormat::P010 => (DType::U16, 2),
        };
        // Allocation goes through numr's allocator, which takes the stream's
        // capture permit itself, so it runs before this function takes one.
        let device = self.client.device();
        let allocate = |plane: &'static str, shape: [usize; 2]| {
            Tensor::<CudaRuntime>::empty(&shape, dtype, device).map_err(|error| {
                DecodeError::Allocation {
                    plane,
                    shape,
                    reason: error.to_string(),
                }
            })
        };
        let y = allocate("luma", luma_shape)?;
        let uv = allocate("chroma", chroma)?;
        let (_, target_height) = sequence.target();
        let copies = [
            PlaneCopy {
                source_row: 0,
                destination: y.ptr(),
                width_bytes: luma_shape[1] * bytes,
                rows: luma_shape[0],
            },
            PlaneCopy {
                source_row: target_height as usize,
                destination: uv.ptr(),
                width_bytes: chroma[1] * bytes,
                rows: chroma[0],
            },
        ];

        let stream = self.client.stream();
        let permit = stream.enqueue_permit().map_err(|error| DecodeError::Cuda {
            operation: "binding numr's CUDA context for the frame copy",
            reason: error.to_string(),
        })?;
        let raw_stream = stream.raw().cu_stream();

        let mut params = CuvidProcParams::zeroed();
        params.progressive_frame = info.progressive_frame;
        params.top_field_first = info.top_field_first;
        params.unpaired_field = c_int::from(info.repeat_first_field < 0);
        params.output_stream = raw_stream.cast::<std::ffi::c_void>();
        let mut pointer = 0_u64;
        let mut pitch = 0_u32;
        // SAFETY: `decoder` is live, `info.picture_index` comes from the
        // parser for it, and the out pointers and `params` are valid for the
        // call. The permit made numr's context current and holds the capture
        // lock for the enqueue on `raw_stream`.
        let code = unsafe {
            (self.cuvid.map_video_frame)(
                decoder,
                info.picture_index,
                &mut pointer,
                &mut pitch,
                &mut params,
            )
        };
        ffi::check("cuvidMapVideoFrame64", code, || {
            format!("picture {} of {}", info.picture_index, sequence.describe())
        })?;

        let copied = copies
            .iter()
            .try_for_each(|copy| enqueue_copy(raw_stream, pointer, pitch as usize, copy))
            .and_then(|()| {
                stream
                    .raw()
                    .record_event(None)
                    .map_err(|error| DecodeError::Cuda {
                        operation: "cuEventRecord after the frame copy",
                        reason: error.to_string(),
                    })
            });
        drop(permit);
        let event = match copied {
            Ok(event) => event,
            Err(error) => {
                // The surface must not unmap while an enqueued copy can still
                // read it. Waiting for the stream drains every copy.
                let waited = stream.synchronize().map_err(|wait| DecodeError::Cuda {
                    operation: "cuStreamSynchronize before unmapping a failed copy",
                    reason: wait.to_string(),
                });
                let unmapped = self.unmap(decoder, pointer);
                // The copy error is the cause. A wait or unmap error after it
                // follows from it, and the session stops at the first error.
                let _ = waited.and(unmapped);
                return Err(error);
            }
        };
        self.mapped.push_back(Mapped {
            decoder,
            pointer,
            copied: event,
        });
        self.frames.push_back(DecodedFrame::new(
            y,
            uv,
            sequence.format,
            info.timestamp,
            sequence.display,
            sequence.color,
        ));
        Ok(1)
    }

    /// Unmaps every surface whose copy has completed, oldest first.
    pub(super) fn unmap_completed(&mut self) -> Result<(), DecodeError> {
        while let Some(front) = self.mapped.front() {
            let done = front
                .copied
                .try_is_complete()
                .map_err(|error| DecodeError::Cuda {
                    operation: "cuEventQuery on a frame copy",
                    reason: error.to_string(),
                })?;
            if !done {
                break;
            }
            self.unmap_oldest()?;
        }
        Ok(())
    }

    /// Waits for every copy, then unmaps every surface. Keeps going after an
    /// error and returns the first one.
    pub(super) fn unmap_all(&mut self) -> Result<(), DecodeError> {
        let mut first = Ok(());
        while !self.mapped.is_empty() {
            let result = self.unmap_oldest();
            if first.is_ok() {
                first = result;
            }
        }
        first
    }

    /// Waits for the oldest copy, then unmaps its surface. The surface
    /// leaves the queue even on an error, so a failed wait cannot repeat.
    fn unmap_oldest(&mut self) -> Result<(), DecodeError> {
        let Some(mapped) = self.mapped.pop_front() else {
            return Ok(());
        };
        let waited = mapped
            .copied
            .synchronize()
            .map_err(|error| DecodeError::Cuda {
                operation: "cuEventSynchronize on a frame copy",
                reason: error.to_string(),
            });
        let unmapped = self.unmap(mapped.decoder, mapped.pointer);
        waited.and(unmapped)
    }

    fn unmap(&self, decoder: CuVideoDecoder, pointer: u64) -> Result<(), DecodeError> {
        // SAFETY: `pointer` came from `cuvidMapVideoFrame64` on `decoder`,
        // which `release_decoder` destroys only after this queue is empty.
        // No copy reads the surface any more. numr's context is current: the
        // callers run inside a parse, which binds it, or bind it themselves.
        let code = unsafe { (self.cuvid.unmap_video_frame)(decoder, pointer) };
        ffi::check("cuvidUnmapVideoFrame64", code, || {
            format!("surface {pointer:#x}")
        })
    }
}

/// Enqueues one plane copy from the mapped surface at `surface` with
/// `pitch` bytes per row, on `stream`.
fn enqueue_copy(
    stream: sys::CUstream,
    surface: u64,
    pitch: usize,
    copy: &PlaneCopy,
) -> Result<(), DecodeError> {
    let parameters = CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: copy.source_row,
        srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null(),
        srcDevice: surface,
        srcArray: std::ptr::null_mut(),
        srcPitch: pitch,
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(),
        dstDevice: copy.destination,
        dstArray: std::ptr::null_mut(),
        dstPitch: copy.width_bytes,
        WidthInBytes: copy.width_bytes,
        Height: copy.rows,
    };
    // SAFETY: the source rows `source_row .. source_row + rows` of `pitch`
    // bytes lie inside the mapped surface: the luma plane holds the target
    // height and the chroma plane follows it with half as many rows. Each
    // row reads `width_bytes`, at most the target width times the sample
    // size, which the pitch covers. The destination is a dense tensor of
    // `rows * width_bytes` bytes. The caller holds numr's enqueue permit.
    let result = unsafe { sys::cuMemcpy2DAsync_v2(&parameters, stream) };
    if result == CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(DecodeError::Cuda {
            operation: "cuMemcpy2DAsync from the NVDEC surface",
            reason: format!(
                "{result:?} for {} rows of {} bytes from row {} at pitch {pitch}",
                copy.rows, copy.width_bytes, copy.source_row
            ),
        })
    }
}
