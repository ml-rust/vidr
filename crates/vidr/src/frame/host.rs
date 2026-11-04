//! Copies of 4:2:0 planes between host memory and numr tensors, for a CPU codec.
//!
//! This is a host copy by definition. The codec declares it through
//! [`FrameRoute`](crate::route::FrameRoute). This module is the only place
//! outside the test code that reads frame bytes to the host.
//!
//! The host side is planar: I420 for [`PixelFormat::Nv12`], I010 for
//! [`PixelFormat::P010`]. An I010 sample holds its value in the low 10 bits of
//! two little-endian bytes. The tensor side holds the planes `DecodedFrame`
//! and `EncodeFrame` document, in the form [`Runtime::packed_color_planes`]
//! asks for.
//!
//! For a display size of `W x H`, each chroma plane has `ceil(W / 2)` samples
//! per row and `ceil(H / 2)` rows, so odd sizes round up.

use std::fmt;

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::convert::{check_plane, numr_error, plane_shapes};
use crate::decode::{PixelFormat, PlaneForm};
use crate::encode::EncodeFrame;
use crate::error::FrameError;
use crate::geometry::Dimensions;

/// The largest I010 sample value.
const I010_MAX: u16 = 1023;
/// P010 stores `value << 6`.
const P010_SHIFT: u32 = 6;

/// Borrowed I420 or I010 planes with their row strides, as a CPU decoder
/// returns them.
///
/// Each stride is in bytes from one row's start to the next. It is at least the
/// plane's row width in bytes. Bytes between rows are not read. The last row
/// needs no bytes past its width.
#[derive(Clone, Copy)]
pub struct HostPlanes<'a> {
    /// Luma, `height` rows of `width` samples.
    pub y: &'a [u8],
    /// Bytes from one luma row to the next.
    pub y_stride: usize,
    /// Cb, `ceil(height / 2)` rows of `ceil(width / 2)` samples.
    pub u: &'a [u8],
    /// Bytes from one Cb row to the next.
    pub u_stride: usize,
    /// Cr, the shape of Cb.
    pub v: &'a [u8],
    /// Bytes from one Cr row to the next.
    pub v_stride: usize,
    /// Display width in pixels.
    pub width: usize,
    /// Display height in pixels.
    pub height: usize,
}

// A manual impl: a derive prints every byte of each plane.
impl fmt::Debug for HostPlanes<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostPlanes")
            .field("y_bytes", &self.y.len())
            .field("y_stride", &self.y_stride)
            .field("u_bytes", &self.u.len())
            .field("u_stride", &self.u_stride)
            .field("v_bytes", &self.v.len())
            .field("v_stride", &self.v_stride)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

/// Owned tight I420 or I010 planes, as [`download_planes`] writes them.
///
/// Keep one per encoder. A frame of the same size allocates nothing on the host
/// after the first.
#[derive(Clone)]
pub struct HostFrame {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    width: usize,
    height: usize,
    format: PixelFormat,
    /// The bytes of one plane as the device holds them.
    scratch: Vec<u8>,
}

impl HostFrame {
    /// An empty frame. The first [`download_planes`] allocates its buffers.
    pub const fn new() -> Self {
        Self {
            y: Vec::new(),
            u: Vec::new(),
            v: Vec::new(),
            width: 0,
            height: 0,
            format: PixelFormat::Nv12,
            scratch: Vec::new(),
        }
    }

    /// Tight luma.
    pub fn y(&self) -> &[u8] {
        &self.y
    }

    /// Tight Cb.
    pub fn u(&self) -> &[u8] {
        &self.u
    }

    /// Tight Cr.
    pub fn v(&self) -> &[u8] {
        &self.v
    }

    /// Display width in pixels, or 0 before the first download.
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Display height in pixels, or 0 before the first download.
    pub const fn height(&self) -> usize {
        self.height
    }

    /// `Nv12` for I420 planes, `P010` for I010 planes.
    pub const fn format(&self) -> PixelFormat {
        self.format
    }

    /// The planes with their tight strides.
    pub fn planes(&self) -> HostPlanes<'_> {
        let sample = self.format.bytes_per_sample();
        let chroma_stride = self.width.div_ceil(2) * sample;
        HostPlanes {
            y: &self.y,
            y_stride: self.width * sample,
            u: &self.u,
            u_stride: chroma_stride,
            v: &self.v,
            v_stride: chroma_stride,
            width: self.width,
            height: self.height,
        }
    }
}

impl Default for HostFrame {
    fn default() -> Self {
        Self::new()
    }
}

// A manual impl: a derive prints every byte of each plane.
impl fmt::Debug for HostFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostFrame")
            .field("y_bytes", &self.y.len())
            .field("u_bytes", &self.u.len())
            .field("v_bytes", &self.v.len())
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format)
            .finish()
    }
}

/// Copies I420 or I010 planes to new NV12 or P010 tensors on `device`.
///
/// Returns `(y, uv)`, both dense. When [`Runtime::packed_color_planes`] is true
/// they are U32 words with 0 past each row's data, for
/// `DecodedFrame::from_packed_planes`. Otherwise they are U8 or U16 samples, for
/// `DecodedFrame::from_planes`. `U, V` interleave, and P010 stores `value << 6`.
///
/// # Errors
///
/// - [`FrameError::HostFrameSize`] for a zero or oversized display size.
/// - [`FrameError::HostStride`] for a stride below the row width.
/// - [`FrameError::HostPlaneShort`] for a plane that ends before its last row.
/// - [`FrameError::HostSampleRange`] for an I010 value above 1023.
/// - [`FrameError::Numr`] or [`FrameError::OutOfMemory`] when the upload fails.
pub fn upload_planes<R>(
    device: &R::Device,
    planes: HostPlanes<'_>,
    format: PixelFormat,
) -> Result<(Tensor<R>, Tensor<R>), FrameError>
where
    R: Runtime<DType = DType>,
{
    let form = if R::packed_color_planes() {
        PlaneForm::Words
    } else {
        PlaneForm::Samples
    };
    upload_in_form(device, planes, format, form)
}

/// Copies a frame's planes to `out` as tight I420 or I010, reversing
/// [`upload_planes`].
///
/// `U, V` deinterleave, and P010 stores back `raw >> 6`, which drops the low six
/// bits that P010 holds as zero. Dense planes read straight into `out`'s
/// buffers. A plane view with an offset on a backend whose handles take no byte
/// offset, or a view with other strides, first copies to new device memory.
///
/// `out` holds no complete frame after an error.
///
/// # Errors
///
/// - [`FrameError::PlaneMismatch`] for a plane that does not match the frame.
/// - [`FrameError::HostFrameSize`] for a sample-form luma plane of zero size.
/// - [`FrameError::Numr`] or [`FrameError::OutOfMemory`] when the read fails.
pub fn download_planes<R>(frame: &EncodeFrame<'_, R>, out: &mut HostFrame) -> Result<(), FrameError>
where
    R: Runtime<DType = DType>,
{
    let (format, display) = frame_layout(frame)?;
    let (luma, chroma, dtype) = plane_shapes::<R>(frame.plane_form(), format, display);
    check_plane("luma", frame.y(), luma, dtype)?;
    check_plane("chroma", frame.uv(), chroma, dtype)?;
    let layout = HostLayout::new(format, display.width() as usize, display.height() as usize)?;
    let chroma_bytes = layout.chroma_height * layout.chroma_row;
    out.y.resize(layout.height * layout.luma_row, 0);
    out.u.resize(chroma_bytes, 0);
    out.v.resize(chroma_bytes, 0);
    out.width = layout.width;
    out.height = layout.height;
    out.format = format;

    let pitch = read_plane(frame.y(), &mut out.scratch)?;
    let rows = out.scratch.chunks(pitch);
    for (source, row) in rows.zip(out.y.chunks_exact_mut(layout.luma_row)) {
        unpack_luma(format, &source[..layout.luma_row], row);
    }

    let pitch = read_plane(frame.uv(), &mut out.scratch)?;
    let targets = out
        .u
        .chunks_exact_mut(layout.chroma_row)
        .zip(out.v.chunks_exact_mut(layout.chroma_row));
    for (source, (u, v)) in out.scratch.chunks(pitch).zip(targets) {
        unpack_chroma(format, &source[..2 * layout.chroma_row], u, v);
    }
    Ok(())
}

/// The sizes of one frame's host planes.
struct HostLayout {
    width: usize,
    height: usize,
    chroma_height: usize,
    /// Bytes of one luma row.
    luma_row: usize,
    /// Bytes of one Cb or Cr row.
    chroma_row: usize,
}

impl HostLayout {
    fn new(format: PixelFormat, width: usize, height: usize) -> Result<Self, FrameError> {
        let sample = format.bytes_per_sample();
        let chroma_width = width.div_ceil(2);
        let chroma_height = height.div_ceil(2);
        let too_large = || FrameError::HostFrameSize { width, height };
        let luma_row = width.checked_mul(sample).ok_or_else(too_large)?;
        // Interleaved chroma rows are twice a Cb row, so this bounds them too.
        let chroma_row = chroma_width
            .checked_mul(2 * sample)
            .map(|row| row / 2)
            .ok_or_else(too_large)?;
        Ok(Self {
            width,
            height,
            chroma_height,
            luma_row,
            chroma_row,
        })
    }
}

/// The display size as [`Dimensions`], which refuses a zero side.
fn host_display(width: usize, height: usize) -> Result<Dimensions, FrameError> {
    u32::try_from(width)
        .ok()
        .zip(u32::try_from(height).ok())
        .and_then(|(w, h)| Dimensions::new(w, h).ok())
        .ok_or(FrameError::HostFrameSize { width, height })
}

/// Checks that `data` holds `rows` rows of `row_bytes` bytes at `stride`.
fn check_host_plane(
    plane: &'static str,
    data: &[u8],
    stride: usize,
    rows: usize,
    row_bytes: usize,
) -> Result<(), FrameError> {
    if stride < row_bytes {
        return Err(FrameError::HostStride {
            plane,
            stride,
            row_bytes,
        });
    }
    let needed = (rows - 1)
        .checked_mul(stride)
        .and_then(|before_last| before_last.checked_add(row_bytes))
        .unwrap_or(usize::MAX);
    if data.len() < needed {
        return Err(FrameError::HostPlaneShort {
            plane,
            needed,
            found: data.len(),
        });
    }
    Ok(())
}

/// [`upload_planes`] in a named form. A unit test passes `Words` on the CPU
/// backend, so the word path runs on a build with no GPU.
fn upload_in_form<R>(
    device: &R::Device,
    planes: HostPlanes<'_>,
    format: PixelFormat,
    form: PlaneForm,
) -> Result<(Tensor<R>, Tensor<R>), FrameError>
where
    R: Runtime<DType = DType>,
{
    let display = host_display(planes.width, planes.height)?;
    let layout = HostLayout::new(format, planes.width, planes.height)?;
    check_host_plane(
        "y",
        planes.y,
        planes.y_stride,
        layout.height,
        layout.luma_row,
    )?;
    check_host_plane(
        "u",
        planes.u,
        planes.u_stride,
        layout.chroma_height,
        layout.chroma_row,
    )?;
    check_host_plane(
        "v",
        planes.v,
        planes.v_stride,
        layout.chroma_height,
        layout.chroma_row,
    )?;

    let (luma_shape, chroma_shape, dtype) = plane_shapes::<R>(form, format, display);
    let element = dtype.size_in_bytes();
    let y = pack_luma(&planes, &layout, format, luma_shape[1] * element)?;
    let uv = pack_chroma(&planes, &layout, format, chroma_shape[1] * element)?;
    let upload = |bytes: &[u8], shape: [usize; 2]| {
        Tensor::<R>::from_bytes(bytes, &shape, dtype, device).map_err(|error| {
            numr_error(
                "Tensor::from_bytes",
                format!("{format} plane of shape {shape:?} in {dtype:?}"),
                &error,
            )
        })
    };
    Ok((upload(&y, luma_shape)?, upload(&uv, chroma_shape)?))
}

/// Luma rows at `pitch` bytes, with 0 past each row's data.
fn pack_luma(
    planes: &HostPlanes<'_>,
    layout: &HostLayout,
    format: PixelFormat,
    pitch: usize,
) -> Result<Vec<u8>, FrameError> {
    let mut out = vec![0_u8; layout.height * pitch];
    let rows = planes.y.chunks(planes.y_stride);
    for (row, (source, target)) in rows.zip(out.chunks_exact_mut(pitch)).enumerate() {
        let source = &source[..layout.luma_row];
        match format {
            PixelFormat::Nv12 => target[..layout.luma_row].copy_from_slice(source),
            PixelFormat::P010 => {
                check_i010("y", row, source)?;
                for (stored, sample) in target
                    .as_chunks_mut::<2>()
                    .0
                    .iter_mut()
                    .zip(source.as_chunks::<2>().0.iter())
                {
                    stored.copy_from_slice(&raise(sample));
                }
            }
        }
    }
    Ok(out)
}

/// Interleaved chroma rows at `pitch` bytes, with 0 past each row's data.
fn pack_chroma(
    planes: &HostPlanes<'_>,
    layout: &HostLayout,
    format: PixelFormat,
    pitch: usize,
) -> Result<Vec<u8>, FrameError> {
    let mut out = vec![0_u8; layout.chroma_height * pitch];
    let rows = planes
        .u
        .chunks(planes.u_stride)
        .zip(planes.v.chunks(planes.v_stride));
    for (row, ((u, v), target)) in rows.zip(out.chunks_exact_mut(pitch)).enumerate() {
        let (u, v) = (&u[..layout.chroma_row], &v[..layout.chroma_row]);
        match format {
            PixelFormat::Nv12 => {
                for (pair, (&cb, &cr)) in target
                    .as_chunks_mut::<2>()
                    .0
                    .iter_mut()
                    .zip(u.iter().zip(v))
                {
                    pair.copy_from_slice(&[cb, cr]);
                }
            }
            PixelFormat::P010 => {
                check_i010("u", row, u)?;
                check_i010("v", row, v)?;
                let samples = u.as_chunks::<2>().0.iter().zip(v.as_chunks::<2>().0.iter());
                for (pair, (cb, cr)) in target.as_chunks_mut::<4>().0.iter_mut().zip(samples) {
                    let (stored_cb, stored_cr) = pair.split_at_mut(2);
                    stored_cb.copy_from_slice(&raise(cb));
                    stored_cr.copy_from_slice(&raise(cr));
                }
            }
        }
    }
    Ok(out)
}

/// One sample of two little-endian bytes. Every caller passes a two-byte chunk.
fn sample_value(sample: &[u8]) -> u16 {
    sample
        .iter()
        .rev()
        .fold(0, |value, &byte| (value << 8) | u16::from(byte))
}

/// An I010 sample as P010 bytes.
fn raise(sample: &[u8]) -> [u8; 2] {
    (sample_value(sample) << P010_SHIFT).to_le_bytes()
}

/// A P010 sample as I010 bytes.
fn lower(sample: &[u8]) -> [u8; 2] {
    (sample_value(sample) >> P010_SHIFT).to_le_bytes()
}

/// Refuses a row with a value that `value << 6` cuts. One OR over the row keeps
/// the common path free of a branch per sample.
fn check_i010(plane: &'static str, row: usize, samples: &[u8]) -> Result<(), FrameError> {
    let combined = samples
        .as_chunks::<2>()
        .0
        .iter()
        .fold(0, |bits, sample| bits | sample_value(sample));
    if combined <= I010_MAX {
        return Ok(());
    }
    let found = samples
        .as_chunks::<2>()
        .0
        .iter()
        .map(|sample| sample_value(sample))
        .enumerate()
        .find(|&(_, value)| value > I010_MAX);
    let (column, value) = found.unwrap_or((0, combined));
    Err(FrameError::HostSampleRange {
        plane,
        row,
        column,
        value,
    })
}

/// One device luma row as a host row.
fn unpack_luma(format: PixelFormat, source: &[u8], row: &mut [u8]) {
    match format {
        PixelFormat::Nv12 => row.copy_from_slice(source),
        PixelFormat::P010 => {
            for (sample, stored) in row
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(source.as_chunks::<2>().0.iter())
            {
                sample.copy_from_slice(&lower(stored));
            }
        }
    }
}

/// One device chroma row as a Cb row and a Cr row.
fn unpack_chroma(format: PixelFormat, source: &[u8], u: &mut [u8], v: &mut [u8]) {
    match format {
        PixelFormat::Nv12 => {
            for (pair, (cb, cr)) in source
                .as_chunks::<2>()
                .0
                .iter()
                .zip(u.iter_mut().zip(v.iter_mut()))
            {
                [*cb, *cr] = *pair;
            }
        }
        PixelFormat::P010 => {
            let targets = u
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(v.as_chunks_mut::<2>().0.iter_mut());
            for (pair, (cb, cr)) in source.as_chunks::<4>().0.iter().zip(targets) {
                let (stored_cb, stored_cr) = pair.split_at(2);
                cb.copy_from_slice(&lower(stored_cb));
                cr.copy_from_slice(&lower(stored_cr));
            }
        }
    }
}

/// The format and display size of a frame. A sample-form frame states them
/// through its luma plane, and a word-form frame states them itself.
fn frame_layout<R>(frame: &EncodeFrame<'_, R>) -> Result<(PixelFormat, Dimensions), FrameError>
where
    R: Runtime<DType = DType>,
{
    if let Some(layout) = frame.word_layout() {
        return Ok(layout);
    }
    let y = frame.y();
    let format = match y.dtype() {
        DType::U8 => PixelFormat::Nv12,
        DType::U16 => PixelFormat::P010,
        other => {
            return Err(FrameError::PlaneMismatch {
                plane: "luma",
                expected: "[rows, columns] in U8 or U16".to_owned(),
                found: format!("{:?} in {other:?}", y.shape()),
            });
        }
    };
    let &[height, width] = y.shape() else {
        return Err(FrameError::PlaneMismatch {
            plane: "luma",
            expected: "[rows, columns] in U8 or U16".to_owned(),
            found: format!("{:?} in {:?}", y.shape(), y.dtype()),
        });
    };
    Ok((format, host_display(width, height)?))
}

/// Reads a checked two-dimensional plane to `scratch` in the fewest device
/// copies, and returns the bytes from one row to the next.
fn read_plane<R>(plane: &Tensor<R>, scratch: &mut Vec<u8>) -> Result<usize, FrameError>
where
    R: Runtime<DType = DType>,
{
    let element = plane.dtype().size_in_bytes();
    let &[rows, columns] = plane.shape() else {
        return Err(FrameError::PlaneMismatch {
            plane: "frame",
            expected: "[rows, columns]".to_owned(),
            found: format!("{:?}", plane.shape()),
        });
    };
    let row_bytes = columns * element;
    if plane.is_contiguous() && plane.offset() == 0 {
        read_bytes(plane, rows * row_bytes, scratch)?;
        return Ok(row_bytes);
    }
    // A pointer backend reads the span of a pitched view in one copy. The span
    // stays inside the parent surface, and the gaps between rows are not used.
    if R::handles_take_byte_offsets()
        && let &[row_stride, 1] = plane.strides()
        && let Ok(row_stride) = usize::try_from(row_stride)
        && row_stride >= columns
    {
        let pitch = row_stride * element;
        read_bytes(plane, (rows - 1) * pitch + row_bytes, scratch)?;
        return Ok(pitch);
    }
    let dense = plane.contiguous().map_err(|error| {
        numr_error(
            "Tensor::contiguous",
            format!(
                "a plane view of shape {:?} with strides {:?}",
                plane.shape(),
                plane.strides()
            ),
            &error,
        )
    })?;
    read_bytes(&dense, rows * row_bytes, scratch)?;
    Ok(row_bytes)
}

/// Copies `bytes` bytes from the plane's first element to `scratch`.
fn read_bytes<R>(plane: &Tensor<R>, bytes: usize, scratch: &mut Vec<u8>) -> Result<(), FrameError>
where
    R: Runtime<DType = DType>,
{
    scratch.resize(bytes, 0);
    R::copy_from_device(plane.ptr(), scratch, plane.device()).map_err(|error| {
        numr_error(
            "Runtime::copy_from_device",
            format!(
                "{bytes} bytes of a plane of shape {:?} in {:?}",
                plane.shape(),
                plane.dtype()
            ),
            &error,
        )
    })
}

#[cfg(test)]
mod tests {
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    use super::*;

    /// Planes of 3x3 I420 samples with a stride of 5, and 0xA5 between rows.
    fn i420_3x3() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let y = vec![
            1, 2, 3, 0xA5, 0xA5, 4, 5, 6, 0xA5, 0xA5, 7, 8, 9, 0xA5, 0xA5,
        ];
        (y, vec![10, 11, 12, 13], vec![20, 21, 22, 23])
    }

    fn view<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8]) -> HostPlanes<'a> {
        HostPlanes {
            y,
            y_stride: 5,
            u,
            u_stride: 2,
            v,
            v_stride: 2,
            width: 3,
            height: 3,
        }
    }

    /// The word path writes the bytes numr's packed form documents: each row
    /// from the start of its first word, with 0 past its data. On a build with
    /// no GPU, only this test runs that path of `upload_planes`.
    #[test]
    fn words_hold_each_row_from_its_first_word() {
        let (y, u, v) = i420_3x3();
        let device = CpuDevice::new();
        let (y, uv) = upload_in_form::<CpuRuntime>(
            &device,
            view(&y, &u, &v),
            PixelFormat::Nv12,
            PlaneForm::Words,
        )
        .unwrap();
        assert_eq!(y.dtype(), DType::U32);
        assert_eq!(y.shape(), [3, 1]);
        assert_eq!(y.to_bytes().unwrap(), [1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0]);
        assert_eq!(uv.shape(), [2, 1]);
        assert_eq!(uv.to_bytes().unwrap(), [10, 20, 11, 21, 12, 22, 13, 23]);
    }

    /// A word-form frame downloads to the planes it came from, and a second
    /// frame keeps every buffer. So the frame allocates nothing on the host.
    #[test]
    fn a_word_frame_round_trips_into_the_same_buffers() {
        let (y, u, v) = i420_3x3();
        let device = CpuDevice::new();
        let (words_y, words_uv) = upload_in_form::<CpuRuntime>(
            &device,
            view(&y, &u, &v),
            PixelFormat::Nv12,
            PlaneForm::Words,
        )
        .unwrap();
        let display = Dimensions::new(3, 3).unwrap();
        let frame =
            EncodeFrame::from_packed_planes(&words_y, &words_uv, PixelFormat::Nv12, display);
        let mut out = HostFrame::new();
        download_planes(&frame, &mut out).unwrap();
        assert_eq!(out.y(), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(out.u(), u);
        assert_eq!(out.v(), v);

        let buffers = (
            out.y.as_ptr(),
            out.u.as_ptr(),
            out.v.as_ptr(),
            out.scratch.as_ptr(),
        );
        let capacity = out.scratch.capacity();
        download_planes(&frame, &mut out).unwrap();
        assert_eq!(
            buffers,
            (
                out.y.as_ptr(),
                out.u.as_ptr(),
                out.v.as_ptr(),
                out.scratch.as_ptr(),
            )
        );
        assert_eq!(capacity, out.scratch.capacity());
    }

    #[test]
    fn i010_values_shift_up_and_back_in_words() {
        let device = CpuDevice::new();
        let samples = |values: &[u16]| -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        };
        let (y, u, v) = (samples(&[1023, 1, 512]), samples(&[2, 3]), samples(&[4, 5]));
        let planes = HostPlanes {
            y: &y,
            y_stride: 6,
            u: &u,
            u_stride: 4,
            v: &v,
            v_stride: 4,
            width: 3,
            height: 1,
        };
        let (words_y, words_uv) =
            upload_in_form::<CpuRuntime>(&device, planes, PixelFormat::P010, PlaneForm::Words)
                .unwrap();
        assert_eq!(words_y.shape(), [1, 2]);
        assert_eq!(
            words_y.to_bytes().unwrap(),
            [
                (1023_u16 << 6).to_le_bytes(),
                (1_u16 << 6).to_le_bytes(),
                (512_u16 << 6).to_le_bytes(),
                [0, 0],
            ]
            .concat()
        );
        let display = Dimensions::new(3, 1).unwrap();
        let frame =
            EncodeFrame::from_packed_planes(&words_y, &words_uv, PixelFormat::P010, display);
        let mut out = HostFrame::new();
        download_planes(&frame, &mut out).unwrap();
        assert_eq!(out.y(), y);
        assert_eq!(out.u(), u);
        assert_eq!(out.v(), v);
    }

    #[test]
    fn a_value_above_ten_bits_names_its_position() {
        let row = [1_u16, 1024, 3]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>();
        assert_eq!(
            check_i010("y", 7, &row),
            Err(FrameError::HostSampleRange {
                plane: "y",
                row: 7,
                column: 1,
                value: 1024,
            })
        );
    }
}
