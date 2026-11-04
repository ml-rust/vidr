//! `to_rgb` and `from_rgb` over numr's `ColorOps`.

use std::fmt;

use numr::dtype::DType;
use numr::ops::{ColorOps, ConvLayout, PackedPlane, PadMode, ShapeOps, YuvPlanes, YuvSpec};
use numr::runtime::{Device, Runtime};
use numr::tensor::Tensor;

use crate::decode::frame::chroma_shape;
use crate::decode::{DecodedFrame, PixelFormat, PlaneForm};
use crate::encode::EncodeFrame;
use crate::error::FrameError;
use crate::geometry::Dimensions;

/// Converts a decoded frame to float RGB on the frame's device.
///
/// The result has shape `[1, 3, H, W]` for `ConvLayout::Nchw` and
/// `[1, H, W, 3]` for `ConvLayout::Nhwc`, where `W x H` is the display size.
/// Values are R, G, B in `[0, 1]`. `dtype` takes what the backend's
/// `ColorOps` writes: F32, F64, F16, and BF16 on CPU and CUDA, F32 on WebGPU.
/// A frame's [`PlaneForm`] selects the numr form of each call.
///
/// numr converts even sizes only, and only AV1 and VP9 give odd ones. The
/// chroma plane of an odd frame already has the even-frame shape. So an odd
/// frame takes these steps, all on the device:
///
/// 1. `pad_mode(PadMode::Replicate)` for samples, or `pad_packed_luma` for words,
///    adds one luma row, one luma column, or both, on the odd side only.
/// 2. `yuv_to_rgb` converts the even frame.
/// 3. `narrow` and `contiguous` cut the RGB back to `[H, W]`.
///
/// Each output pixel takes its own luma sample and the chroma samples numr
/// selects for that position. The added luma never reaches the output.
///
/// # Errors
///
/// - [`FrameError::PlaneMismatch`] when a plane's shape or dtype does not
///   match the frame's format and display size.
/// - [`FrameError::Numr`] when numr refuses the call, for example for an
///   integer `dtype` or for sample-form planes on WebGPU.
pub fn to_rgb<R, C>(
    client: &C,
    frame: &DecodedFrame<R>,
    spec: &YuvSpec,
    dtype: DType,
    layout: ConvLayout,
) -> Result<Tensor<R>, FrameError>
where
    R: Runtime<DType = DType>,
    C: ColorOps<R> + ShapeOps<R>,
{
    let [height, width] = check_planes(frame)?;
    let padded = pad_luma(client, frame, height, width)?;
    let y = padded.as_ref().unwrap_or(frame.y());
    let even_width = width + width % 2;
    let planes = yuv_planes(
        frame.plane_form(),
        frame.format(),
        y,
        frame.uv(),
        even_width,
    );
    let rgb = client
        .yuv_to_rgb(planes, spec, dtype, layout)
        .map_err(|error| {
            numr_error(
                "ColorOps::yuv_to_rgb",
                format!(
                    "{} luma of shape {:?}, {spec:?}, {dtype:?}, {layout:?}",
                    frame.format(),
                    y.shape()
                ),
                &error,
            )
        })?;
    if padded.is_none() {
        return Ok(rgb);
    }
    crop(&rgb, layout, height, width)
}

/// Converts a decoded frame like [`to_rgb`], and reuses `buffer` when it can.
///
/// An even frame writes into `buffer`, which [`reuse_tensor`] allocates or keeps.
/// So a stream of one size allocates its RGB once. The result shares storage
/// with `buffer`, and the next call overwrites it. An odd frame takes the
/// [`to_rgb`] path, which allocates, and leaves `buffer` unchanged.
///
/// # Errors
///
/// The errors of [`to_rgb`]. [`FrameError::OutOfMemory`] when the allocation of
/// `buffer` fails.
pub fn to_rgb_reusing<R, C>(
    client: &C,
    frame: &DecodedFrame<R>,
    spec: &YuvSpec,
    dtype: DType,
    layout: ConvLayout,
    buffer: &mut Option<Tensor<R>>,
) -> Result<Tensor<R>, FrameError>
where
    R: Runtime<DType = DType>,
    C: ColorOps<R> + ShapeOps<R>,
{
    let [height, width] = check_planes(frame)?;
    if !height.is_multiple_of(2) || !width.is_multiple_of(2) {
        return to_rgb(client, frame, spec, dtype, layout);
    }
    let shape = match layout {
        ConvLayout::Nchw => [1, 3, height, width],
        ConvLayout::Nhwc => [1, height, width, 3],
    };
    let rgb = reuse_tensor(buffer, &shape, dtype, frame.y().device()).map_err(|error| {
        numr_error(
            "Tensor::empty",
            format!("RGB of shape {shape:?} in {dtype:?}"),
            &error,
        )
    })?;
    client
        .yuv_to_rgb_into(&rgb, frame.planes(), spec, layout)
        .map_err(|error| {
            numr_error(
                "ColorOps::yuv_to_rgb_into",
                format!(
                    "{} luma of shape {:?}, {spec:?}, {dtype:?}, {layout:?}",
                    frame.format(),
                    frame.y().shape()
                ),
                &error,
            )
        })?;
    Ok(rgb)
}

/// Returns the tensor in `slot` when its shape, dtype and device match.
/// Otherwise it frees the old tensor, allocates a new one, and keeps it in `slot`.
///
/// The result shares storage with `slot`, so a kernel that writes the result
/// overwrites what an earlier call wrote.
///
/// # Errors
///
/// numr's error when the allocation fails.
pub fn reuse_tensor<R>(
    slot: &mut Option<Tensor<R>>,
    shape: &[usize],
    dtype: DType,
    device: &R::Device,
) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
{
    if let Some(tensor) = slot.as_ref()
        && tensor.shape() == shape
        && tensor.dtype() == dtype
        && tensor.device().is_same(device)
    {
        return Ok(tensor.clone());
    }
    *slot = None;
    let tensor = Tensor::<R>::empty(shape, dtype, device)?;
    *slot = Some(tensor.clone());
    Ok(tensor)
}

/// Y and UV planes that [`from_rgb`] writes, on the device of its RGB input.
///
/// Both planes are dense. For a frame of `W x H`, `y` has shape `[H, W]` and
/// `uv` has shape `[H / 2, W]`, in U8 for `Nv12` and U16 for `P010`. In
/// [`PlaneForm::Words`] both hold U32 words of the same bytes, in the shapes
/// `DecodedFrame` documents. `EncodeFrame::from(&planes)` borrows both for an
/// encoder without a copy.
pub struct OutputPlanes<R: Runtime> {
    /// The luma plane.
    pub y: Tensor<R>,
    /// The interleaved chroma plane.
    pub uv: Tensor<R>,
    words: Option<(PixelFormat, Dimensions)>,
}

impl<R: Runtime> OutputPlanes<R> {
    /// Planes of U8 or U16 samples.
    pub fn new(y: Tensor<R>, uv: Tensor<R>) -> Self {
        Self { y, uv, words: None }
    }

    /// Planes of packed U32 words. Words record neither sample size nor row
    /// width, so this states the format and display size.
    pub fn from_packed_planes(
        y: Tensor<R>,
        uv: Tensor<R>,
        format: PixelFormat,
        display: Dimensions,
    ) -> Self {
        Self {
            y,
            uv,
            words: Some((format, display)),
        }
    }

    /// How the planes hold their samples.
    pub fn plane_form(&self) -> PlaneForm {
        match self.words {
            Some(_) => PlaneForm::Words,
            None => PlaneForm::Samples,
        }
    }

    /// The format and display size of word-form planes, or `None` for samples.
    pub fn word_layout(&self) -> Option<(PixelFormat, Dimensions)> {
        self.words
    }
}

impl<'a, R: Runtime> From<&'a OutputPlanes<R>> for EncodeFrame<'a, R> {
    fn from(planes: &'a OutputPlanes<R>) -> Self {
        match planes.words {
            Some((format, display)) => {
                Self::from_packed_planes(&planes.y, &planes.uv, format, display)
            }
            None => Self::new(&planes.y, &planes.uv),
        }
    }
}

// Manual impls: a derive bounds `R: Clone` and `R: Debug`, which `Runtime` does
// not require.
impl<R: Runtime> Clone for OutputPlanes<R> {
    fn clone(&self) -> Self {
        Self {
            y: self.y.clone(),
            uv: self.uv.clone(),
            words: self.words,
        }
    }
}

impl<R: Runtime> fmt::Debug for OutputPlanes<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OutputPlanes")
            .field("y", &self.y)
            .field("uv", &self.uv)
            .field("words", &self.words)
            .finish()
    }
}

/// Converts float RGB to new 4:2:0 planes on the device of `rgb`.
///
/// `rgb` holds one frame: `[1, 3, H, W]` for `ConvLayout::Nchw` and
/// `[1, H, W, 3]` for `ConvLayout::Nhwc`. Values outside `[0, 1]` clamp. The
/// planes hold U32 words when [`Runtime::packed_color_planes`] is true.
///
/// # Errors
///
/// - [`FrameError::RgbShape`] when `rgb` is not one frame in `layout`.
/// - [`FrameError::OddRgbSize`] when `W` or `H` is odd.
/// - [`FrameError::Numr`] when numr refuses the call, for example for a dtype
///   the backend does not read.
pub fn from_rgb<R, C>(
    client: &C,
    rgb: &Tensor<R>,
    spec: &YuvSpec,
    format: PixelFormat,
    layout: ConvLayout,
) -> Result<OutputPlanes<R>, FrameError>
where
    R: Runtime<DType = DType>,
    C: ColorOps<R>,
{
    let [height, width] = rgb_size(rgb, layout)?;
    if !height.is_multiple_of(2) || !width.is_multiple_of(2) {
        return Err(FrameError::OddRgbSize { width, height });
    }
    let display = u32::try_from(width)
        .ok()
        .zip(u32::try_from(height).ok())
        .and_then(|(columns, rows)| Dimensions::new(columns, rows).ok())
        .ok_or_else(|| FrameError::RgbShape {
            shape: rgb.shape().to_vec(),
            layout,
        })?;
    let form = if R::packed_color_planes() {
        PlaneForm::Words
    } else {
        PlaneForm::Samples
    };
    let (luma, chroma, dtype) = plane_shapes::<R>(form, format, display);
    let empty = |shape: [usize; 2]| {
        Tensor::<R>::empty(&shape, dtype, rgb.device()).map_err(|error| {
            numr_error(
                "Tensor::empty",
                format!("a plane of shape {shape:?} in {dtype:?}"),
                &error,
            )
        })
    };
    let y = empty(luma)?;
    let uv = empty(chroma)?;
    client
        .rgb_to_yuv_into(rgb, yuv_planes(form, format, &y, &uv, width), spec, layout)
        .map_err(|error| {
            numr_error(
                "ColorOps::rgb_to_yuv_into",
                format!(
                    "{:?} RGB of shape {:?}, {spec:?}, {format}, {layout:?}",
                    rgb.dtype(),
                    rgb.shape()
                ),
                &error,
            )
        })?;
    Ok(match form {
        PlaneForm::Samples => OutputPlanes::new(y, uv),
        PlaneForm::Words => OutputPlanes::from_packed_planes(y, uv, format, display),
    })
}

/// Pads the luma of an odd frame to the next even size on the device, or returns
/// `None` for an even frame.
fn pad_luma<R, C>(
    client: &C,
    frame: &DecodedFrame<R>,
    height: usize,
    width: usize,
) -> Result<Option<Tensor<R>>, FrameError>
where
    R: Runtime<DType = DType>,
    C: ColorOps<R> + ShapeOps<R>,
{
    let padding = [0, width % 2, 0, height % 2];
    if padding == [0; 4] {
        return Ok(None);
    }
    let shape = frame.y().shape();
    let padded = match frame.plane_form() {
        PlaneForm::Samples => client
            .pad_mode(frame.y(), &padding, PadMode::Replicate)
            .map_err(|error| {
                numr_error(
                    "ShapeOps::pad_mode",
                    format!("luma of shape {shape:?} with padding {padding:?}"),
                    &error,
                )
            })?,
        PlaneForm::Words => {
            let (rows, columns) = (height + height % 2, width + width % 2);
            let sample = frame.format().sample_dtype();
            client
                .pad_packed_luma(PackedPlane::new(frame.y(), width), rows, columns, sample)
                .map_err(|error| {
                    numr_error(
                        "ColorOps::pad_packed_luma",
                        format!(
                            "luma words of shape {shape:?}, width {width}, to \
                             {rows}x{columns} in {sample:?}"
                        ),
                        &error,
                    )
                })?
        }
    };
    Ok(Some(padded))
}

/// The `(luma, chroma, dtype)` shapes `[rows, columns]` of a frame's planes.
pub(super) fn plane_shapes<R: Runtime>(
    form: PlaneForm,
    format: PixelFormat,
    display: Dimensions,
) -> ([usize; 2], [usize; 2], DType) {
    let (width, height) = (display.width(), display.height());
    let luma = [height as usize, width as usize];
    let chroma = chroma_shape(width, height);
    match form {
        PlaneForm::Samples => (luma, chroma, format.sample_dtype()),
        PlaneForm::Words => {
            let sample = format.bytes_per_sample();
            let words =
                |[rows, columns]: [usize; 2]| [rows, PackedPlane::<R>::row_words(columns * sample)];
            (words(luma), words(chroma), DType::U32)
        }
    }
}

/// Checks both planes against the format and display size, and returns the
/// frame's `[H, W]`.
fn check_planes<R>(frame: &DecodedFrame<R>) -> Result<[usize; 2], FrameError>
where
    R: Runtime<DType = DType>,
{
    let display = frame.display();
    let (luma, chroma, dtype) = plane_shapes::<R>(frame.plane_form(), frame.format(), display);
    check_plane("luma", frame.y(), luma, dtype)?;
    check_plane("chroma", frame.uv(), chroma, dtype)?;
    Ok([display.height() as usize, display.width() as usize])
}

pub(super) fn check_plane<R>(
    plane: &'static str,
    tensor: &Tensor<R>,
    expected: [usize; 2],
    dtype: DType,
) -> Result<(), FrameError>
where
    R: Runtime<DType = DType>,
{
    if tensor.shape() == expected.as_slice() && tensor.dtype() == dtype {
        return Ok(());
    }
    Err(FrameError::PlaneMismatch {
        plane,
        expected: format!("{expected:?} in {dtype:?}"),
        found: format!("{:?} in {:?}", tensor.shape(), tensor.dtype()),
    })
}

/// The frame size `[H, W]` of `rgb`, which must hold one frame in `layout`.
fn rgb_size<R: Runtime>(rgb: &Tensor<R>, layout: ConvLayout) -> Result<[usize; 2], FrameError> {
    match (layout, rgb.shape()) {
        (ConvLayout::Nchw, &[1, 3, height, width]) | (ConvLayout::Nhwc, &[1, height, width, 3])
            if height > 0 && width > 0 =>
        {
            Ok([height, width])
        }
        (_, shape) => Err(FrameError::RgbShape {
            shape: shape.to_vec(),
            layout,
        }),
    }
}

/// The planes of an even frame `width` columns wide, in numr's form for `form`
/// and `format`.
fn yuv_planes<'a, R: Runtime>(
    form: PlaneForm,
    format: PixelFormat,
    y: &'a Tensor<R>,
    uv: &'a Tensor<R>,
    width: usize,
) -> YuvPlanes<'a, R> {
    let words = |plane| PackedPlane::new(plane, width);
    match (form, format) {
        (PlaneForm::Samples, PixelFormat::Nv12) => YuvPlanes::Nv12 { y, uv },
        (PlaneForm::Samples, PixelFormat::P010) => YuvPlanes::P010 { y, uv },
        (PlaneForm::Words, PixelFormat::Nv12) => YuvPlanes::Nv12Packed {
            y: words(y),
            uv: words(uv),
        },
        (PlaneForm::Words, PixelFormat::P010) => YuvPlanes::P010Packed {
            y: words(y),
            uv: words(uv),
        },
    }
}

/// Cuts RGB of an even frame back to `[height, width]`, in dense form.
fn crop<R: Runtime>(
    rgb: &Tensor<R>,
    layout: ConvLayout,
    height: usize,
    width: usize,
) -> Result<Tensor<R>, FrameError> {
    let (row_dim, column_dim) = match layout {
        ConvLayout::Nchw => (2, 3),
        ConvLayout::Nhwc => (1, 2),
    };
    rgb.narrow(row_dim, 0, height)
        .and_then(|view| view.narrow(column_dim, 0, width))
        .and_then(|view| view.contiguous())
        .map_err(|error| {
            numr_error(
                "Tensor::narrow and Tensor::contiguous",
                format!("RGB of shape {:?} to {height}x{width}", rgb.shape()),
                &error,
            )
        })
}

pub(super) fn numr_error(
    operation: &'static str,
    input: String,
    error: &numr::error::Error,
) -> FrameError {
    let reason = error.to_string();
    if error.is_out_of_memory() {
        FrameError::OutOfMemory {
            operation,
            input,
            reason,
        }
    } else {
        FrameError::Numr {
            operation,
            input,
            reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_allocation_failure_is_out_of_memory() {
        let error = numr::error::Error::OutOfMemory { size: 8 };
        let error = numr_error("Tensor::empty", "a plane".to_string(), &error);
        assert!(matches!(error, FrameError::OutOfMemory { .. }), "{error}");
        assert!(error.to_string().contains("smaller frame"), "{error}");
    }

    #[test]
    fn another_numr_error_stays_numr() {
        let error = numr::error::Error::NotContiguous;
        let error = numr_error("Tensor::empty", "a plane".to_string(), &error);
        assert!(matches!(error, FrameError::Numr { .. }), "{error}");
    }
}
