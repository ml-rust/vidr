//! [`RemasterTransform`]: the work on one frame of a remaster run.
//!
//! Each frame runs these steps on the frame's device, with no host copy:
//!
//! 1. `to_rgb` at the model's dtype and layout, F16 NHWC by default.
//! 2. The model's native forward, unless the transform is resize-only.
//! 3. A cast to F32 and an antialiased bicubic resize. An equal size skips both.
//! 4. `from_rgb` to the source's pixel format.
//!
//! The transform adds no clamp. `from_rgb` clamps R, G, and B to `[0, 1]` before
//! the colour matrix, and a NaN reads as 0.
//!
//! The RGB and resize buffers persist across frames of one size. Each frame's
//! output planes are new, because the encoder still holds earlier ones. numr has
//! no `cast_into`, and boostr's forward allocates its own intermediates. Both
//! allocate a constant amount per frame.

use std::fmt;
#[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
use std::path::Path;
use std::path::PathBuf;

#[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
use boostr::model::super_resolution::{SrExecution, SrModel, load_sr_dir};
use numr::dtype::DType;
#[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
use numr::ops::ConvAccumulation;
use numr::ops::{
    BinaryOps, ColorOps, ConvLayout, FusedConvOps, ImageOps, ResizeConfig, ResizeFilter, ScalarOps,
    ShapeOps, TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use vidr::color::ColorPlan;
use vidr::decode::DecodedFrame;
use vidr::frame::{OutputPlanes, from_rgb, reuse_tensor, to_rgb_reusing};
use vidr::pipeline::{FrameTransform, PipelineError};
use vidr::{Dimensions, FrameError};

use crate::catalog::{CatalogError, Network};
#[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
use crate::catalog::{catalog_entry, require_model};
use crate::request::ModelId;

/// The dtype of resize-only RGB. The resize runs in F32 in any case.
const RESIZE_ONLY_DTYPE: DType = DType::F32;
/// The layout of resize-only RGB.
const RESIZE_ONLY_LAYOUT: ConvLayout = ConvLayout::Nhwc;

/// The numr ops one frame uses. Any client with every op takes this bound.
///
/// boostr's `SrClient` is the same list without `ColorOps`, so this bound also
/// satisfies it.
pub trait TransformClient<R: Runtime>:
    RuntimeClient<R>
    + BinaryOps<R>
    + ColorOps<R>
    + FusedConvOps<R>
    + ImageOps<R>
    + ScalarOps<R>
    + ShapeOps<R>
    + TypeConversionOps<R>
{
}

impl<R, C> TransformClient<R> for C
where
    R: Runtime,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + ColorOps<R>
        + FusedConvOps<R>
        + ImageOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TypeConversionOps<R>,
{
}

/// How the transform runs a model: F16 in NHWC, with half accumulation allowed.
///
/// A backend can ignore the accumulation permission. Only the native CUDA path
/// honours it.
#[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
pub fn remaster_execution() -> SrExecution {
    SrExecution::new(DType::F16, ConvLayout::Nhwc).with_accumulation(ConvAccumulation::AllowHalf)
}

/// An error of [`RemasterTransform`].
#[derive(Debug)]
#[non_exhaustive]
pub enum TransformError {
    /// The target has an odd side, which 4:2:0 planes cannot hold.
    OddTarget {
        /// The target size.
        target: Dimensions,
    },
    /// An allocation fails on a frame.
    OutOfMemory {
        /// The frame's size.
        frame: Dimensions,
        /// The model, or `None` for a resize-only transform.
        model: Option<ModelId>,
        /// The step and numr's message.
        detail: String,
    },
    /// A frame conversion fails.
    Frame(FrameError),
    /// A numr call on a frame fails.
    Numr {
        /// The numr function.
        operation: &'static str,
        /// The inputs of the call.
        input: String,
        /// numr's message.
        reason: String,
    },
    /// The catalog refuses the model.
    Catalog(CatalogError),
    /// A checkpoint does not load.
    Load {
        /// The model.
        model: ModelId,
        /// The model's directory.
        dir: PathBuf,
        /// boostr's message.
        reason: String,
    },
    /// A checkpoint holds another network than the catalog names.
    WrongNetwork {
        /// The model.
        model: ModelId,
        /// The network the catalog names.
        expected: Network,
        /// The network the checkpoint holds.
        found: Network,
    },
    /// The model's forward fails on a frame.
    Model {
        /// The model.
        model: ModelId,
        /// The frame's size.
        frame: Dimensions,
        /// boostr's message.
        reason: String,
    },
}

impl fmt::Display for TransformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OddTarget { target } => write!(
                f,
                "Output size {target} has an odd side, and 4:2:0 planes need even sides. Pass the \
                 even size `choose_resolution` returns."
            ),
            Self::OutOfMemory {
                frame,
                model,
                detail,
            } => {
                match model {
                    Some(model) => write!(f, "Model {model}")?,
                    None => f.write_str("The resize-only transform")?,
                }
                write!(f, " runs out of memory on a {frame} frame: {detail}. ")?;
                f.write_str(match model {
                    Some(ModelId::Fast) => "Use a smaller source.",
                    Some(_) => "Use --model fast or a smaller source.",
                    None => "Use a smaller source or a smaller --resolution.",
                })
            }
            Self::Frame(error) => fmt::Display::fmt(error, f),
            Self::Numr {
                operation,
                input,
                reason,
            } => write!(
                f,
                "{operation} fails for {input}: {reason}. Check that the backend supports the \
                 dtype and layout."
            ),
            Self::Catalog(error) => fmt::Display::fmt(error, f),
            Self::Load { model, dir, reason } => write!(
                f,
                "Model {model} does not load from {}: {reason}. Keep one valid checkpoint in that \
                 directory.",
                dir.display()
            ),
            Self::WrongNetwork {
                model,
                expected,
                found,
            } => write!(
                f,
                "Model {model} needs a {expected} checkpoint, and its directory holds a {found} \
                 checkpoint. Put the model's own checkpoint in its directory."
            ),
            Self::Model {
                model,
                frame,
                reason,
            } => write!(
                f,
                "Model {model} fails on a {frame} frame: {reason}. Report this error with the \
                 source that causes it."
            ),
        }
    }
}

impl std::error::Error for TransformError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Frame(error) => Some(error),
            Self::Catalog(error) => Some(error),
            _ => None,
        }
    }
}

impl From<CatalogError> for TransformError {
    fn from(error: CatalogError) -> Self {
        Self::Catalog(error)
    }
}

impl From<TransformError> for PipelineError {
    fn from(error: TransformError) -> Self {
        match error {
            TransformError::Frame(error) => Self::Frame(error),
            other => Self::failed("the remaster transform", other.to_string()),
        }
    }
}

/// A loaded model and its name.
#[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
struct LoadedModel<R: Runtime> {
    id: ModelId,
    network: SrModel<R>,
}

/// Restores and scales one frame at a time. It is a [`FrameTransform`] for
/// `vidr::pipeline::run`.
///
/// The output has the target size, and the pixel format of each source frame.
/// [`ColorPlan`] supplies both colour specs. A frame runs as batch 1 with no
/// tiling, so the device must hold the model at the frame's own size.
///
/// The target is the coded output size. For a source rotated by 90 or 270
/// degrees, pass it through [`coded_output_size`](crate::coded_output_size).
pub struct RemasterTransform<R: Runtime> {
    #[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
    model: Option<LoadedModel<R>>,
    target: Dimensions,
    colors: ColorPlan,
    rgb_dtype: DType,
    layout: ConvLayout,
    rgb: Option<Tensor<R>>,
    resized: Option<Tensor<R>>,
}

impl<R: Runtime<DType = DType>> RemasterTransform<R> {
    /// A transform with no model, for `--from-upscaled`: `to_rgb`, the resize,
    /// and `from_rgb`.
    ///
    /// # Errors
    ///
    /// [`TransformError::OddTarget`] for an odd side.
    pub fn resize_only(target: Dimensions, colors: ColorPlan) -> Result<Self, TransformError> {
        check_target(target)?;
        Ok(Self {
            #[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
            model: None,
            target,
            colors,
            rgb_dtype: RESIZE_ONLY_DTYPE,
            layout: RESIZE_ONLY_LAYOUT,
            rgb: None,
            resized: None,
        })
    }

    /// Returns the coded output size.
    pub fn target(&self) -> Dimensions {
        self.target
    }

    /// Returns the model's name, or `None` for a resize-only transform.
    #[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
    pub fn model(&self) -> Option<ModelId> {
        self.model.as_ref().map(|model| model.id)
    }

    /// Returns `None`, because this build holds no model.
    #[cfg(not(any(feature = "model-srvgg", feature = "model-rrdb")))]
    pub fn model(&self) -> Option<ModelId> {
        None
    }
}

#[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
impl<R: Runtime<DType = DType>> RemasterTransform<R> {
    /// Wraps a network the caller loaded, at the execution it loaded with.
    ///
    /// # Errors
    ///
    /// - [`TransformError::OddTarget`] for an odd side.
    /// - [`TransformError::Catalog`] when this build leaves out `model`.
    /// - [`TransformError::WrongNetwork`] when `network` differs from the
    ///   network the catalog names for `model`.
    pub fn from_model(
        model: ModelId,
        network: SrModel<R>,
        target: Dimensions,
        colors: ColorPlan,
    ) -> Result<Self, TransformError> {
        check_target(target)?;
        require_model(model)?;
        let expected = catalog_entry(model).network;
        let found = match &network {
            SrModel::Compact(_) => Network::SrVggCompact,
            SrModel::Rrdb(_) => Network::Rrdb,
        };
        if found != expected {
            return Err(TransformError::WrongNetwork {
                model,
                expected,
                found,
            });
        }
        let execution = *network.execution();
        Ok(Self {
            model: Some(LoadedModel { id: model, network }),
            target,
            colors,
            rgb_dtype: execution.dtype,
            layout: execution.layout,
            rgb: None,
            resized: None,
        })
    }

    /// Loads `model` from `dir` onto `device`, at [`remaster_execution`].
    /// [`resolve_model_dir`](crate::resolve_model_dir) returns `dir`.
    ///
    /// # Errors
    ///
    /// [`TransformError::Load`] when the checkpoint does not load, and the
    /// errors of [`RemasterTransform::from_model`].
    pub fn load(
        model: ModelId,
        dir: &Path,
        device: &R::Device,
        target: Dimensions,
        colors: ColorPlan,
    ) -> Result<Self, TransformError>
    where
        R::Client: TransformClient<R>,
    {
        check_target(target)?;
        require_model(model)?;
        let network = load_sr_dir::<R>(dir, device, remaster_execution()).map_err(|error| {
            TransformError::Load {
                model,
                dir: dir.to_path_buf(),
                reason: error.to_string(),
            }
        })?;
        Self::from_model(model, network, target, colors)
    }
}

impl<R> RemasterTransform<R>
where
    R: Runtime<DType = DType>,
    R::Client: TransformClient<R>,
{
    fn run(&mut self, frame: &DecodedFrame<R>) -> Result<OutputPlanes<R>, TransformError> {
        let client = R::default_client(frame.y().device());
        let size = frame.display();
        let model = self.model();
        let failed = |error: FrameError| frame_failure(size, model, error);
        let rgb = to_rgb_reusing(
            &client,
            frame,
            &self.colors.decode,
            self.rgb_dtype,
            self.layout,
            &mut self.rgb,
        )
        .map_err(failed)?;
        let image = self.forward(&client, &rgb, size)?;
        let output = self.resize(&client, &image, size)?;
        from_rgb(
            &client,
            &output,
            &self.colors.encode,
            frame.format(),
            self.layout,
        )
        .map_err(failed)
    }

    /// The model's native forward, or `rgb` for a resize-only transform.
    #[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
    fn forward(
        &self,
        client: &R::Client,
        rgb: &Tensor<R>,
        frame: Dimensions,
    ) -> Result<Tensor<R>, TransformError> {
        let Some(model) = &self.model else {
            return Ok(rgb.clone());
        };
        model.network.forward_native(client, rgb).map_err(|error| {
            if error.is_out_of_memory() {
                TransformError::OutOfMemory {
                    frame,
                    model: Some(model.id),
                    detail: error.to_string(),
                }
            } else {
                TransformError::Model {
                    model: model.id,
                    frame,
                    reason: error.to_string(),
                }
            }
        })
    }

    /// `rgb`, because this build holds no model.
    #[cfg(not(any(feature = "model-srvgg", feature = "model-rrdb")))]
    fn forward(
        &self,
        _client: &R::Client,
        rgb: &Tensor<R>,
        _frame: Dimensions,
    ) -> Result<Tensor<R>, TransformError> {
        Ok(rgb.clone())
    }

    /// Resizes `image` to the target in F32, as the PoC's `resize_output` does.
    /// An image at the target size passes through.
    fn resize(
        &mut self,
        client: &R::Client,
        image: &Tensor<R>,
        frame: Dimensions,
    ) -> Result<Tensor<R>, TransformError> {
        let (height, width) = (self.target.height() as usize, self.target.width() as usize);
        let layout = self.layout;
        let shape = match layout {
            ConvLayout::Nchw => [1, 3, height, width],
            ConvLayout::Nhwc => [1, height, width, 3],
        };
        if image.shape() == shape.as_slice() {
            return Ok(image.clone());
        }
        let model = self.model();
        let wide = if image.dtype() == DType::F32 {
            image.clone()
        } else {
            client.cast(image, DType::F32).map_err(|error| {
                numr_failure(
                    frame,
                    model,
                    "TypeConversionOps::cast",
                    format!("{:?} of shape {:?} to F32", image.dtype(), image.shape()),
                    &error,
                )
            })?
        };
        let output = reuse_tensor(&mut self.resized, &shape, DType::F32, image.device()).map_err(
            |error| {
                numr_failure(
                    frame,
                    model,
                    "Tensor::empty",
                    format!("the resize output of shape {shape:?} in F32"),
                    &error,
                )
            },
        )?;
        let config = ResizeConfig::new((height, width), ResizeFilter::Bicubic).with_antialias(true);
        client
            .resize_into(&output, &wide, &config, layout)
            .map_err(|error| {
                numr_failure(
                    frame,
                    model,
                    "ImageOps::resize_into",
                    format!("{:?} to {height}x{width} in {layout:?}", wide.shape()),
                    &error,
                )
            })?;
        Ok(output)
    }
}

impl<R> FrameTransform<R> for RemasterTransform<R>
where
    R: Runtime<DType = DType>,
    R::Client: TransformClient<R>,
{
    fn apply(&mut self, frame: DecodedFrame<R>) -> Result<OutputPlanes<R>, PipelineError> {
        Ok(self.run(&frame)?)
    }
}

fn check_target(target: Dimensions) -> Result<(), TransformError> {
    if target.width().is_multiple_of(2) && target.height().is_multiple_of(2) {
        Ok(())
    } else {
        Err(TransformError::OddTarget { target })
    }
}

fn frame_failure(frame: Dimensions, model: Option<ModelId>, error: FrameError) -> TransformError {
    match error {
        FrameError::OutOfMemory {
            operation,
            input,
            reason,
        } => TransformError::OutOfMemory {
            frame,
            model,
            detail: format!("{operation} for {input}: {reason}"),
        },
        other => TransformError::Frame(other),
    }
}

fn numr_failure(
    frame: Dimensions,
    model: Option<ModelId>,
    operation: &'static str,
    input: String,
    error: &numr::error::Error,
) -> TransformError {
    if error.is_out_of_memory() {
        TransformError::OutOfMemory {
            frame,
            model,
            detail: format!("{operation} for {input}: {error}"),
        }
    } else {
        TransformError::Numr {
            operation,
            input,
            reason: error.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use vidr::color::color_plan;
    use vidr::decode::{ColorFacts, MATRIX_BT709, PixelFormat};

    use super::*;

    fn plan() -> ColorPlan {
        color_plan(&ColorFacts {
            matrix_coefficients: MATRIX_BT709,
            transfer_characteristics: 1,
            colour_primaries: 1,
            full_range: false,
            chroma_location: None,
        })
        .expect("BT.709 plan")
    }

    fn dims(width: u32, height: u32) -> Dimensions {
        Dimensions::new(width, height).expect("nonzero size")
    }

    fn frame(width: usize, height: usize) -> DecodedFrame<CpuRuntime> {
        let device = CpuDevice::new();
        let y = Tensor::<CpuRuntime>::from_slice(
            &vec![128_u8; width * height],
            &[height, width],
            &device,
        )
        .expect("luma");
        let uv = Tensor::<CpuRuntime>::from_slice(
            &vec![128_u8; height / 2 * width],
            &[height / 2, width],
            &device,
        )
        .expect("chroma");
        let display = dims(width as u32, height as u32);
        let facts = plan().tags;
        DecodedFrame::from_planes(y, uv, PixelFormat::Nv12, 0, display, facts)
    }

    fn buffer_ptr(slot: &Option<Tensor<CpuRuntime>>) -> u64 {
        slot.as_ref().expect("buffer").ptr()
    }

    #[test]
    fn an_equal_size_allocates_no_resize_buffer() {
        let mut transform =
            RemasterTransform::<CpuRuntime>::resize_only(dims(32, 18), plan()).expect("transform");
        transform.apply(frame(32, 18)).expect("apply");
        assert!(transform.resized.is_none());
    }

    #[test]
    fn frames_of_one_size_reuse_both_buffers() {
        let mut transform =
            RemasterTransform::<CpuRuntime>::resize_only(dims(64, 36), plan()).expect("transform");
        transform.apply(frame(32, 18)).expect("first frame");
        let rgb = buffer_ptr(&transform.rgb);
        let resized = buffer_ptr(&transform.resized);
        transform.apply(frame(32, 18)).expect("second frame");
        assert_eq!(buffer_ptr(&transform.rgb), rgb);
        assert_eq!(buffer_ptr(&transform.resized), resized);
    }

    #[test]
    fn a_new_frame_size_replaces_the_rgb_buffer_and_keeps_the_target() {
        let mut transform =
            RemasterTransform::<CpuRuntime>::resize_only(dims(64, 36), plan()).expect("transform");
        transform.apply(frame(32, 18)).expect("first frame");
        let planes = transform.apply(frame(16, 10)).expect("second frame");
        let shape = transform.rgb.as_ref().expect("buffer").shape().to_vec();
        assert_eq!(shape, [1, 10, 16, 3]);
        assert_eq!(planes.y.shape(), [36, 64]);
    }

    #[test]
    fn an_out_of_memory_error_names_the_next_action_for_each_model() {
        let error = |model| {
            TransformError::OutOfMemory {
                frame: dims(1920, 1080),
                model,
                detail: "Tensor::empty".to_string(),
            }
            .to_string()
        };
        let fast = error(Some(ModelId::Fast));
        assert!(
            fast.contains("fast") && fast.contains("1920x1080"),
            "{fast}"
        );
        assert!(
            fast.contains("smaller source") && !fast.contains("--model"),
            "{fast}"
        );
        let full = error(Some(ModelId::PureVision));
        assert!(full.contains("purevision"), "{full}");
        assert!(full.contains("--model fast or a smaller source"), "{full}");
        let resize = error(None);
        assert!(resize.contains("resize-only"), "{resize}");
        assert!(resize.contains("smaller source"), "{resize}");
    }
}
