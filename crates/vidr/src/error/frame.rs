//! Errors in frame conversion and host plane copies.

use std::error::Error as StdError;
use std::fmt;

use numr::ops::ConvLayout;

/// An error in a conversion between 4:2:0 planes and float RGB, or in a copy of
/// planes between host memory and tensors.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FrameError {
    /// A decoded plane's shape or dtype does not match the frame's format and
    /// display size.
    PlaneMismatch {
        /// The plane: `luma` or `chroma`.
        plane: &'static str,
        /// The shape `[rows, columns]` and dtype the frame needs.
        expected: String,
        /// The plane's shape and dtype.
        found: String,
    },
    /// An RGB tensor does not hold one frame in its layout.
    RgbShape {
        /// The tensor's shape.
        shape: Vec<usize>,
        /// The layout the call names.
        layout: ConvLayout,
    },
    /// RGB of an odd size cannot become 4:2:0 planes.
    OddRgbSize {
        /// Width in pixels.
        width: usize,
        /// Height in pixels.
        height: usize,
    },
    /// A host frame's display size is zero or too large to address.
    HostFrameSize {
        /// Width in pixels.
        width: usize,
        /// Height in pixels.
        height: usize,
    },
    /// A host plane's stride is below its row width.
    HostStride {
        /// The plane: `y`, `u`, or `v`.
        plane: &'static str,
        /// The stride in bytes.
        stride: usize,
        /// The row width in bytes.
        row_bytes: usize,
    },
    /// A host plane ends before its last row does.
    HostPlaneShort {
        /// The plane: `y`, `u`, or `v`.
        plane: &'static str,
        /// The bytes its rows and stride need.
        needed: usize,
        /// The bytes it holds.
        found: usize,
    },
    /// An I010 sample holds a value above 1023.
    HostSampleRange {
        /// The plane: `y`, `u`, or `v`.
        plane: &'static str,
        /// The sample's row.
        row: usize,
        /// The sample's column.
        column: usize,
        /// The value.
        value: u16,
    },
    /// A numr call cannot allocate its memory.
    OutOfMemory {
        /// The numr function.
        operation: &'static str,
        /// The inputs of the call.
        input: String,
        /// numr's message.
        reason: String,
    },
    /// A numr call refuses its input or fails.
    Numr {
        /// The numr function.
        operation: &'static str,
        /// The inputs of the call.
        input: String,
        /// numr's message.
        reason: String,
    },
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlaneMismatch {
                plane,
                expected,
                found,
            } => write!(
                f,
                "the {plane} plane is {found}, and the frame's format and display size \
                 need {expected}. Build the frame with planes in the layout `DecodedFrame` \
                 documents."
            ),
            Self::RgbShape { shape, layout } => {
                let needed = match layout {
                    ConvLayout::Nchw => "[1, 3, H, W]",
                    ConvLayout::Nhwc => "[1, H, W, 3]",
                };
                write!(
                    f,
                    "RGB of shape {shape:?} does not hold one {layout:?} frame. That layout needs \
                     {needed} with H and W above zero. Pass one frame, or name its layout."
                )
            }
            Self::OddRgbSize { width, height } => write!(
                f,
                "RGB of size {width}x{height} cannot become 4:2:0 planes, because `from_rgb` \
                 needs an even width and height. Crop or pad the RGB to an even size."
            ),
            Self::HostFrameSize { width, height } => write!(
                f,
                "a host frame of size {width}x{height} has no valid plane layout, because each \
                 side must be above zero and fit 32 bits. Pass the frame's display size."
            ),
            Self::HostStride {
                plane,
                stride,
                row_bytes,
            } => write!(
                f,
                "host plane {plane} has a stride of {stride} bytes, below its row width of \
                 {row_bytes} bytes. Pass the stride in bytes, and the plane's display size."
            ),
            Self::HostPlaneShort {
                plane,
                needed,
                found,
            } => write!(
                f,
                "host plane {plane} holds {found} bytes, and its rows and stride need {needed}. \
                 Pass the whole plane, or correct the size and stride."
            ),
            Self::HostSampleRange {
                plane,
                row,
                column,
                value,
            } => write!(
                f,
                "host plane {plane} holds {value} at row {row}, column {column}, above the \
                 10-bit maximum of 1023. Pass I010 samples in the low 10 bits."
            ),
            Self::OutOfMemory {
                operation,
                input,
                reason,
            } => write!(
                f,
                "{operation} runs out of memory for {input}: {reason}. Free device memory, or \
                 use a smaller frame."
            ),
            Self::Numr {
                operation,
                input,
                reason,
            } => write!(
                f,
                "{operation} fails for {input}: {reason}. Check that the backend supports the \
                 dtype and plane form, and that the device has free memory, then retry."
            ),
        }
    }
}

impl StdError for FrameError {}
