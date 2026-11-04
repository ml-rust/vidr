//! Errors in video encoding.

use std::error::Error as StdError;
use std::fmt;

#[cfg(any(feature = "h264", feature = "hevc", feature = "av1"))]
use super::BitstreamError;
#[cfg(feature = "frame")]
use super::FrameError;
use crate::decode::PixelFormat;
use crate::encode::EncodeCodec;

/// An error in an encoder: loading the driver library, checking the GPU,
/// checking a frame, encoding, or converting the output.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    /// The encoder's driver library does not load.
    LibraryMissing {
        /// The library file name, for example `libnvidia-encode.so.1`.
        library: &'static str,
        /// The loader's message.
        reason: String,
    },
    /// The driver library lacks a function vidr calls.
    SymbolMissing {
        /// The library file name.
        library: &'static str,
        /// The function name.
        symbol: &'static str,
        /// The loader's message.
        reason: String,
    },
    /// The driver supports an older API than vidr pins.
    DriverTooOld {
        /// The newest `(major, minor)` API the driver reports.
        found: (u32, u32),
        /// The `(major, minor)` API vidr pins.
        required: (u32, u32),
    },
    /// An NVENC call returns an error status.
    Nvenc {
        /// The function called.
        call: &'static str,
        /// The `NVENCSTATUS` code.
        code: i32,
        /// The code's name, or `unknown` for a code vidr does not name.
        name: &'static str,
        /// The inputs of the call.
        input: String,
        /// The driver's last error string. Empty when it has none.
        detail: String,
    },
    /// A CUDA call through numr or cudarc fails.
    Cuda {
        /// The operation, for example `cuMemcpy2DAsync into an NVENC surface`.
        operation: &'static str,
        /// The driver's message.
        reason: String,
    },
    /// The GPU has no encoder for the codec.
    CodecUnsupported {
        /// The codec.
        codec: EncodeCodec,
        /// The codecs the GPU lists.
        available: String,
    },
    /// The GPU does not encode the codec at this bit depth.
    BitDepthUnsupported {
        /// The codec.
        codec: EncodeCodec,
        /// The sample layout.
        format: PixelFormat,
    },
    /// The GPU encodes fewer consecutive B-frames than the config asks.
    BFramesUnsupported {
        /// The codec.
        codec: EncodeCodec,
        /// The config's `b_frames`.
        requested: u32,
        /// The GPU's maximum. 0 means no B-frames.
        maximum: u32,
    },
    /// An odd size for a codec that crops 4:2:0 in units of two samples.
    OddSize {
        /// The codec.
        codec: EncodeCodec,
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// The size is outside the GPU's range for the codec.
    SizeOutOfRange {
        /// The codec.
        codec: EncodeCodec,
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// The GPU's minimum `(width, height)`.
        minimum: (u32, u32),
        /// The GPU's maximum `(width, height)`.
        maximum: (u32, u32),
    },
    /// A config value is out of range.
    InvalidConfig {
        /// The field.
        field: &'static str,
        /// The value.
        value: String,
        /// What the field needs.
        reason: &'static str,
    },
    /// A plane's shape or dtype does not match the config.
    SurfaceMismatch {
        /// The plane: `luma` or `chroma`.
        plane: &'static str,
        /// The shape `[rows, columns]` and dtype the config needs.
        expected: String,
        /// The plane's shape and dtype.
        found: String,
    },
    /// A plane's strides are not a row pitch over adjacent columns.
    PlaneStrides {
        /// The plane: `luma` or `chroma`.
        plane: &'static str,
        /// The plane's strides, in samples.
        strides: Vec<isize>,
        /// Samples per row.
        columns: usize,
    },
    /// A frame's pts does not increase.
    PtsNotIncreasing {
        /// The frame's pts.
        pts: i64,
        /// The previous frame's pts.
        previous: i64,
    },
    /// A frame's duration is not above zero.
    NonPositiveDuration {
        /// The frame's pts.
        pts: i64,
        /// The duration.
        duration: i64,
    },
    /// `send` follows `flush`.
    AfterFlush {
        /// The frame's pts.
        pts: i64,
    },
    /// A decode timestamp does not fit `i64`.
    TimestampOverflow {
        /// The first frame's pts.
        pts: i64,
        /// The first frame's duration.
        duration: i64,
        /// The reorder delay in frames.
        delay: u32,
    },
    /// The encoder returns a picture that decodes after it displays. The reorder
    /// depth is deeper than the config's `b_frames`.
    DtsAfterPts {
        /// The computed dts.
        dts: i64,
        /// The packet's pts.
        pts: i64,
        /// The reorder delay in frames.
        delay: u32,
    },
    /// The encoder returns more pictures than the frames it received.
    UnexpectedOutput {
        /// The packet's pts.
        pts: i64,
    },
    /// Device memory for an input surface cannot be allocated.
    Allocation {
        /// The size in bytes.
        bytes: usize,
        /// The driver's message.
        reason: String,
    },
    /// A software encoder's call returns an error.
    Software {
        /// The encoder, for example `OpenH264`.
        encoder: &'static str,
        /// The call that fails.
        call: &'static str,
        /// The inputs of the call.
        input: String,
        /// The library's message.
        reason: String,
    },
    /// A software encoder returns no picture for a frame, so the output would
    /// lose the frame.
    NoPicture {
        /// The encoder.
        encoder: &'static str,
        /// The frame's pts.
        pts: i64,
        /// The frame type the encoder reports.
        frame_type: String,
    },
    /// The OpenH264 binary at the caller's path does not load, or its hash is
    /// not on the list the `openh264` crate holds.
    Openh264Library {
        /// The path the caller passes.
        path: String,
        /// The loader's message.
        reason: String,
    },
    /// A Vulkan call returns an error.
    Vulkan {
        /// The function called.
        call: &'static str,
        /// The `VkResult` code.
        code: i32,
        /// The code's name.
        name: String,
        /// The inputs of the call.
        input: String,
    },
    /// No device on this machine runs Vulkan Video encode.
    VulkanUnavailable {
        /// What each device lacks.
        reason: String,
    },
    /// The Vulkan driver refuses what this config needs, or vidr has no path for it.
    VulkanRefused {
        /// The codec.
        codec: EncodeCodec,
        /// What the driver or vidr refuses.
        what: String,
        /// What the driver lists.
        reason: String,
    },
    /// An encode feedback query states a status other than complete.
    VulkanEncodeStatus {
        /// The `VkQueryResultStatusKHR` value.
        status: i32,
        /// The status's name.
        name: &'static str,
        /// The frame's pts.
        pts: i64,
    },
    /// A picture does not fit the bitstream buffer.
    VulkanBitstreamOverrun {
        /// The frame's pts.
        pts: i64,
        /// The buffer's range in bytes.
        capacity: u64,
    },
    /// An encode feedback query states a byte range outside the bitstream buffer.
    VulkanFeedbackRange {
        /// The frame's pts.
        pts: i64,
        /// The stated offset in bytes.
        offset: u32,
        /// The stated size in bytes.
        written: u32,
        /// The buffer's range in bytes.
        capacity: u64,
    },
    /// A wait on the GPU does not end in time.
    VulkanWaitTimeout {
        /// How long the wait ran, in seconds.
        seconds: u64,
        /// What the wait covers.
        input: String,
    },
    /// No memory type of the device holds what a Vulkan object needs.
    VulkanNoMemoryType {
        /// The object.
        what: String,
        /// The memory property flags it needs.
        required: String,
        /// The memory types the object allows, as a bit mask.
        type_bits: u32,
    },
    /// The Vulkan core returns a failure with no encode name of its own.
    VulkanCore {
        /// The core's message.
        reason: String,
    },
    /// The Vulkan device is lost, so nothing on it runs again.
    VulkanDeviceLost {
        /// The function that reports the loss.
        call: &'static str,
        /// What the call covers.
        input: String,
    },
    /// A frame's planes fail the copy to host memory.
    #[cfg(feature = "frame")]
    Frame(FrameError),
    /// The encoder's output fails the bitstream filter.
    #[cfg(any(feature = "h264", feature = "hevc", feature = "av1"))]
    Bitstream(BitstreamError),
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LibraryMissing { library, reason } => write!(
                f,
                "{library} does not load: {reason}. Install the NVIDIA driver, which ships \
                 {library}, and check that the loader path finds it."
            ),
            Self::SymbolMissing {
                library,
                symbol,
                reason,
            } => write!(
                f,
                "{library} has no function `{symbol}`: {reason}. The driver is older than \
                 vidr needs. Update the NVIDIA driver."
            ),
            Self::DriverTooOld { found, required } => write!(
                f,
                "the NVIDIA driver supports NVENC API {}.{}, and vidr needs {}.{}. Update the \
                 NVIDIA driver.",
                found.0, found.1, required.0, required.1
            ),
            Self::Nvenc {
                call,
                code,
                name,
                input,
                detail,
            } => write!(
                f,
                "{call} returns NVENCSTATUS {code} ({name}) for {input}. Driver says: \
                 \"{detail}\". Check the encoder config against the GPU's caps, and check that \
                 another process has not used every encode session, then retry."
            ),
            Self::Cuda { operation, reason } => write!(
                f,
                "{operation} fails: {reason}. Check that the CUDA device is present and has \
                 free memory, then retry."
            ),
            Self::CodecUnsupported { codec, available } => write!(
                f,
                "the GPU has no {codec} encoder. It lists: {available}. Pick a listed codec, or \
                 encode on a GPU that has {codec}."
            ),
            Self::BitDepthUnsupported { codec, format } => write!(
                f,
                "the GPU does not encode {codec} from {format} at {} bits. Convert the frames \
                 to NV12, or pick a codec the GPU encodes at this depth.",
                format.bit_depth()
            ),
            Self::BFramesUnsupported {
                codec,
                requested,
                maximum,
            } => write!(
                f,
                "the config asks for {requested} consecutive B-frames, and the GPU encodes at \
                 most {maximum} for {codec}. Set `b_frames` to at most {maximum}."
            ),
            Self::OddSize {
                codec,
                width,
                height,
            } => write!(
                f,
                "{codec} cannot code the odd size {width}x{height}, because it crops 4:2:0 in \
                 units of two samples. Scale or pad the frames to an even size."
            ),
            Self::SizeOutOfRange {
                codec,
                width,
                height,
                minimum,
                maximum,
            } => write!(
                f,
                "{codec} size {width}x{height} is outside the GPU's range: {}x{} to {}x{}. \
                 Scale the frames into that range.",
                minimum.0, minimum.1, maximum.0, maximum.1
            ),
            Self::InvalidConfig {
                field,
                value,
                reason,
            } => write!(
                f,
                "encoder config `{field}` is {value}: {reason}. Change the value and open the \
                 encoder again."
            ),
            Self::SurfaceMismatch {
                plane,
                expected,
                found,
            } => write!(
                f,
                "the {plane} plane is {found}, and the encoder config needs {expected}. Convert \
                 the frame to the config's size and pixel format."
            ),
            Self::PlaneStrides {
                plane,
                strides,
                columns,
            } => write!(
                f,
                "the {plane} plane has strides {strides:?} for {columns} samples per row. The \
                 encoder needs a column stride of 1 and a row stride of at least {columns}. Pass \
                 a dense plane, or a `narrow` of a pitched surface."
            ),
            Self::PtsNotIncreasing { pts, previous } => write!(
                f,
                "the frame at pts {pts} follows pts {previous}. Send frames in display order \
                 with increasing pts."
            ),
            Self::NonPositiveDuration { pts, duration } => write!(
                f,
                "the frame at pts {pts} has duration {duration}. Pass a duration above zero."
            ),
            Self::AfterFlush { pts } => write!(
                f,
                "the frame at pts {pts} follows flush. Create a new encoder for a new stream."
            ),
            Self::TimestampOverflow {
                pts,
                duration,
                delay,
            } => write!(
                f,
                "the first dts, pts {pts} minus {delay} frames of duration {duration}, does not \
                 fit 64 bits. Rebase the pts near zero, or lower `b_frames`."
            ),
            Self::DtsAfterPts { dts, pts, delay } => write!(
                f,
                "the packet at pts {pts} gets dts {dts} with a reorder delay of {delay} frames, \
                 so it decodes after it displays. Report the NVIDIA driver version as a vidr \
                 bug, and encode with `b_frames` 0 until a fix lands."
            ),
            Self::UnexpectedOutput { pts } => write!(
                f,
                "the encoder returns a packet at pts {pts} with no frame left to match it. \
                 Report the NVIDIA driver version as a vidr bug, and retry with the driver \
                 updated."
            ),
            Self::Allocation { bytes, reason } => write!(
                f,
                "an input surface of {bytes} bytes does not allocate: {reason}. Free device \
                 memory, or lower the frame size or `b_frames`."
            ),
            Self::Software {
                encoder,
                call,
                input,
                reason,
            } => write!(
                f,
                "{encoder} call {call} fails for {input}: {reason}. Check the encoder config \
                 against {encoder}'s limits, and report this as a vidr bug if the config fits."
            ),
            Self::NoPicture {
                encoder,
                pts,
                frame_type,
            } => write!(
                f,
                "{encoder} returns frame type {frame_type} and no picture for the frame at pts \
                 {pts}, so the output would lose it. Report this as a vidr bug, with the encoder \
                 config."
            ),
            Self::Openh264Library { path, reason } => write!(
                f,
                "the OpenH264 binary at {path} does not load: {reason}. Pass the path of an \
                 unmodified Cisco release that the `openh264` crate's hash list names."
            ),
            Self::Vulkan {
                call,
                code,
                name,
                input,
            } => write!(
                f,
                "{call} returns VkResult {code} ({name}) for {input}. Check the encoder config \
                 against the driver's capabilities, then retry."
            ),
            Self::VulkanUnavailable { reason } => write!(
                f,
                "Vulkan Video encode is unavailable: {reason}. Install a driver with \
                 VK_KHR_video_encode_queue, or pick another encoder."
            ),
            Self::VulkanRefused {
                codec,
                what,
                reason,
            } => write!(
                f,
                "Vulkan Video does not encode {codec} with {what}: {reason}. Change the config, \
                 or pick another encoder."
            ),
            Self::VulkanEncodeStatus { status, name, pts } => write!(
                f,
                "the encode of the frame at pts {pts} ends with status {status} ({name}). Check \
                 the driver, then retry."
            ),
            Self::VulkanBitstreamOverrun { pts, capacity } => write!(
                f,
                "the frame at pts {pts} does not fit the {capacity}-byte bitstream buffer. Raise \
                 the config's quantiser or lower its bitrate, then retry."
            ),
            Self::VulkanFeedbackRange {
                pts,
                offset,
                written,
                capacity,
            } => write!(
                f,
                "the driver states {written} bytes at offset {offset} for the frame at pts {pts}, \
                 outside the {capacity}-byte buffer. Report the driver version as a vidr bug."
            ),
            Self::VulkanWaitTimeout { seconds, input } => write!(
                f,
                "a wait on the GPU for {input} does not end in {seconds} s. Check that the GPU is \
                 not stalled or busy, then retry."
            ),
            Self::VulkanNoMemoryType {
                what,
                required,
                type_bits,
            } => write!(
                f,
                "no Vulkan memory type holds {required} for {what}, in type bits {type_bits:#x}. \
                 Check the driver, then retry."
            ),
            Self::VulkanCore { reason } => write!(
                f,
                "the Vulkan core stops the encode: {reason}. Report this as a vidr bug."
            ),
            Self::VulkanDeviceLost { call, input } => write!(
                f,
                "{call} reports a lost Vulkan device for {input}. Nothing on this device runs \
                 again. Open a new device and a new encoder."
            ),
            #[cfg(feature = "frame")]
            Self::Frame(error) => fmt::Display::fmt(error, f),
            #[cfg(any(feature = "h264", feature = "hevc", feature = "av1"))]
            Self::Bitstream(error) => fmt::Display::fmt(error, f),
        }
    }
}

impl StdError for EncodeError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        #[cfg(any(feature = "h264", feature = "hevc", feature = "av1"))]
        if let Self::Bitstream(error) = self {
            return Some(error);
        }
        #[cfg(feature = "frame")]
        if let Self::Frame(error) = self {
            return Some(error);
        }
        None
    }
}

#[cfg(any(feature = "h264", feature = "hevc", feature = "av1"))]
impl From<BitstreamError> for EncodeError {
    fn from(error: BitstreamError) -> Self {
        Self::Bitstream(error)
    }
}

#[cfg(feature = "frame")]
impl From<FrameError> for EncodeError {
    fn from(error: FrameError) -> Self {
        Self::Frame(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_refusal_names_the_codec_and_the_list() {
        let error = EncodeError::CodecUnsupported {
            codec: EncodeCodec::Av1,
            available: "H.264, HEVC".to_owned(),
        };
        let message = error.to_string();
        assert!(message.contains("no AV1 encoder"));
        assert!(message.contains("H.264, HEVC"));
    }

    #[test]
    fn old_driver_names_both_versions() {
        let error = EncodeError::DriverTooOld {
            found: (12, 2),
            required: (13, 1),
        };
        let message = error.to_string();
        assert!(message.contains("API 12.2"));
        assert!(message.contains("needs 13.1"));
        assert!(message.contains("Update the NVIDIA driver"));
    }

    #[test]
    fn odd_size_says_what_to_do() {
        let error = EncodeError::OddSize {
            codec: EncodeCodec::Hevc,
            width: 321,
            height: 180,
        };
        let message = error.to_string();
        assert!(message.contains("HEVC cannot code the odd size 321x180"));
        assert!(message.contains("even size"));
    }
}
