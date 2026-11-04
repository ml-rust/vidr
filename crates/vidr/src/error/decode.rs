//! Errors in video decoding.

use std::error::Error as StdError;
use std::fmt;

#[cfg(any(feature = "h264", feature = "hevc", feature = "av1", feature = "vp9"))]
use super::BitstreamError;
#[cfg(feature = "frame")]
use super::FrameError;
use crate::decode::{ChromaFormat, DecodeCodec};

/// An error in a decoder: loading the driver library, querying the GPU,
/// parsing the stream, decoding, or copying a frame.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeError {
    /// The decoder's driver library does not load.
    LibraryMissing {
        /// The library file name, for example `libnvcuvid.so.1`.
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
    /// A driver call returns an error code.
    Driver {
        /// The function called.
        call: &'static str,
        /// The `CUresult` code.
        code: i32,
        /// The code's name, or `unknown` for a code vidr does not name.
        name: &'static str,
        /// The inputs of the call.
        input: String,
    },
    /// A CUDA call through numr or cudarc fails.
    Cuda {
        /// The operation, for example `cuMemcpy2DAsync from the NVDEC surface`.
        operation: &'static str,
        /// The driver's message.
        reason: String,
    },
    /// The GPU does not decode the stream's codec, bit depth, and chroma
    /// format.
    Unsupported {
        /// The codec.
        codec: DecodeCodec,
        /// Bits per sample.
        bit_depth: u8,
        /// Chroma subsampling.
        chroma: ChromaFormat,
        /// What the GPU reports.
        reason: String,
    },
    /// The decoder takes the stream, but vidr has no frame layout for it.
    UnsupportedLayout {
        /// The codec.
        codec: DecodeCodec,
        /// Luma bits per sample.
        bit_depth_luma: u8,
        /// Chroma bits per sample.
        bit_depth_chroma: u8,
        /// Chroma subsampling.
        chroma: ChromaFormat,
    },
    /// The coded size is outside the GPU's limits for the codec.
    CodedSize {
        /// The codec.
        codec: DecodeCodec,
        /// Coded width in pixels.
        width: u32,
        /// Coded height in pixels.
        height: u32,
        /// The GPU's minimum `(width, height)`.
        min: (u32, u32),
        /// The GPU's maximum `(width, height)`.
        max: (u32, u32),
        /// The GPU's maximum count of 16x16 macroblocks.
        max_macroblocks: u32,
    },
    /// The sequence header's display rectangle does not fit the coded frame.
    DisplayArea {
        /// The rectangle `(left, top, right, bottom)`.
        rect: (i32, i32, i32, i32),
        /// Coded `(width, height)`.
        coded: (u32, u32),
    },
    /// The parser reports a codec other than the one the decoder opened.
    CodecMismatch {
        /// The codec the decoder opened.
        expected: DecodeCodec,
        /// The parser's `cudaVideoCodec` value.
        found: i32,
    },
    /// A picture arrives before a sequence header creates the decoder.
    NoSequence {
        /// The parser callback.
        callback: &'static str,
    },
    /// The driver passes a parser callback a null pointer argument.
    NullCallbackArgument {
        /// The parser callback.
        callback: &'static str,
        /// The null argument.
        argument: &'static str,
    },
    /// A packet is empty.
    EmptyPacket {
        /// The packet's pts.
        pts: i64,
    },
    /// `send` follows `flush`.
    AfterFlush {
        /// The packet's pts.
        pts: i64,
    },
    /// A parser callback panics. The panic stops at the callback.
    CallbackPanic {
        /// The parser callback.
        callback: &'static str,
        /// The panic message.
        message: String,
    },
    /// A frame plane cannot be allocated.
    Allocation {
        /// The plane: `luma` or `chroma`.
        plane: &'static str,
        /// The plane shape `[rows, columns]`.
        shape: [usize; 2],
        /// numr's message.
        reason: String,
    },
    /// A software decoder's call returns an error.
    Software {
        /// The decoder, for example `dav1d`.
        decoder: &'static str,
        /// The call that fails.
        call: &'static str,
        /// The inputs of the call.
        input: String,
        /// The library's message.
        reason: String,
    },
    /// A software decoder returns a picture with no packet left to match it.
    UnmatchedPicture {
        /// The decoder.
        decoder: &'static str,
    },
    /// Packets that hold a picture have no frame at the end of the stream.
    MissingPictures {
        /// The decoder.
        decoder: &'static str,
        /// Packets with no frame.
        count: usize,
        /// The lowest pts among them.
        first_pts: i64,
    },
    /// An SPS states a bit depth the H.264 decoder does not decode.
    H264BitDepth {
        /// The decoder.
        decoder: &'static str,
        /// Luma bits per sample.
        luma: u8,
        /// Chroma bits per sample.
        chroma: u8,
    },
    /// An SPS has `frame_mbs_only_flag` 0, and the H.264 decoder takes frames
    /// only.
    H264Interlaced {
        /// The decoder.
        decoder: &'static str,
    },
    /// An SPS states a profile vidr does not pass to the H.264 decoder.
    H264Profile {
        /// The decoder.
        decoder: &'static str,
        /// `profile_idc`.
        profile_idc: u8,
    },
    /// A VP9 frame header states a layout the VP9 decoder does not take.
    Vp9Layout {
        /// The decoder.
        decoder: &'static str,
        /// `Profile`, 0 to 3.
        profile: u8,
        /// `BitDepth`.
        bit_depth: u8,
        /// `subsampling_x`.
        subsampling_x: bool,
        /// `subsampling_y`.
        subsampling_y: bool,
    },
    /// An HEVC stream uses a tool the CPU HEVC decoder does not decode.
    HevcTool {
        /// The decoder.
        decoder: &'static str,
        /// The syntax element that names the tool.
        field: &'static str,
        /// The value the stream holds.
        value: u64,
        /// The values the decoder takes.
        supported: &'static str,
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
    /// No device on this machine runs Vulkan Video decode, or the device does not open.
    VulkanUnavailable {
        /// What each device lacks.
        reason: String,
    },
    /// The Vulkan driver refuses what this stream needs, or vidr has no path for it.
    VulkanRefused {
        /// The codec.
        codec: DecodeCodec,
        /// What the stream needs.
        what: String,
        /// Why it is out.
        reason: String,
    },
    /// The query of one decode reports a status other than complete.
    VulkanDecodeStatus {
        /// The `VkQueryResultStatusKHR` value.
        status: i32,
        /// The status's name.
        name: &'static str,
        /// The picture's pts.
        pts: i64,
    },
    /// A plane copy on wgpu's queue does not finish in time.
    VulkanCopyTimeout {
        /// How long the copy ran, in seconds.
        seconds: u64,
        /// What wgpu reports.
        reason: String,
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
    /// A codec hands the Vulkan core a picture it cannot run.
    VulkanDecodeRequest {
        /// The picture's pts.
        pts: i64,
        /// What the request lacks.
        reason: String,
    },
    /// A picture binds more active references than its session holds.
    VulkanTooManyReferences {
        /// The picture's pts.
        pts: i64,
        /// The active references the picture binds.
        references: usize,
        /// `maxActiveReferencePictures` of the session.
        limit: u32,
    },
    /// A picture names one DPB slot twice.
    VulkanDuplicateSlot {
        /// The picture's pts.
        pts: i64,
        /// The slot.
        slot: u32,
    },
    /// A VP9 frame ends at or before its tile data.
    VulkanNoTileBytes {
        /// The frame's pts.
        pts: i64,
        /// Where the tile data starts, in bytes from the frame's start.
        tiles_offset: u32,
        /// The bytes of the frame.
        frame_bytes: usize,
    },
    /// The Vulkan device is lost, so nothing on it runs again.
    VulkanDeviceLost {
        /// The call that reports it.
        call: &'static str,
        /// The inputs of the call.
        input: String,
    },
    /// An H.264 NAL unit does not parse.
    H264Header {
        /// The decoder.
        decoder: &'static str,
        /// The unit: `SPS`, `PPS`, or `slice`.
        unit: &'static str,
        /// The parser's message.
        reason: String,
    },
    /// An H.264 stream uses a tool the decoder does not decode.
    H264Tool {
        /// The decoder.
        decoder: &'static str,
        /// The syntax element that names the tool.
        field: &'static str,
        /// The value the stream holds.
        value: u64,
        /// The values the decoder takes.
        supported: &'static str,
    },
    /// The reference state of an H.264 stream does not hold together.
    H264References {
        /// The decoder.
        decoder: &'static str,
        /// The picture's pts.
        pts: i64,
        /// What does not hold.
        reason: String,
    },
    /// A frame's planes fail the copy to the decoder's device.
    #[cfg(feature = "frame")]
    Frame(FrameError),
    /// The config record or a packet fails the bitstream filter.
    #[cfg(any(feature = "h264", feature = "hevc", feature = "av1", feature = "vp9"))]
    Bitstream(BitstreamError),
}

impl fmt::Display for DecodeError {
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
            Self::Driver {
                call,
                code,
                name,
                input,
            } => write!(
                f,
                "{call} returns CUresult {code} ({name}) for {input}. Check that the GPU \
                 supports the stream and that no other process holds every decode engine, \
                 then retry."
            ),
            Self::Cuda { operation, reason } => write!(
                f,
                "{operation} fails: {reason}. Check that the CUDA device is present and has \
                 free memory, then retry."
            ),
            Self::Unsupported {
                codec,
                bit_depth,
                chroma,
                reason,
            } => write!(
                f,
                "the GPU refuses {codec} {bit_depth}-bit {chroma}: {reason}. Decode this \
                 stream on a GPU that supports it, or on the CPU."
            ),
            Self::UnsupportedLayout {
                codec,
                bit_depth_luma,
                bit_depth_chroma,
                chroma,
            } => write!(
                f,
                "{codec} stream is {chroma} with {bit_depth_luma}-bit luma and \
                 {bit_depth_chroma}-bit chroma. vidr outputs 4:2:0 frames at 8 or 10 bits \
                 with equal luma and chroma depths only. Convert the source to 4:2:0 at 8 or \
                 10 bits."
            ),
            Self::CodedSize {
                codec,
                width,
                height,
                min,
                max,
                max_macroblocks,
            } => write!(
                f,
                "{codec} coded size {width}x{height} is outside the GPU's range: {}x{} to \
                 {}x{}, at most {max_macroblocks} macroblocks. Scale the source into that \
                 range, or decode it on the CPU.",
                min.0, min.1, max.0, max.1
            ),
            Self::DisplayArea { rect, coded } => write!(
                f,
                "display rectangle left {} top {} right {} bottom {} does not fit the coded \
                 frame {}x{}, or has an empty side. The sequence header is corrupt. Remux the \
                 source, and re-encode it if the remux reports the same error.",
                rect.0, rect.1, rect.2, rect.3, coded.0, coded.1
            ),
            Self::CodecMismatch { expected, found } => write!(
                f,
                "the parser reports cudaVideoCodec {found}, but the decoder opened {expected}. \
                 Open the decoder with the stream's codec."
            ),
            Self::NoSequence { callback } => write!(
                f,
                "the {callback} callback runs before any sequence header. Start the stream at a \
                 random access point, or pass the config record to the decoder."
            ),
            Self::NullCallbackArgument { callback, argument } => write!(
                f,
                "the driver passes the {callback} callback a null `{argument}`. Report the \
                 NVIDIA driver version as a vidr bug, and retry with the driver updated."
            ),
            Self::EmptyPacket { pts } => write!(
                f,
                "the packet at pts {pts} is empty. Pass each sample with its bytes, and skip \
                 empty samples."
            ),
            Self::AfterFlush { pts } => write!(
                f,
                "the packet at pts {pts} follows flush. Create a new decoder for a new stream."
            ),
            Self::CallbackPanic { callback, message } => write!(
                f,
                "the {callback} callback panics: {message}. Report this as a vidr bug, with the \
                 stream that triggers it."
            ),
            Self::Allocation {
                plane,
                shape,
                reason,
            } => write!(
                f,
                "the {plane} plane [{}, {}] does not allocate: {reason}. Free device memory, \
                 or take frames from the decoder sooner.",
                shape[0], shape[1]
            ),
            Self::Software {
                decoder,
                call,
                input,
                reason,
            } => write!(
                f,
                "{decoder} call {call} fails for {input}: {reason}. Run `ffmpeg -v error -i INPUT \
                 -f null -` on the source. Remux or re-encode the source if ffmpeg reports the \
                 same stream as damaged, and report this as a vidr bug if it does not."
            ),
            Self::UnmatchedPicture { decoder } => write!(
                f,
                "{decoder} returns a picture with no packet left to match it, so the picture has \
                 no pts. Pass one access unit per packet, and report this as a vidr bug if each \
                 packet holds one."
            ),
            Self::MissingPictures {
                decoder,
                count,
                first_pts,
            } => write!(
                f,
                "{decoder} returns no frame for {count} packets at the end of the stream, the \
                 first at pts {first_pts}. The output would lose frames. Pass one access unit \
                 per packet, and check the source with `ffmpeg -v error -i INPUT -f null -`."
            ),
            Self::H264BitDepth {
                decoder,
                luma,
                chroma,
            } => write!(
                f,
                "the H.264 SPS states {luma}-bit luma and {chroma}-bit chroma, and {decoder} \
                 decodes 8 bits only. Decode this stream with another decoder, or re-encode the \
                 source at 8 bits."
            ),
            Self::H264Interlaced { decoder } => write!(
                f,
                "the H.264 SPS has frame_mbs_only_flag 0, and {decoder} decodes frames only, with \
                 no fields or MBAFF. Decode this stream with another decoder, or re-encode the \
                 source as progressive."
            ),
            Self::H264Profile {
                decoder,
                profile_idc,
            } => write!(
                f,
                "the H.264 SPS states profile_idc {profile_idc}, and vidr passes {decoder} \
                 Baseline only, profile_idc 66, because its B-slice output differs from the \
                 reference decode. Decode this stream with another decoder, or re-encode the \
                 source as Baseline."
            ),
            Self::Vp9Layout {
                decoder,
                profile,
                bit_depth,
                subsampling_x,
                subsampling_y,
            } => {
                let chroma = match (subsampling_x, subsampling_y) {
                    (true, true) => "4:2:0",
                    (true, false) => "4:2:2",
                    (false, true) => "4:4:0",
                    (false, false) => "4:4:4",
                };
                write!(
                    f,
                    "the VP9 frame header states profile {profile}, {chroma}, {bit_depth} bits, \
                     and {decoder} takes profiles 0 and 2, 4:2:0 at 8 or 10 bits. Decode this \
                     stream with another decoder, or convert the source to 4:2:0 at 8 or 10 bits."
                )
            }
            Self::HevcTool {
                decoder,
                field,
                value,
                supported,
            } => write!(
                f,
                "the HEVC stream holds {field} equal to {value}, and {decoder} takes {supported}. \
                 Decode this stream with another decoder, or transcode it to Main or Main 10."
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
                "{call} returns VkResult {code} ({name}) for {input}. Update the GPU driver, \
                 free device memory, and retry. Report a repeat as a vidr bug."
            ),
            Self::VulkanUnavailable { reason } => write!(
                f,
                "Vulkan Video decode is unavailable: {reason}. Install a driver with \
                 VK_KHR_video_decode_h264, or pass --decoder cpu."
            ),
            Self::VulkanRefused {
                codec,
                what,
                reason,
            } => write!(
                f,
                "Vulkan Video does not decode {what} of this {codec} stream: {reason}. Decode \
                 this stream with another decoder, or re-encode it within that limit."
            ),
            Self::VulkanDecodeStatus { status, name, pts } => write!(
                f,
                "the decode of the picture at pts {pts} ends with status {status} ({name}). The \
                 stream is damaged, or the driver refuses it. Run `ffmpeg -v error -i INPUT -f \
                 null -` on the source, and decode it with another decoder if ffmpeg passes."
            ),
            Self::VulkanCopyTimeout { seconds, reason } => write!(
                f,
                "the plane copy of a Vulkan Video frame does not finish in {seconds} s: {reason}. \
                 The GPU queue stalls, so vidr keeps the images it uses allocated. Check the GPU \
                 and the kernel log, then rerun."
            ),
            Self::VulkanNoMemoryType {
                what,
                required,
                type_bits,
            } => write!(
                f,
                "no Vulkan memory type holds {required} for {what}, in type bits {type_bits:#x}. \
                 Decode this stream with another decoder, or update the GPU driver."
            ),
            Self::VulkanDecodeRequest { pts, reason } => write!(
                f,
                "the Vulkan core cannot run the picture at pts {pts}: {reason}. Report this as a \
                 vidr bug, with the stream that triggers it."
            ),
            Self::VulkanTooManyReferences {
                pts,
                references,
                limit,
            } => write!(
                f,
                "the picture at pts {pts} binds {references} active references, and its session \
                 holds {limit}. The stream is damaged. Run `ffmpeg -v error -i INPUT -f null -` \
                 on the source, and re-encode it if ffmpeg reports an error."
            ),
            Self::VulkanDuplicateSlot { pts, slot } => write!(
                f,
                "the picture at pts {pts} names DPB slot {slot} twice. The stream is damaged. \
                 Run `ffmpeg -v error -i INPUT -f null -` on the source, and re-encode it if \
                 ffmpeg reports an error."
            ),
            Self::VulkanNoTileBytes {
                pts,
                tiles_offset,
                frame_bytes,
            } => write!(
                f,
                "the VP9 frame at pts {pts} holds {frame_bytes} bytes, and its tile data starts \
                 at byte {tiles_offset}. The frame has no tile bytes, so the stream is damaged. \
                 Run `ffmpeg -v error -i INPUT -f null -` on the source, and re-encode it if \
                 ffmpeg reports an error."
            ),
            Self::VulkanDeviceLost { call, input } => write!(
                f,
                "{call} reports a lost Vulkan device for {input}. Nothing on this device runs \
                 again. Check the kernel log for a GPU reset, open a new device, and decode the \
                 stream with another decoder if the loss repeats."
            ),
            Self::H264Header {
                decoder,
                unit,
                reason,
            } => write!(
                f,
                "{decoder} cannot parse an H.264 {unit}: {reason}. The stream is damaged. Run \
                 `ffmpeg -v error -i INPUT -f null -` on the source, and re-encode it if ffmpeg \
                 reports the same unit."
            ),
            Self::H264Tool {
                decoder,
                field,
                value,
                supported,
            } => write!(
                f,
                "the H.264 stream holds {field} equal to {value}, and {decoder} takes {supported}. \
                 Decode this stream with another decoder, or re-encode it without that tool."
            ),
            Self::H264References {
                decoder,
                pts,
                reason,
            } => write!(
                f,
                "{decoder} cannot track the references of the picture at pts {pts}: {reason}. \
                 Start the stream at an IDR picture, or re-encode the source."
            ),
            #[cfg(feature = "frame")]
            Self::Frame(error) => fmt::Display::fmt(error, f),
            #[cfg(any(feature = "h264", feature = "hevc", feature = "av1", feature = "vp9"))]
            Self::Bitstream(error) => fmt::Display::fmt(error, f),
        }
    }
}

impl StdError for DecodeError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        #[cfg(any(feature = "h264", feature = "hevc", feature = "av1", feature = "vp9"))]
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

#[cfg(any(feature = "h264", feature = "hevc", feature = "av1", feature = "vp9"))]
impl From<BitstreamError> for DecodeError {
    fn from(error: BitstreamError) -> Self {
        Self::Bitstream(error)
    }
}

#[cfg(feature = "frame")]
impl From<FrameError> for DecodeError {
    fn from(error: FrameError) -> Self {
        Self::Frame(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_names_codec_depth_and_chroma() {
        let error = DecodeError::Unsupported {
            codec: DecodeCodec::H264,
            bit_depth: 10,
            chroma: ChromaFormat::Yuv420,
            reason: "bIsSupported is 0".to_owned(),
        };
        let message = error.to_string();
        assert!(message.contains("H.264 10-bit 4:2:0"));
        assert!(message.contains("bIsSupported is 0"));
    }

    #[test]
    fn null_callback_argument_names_callback_argument_and_next_action() {
        let error = DecodeError::NullCallbackArgument {
            callback: "sequence",
            argument: "format",
        };
        let message = error.to_string();
        assert!(message.contains("sequence callback"));
        assert!(message.contains("null `format`"));
        assert!(message.contains("driver version"));
        assert!(message.contains("updated"));
        assert!(!message.contains("before any sequence header"));
    }

    #[test]
    fn h264_refusals_name_the_field_and_the_next_action() {
        let depth = DecodeError::H264BitDepth {
            decoder: "OpenH264",
            luma: 10,
            chroma: 10,
        };
        assert!(depth.to_string().contains("10-bit luma"));
        assert!(depth.to_string().contains("re-encode"));
        let interlaced = DecodeError::H264Interlaced {
            decoder: "OpenH264",
        };
        assert!(interlaced.to_string().contains("frame_mbs_only_flag 0"));
        let profile = DecodeError::H264Profile {
            decoder: "OpenH264",
            profile_idc: 100,
        };
        assert!(profile.to_string().contains("profile_idc 100"));
        assert!(profile.to_string().contains("as Baseline"));
    }

    #[test]
    fn vulkan_errors_name_the_input_and_the_next_action() {
        let call = DecodeError::Vulkan {
            call: "vkCreateImage",
            code: -2,
            name: "ERROR_OUT_OF_DEVICE_MEMORY".to_owned(),
            input: "a 320x192 NV12 image".to_owned(),
        };
        let text = call.to_string();
        assert!(text.contains("vkCreateImage"));
        assert!(text.contains("ERROR_OUT_OF_DEVICE_MEMORY"));
        assert!(text.contains("a 320x192 NV12 image"));
        assert!(text.contains("retry"));

        let refused = DecodeError::VulkanRefused {
            codec: DecodeCodec::H264,
            what: "a coded size of 8192x8192".to_owned(),
            reason: "the driver lists at most 4096x4096".to_owned(),
        };
        let text = refused.to_string();
        assert!(text.contains("8192x8192"));
        assert!(text.contains("at most 4096x4096"));
        assert!(text.contains("another decoder"));

        let timeout = DecodeError::VulkanCopyTimeout {
            seconds: 60,
            reason: "Timeout".to_owned(),
        };
        let text = timeout.to_string();
        assert!(text.contains("60 s"));
        assert!(text.contains("rerun"));
        assert!(!text.contains("Install"));

        let memory = DecodeError::VulkanNoMemoryType {
            what: "a DPB image".to_owned(),
            required: "DEVICE_LOCAL".to_owned(),
            type_bits: 0b101,
        };
        let text = memory.to_string();
        assert!(text.contains("a DPB image"));
        assert!(text.contains("0x5"));
        assert!(text.contains("another decoder"));
        assert!(!text.contains("H.264"));

        let request = DecodeError::VulkanDecodeRequest {
            pts: 9,
            reason: "DPB slot 7 does not exist".to_owned(),
        };
        let text = request.to_string();
        assert!(text.contains("pts 9"));
        assert!(text.contains("DPB slot 7"));
        assert!(text.contains("vidr bug"));

        let status = DecodeError::VulkanDecodeStatus {
            status: -1,
            name: "ERROR",
            pts: 512,
        };
        assert!(status.to_string().contains("pts 512"));

        let tool = DecodeError::H264Tool {
            decoder: "Vulkan Video",
            field: "num_slice_groups_minus1",
            value: 3,
            supported: "0",
        };
        assert!(
            tool.to_string()
                .contains("num_slice_groups_minus1 equal to 3")
        );
    }

    #[test]
    fn vp9_layout_names_profile_chroma_depth_and_next_action() {
        let error = DecodeError::Vp9Layout {
            decoder: "rusty_vp9",
            profile: 3,
            bit_depth: 12,
            subsampling_x: false,
            subsampling_y: true,
        };
        let message = error.to_string();
        assert!(message.contains("profile 3, 4:4:0, 12 bits"));
        assert!(message.contains("rusty_vp9 takes profiles 0 and 2"));
        assert!(message.contains("convert the source"));
    }

    #[test]
    fn hevc_tool_names_the_field_the_value_and_the_next_action() {
        let error = DecodeError::HevcTool {
            decoder: "HevcDecoder",
            field: "chroma_format_idc",
            value: 3,
            supported: "1, which is 4:2:0",
        };
        let message = error.to_string();
        assert!(message.contains("chroma_format_idc equal to 3"));
        assert!(message.contains("HevcDecoder takes 1, which is 4:2:0"));
        assert!(message.contains("transcode"));
    }

    #[test]
    fn missing_library_says_to_install_the_driver() {
        let error = DecodeError::LibraryMissing {
            library: "libnvcuvid.so.1",
            reason: "not found".to_owned(),
        };
        let message = error.to_string();
        assert!(message.contains("libnvcuvid.so.1 does not load"));
        assert!(message.contains("Install the NVIDIA driver"));
    }
}
