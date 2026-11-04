//! Errors of the CPU HEVC decoder core.

use std::error::Error as StdError;
use std::fmt;

use crate::error::BitstreamError;

/// An error while checking a stream or parsing its slice data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum HevcError {
    /// A parameter set or a slice header check fails.
    Header(BitstreamError),
    /// The stream uses a tool outside Main and Main 10 at 4:2:0.
    Unsupported {
        /// The syntax element that names the tool.
        field: &'static str,
        /// The value the stream holds.
        value: u64,
        /// The values this decoder takes.
        supported: &'static str,
    },
    /// The picture exceeds the largest size of any level below 8.5 (A.4.1).
    PictureTooLarge {
        /// `pic_width_in_luma_samples`.
        width: u32,
        /// `pic_height_in_luma_samples`.
        height: u32,
    },
    /// A buffer does not fit in memory.
    Allocation {
        /// What the buffer holds.
        what: &'static str,
        /// Elements the buffer needs.
        count: usize,
    },
    /// A display window does not fit its picture, or starts on an odd sample.
    DisplayWindow {
        /// Left edge in luma samples.
        x: usize,
        /// Top edge in luma samples.
        y: usize,
        /// Width in luma samples.
        width: usize,
        /// Height in luma samples.
        height: usize,
    },
    /// Slice data ends before the syntax does.
    SliceDataTruncated {
        /// Bits the parser has read.
        read_bits: u64,
        /// Slice data size in bytes.
        size: usize,
    },
    /// A slice names a parameter set the stream has not sent.
    MissingParameterSet {
        /// `SPS` or `PPS`.
        kind: &'static str,
        /// The identifier the slice or PPS names.
        id: u32,
    },
    /// A picture ends before its slices cover every CTB.
    IncompletePicture {
        /// CTBs the slices reach.
        decoded: usize,
        /// `PicSizeInCtbsY`.
        expected: usize,
    },
    /// A decoded picture differs from its picture hash SEI message (D.3.20).
    PictureHashMismatch {
        /// `MD5`, `CRC`, or `checksum`.
        method: &'static str,
        /// `cIdx` of the plane.
        c_idx: usize,
        /// The value in the SEI message.
        expected: u128,
        /// The value of the decoded plane.
        computed: u128,
    },
    /// A reference picture list names a picture the buffer does not hold.
    MissingReference {
        /// The POC of the picture, or its LSB for a long-term entry with no MSB.
        poc: i32,
    },
    /// A syntax element holds a value the specification forbids.
    InvalidSyntax {
        /// The syntax element or derived variable.
        element: &'static str,
        /// The value read.
        value: i64,
        /// The values the specification allows.
        allowed: &'static str,
    },
}

impl fmt::Display for HevcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Header(error) => write!(f, "HEVC header check fails: {error}"),
            Self::Unsupported {
                field,
                value,
                supported,
            } => write!(
                f,
                "The CPU HEVC decoder does not support {field} equal to {value}. It takes \
                 {supported}. Decode this stream with NVDEC, or transcode it to Main or Main 10."
            ),
            Self::PictureTooLarge { width, height } => write!(
                f,
                "The picture size {width}x{height} exceeds level 6.2: 35651584 luma samples and \
                 16888 samples per side. Transcode the stream to a smaller size."
            ),
            Self::Allocation { what, count } => write!(
                f,
                "No memory for {count} elements of {what}. Decode a smaller picture, or free \
                 memory."
            ),
            Self::DisplayWindow {
                x,
                y,
                width,
                height,
            } => write!(
                f,
                "The display window at ({x}, {y}) of size {width}x{height} does not fit the \
                 picture or starts on an odd sample. The conformance window in the SPS is \
                 corrupt. Remux the source."
            ),
            Self::SliceDataTruncated { read_bits, size } => write!(
                f,
                "HEVC slice data of {size} bytes ends after the parser reads {read_bits} bits. \
                 The input is truncated or corrupt. Remux the source."
            ),
            Self::MissingParameterSet { kind, id } => write!(
                f,
                "The HEVC stream names {kind} {id} before sending it. Send the parameter sets \
                 before the slices, or remux the source."
            ),
            Self::IncompletePicture { decoded, expected } => write!(
                f,
                "An HEVC picture ends with {decoded} of {expected} CTBs decoded. The input is \
                 truncated or corrupt. Remux the source."
            ),
            Self::PictureHashMismatch {
                method,
                c_idx,
                expected,
                computed,
            } => write!(
                f,
                "The {method} of plane {c_idx} is {computed:x}, and the picture hash SEI message \
                 holds {expected:x}. The input is corrupt or this decoder has a defect. Decode \
                 the stream with another decoder to tell which."
            ),
            Self::MissingReference { poc } => write!(
                f,
                "An HEVC picture refers to POC {poc}, which the decoded picture buffer does not \
                 hold. The input is cut or corrupt. Start from a random access point, or remux \
                 the source."
            ),
            Self::InvalidSyntax {
                element,
                value,
                allowed,
            } => write!(
                f,
                "HEVC slice data holds {element} equal to {value}, and the specification allows \
                 {allowed}. The input is corrupt. Remux the source, and re-encode it if the remux \
                 reports the same error."
            ),
        }
    }
}

impl StdError for HevcError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Header(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BitstreamError> for HevcError {
    fn from(error: BitstreamError) -> Self {
        Self::Header(error)
    }
}
