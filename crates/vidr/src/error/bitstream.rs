//! Errors in bitstream filters and codec config records.

use std::error::Error as StdError;
use std::fmt;

/// An error in a bitstream filter, a NAL unit or OBU framing, or a codec
/// config record.
///
/// `structure` names the syntax structure, for example `avcC` or
/// `HEVC SPS`. `field` uses the field name from the specification that
/// defines the structure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BitstreamError {
    /// A structure ends before one of its fields.
    Truncated {
        /// The structure.
        structure: &'static str,
        /// The field that does not fit.
        field: &'static str,
        /// Byte offset of the field in the structure.
        offset: usize,
        /// Size of the structure in bytes.
        size: usize,
    },
    /// A field holds a value its specification forbids.
    InvalidField {
        /// The structure.
        structure: &'static str,
        /// The field.
        field: &'static str,
        /// The value read or passed.
        value: u64,
        /// The values the specification allows.
        allowed: &'static str,
    },
    /// A signed field holds a value its specification forbids.
    InvalidSignedField {
        /// The structure.
        structure: &'static str,
        /// The field.
        field: &'static str,
        /// The value read.
        value: i64,
        /// The values the specification allows.
        allowed: &'static str,
    },
    /// Bytes follow the last field of a structure.
    TrailingBytes {
        /// The structure.
        structure: &'static str,
        /// Byte offset of the first extra byte.
        offset: usize,
        /// Size of the structure in bytes.
        size: usize,
    },
    /// A length-prefixed NAL unit runs past the end of its sample.
    NalUnitOverrun {
        /// Index of the NAL unit in the sample.
        unit: usize,
        /// Byte offset of its length field.
        offset: usize,
        /// The length the field declares.
        length: u64,
        /// Bytes left in the sample after the length field.
        remaining: usize,
    },
    /// A NAL unit is too long for the length field size.
    NalUnitTooLong {
        /// Index of the NAL unit.
        unit: usize,
        /// Its length in bytes.
        length: usize,
        /// The length field size in bytes.
        length_size: u8,
    },
    /// A NAL unit to write is empty.
    EmptyNalUnit {
        /// Index of the NAL unit.
        unit: usize,
    },
    /// Annex-B data holds a nonzero byte before its first start code.
    MissingStartCode {
        /// Byte offset of the nonzero byte.
        offset: usize,
    },
    /// The first access unit lacks a parameter set the config record needs.
    MissingParameterSet {
        /// The codec: `H.264`, `HEVC`, or `AV1`.
        codec: &'static str,
        /// The parameter set kind, for example `SPS` or `sequence header`.
        kind: &'static str,
        /// Index of the access unit, counted from 0.
        access_unit: u64,
    },
    /// An access unit carries a parameter set the config record does not
    /// hold.
    ParameterSetChanged {
        /// The codec: `H.264`, `HEVC`, or `AV1`.
        codec: &'static str,
        /// The parameter set kind, for example `SPS` or `sequence header`.
        kind: &'static str,
        /// Index of the access unit, counted from 0.
        access_unit: u64,
    },
}

impl fmt::Display for BitstreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated {
                structure,
                field,
                offset,
                size,
            } => write!(
                f,
                "{structure} ends after {size} bytes, before field `{field}` at byte {offset}. \
                 The input is truncated or corrupt. Remux the source, and re-encode it if the \
                 remux reports the same error."
            ),
            Self::InvalidField {
                structure,
                field,
                value,
                allowed,
            } => write!(
                f,
                "{structure} field `{field}` is {value}. It must be {allowed}. \
                 The input is corrupt or does not follow its specification. Remux the source, \
                 and re-encode it if the remux reports the same error."
            ),
            Self::InvalidSignedField {
                structure,
                field,
                value,
                allowed,
            } => write!(
                f,
                "{structure} field `{field}` is {value}. It must be {allowed}. \
                 The input is corrupt or does not follow its specification. Remux the source, \
                 and re-encode it if the remux reports the same error."
            ),
            Self::TrailingBytes {
                structure,
                offset,
                size,
            } => write!(
                f,
                "{structure} holds {} bytes after its last field, from byte {offset} of {size}. \
                 Pass exactly one {structure}, without padding.",
                size.saturating_sub(*offset)
            ),
            Self::NalUnitOverrun {
                unit,
                offset,
                length,
                remaining,
            } => write!(
                f,
                "NAL unit {unit} declares {length} bytes at byte {offset}, but {remaining} bytes \
                 remain in the sample. Check that the length size matches the config record's \
                 `lengthSizeMinusOne`, and remux the source if it does."
            ),
            Self::NalUnitTooLong {
                unit,
                length,
                length_size,
            } => write!(
                f,
                "NAL unit {unit} holds {length} bytes, more than a {length_size}-byte length \
                 field can state. Write the samples with a 4-byte length size."
            ),
            Self::EmptyNalUnit { unit } => write!(
                f,
                "NAL unit {unit} is empty, and an empty NAL unit has no valid framing. \
                 Remove it from the input."
            ),
            Self::MissingStartCode { offset } => write!(
                f,
                "Annex-B data holds a nonzero byte at {offset}, before its first start code. \
                 Pass data that starts with a start code `00 00 01` or `00 00 00 01`."
            ),
            Self::MissingParameterSet {
                codec,
                kind,
                access_unit,
            } => write!(
                f,
                "{codec} access unit {access_unit} carries no {kind}, so the config record \
                 cannot be built. Start the stream at a random access point that carries its \
                 parameter sets in band."
            ),
            Self::ParameterSetChanged {
                codec,
                kind,
                access_unit,
            } => write!(
                f,
                "{codec} access unit {access_unit} carries a {kind} that the config record \
                 does not hold. The mux needs a new sample description from this access unit \
                 on, and vidr does not write one. Encode with fixed parameter sets, or split \
                 the output at this access unit."
            ),
        }
    }
}

impl StdError for BitstreamError {}
