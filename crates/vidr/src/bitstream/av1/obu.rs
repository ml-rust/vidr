//! AV1 open bitstream units (AV1 specification 5.3): headers, `leb128()`
//! sizes, and the low-overhead bitstream format of section 5.2.

use crate::error::BitstreamError;

const STRUCTURE: &str = "OBU";

/// An AV1 `obu_type`, 0 to 15.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObuType(u8);

impl ObuType {
    /// Sequence header.
    pub const SEQUENCE_HEADER: Self = Self(1);
    /// Temporal delimiter.
    pub const TEMPORAL_DELIMITER: Self = Self(2);
    /// Frame header.
    pub const FRAME_HEADER: Self = Self(3);
    /// Tile group.
    pub const TILE_GROUP: Self = Self(4);
    /// Metadata.
    pub const METADATA: Self = Self(5);
    /// Frame: a frame header and a tile group.
    pub const FRAME: Self = Self(6);
    /// Redundant frame header.
    pub const REDUNDANT_FRAME_HEADER: Self = Self(7);
    /// Tile list.
    pub const TILE_LIST: Self = Self(8);
    /// Padding.
    pub const PADDING: Self = Self(15);

    /// The numeric `obu_type`.
    pub const fn value(self) -> u8 {
        self.0
    }
}

/// The optional `obu_extension_header()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ObuExtension {
    /// `temporal_id`, 0 to 7.
    pub temporal_id: u8,
    /// `spatial_id`, 0 to 3.
    pub spatial_id: u8,
}

/// A parsed `obu_header()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ObuHeader {
    /// `obu_type`.
    pub obu_type: ObuType,
    /// The extension header, when `obu_extension_flag` is 1.
    pub extension: Option<ObuExtension>,
    /// `obu_has_size_field`.
    pub has_size_field: bool,
}

/// One OBU borrowed from a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Obu<'a> {
    /// The parsed header.
    pub header: ObuHeader,
    /// The header bytes: 1, or 2 with the extension.
    pub header_bytes: &'a [u8],
    /// The payload after the header and size field.
    pub payload: &'a [u8],
    /// The whole OBU as stored: header, size field, and payload.
    pub bytes: &'a [u8],
}

impl Obu<'_> {
    /// Appends the OBU with `obu_has_size_field` 1 and a minimal
    /// `leb128()` size, the form NVDEC and ISO-BMFF samples take.
    ///
    /// The header bytes come from [`Obu::header`], so the reserved bits are
    /// written as 0 and `header_bytes` is not read.
    pub fn write_sized(&self, out: &mut Vec<u8>) {
        let extension_flag = u8::from(self.header.extension.is_some());
        out.push(((self.header.obu_type.value() & 0x0f) << 3) | (extension_flag << 2) | 0x02);
        if let Some(extension) = self.header.extension {
            out.push(((extension.temporal_id & 0x07) << 5) | ((extension.spatial_id & 0x03) << 3));
        }
        write_leb128(out, self.payload.len() as u64);
        out.extend_from_slice(self.payload);
    }
}

/// Reads a `leb128()` value at `offset`, and returns it with its length in
/// bytes.
///
/// # Errors
///
/// [`BitstreamError::Truncated`] when the data ends inside the value, and
/// [`BitstreamError::InvalidField`] for a value of more than 8 bytes or above
/// `2^32 - 1`, which the AV1 specification forbids.
pub fn read_leb128(
    data: &[u8],
    offset: usize,
    field: &'static str,
) -> Result<(u64, usize), BitstreamError> {
    let mut value = 0u64;
    for index in 0..8 {
        let byte = *data.get(offset + index).ok_or(BitstreamError::Truncated {
            structure: STRUCTURE,
            field,
            offset: offset + index,
            size: data.len(),
        })?;
        value |= u64::from(byte & 0x7f) << (7 * index);
        if byte & 0x80 == 0 {
            if value > u64::from(u32::MAX) {
                return Err(BitstreamError::InvalidField {
                    structure: STRUCTURE,
                    field,
                    value,
                    allowed: "at most 2^32 - 1",
                });
            }
            return Ok((value, index + 1));
        }
    }
    Err(BitstreamError::InvalidField {
        structure: STRUCTURE,
        field,
        value,
        allowed: "a leb128() value of at most 8 bytes",
    })
}

/// Appends `value` as a minimal `leb128()`.
pub fn write_leb128(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Splits low-overhead AV1 data into OBUs.
///
/// An OBU without a size field runs to the end of the data. A parse error
/// yields one error and ends the iteration.
pub fn split(data: &[u8]) -> Obus<'_> {
    Obus {
        data,
        pos: 0,
        done: false,
    }
}

/// Iterator over the OBUs of low-overhead data. See [`split`].
#[derive(Clone, Debug)]
pub struct Obus<'a> {
    data: &'a [u8],
    pos: usize,
    done: bool,
}

impl<'a> Iterator for Obus<'a> {
    type Item = Result<Obu<'a>, BitstreamError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.pos >= self.data.len() {
            return None;
        }
        let item = self.read();
        self.done = item.is_err();
        Some(item)
    }
}

impl<'a> Obus<'a> {
    fn read(&mut self) -> Result<Obu<'a>, BitstreamError> {
        let start = self.pos;
        let first = self.data[start];
        if first & 0x80 != 0 {
            return Err(BitstreamError::InvalidField {
                structure: STRUCTURE,
                field: "obu_forbidden_bit",
                value: 1,
                allowed: "0",
            });
        }
        let has_extension = first & 0x04 != 0;
        let has_size_field = first & 0x02 != 0;
        let header_len = 1 + usize::from(has_extension);
        let header_bytes =
            self.data
                .get(start..start + header_len)
                .ok_or(BitstreamError::Truncated {
                    structure: STRUCTURE,
                    field: "obu_extension_header",
                    offset: start + 1,
                    size: self.data.len(),
                })?;
        let extension = has_extension.then(|| ObuExtension {
            temporal_id: header_bytes[1] >> 5,
            spatial_id: (header_bytes[1] >> 3) & 0x03,
        });
        let mut body = start + header_len;
        let end = if has_size_field {
            let (size, width) = read_leb128(self.data, body, "obu_size")?;
            body += width;
            let remaining = self.data.len() - body;
            if size > remaining as u64 {
                return Err(BitstreamError::Truncated {
                    structure: STRUCTURE,
                    field: "obu_size",
                    offset: body + remaining,
                    size: self.data.len(),
                });
            }
            body + size as usize
        } else {
            self.data.len()
        };
        self.pos = end;
        Ok(Obu {
            header: ObuHeader {
                obu_type: ObuType((first >> 3) & 0x0f),
                extension,
                has_size_field,
            },
            header_bytes,
            payload: &self.data[body..end],
            bytes: &self.data[start..end],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obus(data: &[u8]) -> Result<Vec<Obu<'_>>, BitstreamError> {
        split(data).collect()
    }

    #[test]
    fn leb128_round_trips_and_is_minimal() {
        for value in [0u64, 1, 127, 128, 300, 16_383, 16_384, u64::from(u32::MAX)] {
            let mut out = Vec::new();
            write_leb128(&mut out, value);
            assert_eq!(read_leb128(&out, 0, "v"), Ok((value, out.len())));
        }
        let mut out = Vec::new();
        write_leb128(&mut out, 300);
        assert_eq!(out, vec![0xac, 0x02]);
        // A padded encoding of 1 is valid input.
        assert_eq!(read_leb128(&[0x81, 0x80, 0x00], 0, "v"), Ok((1, 3)));
    }

    #[test]
    fn leb128_limits() {
        assert!(matches!(
            read_leb128(&[0x80; 9], 0, "v"),
            Err(BitstreamError::InvalidField { .. })
        ));
        assert!(matches!(
            read_leb128(&[0x80, 0x80, 0x80, 0x80, 0x10], 0, "v"),
            Err(BitstreamError::InvalidField {
                allowed: "at most 2^32 - 1",
                ..
            })
        ));
        assert!(matches!(
            read_leb128(&[0x80], 0, "v"),
            Err(BitstreamError::Truncated { .. })
        ));
    }

    #[test]
    fn splits_sized_and_unsized_obus() {
        // Temporal delimiter, a sized frame with an extension, an unsized frame.
        let data = [0x12, 0x00, 0x36, 0x28, 0x02, 0xaa, 0xbb, 0x30, 0xcc, 0xdd];
        let Ok(parsed) = obus(&data) else {
            panic!("split failed");
        };
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].header.obu_type, ObuType::TEMPORAL_DELIMITER);
        assert!(parsed[0].payload.is_empty());
        assert_eq!(parsed[1].header.obu_type, ObuType::FRAME);
        assert_eq!(
            parsed[1].header.extension,
            Some(ObuExtension {
                temporal_id: 1,
                spatial_id: 1,
            })
        );
        assert_eq!(parsed[1].payload, &[0xaa, 0xbb]);
        assert!(!parsed[2].header.has_size_field);
        assert_eq!(parsed[2].payload, &[0xcc, 0xdd]);
    }

    #[test]
    fn write_sized_adds_the_size_field() {
        let data = [0x30, 0xcc, 0xdd];
        let Ok(parsed) = obus(&data) else {
            panic!("split failed");
        };
        let mut out = Vec::new();
        parsed[0].write_sized(&mut out);
        assert_eq!(out, vec![0x32, 0x02, 0xcc, 0xdd]);
    }

    #[test]
    fn corrupt_obus_are_errors() {
        assert!(matches!(
            obus(&[0x92, 0x00]),
            Err(BitstreamError::InvalidField {
                field: "obu_forbidden_bit",
                ..
            })
        ));
        assert!(matches!(
            obus(&[0x32, 0x05, 1]),
            Err(BitstreamError::Truncated {
                field: "obu_size",
                ..
            })
        ));
        assert!(obus(&[0x36]).is_err());
        assert!(obus(&[0x32]).is_err());
    }
}
