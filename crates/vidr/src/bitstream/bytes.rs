//! A bounds-checked big-endian byte reader for config records.

use crate::error::BitstreamError;

/// Reads big-endian fields from a byte slice. Every read names its field.
pub(crate) struct ByteReader<'a> {
    data: &'a [u8],
    pos: usize,
    structure: &'static str,
}

impl<'a> ByteReader<'a> {
    /// Creates a reader at the first byte of `data`.
    pub(crate) const fn new(data: &'a [u8], structure: &'static str) -> Self {
        Self {
            data,
            pos: 0,
            structure,
        }
    }

    /// Bytes not yet read.
    pub(crate) fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    /// Reads `count` bytes.
    pub(crate) fn take(
        &mut self,
        count: usize,
        field: &'static str,
    ) -> Result<&'a [u8], BitstreamError> {
        if count > self.remaining() {
            return Err(BitstreamError::Truncated {
                structure: self.structure,
                field,
                offset: self.pos,
                size: self.data.len(),
            });
        }
        let bytes = &self.data[self.pos..self.pos + count];
        self.pos += count;
        Ok(bytes)
    }

    /// Reads one byte.
    pub(crate) fn u8(&mut self, field: &'static str) -> Result<u8, BitstreamError> {
        Ok(self.take(1, field)?[0])
    }

    /// Reads a big-endian `u16`.
    #[cfg(any(feature = "h264", feature = "hevc"))]
    pub(crate) fn u16(&mut self, field: &'static str) -> Result<u16, BitstreamError> {
        let bytes = self.take(2, field)?;
        Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
    }

    /// Reads the bytes left.
    #[cfg(feature = "av1")]
    pub(crate) fn rest(&mut self) -> &'a [u8] {
        let bytes = &self.data[self.pos..];
        self.pos = self.data.len();
        bytes
    }

    /// Returns [`BitstreamError::TrailingBytes`] when bytes remain.
    #[cfg(any(feature = "h264", feature = "hevc"))]
    pub(crate) fn finish(&self) -> Result<(), BitstreamError> {
        if self.remaining() > 0 {
            return Err(BitstreamError::TrailingBytes {
                structure: self.structure,
                offset: self.pos,
                size: self.data.len(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(feature = "h264", feature = "hevc"))]
    #[test]
    fn reads_fields_and_reports_the_rest() {
        let mut reader = ByteReader::new(&[1, 0x12, 0x34, 9, 9], "test");
        assert_eq!(reader.u8("a"), Ok(1));
        assert_eq!(reader.take(2, "b"), Ok(&[0x12, 0x34][..]));
        assert_eq!(
            reader.finish(),
            Err(BitstreamError::TrailingBytes {
                structure: "test",
                offset: 3,
                size: 5,
            })
        );
        assert_eq!(
            reader.take(3, "c"),
            Err(BitstreamError::Truncated {
                structure: "test",
                field: "c",
                offset: 3,
                size: 5,
            })
        );
        assert_eq!(reader.take(2, "d"), Ok(&[9, 9][..]));
        assert_eq!(reader.finish(), Ok(()));
    }

    #[cfg(any(feature = "h264", feature = "hevc"))]
    #[test]
    fn reads_big_endian_u16() {
        let mut reader = ByteReader::new(&[0x12, 0x34], "test");
        assert_eq!(reader.u16("a"), Ok(0x1234));
        assert!(reader.u16("b").is_err());
    }

    #[cfg(feature = "av1")]
    #[test]
    fn rest_takes_every_byte_left() {
        let mut reader = ByteReader::new(&[1, 2, 3], "test");
        assert_eq!(reader.u8("a"), Ok(1));
        assert_eq!(reader.rest(), &[2, 3]);
        assert_eq!(reader.remaining(), 0);
    }
}
