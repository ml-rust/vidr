//! A most-significant-bit-first reader for header syntax.
//!
//! Every read names its field, so a short buffer returns
//! [`BitstreamError::Truncated`] with that name.

use crate::error::BitstreamError;

/// Reads bits from a byte slice, most significant bit first.
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// Position in bits from the start of `data`.
    pos: u64,
    structure: &'static str,
}

impl<'a> BitReader<'a> {
    /// Creates a reader at the first bit of `data`.
    pub(crate) const fn new(data: &'a [u8], structure: &'static str) -> Self {
        Self {
            data,
            pos: 0,
            structure,
        }
    }

    /// Bits not yet read.
    pub(crate) fn bits_left(&self) -> u64 {
        (self.data.len() as u64 * 8).saturating_sub(self.pos)
    }

    /// The error for a read of `field` past the end.
    pub(crate) fn truncated(&self, field: &'static str) -> BitstreamError {
        BitstreamError::Truncated {
            structure: self.structure,
            field,
            offset: (self.pos / 8) as usize,
            size: self.data.len(),
        }
    }

    /// The error for a `field` that holds a forbidden `value`.
    #[cfg(any(feature = "hevc", feature = "vp9", feature = "av1"))]
    pub(crate) const fn invalid(
        &self,
        field: &'static str,
        value: u64,
        allowed: &'static str,
    ) -> BitstreamError {
        BitstreamError::InvalidField {
            structure: self.structure,
            field,
            value,
            allowed,
        }
    }

    /// Bits read so far.
    #[cfg(any(feature = "hevc", feature = "vp9", feature = "av1"))]
    pub(crate) const fn position(&self) -> u64 {
        self.pos
    }

    /// Reads one bit.
    pub(crate) fn bit(&mut self, field: &'static str) -> Result<bool, BitstreamError> {
        let byte = self
            .data
            .get((self.pos / 8) as usize)
            .ok_or_else(|| self.truncated(field))?;
        let bit = (byte >> (7 - (self.pos % 8))) & 1;
        self.pos += 1;
        Ok(bit == 1)
    }

    /// Reads `count` bits, at most 64, as an unsigned value.
    pub(crate) fn bits(&mut self, count: u32, field: &'static str) -> Result<u64, BitstreamError> {
        debug_assert!(count <= 64, "a read holds at most 64 bits");
        if u64::from(count) > self.bits_left() {
            return Err(self.truncated(field));
        }
        let mut value = 0u64;
        for _ in 0..count {
            value = (value << 1) | u64::from(self.bit(field)?);
        }
        Ok(value)
    }

    /// Reads `count` bits, at most 8.
    pub(crate) fn u8(&mut self, count: u32, field: &'static str) -> Result<u8, BitstreamError> {
        debug_assert!(count <= 8, "a u8 read holds at most 8 bits");
        Ok(self.bits(count.min(8), field)? as u8)
    }

    /// Skips `count` bits.
    #[cfg(any(test, feature = "h264", feature = "hevc"))]
    pub(crate) fn skip(&mut self, count: u64, field: &'static str) -> Result<(), BitstreamError> {
        if count > self.bits_left() {
            return Err(self.truncated(field));
        }
        self.pos += count;
        Ok(())
    }

    /// Reads an unsigned Exp-Golomb code, `ue(v)` in H.264 and HEVC.
    ///
    /// A code with more than 31 leading zero bits does not fit a `u32` and
    /// returns [`BitstreamError::InvalidField`].
    #[cfg(any(feature = "h264", feature = "hevc"))]
    pub(crate) fn ue(&mut self, field: &'static str) -> Result<u32, BitstreamError> {
        let mut zeros = 0u32;
        while !self.bit(field)? {
            zeros += 1;
            if zeros > 31 {
                return Err(BitstreamError::InvalidField {
                    structure: self.structure,
                    field,
                    value: u64::from(zeros),
                    allowed: "an Exp-Golomb code with at most 31 leading zero bits",
                });
            }
        }
        let suffix = self.bits(zeros, field)?;
        Ok(((1u64 << zeros) - 1 + suffix) as u32)
    }

    /// Reads a signed Exp-Golomb code, `se(v)` in H.264 and HEVC.
    #[cfg(any(feature = "h264", feature = "hevc"))]
    pub(crate) fn se(&mut self, field: &'static str) -> Result<i64, BitstreamError> {
        let code = i64::from(self.ue(field)?);
        Ok(if code % 2 == 1 {
            (code + 1) / 2
        } else {
            -(code / 2)
        })
    }

    /// Reads a `ue(v)` and refuses a value above `max`.
    #[cfg(feature = "hevc")]
    pub(crate) fn ue_max(
        &mut self,
        field: &'static str,
        max: u32,
        allowed: &'static str,
    ) -> Result<u32, BitstreamError> {
        let value = self.ue(field)?;
        if value > max {
            return Err(self.invalid(field, u64::from(value), allowed));
        }
        Ok(value)
    }

    /// Reads a `se(v)` and refuses a value outside `min..=max`.
    #[cfg(feature = "hevc")]
    pub(crate) fn se_range(
        &mut self,
        field: &'static str,
        min: i64,
        max: i64,
        allowed: &'static str,
    ) -> Result<i64, BitstreamError> {
        let value = self.se(field)?;
        if !(min..=max).contains(&value) {
            return Err(BitstreamError::InvalidSignedField {
                structure: self.structure,
                field,
                value,
                allowed,
            });
        }
        Ok(value)
    }

    /// Reads an AV1 `uvlc()` value.
    ///
    /// A code with 32 or more leading zero bits is `u32::MAX`, as the AV1
    /// specification defines.
    #[cfg(feature = "av1")]
    pub(crate) fn uvlc(&mut self, field: &'static str) -> Result<u32, BitstreamError> {
        let mut zeros = 0u32;
        while !self.bit(field)? {
            zeros += 1;
        }
        if zeros >= 32 {
            return Ok(u32::MAX);
        }
        let suffix = self.bits(zeros, field)?;
        Ok(((1u64 << zeros) - 1 + suffix) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_bits_across_bytes() {
        let mut reader = BitReader::new(&[0b1010_1100, 0b0101_0000], "test");
        assert_eq!(reader.bits(3, "a"), Ok(0b101));
        assert_eq!(reader.bits(7, "b"), Ok(0b011_0001));
        assert_eq!(reader.bits(6, "c"), Ok(0b01_0000));
        assert!(reader.bit("d").is_err());
    }

    #[test]
    fn short_read_names_the_field() {
        let mut reader = BitReader::new(&[0xff], "test");
        assert_eq!(
            reader.bits(9, "wide"),
            Err(BitstreamError::Truncated {
                structure: "test",
                field: "wide",
                offset: 0,
                size: 1,
            })
        );
    }

    #[cfg(any(feature = "hevc", feature = "av1"))]
    #[test]
    fn skip_past_the_end_is_an_error() {
        let mut reader = BitReader::new(&[0xff], "test");
        assert!(reader.skip(9, "skip").is_err());
        assert_eq!(reader.skip(8, "skip"), Ok(()));
    }

    #[cfg(any(feature = "h264", feature = "hevc"))]
    #[test]
    fn exp_golomb_codes_decode() {
        // 1 | 010 | 011 | 00100, then zeros to the end.
        let mut reader = BitReader::new(&[0b1010_0110, 0b0100_0000, 0, 0], "test");
        assert_eq!(reader.ue("a"), Ok(0));
        assert_eq!(reader.ue("b"), Ok(1));
        assert_eq!(reader.ue("c"), Ok(2));
        assert_eq!(reader.ue("d"), Ok(3));
        assert!(reader.ue("e").is_err());
    }

    #[cfg(any(feature = "h264", feature = "hevc"))]
    #[test]
    fn exp_golomb_with_32_zeros_is_refused() {
        let mut reader = BitReader::new(&[0, 0, 0, 0, 0xff], "test");
        assert!(matches!(
            reader.ue("long"),
            Err(BitstreamError::InvalidField { field: "long", .. })
        ));
    }

    #[cfg(feature = "hevc")]
    #[test]
    fn signed_exp_golomb_alternates_sign() {
        // codes 1, 2, 3, 4: 010 011 00100 00101
        let mut reader = BitReader::new(&[0b0100_1100, 0b1000_0101], "test");
        assert_eq!(reader.se("a"), Ok(1));
        assert_eq!(reader.se("b"), Ok(-1));
        assert_eq!(reader.se("c"), Ok(2));
        assert_eq!(reader.se("d"), Ok(-2));
    }

    #[cfg(feature = "av1")]
    #[test]
    fn uvlc_saturates_at_32_zeros() {
        let mut reader = BitReader::new(&[0, 0, 0, 0, 0x80], "test");
        assert_eq!(reader.uvlc("u"), Ok(u32::MAX));
        let mut reader = BitReader::new(&[0b0010_1000], "test");
        assert_eq!(reader.uvlc("u"), Ok(4));
    }
}
