//! Length-prefixed NAL units, the sample format of MP4 (ISO/IEC 14496-15)
//! and Matroska.
//!
//! Each NAL unit follows a big-endian length field of 1, 2, or 4 bytes. The
//! config record's `lengthSizeMinusOne` gives the size.

use crate::error::BitstreamError;

const STRUCTURE: &str = "length-prefixed sample";

/// The size of a NAL unit length field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LengthSize {
    /// 1 byte, `lengthSizeMinusOne` 0.
    One,
    /// 2 bytes, `lengthSizeMinusOne` 1.
    Two,
    /// 4 bytes, `lengthSizeMinusOne` 3.
    Four,
}

impl LengthSize {
    /// The size for a `lengthSizeMinusOne` value. `None` for 2, which
    /// ISO/IEC 14496-15 does not allow, and for values above 3.
    pub const fn from_minus_one(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::One),
            1 => Some(Self::Two),
            3 => Some(Self::Four),
            _ => None,
        }
    }

    /// The size in bytes: 1, 2, or 4.
    pub const fn bytes(self) -> u8 {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Four => 4,
        }
    }

    /// The `lengthSizeMinusOne` value: 0, 1, or 3.
    pub const fn minus_one(self) -> u8 {
        self.bytes() - 1
    }

    /// The longest NAL unit the field can state.
    pub const fn max_length(self) -> u64 {
        match self {
            Self::One => u8::MAX as u64,
            Self::Two => u16::MAX as u64,
            Self::Four => u32::MAX as u64,
        }
    }
}

/// Splits a sample into its NAL units.
///
/// The iterator yields each unit without its length field. A zero length
/// yields an empty slice, and the caller decides whether to skip it. A length
/// field cut short, or a length past the end of the sample, yields one error
/// and ends the iteration.
pub fn split(sample: &[u8], length_size: LengthSize) -> NalUnits<'_> {
    NalUnits {
        data: sample,
        pos: 0,
        unit: 0,
        length_size,
        done: false,
    }
}

/// Iterator over the NAL units of a length-prefixed sample. See [`split`].
#[derive(Clone, Debug)]
pub struct NalUnits<'a> {
    data: &'a [u8],
    pos: usize,
    unit: usize,
    length_size: LengthSize,
    done: bool,
}

impl<'a> Iterator for NalUnits<'a> {
    type Item = Result<&'a [u8], BitstreamError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.pos >= self.data.len() {
            return None;
        }
        let item = self.read();
        self.done = item.is_err();
        Some(item)
    }
}

impl<'a> NalUnits<'a> {
    fn read(&mut self) -> Result<&'a [u8], BitstreamError> {
        let start = self.pos;
        let width = usize::from(self.length_size.bytes());
        let field = self
            .data
            .get(start..start + width)
            .ok_or(BitstreamError::Truncated {
                structure: STRUCTURE,
                field: "NAL unit length",
                offset: start,
                size: self.data.len(),
            })?;
        let length = field
            .iter()
            .fold(0u64, |acc, &byte| (acc << 8) | u64::from(byte));
        let body = start + width;
        let remaining = self.data.len() - body;
        if length > remaining as u64 {
            return Err(BitstreamError::NalUnitOverrun {
                unit: self.unit,
                offset: start,
                length,
                remaining,
            });
        }
        let end = body + length as usize;
        self.pos = end;
        self.unit += 1;
        Ok(&self.data[body..end])
    }
}

/// Appends one NAL unit with its length field.
///
/// # Errors
///
/// [`BitstreamError::EmptyNalUnit`] for an empty unit, and
/// [`BitstreamError::NalUnitTooLong`] when the length does not fit the
/// field. `out` is unchanged on an error.
pub fn push(out: &mut Vec<u8>, nal: &[u8], length_size: LengthSize) -> Result<(), BitstreamError> {
    push_unit(out, nal, length_size, 0)
}

fn push_unit(
    out: &mut Vec<u8>,
    nal: &[u8],
    length_size: LengthSize,
    unit: usize,
) -> Result<(), BitstreamError> {
    if nal.is_empty() {
        return Err(BitstreamError::EmptyNalUnit { unit });
    }
    if nal.len() as u64 > length_size.max_length() {
        return Err(BitstreamError::NalUnitTooLong {
            unit,
            length: nal.len(),
            length_size: length_size.bytes(),
        });
    }
    let length = (nal.len() as u32).to_be_bytes();
    out.extend_from_slice(&length[4 - usize::from(length_size.bytes())..]);
    out.extend_from_slice(nal);
    Ok(())
}

/// Appends NAL units, each with its length field, to `out`.
///
/// # Errors
///
/// As [`push`], with the index of the failing unit. `out` is unchanged on an
/// error.
pub fn join<'a, I>(
    units: I,
    length_size: LengthSize,
    out: &mut Vec<u8>,
) -> Result<(), BitstreamError>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    let start = out.len();
    for (unit, nal) in units.into_iter().enumerate() {
        if let Err(error) = push_unit(out, nal, length_size, unit) {
            out.truncate(start);
            return Err(error);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn units(sample: &[u8], size: LengthSize) -> Result<Vec<&[u8]>, BitstreamError> {
        split(sample, size).collect()
    }

    #[test]
    fn splits_each_length_size() {
        assert_eq!(
            units(&[2, 0xaa, 0xbb, 1, 0xcc], LengthSize::One),
            Ok(vec![&[0xaa, 0xbb][..], &[0xcc][..]])
        );
        assert_eq!(
            units(&[0, 1, 0xaa, 0, 0], LengthSize::Two),
            Ok(vec![&[0xaa][..], &[][..]])
        );
        assert_eq!(
            units(&[0, 0, 0, 2, 0xaa, 0xbb], LengthSize::Four),
            Ok(vec![&[0xaa, 0xbb][..]])
        );
        assert_eq!(units(&[], LengthSize::Four), Ok(vec![]));
    }

    #[test]
    fn length_past_the_end_is_an_error() {
        assert_eq!(
            units(&[0, 0, 0, 1, 0xaa, 0, 0, 0, 9, 1], LengthSize::Four),
            Err(BitstreamError::NalUnitOverrun {
                unit: 1,
                offset: 5,
                length: 9,
                remaining: 1,
            })
        );
        // A 4-byte length of u32::MAX on a 64-bit or 32-bit target.
        assert!(units(&[0xff, 0xff, 0xff, 0xff], LengthSize::Four).is_err());
    }

    #[test]
    fn cut_length_field_is_an_error() {
        assert_eq!(
            units(&[0, 0, 0, 1, 0xaa, 0, 0], LengthSize::Four),
            Err(BitstreamError::Truncated {
                structure: STRUCTURE,
                field: "NAL unit length",
                offset: 5,
                size: 7,
            })
        );
    }

    #[test]
    fn iteration_ends_after_an_error() {
        let mut iter = split(&[5, 1], LengthSize::One);
        assert!(matches!(iter.next(), Some(Err(_))));
        assert!(iter.next().is_none());
    }

    #[test]
    fn join_round_trips() {
        let nals: [&[u8]; 2] = [&[0x67, 1, 2], &[0x68]];
        for size in [LengthSize::One, LengthSize::Two, LengthSize::Four] {
            let mut out = Vec::new();
            assert_eq!(join(nals, size, &mut out), Ok(()));
            assert_eq!(units(&out, size), Ok(nals.to_vec()));
        }
    }

    #[test]
    fn join_refuses_and_restores() {
        let long = vec![1u8; 256];
        let mut out = vec![9];
        assert_eq!(
            join([&[1u8][..], &long[..]], LengthSize::One, &mut out),
            Err(BitstreamError::NalUnitTooLong {
                unit: 1,
                length: 256,
                length_size: 1,
            })
        );
        assert_eq!(out, vec![9]);
        assert_eq!(
            join([&[][..]], LengthSize::Four, &mut out),
            Err(BitstreamError::EmptyNalUnit { unit: 0 })
        );
    }

    #[test]
    fn length_size_values() {
        assert_eq!(LengthSize::from_minus_one(2), None);
        assert_eq!(LengthSize::from_minus_one(4), None);
        assert_eq!(LengthSize::from_minus_one(3), Some(LengthSize::Four));
        assert_eq!(LengthSize::Two.minus_one(), 1);
    }
}
