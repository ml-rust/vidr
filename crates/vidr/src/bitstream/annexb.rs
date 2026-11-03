//! Annex-B byte streams (ITU-T H.264 and H.265 annex B): NAL units after
//! start codes.
//!
//! A start code is `00 00 01`. A `zero_byte` before it makes the 4-byte form
//! `00 00 00 01`. Zero bytes after a NAL unit (`trailing_zero_8bits`) belong
//! to the byte stream, never to the unit: annex B forbids a NAL unit that
//! ends in `00`. The emulation prevention byte `03` in `00 00 03` keeps
//! payload bytes from forming a start code, so a unit that ends in
//! `00 00 03` keeps its `03`.

use crate::error::BitstreamError;

/// The start code form to write before a NAL unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StartCode {
    /// `00 00 01`.
    Three,
    /// `00 00 00 01`.
    Four,
}

impl StartCode {
    /// The start code bytes.
    pub const fn bytes(self) -> &'static [u8] {
        match self {
            Self::Three => &[0, 0, 1],
            Self::Four => &[0, 0, 0, 1],
        }
    }
}

/// Finds the next `00 00 01` at or after `from`, and returns the index of
/// its first byte.
fn find_start_code(data: &[u8], from: usize) -> Option<usize> {
    let mut i = from;
    while i + 2 < data.len() {
        match data[i + 2] {
            // No start code can begin at i, i + 1, or i + 2.
            2.. => i += 3,
            1 if data[i] == 0 && data[i + 1] == 0 => return Some(i),
            // A start code at i + 1 or i + 2 needs data[i + 2] == 0.
            1 => i += 3,
            0 => i += 1,
        }
    }
    None
}

/// Splits Annex-B data into NAL units.
///
/// Leading zero bytes are allowed. Units come without start codes and
/// without trailing zero bytes. A start code followed by nothing but zeros
/// yields no unit.
///
/// # Errors
///
/// [`BitstreamError::MissingStartCode`] when a nonzero byte comes before the
/// first start code.
pub fn split(data: &[u8]) -> Result<AnnexBUnits<'_>, BitstreamError> {
    let first = find_start_code(data, 0);
    let lead = first.unwrap_or(data.len());
    if let Some(offset) = data[..lead].iter().position(|&byte| byte != 0) {
        return Err(BitstreamError::MissingStartCode { offset });
    }
    Ok(AnnexBUnits { data, next: first })
}

/// Iterator over the NAL units of Annex-B data. See [`split`].
#[derive(Clone, Debug)]
pub struct AnnexBUnits<'a> {
    data: &'a [u8],
    /// Index of the next start code.
    next: Option<usize>,
}

impl<'a> Iterator for AnnexBUnits<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(code) = self.next {
            let start = code + 3;
            self.next = find_start_code(self.data, start);
            let end = self.next.unwrap_or(self.data.len());
            let unit = &self.data[start..end];
            let kept = unit
                .iter()
                .rposition(|&byte| byte != 0)
                .map_or(0, |last| last + 1);
            if kept > 0 {
                return Some(&unit[..kept]);
            }
        }
        None
    }
}

/// Each unit after a 4-byte start code. Units are not checked.
pub(crate) fn framed<'a>(units: impl IntoIterator<Item = &'a [u8]>) -> Vec<u8> {
    let mut out = Vec::new();
    for unit in units {
        out.extend_from_slice(StartCode::Four.bytes());
        out.extend_from_slice(unit);
    }
    out
}

/// Appends one NAL unit after a start code.
///
/// The unit is written as given. Its emulation prevention bytes are the
/// caller's.
///
/// # Errors
///
/// [`BitstreamError::EmptyNalUnit`] for an empty unit. `out` is unchanged on
/// an error.
pub fn push(out: &mut Vec<u8>, nal: &[u8], start_code: StartCode) -> Result<(), BitstreamError> {
    if nal.is_empty() {
        return Err(BitstreamError::EmptyNalUnit { unit: 0 });
    }
    out.extend_from_slice(start_code.bytes());
    out.extend_from_slice(nal);
    Ok(())
}

/// Appends NAL units to `out`, each after a 4-byte start code.
///
/// # Errors
///
/// [`BitstreamError::EmptyNalUnit`] with the index of an empty unit. `out`
/// is unchanged on an error.
pub fn join<'a, I>(units: I, out: &mut Vec<u8>) -> Result<(), BitstreamError>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    let start = out.len();
    for (unit, nal) in units.into_iter().enumerate() {
        if nal.is_empty() {
            out.truncate(start);
            return Err(BitstreamError::EmptyNalUnit { unit });
        }
        out.extend_from_slice(StartCode::Four.bytes());
        out.extend_from_slice(nal);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn units(data: &[u8]) -> Vec<&[u8]> {
        match split(data) {
            Ok(iter) => iter.collect(),
            Err(error) => panic!("split failed: {error}"),
        }
    }

    #[test]
    fn three_and_four_byte_start_codes() {
        let data = [
            0, 0, 0, 1, 0x67, 0xaa, 0, 0, 1, 0x68, 0xbb, 0, 0, 0, 1, 0x65,
        ];
        assert_eq!(
            units(&data),
            vec![&[0x67, 0xaa][..], &[0x68, 0xbb][..], &[0x65][..]]
        );
    }

    #[test]
    fn leading_zeros_and_trailing_zeros_are_framing() {
        let data = [
            0, 0, 0, 0, 0, 1, 0x09, 0xf0, 0, 0, 0, 0, 0, 1, 0x65, 0x88, 0, 0,
        ];
        assert_eq!(units(&data), vec![&[0x09, 0xf0][..], &[0x65, 0x88][..]]);
    }

    #[test]
    fn emulation_prevention_byte_stays_in_the_unit() {
        // A unit ending in a cabac_zero_word `00 00 03`, then a 4-byte code.
        let data = [0, 0, 1, 0x65, 0x11, 0, 0, 3, 0, 0, 0, 1, 0x41, 0, 0, 3, 1];
        assert_eq!(
            units(&data),
            vec![&[0x65, 0x11, 0, 0, 3][..], &[0x41, 0, 0, 3, 1][..]]
        );
    }

    #[test]
    fn empty_units_are_skipped() {
        let data = [0, 0, 1, 0, 0, 1, 0x67, 0, 0, 1];
        assert_eq!(units(&data), vec![&[0x67][..]]);
        assert!(units(&[]).is_empty());
        assert!(units(&[0, 0]).is_empty());
    }

    #[test]
    fn data_before_the_first_start_code_is_an_error() {
        assert_eq!(
            split(&[0, 7, 0, 0, 1, 0x67]).err(),
            Some(BitstreamError::MissingStartCode { offset: 1 })
        );
        assert_eq!(
            split(&[0x67, 0x42]).err(),
            Some(BitstreamError::MissingStartCode { offset: 0 })
        );
    }

    #[test]
    fn start_code_search_skips_near_misses() {
        assert_eq!(find_start_code(&[0, 1, 0, 0, 2, 0, 0, 1], 0), Some(5));
        assert_eq!(find_start_code(&[1, 0, 0, 1], 0), Some(1));
        assert_eq!(find_start_code(&[0, 0, 0, 0], 0), None);
        assert_eq!(find_start_code(&[0, 0, 1], 1), None);
    }

    #[test]
    fn join_and_push_frame_units() {
        let mut out = Vec::new();
        assert_eq!(join([&[0x67u8][..], &[0x68][..]], &mut out), Ok(()));
        assert_eq!(push(&mut out, &[0x65], StartCode::Three), Ok(()));
        assert_eq!(out, vec![0, 0, 0, 1, 0x67, 0, 0, 0, 1, 0x68, 0, 0, 1, 0x65]);
        assert_eq!(units(&out), vec![&[0x67][..], &[0x68][..], &[0x65][..]]);
        assert_eq!(
            join([&[0x67u8][..], &[][..]], &mut out),
            Err(BitstreamError::EmptyNalUnit { unit: 1 })
        );
        assert_eq!(out.len(), 14);
        assert!(push(&mut out, &[], StartCode::Four).is_err());
    }
}
