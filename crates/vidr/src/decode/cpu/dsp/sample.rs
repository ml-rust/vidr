//! The sample types of a decoded plane.

use std::collections::TryReserveError;
use std::fmt::Debug;

/// One plane sample. `u8` holds 8-bit video and `u16` holds 9-bit and 10-bit
/// video, so 8-bit video keeps the smaller type.
pub(crate) trait Sample: Copy + Default + Eq + Debug + Send + Sync + 'static {
    /// Bytes of one sample in host planes.
    const BYTES: usize;

    /// This sample as an `i32`.
    fn to_i32(self) -> i32;

    /// `Clip3( 0, max, value )` (5.8). The caller passes `max` from a checked
    /// bit depth, so the result fits the sample type.
    fn clipped(value: i32, max: i32) -> Self;

    /// The little-endian bytes of `samples` for a host plane. `u8` borrows
    /// `samples` and leaves `scratch` alone. `u16` writes `scratch`.
    fn host_bytes<'a>(
        samples: &'a [Self],
        scratch: &'a mut Vec<u8>,
    ) -> Result<&'a [u8], TryReserveError>;
}

impl Sample for u8 {
    const BYTES: usize = 1;

    #[inline]
    fn to_i32(self) -> i32 {
        i32::from(self)
    }

    #[inline]
    fn clipped(value: i32, max: i32) -> Self {
        value.max(0).min(max) as u8
    }

    fn host_bytes<'a>(
        samples: &'a [Self],
        _scratch: &'a mut Vec<u8>,
    ) -> Result<&'a [u8], TryReserveError> {
        Ok(samples)
    }
}

impl Sample for u16 {
    const BYTES: usize = 2;

    #[inline]
    fn to_i32(self) -> i32 {
        i32::from(self)
    }

    #[inline]
    fn clipped(value: i32, max: i32) -> Self {
        value.max(0).min(max) as u16
    }

    fn host_bytes<'a>(
        samples: &'a [Self],
        scratch: &'a mut Vec<u8>,
    ) -> Result<&'a [u8], TryReserveError> {
        scratch.clear();
        scratch.try_reserve_exact(samples.len() * Self::BYTES)?;
        scratch.extend(samples.iter().flat_map(|sample| sample.to_le_bytes()));
        Ok(scratch.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipped_holds_the_range() {
        assert_eq!(u8::clipped(-3, 255), 0);
        assert_eq!(u8::clipped(300, 255), 255);
        assert_eq!(u8::clipped(17, 255), 17);
        assert_eq!(u16::clipped(-1, 1023), 0);
        assert_eq!(u16::clipped(5000, 1023), 1023);
        assert_eq!(u16::clipped(511, 511), 511);
    }

    #[test]
    fn eight_bit_samples_borrow() {
        let samples = [1u8, 2, 3];
        let mut scratch = Vec::new();
        let bytes = u8::host_bytes(&samples, &mut scratch);
        assert_eq!(bytes, Ok(&samples[..]));
    }

    #[test]
    fn ten_bit_samples_write_little_endian() {
        let samples = [0x0102u16, 0x03ff];
        let mut scratch = vec![9u8; 40];
        let bytes = u16::host_bytes(&samples, &mut scratch);
        assert_eq!(bytes, Ok(&[0x02u8, 0x01, 0xff, 0x03][..]));
    }
}
