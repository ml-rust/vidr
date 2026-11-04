//! The sample types of a decoded plane.

use std::collections::TryReserveError;
use std::fmt::Debug;

use fearless_simd::{Bytes, Simd, SimdFrom, i16x8, i16x16, u8x16, u16x8, u16x16};

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

    /// Eight samples as 16-bit lanes. Bit depths up to 10 keep every value.
    fn load_x8<V: Simd>(simd: V, src: &[Self; 8]) -> i16x8<V>;

    /// Sixteen samples as 16-bit lanes, as [`Sample::load_x8`] reads them.
    fn load_x16<V: Simd>(simd: V, src: &[Self; 16]) -> i16x16<V>;

    /// Eight 16-bit lanes as samples. Each lane already lies within the sample range.
    fn pack_x8<V: Simd>(simd: V, value: i16x8<V>) -> [Self; 8];

    /// Sixteen 16-bit lanes as samples, as [`Sample::pack_x8`] takes them.
    fn pack_x16<V: Simd>(simd: V, value: i16x16<V>) -> [Self; 16];

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

    #[inline(always)]
    fn load_x8<V: Simd>(simd: V, src: &[Self; 8]) -> i16x8<V> {
        let mut bytes = [0u8; 16];
        if let Some(low) = bytes.first_chunk_mut::<8>() {
            *low = *src;
        }
        let (low, _) = simd.widen_u8x16(u8x16::simd_from(simd, bytes));
        low.bitcast()
    }

    #[inline(always)]
    fn load_x16<V: Simd>(simd: V, src: &[Self; 16]) -> i16x16<V> {
        let (low, high) = simd.widen_u8x16(u8x16::simd_from(simd, *src));
        simd.combine_u16x8(low, high).bitcast()
    }

    #[inline(always)]
    fn pack_x8<V: Simd>(simd: V, value: i16x8<V>) -> [Self; 8] {
        let lanes: u16x8<V> = value.bitcast();
        let bytes: [u8; 16] = simd.relaxed_narrow_u16x8(lanes, lanes).into();
        bytes.first_chunk::<8>().copied().unwrap_or_default()
    }

    #[inline(always)]
    fn pack_x16<V: Simd>(simd: V, value: i16x16<V>) -> [Self; 16] {
        let lanes: u16x16<V> = value.bitcast();
        let (low, _) = simd.split_u8x32(simd.relaxed_narrow_u16x16(lanes, lanes));
        low.into()
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

    #[inline(always)]
    fn load_x8<V: Simd>(simd: V, src: &[Self; 8]) -> i16x8<V> {
        u16x8::simd_from(simd, *src).bitcast()
    }

    #[inline(always)]
    fn load_x16<V: Simd>(simd: V, src: &[Self; 16]) -> i16x16<V> {
        u16x16::simd_from(simd, *src).bitcast()
    }

    #[inline(always)]
    fn pack_x8<V: Simd>(_simd: V, value: i16x8<V>) -> [Self; 8] {
        let lanes: u16x8<V> = value.bitcast();
        lanes.into()
    }

    #[inline(always)]
    fn pack_x16<V: Simd>(_simd: V, value: i16x16<V>) -> [Self; 16] {
        let lanes: u16x16<V> = value.bitcast();
        lanes.into()
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
    use crate::decode::cpu::dsp::simd::{Cpu, with_simd};

    #[test]
    fn clipped_holds_the_range() {
        assert_eq!(u8::clipped(-3, 255), 0);
        assert_eq!(u8::clipped(300, 255), 255);
        assert_eq!(u8::clipped(17, 255), 17);
        assert_eq!(u16::clipped(-1, 1023), 0);
        assert_eq!(u16::clipped(5000, 1023), 1023);
        assert_eq!(u16::clipped(511, 511), 511);
    }

    /// A round trip through 16-bit lanes keeps every sample, in every form.
    fn round_trip<S: Sample>(cpu: Cpu, make: fn(usize) -> S) {
        let short: [S; 8] = std::array::from_fn(make);
        let long: [S; 16] = std::array::from_fn(make);
        let (out_short, out_long) = with_simd!(cpu, simd => {
            (
                S::pack_x8(simd, S::load_x8(simd, &short)),
                S::pack_x16(simd, S::load_x16(simd, &long)),
            )
        });
        assert_eq!(out_short, short);
        assert_eq!(out_long, long);
    }

    #[test]
    fn samples_survive_16_bit_lanes() {
        for cpu in [Cpu::scalar(), Cpu::detected()] {
            round_trip::<u8>(cpu, |index| (index * 17) as u8);
            round_trip::<u16>(cpu, |index| (index * 68) as u16);
        }
    }

    /// `u8` lanes extend with zeros: 255 stays positive.
    #[test]
    fn eight_bit_lanes_extend_with_zeros() {
        for cpu in [Cpu::scalar(), Cpu::detected()] {
            let lanes = with_simd!(cpu, simd => <[i16; 8]>::from(u8::load_x8(simd, &[255u8; 8])));
            assert_eq!(lanes, [255; 8]);
        }
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
