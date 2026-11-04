//! Runtime selection of the CPU features the kernels compile under, and the vector
//! load and store helpers they share.
//!
//! A kernel is generic over a [`Simd`] token and works on explicit vector types. The
//! [`with_simd`] macro instantiates it once for each form of [`Cpu`], inside a function
//! compiled for that token's features. Everything called inside carries
//! `#[inline(always)]`. A call that stays a call compiles for the target's baseline.
//!
//! x86-64 selects AVX2 and AArch64 selects NEON. Any other CPU takes the scalar token,
//! the reference form of every kernel. The crate providing the tokens holds the only
//! `unsafe` of this path.

use fearless_simd::{Fallback, Simd, SimdFrom, i16x8, i16x16, i32x4, i32x8};

/// The feature set one kernel call compiles under.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Cpu {
    /// One scalar operation per lane: the reference form of every kernel.
    Scalar(Fallback),
    /// The x86-64-v3 feature set, AVX2 included. Holding the token proves the features.
    #[cfg(target_arch = "x86_64")]
    Avx2(fearless_simd::x86::Avx2),
    /// NEON. Holding the token proves the feature.
    #[cfg(target_arch = "aarch64")]
    Neon(fearless_simd::aarch64::Neon),
}

impl Cpu {
    /// The scalar form, on every machine.
    #[inline(always)]
    pub(crate) const fn scalar() -> Self {
        Self::Scalar(Fallback::new())
    }

    /// The best feature set of this machine. The detection result stays cached, so a
    /// call costs one load and a few branches.
    #[inline(always)]
    pub(crate) fn detected() -> Self {
        #[cfg(target_arch = "x86_64")]
        if let Some(token) = fearless_simd::Level::new().as_avx2() {
            return Self::Avx2(token);
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(token) = fearless_simd::Level::new().as_neon() {
            return Self::Neon(token);
        }
        Self::scalar()
    }
}

/// Runs `$body` compiled for the features of `$cpu`, with `$simd` bound to its token.
/// The body expands once per form of [`Cpu`], and calls only `#[inline(always)]` kernels.
macro_rules! with_simd {
    ($cpu:expr, $simd:ident => $body:expr) => {
        match $cpu {
            $crate::decode::cpu::dsp::simd::Cpu::Scalar(token) => {
                let $simd = token;
                ::fearless_simd::Simd::vectorize(
                    token,
                    #[inline(always)]
                    || $body,
                )
            }
            #[cfg(target_arch = "x86_64")]
            $crate::decode::cpu::dsp::simd::Cpu::Avx2(token) => {
                let $simd = token;
                ::fearless_simd::Simd::vectorize(
                    token,
                    #[inline(always)]
                    || $body,
                )
            }
            #[cfg(target_arch = "aarch64")]
            $crate::decode::cpu::dsp::simd::Cpu::Neon(token) => {
                let $simd = token;
                ::fearless_simd::Simd::vectorize(
                    token,
                    #[inline(always)]
                    || $body,
                )
            }
        }
    };
}
pub(crate) use with_simd;

/// Eight values of `src` from index `at`. A range past the end reads zeros, and a kernel
/// never asks for one. A constant or masked `at` folds the check away.
#[inline(always)]
pub(crate) fn load_i16x8<V: Simd>(simd: V, src: &[i16], at: usize) -> i16x8<V> {
    match src.get(at..).and_then(<[i16]>::first_chunk::<8>) {
        Some(lanes) => i16x8::simd_from(simd, *lanes),
        None => simd.splat_i16x8(0),
    }
}

/// Sixteen values of `src` from index `at`, as [`load_i16x8`] reads them.
#[inline(always)]
pub(crate) fn load_i16x16<V: Simd>(simd: V, src: &[i16], at: usize) -> i16x16<V> {
    match src.get(at..).and_then(<[i16]>::first_chunk::<16>) {
        Some(lanes) => i16x16::simd_from(simd, *lanes),
        None => simd.splat_i16x16(0),
    }
}

/// Writes eight values to `target` at index `at`. A range past the end writes nothing.
#[inline(always)]
pub(crate) fn store_i16x8<V: Simd>(value: i16x8<V>, target: &mut [i16], at: usize) {
    if let Some(lanes) = target.get_mut(at..).and_then(<[i16]>::first_chunk_mut::<8>) {
        *lanes = value.into();
    }
}

/// Writes sixteen values to `target` at index `at`, as [`store_i16x8`] writes them.
#[inline(always)]
pub(crate) fn store_i16x16<V: Simd>(value: i16x16<V>, target: &mut [i16], at: usize) {
    if let Some(lanes) = target
        .get_mut(at..)
        .and_then(<[i16]>::first_chunk_mut::<16>)
    {
        *lanes = value.into();
    }
}

/// `Clip3( 0, max, value )` on eight lanes.
#[inline(always)]
pub(crate) fn clip_x8<V: Simd>(simd: V, value: i16x8<V>, max: i16x8<V>) -> i16x8<V> {
    simd.min_i16x8(simd.max_i16x8(value, simd.splat_i16x8(0)), max)
}

/// `Clip3( 0, max, value )` on sixteen lanes.
#[inline(always)]
pub(crate) fn clip_x16<V: Simd>(simd: V, value: i16x16<V>, max: i16x16<V>) -> i16x16<V> {
    simd.min_i16x16(simd.max_i16x16(value, simd.splat_i16x16(0)), max)
}

/// Two 16-bit values in one 32-bit lane: `first` in the low half. A splat of it gives the
/// weights of a dot product on pairs of lanes.
#[inline(always)]
pub(crate) const fn tap_pair(first: i16, second: i16) -> i32 {
    ((second as i32) << 16) | (first as u16 as i32)
}

/// Four values of an `i32` slice as a vector. A shorter slice reads zeros.
#[inline(always)]
pub(crate) fn load_i32x4<V: Simd>(simd: V, src: &[i32]) -> i32x4<V> {
    match src.first_chunk::<4>() {
        Some(lanes) => i32x4::simd_from(simd, *lanes),
        None => simd.splat_i32x4(0),
    }
}

/// Eight values of an `i32` slice as a vector. A shorter slice reads zeros.
#[inline(always)]
pub(crate) fn load_i32x8<V: Simd>(simd: V, src: &[i32]) -> i32x8<V> {
    match src.first_chunk::<8>() {
        Some(lanes) => i32x8::simd_from(simd, *lanes),
        None => simd.splat_i32x8(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both forms run the operation and return its result.
    #[test]
    fn every_form_runs_the_operation() {
        for cpu in [Cpu::scalar(), Cpu::detected()] {
            let values = [1i16, 2, 3, 4, 5, 6, 7, 8];
            let mut out = [0i16; 8];
            with_simd!(cpu, simd => {
                let doubled = simd.add_i16x8(load_i16x8(simd, &values, 0), load_i16x8(simd, &values, 0));
                store_i16x8(doubled, &mut out, 0);
            });
            assert_eq!(out, [2, 4, 6, 8, 10, 12, 14, 16]);
        }
    }

    /// A CPU that reports AVX2 gets the AVX2 form.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_is_selected_when_the_cpu_has_it() {
        if fearless_simd::Level::new().as_avx2().is_some() {
            assert!(matches!(Cpu::detected(), Cpu::Avx2(_)));
        }
    }

    /// Ranges past the end stay harmless.
    #[test]
    fn a_range_past_the_end_reads_zeros_and_writes_nothing() {
        let values = [7i16; 8];
        let mut out = [1i16; 8];
        with_simd!(Cpu::scalar(), simd => {
            let read = load_i16x8(simd, &values, 1);
            store_i16x8(read, &mut out, 1);
        });
        assert_eq!(out, [1; 8]);
        with_simd!(Cpu::scalar(), simd => store_i16x8(load_i16x8(simd, &values, 1), &mut out, 0));
        assert_eq!(out, [0; 8]);
    }
}
