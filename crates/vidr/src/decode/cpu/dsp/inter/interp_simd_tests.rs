//! Every form of the interpolation kernels against the formulas of 8.5.3.3.3, written
//! sample by sample with clamped coordinates.

use super::*;
use crate::decode::cpu::dsp::test_rng::Rng;
use crate::decode::cpu::hevc::picture::Plane;

/// The sample at a clamped location, (8-222), (8-223), (8-239), and (8-240).
fn clamped<S: Sample>(plane: &Plane<S>, x: i32, y: i32) -> i32 {
    let column = x.clamp(0, plane.width() as i32 - 1) as usize;
    let row = y.clamp(0, plane.height() as i32 - 1) as usize;
    plane.row(row).unwrap()[column].to_i32()
}

/// One output sample with every term spelled out in 32 bits.
fn reference_sample<S: Sample, const TAPS: usize>(
    plane: &Plane<S>,
    at: (i32, i32),
    filters: (&[i16; TAPS], &[i16; TAPS]),
    fractions: (bool, bool),
    bit_depth: u32,
) -> i16 {
    let before = TAPS as i32 / 2 - 1;
    let shift1 = bit_depth.saturating_sub(8).min(4);
    let shift3 = 14u32.saturating_sub(bit_depth).max(2);
    let across = |y: i32| -> i32 {
        filters
            .0
            .iter()
            .enumerate()
            .map(|(k, &tap)| clamped(plane, at.0 + k as i32 - before, y) * i32::from(tap))
            .sum::<i32>()
            >> shift1
    };
    let down = |column: &dyn Fn(i32) -> i32, shift: u32| -> i32 {
        filters
            .1
            .iter()
            .enumerate()
            .map(|(k, &tap)| column(at.1 + k as i32 - before) * i32::from(tap))
            .sum::<i32>()
            >> shift
    };
    let value = match fractions {
        (false, false) => clamped(plane, at.0, at.1) << shift3,
        (true, false) => across(at.1),
        (false, true) => down(&|y| clamped(plane, at.0, y), shift1),
        (true, true) => down(&|y| i32::from(across(y) as i16), 6),
    };
    value as i16
}

/// Random samples with flat areas, steps, and full-range values.
fn random_plane<S: Sample>(
    width: usize,
    height: usize,
    bit_depth: u32,
    make: fn(i32) -> S,
    rng: &mut Rng,
) -> Plane<S> {
    let mut plane = Plane::new(width, height).unwrap();
    for y in 0..height {
        for sample in plane.row_mut(y).unwrap().iter_mut() {
            *sample = make(rng.sample(1 << (bit_depth - 1), bit_depth));
        }
    }
    plane
}

/// Locations inside the plane, across each edge, and far outside it.
fn locations(width: i32, height: i32, block_width: i32, block_height: i32) -> Vec<(i32, i32)> {
    vec![
        (16, 16),
        (-2, 16),
        (16, -2),
        (width - block_width + 2, 16),
        (16, height - block_height + 2),
        (-1, -1),
        (-500, 500),
        (500, -500),
    ]
}

/// Every block size of Main in the kernel's units, all fractions, each location, in
/// each form.
fn compare_luma<S: Sample>(bit_depth: u32, make: fn(i32) -> S, seed: u64) {
    let mut rng = Rng::new(seed);
    let plane = random_plane::<S>(96, 96, bit_depth, make, &mut rng);
    let view = plane.view();
    let mut scratch = Box::new(InterpScratch::new());
    let sizes = [
        (4, 8),
        (8, 4),
        (8, 8),
        (12, 16),
        (16, 4),
        (16, 16),
        (24, 8),
        (32, 8),
        (48, 16),
        (64, 8),
        (64, 64),
    ];
    for (width, height) in sizes {
        for at in locations(96, 96, width as i32, height as i32) {
            for (x_frac, x_taps) in LUMA_FILTERS.iter().enumerate() {
                for (y_frac, y_taps) in LUMA_FILTERS.iter().enumerate() {
                    let block = RefBlock {
                        x: at.0,
                        y: at.1,
                        width,
                        height,
                    };
                    let mut baseline = Box::new(PredSamples::new());
                    let mut detected = Box::new(PredSamples::new());
                    let frac = (x_frac, y_frac);
                    luma_on(
                        Cpu::scalar(),
                        &view,
                        block,
                        frac,
                        bit_depth,
                        &mut scratch,
                        &mut baseline,
                    );
                    luma_on(
                        Cpu::detected(),
                        &view,
                        block,
                        frac,
                        bit_depth,
                        &mut scratch,
                        &mut detected,
                    );
                    for y in 0..height {
                        let expected: Vec<i16> = (0..width)
                            .map(|x| {
                                reference_sample(
                                    &plane,
                                    (at.0 + x as i32, at.1 + y as i32),
                                    (x_taps, y_taps),
                                    (x_frac != 0, y_frac != 0),
                                    bit_depth,
                                )
                            })
                            .collect();
                        let context = format!("{width}x{height} at {at:?} frac {frac:?} row {y}");
                        assert_eq!(baseline.row(y, width), expected, "baseline {context}");
                        assert_eq!(detected.row(y, width), expected, "detected {context}");
                    }
                }
            }
        }
    }
}

fn compare_chroma<S: Sample>(bit_depth: u32, make: fn(i32) -> S, seed: u64) {
    let mut rng = Rng::new(seed);
    let plane = random_plane::<S>(48, 48, bit_depth, make, &mut rng);
    let view = plane.view();
    let mut scratch = Box::new(InterpScratch::new());
    let sizes = [
        (2, 4),
        (4, 2),
        (4, 4),
        (6, 8),
        (8, 2),
        (8, 8),
        (12, 16),
        (16, 4),
        (24, 32),
        (32, 32),
    ];
    for (width, height) in sizes {
        for at in locations(48, 48, width as i32, height as i32) {
            for (x_frac, x_taps) in CHROMA_FILTERS.iter().enumerate() {
                for (y_frac, y_taps) in CHROMA_FILTERS.iter().enumerate() {
                    let block = RefBlock {
                        x: at.0,
                        y: at.1,
                        width,
                        height,
                    };
                    let mut baseline = Box::new(PredSamples::new());
                    let mut detected = Box::new(PredSamples::new());
                    let frac = (x_frac, y_frac);
                    chroma_on(
                        Cpu::scalar(),
                        &view,
                        block,
                        frac,
                        bit_depth,
                        &mut scratch,
                        &mut baseline,
                    );
                    chroma_on(
                        Cpu::detected(),
                        &view,
                        block,
                        frac,
                        bit_depth,
                        &mut scratch,
                        &mut detected,
                    );
                    for y in 0..height {
                        let expected: Vec<i16> = (0..width)
                            .map(|x| {
                                reference_sample(
                                    &plane,
                                    (at.0 + x as i32, at.1 + y as i32),
                                    (x_taps, y_taps),
                                    (x_frac != 0, y_frac != 0),
                                    bit_depth,
                                )
                            })
                            .collect();
                        let context = format!("{width}x{height} at {at:?} frac {frac:?} row {y}");
                        assert_eq!(baseline.row(y, width), expected, "baseline {context}");
                        assert_eq!(detected.row(y, width), expected, "detected {context}");
                    }
                }
            }
        }
    }
}

#[test]
fn luma_forms_equal_the_formulas_at_8_bits() {
    compare_luma::<u8>(8, |value| value as u8, 0x51a2_0c33_7700_11ef);
}

#[test]
fn luma_forms_equal_the_formulas_at_10_bits() {
    compare_luma::<u16>(10, |value| value as u16, 0x1f2e_3d4c_5b6a_7988);
}

/// 8-bit video in `u16` samples takes the same path as a `u8` edge block.
#[test]
fn luma_forms_equal_the_formulas_at_8_bits_in_16_bit_samples() {
    compare_luma::<u16>(8, |value| value as u16, 0x77aa_55cc_1133_9921);
}

#[test]
fn chroma_forms_equal_the_formulas_at_8_bits() {
    compare_chroma::<u8>(8, |value| value as u8, 0x6b8b_4567_327b_23c6);
}

#[test]
fn chroma_forms_equal_the_formulas_at_10_bits() {
    compare_chroma::<u16>(10, |value| value as u16, 0x643c_9869_6633_4873);
}

/// A plane smaller than the filter reach falls back to the clamped copy.
#[test]
fn a_tiny_plane_takes_the_clamped_copy() {
    let mut rng = Rng::new(0x0074_b0dc_5119_495c);
    let plane = random_plane::<u8>(4, 4, 8, |value| value as u8, &mut rng);
    let mut scratch = Box::new(InterpScratch::new());
    let mut out = Box::new(PredSamples::new());
    let block = RefBlock {
        x: 0,
        y: 0,
        width: 4,
        height: 4,
    };
    luma_on(
        Cpu::detected(),
        &plane.view(),
        block,
        (2, 2),
        8,
        &mut scratch,
        &mut out,
    );
    for y in 0..4 {
        let expected: Vec<i16> = (0..4)
            .map(|x| {
                reference_sample(
                    &plane,
                    (x, y),
                    (&LUMA_FILTERS[2], &LUMA_FILTERS[2]),
                    (true, true),
                    8,
                )
            })
            .collect();
        assert_eq!(out.row(y as usize, 4), expected);
    }
}
