//! Hand-worked values from (8-224) to (8-261), and the clamped coordinates.

use super::*;
use crate::decode::cpu::hevc::picture::Plane;

fn scratch() -> Box<InterpScratch> {
    Box::new(InterpScratch::new())
}

fn plane<S: Sample>(width: usize, height: usize, at: impl Fn(usize, usize) -> i32) -> Plane<S> {
    let mut plane = Plane::new(width, height).unwrap();
    for y in 0..height {
        for (x, sample) in plane.row_mut(y).unwrap().iter_mut().enumerate() {
            *sample = S::clipped(at(x, y), 65_535);
        }
    }
    plane
}

fn block(x: i32, y: i32, width: usize, height: usize) -> RefBlock {
    RefBlock {
        x,
        y,
        width,
        height,
    }
}

/// A flat plane gives value << 6 at every fraction, because each filter sums to 64.
/// Luma, 8 bits.
#[test]
fn a_flat_plane_gives_the_same_value_at_every_luma_fraction() {
    let source = plane::<u8>(32, 32, |_, _| 100);
    let mut out = PredSamples::new();
    let mut work = scratch();
    for x_frac in 0..4 {
        for y_frac in 0..4 {
            interpolate_luma(
                &source.view(),
                block(8, 8, 8, 8),
                (x_frac, y_frac),
                8,
                &mut work,
                &mut out,
            );
            for y in 0..8 {
                assert_eq!(out.row(y, 8), &[6400i16; 8], "{x_frac} {y_frac}");
            }
        }
    }
}

/// Chroma, 8 bits. Same reason.
#[test]
fn a_flat_plane_gives_the_same_value_at_every_chroma_fraction() {
    let source = plane::<u8>(32, 32, |_, _| 50);
    let mut out = PredSamples::new();
    let mut work = scratch();
    for x_frac in 0..8 {
        for y_frac in 0..8 {
            interpolate_chroma(
                &source.view(),
                block(8, 8, 4, 2),
                (x_frac, y_frac),
                8,
                &mut work,
                &mut out,
            );
            assert_eq!(out.row(1, 4), &[3200i16; 4], "{x_frac} {y_frac}");
        }
    }
}

/// 10 bits: shift1 2, shift3 4. A flat 400: full-sample 400 << 4 = 6400.
/// One-stage ( 400 * 64 ) >> 2 = 6400. Two-stage ( 6400 * 64 ) >> 6 = 6400.
#[test]
fn ten_bit_shifts_keep_14_bits() {
    let source = plane::<u16>(32, 32, |_, _| 400);
    let mut out = PredSamples::new();
    let mut work = scratch();
    for frac in [(0, 0), (1, 0), (0, 2), (3, 3)] {
        interpolate_luma(
            &source.view(),
            block(4, 4, 8, 8),
            frac,
            10,
            &mut work,
            &mut out,
        );
        assert_eq!(out.row(7, 8), &[6400i16; 8], "{frac:?}");
    }
}

/// A step edge: samples are 0 left of x = 10, and 64 from there on. At the block's
/// first sample x = 10, A-3 to A-1 are 0 and A0 to A4 are 64.
/// a = 64 * ( 58 + 17 - 5 + 1 ) = 4544.
/// b = 64 * ( 40 + 40 - 11 + 4 - 1 ) = 4608.
/// c = 64 * ( 17 + 58 - 10 + 4 - 1 ) = 4352.
#[test]
fn a_step_edge_follows_8_224_to_8_226() {
    let source = plane::<u8>(32, 8, |x, _| if x >= 10 { 64 } else { 0 });
    let mut out = PredSamples::new();
    let mut work = scratch();
    let area = block(10, 2, 8, 1);
    interpolate_luma(&source.view(), area, (1, 0), 8, &mut work, &mut out);
    assert_eq!(out.row(0, 1), &[4544]);
    interpolate_luma(&source.view(), area, (2, 0), 8, &mut work, &mut out);
    assert_eq!(out.row(0, 1), &[4608]);
    interpolate_luma(&source.view(), area, (3, 0), 8, &mut work, &mut out);
    assert_eq!(out.row(0, 1), &[4352]);
}

/// The vertical filters match the horizontal ones on a transposed step.
#[test]
fn a_vertical_step_follows_8_227_to_8_229() {
    let source = plane::<u8>(8, 32, |_, y| if y >= 10 { 64 } else { 0 });
    let mut out = PredSamples::new();
    let mut work = scratch();
    let area = block(2, 10, 1, 8);
    interpolate_luma(&source.view(), area, (0, 1), 8, &mut work, &mut out);
    assert_eq!(out.row(0, 1), &[4544]);
    interpolate_luma(&source.view(), area, (0, 2), 8, &mut work, &mut out);
    assert_eq!(out.row(0, 1), &[4608]);
    interpolate_luma(&source.view(), area, (0, 3), 8, &mut work, &mut out);
    assert_eq!(out.row(0, 1), &[4352]);
}

/// j at a corner of 64: samples are 64 where x >= 10 and y >= 10.
/// The first stage on rows 10 and later is b = 4608, and on earlier rows 0.
/// j = ( 4608 * ( 40 + 40 - 11 + 4 - 1 ) ) >> 6 = ( 4608 * 72 ) >> 6 = 5184.
#[test]
fn two_stages_follow_8_234() {
    let source = plane::<u8>(32, 32, |x, y| if x >= 10 && y >= 10 { 64 } else { 0 });
    let mut out = PredSamples::new();
    let mut work = scratch();
    interpolate_luma(
        &source.view(),
        block(10, 10, 4, 4),
        (2, 2),
        8,
        &mut work,
        &mut out,
    );
    assert_eq!(out.row(0, 1), &[5184]);
}

/// Chroma step: samples are 0 left of x = 4, and 100 from there on, at x = 4.
/// B-1 is 0, and B0 to B2 are 100. ab = 100 * ( 58 + 10 - 2 ) = 6600.
/// ae = 100 * ( 36 + 36 - 4 ) = 6800. ah = 100 * ( 10 + 58 - 2 ) = 6600.
#[test]
fn a_chroma_step_follows_8_241_to_8_247() {
    let source = plane::<u8>(16, 4, |x, _| if x >= 4 { 100 } else { 0 });
    let mut out = PredSamples::new();
    let mut work = scratch();
    let area = block(4, 1, 2, 1);
    interpolate_chroma(&source.view(), area, (1, 0), 8, &mut work, &mut out);
    assert_eq!(out.row(0, 1), &[6600]);
    interpolate_chroma(&source.view(), area, (4, 0), 8, &mut work, &mut out);
    assert_eq!(out.row(0, 1), &[6800]);
    interpolate_chroma(&source.view(), area, (7, 0), 8, &mut work, &mut out);
    assert_eq!(out.row(0, 1), &[6600]);
}

/// Far outside the plane, (8-222) and (8-223) read the corner sample.
#[test]
fn coordinates_clamp_far_outside_the_plane() {
    let source = plane::<u8>(16, 16, |x, y| (x + 16 * y) as i32);
    let mut out = PredSamples::new();
    let mut work = scratch();
    interpolate_luma(
        &source.view(),
        block(-5000, -5000, 8, 8),
        (0, 0),
        8,
        &mut work,
        &mut out,
    );
    assert_eq!(out.row(7, 8), &[0i16; 8]);
    interpolate_luma(
        &source.view(),
        block(5000, 5000, 8, 8),
        (3, 1),
        8,
        &mut work,
        &mut out,
    );
    assert_eq!(out.row(7, 8), &[(255i32 << 6) as i16; 8]);
}

/// A block across the left edge repeats column 0. Column values are 0, 1, 2, and on.
#[test]
fn a_block_across_the_left_edge_repeats_column_0() {
    let source = plane::<u8>(16, 4, |x, _| x as i32);
    let mut out = PredSamples::new();
    let mut work = scratch();
    interpolate_luma(
        &source.view(),
        block(-2, 0, 5, 1),
        (0, 0),
        8,
        &mut work,
        &mut out,
    );
    assert_eq!(out.row(0, 5), &[0, 0, 0, 64, 128]);
}

/// A block across the right edge repeats the last column.
#[test]
fn a_block_across_the_right_edge_repeats_the_last_column() {
    let source = plane::<u8>(16, 4, |x, _| x as i32);
    let mut out = PredSamples::new();
    let mut work = scratch();
    interpolate_luma(
        &source.view(),
        block(13, 0, 5, 1),
        (0, 0),
        8,
        &mut work,
        &mut out,
    );
    assert_eq!(
        out.row(0, 5),
        &[13 * 64, 14 * 64, 15 * 64, 15 * 64, 15 * 64]
    );
}

/// The largest block with two fractions fits the scratch buffers.
#[test]
fn the_largest_block_fits() {
    let source = plane::<u16>(128, 128, |_, _| 1000);
    let mut out = PredSamples::new();
    let mut work = scratch();
    interpolate_luma(
        &source.view(),
        block(30, 30, 64, 64),
        (1, 3),
        10,
        &mut work,
        &mut out,
    );
    assert_eq!(out.row(63, 64), &[16000i16; 64]);
}
