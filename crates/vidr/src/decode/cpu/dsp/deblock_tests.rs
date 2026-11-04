//! Hand-worked decisions and filters of 8.7.2.5, and every form against the formulas.

use super::*;
use crate::decode::cpu::dsp::simd::{Cpu, with_simd};

/// [`luma_section`] on the CPU's best form.
fn filter_luma_section<S: Sample>(samples: &mut [S], section: EdgeSection, filter: &EdgeFilter) {
    with_simd!(Cpu::detected(), simd => luma_section(simd, samples, section, filter));
}

const STRIDE: usize = 8;

const fn filter(beta: i32, tc: i32) -> EdgeFilter {
    EdgeFilter {
        beta,
        tc,
        modify_p: true,
        modify_q: true,
        max_value: 255,
    }
}

/// Four lines of eight samples with the same values, and the edge in the middle.
fn lines(line: [u8; 8]) -> Vec<u8> {
    line.repeat(4)
}

const VERTICAL: EdgeSection = EdgeSection {
    q0: 4,
    across: 1,
    along: STRIDE,
};

#[test]
fn the_tables_read_table_8_12() {
    assert_eq!(beta(15, 0, 8), 0);
    assert_eq!(beta(16, 0, 8), 6);
    assert_eq!(beta(28, 0, 8), 18);
    assert_eq!(beta(29, 0, 8), 20);
    assert_eq!(beta(51, 0, 8), 64);
    // Slice offset 3: Q = 25 + 6.
    assert_eq!(beta(25, 3, 8), 24);
    // Q clips to 51, and 10 bits scale by 4.
    assert_eq!(beta(60, 6, 10), 256);
    assert_eq!(tc(17, 0, 8), 0);
    assert_eq!(tc(18, 0, 8), 1);
    assert_eq!(tc(27, 0, 8), 2);
    assert_eq!(tc(53, 0, 8), 24);
    assert_eq!(tc(70, 0, 10), 96);
    assert_eq!(tc(-5, -6, 8), 0);
}

/// Flat sides and a step of 4, with beta 16 and tC 2: d is 0. dSam needs
/// Abs( p0 - q0 ) below ( 5 * 2 + 1 ) >> 1 = 5, so the strong filter runs.
#[test]
fn a_small_step_takes_the_strong_filter() {
    let mut samples = lines([100, 100, 100, 100, 104, 104, 104, 104]);
    filter_luma_section(&mut samples, VERTICAL, &filter(16, 2));
    // p0: ( 100 + 200 + 200 + 208 + 104 + 4 ) >> 3 = 102. p1: 406 >> 2 = 101.
    // p2: ( 200 + 300 + 100 + 100 + 104 + 4 ) >> 3 = 101.
    // q0: ( 100 + 200 + 208 + 208 + 104 + 4 ) >> 3 = 103. q1: 414 >> 2 = 103.
    // q2: ( 100 + 104 + 104 + 312 + 208 + 4 ) >> 3 = 104.
    let expected = [100, 101, 101, 102, 103, 103, 104, 104];
    for row in samples.chunks(STRIDE) {
        assert_eq!(row, expected);
    }
}

/// d = 0 again, but a step of 20 fails the third dSam test. The weak filter gets
/// Δ = ( 9 * 20 - 3 * 20 + 8 ) >> 4 = 8, clipped to tC 4. dp is 0, below the side
/// limit.
#[test]
fn a_large_step_takes_the_weak_filter() {
    let mut samples = lines([100, 100, 100, 100, 120, 120, 120, 120]);
    filter_luma_section(&mut samples, VERTICAL, &filter(16, 4));
    // p1: ( 100 - 100 + 4 ) >> 1 = 2, within tC >> 1. q1: ( 120 - 120 - 4 ) >> 1 = -2.
    let expected = [100, 100, 102, 104, 116, 118, 120, 120];
    for row in samples.chunks(STRIDE) {
        assert_eq!(row, expected);
    }
}

/// Abs( Δ ) at tC * 10 turns the line off: Δ = ( 9 * 60 - 3 * 60 + 8 ) >> 4 = 23,
/// and tC 2 gives a limit of 20.
#[test]
fn a_very_large_step_stays() {
    let line = [50, 50, 50, 50, 110, 110, 110, 110];
    let mut samples = lines(line);
    filter_luma_section(&mut samples, VERTICAL, &filter(16, 2));
    assert_eq!(samples, lines(line));
}

/// dp is 8 on both lines, so d = 16 reaches beta 16 and dE stays 0.
#[test]
fn a_busy_edge_stays() {
    let line = [100, 100, 104, 100, 100, 100, 100, 100];
    let mut samples = lines(line);
    filter_luma_section(&mut samples, VERTICAL, &filter(16, 2));
    assert_eq!(samples, lines(line));
}

/// nDp 0 keeps the p side, and q still changes (8.7.2.5.7).
#[test]
fn a_protected_side_keeps_its_samples() {
    let mut samples = lines([100, 100, 100, 100, 104, 104, 104, 104]);
    let protected = EdgeFilter {
        modify_p: false,
        ..filter(16, 2)
    };
    filter_luma_section(&mut samples, VERTICAL, &protected);
    for row in samples.chunks(STRIDE) {
        assert_eq!(row, [100, 100, 100, 100, 103, 103, 104, 104]);
    }
}

/// The same filter on rows: a horizontal edge swaps the steps.
#[test]
fn a_horizontal_edge_filters_columns() {
    let mut samples = vec![0u16; 8 * 4];
    for row in 0..8 {
        let value = if row < 4 { 100 } else { 104 };
        samples[row * 4..row * 4 + 4].fill(value);
    }
    let section = EdgeSection {
        q0: 16,
        across: 4,
        along: 1,
    };
    filter_luma_section(&mut samples, section, &filter(16, 2));
    for x in 0..4 {
        let column: Vec<u16> = (0..8).map(|y| samples[y * 4 + x]).collect();
        assert_eq!(column, [100, 101, 101, 102, 103, 103, 104, 104]);
    }
}

/// (8-403): Δ = ( ( 8 << 2 ) + 100 - 108 + 4 ) >> 3 = 3, clipped to tC 2.
#[test]
fn chroma_moves_one_sample_on_each_side() {
    let mut samples = lines([0, 0, 100, 100, 108, 108, 0, 0]);
    let section = EdgeSection {
        q0: 4,
        across: 1,
        along: STRIDE,
    };
    filter_chroma_section(&mut samples, section, &filter(0, 2));
    for row in samples.chunks(STRIDE) {
        assert_eq!(row, [0, 0, 100, 102, 106, 108, 0, 0]);
    }
}

#[test]
fn a_small_chroma_step_rounds_to_no_change() {
    let mut samples = lines([0, 0, 254, 254, 255, 255, 0, 0]);
    let section = EdgeSection {
        q0: 4,
        across: 1,
        along: STRIDE,
    };
    filter_chroma_section(&mut samples, section, &filter(0, 8));
    // Δ = ( 4 + 254 - 255 + 4 ) >> 3 = 0, so nothing moves.
    assert_eq!(samples, lines([0, 0, 254, 254, 255, 255, 0, 0]));
}

#[test]
fn a_section_outside_the_samples_is_left_alone() {
    let mut samples = vec![7u8; 16];
    let at_start = EdgeSection {
        q0: 2,
        across: 1,
        along: 4,
    };
    filter_luma_section(&mut samples, at_start, &filter(64, 24));
    let past_end = EdgeSection {
        q0: 12,
        across: 1,
        along: 4,
    };
    filter_luma_section(&mut samples, past_end, &filter(64, 24));
    filter_chroma_section(&mut samples, past_end, &filter(64, 24));
    assert_eq!(samples, vec![7u8; 16]);
}
