//! Every form of the luma filter against the same process written one line at a time in
//! 32 bits, with one read and one write per sample.

use super::*;
use crate::decode::cpu::dsp::simd::{Cpu, with_simd};
use crate::decode::cpu::dsp::test_rng::Rng;

/// `p0` to `p3`, or `q0` to `q3`, of one line.
fn side<S: Sample>(samples: &[S], first: usize, step: isize) -> [i32; 4] {
    let at = |distance: isize| samples[first.wrapping_add_signed(distance * step)].to_i32();
    [at(0), at(1), at(2), at(3)]
}

fn second(side: &[i32; 4]) -> i32 {
    (side[2] - 2 * side[1] + side[0]).abs()
}

fn line_is_strong(p: &[i32; 4], q: &[i32; 4], dpq: i32, filter: &EdgeFilter) -> bool {
    dpq < (filter.beta >> 2)
        && (p[3] - p[0]).abs() + (q[0] - q[3]).abs() < (filter.beta >> 3)
        && (p[0] - q[0]).abs() < ((5 * filter.tc + 1) >> 1)
}

fn write<S: Sample>(
    samples: &mut [S],
    first: usize,
    step: isize,
    values: &[Option<i32>; 3],
    max_value: i32,
) {
    for (distance, value) in values.iter().enumerate() {
        if let Some(value) = *value {
            let index = first.wrapping_add_signed(distance as isize * step);
            samples[index] = S::clipped(value, max_value);
        }
    }
}

/// (8-389) to (8-394) on one line.
fn strong_values(p: &[i32; 4], q: &[i32; 4], tc: i32) -> ([Option<i32>; 3], [Option<i32>; 3]) {
    let limit = 2 * tc;
    let clip = |value: i32, centre: i32| Some(value.clamp(centre - limit, centre + limit));
    (
        [
            clip(
                (p[2] + 2 * p[1] + 2 * p[0] + 2 * q[0] + q[1] + 4) >> 3,
                p[0],
            ),
            clip((p[2] + p[1] + p[0] + q[0] + 2) >> 2, p[1]),
            clip((2 * p[3] + 3 * p[2] + p[1] + p[0] + q[0] + 4) >> 3, p[2]),
        ],
        [
            clip(
                (p[1] + 2 * p[0] + 2 * q[0] + 2 * q[1] + q[2] + 4) >> 3,
                q[0],
            ),
            clip((p[0] + q[0] + q[1] + q[2] + 2) >> 2, q[1]),
            clip((p[0] + q[0] + q[1] + 3 * q[2] + 2 * q[3] + 4) >> 3, q[2]),
        ],
    )
}

/// The changed `p` and `q` samples of one line.
type LineChanges = ([Option<i32>; 3], [Option<i32>; 3]);

/// (8-395) to (8-402) on one line, or `None` when `Abs( Δ )` reaches `tC * 10`.
fn weak_values(
    p: &[i32; 4],
    q: &[i32; 4],
    tc: i32,
    extend_p: bool,
    extend_q: bool,
) -> Option<LineChanges> {
    let delta = (9 * (q[0] - p[0]) - 3 * (q[1] - p[1]) + 8) >> 4;
    if delta.abs() >= tc * 10 {
        return None;
    }
    let delta = delta.clamp(-tc, tc);
    let half = tc >> 1;
    let p1 = extend_p.then(|| {
        let change = (((p[2] + p[0] + 1) >> 1) - p[1] + delta) >> 1;
        p[1] + change.clamp(-half, half)
    });
    let q1 = extend_q.then(|| {
        let change = (((q[2] + q[0] + 1) >> 1) - q[1] - delta) >> 1;
        q[1] + change.clamp(-half, half)
    });
    Some((
        [Some(p[0] + delta), p1, None],
        [Some(q[0] - delta), q1, None],
    ))
}

/// The process of 8.7.2.5.3 and 8.7.2.5.4 on one section.
fn reference_section<S: Sample>(samples: &mut [S], section: EdgeSection, filter: &EdgeFilter) {
    let across = section.across as isize;
    let last = section.q0 + 3 * section.along;
    let (p_first, q_first) = (
        side(samples, section.q0 - section.across, -across),
        side(samples, section.q0, across),
    );
    let (p_last, q_last) = (
        side(samples, last - section.across, -across),
        side(samples, last, across),
    );
    let (dp0, dp3) = (second(&p_first), second(&p_last));
    let (dq0, dq3) = (second(&q_first), second(&q_last));
    let (dpq0, dpq3) = (dp0 + dq0, dp3 + dq3);
    if dpq0 + dpq3 >= filter.beta {
        return;
    }
    let strong = line_is_strong(&p_first, &q_first, 2 * dpq0, filter)
        && line_is_strong(&p_last, &q_last, 2 * dpq3, filter);
    let side_limit = (filter.beta + (filter.beta >> 1)) >> 3;
    let extend_p = dp0 + dp3 < side_limit;
    let extend_q = dq0 + dq3 < side_limit;
    for line in 0..4 {
        let q0 = section.q0 + line * section.along;
        let p = side(samples, q0 - section.across, -across);
        let q = side(samples, q0, across);
        let (new_p, new_q) = if strong {
            strong_values(&p, &q, filter.tc)
        } else if let Some(values) = weak_values(&p, &q, filter.tc, extend_p, extend_q) {
            values
        } else {
            continue;
        };
        if filter.modify_p {
            write(
                samples,
                q0 - section.across,
                -across,
                &new_p,
                filter.max_value,
            );
        }
        if filter.modify_q {
            write(samples, q0, across, &new_q, filter.max_value);
        }
    }
}

/// Random lines around one edge: a flat side, a small or large step, and noise, so the
/// decisions reach the strong filter, the weak filter, and no filter.
fn random_area<S: Sample>(
    rng: &mut Rng,
    bit_depth: u32,
    make: fn(i32) -> S,
    section: EdgeSection,
    len: usize,
) -> Vec<S> {
    let max = (1i32 << bit_depth) - 1;
    let scale = 1 << (bit_depth - 8);
    let mut area = vec![make(0); len];
    let base = rng.below(max as u64 + 1) as i32;
    let step = match rng.below(3) {
        0 => rng.below(4) as i32 * scale,
        1 => rng.below(24) as i32 * scale,
        _ => rng.below(120) as i32 * scale,
    };
    let noise = [0, 1, 3, 9][rng.below(4) as usize] * scale;
    for line in 0..4 {
        for distance in 0..8usize {
            let index =
                section.q0 - 4 * section.across + distance * section.across + line * section.along;
            let level = if distance < 4 { base } else { base + step };
            let wobble = if noise == 0 {
                0
            } else {
                rng.below(2 * noise as u64 + 1) as i32 - noise
            };
            area[index] = make((level + wobble).clamp(0, max));
        }
    }
    area
}

fn compare<S: Sample>(bit_depth: u32, make: fn(i32) -> S, seed: u64) {
    let mut rng = Rng::new(seed);
    let scale = 1 << (bit_depth - 8);
    let forms = [
        EdgeSection {
            q0: 4,
            across: 1,
            along: 16,
        },
        EdgeSection {
            q0: 64,
            across: 16,
            along: 1,
        },
        EdgeSection {
            q0: 2 * 3 + 4 * 16,
            across: 2,
            along: 16,
        },
    ];
    let mut touched = 0usize;
    for section in forms {
        for _ in 0..4000 {
            let start = random_area(&mut rng, bit_depth, make, section, 256);
            let filter = EdgeFilter {
                beta: rng.below(65) as i32 * scale,
                tc: rng.below(25) as i32 * scale,
                modify_p: rng.below(4) != 0,
                modify_q: rng.below(4) != 0,
                max_value: (1 << bit_depth) - 1,
            };
            let mut expected = start.clone();
            reference_section(&mut expected, section, &filter);
            touched += usize::from(expected != start);
            for cpu in [Cpu::scalar(), Cpu::detected()] {
                let mut out = start.clone();
                with_simd!(cpu, simd => luma_section(simd, &mut out, section, &filter));
                assert_eq!(out, expected, "{cpu:?} {section:?} {filter:?}");
            }
        }
    }
    assert!(
        touched > 1000,
        "the inputs reach the filters {touched} times"
    );
}

#[test]
fn every_form_equals_the_process_at_8_bits() {
    compare::<u8>(8, |value| value as u8, 0xfeed_c0de_1357_9bdf);
}

#[test]
fn every_form_equals_the_process_at_10_bits() {
    compare::<u16>(10, |value| value as u16, 0xbead_f00d_2468_ace1);
}
