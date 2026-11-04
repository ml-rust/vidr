//! Deblocking kernels: the sample decisions and filters of 8.7.2.5.3 and 8.7.2.5.6
//! to 8.7.2.5.8, over a plane's samples. A vertical edge and a horizontal edge
//! differ only in the two index steps.

use super::sample::Sample;

/// `β′` (Table 8-12), for `Q` 0 to 51.
const BETA_PRIME: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
];
/// `tC′` (Table 8-12), for `Q` 0 to 53.
const TC_PRIME: [u8; 54] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3,
    3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24,
];
/// Lines in one edge section.
const SECTION_LINES: usize = 4;

/// `β` (8-348, 8-349) for `qPL` and the slice's `slice_beta_offset_div2`.
pub(crate) fn beta(qp_l: i32, beta_offset_div2: i32, bit_depth: u32) -> i32 {
    let q = (qp_l + (beta_offset_div2 << 1)).clamp(0, 51) as usize;
    i32::from(BETA_PRIME[q]) << (bit_depth - 8)
}

/// `tC` from a `qP` that holds every term of (8-350) or (8-383) but the slice offset.
pub(crate) fn tc(qp: i32, tc_offset_div2: i32, bit_depth: u32) -> i32 {
    let q = (qp + (tc_offset_div2 << 1)).clamp(0, 53) as usize;
    i32::from(TC_PRIME[q]) << (bit_depth - 8)
}

/// Where one edge section lies in a plane's samples.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EdgeSection {
    /// The index of `q0` on the first line.
    pub(crate) q0: usize,
    /// The step from `q0` to `q1`.
    pub(crate) across: usize,
    /// The step from one line to the next.
    pub(crate) along: usize,
}

impl EdgeSection {
    /// True when every sample `reach` steps either side of the edge is in bounds.
    fn fits(self, len: usize, reach: usize) -> bool {
        self.q0 >= reach * self.across
            && self.q0 + (SECTION_LINES - 1) * self.along + (reach - 1) * self.across < len
    }
}

/// What the filters of one edge section read.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EdgeFilter {
    /// `β`. The chroma filter ignores it.
    pub(crate) beta: i32,
    /// `tC`.
    pub(crate) tc: i32,
    /// False when `nDp` is forced to 0 (8.7.2.5.7, 8.7.2.5.8).
    pub(crate) modify_p: bool,
    /// False when `nDq` is forced to 0 (8.7.2.5.7, 8.7.2.5.8).
    pub(crate) modify_q: bool,
    /// `( 1 << BitDepth ) - 1`.
    pub(crate) max_value: i32,
}

/// `p0` to `p3`, or `q0` to `q3`, of one line.
fn side<S: Sample>(samples: &[S], first: usize, step: isize) -> [i32; 4] {
    let at = |distance: isize| samples[first.wrapping_add_signed(distance * step)].to_i32();
    [at(0), at(1), at(2), at(3)]
}

/// `Abs( x2 - 2 * x1 + x0 )`, (8-352) to (8-355).
const fn second_difference(side: &[i32; 4]) -> i32 {
    (side[2] - 2 * side[1] + side[0]).abs()
}

/// The decision process for a luma sample (8.7.2.5.6).
const fn strong_line(p: &[i32; 4], q: &[i32; 4], dpq: i32, filter: &EdgeFilter) -> bool {
    dpq < (filter.beta >> 2)
        && (p[3] - p[0]).abs() + (q[0] - q[3]).abs() < (filter.beta >> 3)
        && (p[0] - q[0]).abs() < ((5 * filter.tc + 1) >> 1)
}

/// The decision and filtering of one luma section (8.7.2.5.3, 8.7.2.5.4). A section
/// outside `samples` is left alone.
pub(crate) fn filter_luma_section<S: Sample>(
    samples: &mut [S],
    section: EdgeSection,
    filter: &EdgeFilter,
) {
    if !section.fits(samples.len(), 4) {
        return;
    }
    let across = section.across as isize;
    let last = section.q0 + (SECTION_LINES - 1) * section.along;
    let (p_first, q_first) = (
        side(samples, section.q0 - section.across, -across),
        side(samples, section.q0, across),
    );
    let (p_last, q_last) = (
        side(samples, last - section.across, -across),
        side(samples, last, across),
    );
    let (dp0, dp3) = (second_difference(&p_first), second_difference(&p_last));
    let (dq0, dq3) = (second_difference(&q_first), second_difference(&q_last));
    let (dpq0, dpq3) = (dp0 + dq0, dp3 + dq3);
    if dpq0 + dpq3 >= filter.beta {
        return;
    }
    let strong = strong_line(&p_first, &q_first, 2 * dpq0, filter)
        && strong_line(&p_last, &q_last, 2 * dpq3, filter);
    let side_limit = (filter.beta + (filter.beta >> 1)) >> 3;
    let extend_p = dp0 + dp3 < side_limit;
    let extend_q = dq0 + dq3 < side_limit;
    for line in 0..SECTION_LINES {
        let q0 = section.q0 + line * section.along;
        let p = side(samples, q0 - section.across, -across);
        let q = side(samples, q0, across);
        let (new_p, new_q) = if strong {
            strong_filter(&p, &q, filter.tc)
        } else if let Some(values) = weak_filter(&p, &q, filter, extend_p, extend_q) {
            values
        } else {
            continue;
        };
        if filter.modify_p {
            write_side(
                samples,
                q0 - section.across,
                -across,
                &new_p,
                filter.max_value,
            );
        }
        if filter.modify_q {
            write_side(samples, q0, across, &new_q, filter.max_value);
        }
    }
}

/// Writes the entries that hold a value. `first` and `step` follow [`side`].
fn write_side<S: Sample>(
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

/// (8-389) to (8-394).
fn strong_filter(p: &[i32; 4], q: &[i32; 4], tc: i32) -> ([Option<i32>; 3], [Option<i32>; 3]) {
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

/// Filtered `p` and `q` samples. `None` leaves that sample as it is.
type LumaFilterResult = ([Option<i32>; 3], [Option<i32>; 3]);

/// (8-395) to (8-402), or `None` when `Abs( Δ )` reaches `tC * 10`.
fn weak_filter(
    p: &[i32; 4],
    q: &[i32; 4],
    filter: &EdgeFilter,
    extend_p: bool,
    extend_q: bool,
) -> Option<LumaFilterResult> {
    let tc = filter.tc;
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

/// The filtering of one chroma section (8.7.2.5.5, 8.7.2.5.8). A section outside
/// `samples` is left alone.
pub(crate) fn filter_chroma_section<S: Sample>(
    samples: &mut [S],
    section: EdgeSection,
    filter: &EdgeFilter,
) {
    if !section.fits(samples.len(), 2) {
        return;
    }
    let tc = filter.tc;
    for line in 0..SECTION_LINES {
        let q0_at = section.q0 + line * section.along;
        let p0_at = q0_at - section.across;
        let p0 = samples[p0_at].to_i32();
        let p1 = samples[p0_at - section.across].to_i32();
        let q0 = samples[q0_at].to_i32();
        let q1 = samples[q0_at + section.across].to_i32();
        let delta = (((q0 - p0) << 2) + p1 - q1 + 4) >> 3;
        let delta = delta.clamp(-tc, tc);
        if filter.modify_p {
            samples[p0_at] = S::clipped(p0 + delta, filter.max_value);
        }
        if filter.modify_q {
            samples[q0_at] = S::clipped(q0 - delta, filter.max_value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
