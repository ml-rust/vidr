//! Deblocking kernels: the sample decisions and filters of 8.7.2.5.3 and 8.7.2.5.6
//! to 8.7.2.5.8, over a plane's samples. A vertical edge and a horizontal edge
//! differ only in the two index steps. A section loads into one vector per distance from
//! the edge, with one lane per line, and the filters run with no branch per line.
//! Samples of up to 10 bits keep every sum inside 16 bits.

use fearless_simd::{Bytes, Simd, i16x8, i32x4, i64x2, mask16x8};

use super::sample::Sample;
use super::simd::clip_x8;

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

/// The samples of one section: index 0 to 3 is `p3` to `p0`, and index 4 to 7 is `q0` to
/// `q3`. Lanes 0 to 3 of each vector hold the lines. The other lanes hold no meaning.
type SectionLanes<V> = [i16x8<V>; 8];

/// Two vectors of 32-bit lanes zipped: `( a0, b0, a1, b1 )`, `( a2, b2, a3, b3 )`.
#[inline(always)]
fn zip32<V: Simd>(simd: V, a: i16x8<V>, b: i16x8<V>) -> (i16x8<V>, i16x8<V>) {
    let (low, high) = simd.interleave_i32x4(a.bitcast::<i32x4<V>>(), b.bitcast());
    (low.bitcast(), high.bitcast())
}

/// The upper 64 bits of `a` moved to its lower 64 bits.
#[inline(always)]
fn upper_half<V: Simd>(simd: V, a: i16x8<V>) -> i16x8<V> {
    let lanes: i64x2<V> = a.bitcast();
    simd.zip_high_i64x2(lanes, lanes).bitcast()
}

/// The lower 64 bits of `a` then the lower 64 bits of `b`, and the same of the upper 64
/// bits.
#[inline(always)]
fn zip64<V: Simd>(simd: V, a: i16x8<V>, b: i16x8<V>) -> (i16x8<V>, i16x8<V>) {
    let (low, high) = simd.interleave_i64x2(a.bitcast::<i64x2<V>>(), b.bitcast());
    (low.bitcast(), high.bitcast())
}

/// One line of eight samples of a vertical edge.
#[inline(always)]
fn load_row<V: Simd, S: Sample>(simd: V, samples: &[S], start: usize) -> Option<i16x8<V>> {
    let lanes = samples.get(start..)?.first_chunk::<8>()?;
    Some(S::load_x8(simd, lanes))
}

/// The four lines at one distance from an edge that is not vertical.
#[inline(always)]
fn load_lines<V: Simd, S: Sample>(
    simd: V,
    samples: &[S],
    start: usize,
    along: usize,
) -> Option<i16x8<V>> {
    let mut lines = [S::default(); 8];
    if along == 1 {
        let run = samples.get(start..)?.first_chunk::<SECTION_LINES>()?;
        for (line, &sample) in lines.iter_mut().zip(run) {
            *line = sample;
        }
    } else {
        for (index, line) in lines.iter_mut().enumerate().take(SECTION_LINES) {
            *line = *samples.get(start + index * along)?;
        }
    }
    Some(S::load_x8(simd, &lines))
}

/// Reads a section that [`EdgeSection::fits`] accepted. A vertical edge reads four rows
/// of eight samples and transposes them. Any other edge reads eight runs of four lines.
#[inline(always)]
fn load_section<V: Simd, S: Sample>(
    simd: V,
    samples: &[S],
    section: EdgeSection,
) -> Option<SectionLanes<V>> {
    let (first, across, along) = (
        section.q0.checked_sub(4 * section.across)?,
        section.across,
        section.along,
    );
    if across == 1 {
        let r0 = load_row(simd, samples, first)?;
        let r1 = load_row(simd, samples, first + along)?;
        let r2 = load_row(simd, samples, first + 2 * along)?;
        let r3 = load_row(simd, samples, first + 3 * along)?;
        let (a0, a1) = simd.interleave_i16x8(r0, r1);
        let (b0, b1) = simd.interleave_i16x8(r2, r3);
        let (p3, p1) = zip32(simd, a0, b0);
        let (q0, q2) = zip32(simd, a1, b1);
        return Some([
            p3,
            upper_half(simd, p3),
            p1,
            upper_half(simd, p1),
            q0,
            upper_half(simd, q0),
            q2,
            upper_half(simd, q2),
        ]);
    }
    Some([
        load_lines(simd, samples, first, along)?,
        load_lines(simd, samples, first + across, along)?,
        load_lines(simd, samples, first + 2 * across, along)?,
        load_lines(simd, samples, first + 3 * across, along)?,
        load_lines(simd, samples, first + 4 * across, along)?,
        load_lines(simd, samples, first + 5 * across, along)?,
        load_lines(simd, samples, first + 6 * across, along)?,
        load_lines(simd, samples, first + 7 * across, along)?,
    ])
}

/// Writes a section that [`EdgeSection::fits`] accepted. A vertical edge writes all eight
/// samples of each line, and any other edge skips `p3` and `q3`. Unchanged lanes hold
/// their old samples.
#[inline(always)]
fn store_section<V: Simd, S: Sample>(
    simd: V,
    samples: &mut [S],
    section: EdgeSection,
    lanes: &SectionLanes<V>,
) {
    let first = section.q0 - 4 * section.across;
    if section.across == 1 {
        let [p3, p2, p1, p0, q0, q1, q2, q3] = *lanes;
        let (e0, _) = simd.interleave_i16x8(p3, p2);
        let (e1, _) = simd.interleave_i16x8(p1, p0);
        let (e2, _) = simd.interleave_i16x8(q0, q1);
        let (e3, _) = simd.interleave_i16x8(q2, q3);
        let (f0, f1) = zip32(simd, e0, e1);
        let (g0, g1) = zip32(simd, e2, e3);
        let (r0, r1) = zip64(simd, f0, g0);
        let (r2, r3) = zip64(simd, f1, g1);
        for (line, row) in [r0, r1, r2, r3].into_iter().enumerate() {
            if let Some(target) = samples
                .get_mut(first + line * section.along..)
                .and_then(<[S]>::first_chunk_mut::<8>)
            {
                *target = S::pack_x8(simd, row);
            }
        }
        return;
    }
    for (distance, &lane) in lanes.iter().enumerate().take(7).skip(1) {
        let start = first + distance * section.across;
        let packed = S::pack_x8(simd, lane);
        if section.along == 1 {
            if let Some(target) = samples
                .get_mut(start..)
                .and_then(<[S]>::first_chunk_mut::<SECTION_LINES>)
            {
                for (sample, &value) in target.iter_mut().zip(&packed) {
                    *sample = value;
                }
            }
        } else {
            for (index, &value) in packed.iter().enumerate().take(SECTION_LINES) {
                if let Some(target) = samples.get_mut(start + index * section.along) {
                    *target = value;
                }
            }
        }
    }
}

/// `Abs( x2 - 2 * x1 + x0 )`, (8-352) to (8-355), in each lane.
#[inline(always)]
fn second_difference<V: Simd>(simd: V, x0: i16x8<V>, x1: i16x8<V>, x2: i16x8<V>) -> i16x8<V> {
    let doubled = simd.add_i16x8(x1, x1);
    simd.abs_i16x8(simd.add_i16x8(simd.sub_i16x8(x2, doubled), x0))
}

/// `Abs( a - b )` in each lane.
#[inline(always)]
fn distance<V: Simd>(simd: V, a: i16x8<V>, b: i16x8<V>) -> i16x8<V> {
    simd.abs_i16x8(simd.sub_i16x8(a, b))
}

/// The decision and filtering of one luma section (8.7.2.5.3, 8.7.2.5.4). A section
/// outside `samples` is left alone.
#[inline(always)]
pub(crate) fn luma_section<V: Simd, S: Sample>(
    simd: V,
    samples: &mut [S],
    section: EdgeSection,
    filter: &EdgeFilter,
) {
    if !section.fits(samples.len(), 4) {
        return;
    }
    let Some(lanes) = load_section(simd, samples, section) else {
        return;
    };
    let [p3, p2, p1, p0, q0, q1, q2, q3] = lanes;
    let dp: [i16; 8] = second_difference(simd, p0, p1, p2).into();
    let dq: [i16; 8] = second_difference(simd, q0, q1, q2).into();
    let (dp0, dp3) = (i32::from(dp[0]), i32::from(dp[3]));
    let (dq0, dq3) = (i32::from(dq[0]), i32::from(dq[3]));
    let (dpq0, dpq3) = (dp0 + dq0, dp3 + dq3);
    if dpq0 + dpq3 >= filter.beta {
        return;
    }
    // The decision process for a luma sample (8.7.2.5.6), on lines 0 and 3.
    let outer: [i16; 8] = simd
        .add_i16x8(distance(simd, p3, p0), distance(simd, q0, q3))
        .into();
    let across: [i16; 8] = distance(simd, p0, q0).into();
    let line_is_strong = |line: usize, dpq: i32| {
        dpq < (filter.beta >> 2)
            && i32::from(outer[line]) < (filter.beta >> 3)
            && i32::from(across[line]) < ((5 * filter.tc + 1) >> 1)
    };
    let strong = line_is_strong(0, 2 * dpq0) && line_is_strong(3, 2 * dpq3);
    let mut new_lanes = if strong {
        strong_filter(simd, &lanes, filter)
    } else {
        let side_limit = (filter.beta + (filter.beta >> 1)) >> 3;
        let extend = (dp0 + dp3 < side_limit, dq0 + dq3 < side_limit);
        weak_filter(simd, &lanes, filter, extend)
    };
    let max = simd.splat_i16x8(filter.max_value as i16);
    for lane in &mut new_lanes {
        *lane = clip_x8(simd, *lane, max);
    }
    if !filter.modify_p {
        new_lanes[..4].copy_from_slice(&lanes[..4]);
    }
    if !filter.modify_q {
        new_lanes[4..].copy_from_slice(&lanes[4..]);
    }
    store_section(simd, samples, section, &new_lanes);
}

/// One sum of (8-389) to (8-394), clipped to `2 * tC` around `centre`.
#[inline(always)]
fn strong_value<V: Simd>(
    simd: V,
    sum: i16x8<V>,
    shift: u32,
    centre: i16x8<V>,
    limit: i16x8<V>,
) -> i16x8<V> {
    let value = simd.shr_i16x8(sum, shift);
    let low = simd.sub_i16x8(centre, limit);
    let high = simd.add_i16x8(centre, limit);
    simd.min_i16x8(simd.max_i16x8(value, low), high)
}

/// The sum of `terms` in each lane.
#[inline(always)]
fn sum_of<V: Simd, const N: usize>(simd: V, terms: [i16x8<V>; N]) -> i16x8<V> {
    let mut sum = simd.splat_i16x8(0);
    for term in terms {
        sum = simd.add_i16x8(sum, term);
    }
    sum
}

/// (8-389) to (8-394) on all lines of a section. `p3` and `q3` stay.
#[inline(always)]
fn strong_filter<V: Simd>(
    simd: V,
    lanes: &SectionLanes<V>,
    filter: &EdgeFilter,
) -> SectionLanes<V> {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = *lanes;
    let limit = simd.splat_i16x8((2 * filter.tc) as i16);
    let (two, four) = (simd.splat_i16x8(2), simd.splat_i16x8(4));
    // `p0 + q0`, a term of every sum.
    let c = simd.add_i16x8(p0, q0);
    let new_p0 = strong_value(
        simd,
        sum_of(simd, [p2, p1, p1, c, c, q1, four]),
        3,
        p0,
        limit,
    );
    let new_p1 = strong_value(simd, sum_of(simd, [p2, p1, c, two]), 2, p1, limit);
    let new_p2 = strong_value(
        simd,
        sum_of(simd, [p3, p3, p2, p2, p2, p1, c, four]),
        3,
        p2,
        limit,
    );
    let new_q0 = strong_value(
        simd,
        sum_of(simd, [p1, c, c, q1, q1, q2, four]),
        3,
        q0,
        limit,
    );
    let new_q1 = strong_value(simd, sum_of(simd, [c, q1, q2, two]), 2, q1, limit);
    let new_q2 = strong_value(
        simd,
        sum_of(simd, [c, q1, q2, q2, q2, q3, q3, four]),
        3,
        q2,
        limit,
    );
    [p3, new_p2, new_p1, new_p0, new_q0, new_q1, new_q2, q3]
}

/// The change of `p1` or `q1`, (8-398) and (8-399), added to `middle` where `apply` holds.
/// `signed_delta` is `Δ` for `p1` and `-Δ` for `q1`.
#[inline(always)]
fn weak_side<V: Simd>(
    simd: V,
    far: i16x8<V>,
    near: i16x8<V>,
    middle: i16x8<V>,
    signed_delta: i16x8<V>,
    apply: mask16x8<V>,
    half: i16,
) -> i16x8<V> {
    let mean = simd.shr_i16x8(sum_of(simd, [far, near, simd.splat_i16x8(1)]), 1);
    let step = simd.shr_i16x8(
        simd.add_i16x8(simd.sub_i16x8(mean, middle), signed_delta),
        1,
    );
    let step = simd.min_i16x8(
        simd.max_i16x8(step, simd.splat_i16x8(-half)),
        simd.splat_i16x8(half),
    );
    simd.add_i16x8(middle, simd.select_i16x8(apply, step, simd.splat_i16x8(0)))
}

/// (8-395) to (8-402) on all lines of a section. A line with `Abs( Δ )` at `tC * 10` or
/// more takes a zero change. `extend` holds `dEp` and `dEq`. A plane sample holds at most
/// `max_value`, so writing an unchanged line changes nothing.
#[inline(always)]
fn weak_filter<V: Simd>(
    simd: V,
    lanes: &SectionLanes<V>,
    filter: &EdgeFilter,
    extend: (bool, bool),
) -> SectionLanes<V> {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = *lanes;
    let tc = filter.tc as i16;
    let half = tc >> 1;
    let zero = simd.splat_i16x8(0);
    let delta = simd.shr_i16x8(
        simd.add_i16x8(
            simd.sub_i16x8(
                simd.mul_i16x8(simd.sub_i16x8(q0, p0), simd.splat_i16x8(9)),
                simd.mul_i16x8(simd.sub_i16x8(q1, p1), simd.splat_i16x8(3)),
            ),
            simd.splat_i16x8(8),
        ),
        4,
    );
    let apply = simd.simd_lt_i16x8(simd.abs_i16x8(delta), simd.splat_i16x8(tc * 10));
    let clipped = simd.min_i16x8(
        simd.max_i16x8(delta, simd.splat_i16x8(-tc)),
        simd.splat_i16x8(tc),
    );
    let change = simd.select_i16x8(apply, clipped, zero);
    let new_p0 = simd.add_i16x8(p0, change);
    let new_q0 = simd.sub_i16x8(q0, change);
    let new_p1 = if extend.0 {
        weak_side(simd, p2, p0, p1, clipped, apply, half)
    } else {
        p1
    };
    let new_q1 = if extend.1 {
        weak_side(simd, q2, q0, q1, simd.sub_i16x8(zero, clipped), apply, half)
    } else {
        q1
    };
    [p3, p2, new_p1, new_p0, new_q0, new_q1, q2, q3]
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
#[path = "deblock_simd_tests.rs"]
mod simd_tests;

#[cfg(test)]
#[path = "deblock_tests.rs"]
mod tests;
