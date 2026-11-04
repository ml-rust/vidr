//! Seeded damage to a stream, for each decoder's mutation harness. Zip name, seed, and kind
//! fix the mutated bytes.

use std::ops::Range;

/// Bits per flip at the low rate.
const LOW_RATE_BITS: usize = 100_000;
/// Bits per flip at the high rate.
const HIGH_RATE_BITS: usize = 512;
/// The longest overwritten run in bytes.
const MAX_RUN: usize = 4_096;
/// Bytes of a VCL NAL unit that count as its slice header.
pub(super) const SLICE_HEADER_BYTES: usize = 8;
/// The bytes of the NAL unit header.
const NAL_HEADER_BYTES: usize = 2;
/// The most NAL units one corruption kind touches.
pub(super) const MAX_CORRUPT_UNITS: usize = 4;

/// One way to damage a stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    FlipLow,
    FlipHigh,
    RunZeros,
    RunOnes,
    Truncate,
    DropNal,
    DuplicateNal,
    SwapNals,
    CorruptParameterSets,
    CorruptSliceHeaders,
}

impl Kind {
    pub(super) const ALL: [Self; 10] = [
        Self::FlipLow,
        Self::FlipHigh,
        Self::RunZeros,
        Self::RunOnes,
        Self::Truncate,
        Self::DropNal,
        Self::DuplicateNal,
        Self::SwapNals,
        Self::CorruptParameterSets,
        Self::CorruptSliceHeaders,
    ];

    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::FlipLow => "flip-low",
            Self::FlipHigh => "flip-high",
            Self::RunZeros => "run-zeros",
            Self::RunOnes => "run-ones",
            Self::Truncate => "truncate",
            Self::DropNal => "drop-nal",
            Self::DuplicateNal => "duplicate-nal",
            Self::SwapNals => "swap-nals",
            Self::CorruptParameterSets => "corrupt-parameter-sets",
            Self::CorruptSliceHeaders => "corrupt-slice-headers",
        }
    }
}

/// SplitMix64: a small generator with a full 64-bit state.
pub(super) struct Prng(u64);

impl Prng {
    /// A generator for one case. FNV-1a folds the zip name into the state.
    pub(super) fn new(name: &str, kind: Kind, seed: u64) -> Self {
        let mut state = 0xcbf2_9ce4_8422_2325u64;
        for byte in name.bytes().chain(kind.label().bytes()) {
            state = (state ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
        }
        Self(state ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    /// A value in `0..limit`, or 0 for a limit of 0.
    pub(super) fn below(&mut self, limit: usize) -> usize {
        if limit == 0 {
            return 0;
        }
        (self.next_u64() % limit as u64) as usize
    }
}

/// The start code offsets of `stream`, each with the bytes up to the next start code. A
/// start code of `00 00 00 01` starts at its second zero.
fn nal_spans(stream: &[u8]) -> Vec<Range<usize>> {
    let starts: Vec<usize> = stream
        .windows(3)
        .enumerate()
        .filter(|(_, window)| *window == [0, 0, 1])
        .map(|(index, _)| index)
        .collect();
    let ends = starts.iter().skip(1).copied().chain([stream.len()]);
    starts
        .iter()
        .copied()
        .zip(ends)
        .map(|(a, b)| a..b)
        .collect()
}

/// `nal_unit_type` of the unit in `span`, or `None` for a span too short to hold one.
fn nal_type(stream: &[u8], span: &Range<usize>) -> Option<u8> {
    stream.get(span.start + 3).map(|byte| (byte >> 1) & 0x3f)
}

pub(super) fn flip_bits(stream: &mut [u8], range: Range<usize>, count: usize, prng: &mut Prng) {
    for _ in 0..count {
        let at = range.start + prng.below(range.len());
        if let Some(byte) = stream.get_mut(at) {
            *byte ^= 1 << prng.below(8);
        }
    }
}

fn overwrite_run(stream: &mut [u8], value: u8, prng: &mut Prng) {
    let length = 1 + prng.below(MAX_RUN.min(stream.len() / 4 + 1));
    let start = prng.below(stream.len().saturating_sub(length) + 1);
    let end = (start + length).min(stream.len());
    if let Some(run) = stream.get_mut(start..end) {
        run.fill(value);
    }
}

/// Removes the unit at a random index.
fn drop_unit(stream: &[u8], spans: &[Range<usize>], prng: &mut Prng) -> Vec<u8> {
    let Some(span) = spans.get(prng.below(spans.len())) else {
        return stream.to_vec();
    };
    let mut out = stream[..span.start].to_vec();
    out.extend_from_slice(&stream[span.end..]);
    out
}

/// Sends the unit at a random index twice in a row.
fn duplicate_unit(stream: &[u8], spans: &[Range<usize>], prng: &mut Prng) -> Vec<u8> {
    let Some(span) = spans.get(prng.below(spans.len())) else {
        return stream.to_vec();
    };
    let mut out = stream[..span.end].to_vec();
    out.extend_from_slice(&stream[span.clone()]);
    out.extend_from_slice(&stream[span.end..]);
    out
}

/// Exchanges the units at two random indices.
fn swap_units(stream: &[u8], spans: &[Range<usize>], prng: &mut Prng) -> Vec<u8> {
    let (a, b) = (prng.below(spans.len()), prng.below(spans.len()));
    let (Some(first), Some(second)) = (spans.get(a.min(b)), spans.get(a.max(b))) else {
        return stream.to_vec();
    };
    if a == b {
        return stream.to_vec();
    }
    let mut out = stream[..first.start].to_vec();
    out.extend_from_slice(&stream[second.clone()]);
    out.extend_from_slice(&stream[first.end..second.start]);
    out.extend_from_slice(&stream[first.clone()]);
    out.extend_from_slice(&stream[second.end..]);
    out
}

/// Flips bits in the payload of up to [`MAX_CORRUPT_UNITS`] units that `pick` selects. A
/// unit takes one to three flips in its first `reach` payload bytes.
fn corrupt_units(
    stream: &mut [u8],
    spans: &[Range<usize>],
    pick: impl Fn(u8) -> bool,
    reach: usize,
    prng: &mut Prng,
) {
    let chosen: Vec<&Range<usize>> = spans
        .iter()
        .filter(|span| nal_type(stream, span).is_some_and(&pick))
        .collect();
    for _ in 0..=prng.below(MAX_CORRUPT_UNITS) {
        let Some(span) = chosen.get(prng.below(chosen.len())) else {
            return;
        };
        let start = span.start + 3 + NAL_HEADER_BYTES;
        let end = span.end.min(start.saturating_add(reach));
        if start < end {
            let flips = 1 + prng.below(3);
            flip_bits(stream, start..end, flips, prng);
        }
    }
}

/// The stream of one case. Name, seed, and kind fix its bytes.
pub(super) fn mutate(stream: &[u8], name: &str, kind: Kind, seed: u64) -> Vec<u8> {
    let mut prng = Prng::new(name, kind, seed);
    let spans = nal_spans(stream);
    let mut out = stream.to_vec();
    let whole = 0..out.len();
    match kind {
        Kind::FlipLow => {
            let count = (out.len() * 8 / LOW_RATE_BITS).max(1);
            flip_bits(&mut out, whole, count, &mut prng);
        }
        Kind::FlipHigh => {
            let count = (out.len() * 8 / HIGH_RATE_BITS).max(1);
            flip_bits(&mut out, whole, count, &mut prng);
        }
        Kind::RunZeros => overwrite_run(&mut out, 0, &mut prng),
        Kind::RunOnes => overwrite_run(&mut out, 0xff, &mut prng),
        Kind::Truncate => out.truncate(prng.below(stream.len())),
        Kind::DropNal => return drop_unit(stream, &spans, &mut prng),
        Kind::DuplicateNal => return duplicate_unit(stream, &spans, &mut prng),
        Kind::SwapNals => return swap_units(stream, &spans, &mut prng),
        Kind::CorruptParameterSets => {
            let pick = |nal: u8| matches!(nal, 32..=34);
            corrupt_units(&mut out, &spans, pick, usize::MAX, &mut prng);
        }
        Kind::CorruptSliceHeaders => {
            let pick = |nal: u8| nal < 32;
            corrupt_units(&mut out, &spans, pick, SLICE_HEADER_BYTES, &mut prng);
        }
    }
    out
}

/// Counts by outcome.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Counts {
    pub(super) decoded: usize,
    pub(super) error: usize,
    pub(super) panic: usize,
    pub(super) deadline: usize,
}

impl Counts {
    pub(super) fn add(&mut self, other: &Self) {
        self.decoded += other.decoded;
        self.error += other.error;
        self.panic += other.panic;
        self.deadline += other.deadline;
    }
}

/// The value of variable `name`, or `default` when it is unset.
pub(super) fn env_number(name: &str, default: u64) -> u64 {
    std::env::var(name).map_or(default, |text| {
        text.parse::<u64>().unwrap_or_else(|error| {
            panic!("{name} holds `{text}`: {error}. Set it to a whole number, or unset it.")
        })
    })
}

/// Three units: VPS, SPS, and a slice with a 4-byte start code.
#[cfg(test)]
pub(super) fn sample() -> Vec<u8> {
    let mut stream = vec![0, 0, 1, 32 << 1, 1, 0xaa, 0xbb];
    stream.extend_from_slice(&[0, 0, 1, 33 << 1, 1, 0xcc, 0xdd, 0xee]);
    stream.extend_from_slice(&[0, 0, 0, 1, 1 << 1, 1]);
    stream.extend_from_slice(&[0x55; 40]);
    stream
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_cover_each_unit() {
        let stream = sample();
        let spans = nal_spans(&stream);
        assert_eq!(spans.len(), 3);
        let types: Vec<_> = spans.iter().filter_map(|s| nal_type(&stream, s)).collect();
        assert_eq!(types, [32, 33, 1]);
        assert_eq!(spans[2].end, stream.len());
    }

    #[test]
    fn a_case_fixes_its_bytes() {
        let stream = sample();
        for kind in Kind::ALL {
            let first = mutate(&stream, "zip", kind, 5);
            assert_eq!(first, mutate(&stream, "zip", kind, 5), "{}", kind.label());
        }
        assert_ne!(
            Prng::new("zip", Kind::FlipLow, 1).next_u64(),
            Prng::new("zip", Kind::FlipLow, 2).next_u64()
        );
        assert_ne!(
            Prng::new("zip", Kind::FlipLow, 1).next_u64(),
            Prng::new("other", Kind::FlipLow, 1).next_u64()
        );
    }

    #[test]
    fn parameter_set_corruption_leaves_the_slices() {
        let stream = sample();
        let spans = nal_spans(&stream);
        for seed in 0..64 {
            let out = mutate(&stream, "zip", Kind::CorruptParameterSets, seed);
            assert_eq!(out.len(), stream.len());
            assert_eq!(out[spans[2].clone()], stream[spans[2].clone()]);
            assert_eq!(out[..5], stream[..5]);
        }
    }

    #[test]
    fn slice_header_corruption_stays_in_its_bytes() {
        let stream = sample();
        let spans = nal_spans(&stream);
        let payload = spans[2].start + 3 + NAL_HEADER_BYTES;
        for seed in 0..64 {
            let out = mutate(&stream, "zip", Kind::CorruptSliceHeaders, seed);
            assert_eq!(out[..payload], stream[..payload]);
            let end = payload + SLICE_HEADER_BYTES;
            assert_eq!(out[end..], stream[end..]);
        }
    }

    #[test]
    fn unit_edits_keep_every_other_unit() {
        let stream = sample();
        let spans = nal_spans(&stream);
        let units: Vec<&[u8]> = spans.iter().map(|span| &stream[span.clone()]).collect();
        for seed in 0..32 {
            let dropped = mutate(&stream, "zip", Kind::DropNal, seed);
            assert!(
                units
                    .iter()
                    .any(|u| dropped.len() == stream.len() - u.len())
            );
            let doubled = mutate(&stream, "zip", Kind::DuplicateNal, seed);
            assert!(
                units
                    .iter()
                    .any(|u| doubled.len() == stream.len() + u.len())
            );
            let swapped = mutate(&stream, "zip", Kind::SwapNals, seed);
            assert_eq!(swapped.len(), stream.len());
            let mut sorted = swapped;
            sorted.sort_unstable();
            let mut expected = stream.clone();
            expected.sort_unstable();
            assert_eq!(sorted, expected);
        }
    }

    #[test]
    fn a_cut_stays_inside_the_stream() {
        let stream = sample();
        for seed in 0..32 {
            let out = mutate(&stream, "zip", Kind::Truncate, seed);
            assert!(out.len() < stream.len());
            assert_eq!(out[..], stream[..out.len()]);
        }
    }

    #[test]
    fn runs_write_one_value() {
        let stream = sample();
        for seed in 0..32 {
            let out = mutate(&stream, "zip", Kind::RunOnes, seed);
            assert_eq!(out.len(), stream.len());
            assert!(out.iter().zip(&stream).all(|(a, b)| a == b || *a == 0xff));
        }
    }
}
