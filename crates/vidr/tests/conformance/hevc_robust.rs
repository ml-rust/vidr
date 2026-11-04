//! Bad input through `HevcDecoder`: seeded mutations of the JCT-VC HEVC_v1 zips. Each
//! stream ends in a decoded picture or a named error. A panic, a stall, or a frame pile
//! fails the test.
//!
//! `scripts/conformance/fetch_hevc.sh` fetches the zips. Pass `--no-capture` to see the
//! table on a run that passes.
//!
//! | Variable | Meaning | Default |
//! | --- | --- | --- |
//! | `VIDR_HEVC_CONFORMANCE` | The directory of zips | Skip with a notice |
//! | `VIDR_HEVC_ONLY` | Run zips whose name holds this text | Every zip |
//! | `VIDR_HEVC_ROBUST_SEEDS` | Seeds per zip and kind, from 0 | 8 |
//! | `VIDR_HEVC_ROBUST_DEADLINE_SECS` | Seconds one stream can take | 600 |
//! | `VIDR_HEVC_ROBUST_KEEP` | A directory for each failing stream | No file |
//! | `VIDR_HEVC_ROBUST_TRACE` | Set to print each case before it runs | No line |
//! | `VIDR_CONFORMANCE_JOBS` | Zips decoded at once | Half the logical CPUs |
//! | `VIDR_HEVC_THREADS` | Worker threads per decoder. 1 decodes on the calling thread | 1 |
//!
//! Zip name, seed, and kind fix the mutated bytes. To run one case again, set
//! `VIDR_HEVC_ONLY` to its zip and `VIDR_HEVC_ROBUST_SEEDS` above its seed.
//!
//! Each seed with remainder 3 by 4 opens the decoder with the picture hash check, so
//! hash messages take bad input too. The other seeds run without it, because a changed
//! picture then keeps feeding later pictures.
//!
//! Rust cannot stop a thread. A stream over the deadline fails, and its thread runs on
//! until the test process ends.
//!
//! An abort or a stack overflow ends the process with no table. `VIDR_HEVC_ROBUST_TRACE`
//! names the cases that were running. This test sets no counting allocator: the test
//! crate holds every suite, and one peak over parallel workers names no stream. The
//! limits on `HevcDecoder` hold memory.

use std::fmt;
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::Duration;

use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use vidr::decode::Decoder;
use vidr::decode::hevc::{HevcDecoder, HevcOptions};

use super::hevc::{read_stream, selected_zips, zip_name};
use super::hevc_decode::hevc_threads;
use super::runner::{panic_text, run_each};
use crate::cpu_hevc::packets::nal_packets;

const DEFAULT_SEEDS: u64 = 8;
const DEFAULT_DEADLINE_SECS: u64 = 600;
/// Frames one send can leave queued on one thread: the largest `MaxDpbSize` (A-2), and the
/// picture that the send finishes.
const MAX_QUEUED_FRAMES: usize = 17;
/// Bits per flip at the low rate.
const LOW_RATE_BITS: usize = 100_000;
/// Bits per flip at the high rate.
const HIGH_RATE_BITS: usize = 512;
/// The longest overwritten run in bytes.
const MAX_RUN: usize = 4_096;
/// Bytes of a VCL NAL unit that count as its slice header.
const SLICE_HEADER_BYTES: usize = 8;
/// The bytes of the NAL unit header.
const NAL_HEADER_BYTES: usize = 2;
/// The most NAL units one corruption kind touches.
const MAX_CORRUPT_UNITS: usize = 4;

/// One way to damage a stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
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
    const ALL: [Self; 10] = [
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

    const fn label(self) -> &'static str {
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
struct Prng(u64);

impl Prng {
    /// A generator for one case. FNV-1a folds the zip name into the state.
    fn new(name: &str, kind: Kind, seed: u64) -> Self {
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
    fn below(&mut self, limit: usize) -> usize {
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

fn flip_bits(stream: &mut [u8], range: Range<usize>, count: usize, prng: &mut Prng) {
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
fn mutate(stream: &[u8], name: &str, kind: Kind, seed: u64) -> Vec<u8> {
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

/// Frames one send can leave queued. Workers add the pictures in flight: their thread count
/// plus the largest `MaxDpbSize`, each with one output waiting.
const fn max_queued_frames(threads: usize) -> usize {
    if threads == 1 {
        MAX_QUEUED_FRAMES
    } else {
        MAX_QUEUED_FRAMES + threads + 16
    }
}

/// How one stream ended.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Decoded,
    Error,
    Panic(String),
    Deadline,
}

/// What a decode reports to the pool worker.
#[derive(Debug)]
struct Decode {
    outcome: Outcome,
    peak_queued: usize,
}

/// Sends every NAL unit, and drains the frames after each. The first error ends the
/// stream, because each later call returns the same error.
fn decode(stream: &[u8], check_hash: bool) -> Decode {
    let device = CpuDevice::new();
    let options = HevcOptions {
        threads: Some(hevc_threads()),
        check_picture_hash: check_hash,
    };
    let opened = HevcDecoder::<CpuRuntime>::with_options(&device, &[], options);
    let mut peak_queued = 0;
    let mut finish = |decoder: &mut HevcDecoder<CpuRuntime>| {
        peak_queued = peak_queued.max(decoder.queued());
        while decoder.receive().is_some() {}
    };
    let ended = opened.ok().and_then(|mut decoder| {
        let packets = nal_packets(stream).ok()?;
        for (packet, pts) in packets {
            let sent = decoder.send(&packet, pts);
            finish(&mut decoder);
            sent.ok()?;
        }
        let flushed = decoder.flush();
        finish(&mut decoder);
        flushed.ok()
    });
    Decode {
        outcome: ended.map_or(Outcome::Error, |()| Outcome::Decoded),
        peak_queued,
    }
}

/// Runs one stream on its own thread, and waits up to `deadline`. A panic is a result of
/// this stream alone.
fn decode_with_deadline(stream: Arc<Vec<u8>>, check_hash: bool, deadline: Duration) -> Decode {
    let (sender, receiver) = mpsc::channel();
    let spawned = thread::Builder::new()
        .name("hevc-robust".to_owned())
        .spawn(move || {
            let result = catch_unwind(AssertUnwindSafe(|| decode(&stream, check_hash)));
            let decode = result.unwrap_or_else(|panic| Decode {
                outcome: Outcome::Panic(panic_text(&*panic).to_owned()),
                peak_queued: 0,
            });
            // The receiver leaves after the deadline.
            let _ = sender.send(decode);
        });
    let stopped = |text: String| Decode {
        outcome: Outcome::Panic(text),
        peak_queued: 0,
    };
    if let Err(error) = spawned {
        return stopped(format!("spawning the thread: {error}"));
    }
    match receiver.recv_timeout(deadline) {
        Ok(decode) => decode,
        Err(RecvTimeoutError::Timeout) => Decode {
            outcome: Outcome::Deadline,
            peak_queued: 0,
        },
        Err(RecvTimeoutError::Disconnected) => {
            stopped("the thread ended with no result".to_owned())
        }
    }
}

/// Counts by outcome.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counts {
    decoded: usize,
    error: usize,
    panic: usize,
    deadline: usize,
}

impl Counts {
    fn add(&mut self, other: &Self) {
        self.decoded += other.decoded;
        self.error += other.error;
        self.panic += other.panic;
        self.deadline += other.deadline;
    }
}

/// The run's settings.
#[derive(Clone, Debug)]
struct Settings {
    seeds: u64,
    deadline: Duration,
    keep: Option<PathBuf>,
    trace: bool,
}

/// One failing case, with its reproduction.
#[derive(Debug)]
struct Failure {
    zip: String,
    seed: u64,
    kind: Kind,
    detail: String,
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "zip {} seed {} kind {}: {}",
            self.zip,
            self.seed,
            self.kind.label(),
            self.detail
        )
    }
}

/// What one zip did.
#[derive(Debug, Default)]
struct ZipReport {
    /// One entry per [`Kind::ALL`].
    counts: [Counts; Kind::ALL.len()],
    failures: Vec<Failure>,
    peak_queued: usize,
}

/// Writes a failing stream to the keep directory, and returns the text for the report.
fn keep_stream(settings: &Settings, failure: &Failure, stream: &[u8]) -> String {
    let Some(directory) = &settings.keep else {
        return String::new();
    };
    let name = format!(
        "{}.{}.{}.hevc",
        failure.zip,
        failure.kind.label(),
        failure.seed
    );
    let path = directory.join(name);
    match std::fs::write(&path, stream) {
        Ok(()) => format!(". The stream is at {}", path.display()),
        Err(error) => format!(". Writing {} fails: {error}", path.display()),
    }
}

/// Every mutation of one zip.
fn run_zip(zip: &Path, settings: &Settings) -> ZipReport {
    let name = zip_name(zip);
    let mut report = ZipReport::default();
    let stream = match read_stream(zip) {
        Ok((stream, _)) => stream,
        Err(detail) => {
            report.failures.push(Failure {
                zip: name,
                seed: 0,
                kind: Kind::FlipLow,
                detail: format!("reading the zip: {detail}"),
            });
            return report;
        }
    };
    for (index, kind) in Kind::ALL.into_iter().enumerate() {
        for seed in 0..settings.seeds {
            if settings.trace {
                eprintln!("running zip {name} seed {seed} kind {}", kind.label());
            }
            let mutated = Arc::new(mutate(&stream, &name, kind, seed));
            let check_hash = seed % 4 == 3;
            let run = decode_with_deadline(Arc::clone(&mutated), check_hash, settings.deadline);
            report.peak_queued = report.peak_queued.max(run.peak_queued);
            let counts = &mut report.counts[index];
            let mut detail = match run.outcome {
                Outcome::Decoded => {
                    counts.decoded += 1;
                    None
                }
                Outcome::Error => {
                    counts.error += 1;
                    None
                }
                Outcome::Panic(text) => {
                    counts.panic += 1;
                    Some(format!("panic: {text}"))
                }
                Outcome::Deadline => {
                    counts.deadline += 1;
                    Some(format!(
                        "over the deadline of {} s",
                        settings.deadline.as_secs()
                    ))
                }
            };
            let limit = max_queued_frames(hevc_threads());
            if run.peak_queued > limit {
                detail = Some(format!(
                    "{} frames queued after one send, and the limit is {limit}",
                    run.peak_queued
                ));
            }
            if let Some(detail) = detail {
                let mut failure = Failure {
                    zip: name.clone(),
                    seed,
                    kind,
                    detail,
                };
                let kept = keep_stream(settings, &failure, &mutated);
                failure.detail.push_str(&kept);
                eprintln!("{failure}");
                report.failures.push(failure);
            }
        }
    }
    report
}

/// The value of variable `name`, or `default` when it is unset.
fn env_number(name: &str, default: u64) -> u64 {
    std::env::var(name).map_or(default, |text| {
        text.parse::<u64>().unwrap_or_else(|error| {
            panic!("{name} holds `{text}`: {error}. Set it to a whole number, or unset it.")
        })
    })
}

#[test]
fn hevc_mutations_stay_safe() {
    let Some(zips) = selected_zips() else {
        return;
    };
    let settings = Settings {
        seeds: env_number("VIDR_HEVC_ROBUST_SEEDS", DEFAULT_SEEDS),
        deadline: Duration::from_secs(env_number(
            "VIDR_HEVC_ROBUST_DEADLINE_SECS",
            DEFAULT_DEADLINE_SECS,
        )),
        keep: std::env::var_os("VIDR_HEVC_ROBUST_KEEP").map(PathBuf::from),
        trace: std::env::var_os("VIDR_HEVC_ROBUST_TRACE").is_some(),
    };
    if let Some(keep) = &settings.keep {
        std::fs::create_dir_all(keep).unwrap_or_else(|error| {
            panic!(
                "creating {}: {error}. Set VIDR_HEVC_ROBUST_KEEP to a path you can write.",
                keep.display()
            )
        });
    }
    let reports = run_each(&zips, |zip| run_zip(zip, &settings));

    println!("| Kind | Decoded | Error | Panic | Deadline |");
    println!("| --- | --- | --- | --- | --- |");
    let mut total = Counts::default();
    for (index, kind) in Kind::ALL.into_iter().enumerate() {
        let mut counts = Counts::default();
        for report in &reports {
            counts.add(&report.counts[index]);
        }
        total.add(&counts);
        println!(
            "| {} | {} | {} | {} | {} |",
            kind.label(),
            counts.decoded,
            counts.error,
            counts.panic,
            counts.deadline
        );
    }
    println!(
        "| total | {} | {} | {} | {} |",
        total.decoded, total.error, total.panic, total.deadline
    );
    let peak = reports.iter().map(|report| report.peak_queued).max();
    println!(
        "{} zips, {} seeds per kind, deadline {} s, peak queued frames {}",
        reports.len(),
        settings.seeds,
        settings.deadline.as_secs(),
        peak.unwrap_or(0)
    );
    let failures: Vec<String> = reports
        .iter()
        .flat_map(|report| &report.failures)
        .map(Failure::to_string)
        .collect();
    assert!(
        failures.is_empty(),
        "{} cases fail:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three units: VPS, SPS, and a slice with a 4-byte start code.
    fn sample() -> Vec<u8> {
        let mut stream = vec![0, 0, 1, 32 << 1, 1, 0xaa, 0xbb];
        stream.extend_from_slice(&[0, 0, 1, 33 << 1, 1, 0xcc, 0xdd, 0xee]);
        stream.extend_from_slice(&[0, 0, 0, 1, 1 << 1, 1]);
        stream.extend_from_slice(&[0x55; 40]);
        stream
    }

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

    /// Garbage ends in an error or a clean flush on the pool's own path.
    #[test]
    fn garbage_ends_as_an_outcome() {
        let run = decode_with_deadline(
            Arc::new(vec![0xff; 64]),
            false,
            Duration::from_secs(DEFAULT_DEADLINE_SECS),
        );
        assert_eq!(run.outcome, Outcome::Error);
        let run = decode_with_deadline(
            Arc::new(sample()),
            true,
            Duration::from_secs(DEFAULT_DEADLINE_SECS),
        );
        assert_eq!(run.outcome, Outcome::Error);
    }

    /// A zero deadline can lose the race to a fast thread. Either way it returns.
    #[test]
    fn a_deadline_does_not_wait_for_the_thread() {
        let run = decode_with_deadline(Arc::new(Vec::new()), false, Duration::ZERO);
        assert!(matches!(run.outcome, Outcome::Deadline | Outcome::Decoded));
    }

    #[test]
    fn a_failure_names_its_case() {
        let failure = Failure {
            zip: "AMP_A_Samsung_7".to_owned(),
            seed: 3,
            kind: Kind::SwapNals,
            detail: "panic: index".to_owned(),
        };
        assert_eq!(
            failure.to_string(),
            "zip AMP_A_Samsung_7 seed 3 kind swap-nals: panic: index"
        );
    }
}
