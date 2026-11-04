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
use super::mutate::{Counts, Kind, env_number, mutate};
use super::runner::{panic_text, run_each};
use crate::cpu_hevc::packets::nal_packets;

const DEFAULT_SEEDS: u64 = 8;
const DEFAULT_DEADLINE_SECS: u64 = 600;
/// Frames one send can leave queued on one thread: the largest `MaxDpbSize` (A-2), and the
/// picture that the send finishes.
const MAX_QUEUED_FRAMES: usize = 17;
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
    use super::super::mutate::sample;
    use super::*;

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
