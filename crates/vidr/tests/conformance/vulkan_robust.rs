//! Bad input through `VulkanDecoder`: seeded mutations of the HEVC_v1 zips and the libvpx
//! vectors. Each stream ends in decoded pictures or a named error. A panic, a deadline, a
//! validation error, or an error that does not repeat fails the test.
//!
//! numr holds one device per index, and concurrent use loses it. So cases run one after another
//! on the shared device. Pass `--no-capture` to see the table on a run that passes.
//!
//! | Variable | Meaning | Default |
//! | --- | --- | --- |
//! | `VIDR_HEVC_CONFORMANCE` | The directory of zips | Skip with a notice |
//! | `VIDR_HEVC_ONLY` | Run zips whose name holds this text | Every zip |
//! | `VIDR_VP9_VECTORS` | The directory of vectors | Skip with a notice |
//! | `VIDR_VP9_FILTER` | Run vectors whose name holds this text | Every vector |
//! | `VIDR_HEVC_ROBUST_SEEDS` | Seeds per stream and kind, from 0, for both codecs | 2 |
//! | `VIDR_HEVC_ROBUST_DEADLINE_SECS` | Seconds one stream can take, drop included | 600 |
//! | `VIDR_HEVC_ROBUST_TRACE` | Set to print each case before it runs | No line |
//!
//! A driver reached with bad data can hang or lose the device. A lost device counts as an error,
//! prints a line for its case, and the sweep opens a new device. When that open fails, the
//! remaining cases do not run, and the test fails. A stream over the deadline fails the test and
//! ends the sweep, because its thread still holds the device. Rust cannot stop that thread.
//!
//! A VP9 stream has no NAL units. So the kinds that name a unit act on a packet, and the two
//! corruption kinds flip bits at the start of a packet.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::Duration;

use vidr::DecodeError;
use vidr::decode::{DecodeCodec, Decoder};
use vidr::vulkan::{VulkanDecoder, VulkanDevice, VulkanOptions};

use super::hevc::{read_stream, selected_zips, zip_name};
use super::mutate::{Counts, Kind, env_number, mutate};
use super::runner::panic_text;
use crate::vulkan::support::{codec_present, device, try_hevc_units};

const DEFAULT_SEEDS: u64 = 2;
const DEFAULT_DEADLINE_SECS: u64 = 600;

type Packets = Vec<(Vec<u8>, i64)>;

/// How one stream ended.
#[derive(Debug)]
enum Outcome {
    Decoded,
    Error {
        error: DecodeError,
        /// What a later call returned, when it differs from `error`.
        changed: Option<String>,
    },
    Panic(String),
    Deadline,
}

/// Checks that each later call returns `error`.
fn after_error(decoder: &mut VulkanDecoder, error: DecodeError, pts: i64) -> Outcome {
    let sent = decoder.send(&[0], pts);
    let flushed = decoder.flush();
    let expected = Err(error.clone());
    let changed = (sent != expected || flushed != expected)
        .then(|| format!("after `{error}`, send gives {sent:?} and flush gives {flushed:?}"));
    Outcome::Error { error, changed }
}

/// A decoder for one stream. A VP9 session takes the sweep's size, so a `resize` vector grows.
fn open(device: &Arc<VulkanDevice>, codec: DecodeCodec) -> Result<VulkanDecoder, DecodeError> {
    #[cfg(feature = "mkv")]
    if codec == DecodeCodec::Vp9 {
        let size = super::vulkan_vp9::MAX_SESSION;
        return VulkanDecoder::with_max_coded_size(device, codec, &[], size);
    }
    VulkanDecoder::new(device, codec, &[])
}

/// Sends every packet, and drains the frames after each. The decoder drops on this thread, so
/// the deadline covers its drop.
fn decode(device: &Arc<VulkanDevice>, codec: DecodeCodec, packets: &Packets) -> Outcome {
    let mut decoder = match open(device, codec) {
        Ok(decoder) => decoder,
        Err(error) => {
            return Outcome::Error {
                error,
                changed: None,
            };
        }
    };
    for (packet, pts) in packets {
        let sent = decoder.send(packet, *pts);
        while decoder.receive().is_some() {}
        if let Err(error) = sent {
            return after_error(&mut decoder, error, *pts);
        }
    }
    let flushed = decoder.flush();
    while decoder.receive().is_some() {}
    match flushed {
        Ok(()) => Outcome::Decoded,
        Err(error) => after_error(&mut decoder, error, 0),
    }
}

/// Runs one stream on its own thread, and waits up to `deadline`.
fn decode_with_deadline(
    device: &Arc<VulkanDevice>,
    codec: DecodeCodec,
    packets: Packets,
    deadline: Duration,
) -> Outcome {
    let (sender, receiver) = mpsc::channel();
    let device = Arc::clone(device);
    let spawned = thread::Builder::new()
        .name("vulkan-robust".to_owned())
        .spawn(move || {
            let result = catch_unwind(AssertUnwindSafe(|| decode(&device, codec, &packets)));
            let outcome =
                result.unwrap_or_else(|panic| Outcome::Panic(panic_text(&*panic).to_owned()));
            // The receiver leaves after the deadline.
            let _ = sender.send(outcome);
        });
    if let Err(error) = spawned {
        return Outcome::Panic(format!("spawning the thread: {error}"));
    }
    match receiver.recv_timeout(deadline) {
        Ok(outcome) => outcome,
        Err(RecvTimeoutError::Timeout) => Outcome::Deadline,
        Err(RecvTimeoutError::Disconnected) => {
            Outcome::Panic("the thread ended with no result".to_owned())
        }
    }
}

/// One case of a sweep.
struct Case<'a> {
    stream: &'a str,
    kind: Kind,
    seed: u64,
}

impl Case<'_> {
    fn label(&self) -> String {
        format!(
            "{} seed {} kind {}",
            self.stream,
            self.seed,
            self.kind.label()
        )
    }
}

/// One sweep, on one device at a time.
struct Sweep {
    codec: DecodeCodec,
    device: Arc<VulkanDevice>,
    seeds: u64,
    deadline: Duration,
    trace: bool,
    /// One entry per [`Kind::ALL`].
    counts: [Counts; Kind::ALL.len()],
    /// Validation errors of this device already reported.
    seen_validation: usize,
    lost_devices: usize,
    failures: Vec<String>,
    /// Why the sweep ended early.
    stopped: Option<String>,
}

impl Sweep {
    fn new(codec: DecodeCodec, device: Arc<VulkanDevice>) -> Self {
        Self {
            codec,
            device,
            seeds: env_number("VIDR_HEVC_ROBUST_SEEDS", DEFAULT_SEEDS),
            deadline: Duration::from_secs(env_number(
                "VIDR_HEVC_ROBUST_DEADLINE_SECS",
                DEFAULT_DEADLINE_SECS,
            )),
            trace: std::env::var_os("VIDR_HEVC_ROBUST_TRACE").is_some(),
            counts: [Counts::default(); Kind::ALL.len()],
            seen_validation: 0,
            lost_devices: 0,
            failures: Vec::new(),
            stopped: None,
        }
    }

    /// Every mutation of one stream. `build` gives a case's packets, or the error of a stream
    /// with no packets to send.
    fn run_stream(&mut self, stream: &str, build: impl Fn(Kind, u64) -> Result<Packets, String>) {
        for (index, kind) in Kind::ALL.into_iter().enumerate() {
            for seed in 0..self.seeds {
                if self.stopped.is_some() {
                    return;
                }
                let case = Case { stream, kind, seed };
                if self.trace {
                    eprintln!("running {}", case.label());
                }
                match build(kind, seed) {
                    Ok(packets) => self.run_case(&case, index, packets),
                    Err(_) => self.counts[index].error += 1,
                }
            }
        }
    }

    fn run_case(&mut self, case: &Case<'_>, index: usize, packets: Packets) {
        let outcome = decode_with_deadline(&self.device, self.codec, packets, self.deadline);
        let counts = &mut self.counts[index];
        let mut lost = None;
        match outcome {
            Outcome::Decoded => counts.decoded += 1,
            Outcome::Error { error, changed } => {
                counts.error += 1;
                if let Some(changed) = changed {
                    self.failures.push(format!("{}: {changed}", case.label()));
                }
                if matches!(error, DecodeError::VulkanDeviceLost { .. }) {
                    lost = Some(error.to_string());
                }
            }
            Outcome::Panic(text) => {
                counts.panic += 1;
                self.failures
                    .push(format!("{}: panic: {text}", case.label()));
            }
            Outcome::Deadline => {
                counts.deadline += 1;
                let seconds = self.deadline.as_secs();
                self.failures.push(format!(
                    "{}: over the deadline of {seconds} s",
                    case.label()
                ));
                self.stopped = Some(format!(
                    "the thread of {} still holds the device",
                    case.label()
                ));
                return;
            }
        }
        self.read_validation(case);
        if let Some(text) = lost.or_else(|| {
            self.device
                .is_lost()
                .then(|| "the device reports a loss".to_owned())
        }) {
            self.replace_device(case, &text);
        }
    }

    /// Reports each validation error this case added. Bad data past a check reaches the driver.
    fn read_validation(&mut self, case: &Case<'_>) {
        let errors = self.device.validation_errors();
        for error in errors.iter().skip(self.seen_validation) {
            self.failures
                .push(format!("{}: validation error: {error}", case.label()));
        }
        self.seen_validation = errors.len();
    }

    fn replace_device(&mut self, case: &Case<'_>, text: &str) {
        self.lost_devices += 1;
        println!("device lost at {}: {text}", case.label());
        match VulkanDevice::open_shared(VulkanOptions { validation: true }) {
            Ok(device) => {
                self.device = device;
                self.seen_validation = 0;
            }
            Err(error) => {
                let reason = format!("no new device opens after {}: {error}", case.label());
                self.failures.push(reason.clone());
                self.stopped = Some(reason);
            }
        }
    }

    /// Prints the table, and fails on what the sweep found.
    fn finish(self, streams: usize) {
        println!("| Kind | Decoded | Error | Panic | Deadline |");
        println!("| --- | --- | --- | --- | --- |");
        let mut total = Counts::default();
        for (kind, counts) in Kind::ALL.into_iter().zip(&self.counts) {
            total.add(counts);
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
        println!(
            "{streams} streams, {} seeds per kind, deadline {} s, {} lost devices",
            self.seeds,
            self.deadline.as_secs(),
            self.lost_devices
        );
        if let Some(reason) = &self.stopped {
            println!("the sweep ended early: {reason}");
        }
        assert!(
            self.failures.is_empty(),
            "{} failures:\n{}",
            self.failures.len(),
            self.failures.join("\n")
        );
    }
}

#[test]
fn hevc_vulkan_mutations_stay_safe() {
    let Some(zips) = selected_zips() else {
        return;
    };
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, DecodeCodec::Hevc) {
        return;
    }
    let mut sweep = Sweep::new(DecodeCodec::Hevc, device);
    for zip in &zips {
        let name = zip_name(zip);
        match read_stream(zip) {
            Ok((stream, _)) => sweep.run_stream(&name, |kind, seed| {
                try_hevc_units(&mutate(&stream, &name, kind, seed))
                    .map_err(|error| error.to_string())
            }),
            Err(detail) => sweep
                .failures
                .push(format!("{name}: reading the zip: {detail}")),
        }
    }
    sweep.finish(zips.len());
}

#[cfg(feature = "mkv")]
mod vp9 {
    use std::path::PathBuf;

    use super::super::mutate::{MAX_CORRUPT_UNITS, Prng, SLICE_HEADER_BYTES, flip_bits};
    use super::super::vp9_vectors::{matroska_of, vector_name, vectors};
    use super::super::vulkan_vp9::read_packets;
    use super::*;

    /// Bytes at a packet's start that a parameter corruption reaches. They hold the uncompressed
    /// header of a frame with few segmentation or loop filter updates.
    const FRAME_HEADER_BYTES: usize = 32;

    /// The bytes of `mutated` cut back into the packets of `packets`, as far as they reach.
    fn cut_like(mutated: &[u8], packets: &[(Vec<u8>, i64)]) -> Packets {
        let mut rest = mutated;
        let mut out = Vec::with_capacity(packets.len());
        for (packet, pts) in packets {
            let (head, tail) = rest.split_at(packet.len().min(rest.len()));
            if head.is_empty() {
                break;
            }
            out.push((head.to_vec(), *pts));
            rest = tail;
        }
        out
    }

    /// The packets of one case. Name, seed, and kind fix their bytes.
    pub(super) fn mutate_packets(
        packets: &[(Vec<u8>, i64)],
        name: &str,
        kind: Kind,
        seed: u64,
    ) -> Packets {
        let mut prng = Prng::new(name, kind, seed);
        let mut out = packets.to_vec();
        let count = out.len();
        match kind {
            Kind::FlipLow | Kind::FlipHigh | Kind::RunZeros | Kind::RunOnes | Kind::Truncate => {
                let joined: Vec<u8> = packets
                    .iter()
                    .flat_map(|(packet, _)| packet.iter().copied())
                    .collect();
                return cut_like(&mutate(&joined, name, kind, seed), packets);
            }
            Kind::DropNal => {
                if count > 0 {
                    out.remove(prng.below(count));
                }
            }
            Kind::DuplicateNal => {
                if count > 0 {
                    let index = prng.below(count);
                    out.insert(index, out[index].clone());
                }
            }
            Kind::SwapNals => {
                if count > 0 {
                    out.swap(prng.below(count), prng.below(count));
                }
            }
            Kind::CorruptParameterSets | Kind::CorruptSliceHeaders => {
                let reach = if kind == Kind::CorruptParameterSets {
                    FRAME_HEADER_BYTES
                } else {
                    SLICE_HEADER_BYTES
                };
                for _ in 0..=prng.below(MAX_CORRUPT_UNITS) {
                    let index = prng.below(count);
                    if let Some((packet, _)) = out.get_mut(index) {
                        let end = packet.len().min(reach);
                        if end > 0 {
                            let flips = 1 + prng.below(3);
                            flip_bits(packet, 0..end, flips, &mut prng);
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn vp9_vulkan_mutations_stay_safe() {
        let Some(directory) = std::env::var_os("VIDR_VP9_VECTORS") else {
            eprintln!(
                "skipping: VIDR_VP9_VECTORS is unset. Run `scripts/conformance/fetch_vp9.sh DIR`, \
                 then set it to DIR."
            );
            return;
        };
        let directory = PathBuf::from(directory);
        let filter = std::env::var("VIDR_VP9_FILTER").ok();
        let vectors = vectors(&directory, filter.as_deref());
        assert!(
            !vectors.is_empty(),
            "{} holds no vector for filter {filter:?}. Run `scripts/conformance/fetch_vp9.sh {}`.",
            directory.display(),
            directory.display()
        );
        let Some((device, _guard)) = device() else {
            return;
        };
        if !codec_present(&device, DecodeCodec::Vp9) {
            return;
        }
        let work = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("conformance_vulkan_robust");
        let mut sweep = Sweep::new(DecodeCodec::Vp9, device);
        for vector in &vectors {
            let name = vector_name(vector);
            let packets = matroska_of(vector, &work).and_then(|path| match path {
                Some(path) => read_packets(&path),
                None => Err("ffmpeg is not on PATH, and it remuxes IVF files".to_owned()),
            });
            match packets {
                Ok(packets) => sweep.run_stream(&name, |kind, seed| {
                    Ok(mutate_packets(&packets, &name, kind, seed))
                }),
                Err(detail) => sweep.failures.push(format!("{name}: {detail}")),
            }
        }
        sweep.finish(vectors.len());
    }

    #[test]
    fn a_case_fixes_its_packets_and_a_cut_keeps_their_order() {
        let packets: Packets = vec![(vec![1; 40], 0), (vec![2; 40], 1), (vec![3; 40], 2)];
        for kind in Kind::ALL {
            let first = mutate_packets(&packets, "vector", kind, 5);
            assert_eq!(
                first,
                mutate_packets(&packets, "vector", kind, 5),
                "{}",
                kind.label()
            );
            assert!(
                first.iter().all(|(packet, _)| !packet.is_empty()),
                "{}",
                kind.label()
            );
        }
        let cut = cut_like(&[9; 50], &packets);
        assert_eq!(cut, [(vec![9; 40], 0), (vec![9; 10], 1)]);
        assert!(mutate_packets(&[], "vector", Kind::DropNal, 0).is_empty());
        assert!(mutate_packets(&[], "vector", Kind::CorruptSliceHeaders, 0).is_empty());
    }
}
