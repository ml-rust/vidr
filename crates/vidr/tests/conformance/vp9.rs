//! libvpx's VP9 test vectors through `RustyVp9Decoder`, frame by frame.
//!
//! `scripts/conformance/fetch_vp9.sh` fetches the vectors and `.md5` files. Set
//! `VIDR_VP9_VECTORS` to that directory. Pass `--no-capture` to see the table on
//! a run that passes.
//!
//! | Variable | Meaning | Default |
//! | --- | --- | --- |
//! | `VIDR_VP9_VECTORS` | The directory of vectors | Skip with a notice |
//! | `VIDR_VP9_FILTER` | Run vectors whose name holds this text | Every vector |
//! | `VIDR_CONFORMANCE_JOBS` | Vectors decoded at once | Half the logical CPUs |
//!
//! Each line of `NAME.md5` holds one frame's hash. libvpx's `test/md5_helper.h`
//! hashes the whole visible Y plane, then U, then V, row by row, with no padding.
//! A chroma plane has `ceil(W / 2)` samples per row and `ceil(H / 2)` rows. An
//! 8-bit sample is one byte. A 10-bit or 12-bit sample is two little-endian bytes
//! holding its value. The tolerance is zero.
//!
//! A vector with `.ivf` passes through ffmpeg into Matroska first. containr reads
//! no IVF, and this repository parses no container. Without ffmpeg these vectors
//! skip by name.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use containr::TrackKind;
use containr::mkv::Demuxer;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use vidr::DecodeError;
use vidr::decode::Decoder;
use vidr::decode::rusty_vp9::RustyVp9Decoder;

use super::cpu_frame_hash::frame_hash;
use super::runner::{panic_text, run_each};
use super::vector_files::{expected_hashes, vector_name, vectors};
use super::vp9_vectors::{matroska_of, refusal_expected};

/// What one vector did.
enum Outcome {
    /// Every frame hash matches.
    Pass { frames: usize },
    /// The adapter refuses it by name, as its profile or depth requires.
    Refused { reason: String },
    /// The run has no tool to read it. The summary names it.
    Skipped { reason: String },
    /// A frame or the frame count differs, or the decoder returns an error.
    Fail { reason: String },
}

/// Compares frames against the hashes in order, and returns how many passed.
struct HashCheck {
    expected: Vec<String>,
    next: usize,
}

impl HashCheck {
    fn drain(&mut self, decoder: &mut RustyVp9Decoder<CpuRuntime>) -> Result<(), String> {
        while let Some(frame) = decoder.receive() {
            let hash = frame_hash(&frame)?;
            let Some(expected) = self.expected.get(self.next) else {
                return Err(format!(
                    "the decoder gives more frames than the {} hashes",
                    self.expected.len()
                ));
            };
            if &hash != expected {
                return Err(format!(
                    "frame {} ({}x{}): md5 {hash}, libvpx has {expected}",
                    self.next,
                    frame.display().width(),
                    frame.display().height()
                ));
            }
            self.next += 1;
        }
        Ok(())
    }
}

fn decode_vector(vector: &Path, work: &Path) -> Result<Outcome, String> {
    let expected = expected_hashes(vector)?;
    let Some(path) = matroska_of(vector, work)? else {
        return Ok(Outcome::Skipped {
            reason: "ffmpeg is not on PATH, and it remuxes IVF files".to_owned(),
        });
    };
    let mut demuxer = Demuxer::open_path(&path).map_err(|error| format!("demux: {error}"))?;
    let track = demuxer
        .tracks()
        .iter()
        .find(|track| track.kind == TrackKind::Video)
        .map(|track| track.id)
        .ok_or_else(|| "the file has no video track".to_owned())?;
    let packets = demuxer
        .track_packets(track)
        .map_err(|error| format!("demux: {error}"))?;

    let mut decoder = RustyVp9Decoder::<CpuRuntime>::new(&CpuDevice::new());
    let mut check = HashCheck { expected, next: 0 };
    let mut data = Vec::new();
    for packet in packets {
        let packet = packet.map_err(|error| format!("demux: {error}"))?;
        demuxer
            .read_packet_into(&packet, &mut data)
            .map_err(|error| format!("demux at packet {}: {error}", packet.index))?;
        match decoder.send(&data, packet.pts) {
            Ok(()) => {}
            Err(error @ DecodeError::Vp9Layout { .. }) => {
                return Ok(Outcome::Refused {
                    reason: error.to_string(),
                });
            }
            Err(error) => return Err(format!("packet {}: {error}", packet.index)),
        }
        check.drain(&mut decoder)?;
    }
    decoder.flush().map_err(|error| format!("flush: {error}"))?;
    check.drain(&mut decoder)?;
    if check.next != check.expected.len() {
        return Err(format!(
            "the decoder gives {} frames, and libvpx has {}",
            check.next,
            check.expected.len()
        ));
    }
    Ok(Outcome::Pass { frames: check.next })
}

/// Runs one vector. A panic in the decoder or an unexpected refusal is a failure.
fn run_vector(vector: &Path, work: &Path) -> Outcome {
    let name = vector_name(vector);
    let decoded = catch_unwind(AssertUnwindSafe(|| decode_vector(vector, work)));
    let outcome = match decoded {
        Ok(Ok(outcome)) => outcome,
        Ok(Err(reason)) => Outcome::Fail { reason },
        Err(panic) => Outcome::Fail {
            reason: format!("panic: {}", panic_text(&*panic)),
        },
    };
    match outcome {
        Outcome::Refused { reason } if !refusal_expected(&name) => Outcome::Fail {
            reason: format!("refused, and its name states a layout the adapter takes: {reason}"),
        },
        Outcome::Pass { .. } if refusal_expected(&name) => Outcome::Fail {
            reason: "decoded, and its name states a layout the adapter refuses".to_owned(),
        },
        other => other,
    }
}

#[test]
fn libvpx_vectors_match_frame_by_frame() {
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
        "{} holds no vector with a `.md5` file for filter {filter:?}. Run \
         `scripts/conformance/fetch_vp9.sh {}`.",
        directory.display(),
        directory.display()
    );
    let work = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("conformance_vp9");
    let outcomes = run_each(&vectors, |vector| run_vector(vector, &work));

    println!("| Vector | Result | Detail |");
    println!("| --- | --- | --- |");
    let (mut passed, mut refused, mut skipped) = (0, 0, 0);
    let mut failed = Vec::new();
    for (vector, outcome) in vectors.iter().zip(&outcomes) {
        let name = vector_name(vector);
        match outcome {
            Outcome::Pass { frames } => {
                passed += 1;
                println!("| {name} | pass | {frames} frames |");
            }
            Outcome::Refused { reason } => {
                refused += 1;
                println!("| {name} | refused | {reason} |");
            }
            Outcome::Skipped { reason } => {
                skipped += 1;
                println!("| {name} | skipped | {reason} |");
            }
            Outcome::Fail { reason } => {
                println!("| {name} | FAIL | {reason} |");
                failed.push(format!("{name}: {reason}"));
            }
        }
    }
    println!(
        "{} vectors: {passed} pass, {} fail, {refused} refused by name, {skipped} skipped",
        vectors.len(),
        failed.len()
    );
    assert!(
        failed.is_empty(),
        "{} vectors fail:\n{}",
        failed.len(),
        failed.join("\n")
    );
}
