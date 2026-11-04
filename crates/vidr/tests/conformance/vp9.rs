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

use std::ffi::OsStr;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::process::Command;

use containr::TrackKind;
use containr::mkv::Demuxer;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use vidr::DecodeError;
use vidr::decode::rusty_vp9::RustyVp9Decoder;
use vidr::decode::{DecodedFrame, Decoder, PixelFormat};

use super::md5::hex_digest;
use super::runner::{panic_text, run_each};

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

/// Whether the vector's name states a layout the adapter refuses.
fn refusal_expected(name: &str) -> bool {
    name.starts_with("vp91-") || name.starts_with("vp93-") || name.contains("12bit")
}

/// The vectors in `directory`: each file that has a `.md5` beside it, by name.
fn vectors(directory: &Path, filter: Option<&str>) -> Vec<PathBuf> {
    let entries = std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("reading {}: {error}", directory.display()));
    let mut found: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension() == Some(OsStr::new("md5")))
        .map(|path| path.with_extension(""))
        .filter(|path| path.is_file())
        .filter(|path| filter.is_none_or(|text| vector_name(path).contains(text)))
        .collect();
    found.sort();
    found
}

fn vector_name(path: &Path) -> String {
    path.file_name()
        .map_or_else(String::new, |name| name.to_string_lossy().into_owned())
}

/// The hash of each frame: the first word of each line.
fn expected_hashes(vector: &Path) -> Result<Vec<String>, String> {
    let mut path = vector.as_os_str().to_owned();
    path.push(".md5");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| format!("reading {}: {error}", PathBuf::from(&path).display()))?;
    Ok(text
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .map(str::to_ascii_lowercase)
        .collect())
}

/// One frame's hash in libvpx's layout. numr's CPU backend holds samples.
fn frame_hash(frame: &DecodedFrame<CpuRuntime>) -> Result<String, String> {
    let read = |plane: &numr::tensor::Tensor<CpuRuntime>| {
        plane
            .to_bytes()
            .map_err(|error| format!("reading a plane: {error}"))
    };
    let (mut y, uv) = (read(frame.y())?, read(frame.uv())?);
    let sample = frame.format().bytes_per_sample();
    let mut u = Vec::with_capacity(uv.len() / 2);
    let mut v = Vec::with_capacity(uv.len() / 2);
    for pair in uv.chunks_exact(2 * sample) {
        let (cb, cr) = pair.split_at(sample);
        u.extend_from_slice(cb);
        v.extend_from_slice(cr);
    }
    if frame.format() == PixelFormat::P010 {
        for plane in [&mut y, &mut u, &mut v] {
            for stored in plane.as_chunks_mut::<2>().0 {
                *stored = (u16::from_le_bytes(*stored) >> 6).to_le_bytes();
            }
        }
    }
    Ok(hex_digest(&[&y, &u, &v]))
}

/// The Matroska file of a vector: the WebM file itself, or an IVF file's remux
/// under `work`. `None` means ffmpeg did not run.
fn matroska_of(vector: &Path, work: &Path) -> Result<Option<PathBuf>, String> {
    if vector.extension() != Some(OsStr::new("ivf")) {
        return Ok(Some(vector.to_path_buf()));
    }
    std::fs::create_dir_all(work)
        .map_err(|error| format!("creating {}: {error}", work.display()))?;
    let remuxed = work.join(format!("{}.mkv", vector_name(vector)));
    let output = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(vector)
        .args(["-map", "0:v:0", "-c", "copy", "-f", "matroska"])
        .arg(&remuxed)
        .output();
    match output {
        Err(_) => Ok(None),
        Ok(output) if output.status.success() => Ok(Some(remuxed)),
        Ok(output) => Err(format!(
            "ffmpeg cannot remux the IVF file: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )),
    }
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
