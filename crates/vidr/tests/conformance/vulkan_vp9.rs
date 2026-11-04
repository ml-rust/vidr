//! libvpx's VP9 test vectors through `VulkanDecoder`, frame by frame, on one GPU device.
//!
//! `scripts/conformance/fetch_vp9.sh` fetches the vectors and `.md5` files. Pass
//! `--no-capture` to see the table on a run that passes. numr holds one device at index 0,
//! and concurrent use loses it, so vectors run one after another.
//!
//! | Variable | Meaning | Default |
//! | --- | --- | --- |
//! | `VIDR_VP9_VECTORS` | The directory of vectors | Skip with a notice |
//! | `VIDR_VP9_FILTER` | Run vectors whose name holds this text | Every vector |
//!
//! Each session holds up to the driver's size, or 8192x4320 at most. A `resize` vector grows
//! at frames that keep references, so it needs that room. A vector past that counts as refused
//! by name.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use containr::TrackKind;
use containr::mkv::Demuxer;
use vidr::DecodeError;
use vidr::decode::{DecodeCodec, Decoder};
use vidr::vulkan::{VulkanDecoder, VulkanDevice};

use super::md5::hex_digest;
use super::runner::panic_text;
use super::vp9_vectors::{expected_hashes, matroska_of, refusal_expected, vector_name, vectors};
use crate::vulkan::support::{assert_no_validation_errors, device, planar_planes};

/// The largest session a sweep opens.
pub(super) const MAX_SESSION: (u32, u32) = (8192, 4320);

/// What one vector did.
enum Outcome {
    Pass { frames: usize },
    Refused { reason: String },
    Skipped { reason: String },
    Fail { reason: String },
}

/// Compares frames against the hashes in order.
struct HashCheck {
    expected: Vec<String>,
    next: usize,
}

impl HashCheck {
    fn drain(&mut self, decoder: &mut VulkanDecoder, name: &str) -> Result<(), String> {
        while let Some(frame) = decoder.receive() {
            let (y, u, v) = planar_planes(&frame, name);
            let hash = hex_digest(&[&y, &u, &v]);
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

/// Every packet of the video track of a Matroska file, with its pts.
pub(super) fn read_packets(path: &Path) -> Result<Vec<(Vec<u8>, i64)>, String> {
    let mut demuxer = Demuxer::open_path(path).map_err(|error| format!("demux: {error}"))?;
    let track = demuxer
        .tracks()
        .iter()
        .find(|track| track.kind == TrackKind::Video)
        .map(|track| track.id)
        .ok_or_else(|| "the file has no video track".to_owned())?;
    let packets = demuxer
        .track_packets(track)
        .map_err(|error| format!("demux: {error}"))?;
    let mut read = Vec::new();
    for packet in packets {
        let packet = packet.map_err(|error| format!("demux: {error}"))?;
        let mut data = Vec::new();
        demuxer
            .read_packet_into(&packet, &mut data)
            .map_err(|error| format!("demux at packet {}: {error}", packet.index))?;
        read.push((data, packet.pts));
    }
    Ok(read)
}

fn decode_vector(
    device: &Arc<VulkanDevice>,
    vector: &Path,
    work: &Path,
) -> Result<Outcome, String> {
    let name = vector_name(vector);
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

    let mut decoder =
        VulkanDecoder::with_max_coded_size(device, DecodeCodec::Vp9, &[], MAX_SESSION)
            .map_err(|error| format!("open: {error}"))?;
    let mut check = HashCheck { expected, next: 0 };
    let mut data = Vec::new();
    for packet in packets {
        let packet = packet.map_err(|error| format!("demux: {error}"))?;
        demuxer
            .read_packet_into(&packet, &mut data)
            .map_err(|error| format!("demux at packet {}: {error}", packet.index))?;
        match decoder.send(&data, packet.pts) {
            Ok(()) => {}
            Err(error @ (DecodeError::Vp9Layout { .. } | DecodeError::VulkanRefused { .. })) => {
                return Ok(Outcome::Refused {
                    reason: error.to_string(),
                });
            }
            Err(error) => return Err(format!("packet {}: {error}", packet.index)),
        }
        check.drain(&mut decoder, &name)?;
    }
    decoder.flush().map_err(|error| format!("flush: {error}"))?;
    check.drain(&mut decoder, &name)?;
    if check.next != check.expected.len() {
        return Err(format!(
            "the decoder gives {} frames, and libvpx has {}",
            check.next,
            check.expected.len()
        ));
    }
    Ok(Outcome::Pass { frames: check.next })
}

/// Runs one vector. A panic, or a pass where the name states a refused layout, is a failure.
fn run_vector(device: &Arc<VulkanDevice>, vector: &Path, work: &Path) -> Outcome {
    let name = vector_name(vector);
    let decoded = catch_unwind(AssertUnwindSafe(|| decode_vector(device, vector, work)));
    let outcome = match decoded {
        Ok(Ok(outcome)) => outcome,
        Ok(Err(reason)) => Outcome::Fail { reason },
        Err(panic) => Outcome::Fail {
            reason: format!("panic: {}", panic_text(&*panic)),
        },
    };
    match outcome {
        Outcome::Pass { .. } if refusal_expected(&name) => Outcome::Fail {
            reason: "decoded, and its name states a layout this backend refuses".to_owned(),
        },
        other => other,
    }
}

#[test]
fn libvpx_vectors_match_on_vulkan_video() {
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
    let Some((device, _guard)) = device() else {
        return;
    };
    let work = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("conformance_vulkan_vp9");

    println!("| Vector | Result | Detail |");
    println!("| --- | --- | --- |");
    let (mut passed, mut refused, mut skipped) = (0, 0, 0);
    let mut failed = Vec::new();
    for vector in &vectors {
        let name = vector_name(vector);
        match run_vector(&device, vector, &work) {
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
    assert_no_validation_errors(&device, "the VP9 sweep");
    assert!(
        failed.is_empty(),
        "{} vectors fail:\n{}",
        failed.len(),
        failed.join("\n")
    );
}
