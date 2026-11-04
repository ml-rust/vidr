//! libaom's AV1 test vectors through `VulkanDecoder`, frame by frame, on one GPU device.
//!
//! `scripts/conformance/fetch_av1.sh` fetches the vectors and `.md5` files. Pass `--no-capture`
//! to see the table on a run that passes. numr holds one device at index 0, and concurrent use
//! loses it, so vectors run one after another.
//!
//! | Variable | Meaning | Default |
//! | --- | --- | --- |
//! | `VIDR_AV1_VECTORS` | The directory of vectors | Skip with a notice |
//! | `VIDR_AV1_FILTER` | Run vectors whose name holds this text | Every vector |
//!
//! A vector past what the driver lists counts as refused by name: a size under its smallest coded
//! extent, for one. A `.mkv` vector needs the `mkv` feature.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use vidr::DecodeError;
use vidr::decode::{DecodeCodec, Decoder};
use vidr::vulkan::{VulkanDecoder, VulkanDevice};

use super::av1_vectors::{HashCheck, Outcome, read_packets, refusal_expected};
use super::md5::hex_digest;
use super::runner::panic_text;
use super::vector_files::{expected_hashes, vector_name, vectors};
use crate::vulkan::support::{assert_no_validation_errors, codec_present, device, planar_planes};

/// Compares each frame the decoder holds.
fn drain(decoder: &mut VulkanDecoder, check: &mut HashCheck, name: &str) -> Result<(), String> {
    while let Some(frame) = decoder.receive() {
        let (y, u, v) = planar_planes(&frame, name);
        check.compare(&hex_digest(&[&y, &u, &v]), frame.display())?;
    }
    Ok(())
}

fn decode_vector(device: &Arc<VulkanDevice>, vector: &Path) -> Result<Outcome, String> {
    let name = vector_name(vector);
    let expected = expected_hashes(vector)?;
    let packets = read_packets(vector)?;
    let mut decoder = VulkanDecoder::new(device, DecodeCodec::Av1, &[])
        .map_err(|error| format!("open: {error}"))?;
    let mut check = HashCheck::new(expected);
    for (index, (packet, pts)) in packets.iter().enumerate() {
        match decoder.send(packet, *pts) {
            Ok(()) => {}
            Err(
                error @ (DecodeError::UnsupportedLayout { .. } | DecodeError::VulkanRefused { .. }),
            ) => {
                return Ok(Outcome::Refused {
                    reason: error.to_string(),
                });
            }
            Err(error) => return Err(format!("packet {index}: {error}")),
        }
        drain(&mut decoder, &mut check, &name)?;
    }
    decoder.flush().map_err(|error| format!("flush: {error}"))?;
    drain(&mut decoder, &mut check, &name)?;
    check.finish()
}

/// Runs one vector. A panic, or a pass where the name states a refused layout, is a failure.
fn run_vector(device: &Arc<VulkanDevice>, vector: &Path) -> Outcome {
    let name = vector_name(vector);
    let decoded = catch_unwind(AssertUnwindSafe(|| decode_vector(device, vector)));
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
fn libaom_vectors_match_on_vulkan_video() {
    let Some(directory) = std::env::var_os("VIDR_AV1_VECTORS") else {
        eprintln!(
            "skipping: VIDR_AV1_VECTORS is unset. Run `scripts/conformance/fetch_av1.sh DIR`, \
             then set it to DIR."
        );
        return;
    };
    let directory = PathBuf::from(directory);
    let filter = std::env::var("VIDR_AV1_FILTER").ok();
    let vectors = vectors(&directory, filter.as_deref());
    assert!(
        !vectors.is_empty(),
        "{} holds no vector with a `.md5` file for filter {filter:?}. Run \
         `scripts/conformance/fetch_av1.sh {}`.",
        directory.display(),
        directory.display()
    );
    let Some((device, _guard)) = device() else {
        return;
    };
    if !codec_present(&device, DecodeCodec::Av1) {
        return;
    }

    println!("| Vector | Result | Detail |");
    println!("| --- | --- | --- |");
    let (mut passed, mut refused) = (0, 0);
    let mut failed = Vec::new();
    for vector in &vectors {
        let name = vector_name(vector);
        match run_vector(&device, vector) {
            Outcome::Pass { frames } => {
                passed += 1;
                println!("| {name} | pass | {frames} frames |");
            }
            Outcome::Refused { reason } => {
                refused += 1;
                println!("| {name} | refused | {reason} |");
            }
            Outcome::Fail { reason } => {
                println!("| {name} | FAIL | {reason} |");
                failed.push(format!("{name}: {reason}"));
            }
        }
    }
    println!(
        "{} vectors: {passed} pass, {} fail, {refused} refused by name",
        vectors.len(),
        failed.len()
    );
    assert_no_validation_errors(&device, "the AV1 sweep");
    assert!(
        failed.is_empty(),
        "{} vectors fail:\n{}",
        failed.len(),
        failed.join("\n")
    );
}
