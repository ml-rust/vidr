//! libaom's AV1 test vectors through `Dav1dDecoder`, frame by frame.
//!
//! `scripts/conformance/fetch_av1.sh` fetches the vectors and `.md5` files. Set
//! `VIDR_AV1_VECTORS` to that directory. Pass `--no-capture` to see the table on a run that
//! passes.
//!
//! | Variable | Meaning | Default |
//! | --- | --- | --- |
//! | `VIDR_AV1_VECTORS` | The directory of vectors | Skip with a notice |
//! | `VIDR_AV1_FILTER` | Run vectors whose name holds this text | Every vector |
//! | `VIDR_CONFORMANCE_JOBS` | Vectors decoded at once | Half the logical CPUs |
//!
//! Each line of `NAME.md5` holds one frame's hash: the visible Y plane, then U, then V, row by
//! row, with no padding. An 8-bit sample is one byte, and a 10-bit sample is two little-endian
//! bytes holding its value. The tolerance is zero. A `.mkv` vector needs the `mkv` feature.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use vidr::DecodeError;
use vidr::decode::Decoder;
use vidr::decode::dav1d::Dav1dDecoder;

use super::av1_vectors::{HashCheck, Outcome, read_packets, refusal_expected};
use super::cpu_frame_hash::frame_hash;
use super::runner::{panic_text, run_each};
use super::vector_files::{expected_hashes, vector_name, vectors};

/// Compares each frame the decoder holds.
fn drain(decoder: &mut Dav1dDecoder<CpuRuntime>, check: &mut HashCheck) -> Result<(), String> {
    while let Some(frame) = decoder.receive() {
        check.compare(&frame_hash(&frame)?, frame.display())?;
    }
    Ok(())
}

fn decode_vector(vector: &Path) -> Result<Outcome, String> {
    let expected = expected_hashes(vector)?;
    let packets = read_packets(vector)?;
    let mut decoder = Dav1dDecoder::<CpuRuntime>::new(&CpuDevice::new(), &[])
        .map_err(|error| format!("open: {error}"))?;
    let mut check = HashCheck::new(expected);
    for (index, (packet, pts)) in packets.iter().enumerate() {
        match decoder.send(packet, *pts) {
            Ok(()) => {}
            Err(error @ DecodeError::UnsupportedLayout { .. }) => {
                return Ok(Outcome::Refused {
                    reason: error.to_string(),
                });
            }
            Err(error) => return Err(format!("packet {index}: {error}")),
        }
        drain(&mut decoder, &mut check)?;
    }
    decoder.flush().map_err(|error| format!("flush: {error}"))?;
    drain(&mut decoder, &mut check)?;
    check.finish()
}

/// Runs one vector. A panic or an unexpected refusal is a failure.
fn run_vector(vector: &Path) -> Outcome {
    let name = vector_name(vector);
    let decoded = catch_unwind(AssertUnwindSafe(|| decode_vector(vector)));
    let outcome = match decoded {
        Ok(Ok(outcome)) => outcome,
        Ok(Err(reason)) => Outcome::Fail { reason },
        Err(panic) => Outcome::Fail {
            reason: format!("panic: {}", panic_text(&*panic)),
        },
    };
    match outcome {
        Outcome::Refused { reason } if !refusal_expected(&name) => Outcome::Fail {
            reason: format!("refused, and its name states a layout the decoder takes: {reason}"),
        },
        Outcome::Pass { .. } if refusal_expected(&name) => Outcome::Fail {
            reason: "decoded, and its name states a layout the decoder refuses".to_owned(),
        },
        other => other,
    }
}

#[test]
fn libaom_vectors_match_on_dav1d() {
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
    let outcomes = run_each(&vectors, |vector| run_vector(vector));

    println!("| Vector | Result | Detail |");
    println!("| --- | --- | --- |");
    let (mut passed, mut refused) = (0, 0);
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
    assert!(
        failed.is_empty(),
        "{} vectors fail:\n{}",
        failed.len(),
        failed.join("\n")
    );
}
