//! The JCT-VC HEVC_v1 zips through `VulkanDecoder`, on one GPU device.
//!
//! `scripts/conformance/fetch_hevc.sh` fetches the zips. Pass `--no-capture` to see the table
//! on a run that passes. numr holds one device at index 0, and concurrent use loses it. So
//! vectors run one after another on the device of `VulkanDevice::open_shared`.
//!
//! | Variable | Meaning | Default |
//! | --- | --- | --- |
//! | `VIDR_HEVC_CONFORMANCE` | The directory of zips | Skip with a notice |
//! | `VIDR_HEVC_ONLY` | Run zips whose name holds this text | Every zip |
//!
//! Each output picture feeds one MD5 of the whole stream, as the reference decoder writes its
//! file. This decoder checks no hash SEI message, because the samples stay on the GPU.
//!
//! | Status | Meaning |
//! | --- | --- |
//! | `pass` | The whole-stream MD5 matches |
//! | `unchecked` | The zip has no whole-stream MD5, and the stream decodes |
//! | `refused` | The decoder refuses the stream by name |
//! | `fail` | The MD5 differs, or the decoder returns another error or panics |

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::Arc;

use vidr::decode::{DecodeCodec, Decoder};
use vidr::vulkan::{VulkanDecoder, VulkanDevice};
use vidr::{BitstreamError, DecodeError};

use super::hevc::{selected_zips, zip_name};
use super::hevc_hash::read_zip;
use super::md5::Md5;
use super::runner::panic_text;
use crate::vulkan::support::{assert_no_validation_errors, device, hevc_units, planar_planes};

/// What one vector did.
enum Outcome {
    Pass { frames: usize },
    Unchecked { frames: usize },
    Refused { reason: String },
    Fail { reason: String },
}

/// The refusals of a stream outside what this backend takes.
fn is_refusal(error: &DecodeError) -> bool {
    match error {
        DecodeError::VulkanRefused { .. }
        | DecodeError::UnsupportedLayout { .. }
        | DecodeError::HevcTool { .. } => true,
        DecodeError::Bitstream(BitstreamError::InvalidField { allowed, .. }) => {
            allowed.contains("because this decoder") || allowed.contains("because vidr does not")
        }
        _ => false,
    }
}

/// Takes frames from the decoder, feeds the MD5, and keeps none of them.
fn drain(decoder: &mut VulkanDecoder, md5: &mut Md5, frames: &mut usize, name: &str) {
    while let Some(frame) = decoder.receive() {
        let (y, u, v) = planar_planes(&frame, name);
        md5.update(&y);
        md5.update(&u);
        md5.update(&v);
        *frames += 1;
    }
}

fn decode_zip(device: &Arc<VulkanDevice>, zip: &Path) -> Result<Outcome, String> {
    let name = zip_name(zip);
    let (stream, published) = read_zip(zip)?;
    let mut decoder = match VulkanDecoder::new(device, DecodeCodec::Hevc, &[]) {
        Ok(decoder) => decoder,
        Err(error) if is_refusal(&error) => {
            return Ok(Outcome::Refused {
                reason: error.to_string(),
            });
        }
        Err(error) => return Err(format!("open: {error}")),
    };
    let mut md5 = Md5::new();
    let mut frames = 0;
    for (index, (packet, pts)) in hevc_units(&stream, &name).into_iter().enumerate() {
        match decoder.send(&packet, pts) {
            Ok(()) => {}
            Err(error) if is_refusal(&error) => {
                return Ok(Outcome::Refused {
                    reason: error.to_string(),
                });
            }
            Err(error) => return Err(format!("access unit {index}: {error}")),
        }
        drain(&mut decoder, &mut md5, &mut frames, &name);
    }
    decoder.flush().map_err(|error| format!("flush: {error}"))?;
    drain(&mut decoder, &mut md5, &mut frames, &name);
    if frames == 0 {
        return Err("the decoder gives no frame".to_owned());
    }
    let digest = md5.finish();
    match published {
        Some((_, expected)) if digest == expected => Ok(Outcome::Pass { frames }),
        Some((member, expected)) => Err(format!(
            "the {frames} frames have MD5 {digest}, and {member} states {expected}"
        )),
        None => Ok(Outcome::Unchecked { frames }),
    }
}

#[test]
fn hevc_v1_vectors_match_on_vulkan_video() {
    let Some(zips) = selected_zips() else {
        return;
    };
    let Some((device, _guard)) = device() else {
        return;
    };
    println!("| Vector | Result | Detail |");
    println!("| --- | --- | --- |");
    let (mut passed, mut unchecked, mut refused) = (0, 0, 0);
    let mut failed = Vec::new();
    for zip in &zips {
        let name = zip_name(zip);
        let decoded = catch_unwind(AssertUnwindSafe(|| decode_zip(&device, zip)));
        let outcome = match decoded {
            Ok(Ok(outcome)) => outcome,
            Ok(Err(reason)) => Outcome::Fail { reason },
            Err(panic) => Outcome::Fail {
                reason: format!("panic: {}", panic_text(&*panic)),
            },
        };
        match outcome {
            Outcome::Pass { frames } => {
                passed += 1;
                println!("| {name} | pass | {frames} frames |");
            }
            Outcome::Unchecked { frames } => {
                unchecked += 1;
                println!("| {name} | unchecked | {frames} frames, no MD5 in the zip |");
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
        "{} vectors: {passed} pass, {unchecked} unchecked, {} fail, {refused} refused by name",
        zips.len(),
        failed.len()
    );
    assert_no_validation_errors(&device, "the HEVC sweep");
    assert!(
        failed.is_empty(),
        "{} vectors fail:\n{}",
        failed.len(),
        failed.join("\n")
    );
}
