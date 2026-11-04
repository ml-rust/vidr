//! Whole runs on the WebGPU runtime with Vulkan Video decode and a CPU encoder.
//!
//! Run with `cargo nextest run -p vidr-remaster --features
//! mp4,vulkan-video,openh264,cpu-av1,dav1d --test e2e_vulkan`. No Vulkan video device, a
//! missing ffmpeg, or a missing ffmpeg encoder skips with a printed notice.

#![cfg(all(
    feature = "vulkan-video",
    feature = "transform",
    feature = "mp4",
    feature = "openh264"
))]

#[path = "e2e_cpu/harness.rs"]
mod harness;

use vidr::decode::DecodeCodec;
use vidr::vulkan::{VulkanDecoder, VulkanDevice, VulkanOptions};
use vidr_remaster::ResolutionMode;

use harness::{Case, Run, assert_complete};

const FRAMES: u32 = 5;
/// libx264 High with B-frames, which OpenH264 refuses.
const HIGH: [&str; 6] = ["-c:v", "libx264", "-preset", "fast", "-bf", "2"];

fn start(name: &str) -> Option<Case> {
    if let Err(error) = vidr::vulkan::vulkan_video_available() {
        eprintln!("skipping: {error}");
        return None;
    }
    let case = Case::start(name)?;
    case.first_encoder(&["libx264"])?;
    Some(case)
}

fn assert_vulkan_path(run: &Run, frames: u64) {
    let path = run.frame_path();
    assert_eq!(path["runtime"], "wgpu");
    assert_eq!(path["decoder"]["name"], "vulkan");
    assert_eq!(path["decoder"]["route"], "device");
    assert_eq!(path["encoder"]["name"], "openh264");
    assert_eq!(path["encoder"]["route"], "host download");
    assert_eq!(path["output_check_decoder"], "vulkan");
    assert_eq!(
        path["host_copied_frames"], frames,
        "the encoder's copy only"
    );
}

fn from_upscaled(case: &mut Case, decoder: &str) -> Run {
    from_upscaled_with(case, decoder, &HIGH)
}

fn from_upscaled_with(case: &mut Case, decoder: &str, codec: &[&str]) -> Run {
    let source = case.source("clip.mp4", FRAMES, codec);
    let enhanced = case.enhanced("up.mp4", FRAMES, codec);
    let output = case.path("out.mp4");
    let run = case.run_on(
        "wgpu",
        &source,
        &[
            "--codec",
            "h264",
            "--decoder",
            decoder,
            "--encoder",
            "cpu",
            "--from-upscaled",
            enhanced.to_str().expect("a UTF-8 path"),
            "-o",
            output.to_str().expect("a UTF-8 path"),
        ],
    );
    assert_complete(&run, &source, &output, ResolutionMode::Auto);
    run
}

#[test]
fn an_explicit_vulkan_decoder_keeps_frames_and_timing() {
    let Some(mut case) = start("vulkan_explicit") else {
        return;
    };
    let run = from_upscaled(&mut case, "vulkan");
    assert_vulkan_path(&run, u64::from(FRAMES));
}

#[test]
fn auto_on_webgpu_picks_vulkan_for_h264() {
    let Some(mut case) = start("vulkan_auto") else {
        return;
    };
    let run = from_upscaled(&mut case, "auto");
    assert_vulkan_path(&run, u64::from(FRAMES));
}

/// Whether this device lists `codec`. Prints a notice when it does not.
fn device_decodes(codec: DecodeCodec) -> bool {
    let opened = VulkanDevice::open_shared(VulkanOptions::default())
        .and_then(|device| VulkanDecoder::new(&device, codec, &[]));
    if let Err(error) = &opened {
        eprintln!("skipping: {error}");
    }
    opened.is_ok()
}

/// `--runtime wgpu --decoder vulkan --encoder cpu` on a source from `encoder`: the whole path,
/// with frames and timing kept.
fn run_source_on_vulkan(label: &str, codec: DecodeCodec, encoder: &str, args: &[&str]) {
    let Some(mut case) = start(label) else {
        return;
    };
    if case.first_encoder(&[encoder]).is_none() || !device_decodes(codec) {
        return;
    }
    let mut flags = vec!["-c:v", encoder];
    flags.extend_from_slice(args);
    let run = from_upscaled_with(&mut case, "vulkan", &flags);
    assert_vulkan_path(&run, u64::from(FRAMES));
}

#[test]
fn an_explicit_vulkan_decoder_runs_an_hevc_source() {
    run_source_on_vulkan(
        "vulkan_hevc",
        DecodeCodec::Hevc,
        "libx265",
        &["-tag:v", "hvc1"],
    );
}

#[test]
fn an_explicit_vulkan_decoder_runs_a_vp9_source() {
    run_source_on_vulkan(
        "vulkan_vp9",
        DecodeCodec::Vp9,
        "libvpx-vp9",
        &["-b:v", "0", "-crf", "30"],
    );
}

/// AV1 takes its session from the sequence header, and rav1r writes nothing here. libaom writes
/// the source, and the output check reads the H.264 output on Vulkan Video.
#[test]
fn an_explicit_vulkan_decoder_runs_an_av1_source() {
    run_source_on_vulkan(
        "vulkan_av1",
        DecodeCodec::Av1,
        "libaom-av1",
        &["-cpu-used", "8", "-row-mt", "0"],
    );
}
