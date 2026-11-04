//! The report's frame path on a real GPU: a hardware run copies nothing through host
//! memory, and an explicit CPU side states its copy.

use std::path::Path;

use vidr_remaster::ResolutionMode;

use crate::harness::Case;

#[test]
fn nvdec_and_nvenc_copy_no_frame_through_the_host() {
    let Some(mut case) = Case::start("frame_path_hardware") else {
        return;
    };
    let source = case.clip("clip.mp4", 6);
    let output = case.default_output(&source);
    let run = case.run(&source, &[]);
    case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    let path = &run.report()["frame_path"];
    assert_eq!(path["runtime"], "cuda");
    assert_eq!(path["decoder"]["name"], "nvdec");
    assert_eq!(path["decoder"]["route"], "device");
    assert_eq!(path["encoder"]["name"], "nvenc");
    assert_eq!(path["encoder"]["route"], "device");
    assert_eq!(path["output_check_decoder"], "nvdec");
    assert_eq!(path["host_copied_frames"], 0);
    assert_eq!(path["host_copied_bytes"], 0);
    assert_eq!(path["scan_host_copied_frames"], 0);
    assert_eq!(run.report()["backend"], "boostr on numr CUDA, F16");
}

/// A sidecar of a hardware run has the PoC's keys and no others.
#[test]
fn a_hardware_run_keeps_the_poc_sidecar() {
    let Some(mut case) = Case::start("frame_path_sidecar") else {
        return;
    };
    let source = case.clip("clip.mp4", 6);
    let output = case.default_output(&source);
    let run = case.run(&source, &[]);
    case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    let sidecar = vidr_remaster::record_path(Path::new(&output));
    let text = std::fs::read_to_string(sidecar).expect("the sidecar");
    assert!(!text.contains("\"encoder\""), "{text}");
    assert!(!text.contains("\"decoder\""), "{text}");
}

/// NVDEC with a CPU encoder: one copy per frame, stated in the report. It skips with
/// a notice in a build with no `openh264`.
#[test]
fn a_cpu_encoder_on_cuda_states_its_download() {
    if !cfg!(feature = "openh264") {
        eprintln!("skipping: the openh264 feature is off.");
        return;
    }
    let Some(mut case) = Case::start("frame_path_cpu_encoder") else {
        return;
    };
    let source = case.clip("clip.mp4", 6);
    let output = case.default_output(&source);
    let run = case.run(&source, &["--codec", "h264", "--encoder", "cpu"]);
    case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    let path = &run.report()["frame_path"];
    assert_eq!(path["runtime"], "cuda");
    assert_eq!(path["decoder"]["route"], "device");
    assert_eq!(path["encoder"]["name"], "openh264");
    assert_eq!(path["encoder"]["route"], "host download");
    assert_eq!(path["host_copied_frames"], 6);
}
