//! The two RRDB models, one short clip each. The other tests use `fast`.

use vidr_remaster::{ModelId, ResolutionMode};

use crate::harness::{Case, PICTURE};

/// Four frames at a size RRDB runs in little GPU memory.
fn shortest_clip(case: &Case) -> std::path::PathBuf {
    let picture = PICTURE.replace("320x240", "128x96");
    case.generate(
        "clip.mp4",
        &[
            "-f",
            "lavfi",
            "-i",
            &picture,
            "-frames:v",
            "4",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
        ],
    )
}

#[test]
fn realesrgan_runs_and_states_no_notice() {
    let Some(mut case) = Case::with_model("realesrgan", ModelId::RealEsrgan) else {
        return;
    };
    let source = shortest_clip(&case);
    let run = case.run(&source, &[]);
    let output = case.default_output(&source);
    let probe = case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    assert_eq!(probe.size(), (640, 480));
    assert_eq!(run.report()["model"], "realesrgan");
    assert!(run.report()["licence_notice"].is_null());
    assert_eq!(run.report()["processing"]["ai_executed"], true);
}

#[test]
fn purevision_runs_and_the_report_states_its_licence() {
    let Some(mut case) = Case::with_model("purevision", ModelId::PureVision) else {
        return;
    };
    let source = shortest_clip(&case);
    let run = case.run(&source, &[]);
    let output = case.default_output(&source);
    let probe = case.assert_complete(&run, &source, &output, ResolutionMode::Auto);
    assert_eq!(probe.size(), (640, 480));
    assert_eq!(run.report()["model"], "purevision");
    let notice = run.report()["licence_notice"]
        .as_str()
        .expect("the licence notice");
    assert!(notice.contains("CC BY-NC-SA 4.0"), "{notice}");
    assert!(notice.contains("forbids commercial use"), "{notice}");
}
