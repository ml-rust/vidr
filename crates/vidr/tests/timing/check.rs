use vidr::{
    DURATION_TOLERANCE, Deviation, Dimensions, DurationHints, FrameTiming, Geometry, Rational,
    Rotation, SampleAspectRatio, Side, StreamFacts, check_output,
};

use super::r;

fn stream(
    width: u32,
    height: u32,
    rotation: Rotation,
    time_base: Rational,
    timing: &[(i64, i64)],
) -> StreamFacts {
    StreamFacts::new(
        Geometry::new(
            Dimensions::new(width, height).unwrap(),
            rotation,
            SampleAspectRatio::SQUARE,
        ),
        time_base,
        timing
            .iter()
            .map(|&(pts, duration)| FrameTiming::new(pts, None, duration))
            .collect(),
        DurationHints::default(),
    )
}

fn size(width: u32, height: u32) -> Dimensions {
    Dimensions::new(width, height).unwrap()
}

/// 640x480 upright, four frames at 30 fps.
fn source() -> StreamFacts {
    stream(
        640,
        480,
        Rotation::Upright,
        r(1, 30),
        &[(0, 1), (1, 1), (2, 1), (3, 1)],
    )
}

#[test]
fn faithful_output_passes() {
    let output = stream(
        1280,
        960,
        Rotation::Upright,
        r(1, 30),
        &[(10, 1), (11, 1), (12, 1), (13, 1)],
    );
    assert_eq!(check_output(&source(), &output, size(1280, 960)), Ok(()));
}

#[test]
fn geometry_count_and_pts_mismatches_are_reported_together() {
    // Three frames out of order, a 90-degree rotation, and a 16:9 coded frame.
    let output = stream(
        1280,
        720,
        Rotation::Clockwise90,
        r(1, 30),
        &[(0, 1), (2, 1), (1, 1)],
    );
    let deviations = check_output(&source(), &output, size(1280, 960))
        .unwrap_err()
        .into_deviations();
    let expected = [
        Deviation::FrameCount {
            source: 4,
            output: 3,
        },
        Deviation::Dimensions {
            target: size(1280, 960),
            output: size(720, 1280),
        },
        Deviation::Rotation {
            source: Rotation::Upright,
            output: Rotation::Clockwise90,
        },
        Deviation::AspectRatio {
            source: r(4, 3),
            output: r(9, 16),
            // max(4/3, 9/16) * 2 / 480.
            tolerance: r(1, 180),
        },
        Deviation::PtsOrder {
            side: Side::Output,
            index: 2,
            previous: 2,
            pts: 1,
            violations: 1,
        },
        Deviation::Pts {
            index: 1,
            expected: 1,
            output: 2,
            mismatches: 2,
            time_base: r(1, 30),
        },
    ];
    assert_eq!(deviations, expected);
}

#[test]
fn pts_and_exact_duration_mismatches_are_reported_together() {
    let output = stream(
        1280,
        960,
        Rotation::Upright,
        r(1, 30),
        &[(0, 1), (1, 1), (2, 1), (4, 2)],
    );
    let deviations = check_output(&source(), &output, size(1280, 960))
        .unwrap_err()
        .into_deviations();
    assert_eq!(
        deviations,
        [
            Deviation::Pts {
                index: 3,
                expected: 3,
                output: 4,
                mismatches: 1,
                time_base: r(1, 30),
            },
            Deviation::Duration {
                expected: r(4, 30),
                output: r(6, 30),
                tolerance: None,
            },
        ]
    );
}

#[test]
fn one_time_base_allows_no_duration_slack() {
    // Same clock, last frame one tick longer: inside 2 ms is still a mismatch.
    let base = r(1, 90000);
    let source = stream(64, 64, Rotation::Upright, base, &[(0, 3003), (3003, 3003)]);
    let output = stream(64, 64, Rotation::Upright, base, &[(0, 3003), (3003, 3004)]);
    let deviations = check_output(&source, &output, size(64, 64))
        .unwrap_err()
        .into_deviations();
    assert_eq!(
        deviations,
        [Deviation::Duration {
            expected: r(6006, 90000),
            output: r(6007, 90000),
            tolerance: None,
        }]
    );
}

#[test]
fn millisecond_container_passes_within_tolerance() {
    // NTSC source in 1/30000, written to a 1/1000 clock: pts 33.37 -> 33, 66.73 -> 67.
    let source = stream(
        64,
        64,
        Rotation::Upright,
        r(1, 30000),
        &[(0, 1001), (1001, 1001), (2002, 1001)],
    );
    let output = stream(
        64,
        64,
        Rotation::Upright,
        r(1, 1000),
        &[(0, 33), (33, 34), (67, 33)],
    );
    assert_eq!(check_output(&source, &output, size(64, 64)), Ok(()));
}

#[test]
fn millisecond_container_beyond_tolerance_is_reported() {
    let source = stream(
        64,
        64,
        Rotation::Upright,
        r(1, 30000),
        &[(0, 1001), (1001, 1001), (2002, 1001)],
    );
    let output = stream(
        64,
        64,
        Rotation::Upright,
        r(1, 1000),
        &[(0, 33), (33, 34), (67, 36)],
    );
    let deviations = check_output(&source, &output, size(64, 64))
        .unwrap_err()
        .into_deviations();
    assert_eq!(
        deviations,
        [Deviation::Duration {
            expected: r(3003, 30000),
            output: r(103, 1000),
            tolerance: Some(DURATION_TOLERANCE),
        }]
    );
}

#[test]
fn error_message_lists_every_mismatch() {
    let output = stream(
        640,
        480,
        Rotation::Clockwise180,
        r(1, 30),
        &[(0, 1), (1, 1), (2, 1)],
    );
    let message = check_output(&source(), &output, size(640, 480))
        .unwrap_err()
        .to_string();
    assert!(
        message.starts_with("output check found 3 mismatch(es):"),
        "{message}"
    );
    assert!(
        message.contains("output has 3 frames and source has 4"),
        "{message}"
    );
    assert!(message.contains("Copy the source rotation"), "{message}");
    assert!(message.contains("output duration 1/10 s"), "{message}");
}
