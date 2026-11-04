//! Container colour tags and the merge against ffprobe.
//!
//! ffprobe's stream fields prefer the container. Its frame fields prefer the
//! SPS. `merge_container` follows the rule its documentation states, so each
//! test names the ffprobe field that agrees with it.

use numr::ops::{ColorMatrix, ColorRange};
use vidr::ColorError;
use vidr::color::{check_sdr, color_plan, merge_container};
use vidr::decode::ColorFacts;
use vidr::media::Input;

use super::fixtures::{Fixture, open, video};

const CONTAINERS: [&str; 2] = ["mp4", "mkv"];

/// The colour of the H.264 stream outside its container.
fn bitstream(fixture: &Fixture) -> ColorFacts {
    ColorFacts {
        matrix_coefficients: fixture.code("raw_matrix"),
        transfer_characteristics: fixture.code("raw_transfer"),
        colour_primaries: fixture.code("raw_primaries"),
        full_range: fixture.meta("raw_range") == "pc",
        chroma_location: None,
    }
}

fn merged(fixture: &Fixture, input: &Input) -> ColorFacts {
    let track = video(fixture, input);
    merge_container(bitstream(fixture), input.colour(&track))
}

/// Checks the three codes against the meta lines that start with `prefix`.
fn assert_codes(facts: &ColorFacts, fixture: &Fixture, prefix: &str) {
    let name = &fixture.name;
    assert_eq!(
        facts.matrix_coefficients,
        fixture.code(&format!("{prefix}matrix")),
        "{name}: matrix against ffprobe `{prefix}`"
    );
    assert_eq!(
        facts.transfer_characteristics,
        fixture.code(&format!("{prefix}transfer")),
        "{name}: transfer against ffprobe `{prefix}`"
    );
    assert_eq!(
        facts.colour_primaries,
        fixture.code(&format!("{prefix}primaries")),
        "{name}: primaries against ffprobe `{prefix}`"
    );
}

/// Each `(fixture, facts)` of one colour recipe, in the containers it has.
fn each(stem: &str, check: impl Fn(&Fixture, &ColorFacts)) {
    for ext in CONTAINERS {
        if let Some((fixture, input)) = open(&format!("{stem}.{ext}")) {
            check(&fixture, &merged(&fixture, &input));
        }
    }
}

#[test]
fn agreeing_tags_match_every_ffprobe_field() {
    for (stem, range, full) in [("colour_limited", "tv", false), ("colour_full", "pc", true)] {
        each(stem, |fixture, facts| {
            assert_codes(facts, fixture, "");
            assert_codes(facts, fixture, "frame_");
            assert_eq!(facts.matrix_coefficients, 1);
            assert_eq!(facts.full_range, full);
            assert_eq!(fixture.meta("range"), range);
            assert_eq!(fixture.meta("frame_range"), range);
        });
    }
}

#[test]
fn the_container_fills_a_silent_sps() {
    each("colour_container_only", |fixture, facts| {
        assert_eq!(fixture.code("raw_matrix"), 2);
        assert_eq!(fixture.meta("raw_range"), "unknown");
        assert_codes(facts, fixture, "");
        assert_codes(facts, fixture, "frame_");
        assert!(facts.full_range);
        assert_eq!(fixture.meta("range"), "pc");
    });
}

#[test]
fn the_container_fills_an_sps_that_states_unspecified() {
    // ffprobe's frame fields keep the SPS's unspecified here. Its stream
    // fields take the container, as the merge does.
    each("colour_sps_unspecified", |fixture, facts| {
        assert_eq!(fixture.code("frame_matrix"), 2);
        assert_codes(facts, fixture, "");
    });
}

#[test]
fn a_disagreement_follows_the_sps_as_ffprobes_frame_fields_do() {
    // This is the stated difference: ffprobe's stream fields, which the PoC
    // reads, take the container here.
    each("colour_disagree", |fixture, facts| {
        assert_codes(facts, fixture, "frame_");
        assert_eq!(facts.matrix_coefficients, 5);
        assert_eq!(fixture.code("matrix"), 1);
    });
}

#[test]
fn a_full_range_sps_wins_as_ffprobes_frame_fields_do() {
    each("colour_sps_full", |fixture, facts| {
        assert!(facts.full_range);
        assert_eq!(fixture.meta("frame_range"), "pc");
        assert_eq!(fixture.meta("range"), "tv");
    });
}

#[test]
fn a_limited_range_sps_loses_to_a_full_container_as_ffprobes_stream_fields_do() {
    // This is the stated limit: `ColorFacts::full_range` cannot tell a
    // signalled limited range from no signal, so ffprobe's frame field differs.
    each("colour_sps_limited", |fixture, facts| {
        assert!(facts.full_range);
        assert_eq!(fixture.meta("range"), "pc");
        assert_eq!(fixture.meta("frame_range"), "tv");
    });
}

#[test]
fn the_tags_read_from_the_container_match_ffprobe() {
    for ext in CONTAINERS {
        let Some((fixture, input)) = open(&format!("colour_full.{ext}")) else {
            continue;
        };
        let colour = input.colour(&video(&fixture, &input)).unwrap();
        assert_eq!(colour.matrix, Some(fixture.code("matrix")));
        assert_eq!(colour.transfer, Some(fixture.code("transfer")));
        assert_eq!(colour.primaries, Some(fixture.code("primaries")));
        assert_eq!(colour.full_range, Some(true));
    }
}

#[test]
fn quicktime_nclc_carries_no_range_and_the_sps_sets_it() {
    let Some((fixture, input)) = open("colour_nclc.mov") else {
        return;
    };
    let colour = input.colour(&video(&fixture, &input)).unwrap();
    assert_eq!(colour.full_range, None);
    let facts = merged(&fixture, &input);
    assert_codes(&facts, &fixture, "");
    assert!(facts.full_range);
    assert_eq!(fixture.meta("range"), "pc");
}

#[test]
fn hdr_fixtures_are_refused_with_their_code() {
    each("colour_pq", |fixture, facts| {
        assert_eq!(fixture.code("transfer"), 16);
        let expected = ColorError::Hdr {
            field: "TransferCharacteristics",
            code: 16,
        };
        assert_eq!(check_sdr(facts), Err(expected));
        assert_eq!(color_plan(facts).map(|_| ()), Err(expected));
    });
    each("colour_hlg", |fixture, facts| {
        assert_eq!(fixture.code("transfer"), 18);
        assert_eq!(
            check_sdr(facts),
            Err(ColorError::Hdr {
                field: "TransferCharacteristics",
                code: 18,
            })
        );
    });
}

#[test]
fn an_untagged_fixture_plans_bt601_limited_and_stays_untagged() {
    each("plain", |fixture, facts| {
        assert_eq!(fixture.code("matrix"), 2);
        let plan = color_plan(facts).unwrap();
        assert_eq!(plan.decode.matrix, ColorMatrix::Bt601);
        assert_eq!(plan.encode.matrix, ColorMatrix::Bt601);
        assert_eq!(plan.decode.range, ColorRange::Limited);
        assert_eq!(plan.tags.matrix_coefficients, 2);
        assert_eq!(plan.tags.transfer_characteristics, 2);
        assert_eq!(plan.tags.colour_primaries, 2);
    });
}
