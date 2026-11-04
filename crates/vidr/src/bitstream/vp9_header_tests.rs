//! Hand-built VP9 uncompressed headers.

use super::*;
use crate::bitstream::test_bits::BitWriter;

/// The bytes of compressed header data after each built header.
const COMPRESSED: usize = 3;

/// The fields from `refresh_frame_context` on. The defaults make a valid tail
/// for a frame of 64 luma samples across.
#[derive(Clone)]
struct Tail {
    error_resilient: bool,
    refresh_frame_context: bool,
    frame_parallel: bool,
    frame_context_idx: u64,
    level: u64,
    sharpness: u64,
    /// `None` clears `loop_filter_delta_enabled`.
    deltas: Option<Option<Deltas>>,
    q: (u64, [Option<i64>; 3]),
    segmentation: Option<fn(&mut BitWriter)>,
    sb64_cols: u32,
    tile_cols_log2: u32,
    tile_rows_log2: u32,
    header_size: u64,
}

/// `loop_filter_ref_deltas` and `loop_filter_mode_deltas` updates.
type Deltas = ([Option<i64>; 4], [Option<i64>; 2]);

impl Default for Tail {
    fn default() -> Self {
        Self {
            error_resilient: false,
            refresh_frame_context: true,
            frame_parallel: false,
            frame_context_idx: 2,
            level: 20,
            sharpness: 3,
            deltas: Some(None),
            q: (60, [None, None, None]),
            segmentation: None,
            sb64_cols: 1,
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            header_size: COMPRESSED as u64,
        }
    }
}

fn signed(w: &mut BitWriter, count: u32, value: i64) {
    w.u(count, value.unsigned_abs()).flag(value < 0);
}

fn prefix(w: &mut BitWriter, profile: u64) {
    w.u(2, 2).u(1, profile & 1).u(1, profile >> 1);
    if profile == 3 {
        w.flag(false);
    }
}

fn sync(w: &mut BitWriter) {
    w.u(8, 0x49).u(8, 0x83).u(8, 0x42);
}

/// Writes the tail, its trailing bits, and the compressed header bytes.
fn finish(w: &mut BitWriter, tail: &Tail) -> Vec<u8> {
    if !tail.error_resilient {
        w.flag(tail.refresh_frame_context).flag(tail.frame_parallel);
    }
    w.u(2, tail.frame_context_idx)
        .u(6, tail.level)
        .u(3, tail.sharpness);
    w.flag(tail.deltas.is_some());
    if let Some(update) = &tail.deltas {
        w.flag(update.is_some());
        if let Some((refs, modes)) = update {
            for value in refs.iter().chain(modes) {
                w.flag(value.is_some());
                if let Some(value) = value {
                    signed(w, 6, *value);
                }
            }
        }
    }
    w.u(8, tail.q.0);
    for delta in tail.q.1 {
        w.flag(delta.is_some());
        if let Some(delta) = delta {
            signed(w, 4, delta);
        }
    }
    w.flag(tail.segmentation.is_some());
    if let Some(segmentation) = tail.segmentation {
        segmentation(w);
    }
    let mut minimum = 0;
    while (64 << minimum) < tail.sb64_cols {
        minimum += 1;
    }
    let mut maximum = 1;
    while (tail.sb64_cols >> maximum) >= 4 {
        maximum += 1;
    }
    maximum -= 1;
    for _ in minimum..tail.tile_cols_log2 {
        w.flag(true);
    }
    if tail.tile_cols_log2 < maximum {
        w.flag(false);
    }
    w.flag(tail.tile_rows_log2 > 0);
    if tail.tile_rows_log2 > 0 {
        w.flag(tail.tile_rows_log2 == 2);
    }
    w.u(16, tail.header_size);
    w.align();
    let mut bytes = w.bytes();
    bytes.extend_from_slice(&[0x80; COMPRESSED]);
    bytes
}

/// A profile 0 key frame of `width` by 48, BT.709, limited range.
fn key_with(width: u64, tail: &Tail) -> Vec<u8> {
    let mut w = BitWriter::new();
    prefix(&mut w, 0);
    w.flag(false)
        .flag(false)
        .flag(true)
        .flag(tail.error_resilient);
    sync(&mut w);
    w.u(3, 2).flag(false);
    w.u(16, width - 1).u(16, 47).flag(false);
    finish(&mut w, tail)
}

fn key(tail: &Tail) -> Vec<u8> {
    key_with(64, tail)
}

/// A shown profile 0 inter frame that takes its size from reference 0, with a
/// switchable filter.
fn inter(tail: &Tail) -> Vec<u8> {
    let mut w = BitWriter::new();
    prefix(&mut w, 0);
    w.flag(false)
        .flag(true)
        .flag(true)
        .flag(tail.error_resilient);
    if !tail.error_resilient {
        w.u(2, 1);
    }
    w.u(8, 1);
    for reference in 0..3 {
        w.u(3, reference).flag(reference == 1);
    }
    w.flag(true).flag(false);
    w.flag(true).flag(true);
    finish(&mut w, tail)
}

fn run(parser: &mut Vp9HeaderParser, refs: &mut RefFrameTable, frame: &[u8]) -> UncompressedHeader {
    let header = parser
        .parse(frame, refs)
        .unwrap_or_else(|error| panic!("{error}"));
    refs.update(&header);
    header
}

/// A parser and table that have seen one default key frame.
fn after_key() -> (Vp9HeaderParser, RefFrameTable) {
    let mut parser = Vp9HeaderParser::new();
    let mut refs = RefFrameTable::default();
    run(&mut parser, &mut refs, &key(&Tail::default()));
    (parser, refs)
}

fn parse_alone(frame: &[u8]) -> Result<UncompressedHeader, BitstreamError> {
    Vp9HeaderParser::new().parse(frame, &RefFrameTable::default())
}

#[test]
fn a_key_frame_reads_every_field() {
    let mut w = BitWriter::new();
    prefix(&mut w, 0);
    w.flag(false).flag(false).flag(true).flag(false);
    sync(&mut w);
    w.u(3, 2).flag(true);
    w.u(16, 1919).u(16, 1079);
    w.flag(true).u(16, 1279).u(16, 719);
    let frame = finish(
        &mut w,
        &Tail {
            sb64_cols: 30,
            tile_cols_log2: 1,
            tile_rows_log2: 2,
            ..Tail::default()
        },
    );
    let header = parse_alone(&frame).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(header.profile, 0);
    assert_eq!(header.frame_type, KEY_FRAME);
    assert!(header.show_frame && header.frame_is_intra && !header.intra_only);
    assert_eq!(header.refresh_frame_flags, 0xff);
    assert_eq!(
        header.color_config,
        ColorConfig {
            bit_depth: 8,
            color_space: 2,
            color_range: true,
            subsampling_x: true,
            subsampling_y: true,
        }
    );
    assert_eq!((header.frame_width, header.frame_height), (1920, 1080));
    assert_eq!((header.render_width, header.render_height), (1280, 720));
    assert_eq!((header.mi_cols, header.mi_rows), (240, 135));
    assert_eq!((header.sb64_cols, header.sb64_rows), (30, 17));
    assert_eq!(header.tile_cols_log2, 1);
    assert_eq!(header.tile_rows_log2, 2);
    assert_eq!(header.header_size_in_bytes, COMPRESSED as u16);
    assert_eq!(header.uncompressed_header_size, frame.len() - COMPRESSED);
    assert_eq!(header.last_frame_type, None);
    assert!(header.size_changed && !header.use_prev_frame_mvs);
    assert!(header.past_independence);
}

#[test]
fn colour_syntax_follows_the_profile() {
    type Color = (u8, u8, bool, bool, bool);
    type Case = (u64, fn(&mut BitWriter), Color);
    let cases: [Case; 7] = [
        (
            1,
            |w| {
                w.u(3, 2).flag(true).flag(true).flag(false).flag(false);
            },
            (8, 2, true, true, false),
        ),
        (
            1,
            |w| {
                w.u(3, 2).flag(false).flag(false).flag(true).flag(false);
            },
            (8, 2, false, false, true),
        ),
        (
            1,
            |w| {
                w.u(3, 2).flag(false).flag(false).flag(false).flag(false);
            },
            (8, 2, false, false, false),
        ),
        (
            1,
            |w| {
                w.u(3, 7).flag(false);
            },
            (8, 7, true, false, false),
        ),
        (
            2,
            |w| {
                w.flag(false).u(3, 5).flag(true);
            },
            (10, 5, true, true, true),
        ),
        (
            2,
            |w| {
                w.flag(true).u(3, 5).flag(false);
            },
            (12, 5, false, true, true),
        ),
        (
            3,
            |w| {
                w.flag(true).u(3, 7).flag(false);
            },
            (12, 7, true, false, false),
        ),
    ];
    for (profile, color, expected) in cases {
        let mut w = BitWriter::new();
        prefix(&mut w, profile);
        w.flag(false).flag(false).flag(true).flag(false);
        sync(&mut w);
        color(&mut w);
        w.u(16, 63).u(16, 47).flag(false);
        let header = parse_alone(&finish(&mut w, &Tail::default()))
            .unwrap_or_else(|error| panic!("profile {profile}: {error}"));
        let color = header.color_config;
        assert_eq!(
            (
                color.bit_depth,
                color.color_space,
                color.color_range,
                color.subsampling_x,
                color.subsampling_y
            ),
            expected,
            "profile {profile}"
        );
    }
}

#[test]
fn colour_values_the_specification_forbids_are_refused() {
    type Case<'a> = (u64, fn(&mut BitWriter), &'a str, u64);
    let cases: [Case<'_>; 4] = [
        (
            0,
            |w| {
                w.u(3, 7);
            },
            "color_space",
            7,
        ),
        (
            1,
            |w| {
                w.u(3, 2).flag(false).flag(true).flag(true);
            },
            "subsampling_y",
            1,
        ),
        (
            1,
            |w| {
                w.u(3, 2).flag(false).flag(true).flag(false).flag(true);
            },
            "reserved_zero",
            1,
        ),
        (
            3,
            |w| {
                w.flag(false).u(3, 7).flag(true);
            },
            "reserved_zero",
            1,
        ),
    ];
    for (profile, color, name, expected) in cases {
        let mut w = BitWriter::new();
        prefix(&mut w, profile);
        w.flag(false).flag(false).flag(true).flag(false);
        sync(&mut w);
        color(&mut w);
        w.u(32, 0);
        let result = parse_alone(&w.bytes());
        assert!(
            matches!(
                &result,
                Err(BitstreamError::InvalidField { field, value, .. })
                    if *field == name && *value == expected
            ),
            "{name}: {result:?}"
        );
    }
}

#[test]
fn intra_only_frames_set_colour_refresh_and_size() {
    let (mut parser, mut refs) = after_key();
    // Profile 0: BT.601 and 8 bits are implied, and the range stays.
    let mut w = BitWriter::new();
    prefix(&mut w, 0);
    w.flag(false).flag(true).flag(false).flag(false);
    w.flag(true).u(2, 3);
    sync(&mut w);
    w.u(8, 0x0c).u(16, 31).u(16, 23).flag(false);
    let frame = finish(&mut w, &Tail::default());
    let header = run(&mut parser, &mut refs, &frame);
    assert!(header.intra_only && header.frame_is_intra);
    assert_eq!(header.reset_frame_context, 3);
    assert_eq!(header.refresh_frame_flags, 0x0c);
    assert_eq!(header.color_config.color_space, 1);
    assert_eq!(header.color_config.bit_depth, 8);
    assert!(!header.color_config.color_range);
    assert_eq!((header.frame_width, header.frame_height), (32, 24));
    assert!(!header.use_prev_frame_mvs);
    // Slots 2 and 3 took the new size. Slot 0 kept the key frame's.
    assert_eq!(refs.ref_frame_width[..4], [64, 64, 32, 32]);
    assert_eq!(header.last_frame_type, Some(KEY_FRAME));

    // Profile 2: the header states the colour.
    let mut w = BitWriter::new();
    prefix(&mut w, 2);
    w.flag(false).flag(true).flag(false).flag(false);
    w.flag(true).u(2, 0);
    sync(&mut w);
    w.flag(true).u(3, 5).flag(true);
    w.u(8, 0xff).u(16, 63).u(16, 47).flag(false);
    let header = run(
        &mut Vp9HeaderParser::new(),
        &mut RefFrameTable::default(),
        &finish(&mut w, &Tail::default()),
    );
    assert_eq!(header.color_config.bit_depth, 12);
    assert_eq!(header.color_config.color_space, 5);
    assert!(header.color_config.color_range);
}

#[test]
fn inter_frames_read_references_size_and_filter() {
    let (mut parser, mut refs) = after_key();
    let header = run(&mut parser, &mut refs, &inter(&Tail::default()));
    assert!(!header.frame_is_intra && !header.intra_only);
    assert_eq!(header.reset_frame_context, 1);
    assert_eq!(header.refresh_frame_flags, 1);
    assert_eq!(header.ref_frame_idx, [0, 1, 2]);
    assert_eq!(header.ref_frame_sign_bias, [false, false, true, false]);
    assert_eq!(header.found_ref, Some(0));
    assert_eq!((header.frame_width, header.frame_height), (64, 48));
    assert_eq!((header.render_width, header.render_height), (64, 48));
    assert!(header.allow_high_precision_mv);
    assert!(header.is_filter_switchable);
    assert_eq!(header.interpolation_filter, SWITCHABLE);
    assert_eq!(header.color_config.bit_depth, 8);
    assert_eq!(header.color_config.color_space, 2);
    assert_eq!(header.last_frame_type, Some(KEY_FRAME));
    assert!(!header.size_changed);
    assert!(header.use_prev_frame_mvs);
    assert!(!header.past_independence);
    assert_eq!(header.frame_context_idx, 2);
}

/// The second `found_ref` picks reference 1, whose slot holds another size.
#[test]
fn found_ref_takes_the_size_from_the_callers_table() {
    let (mut parser, mut refs) = after_key();
    refs.ref_frame_width[1] = 100;
    refs.ref_frame_height[1] = 80;
    let mut w = BitWriter::new();
    prefix(&mut w, 0);
    w.flag(false)
        .flag(true)
        .flag(true)
        .flag(false)
        .u(2, 0)
        .u(8, 0);
    for reference in 0..3 {
        w.u(3, reference).flag(false);
    }
    w.flag(false).flag(true).flag(false);
    w.flag(false).flag(false).u(2, 1);
    let frame = finish(
        &mut w,
        &Tail {
            sb64_cols: 2,
            ..Tail::default()
        },
    );
    let header = run(&mut parser, &mut refs, &frame);
    assert_eq!(header.found_ref, Some(1));
    assert_eq!((header.frame_width, header.frame_height), (100, 80));
    assert_eq!(header.raw_interpolation_filter, 1);
    assert_eq!(header.interpolation_filter, 0);
    assert!(header.size_changed && !header.use_prev_frame_mvs);
    assert!(!header.is_filter_switchable);
}

#[test]
fn an_explicit_size_with_render_size_and_every_filter_literal() {
    for (literal, filter) in [(0, 1), (1, 0), (2, 2), (3, 3)] {
        let (mut parser, mut refs) = after_key();
        let mut w = BitWriter::new();
        prefix(&mut w, 0);
        w.flag(false)
            .flag(true)
            .flag(true)
            .flag(false)
            .u(2, 0)
            .u(8, 0);
        for reference in 0..3 {
            w.u(3, reference).flag(false);
        }
        // Three `found_ref` 0, then the size and the render size.
        w.flag(false).flag(false).flag(false);
        w.u(16, 95).u(16, 71);
        w.flag(true).u(16, 9).u(16, 19);
        w.flag(true).flag(false).u(2, literal);
        let frame = finish(
            &mut w,
            &Tail {
                sb64_cols: 2,
                ..Tail::default()
            },
        );
        let header = run(&mut parser, &mut refs, &frame);
        assert_eq!(header.found_ref, None);
        assert_eq!((header.frame_width, header.frame_height), (96, 72));
        assert_eq!((header.render_width, header.render_height), (10, 20));
        assert_eq!(u64::from(header.raw_interpolation_filter), literal);
        assert_eq!(header.interpolation_filter, filter);
    }
}

#[test]
fn references_the_specification_forbids_are_refused() {
    // An inter frame with no colour state.
    let result = parse_alone(&inter(&Tail::default()));
    assert!(matches!(
        result,
        Err(BitstreamError::InvalidField {
            field: "frame_type",
            value: 1,
            ..
        })
    ));
    // A size taken from an empty slot.
    let (mut parser, refs) = after_key();
    let empty = RefFrameTable::default();
    assert!(matches!(
        parser.parse(&inter(&Tail::default()), &empty),
        Err(BitstreamError::InvalidField {
            field: "ref_frame_idx",
            value: 0,
            ..
        })
    ));
    // Another bit depth in a slot.
    let mut deep = refs;
    deep.ref_bit_depth[2] = 10;
    assert!(matches!(
        parser.parse(&inter(&Tail::default()), &deep),
        Err(BitstreamError::InvalidField {
            field: "ref_frame_idx",
            value: 2,
            ..
        })
    ));
    // Other subsampling in a slot.
    let mut other = refs;
    other.ref_subsampling_x[1] = false;
    assert!(matches!(
        parser.parse(&inter(&Tail::default()), &other),
        Err(BitstreamError::InvalidField {
            field: "ref_frame_idx",
            value: 1,
            ..
        })
    ));
    // A size no reference can scale to: more than twice as small in all three.
    let mut wide = refs;
    wide.ref_frame_width = [64; NUM_REF_FRAMES];
    let mut w = BitWriter::new();
    prefix(&mut w, 0);
    w.flag(false)
        .flag(true)
        .flag(true)
        .flag(false)
        .u(2, 0)
        .u(8, 0);
    for reference in 0..3 {
        w.u(3, reference).flag(false);
    }
    w.flag(false)
        .flag(false)
        .flag(false)
        .u(16, 7)
        .u(16, 47)
        .flag(false);
    w.flag(true).flag(true);
    let frame = finish(&mut w, &Tail::default());
    assert!(matches!(
        parser.parse(&frame, &wide),
        Err(BitstreamError::InvalidField {
            field: "frame_width_minus_1",
            value: 7,
            ..
        })
    ));
}

#[test]
fn loop_filter_deltas_persist_until_independence() {
    let (mut parser, mut refs) = after_key();
    // The key frame set the defaults.
    let header = run(&mut parser, &mut refs, &inter(&Tail::default()));
    assert_eq!(header.loop_filter.loop_filter_ref_deltas, [1, 0, -1, -1]);
    assert_eq!(header.loop_filter.loop_filter_mode_deltas, [0, 0]);
    assert!(header.loop_filter.loop_filter_delta_enabled);
    assert!(!header.loop_filter.loop_filter_delta_update);
    assert_eq!(header.loop_filter.loop_filter_level, 20);
    assert_eq!(header.loop_filter.loop_filter_sharpness, 3);

    let update = Tail {
        deltas: Some(Some(([Some(-63), None, Some(5), None], [None, Some(-1)]))),
        ..Tail::default()
    };
    let header = run(&mut parser, &mut refs, &inter(&update));
    assert!(header.loop_filter.loop_filter_delta_update);
    assert_eq!(
        header.loop_filter.update_ref_delta,
        [true, false, true, false]
    );
    assert_eq!(header.loop_filter.update_mode_delta, [false, true]);
    assert_eq!(header.loop_filter.loop_filter_ref_deltas, [-63, 0, 5, -1]);
    assert_eq!(header.loop_filter.loop_filter_mode_deltas, [0, -1]);

    // A later frame without an update, and one with deltas disabled, keep them.
    let header = run(&mut parser, &mut refs, &inter(&Tail::default()));
    assert_eq!(header.loop_filter.loop_filter_ref_deltas, [-63, 0, 5, -1]);
    let off = Tail {
        deltas: None,
        ..Tail::default()
    };
    let header = run(&mut parser, &mut refs, &inter(&off));
    assert!(!header.loop_filter.loop_filter_delta_enabled);
    assert_eq!(header.loop_filter.loop_filter_mode_deltas, [0, -1]);

    // An error-resilient frame resets them.
    let resilient = Tail {
        error_resilient: true,
        ..Tail::default()
    };
    let header = run(&mut parser, &mut refs, &inter(&resilient));
    assert!(header.past_independence);
    assert_eq!(header.loop_filter.loop_filter_ref_deltas, [1, 0, -1, -1]);
    assert_eq!(header.loop_filter.loop_filter_mode_deltas, [0, 0]);
}

#[test]
fn error_resilient_frames_use_the_inferred_context_fields() {
    let (mut parser, mut refs) = after_key();
    let header = run(
        &mut parser,
        &mut refs,
        &inter(&Tail {
            error_resilient: true,
            ..Tail::default()
        }),
    );
    assert_eq!(header.reset_frame_context, 0);
    assert!(!header.refresh_frame_context);
    assert!(header.frame_parallel_decoding_mode);
    assert_eq!(header.coded_frame_context_idx, 2);
    assert_eq!(header.frame_context_idx, 0);
    assert!(!header.use_prev_frame_mvs);
}

/// Segment 2 sets a negative quantizer, segment 3 a loop filter level, and
/// segment 7 a reference and a skip.
fn segmentation_update(w: &mut BitWriter) {
    w.flag(true);
    for index in 0..7 {
        w.flag(index != 1)
            .u(if index == 1 { 0 } else { 8 }, 10 + index);
    }
    w.flag(true);
    for index in 0..3 {
        w.flag(index != 2)
            .u(if index == 2 { 0 } else { 8 }, 20 + index);
    }
    w.flag(true).flag(false);
    for segment in 0..8 {
        for feature in 0..4 {
            match (segment, feature) {
                (2, 0) => {
                    w.flag(true).u(8, 200).flag(true);
                }
                (3, 1) => {
                    w.flag(true).u(6, 40).flag(false);
                }
                (7, 2) => {
                    w.flag(true).u(2, 3);
                }
                (7, 3) => {
                    w.flag(true);
                }
                _ => {
                    w.flag(false);
                }
            }
        }
    }
}

/// No map update and no data update.
fn segmentation_keep(w: &mut BitWriter) {
    w.flag(false).flag(false);
}

#[test]
fn segmentation_features_persist_until_independence() {
    let (mut parser, mut refs) = after_key();
    let update = Tail {
        segmentation: Some(segmentation_update),
        ..Tail::default()
    };
    let header = run(&mut parser, &mut refs, &inter(&update));
    let params = header.segmentation;
    assert!(params.segmentation_enabled && params.segmentation_update_map);
    assert_eq!(
        params.segmentation_tree_probs,
        [10, 255, 12, 13, 14, 15, 16]
    );
    assert_eq!(params.segmentation_tree_prob_coded[..2], [true, false]);
    assert!(params.segmentation_temporal_update);
    assert_eq!(params.segmentation_pred_prob, [20, 21, 255]);
    assert_eq!(params.segmentation_pred_prob_coded, [true, true, false]);
    assert!(params.segmentation_update_data);
    assert!(!params.segmentation_abs_or_delta_update);
    assert!(params.feature_enabled[2][0] && params.feature_sign[2][0]);
    assert_eq!(params.feature_data[2][0], -200);
    assert_eq!(params.feature_data[3][1], 40);
    assert_eq!(params.feature_data[7][2], 3);
    assert!(params.feature_enabled[7][3]);
    assert_eq!(params.feature_data[7][3], 0);

    // Without an update the features stay, and the probabilities do not.
    let keep = Tail {
        segmentation: Some(segmentation_keep),
        ..Tail::default()
    };
    let header = run(&mut parser, &mut refs, &inter(&keep));
    assert_eq!(header.segmentation.feature_data[2][0], -200);
    assert!(header.segmentation.feature_enabled[7][3]);
    assert_eq!(header.segmentation.segmentation_tree_probs, [255; 7]);
    assert!(!header.segmentation.segmentation_update_map);

    // A frame with segmentation off keeps them for later frames.
    let header = run(&mut parser, &mut refs, &inter(&Tail::default()));
    assert!(!header.segmentation.segmentation_enabled);
    assert_eq!(header.segmentation.feature_data[3][1], 40);

    // A key frame clears them.
    let header = run(&mut parser, &mut refs, &key(&keep));
    assert_eq!(header.segmentation.feature_data[2][0], 0);
    assert!(!header.segmentation.feature_enabled[7][3]);
}

#[test]
fn a_feature_sign_with_absolute_values_is_refused() {
    let (mut parser, refs) = after_key();
    let update = Tail {
        segmentation: Some(|w| {
            // No map update. Data update with absolute values, then a sign.
            w.flag(false).flag(true).flag(true);
            w.flag(true).u(8, 1).flag(true);
        }),
        ..Tail::default()
    };
    assert!(matches!(
        parser.parse(&inter(&update), &refs),
        Err(BitstreamError::InvalidField {
            field: "feature_sign",
            value: 1,
            ..
        })
    ));
}

#[test]
fn quantization_deltas_and_lossless_are_read() {
    let header = parse_alone(&key(&Tail {
        q: (0, [None, None, None]),
        ..Tail::default()
    }))
    .unwrap_or_else(|error| panic!("{error}"));
    assert!(header.quantization.lossless);
    assert_eq!(header.quantization.delta_coded, [false; 3]);
    let header = parse_alone(&key(&Tail {
        q: (0, [Some(-15), Some(15), Some(1)]),
        ..Tail::default()
    }))
    .unwrap_or_else(|error| panic!("{error}"));
    assert!(!header.quantization.lossless);
    assert_eq!(header.quantization.delta_q_y_dc, -15);
    assert_eq!(header.quantization.delta_q_uv_dc, 15);
    assert_eq!(header.quantization.delta_q_uv_ac, 1);
    assert_eq!(header.quantization.delta_coded, [true; 3]);
    // A coded delta of 0 still counts as lossless.
    let header = parse_alone(&key(&Tail {
        q: (0, [Some(0), None, None]),
        ..Tail::default()
    }))
    .unwrap_or_else(|error| panic!("{error}"));
    assert!(header.quantization.lossless);
    assert_eq!(header.quantization.delta_coded, [true, false, false]);
}

#[test]
fn tile_info_follows_the_width_limits() {
    // 64 wide: one superblock column, so no increment bits exist.
    let header = parse_alone(&key(&Tail::default())).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(header.tile_cols_log2, 0);
    // 4160 wide: 65 superblock columns, a minimum of 1 and a maximum of 4.
    for columns in 1..=4 {
        let frame = key_with(
            4160,
            &Tail {
                sb64_cols: 65,
                tile_cols_log2: columns,
                tile_rows_log2: 1,
                ..Tail::default()
            },
        );
        let header = parse_alone(&frame).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(u32::from(header.tile_cols_log2), columns);
        assert_eq!(header.tile_rows_log2, 1);
    }
    // 65536 wide: 1024 superblock columns. The syntax reaches 7, the rule 6.
    let frame = key_with(
        65_536,
        &Tail {
            sb64_cols: 1024,
            tile_cols_log2: 7,
            ..Tail::default()
        },
    );
    assert!(matches!(
        parse_alone(&frame),
        Err(BitstreamError::InvalidField {
            field: "tile_cols_log2",
            value: 7,
            ..
        })
    ));
}

#[test]
fn show_existing_frame_reads_an_index_and_touches_no_state() {
    let (mut parser, refs) = after_key();
    let before = parser.clone();
    let mut w = BitWriter::new();
    prefix(&mut w, 3);
    w.flag(true).u(3, 5);
    let header = parser
        .parse(&w.bytes(), &refs)
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(header.show_existing_frame);
    assert_eq!(header.profile, 3);
    assert_eq!(header.frame_to_show_map_idx, 5);
    assert_eq!(header.refresh_frame_flags, 0);
    assert_eq!(header.header_size_in_bytes, 0);
    assert_eq!(parser, before);
}

#[test]
fn previous_frame_rules_follow_7_2_6() {
    let (mut parser, mut refs) = after_key();
    // A hidden inter frame after a shown frame.
    let mut w = BitWriter::new();
    prefix(&mut w, 0);
    w.flag(false).flag(true).flag(false).flag(false);
    w.flag(false).u(2, 0).u(8, 0);
    for reference in 0..3 {
        w.u(3, reference).flag(false);
    }
    w.flag(true).flag(false).flag(true).flag(true);
    let hidden = finish(&mut w, &Tail::default());
    let header = run(&mut parser, &mut refs, &hidden);
    assert!(!header.show_frame && !header.intra_only);
    assert!(header.use_prev_frame_mvs);
    // A frame after a hidden frame has no previous motion vectors.
    let header = run(&mut parser, &mut refs, &inter(&Tail::default()));
    assert!(!header.use_prev_frame_mvs);
    assert_eq!(header.last_frame_type, Some(1));
    // Shown, same size, and not resilient.
    let header = run(&mut parser, &mut refs, &inter(&Tail::default()));
    assert!(header.use_prev_frame_mvs);
}

#[test]
fn bad_markers_and_sizes_are_refused() {
    let mut frame = key(&Tail::default());
    frame[0] &= 0x3f;
    assert!(matches!(
        parse_alone(&frame),
        Err(BitstreamError::InvalidField {
            field: "frame_marker",
            value: 0,
            ..
        })
    ));
    let mut w = BitWriter::new();
    w.u(2, 2).u(2, 3).flag(true);
    assert!(matches!(
        parse_alone(&w.bytes()),
        Err(BitstreamError::InvalidField {
            field: "reserved_zero",
            value: 1,
            ..
        })
    ));
    let mut frame = key(&Tail::default());
    frame[1] ^= 0xff;
    assert!(matches!(
        parse_alone(&frame),
        Err(BitstreamError::InvalidField {
            field: "frame_sync_byte_0",
            ..
        })
    ));
    // A compressed header larger than what follows.
    let frame = key(&Tail {
        header_size: COMPRESSED as u64 + 1,
        ..Tail::default()
    });
    assert!(matches!(
        parse_alone(&frame),
        Err(BitstreamError::InvalidField {
            field: "header_size_in_bytes",
            ..
        })
    ));
    // A frame with no compressed header reads as a header with padding (6.1).
    let frame = key(&Tail {
        header_size: 0,
        ..Tail::default()
    });
    let header = parse_alone(&frame).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(header.header_size_in_bytes, 0);
}

#[test]
fn a_set_trailing_bit_is_refused() {
    // The default key frame holds 113 header bits, so bit 0 of the last header
    // byte is a trailing bit.
    let mut frame = key(&Tail::default());
    let size = frame.len() - COMPRESSED;
    frame[size - 1] |= 1;
    assert!(matches!(
        parse_alone(&frame),
        Err(BitstreamError::InvalidField {
            field: "zero_bit",
            value: 1,
            ..
        })
    ));
}

#[test]
fn truncation_at_every_byte_is_an_error_and_keeps_the_state() {
    let frames = [
        key(&Tail {
            segmentation: Some(segmentation_update),
            deltas: Some(Some((
                [Some(1), Some(2), Some(3), Some(4)],
                [Some(5), Some(6)],
            ))),
            q: (9, [Some(1), Some(2), Some(3)]),
            ..Tail::default()
        }),
        inter(&Tail::default()),
    ];
    let (parser, refs) = after_key();
    for frame in &frames {
        for end in 0..frame.len() {
            let mut attempt = parser.clone();
            assert!(attempt.parse(&frame[..end], &refs).is_err(), "cut at {end}");
            assert_eq!(attempt, parser, "state after a cut at {end}");
        }
    }
}

#[test]
fn the_table_updates_only_refreshed_slots() {
    let mut refs = RefFrameTable::default();
    let header = UncompressedHeader {
        refresh_frame_flags: 0b1000_0001,
        frame_width: 640,
        frame_height: 480,
        color_config: ColorConfig {
            bit_depth: 10,
            subsampling_x: true,
            ..ColorConfig::default()
        },
        ..UncompressedHeader::default()
    };
    refs.update(&header);
    assert_eq!(refs.ref_frame_width[0], 640);
    assert_eq!(refs.ref_frame_height[7], 480);
    assert_eq!(refs.ref_bit_depth[7], 10);
    assert!(refs.ref_subsampling_x[0] && !refs.ref_subsampling_y[0]);
    assert_eq!(refs.ref_frame_width[1], 0);
    assert_eq!(refs.ref_bit_depth[6], 0);
}
