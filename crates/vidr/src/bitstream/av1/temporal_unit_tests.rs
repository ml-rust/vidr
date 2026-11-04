//! The packet walk: temporal units, tile groups, the operating point, and bad input.

use super::*;
use crate::bitstream::av1::test_stream::{
    InterOptions, frame_payload, header_payload, inter_header, key_header, obu_bytes,
    sequence_payload, show_existing_header,
};

const TD: u8 = 2;
const SEQ: u8 = 1;
const HEADER: u8 = 3;
const GROUP: u8 = 4;
const META: u8 = 5;
const FRAME: u8 = 6;
const REDUNDANT: u8 = 7;
const LIST: u8 = 8;
const PADDING: u8 = 15;

fn joined(obus: &[Vec<u8>]) -> Vec<u8> {
    obus.concat()
}

fn key_unit() -> Vec<u8> {
    joined(&[
        obu_bytes(TD, None, &[]),
        obu_bytes(SEQ, None, &sequence_payload(64, 64, 0)),
        obu_bytes(
            FRAME,
            None,
            &frame_payload(&key_header(0, false), &[0xaa; 5]),
        ),
    ])
}

#[test]
fn a_frame_obu_gives_one_frame_with_its_tile_past_the_header() {
    let unit = key_unit();
    let mut parser = Av1Parser::default();
    let frames = parser.parse_packet(&unit).expect("parses");
    assert_eq!(frames.len(), 1);
    let frame = &frames[0];
    assert!(frame.header.show_frame);
    assert_eq!(frame.parts.len(), 1);
    let whole = frame.parts[0];
    let tile = frame.tiles[0];
    assert_eq!(tile.size, 5);
    let start = tile.offset as usize;
    assert_eq!(&whole[start..start + 5], &[0xaa; 5]);
    assert_eq!(whole.len(), start + 5);
    assert_eq!(whole[0] >> 3 & 0x0f, FRAME);
}

/// A hidden frame and a shown frame in one packet give two frames of one temporal unit.
#[test]
fn a_hidden_frame_and_a_shown_frame_share_a_temporal_unit() {
    let mut parser = Av1Parser::default();
    let first = key_unit();
    parser.parse_packet(&first).expect("parses");
    let hidden = inter_header(InterOptions {
        order_hint: 4,
        show_frame: false,
        showable_frame: true,
        refresh_frame_flags: 0b10,
        ..InterOptions::default()
    });
    let shown = inter_header(InterOptions {
        order_hint: 1,
        ..InterOptions::default()
    });
    let unit = joined(&[
        obu_bytes(TD, None, &[]),
        obu_bytes(FRAME, None, &frame_payload(&hidden, &[1])),
        obu_bytes(META, None, &[1, 2, 3]),
        obu_bytes(PADDING, None, &[0, 0]),
        obu_bytes(FRAME, None, &frame_payload(&shown, &[2])),
    ]);
    let frames = parser.parse_packet(&unit).expect("parses");
    assert_eq!(frames.len(), 2);
    assert!(!frames[0].header.show_frame);
    assert!(frames[1].header.show_frame);
    assert_eq!((frames[0].unit, frames[1].unit), (0, 0));
    // The update of the hidden frame reaches the next header.
    assert_eq!(parser.refs().get(1).order_hint, 4);
}

/// `show_existing_frame` of a hidden frame gives a frame with no tiles, in its own temporal unit.
#[test]
fn a_frame_shown_again_is_a_frame_with_no_tiles() {
    let mut parser = Av1Parser::default();
    parser.parse_packet(&key_unit()).expect("parses");
    let hidden = inter_header(InterOptions {
        order_hint: 4,
        show_frame: false,
        showable_frame: true,
        refresh_frame_flags: 0b10,
        ..InterOptions::default()
    });
    let unit = joined(&[
        obu_bytes(TD, None, &[]),
        obu_bytes(FRAME, None, &frame_payload(&hidden, &[1])),
        obu_bytes(TD, None, &[]),
        obu_bytes(HEADER, None, &show_existing_header(1)),
    ]);
    let frames = parser.parse_packet(&unit).expect("parses");
    assert_eq!(frames.len(), 2);
    assert_eq!((frames[0].unit, frames[1].unit), (0, 1));
    let shown = &frames[1];
    assert!(shown.header.show_existing_frame);
    assert_eq!(shown.header.frame_to_show_map_idx, 1);
    assert_eq!(shown.header.refresh_frame_flags, 0);
    assert_eq!(shown.header.order_hint, 4);
    assert!(shown.tiles.is_empty());
}

/// A frame header OBU and two tile group OBUs: each offset counts the bytes before it.
#[test]
fn tile_groups_give_offsets_into_the_joined_parts() {
    let header_bits = key_header(0, true);
    let header = header_payload(&header_bits);
    // Flag 1, start 0, end 0, alignment. Then flag 1, start 1, end 1, alignment.
    let first_group = [0b1000_0000, 0x11, 0x12];
    let second_group = [0b1110_0000, 0x21];
    let unit = joined(&[
        obu_bytes(SEQ, None, &sequence_payload(128, 64, 0)),
        obu_bytes(HEADER, None, &header),
        obu_bytes(GROUP, None, &first_group),
        obu_bytes(REDUNDANT, None, &header),
        obu_bytes(GROUP, None, &second_group),
    ]);
    let mut parser = Av1Parser::default();
    let frames = parser.parse_packet(&unit).expect("parses");
    assert_eq!(frames.len(), 1);
    let frame = &frames[0];
    assert_eq!(frame.parts.len(), 4);
    assert_eq!(frame.groups.len(), 2);
    assert_eq!(frame.tiles.len(), 2);
    let bitstream = frame.parts.concat();
    let first = frame.tiles[0];
    assert_eq!(
        &bitstream[first.offset as usize..(first.offset + first.size) as usize],
        &[0x11, 0x12]
    );
    let second = frame.tiles[1];
    assert_eq!(bitstream[second.offset as usize], 0x21);
    assert_eq!(second.size, 1);
    assert_eq!(bitstream.len(), (second.offset + second.size) as usize);
}

#[test]
fn a_redundant_header_that_differs_is_refused_by_name() {
    let header = header_payload(&key_header(0, true));
    let other = header_payload(&key_header(3, true));
    let unit = joined(&[
        obu_bytes(SEQ, None, &sequence_payload(128, 64, 0)),
        obu_bytes(HEADER, None, &header),
        obu_bytes(REDUNDANT, None, &other),
    ]);
    let error = Av1Parser::default()
        .parse_packet(&unit)
        .expect_err("refused");
    assert!(error.to_string().contains("frame_header_copy"), "{error}");
}

/// Header bits, then one 1 bit and 0 bits. The three header bits past byte 0 are `101`.
#[test]
fn a_header_copy_holds_the_header_bits_then_trailing_bits() {
    let original = [0xab, 0b1011_0000];
    let copies: [(&[u8], bool); 9] = [
        (&[0xab, 0b1011_0000], true),
        (&[0xab, 0b1011_0000, 0, 0], true),
        (&[], false),
        (&[0xab], false),
        (&[0xaa, 0b1011_0000], false),
        (&[0xab, 0b1001_0000], false),
        (&[0xab, 0b1010_1000], false),
        (&[0xab, 0b1011_0001], false),
        (&[0xab, 0b1011_0000, 1], false),
    ];
    for (copy, expected) in copies {
        assert_eq!(is_header_copy(&original, copy, 11), expected, "{copy:02x?}");
    }
}

/// A header that ends on a byte boundary needs a byte for its trailing bits.
#[test]
fn a_header_copy_at_a_byte_boundary_needs_its_trailing_byte() {
    assert!(is_header_copy(&[1, 2, 0x80], &[1, 2, 0x80], 16));
    assert!(is_header_copy(&[1, 2], &[1, 2, 0x80, 0], 16));
    assert!(!is_header_copy(&[1, 2, 0x80], &[1, 2], 16));
    assert!(!is_header_copy(&[1, 2, 0x80], &[1, 2, 0], 16));
    assert!(!is_header_copy(&[1, 2, 0x80], &[1, 3, 0x80], 16));
}

/// A frame header that stops inside its first tile group, then `extra`.
fn open_frame_then(extra: &[Vec<u8>]) -> Vec<u8> {
    let header = header_payload(&key_header(0, true));
    let mut obus = vec![
        obu_bytes(SEQ, None, &sequence_payload(128, 64, 0)),
        obu_bytes(HEADER, None, &header),
        obu_bytes(GROUP, None, &[0b1000_0000, 0x11]),
    ];
    obus.extend_from_slice(extra);
    joined(&obus)
}

#[test]
fn a_redundant_header_that_is_empty_or_a_prefix_is_refused_by_name() {
    let header = header_payload(&key_header(0, true));
    for end in 0..header.len() {
        let unit = open_frame_then(&[obu_bytes(REDUNDANT, None, &header[..end])]);
        let error = Av1Parser::default()
            .parse_packet(&unit)
            .expect_err("refused");
        let text = error.to_string();
        assert!(text.contains("frame_header_copy"), "end {end}: {text}");
        assert!(text.contains("Remux"), "end {end}: {text}");
    }
}

/// Zero bytes past the trailing bits change nothing, as `obu_size` covers them.
#[test]
fn a_redundant_header_with_more_zero_bytes_is_a_copy() {
    let mut header = header_payload(&key_header(0, true));
    header.extend_from_slice(&[0, 0]);
    let unit = open_frame_then(&[
        obu_bytes(REDUNDANT, None, &header),
        obu_bytes(GROUP, None, &[0b1110_0000, 0x21]),
    ]);
    let frames = Av1Parser::default().parse_packet(&unit).expect("parses");
    assert_eq!(frames.len(), 1);
}

/// An OBU_FRAME holds `byte_alignment()` and tiles past its header, and a copy holds trailing bits.
#[test]
fn a_redundant_header_after_a_frame_obu_compares_header_bits_only() {
    let bits = key_header(0, true);
    let unit = joined(&[
        obu_bytes(SEQ, None, &sequence_payload(128, 64, 0)),
        obu_bytes(FRAME, None, &frame_payload(&bits, &[0b1000_0000, 0x11])),
        obu_bytes(REDUNDANT, None, &header_payload(&bits)),
        obu_bytes(GROUP, None, &[0b1110_0000, 0x21]),
    ]);
    let frames = Av1Parser::default().parse_packet(&unit).expect("parses");
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].groups.len(), 2);
}

/// Section 6.8.1 asks SeenFrameHeader 0 of OBU_FRAME_HEADER. So a copy takes the redundant type.
#[test]
fn a_second_frame_header_obu_is_refused_by_name() {
    let same = header_payload(&key_header(0, true));
    let other = header_payload(&key_header(3, true));
    for payload in [same, other] {
        let unit = open_frame_then(&[obu_bytes(HEADER, None, &payload)]);
        let error = Av1Parser::default()
            .parse_packet(&unit)
            .expect_err("refused");
        let text = error.to_string();
        assert!(text.contains("`obu_type` is 3"), "{text}");
        assert!(text.contains("OBU_REDUNDANT_FRAME_HEADER"), "{text}");
    }
}

/// A same or a new sequence header both refuse. A new one clears the names `set_sequence` holds,
/// under a frame that read them.
#[test]
fn a_new_sequence_header_inside_an_open_frame_is_refused_by_name() {
    let unit = open_frame_then(&[obu_bytes(SEQ, None, &sequence_payload(64, 64, 0))]);
    let mut parser = Av1Parser::default();
    let error = parser.parse_packet(&unit).expect_err("refused");
    let text = error.to_string();
    assert!(text.contains("`obu_type` is 1"), "{text}");
    assert!(text.contains("sequence header"), "{text}");
    assert_eq!(
        parser
            .sequence()
            .map(|sequence| sequence.max_frame_width_minus_1),
        Some(127)
    );
}

/// A repeat of the current sequence inside an open frame changes nothing. The frame itself
/// stays open here, so the error names that and not the header.
#[test]
fn a_repeated_sequence_header_inside_an_open_frame_passes() {
    let unit = open_frame_then(&[obu_bytes(SEQ, None, &sequence_payload(128, 64, 0))]);
    let mut parser = Av1Parser::default();
    if let Err(error) = parser.parse_packet(&unit) {
        let text = error.to_string();
        assert!(!text.contains("new sequence header"), "{text}");
    }
}

/// A sequence header between two frames of a packet still parses.
#[test]
fn a_sequence_header_between_frames_passes() {
    let unit = joined(&[
        key_unit(),
        obu_bytes(SEQ, None, &sequence_payload(64, 64, 0)),
    ]);
    let frames = Av1Parser::default().parse_packet(&unit).expect("parses");
    assert_eq!(frames.len(), 1);
}

/// The sequence selects temporal layer 0 and spatial layer 0. A frame of spatial layer 1 drops
/// whole, so the parser reads none of its bytes.
#[test]
fn the_operating_point_drops_obus_outside_its_layers() {
    let key = frame_payload(&key_header(0, false), &[0xaa]);
    let unit = joined(&[
        obu_bytes(TD, None, &[]),
        obu_bytes(SEQ, None, &sequence_payload(64, 64, 0x101)),
        obu_bytes(FRAME, Some((0, 0)), &key),
        obu_bytes(FRAME, Some((0, 1)), &[0xff, 0xff]),
        obu_bytes(FRAME, Some((1, 0)), &[0xff, 0xff]),
    ]);
    let mut parser = Av1Parser::default();
    let frames = parser.parse_packet(&unit).expect("parses");
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].header.temporal_id, 0);
}

#[test]
fn a_frame_with_no_sequence_header_is_refused_by_name() {
    let unit = obu_bytes(FRAME, None, &frame_payload(&key_header(0, false), &[1]));
    let error = Av1Parser::default()
        .parse_packet(&unit)
        .expect_err("refused");
    assert!(error.to_string().contains("sequence_header_obu"), "{error}");
}

#[test]
fn a_tile_group_with_no_frame_header_is_refused_by_name() {
    let unit = joined(&[
        obu_bytes(SEQ, None, &sequence_payload(64, 64, 0)),
        obu_bytes(GROUP, None, &[1, 2]),
    ]);
    let error = Av1Parser::default()
        .parse_packet(&unit)
        .expect_err("refused");
    assert!(error.to_string().contains("tile group"), "{error}");
}

#[test]
fn a_frame_that_stops_before_its_last_tile_is_a_named_error() {
    let header = header_payload(&key_header(0, true));
    let unit = joined(&[
        obu_bytes(SEQ, None, &sequence_payload(128, 64, 0)),
        obu_bytes(HEADER, None, &header),
        obu_bytes(GROUP, None, &[0b1000_0000, 0x11]),
    ]);
    let error = Av1Parser::default()
        .parse_packet(&unit)
        .expect_err("refused");
    assert!(error.to_string().contains("tile_group_obu"), "{error}");
}

#[test]
fn a_tile_list_is_refused_by_name() {
    let unit = joined(&[
        obu_bytes(SEQ, None, &sequence_payload(64, 64, 0)),
        obu_bytes(LIST, None, &[0]),
    ]);
    let error = Av1Parser::default()
        .parse_packet(&unit)
        .expect_err("refused");
    assert!(error.to_string().contains("Large-scale"), "{error}");
}

/// A repeated sequence header keeps its `Arc`, and another one replaces it and clears the names.
#[test]
fn only_a_changed_sequence_header_clears_the_names() {
    let mut parser = Av1Parser::default();
    parser.parse_packet(&key_unit()).expect("parses");
    let first = parser.sequence().cloned().expect("sequence");
    parser
        .set_sequence(&sequence_payload(64, 64, 0))
        .expect("parses");
    assert!(parser.refs().get(0).valid);
    assert!(Arc::ptr_eq(&first, parser.sequence().expect("sequence")));
    parser
        .set_sequence(&sequence_payload(32, 64, 0))
        .expect("parses");
    assert!(!parser.refs().get(0).valid);
    assert!(!Arc::ptr_eq(&first, parser.sequence().expect("sequence")));
}

/// A cut anywhere inside an OBU is a named error. A cut between OBUs is a packet with fewer OBUs.
#[test]
fn a_truncation_at_every_byte_never_panics() {
    let hidden = inter_header(InterOptions {
        order_hint: 4,
        show_frame: false,
        showable_frame: true,
        ..InterOptions::default()
    });
    let obus = [
        obu_bytes(TD, None, &[]),
        obu_bytes(SEQ, None, &sequence_payload(64, 64, 0)),
        obu_bytes(
            FRAME,
            None,
            &frame_payload(&key_header(0, false), &[0xaa; 3]),
        ),
        obu_bytes(FRAME, None, &frame_payload(&hidden, &[0xbb; 3])),
        obu_bytes(HEADER, None, &show_existing_header(0)),
    ];
    let unit = joined(&obus);
    let mut boundaries = vec![0];
    for one in &obus {
        boundaries.push(boundaries[boundaries.len() - 1] + one.len());
    }
    for end in 0..unit.len() {
        let result = Av1Parser::default().parse_packet(&unit[..end]);
        if !boundaries.contains(&end) {
            let error = result.expect_err("cut inside an OBU");
            assert!(
                matches!(error, BitstreamError::Truncated { .. }),
                "end {end}: {error}"
            );
        }
    }
}

#[test]
fn annex_b_framing_is_told_from_low_overhead_framing() {
    // temporal_unit_size 7, frame_unit_size 6, obu_length 1, and a delimiter with no size field.
    // Then obu_length 3 and a frame OBU of two bytes.
    let annex_b = [7, 6, 1, 0x10, 3, 0x30, 0xaa, 0xbb];
    assert!(looks_like_annex_b(&annex_b));
    assert!(!looks_like_annex_b(&key_unit()));
    assert!(!looks_like_annex_b(&[0x12, 0x00]));
    assert!(!looks_like_annex_b(&[]));
    assert!(!looks_like_annex_b(&[1, 0]));
}
