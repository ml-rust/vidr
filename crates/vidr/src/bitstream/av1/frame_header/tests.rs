//! Whole headers: the state that crosses frames, in `uncompressed_header()`'s bit order.

use super::ref_table::SavedRef;
use super::*;
use crate::bitstream::av1::seq_header::SequenceHeader;
use crate::bitstream::av1::test_stream::{
    InterOptions, header_payload, inter_header, key_header, plain_sequence, show_existing_header,
    write_inter_tail,
};
use crate::bitstream::test_bits::BitWriter;
use crate::error::BitstreamError;

fn parsed(sequence: &SequenceHeader, bits: &BitWriter, refs: &mut RefTable) -> FrameHeader {
    parse_frame_header(sequence, None, &header_payload(bits), refs).expect("parses")
}

/// A table after one shown key frame at `order_hint` 0.
fn after_key(sequence: &SequenceHeader) -> RefTable {
    let mut refs = RefTable::default();
    let key = parsed(sequence, &key_header(0, false), &mut refs);
    refs.update(&key);
    refs
}

#[test]
fn a_shown_key_frame_fills_every_name() {
    let sequence = plain_sequence(64, 64);
    let mut refs = RefTable::default();
    let key = parsed(&sequence, &key_header(5, false), &mut refs);
    assert_eq!(key.frame_type, KEY_FRAME);
    assert!(key.error_resilient_mode);
    assert!(key.frame_is_intra);
    assert!(key.force_integer_mv);
    assert!(!key.coded_force_integer_mv);
    assert_eq!(key.refresh_frame_flags, ALL_FRAMES);
    assert_eq!(key.primary_ref_frame, PRIMARY_REF_NONE);
    assert_eq!(
        (key.upscaled_width, key.frame_width, key.frame_height),
        (64, 64, 64)
    );
    assert_eq!((key.render_width, key.render_height), (64, 64));
    assert_eq!((key.mi_cols, key.mi_rows), (16, 16));
    assert_eq!(key.quantization.base_q_idx, 100);
    assert_eq!(key.loop_filter.level, [1, 2, 3, 4]);
    assert_eq!(key.loop_filter.ref_deltas, DEFAULT_REF_DELTAS);
    assert_eq!(key.tx_mode, TX_MODE_SELECT);
    assert_eq!(key.gm_params[1], IDENTITY_WARP);
    refs.update(&key);
    for name in 0..8 {
        let saved = refs.get(name);
        assert!(saved.valid);
        assert_eq!(saved.order_hint, 5);
        assert_eq!(saved.frame_type, KEY_FRAME);
        assert_eq!(saved.upscaled_width, 64);
    }
}

#[test]
fn a_shown_key_frame_clears_validity_at_its_header() {
    let sequence = plain_sequence(64, 64);
    let mut refs = after_key(&sequence);
    refs.names[3].order_hint = 9;
    parsed(&sequence, &key_header(5, false), &mut refs);
    assert!(!refs.get(3).valid);
    assert_eq!(refs.get(3).order_hint, 0);
}

#[test]
fn references_give_order_hints_and_sign_bias() {
    let sequence = plain_sequence(64, 64);
    let mut refs = after_key(&sequence);
    refs.names[1].order_hint = 6;
    refs.names[2].order_hint = 2;
    let header = parsed(
        &sequence,
        &inter_header(InterOptions {
            order_hint: 4,
            ref_frame_idx: [2, 1, 0, 0, 1, 2, 0],
            ..InterOptions::default()
        }),
        &mut refs,
    );
    assert_eq!(header.order_hints, [0, 2, 6, 0, 0, 6, 2, 0]);
    assert_eq!(
        header.ref_frame_sign_bias,
        [false, false, true, false, false, true, false, false]
    );
    assert_eq!(header.ref_frame_idx, [2, 1, 0, 0, 1, 2, 0]);
    assert_eq!(header.gm_type, [IDENTITY; 8]);
}

/// `primary_ref_frame` loads the loop filter deltas and the segmentation features of its name. A
/// later frame with `PRIMARY_REF_NONE` goes back to the defaults.
#[test]
fn a_primary_reference_loads_its_saved_parameters() {
    let sequence = plain_sequence(64, 64);
    let mut refs = after_key(&sequence);
    refs.names[4].loop_filter_ref_deltas = [9, 8, 7, 6, 5, 4, 3, 2];
    refs.names[4].loop_filter_mode_deltas = [1, -1];
    refs.names[4].feature_enabled[1][2] = true;
    refs.names[4].feature_data[1][2] = 33;
    refs.names[4].gm_params[1][2] = 12345;
    let header = parsed(
        &sequence,
        &inter_header(InterOptions {
            primary_ref_frame: 2,
            ref_frame_idx: [0, 0, 4, 0, 0, 0, 0],
            ..InterOptions::default()
        }),
        &mut refs,
    );
    assert_eq!(header.loop_filter.ref_deltas, [9, 8, 7, 6, 5, 4, 3, 2]);
    assert_eq!(header.loop_filter.mode_deltas, [1, -1]);
    // This frame turns segmentation off, which clears what it loaded.
    assert!(!header.segmentation.feature_enabled[1][2]);
    assert_eq!(header.gm_params[1], IDENTITY_WARP, "no global motion coded");
    let reset = parsed(
        &sequence,
        &inter_header(InterOptions {
            ref_frame_idx: [4; 7],
            ..InterOptions::default()
        }),
        &mut refs,
    );
    assert_eq!(reset.loop_filter.ref_deltas, DEFAULT_REF_DELTAS);
    assert_eq!(reset.loop_filter.mode_deltas, [0, 0]);
}

#[test]
fn a_primary_reference_with_an_empty_name_is_refused_by_name() {
    let sequence = plain_sequence(64, 64);
    let mut refs = after_key(&sequence);
    // A lost name passes `ref_frame_idx`, so `primary_ref_frame` is the field that refuses it.
    refs.names[4].valid = false;
    refs.names[4].lost = true;
    let bits = inter_header(InterOptions {
        primary_ref_frame: 0,
        ref_frame_idx: [4; 7],
        ..InterOptions::default()
    });
    let error = parse_frame_header(&sequence, None, &header_payload(&bits), &mut refs)
        .expect_err("refused");
    assert!(error.to_string().contains("primary_ref_frame"), "{error}");
}

#[test]
fn short_signaling_computes_each_reference() {
    let sequence = plain_sequence(64, 64);
    let mut refs = after_key(&sequence);
    for (name, hint) in [0, 1, 2, 3, 5, 6, 7, 2].into_iter().enumerate() {
        refs.names[name].order_hint = hint;
    }
    let mut w = BitWriter::new();
    w.u(1, 0).u(2, 1).u(1, 1);
    w.u(1, 0).u(1, 0).u(1, 0);
    w.u(4, 4).u(3, 7).u(8, 1);
    // Short signaling: LAST is name 3, and GOLDEN is name 0.
    w.u(1, 1).u(3, 3).u(3, 0);
    write_inter_tail(&mut w);
    let header = parsed(&sequence, &w, &mut refs);
    assert!(header.frame_refs_short_signaling);
    assert_eq!(header.ref_frame_idx, [3, 7, 2, 0, 4, 5, 6]);
}

/// A key frame shown again loads what the name holds and refreshes every name. A second show of
/// it is refused, as section 6.8.2 requires.
#[test]
fn a_key_frame_shown_again_refreshes_every_name_once() {
    let sequence = plain_sequence(64, 64);
    let mut refs = after_key(&sequence);
    let mut w = BitWriter::new();
    // A hidden key frame at hint 9 that fills name 2 and is showable.
    w.u(1, 0).u(2, 0).u(1, 0).u(1, 1).u(1, 0);
    w.u(1, 0);
    w.u(1, 0);
    w.u(4, 9);
    w.u(8, 0b100);
    w.u(1, 0);
    w.u(1, 1);
    w.u(1, 1);
    w.u(8, 100).u(1, 0).u(1, 0).u(1, 0).u(1, 0);
    w.u(1, 0);
    w.u(1, 0);
    w.u(6, 1).u(6, 2).u(6, 3).u(6, 4).u(3, 0);
    w.u(1, 1).u(1, 0);
    w.u(1, 1);
    w.u(1, 0);
    let hidden = parsed(&sequence, &w, &mut refs);
    assert!(hidden.showable_frame);
    assert_eq!(hidden.refresh_frame_flags, 0b100);
    assert_eq!(hidden.film_grain, FilmGrain::default());
    refs.update(&hidden);
    assert_eq!(
        refs.get(0).order_hint,
        0,
        "name 0 keeps the first key frame"
    );

    let show = show_existing_header(2);
    let shown = parse_frame_header(&sequence, None, &show, &mut refs).expect("parses");
    assert!(shown.show_existing_frame);
    assert_eq!(shown.frame_type, KEY_FRAME);
    assert_eq!(shown.refresh_frame_flags, ALL_FRAMES);
    assert_eq!(shown.order_hint, 9);
    refs.update(&shown);
    for name in 0..8 {
        assert_eq!(refs.get(name).order_hint, 9, "name {name}");
    }
    let error = parse_frame_header(&sequence, None, &show, &mut refs).expect_err("refused");
    assert!(
        error.to_string().contains("frame_to_show_map_idx"),
        "{error}"
    );
}

#[test]
fn an_inter_frame_shown_again_refreshes_nothing() {
    let sequence = plain_sequence(64, 64);
    let mut refs = after_key(&sequence);
    let hidden = parsed(
        &sequence,
        &inter_header(InterOptions {
            order_hint: 7,
            show_frame: false,
            showable_frame: true,
            refresh_frame_flags: 0b10,
            ..InterOptions::default()
        }),
        &mut refs,
    );
    refs.update(&hidden);
    let shown =
        parse_frame_header(&sequence, None, &show_existing_header(1), &mut refs).expect("parses");
    assert_eq!(shown.refresh_frame_flags, 0);
    assert_eq!(shown.frame_type, INTER_FRAME);
    assert_eq!(shown.upscaled_size(), (64, 64));
    let before = refs.clone();
    refs.update(&shown);
    assert_eq!(refs, before);
}

#[test]
fn showing_an_empty_or_unshowable_name_is_refused_by_name() {
    let sequence = plain_sequence(64, 64);
    let mut refs = RefTable::default();
    let show = show_existing_header(1);
    let error = parse_frame_header(&sequence, None, &show, &mut refs).expect_err("refused");
    assert!(error.to_string().contains("holds a frame"), "{error}");
    let mut refs = after_key(&sequence);
    let error = parse_frame_header(&sequence, None, &show, &mut refs).expect_err("refused");
    assert!(error.to_string().contains("showable_frame"), "{error}");
}

/// Frame size override with `found_ref`: the size comes from the reference, and a frame that
/// sends its own size checks it against the sequence.
#[test]
fn frame_size_comes_from_a_reference_or_from_the_header() {
    let mut sequence = plain_sequence(64, 64);
    sequence.max_frame_width_minus_1 = 319;
    sequence.max_frame_height_minus_1 = 179;
    let mut refs = RefTable::default();
    refs.names[1] = saved_of_size(160, 90);
    let mut header = FrameHeader {
        frame_size_override_flag: true,
        ref_frame_idx: [0, 1, 0, 0, 0, 0, 0],
        ..FrameHeader::default()
    };
    let mut w = BitWriter::new();
    w.u(1, 0).u(1, 1);
    let bytes = w.bytes();
    let mut reader = crate::bitstream::bits::BitReader::new(&bytes, "test");
    super::size::frame_size_with_refs(&mut reader, &sequence, &refs, &mut header).expect("parses");
    assert_eq!(header.found_ref, Some((2, true)));
    assert_eq!(header.upscaled_size(), (160, 90));
    assert_eq!((header.render_width, header.render_height), (160, 90));
    assert_eq!((header.mi_cols, header.mi_rows), (40, 24));

    let mut header = FrameHeader {
        frame_size_override_flag: true,
        ..FrameHeader::default()
    };
    let mut w = BitWriter::new();
    w.u(12, 500).u(12, 99);
    let bytes = w.bytes();
    let mut reader = crate::bitstream::bits::BitReader::new(&bytes, "test");
    let error = super::size::frame_size(&mut reader, &sequence, &mut header).expect_err("refused");
    assert!(error.to_string().contains("frame_width_minus_1"), "{error}");
}

/// A frame of 200 at denominator 16 codes 100 and upscales to 200. `coded_denom` is the
/// denominator minus 9.
#[test]
fn superres_narrows_the_coded_frame_and_keeps_the_upscaled_width() {
    let mut sequence = plain_sequence(64, 64);
    sequence.enable_superres = true;
    sequence.max_frame_width_minus_1 = 199;
    sequence.max_frame_height_minus_1 = 99;
    let mut w = BitWriter::new();
    w.u(1, 1).u(3, 7);
    let bytes = w.bytes();
    let mut reader = crate::bitstream::bits::BitReader::new(&bytes, "test");
    let mut header = FrameHeader::default();
    super::size::frame_size(&mut reader, &sequence, &mut header).expect("parses");
    assert!(header.use_superres);
    assert_eq!(header.coded_denom, 7);
    assert_eq!(header.upscaled_width, 200);
    assert_eq!(header.frame_width, 100);
    assert_eq!(header.frame_height, 100);
    assert_eq!(header.mi_cols, 26);
}

/// `ref_order_hint` that differs marks the name invalid and takes the coded hint.
#[test]
fn a_differing_ref_order_hint_marks_the_name_invalid() {
    let sequence = plain_sequence(64, 64);
    let mut refs = after_key(&sequence);
    let mut w = BitWriter::new();
    // A frame with error_resilient_mode 1: no primary_ref_frame, and eight hints.
    w.u(1, 0).u(2, 1).u(1, 1);
    w.u(1, 1).u(1, 0).u(1, 0);
    w.u(4, 1).u(8, 1);
    for name in 0..8 {
        w.u(4, u64::from(name == 3) * 5);
    }
    w.u(1, 0);
    for _ in 0..7 {
        w.u(3, 0);
    }
    write_inter_tail(&mut w);
    let header = parsed(&sequence, &w, &mut refs);
    assert_eq!(header.ref_order_hint, Some([0, 0, 0, 5, 0, 0, 0, 0]));
    assert!(!refs.get(3).valid);
    assert_eq!(refs.get(3).order_hint, 5);
    assert!(refs.get(2).valid);
}

/// A frame that names a lost name still parses, as Annex C states. The decoder gets the mask.
#[test]
fn a_frame_can_name_a_reference_its_ref_order_hint_states_as_lost() {
    let sequence = plain_sequence(64, 64);
    let mut refs = after_key(&sequence);
    let mut w = BitWriter::new();
    w.u(1, 0).u(2, 1).u(1, 1);
    w.u(1, 1).u(1, 0).u(1, 0);
    w.u(4, 1).u(8, 1);
    for name in 0..8 {
        w.u(4, u64::from(name == 3) * 5);
    }
    w.u(1, 0);
    for index in 0..7 {
        w.u(3, u64::from(index == 2) * 3);
    }
    write_inter_tail(&mut w);
    let header = parsed(&sequence, &w, &mut refs);
    assert_eq!(header.lost_names, 0b1000);
    assert_eq!(header.ref_frame_idx[2], 3);
    assert!(!refs.get(3).valid);
    assert!(refs.get(3).lost);
    refs.update(&FrameHeader {
        refresh_frame_flags: 0b1000,
        ..header
    });
    assert!(refs.get(3).valid);
    assert!(!refs.get(3).lost);
}

/// A hint that differs on a name that holds no frame loses nothing, so the name stays refused.
#[test]
fn a_ref_order_hint_on_an_empty_name_loses_nothing() {
    let sequence = plain_sequence(64, 64);
    let mut refs = RefTable::default();
    let mut w = BitWriter::new();
    w.u(1, 0).u(2, 1).u(1, 1);
    w.u(1, 1).u(1, 0).u(1, 0);
    w.u(4, 1).u(8, 1);
    for _ in 0..8 {
        w.u(4, 5);
    }
    w.u(1, 0);
    w.u(3, 0);
    let error =
        parse_frame_header(&sequence, None, &header_payload(&w), &mut refs).expect_err("refused");
    assert_eq!(refs.get(0).order_hint, 5);
    assert!(!refs.get(0).lost);
    assert!(error.to_string().contains("ref_frame_idx"), "{error}");
}

#[test]
fn each_reference_with_ref_valid_0_is_refused_by_name() {
    let sequence = plain_sequence(64, 64);
    for index in 0..7 {
        let mut refs = after_key(&sequence);
        refs.names[5].valid = false;
        let mut ref_frame_idx = [0; 7];
        ref_frame_idx[index] = 5;
        let bits = inter_header(InterOptions {
            ref_frame_idx,
            ..InterOptions::default()
        });
        let error = parse_frame_header(&sequence, None, &header_payload(&bits), &mut refs)
            .expect_err("refused");
        let text = error.to_string();
        assert!(text.contains("`ref_frame_idx` is 5"), "{text}");
        assert!(text.contains("key frame"), "{text}");
    }
}

/// `frame_refs_short_signaling` computes `ref_frame_idx`, and the rule covers it too.
#[test]
fn short_signaling_of_a_reference_with_ref_valid_0_is_refused_by_name() {
    let sequence = plain_sequence(64, 64);
    let mut refs = after_key(&sequence);
    refs.names[3].valid = false;
    let mut w = BitWriter::new();
    w.u(1, 0).u(2, 1).u(1, 1);
    w.u(1, 0).u(1, 0).u(1, 0);
    w.u(4, 1).u(3, 7).u(8, 1);
    w.u(1, 1).u(3, 3).u(3, 0);
    write_inter_tail(&mut w);
    let error =
        parse_frame_header(&sequence, None, &header_payload(&w), &mut refs).expect_err("refused");
    assert!(
        error.to_string().contains("`ref_frame_idx` is 3"),
        "{error}"
    );
}

/// A sequence with frame ids: 4 bits of delta, 6 bits of id. An inter frame at `current_frame_id`
/// 20 with one `delta_frame_id_minus_1` for all seven references.
fn inter_with_frame_id(
    sequence: &SequenceHeader,
    refs: &mut RefTable,
    delta: u64,
) -> Result<FrameHeader, BitstreamError> {
    let mut w = BitWriter::new();
    w.u(1, 0).u(2, 1).u(1, 1);
    w.u(1, 0).u(1, 0);
    w.u(6, 20);
    w.u(1, 0);
    w.u(4, 1).u(3, 7).u(8, 1);
    w.u(1, 0);
    for _ in 0..7 {
        w.u(3, 0).u(4, delta);
    }
    write_inter_tail(&mut w);
    parse_frame_header(sequence, None, &header_payload(&w), refs)
}

fn sequence_with_frame_ids() -> SequenceHeader {
    let mut sequence = plain_sequence(64, 64);
    sequence.frame_id_numbers_present_flag = true;
    sequence.delta_frame_id_length_minus_2 = 2;
    sequence.additional_frame_id_length_minus_1 = 1;
    sequence
}

#[test]
fn expected_frame_id_must_match_the_name_it_reads() {
    let sequence = sequence_with_frame_ids();
    let mut refs = RefTable::default();
    refs.names[0] = SavedRef {
        valid: true,
        frame_id: 17,
        ..saved_of_size(64, 64)
    };
    // 20 - (2 + 1) is 17.
    let header = inter_with_frame_id(&sequence, &mut refs, 2).expect("parses");
    assert_eq!(header.expected_frame_id, [17; 7]);
    let error = inter_with_frame_id(&sequence, &mut refs, 3).expect_err("refused");
    let text = error.to_string();
    assert!(text.contains("delta_frame_id_minus_1"), "{text}");
    assert!(text.contains("is 3"), "{text}");
}

/// `mark_ref_frames()` retires a frame id past the delta span of 16, and the frame is no reference.
#[test]
fn a_reference_mark_ref_frames_retired_is_refused_by_name() {
    let sequence = sequence_with_frame_ids();
    let mut refs = RefTable::default();
    refs.names[0] = SavedRef {
        valid: true,
        frame_id: 3,
        ..saved_of_size(64, 64)
    };
    let error = inter_with_frame_id(&sequence, &mut refs, 0).expect_err("refused");
    assert!(!refs.get(0).valid);
    assert!(!refs.get(0).lost);
    assert!(
        error.to_string().contains("`ref_frame_idx` is 0"),
        "{error}"
    );
}

#[test]
fn an_intra_only_frame_that_refreshes_every_name_is_refused_by_name() {
    let sequence = plain_sequence(64, 64);
    let mut refs = after_key(&sequence);
    let mut w = BitWriter::new();
    w.u(1, 0).u(2, 2).u(1, 1);
    w.u(1, 0).u(1, 0).u(1, 0);
    w.u(4, 1).u(8, 0xff);
    w.u(32, 0);
    let error = parse_frame_header(&sequence, None, &w.bytes(), &mut refs).expect_err("refused");
    assert!(error.to_string().contains("refresh_frame_flags"), "{error}");
}

/// Every syntax element the helpers write sits before the last byte, so each cut ends early.
#[test]
fn a_truncation_at_every_byte_is_a_named_error() {
    let sequence = plain_sequence(64, 64);
    let refs = after_key(&sequence);
    let headers = [
        header_payload(&key_header(1, false)),
        header_payload(&inter_header(InterOptions::default())),
    ];
    for bytes in headers {
        for end in 0..bytes.len() {
            let mut refs = refs.clone();
            let error =
                parse_frame_header(&sequence, None, &bytes[..end], &mut refs).expect_err("cut");
            assert!(
                matches!(error, BitstreamError::Truncated { .. }),
                "end {end}: {error}"
            );
        }
    }
}

/// `header_bits` stops before the trailing bits.
#[test]
fn the_header_length_excludes_its_trailing_bits() {
    let sequence = plain_sequence(64, 64);
    let mut refs = RefTable::default();
    let bits = key_header(0, false);
    let header = parsed(&sequence, &bits, &mut refs);
    let mut written = bits.clone();
    written.align();
    assert_eq!(header.header_bytes(), written.bytes().len());
}

/// A valid name with a size, for size tests.
fn saved_of_size(width: u32, height: u32) -> SavedRef {
    SavedRef {
        valid: true,
        upscaled_width: width,
        frame_width: width,
        frame_height: height,
        render_width: width,
        render_height: height,
        ..Default::default()
    }
}
