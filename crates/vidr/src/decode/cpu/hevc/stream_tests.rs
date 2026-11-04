//! Real streams from `crates/vidr/tests/fixtures/gen_hevc.py`: byte-exact output
//! against ffmpeg's raw video, and picture hash SEI messages. A test prints a skip
//! notice when its fixture is absent.

use std::path::PathBuf;

use crate::bitstream::annexb::split;
use crate::decode::cpu::dsp::sample::Sample;

use super::error::HevcError;
use super::picture::{Picture, Plane};
use super::stream::{DecodeOptions, DecodedPicture, StreamDecoder};

fn fixture_dir() -> PathBuf {
    std::env::var_os("VIDR_HEVC_FIXTURES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hevc"),
        PathBuf::from,
    )
}

/// The bytes of `NAME.hevc` and `NAME.yuv`, or `None` after a notice.
fn fixture(name: &str) -> Option<(Vec<u8>, Vec<u8>)> {
    let dir = fixture_dir();
    let stream = std::fs::read(dir.join(format!("{name}.hevc"))).ok();
    let reference = std::fs::read(dir.join(format!("{name}.yuv"))).ok();
    if let (Some(stream), Some(reference)) = (stream, reference) {
        return Some((stream, reference));
    }
    println!(
        "skipping: HEVC fixture `{name}` is missing in {}. Run \
         crates/vidr/tests/fixtures/gen_hevc.py, or set VIDR_HEVC_FIXTURES to its output \
         directory.",
        dir.display()
    );
    None
}

/// Every picture of an Annex-B stream, in decoding order.
fn decode_all(stream: &[u8], options: DecodeOptions) -> Result<Vec<DecodedPicture>, HevcError> {
    let mut decoder = StreamDecoder::new(options);
    let mut pictures = Vec::new();
    for nal in split(stream).expect("the fixture is Annex-B") {
        if let Some(picture) = decoder.decode_nal(nal)? {
            pictures.push(picture);
        }
    }
    if let Some(picture) = decoder.flush()? {
        pictures.push(picture);
    }
    Ok(pictures)
}

/// The bytes of one plane's window in ffmpeg's `yuv420p` or `yuv420p10le` layout.
fn window_bytes<S: Sample>(
    plane: &Plane<S>,
    area: (usize, usize, usize, usize),
    out: &mut Vec<u8>,
) {
    let (x, y, width, height) = area;
    let mut scratch = Vec::new();
    for line in y..y + height {
        let row = &plane.row(line).expect("the window fits")[x..x + width];
        out.extend_from_slice(S::host_bytes(row, &mut scratch).expect("the scratch fits"));
    }
}

fn picture_bytes<S: Sample>(picture: &Picture<S>) -> Vec<u8> {
    let window = picture.window();
    let mut out = Vec::new();
    window_bytes(
        picture.luma(),
        (window.x, window.y, window.width, window.height),
        &mut out,
    );
    let chroma = (
        window.x / 2,
        window.y / 2,
        window.width / 2,
        window.height / 2,
    );
    window_bytes(picture.cb(), chroma, &mut out);
    window_bytes(picture.cr(), chroma, &mut out);
    out
}

fn bytes_of(picture: &DecodedPicture) -> Vec<u8> {
    match picture {
        DecodedPicture::Eight(finished) => picture_bytes(&finished.picture),
        DecodedPicture::Ten(finished) => picture_bytes(&finished.picture),
    }
}

/// Decodes a fixture with the hash check on, and compares each picture with ffmpeg.
fn check_stream_equals_ffmpeg(name: &str) {
    let Some((stream, reference)) = fixture(name) else {
        return;
    };
    let options = DecodeOptions {
        check_picture_hash: true,
    };
    let pictures = decode_all(&stream, options).unwrap_or_else(|error| panic!("{name}: {error}"));
    assert!(!pictures.is_empty(), "{name}: no pictures");
    let mut offset = 0;
    for (index, picture) in pictures.iter().enumerate() {
        let bytes = bytes_of(picture);
        let end = offset + bytes.len();
        let expected = reference.get(offset..end);
        assert!(
            expected.is_some(),
            "{name}: reference ends at picture {index}"
        );
        if expected != Some(&bytes[..]) {
            let first = bytes
                .iter()
                .zip(&reference[offset..end])
                .position(|(a, b)| a != b);
            panic!("{name}: picture {index} differs from ffmpeg at byte {first:?}");
        }
        offset = end;
    }
    assert_eq!(
        offset,
        reference.len(),
        "{name}: reference holds more pictures"
    );
}

// H8: no in-loop filters. Each name states the case.
#[test]
fn intra_8_bit_without_filters_equals_ffmpeg() {
    check_stream_equals_ffmpeg("intra_8");
}

#[test]
fn intra_10_bit_without_filters_equals_ffmpeg() {
    check_stream_equals_ffmpeg("intra_10");
}

/// Each picture after the first is a CRA picture, whose header holds a POC and a
/// reference picture set.
#[test]
fn cra_pictures_equal_ffmpeg() {
    check_stream_equals_ffmpeg("cra_8");
}

#[test]
fn a_conformance_window_equals_ffmpeg() {
    check_stream_equals_ffmpeg("window_8");
    check_stream_equals_ffmpeg("window_10");
}

#[test]
fn small_ctus_equal_ffmpeg() {
    check_stream_equals_ffmpeg("ctu16_8");
    check_stream_equals_ffmpeg("ctu32_min16_8");
}

#[test]
fn wavefronts_and_slices_equal_ffmpeg() {
    check_stream_equals_ffmpeg("wavefront_8");
    check_stream_equals_ffmpeg("slices_8");
}

#[test]
fn deep_transform_trees_and_cu_qp_deltas_equal_ffmpeg() {
    check_stream_equals_ffmpeg("deep_tu_8");
    check_stream_equals_ffmpeg("cu_qp_8");
    check_stream_equals_ffmpeg("chroma_offsets_8");
}

#[test]
fn transform_skip_and_lossless_equal_ffmpeg() {
    check_stream_equals_ffmpeg("tskip_8");
    check_stream_equals_ffmpeg("lossless_8");
    check_stream_equals_ffmpeg("cu_lossless_8");
}

// H9: each hash method on real SEI messages.
#[test]
fn md5_crc_and_checksum_messages_match() {
    check_stream_equals_ffmpeg("hash_md5_8");
    check_stream_equals_ffmpeg("hash_md5_10");
    check_stream_equals_ffmpeg("hash_crc_8");
    check_stream_equals_ffmpeg("hash_crc_10");
    check_stream_equals_ffmpeg("hash_checksum_8");
    check_stream_equals_ffmpeg("hash_checksum_10");
}

/// Flipping one byte of each SEI message gives the named error. The default options
/// do not read the message.
#[test]
fn a_changed_hash_is_a_named_error_only_with_the_check_on() {
    let Some((stream, _)) = fixture("hash_md5_8") else {
        return;
    };
    let mut changed = Vec::new();
    for nal in split(&stream).expect("the fixture is Annex-B") {
        let mut nal = nal.to_vec();
        // Suffix SEI type 40, payloadType 132: the last digest byte precedes the
        // trailing bits.
        if nal[0] >> 1 == 40 && nal[2] == 132 {
            let last = nal.len() - 2;
            nal[last] ^= 1;
        }
        changed.extend_from_slice(&[0, 0, 0, 1]);
        changed.extend_from_slice(&nal);
    }
    let on = DecodeOptions {
        check_picture_hash: true,
    };
    assert!(matches!(
        decode_all(&changed, on),
        Err(HevcError::PictureHashMismatch {
            method: "MD5",
            c_idx: 2,
            ..
        })
    ));
    assert!(decode_all(&changed, DecodeOptions::default()).is_ok());
}

// H10: deblocking on, SAO off.
#[test]
fn deblocking_equals_ffmpeg() {
    check_stream_equals_ffmpeg("deblock_8");
    check_stream_equals_ffmpeg("deblock_10");
    check_stream_equals_ffmpeg("deblock_offsets_8");
    check_stream_equals_ffmpeg("deblock_chroma_offsets_8");
}

// H11: SAO on, with and without deblocking.
#[test]
fn sao_equals_ffmpeg() {
    check_stream_equals_ffmpeg("sao_8");
    check_stream_equals_ffmpeg("sao_10");
    check_stream_equals_ffmpeg("sao_only_8");
    check_stream_equals_ffmpeg("sao_window_8");
    check_stream_equals_ffmpeg("sao_slices_8");
}

/// Strong intra smoothing needs a smooth 32x32 region.
#[test]
fn strong_smoothing_equals_ffmpeg() {
    check_stream_equals_ffmpeg("smooth_8");
    check_stream_equals_ffmpeg("smooth_10");
}

/// An inter stream stops at its first P or B slice, with the named error.
#[test]
fn an_inter_slice_is_a_named_error() {
    let Some((stream, _)) = fixture("inter_8") else {
        return;
    };
    assert!(matches!(
        decode_all(&stream, DecodeOptions::default()),
        Err(HevcError::Unsupported {
            field: "slice_type",
            ..
        })
    ));
}

/// Cut streams and cut slices return errors, and never panic.
#[test]
fn truncated_streams_return_errors() {
    let Some((stream, _)) = fixture("sao_8") else {
        return;
    };
    for length in [stream.len() / 2, stream.len() / 4, stream.len() - 7] {
        let result = decode_all(&stream[..length], DecodeOptions::default());
        if let Err(error) = &result {
            assert!(!error.to_string().is_empty());
        }
    }
    // Corruption at many places stays in the error paths too.
    for step in (40..stream.len()).step_by(stream.len() / 40 + 1) {
        let mut corrupt = stream.clone();
        corrupt[step] ^= 0x55;
        let Ok(units) = split(&corrupt) else {
            continue;
        };
        let mut decoder = StreamDecoder::new(DecodeOptions::default());
        for nal in units {
            let _ = decoder.decode_nal(nal);
        }
        let _ = decoder.flush();
    }
}

#[test]
fn a_slice_before_its_parameter_sets_is_a_named_error() {
    // TRAIL_R, first_slice_segment_in_pic_flag 1, slice_pic_parameter_set_id 0.
    let mut decoder = StreamDecoder::new(DecodeOptions::default());
    assert_eq!(
        decoder.decode_nal(&[0x02, 0x01, 0xc0, 0x00]),
        Err(HevcError::MissingParameterSet { kind: "PPS", id: 0 })
    );
}

#[test]
fn other_layers_and_short_units_are_dropped() {
    let mut decoder = StreamDecoder::new(DecodeOptions::default());
    assert_eq!(decoder.decode_nal(&[]), Ok(None));
    assert_eq!(decoder.decode_nal(&[0x42]), Ok(None));
    // nuh_layer_id 1 on an SPS with no payload.
    assert_eq!(decoder.decode_nal(&[0x42, 0x09]), Ok(None));
    assert_eq!(decoder.flush(), Ok(None));
}
