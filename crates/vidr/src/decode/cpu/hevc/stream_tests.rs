//! Real streams from `crates/vidr/tests/fixtures/gen_hevc.py`: byte-exact output
//! against ffmpeg's raw video in output order, and picture hash SEI messages. A test prints a skip
//! notice when its fixture is absent.

use std::path::PathBuf;

use crate::bitstream::annexb::split;
use crate::bitstream::hevc_pps::tests as pps_tests;
use crate::bitstream::hevc_rps::ShortTermRefPicSet;
use crate::bitstream::hevc_slice::SliceSegmentHeader;
use crate::bitstream::hevc_sps_full::tests as sps_tests;
use crate::bitstream::test_bits::{BitWriter, hevc_nal};
use crate::decode::cpu::dsp::sample::Sample;

use super::cabac::context::{self as ctx, ContextTable};
use super::cabac::test_encoder::CabacEncoder;
use super::dpb::rps::RpsPocs;
use super::error::{HevcError, SequenceFormat};
use super::params::{check_dpb_count, max_dpb_size};
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

/// Every output picture of an Annex-B stream, in output order.
fn decode_all(stream: &[u8], options: DecodeOptions) -> Result<Vec<DecodedPicture>, HevcError> {
    let mut decoder = StreamDecoder::new(options);
    let mut pictures = Vec::new();
    for nal in split(stream).expect("the fixture is Annex-B") {
        decoder.decode_nal(nal)?;
        pictures.extend(std::iter::from_fn(|| decoder.next_output()));
    }
    decoder.flush()?;
    pictures.extend(std::iter::from_fn(|| decoder.next_output()));
    Ok(pictures)
}

/// `PicOrderCntVal` of each picture, in output order.
fn output_pocs(name: &str) -> Option<Vec<i32>> {
    let (stream, _) = fixture(name)?;
    let pictures = decode_all(&stream, DecodeOptions::default())
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    Some(
        pictures
            .iter()
            .map(|picture| match picture {
                DecodedPicture::Eight(frame) => frame.poc,
                DecodedPicture::Ten(frame) => frame.poc,
            })
            .collect(),
    )
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

// H12: POC, reference picture sets, reference lists, the collocated picture.
#[test]
fn p_and_b_pictures_equal_ffmpeg() {
    check_stream_equals_ffmpeg("inter_8");
    check_stream_equals_ffmpeg("p_only_8");
    check_stream_equals_ffmpeg("p_only_10");
    check_stream_equals_ffmpeg("p_noisy_8");
    check_stream_equals_ffmpeg("bframes_8");
    check_stream_equals_ffmpeg("bframes_10");
    check_stream_equals_ffmpeg("b_pyramid_8");
    check_stream_equals_ffmpeg("b_pyramid_10");
}

#[test]
fn several_references_equal_ffmpeg() {
    check_stream_equals_ffmpeg("refs_4_8");
    check_stream_equals_ffmpeg("long_gop_8");
}

/// A 4-bit `slice_pic_order_cnt_lsb` wraps twice in 40 frames. A correct MSB gives
/// the display index as POC.
#[test]
fn poc_wraps_equal_ffmpeg() {
    check_stream_equals_ffmpeg("poc_wrap_8");
    if let Some(pocs) = output_pocs("poc_wrap_8") {
        assert_eq!(pocs, (0..40).collect::<Vec<i32>>());
    }
}

// H13: output order as a whole-stream check.
/// Output rises in POC through one sequence, and every encoded frame leaves.
#[test]
fn output_order_rises_through_a_sequence() {
    for (name, count) in [("b_pyramid_8", 24), ("long_gop_8", 90)] {
        if let Some(pocs) = output_pocs(name) {
            assert_eq!(pocs, (0..count).collect::<Vec<i32>>(), "{name}");
        }
    }
}

/// Each IDR picture restarts POC after the waiting pictures leave.
#[test]
fn scene_cuts_equal_ffmpeg() {
    check_stream_equals_ffmpeg("scene_cut_8");
    if let Some(pocs) = output_pocs("scene_cut_8") {
        assert_eq!(pocs.len(), 24);
        assert_eq!(pocs.iter().filter(|&&poc| poc == 0).count(), 4);
    }
}

/// CRA pictures in mid stream have `NoRaslOutputFlag` 0, so RASL pictures decode.
#[test]
fn an_open_gop_equals_ffmpeg() {
    check_stream_equals_ffmpeg("open_gop_8");
}

/// The stream starts at a CRA picture. Its RASL pictures lose their references, stay
/// out of the output, and leave the later pictures intact.
#[test]
fn rasl_pictures_of_a_first_cra_are_skipped() {
    check_stream_equals_ffmpeg("rasl_cut_8");
    if let Some(pocs) = output_pocs("rasl_cut_8") {
        assert_eq!(pocs.len(), 28);
        assert!(pocs.windows(2).all(|pair| pair[0] < pair[1]));
    }
}

/// C.5.2.2: a CRA picture after an end of sequence NAL unit sets
/// `NoOutputOfPriorPicsFlag` to 1, so the pictures still waiting are dropped.
#[test]
fn a_cra_after_an_end_of_sequence_drops_waiting_pictures() {
    check_stream_equals_ffmpeg("end_of_sequence_8");
}

/// C.5.2.2: `no_output_of_prior_pics_flag` 1 on an IDR picture drops waiting pictures.
#[test]
fn no_output_of_prior_pics_equals_ffmpeg() {
    check_stream_equals_ffmpeg("no_output_prior_8");
}

#[test]
fn a_conformance_window_on_inter_pictures_equals_ffmpeg() {
    check_stream_equals_ffmpeg("window_inter_8");
}

// H14: merge candidates and temporal prediction.
#[test]
fn merge_list_sizes_equal_ffmpeg() {
    check_stream_equals_ffmpeg("merge_5_8");
    check_stream_equals_ffmpeg("merge_1_8");
}

#[test]
fn temporal_prediction_on_and_off_equals_ffmpeg() {
    check_stream_equals_ffmpeg("tmvp_on_8");
    check_stream_equals_ffmpeg("tmvp_off_8");
}

/// x265 has no parallel merge option, so the generator rewrites the PPS. The stream
/// drifts from its encoder and holds no hash, so ffmpeg alone is the reference.
#[test]
fn a_parallel_merge_level_equals_ffmpeg() {
    check_stream_equals_ffmpeg("parallel_merge_8");
}

// H15: motion vector prediction over long searches.
#[test]
fn motion_ranges_equal_ffmpeg() {
    check_stream_equals_ffmpeg("merange_8");
    check_stream_equals_ffmpeg("merange_wide_8");
}

// H16: motion compensation over each partition shape.
#[test]
fn partitions_equal_ffmpeg() {
    check_stream_equals_ffmpeg("rect_8");
    check_stream_equals_ffmpeg("amp_8");
    check_stream_equals_ffmpeg("amp_10");
}

/// No in-loop filter: a difference here lies in prediction or residual alone.
#[test]
fn unfiltered_inter_pictures_equal_ffmpeg() {
    check_stream_equals_ffmpeg("p_no_filters_8");
    check_stream_equals_ffmpeg("b_no_filters_8");
}

#[test]
fn inter_transform_trees_equal_ffmpeg() {
    check_stream_equals_ffmpeg("tu_depth_8");
    check_stream_equals_ffmpeg("tu_depth_1_8");
    check_stream_equals_ffmpeg("cu_qp_inter_8");
    check_stream_equals_ffmpeg("lossless_inter_8");
    check_stream_equals_ffmpeg("cu_lossless_inter_8");
}

#[test]
fn inter_block_structures_equal_ffmpeg() {
    check_stream_equals_ffmpeg("ctu16_inter_8");
    check_stream_equals_ffmpeg("ctu32_inter_8");
    check_stream_equals_ffmpeg("wavefront_inter_8");
    check_stream_equals_ffmpeg("slices_inter_8");
}

// H17: weighted prediction.
#[test]
fn explicit_weights_equal_ffmpeg() {
    check_stream_equals_ffmpeg("weightp_8");
    check_stream_equals_ffmpeg("weightp_10");
    check_stream_equals_ffmpeg("weightb_8");
    check_stream_equals_ffmpeg("weightb_10");
}

// `bS` of inter edges (8.7.2.4).
#[test]
fn deblocking_of_inter_pictures_equals_ffmpeg() {
    check_stream_equals_ffmpeg("deblock_only_8");
    check_stream_equals_ffmpeg("sao_only_inter_8");
}

/// Cut and corrupt inter streams stay in the error paths, with every picture and
/// reference still decided by the stream.
#[test]
fn corrupt_inter_streams_return_errors() {
    let Some((stream, _)) = fixture("b_pyramid_8") else {
        return;
    };
    for length in [stream.len() / 2, stream.len() - 7] {
        let result = decode_all(&stream[..length], DecodeOptions::default());
        if let Err(error) = &result {
            assert!(!error.to_string().is_empty());
        }
    }
    for step in (40..stream.len()).step_by(stream.len() / 60 + 1) {
        let mut corrupt = stream.clone();
        corrupt[step] ^= 0x55;
        let Ok(units) = split(&corrupt) else {
            continue;
        };
        let mut decoder = StreamDecoder::new(DecodeOptions::default());
        for nal in units {
            let _ = decoder.decode_nal(nal);
            while decoder.next_output().is_some() {}
        }
        let _ = decoder.flush();
    }
}

/// A P slice whose reference never arrived names the missing picture. The first
/// picture of the stream is removed, and each later picture refers to it or to a
/// picture that refers to it.
#[test]
fn a_lost_reference_is_a_named_error() {
    let Some((stream, _)) = fixture("p_only_8") else {
        return;
    };
    let units = split(&stream).expect("the fixture is Annex-B");
    let mut decoder = StreamDecoder::new(DecodeOptions::default());
    let mut first_error = None;
    let mut dropped = false;
    for nal in units {
        // Types below 32 are slice segments.
        if nal[0] >> 1 < 32 && !dropped {
            dropped = true;
            continue;
        }
        if let Err(error) = decoder.decode_nal(nal) {
            first_error = first_error.or(Some(error));
        }
    }
    assert!(matches!(
        first_error,
        Some(HevcError::MissingReference { .. })
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
    assert_eq!(decoder.decode_nal(&[]), Ok(()));
    assert_eq!(decoder.decode_nal(&[0x42]), Ok(()));
    // nuh_layer_id 1 on an SPS with no payload.
    assert_eq!(decoder.decode_nal(&[0x42, 0x09]), Ok(()));
    assert_eq!(decoder.flush(), Ok(()));
    assert_eq!(decoder.next_output(), None);
}

const TRAIL_R: u8 = 1;
const IDR_W_RADL: u8 = 19;

/// An SPS of 16x16 CTBs and coding blocks with no SAO and no reference sets.
fn small_sps(width: u32, height: u32, depth_minus8: u32) -> Vec<u8> {
    sps_tests::build(&sps_tests::Spec {
        width,
        height,
        bit_depth_luma_minus8: depth_minus8,
        bit_depth_chroma_minus8: depth_minus8,
        dec_pic_buffering: [1, 1],
        num_reorder: [0, 0],
        min_cb_minus3: 1,
        diff_cb: 0,
        diff_tb: 2,
        depth_inter: 0,
        depth_intra: 0,
        sao: false,
        sets: Vec::new(),
        temporal_mvp: false,
        ..sps_tests::Spec::default()
    })
}

fn small_pps(pps_id: u32) -> Vec<u8> {
    pps_tests::build(&pps_tests::Spec {
        pps_id,
        ..pps_tests::Spec::default()
    })
}

/// One I slice segment of `small_sps` under `small_pps`.
#[derive(Clone, Copy)]
struct SmallSlice {
    nal_unit_type: u8,
    pps_id: u32,
    /// `slice_segment_address` in `address_bits` bits, with `first_slice_segment_in_pic_flag`
    /// 0. `None` for the first slice segment of a picture.
    address: Option<(u32, u64)>,
}

impl SmallSlice {
    const fn first(nal_unit_type: u8) -> Self {
        Self {
            nal_unit_type,
            pps_id: 0,
            address: None,
        }
    }

    /// The NAL unit with `data` as its slice data.
    fn nal(&self, data: &[u8]) -> Vec<u8> {
        let mut header = BitWriter::new();
        header.flag(self.address.is_none());
        if self.nal_unit_type >= 16 {
            // no_output_of_prior_pics_flag.
            header.flag(false);
        }
        header.ue(self.pps_id);
        if let Some((bits, address)) = self.address {
            header.u(bits, address);
        }
        // slice_type I.
        header.ue(2);
        if self.nal_unit_type != IDR_W_RADL {
            // slice_pic_order_cnt_lsb, short_term_ref_pic_set_sps_flag 0, and an empty set.
            header.u(8, 1).flag(false).ue(0).ue(0);
        }
        // slice_qp_delta, then slice_loop_filter_across_slices_enabled_flag.
        header.se(0).flag(true).trailing();
        let mut rbsp = header.bytes();
        rbsp.extend_from_slice(data);
        hevc_nal(self.nal_unit_type, &rbsp)
    }
}

/// The slice data of `count` 16x16 intra CTUs with DC prediction and no residual.
fn intra_ctus(count: usize) -> Vec<u8> {
    let mut contexts = ContextTable::new(0, 26);
    let mut encoder = CabacEncoder::new();
    for index in 1..=count {
        // part_mode 2Nx2N, prev_intra_luma_pred_flag, mpm_idx 1, intra_chroma_pred_mode
        // 4, and no coded block flags.
        encoder.encode_decision(contexts.state_mut(ctx::PART_MODE), true);
        encoder.encode_decision(contexts.state_mut(ctx::PREV_INTRA_LUMA_PRED), true);
        encoder.encode_bypass(true);
        encoder.encode_bypass(false);
        encoder.encode_decision(contexts.state_mut(ctx::INTRA_CHROMA_PRED_MODE), false);
        encoder.encode_decision(contexts.state_mut(ctx::CBF_CHROMA), false);
        encoder.encode_decision(contexts.state_mut(ctx::CBF_CHROMA), false);
        encoder.encode_decision(contexts.state_mut(ctx::CBF_LUMA + 1), false);
        encoder.encode_terminate(index == count);
    }
    encoder.finish()
}

/// The NAL units of `count` 16x16 IDR pictures.
pub(crate) fn intra_pictures(count: usize) -> Vec<Vec<u8>> {
    let mut units = vec![small_sps(16, 16, 0), small_pps(0)];
    units.extend((0..count).map(|_| SmallSlice::first(IDR_W_RADL).nal(&intra_ctus(1))));
    units
}

/// The SPS and PPS of pictures of two 16x16 CTBs.
pub(crate) fn two_ctb_parameter_sets() -> Vec<Vec<u8>> {
    vec![small_sps(32, 16, 0), small_pps(0)]
}

/// An IDR picture of two CTBs.
pub(crate) fn two_ctb_picture() -> Vec<u8> {
    SmallSlice::first(IDR_W_RADL).nal(&intra_ctus(2))
}

/// An IDR picture whose slice stops after the first of two CTBs.
pub(crate) fn half_of_two_ctb_picture() -> Vec<u8> {
    SmallSlice::first(IDR_W_RADL).nal(&intra_ctus(1))
}

/// An IDR picture whose slice has no data, which the parse refuses.
pub(crate) fn empty_slice_picture() -> Vec<u8> {
    SmallSlice::first(IDR_W_RADL).nal(&[])
}

/// The second slice segment of a picture of two CTBs, at address 1.
pub(crate) fn second_ctb_slice() -> Vec<u8> {
    SmallSlice {
        nal_unit_type: IDR_W_RADL,
        pps_id: 0,
        address: Some((1, 1)),
    }
    .nal(&intra_ctus(1))
}

/// `units` as one Annex-B packet.
pub(crate) fn annexb_of(units: &[Vec<u8>]) -> Vec<u8> {
    let mut stream = Vec::new();
    for unit in units {
        stream.extend_from_slice(&[0, 0, 0, 1]);
        stream.extend_from_slice(unit);
    }
    stream
}

/// Feeds `units`, and returns the first error.
fn feed_units(decoder: &mut StreamDecoder, units: &[Vec<u8>]) -> Result<(), HevcError> {
    units.iter().try_for_each(|unit| decoder.decode_nal(unit))
}

#[test]
fn hand_built_pictures_decode() {
    let mut decoder = StreamDecoder::new(DecodeOptions::default());
    assert_eq!(feed_units(&mut decoder, &intra_pictures(2)), Ok(()));
    assert_eq!(decoder.flush(), Ok(()));
    let outputs = std::iter::from_fn(|| decoder.next_output()).count();
    assert_eq!(outputs, 2);
}

/// A trailing picture cannot change the format, because its references keep the old
/// one. The check runs before the new tables and buffers.
#[test]
fn a_format_change_at_a_trailing_picture_is_a_named_error() {
    let previous = SequenceFormat {
        width: 16,
        height: 16,
        bit_depth: 8,
    };
    let changes = [(32, 16, 0), (16, 32, 0), (16, 16, 2)];
    for (width, height, depth_minus8) in changes {
        let mut decoder = StreamDecoder::new(DecodeOptions::default());
        feed_units(&mut decoder, &intra_pictures(1)).unwrap();
        decoder
            .decode_nal(&small_sps(width, height, depth_minus8))
            .unwrap();
        let slice = SmallSlice::first(TRAIL_R).nal(&intra_ctus(1));
        assert_eq!(
            decoder.decode_nal(&slice),
            Err(HevcError::SequenceFormatChange {
                previous,
                current: SequenceFormat {
                    width,
                    height,
                    bit_depth: 8 + depth_minus8,
                },
            })
        );
    }
}

#[test]
fn a_format_change_at_an_idr_picture_decodes() {
    let mut decoder = StreamDecoder::new(DecodeOptions::default());
    feed_units(&mut decoder, &intra_pictures(1)).unwrap();
    decoder.decode_nal(&small_sps(32, 16, 0)).unwrap();
    let slice = SmallSlice::first(IDR_W_RADL).nal(&intra_ctus(2));
    assert_eq!(decoder.decode_nal(&slice), Ok(()));
    assert_eq!(decoder.flush(), Ok(()));
    let widths: Vec<usize> = std::iter::from_fn(|| decoder.next_output())
        .map(|picture| match picture {
            DecodedPicture::Eight(frame) => frame.picture.luma().width(),
            DecodedPicture::Ten(frame) => frame.picture.luma().width(),
        })
        .collect();
    assert_eq!(widths, [16, 32]);
}

#[test]
fn a_trailing_picture_of_the_same_format_decodes() {
    let mut decoder = StreamDecoder::new(DecodeOptions::default());
    feed_units(&mut decoder, &intra_pictures(1)).unwrap();
    decoder.decode_nal(&small_sps(16, 16, 0)).unwrap();
    let slice = SmallSlice::first(TRAIL_R).nal(&intra_ctus(1));
    assert_eq!(decoder.decode_nal(&slice), Ok(()));
    assert_eq!(decoder.flush(), Ok(()));
    assert_eq!(std::iter::from_fn(|| decoder.next_output()).count(), 2);
}

#[test]
fn a_slice_that_names_another_pps_is_a_named_error() {
    let mut decoder = StreamDecoder::new(DecodeOptions::default());
    let first = SmallSlice::first(IDR_W_RADL).nal(&intra_ctus(1));
    let units = [small_sps(32, 16, 0), small_pps(0), small_pps(1), first];
    feed_units(&mut decoder, &units).unwrap();
    let second = SmallSlice {
        nal_unit_type: IDR_W_RADL,
        pps_id: 1,
        address: Some((1, 1)),
    };
    assert!(matches!(
        decoder.decode_nal(&second.nal(&intra_ctus(1))),
        Err(HevcError::InvalidSyntax {
            element: "slice_pic_parameter_set_id",
            value: 1,
            ..
        })
    ));
}

/// A slice segment over CTBs that a slice segment before it holds decodes nothing.
#[test]
fn a_repeated_slice_address_is_a_named_error() {
    let mut decoder = StreamDecoder::new(DecodeOptions::default());
    let first = SmallSlice::first(IDR_W_RADL).nal(&intra_ctus(1));
    let second = SmallSlice {
        nal_unit_type: IDR_W_RADL,
        pps_id: 0,
        address: Some((1, 1)),
    }
    .nal(&intra_ctus(1));
    let units = [small_sps(32, 16, 0), small_pps(0), first, second.clone()];
    feed_units(&mut decoder, &units).unwrap();
    assert!(matches!(
        decoder.decode_nal(&second),
        Err(HevcError::InvalidSyntax {
            element: "slice_segment_address",
            value: 1,
            ..
        })
    ));
}

/// (A-2) with `MaxLumaPs` 35651584.
#[test]
fn max_dpb_size_follows_the_picture_size() {
    assert_eq!(max_dpb_size(4_096, 2_176), 16);
    assert_eq!(max_dpb_size(4_096, 2_184), 12);
    assert_eq!(max_dpb_size(4_096, 4_352), 12);
    assert_eq!(max_dpb_size(4_096, 4_360), 8);
    assert_eq!(max_dpb_size(8_192, 3_264), 8);
    assert_eq!(max_dpb_size(8_192, 3_272), 6);
    assert_eq!(max_dpb_size(8_192, 4_320), 6);
}

/// The error comes from a few bytes, before any plane exists.
#[test]
fn a_buffer_above_max_dpb_size_is_a_named_error() {
    let sps = sps_tests::build(&sps_tests::Spec {
        width: 8_192,
        height: 4_352,
        dec_pic_buffering: [15, 15],
        num_reorder: [0, 0],
        min_cb_minus3: 1,
        diff_cb: 0,
        diff_tb: 2,
        depth_inter: 0,
        depth_intra: 0,
        sao: false,
        sets: Vec::new(),
        temporal_mvp: false,
        ..sps_tests::Spec::default()
    });
    let mut decoder = StreamDecoder::new(DecodeOptions::default());
    feed_units(&mut decoder, &[sps, small_pps(0)]).unwrap();
    assert_eq!(
        decoder.decode_nal(&SmallSlice::first(IDR_W_RADL).nal(&[])),
        Err(HevcError::DpbTooLarge {
            count: 16,
            limit: 6,
            width: 8_192,
            height: 4_352,
        })
    );
}

/// 8.3.3 allocates a picture for each missing entry, so the entries stay under the limit.
#[test]
fn a_reference_set_above_max_dpb_size_is_refused() {
    let set = ShortTermRefPicSet {
        num_negative_pics: 7,
        ..ShortTermRefPicSet::default()
    };
    let header = SliceSegmentHeader {
        nal_unit_type: 21,
        short_term_ref_pic_set: Some(set),
        ..SliceSegmentHeader::default()
    };
    let sps = sps_tests::valid(&sps_tests::Spec::default());
    let count = RpsPocs::derive(&header, &sps, 0).entry_count();
    assert_eq!(count, 7);
    assert_eq!(check_dpb_count(count, 416, 240), Ok(()));
    assert_eq!(
        check_dpb_count(count, 8_192, 4_352),
        Err(HevcError::DpbTooLarge {
            count: 7,
            limit: 6,
            width: 8_192,
            height: 4_352,
        })
    );
}

/// Bits past the end read as zero, so these parse until a named error ends them.
#[test]
fn slices_of_zeros_or_no_data_end_with_errors() {
    let cases: [&[u8]; 3] = [&[], &[0; 4], &[0; 512]];
    for data in cases {
        let mut decoder = StreamDecoder::new(DecodeOptions::default());
        feed_units(&mut decoder, &[small_sps(32, 16, 0), small_pps(0)]).unwrap();
        let slice = SmallSlice::first(IDR_W_RADL).nal(data);
        let ended = decoder.decode_nal(&slice).and_then(|()| decoder.flush());
        assert!(ended.is_err(), "{} bytes of zeros", data.len());
    }
}
