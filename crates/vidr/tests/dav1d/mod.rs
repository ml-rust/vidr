//! dav1d decode on numr's CPU backend, against ffmpeg's decode.
//!
//! Every test skips with a notice when its fixture is missing. Run
//! `python3 crates/vidr/tests/fixtures/gen_nvdec.py` to write them.

use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use vidr::bitstream::av1::{Av1Config, Av1ToDecoder};
use vidr::decode::dav1d::Dav1dDecoder;
use vidr::decode::{ChromaFormat, ChromaLocation, DecodeCodec, Decoder};
use vidr::{DecodeError, FrameRoute};

use crate::decode_fixtures::cpu::{check_frames, decode_all, samples};
use crate::decode_fixtures::{Stream, load};

fn open(config: &[u8]) -> Result<Dav1dDecoder<CpuRuntime>, DecodeError> {
    Dav1dDecoder::new(&CpuDevice::new(), config)
}

fn location(stream: &Stream) -> Option<ChromaLocation> {
    let config = Av1Config::parse(&stream.config)
        .unwrap_or_else(|error| panic!("{}: av1C: {error}", stream.name));
    ChromaLocation::from_av1(config.chroma_sample_position)
}

/// Decodes container samples with `av1C`, and checks every byte.
fn decodes_exactly(name: &'static str) {
    let Some(stream) = load(name) else {
        return;
    };
    let mut decoder = open(&stream.config).unwrap_or_else(|error| panic!("{name}: open: {error}"));
    assert_eq!(decoder.route(), FrameRoute::HostOnly, "{name}: route");
    let frames = decode_all(&mut decoder, samples(&stream), name);
    check_frames(&stream, &frames, location(&stream));
}

#[test]
fn av1_8bit_equals_ffmpeg() {
    decodes_exactly("av1_8bit");
}

#[test]
fn av1_10bit_equals_ffmpeg() {
    decodes_exactly("av1_10bit");
}

#[test]
fn av1_odd_size_equals_ffmpeg() {
    decodes_exactly("av1_odd");
}

/// The same stream as raw temporal units, so the sequence header comes in band.
#[test]
fn raw_temporal_units_equal_ffmpeg() {
    let name = "av1_8bit";
    let Some(stream) = load(name) else {
        return;
    };
    let mut decoder = open(&[]).unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let frames = decode_all(&mut decoder, temporal_units(&stream), name);
    check_frames(&stream, &frames, location(&stream));
}

fn temporal_units(stream: &Stream) -> Vec<(Vec<u8>, i64)> {
    let name = stream.name;
    let config =
        Av1Config::parse(&stream.config).unwrap_or_else(|error| panic!("{name}: av1C: {error}"));
    let mut filter =
        Av1ToDecoder::new(&config).unwrap_or_else(|error| panic!("{name}: filter: {error}"));
    samples(stream)
        .into_iter()
        .map(|(sample, pts)| {
            let unit = filter
                .filter(&sample)
                .unwrap_or_else(|error| panic!("{name}: filter at pts {pts}: {error}"));
            (unit, pts)
        })
        .collect()
}

/// `av1C` and the in-band sequence header both name the refused layout.
fn refuses(name: &'static str, bit_depth: u8, chroma: ChromaFormat) {
    let Some(stream) = load(name) else {
        return;
    };
    let expected = DecodeError::UnsupportedLayout {
        codec: DecodeCodec::Av1,
        bit_depth_luma: bit_depth,
        bit_depth_chroma: bit_depth,
        chroma,
    };
    assert_eq!(
        open(&stream.config).err(),
        Some(expected.clone()),
        "{name}: new"
    );

    let mut decoder = open(&[]).unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let Some((unit, pts)) = temporal_units(&stream).into_iter().next() else {
        panic!("{name}: the fixture has no samples");
    };
    assert_eq!(
        decoder.send(&unit, pts),
        Err(expected.clone()),
        "{name}: send"
    );
    // The refusal stays for every later call.
    assert_eq!(
        decoder.send(&unit, pts + 1),
        Err(expected.clone()),
        "{name}: second send"
    );
    assert_eq!(decoder.flush(), Err(expected), "{name}: flush");
}

#[test]
fn twelve_bits_are_refused() {
    refuses("av1_12bit", 12, ChromaFormat::Yuv420);
}

#[test]
fn monochrome_is_refused() {
    refuses("av1_mono", 8, ChromaFormat::Monochrome);
}

#[test]
fn yuv422_is_refused() {
    refuses("av1_422", 10, ChromaFormat::Yuv422);
}

#[test]
fn yuv444_is_refused() {
    refuses("av1_444", 8, ChromaFormat::Yuv444);
}

#[test]
fn a_corrupt_config_is_a_bitstream_error() {
    assert!(matches!(open(&[0x81]), Err(DecodeError::Bitstream(_))));
}

#[test]
fn calls_out_of_order_are_refused() {
    let mut decoder = open(&[]).unwrap_or_else(|error| panic!("open: {error}"));
    assert_eq!(
        decoder.send(&[], 5),
        Err(DecodeError::EmptyPacket { pts: 5 })
    );
    assert_eq!(decoder.flush(), Ok(()));
    assert_eq!(decoder.flush(), Ok(()));
    assert_eq!(
        decoder.send(&[0x12, 0x00], 6),
        Err(DecodeError::AfterFlush { pts: 6 })
    );
    assert!(decoder.receive().is_none());
}

/// A cut OBU stops before libdav1d reads it, with an error and no panic.
#[test]
fn a_cut_obu_is_an_error_and_no_panic() {
    let mut decoder = open(&[]).unwrap_or_else(|error| panic!("open: {error}"));
    assert!(matches!(
        decoder.send(&[0x32, 0x7f, 0x00], 1),
        Err(DecodeError::Bitstream(_))
    ));
}
