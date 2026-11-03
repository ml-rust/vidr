//! Streams the GPU or vidr refuses, and calls out of order.

use vidr::DecodeError;
use vidr::decode::nvdec::NvdecDecoder;
use vidr::decode::{ChromaFormat, DecodeCodec};

use super::fixtures::{self, Stream};

/// Sends every sample and flushes, and returns the first error.
fn first_error(decoder: &mut NvdecDecoder, stream: &Stream) -> Option<DecodeError> {
    for (sample, pts) in stream.samples.iter().zip(&stream.pts) {
        if let Err(error) = decoder.send(sample, *pts) {
            return Some(error);
        }
    }
    decoder.flush().err()
}

fn refused(name: &'static str) -> Option<(NvdecDecoder, Stream, DecodeError)> {
    let client = fixtures::gpu()?;
    let stream = fixtures::load(name)?;
    let mut decoder = NvdecDecoder::new(&client, stream.codec(), &stream.config)
        .unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let Some(error) = first_error(&mut decoder, &stream) else {
        panic!("{name}: the stream decodes, but the test expects a refusal");
    };
    assert_eq!(decoder.receive().map(|frame| frame.pts()), None);
    Some((decoder, stream, error))
}

/// NVDEC has no H.264 High 10 decoder. The error names the codec, the bit
/// depth, and the chroma format.
#[test]
fn h264_10bit_is_refused_by_the_gpu() {
    let Some((_, _, error)) = refused("h264_10bit") else {
        return;
    };
    match &error {
        DecodeError::Unsupported {
            codec: DecodeCodec::H264,
            bit_depth: 10,
            chroma: ChromaFormat::Yuv420,
            ..
        } => assert!(error.to_string().contains("H.264 10-bit 4:2:0")),
        other => panic!("h264_10bit: expected Unsupported, got {other:?}: {other}"),
    }
}

/// vidr outputs 4:2:0 only, so 4:4:4 is refused before the GPU is asked.
#[test]
fn hevc_444_is_refused_by_vidr() {
    let Some((_, _, error)) = refused("hevc_444") else {
        return;
    };
    assert!(
        matches!(
            error,
            DecodeError::UnsupportedLayout {
                codec: DecodeCodec::Hevc,
                chroma: ChromaFormat::Yuv444,
                bit_depth_luma: 8,
                ..
            }
        ),
        "hevc_444: expected UnsupportedLayout, got {error:?}"
    );
}

/// After an error, every call returns the same error.
#[test]
fn errors_are_sticky() {
    let Some((mut decoder, stream, error)) = refused("hevc_444") else {
        return;
    };
    let pts = stream.pts.last().copied().unwrap_or(0);
    let again = decoder.send(&stream.samples[0], pts + 1);
    assert_eq!(again, Err(error));
}

#[test]
fn empty_packets_and_sends_after_flush_are_errors() {
    let Some(client) = fixtures::gpu() else {
        return;
    };
    let Some(stream) = fixtures::load("h264_8bit") else {
        return;
    };
    let mut decoder = NvdecDecoder::new(&client, DecodeCodec::H264, &stream.config)
        .unwrap_or_else(|error| panic!("open: {error}"));
    assert_eq!(decoder.codec(), DecodeCodec::H264);
    assert_eq!(
        decoder.send(&[], 7),
        Err(DecodeError::EmptyPacket { pts: 7 })
    );
    assert_eq!(decoder.flush(), Ok(()));
    assert_eq!(decoder.queued(), 0);
    assert_eq!(
        decoder.send(&stream.samples[0], 0),
        Err(DecodeError::AfterFlush { pts: 0 })
    );
}

/// A corrupt config record fails at open, with the bitstream error.
#[test]
fn corrupt_config_fails_at_open() {
    let Some(client) = fixtures::gpu() else {
        return;
    };
    let error = NvdecDecoder::new(&client, DecodeCodec::Hevc, &[1, 2, 3]).err();
    assert!(
        matches!(error, Some(DecodeError::Bitstream(_))),
        "expected a bitstream error, got {error:?}"
    );
}
