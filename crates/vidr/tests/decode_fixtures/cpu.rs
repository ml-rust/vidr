//! CPU decoders on numr's CPU backend against ffmpeg's decode, byte for byte.
//!
//! H.264, AV1, and VP9 decoding is bit-exact by specification, film grain aside, and
//! no fixture has film grain. A mismatch is a defect, never a tolerance to widen.

use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;
#[cfg(any(feature = "dav1d", feature = "openh264", feature = "openh264-cisco"))]
use vidr::decode::ChromaLocation;
use vidr::decode::{DecodedFrame, Decoder, PixelFormat};

use super::Stream;

/// Sends every packet, receiving after each, then flushes and drains, through
/// the backend-neutral trait.
pub(crate) fn decode_all(
    decoder: &mut dyn Decoder<CpuRuntime>,
    packets: impl IntoIterator<Item = (Vec<u8>, i64)>,
    name: &str,
) -> Vec<DecodedFrame<CpuRuntime>> {
    let mut frames = Vec::new();
    for (packet, pts) in packets {
        decoder
            .send(&packet, pts)
            .unwrap_or_else(|error| panic!("{name}: send at pts {pts}: {error}"));
        while let Some(frame) = decoder.receive() {
            frames.push(frame);
        }
    }
    decoder
        .flush()
        .unwrap_or_else(|error| panic!("{name}: flush: {error}"));
    while let Some(frame) = decoder.receive() {
        frames.push(frame);
    }
    frames
}

/// Every sample with its pts, in decode order.
pub(crate) fn samples(stream: &Stream) -> Vec<(Vec<u8>, i64)> {
    stream
        .samples
        .iter()
        .cloned()
        .zip(stream.pts.iter().copied())
        .collect()
}

fn plane_bytes(plane: &Tensor<CpuRuntime>, name: &str) -> Vec<u8> {
    plane
        .to_bytes()
        .unwrap_or_else(|error| panic!("{name}: reading a plane: {error}"))
}

/// The fixture's pixel format.
pub(crate) fn stream_format(stream: &Stream) -> PixelFormat {
    match stream.meta("format") {
        "nv12" => PixelFormat::Nv12,
        "p010" => PixelFormat::P010,
        other => panic!("{}: format `{other}`", stream.name),
    }
}

/// Checks every plane byte against the fixture's `.yuv`.
pub(crate) fn check_bytes(stream: &Stream, frames: &[DecodedFrame<CpuRuntime>]) {
    let name = stream.name;
    let Some(yuv) = &stream.yuv else {
        panic!("{name}: the fixture has no .yuv");
    };
    let mut decoded = Vec::with_capacity(yuv.len());
    for frame in frames {
        decoded.extend(plane_bytes(frame.y(), name));
        decoded.extend(plane_bytes(frame.uv(), name));
    }
    assert_eq!(decoded.len(), yuv.len(), "{name}: decoded bytes");
    if let Some(offset) = decoded.iter().zip(yuv).position(|(a, b)| a != b) {
        panic!(
            "{name}: byte {offset} differs from ffmpeg's decode: {} against {}",
            decoded[offset], yuv[offset]
        );
    }
}

/// Checks pts, size, format, colour, and every plane byte against the fixture.
#[cfg(any(feature = "dav1d", feature = "openh264", feature = "openh264-cisco"))]
pub(crate) fn check_frames(
    stream: &Stream,
    frames: &[DecodedFrame<CpuRuntime>],
    location: Option<ChromaLocation>,
) {
    let name = stream.name;
    let pts: Vec<i64> = frames.iter().map(DecodedFrame::pts).collect();
    assert_eq!(pts, stream.frames, "{name}: display-order pts");

    let (width, height) = (stream.number("width"), stream.number("height"));
    let format = stream_format(stream);
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(frame.format(), format, "{name} frame {index}: format");
        assert_eq!(
            (frame.display().width(), frame.display().height()),
            (width, height),
            "{name} frame {index}: display size"
        );
        assert!(frame.progressive(), "{name} frame {index}: progressive");
        let color = frame.color();
        assert_eq!(
            color.chroma_location, location,
            "{name} frame {index}: chroma location"
        );
        assert_eq!(
            u32::from(color.matrix_coefficients),
            stream.number("matrix"),
            "{name} frame {index}: matrix"
        );
        assert_eq!(
            u32::from(color.transfer_characteristics),
            stream.number("transfer"),
            "{name} frame {index}: transfer"
        );
        assert_eq!(
            u32::from(color.colour_primaries),
            stream.number("primaries"),
            "{name} frame {index}: primaries"
        );
        assert_eq!(
            color.full_range,
            stream.meta("range") == "pc",
            "{name} frame {index}: range"
        );
    }
    check_bytes(stream, frames);
}
