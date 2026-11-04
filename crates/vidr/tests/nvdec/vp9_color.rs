//! VP9 colour: NVDEC reports an all-zero video signal description for VP9,
//! so vidr reads `color_space` and `color_range` from the frame headers.
//!
//! The fixtures record the values ffprobe reads from the raw IVF, which
//! holds the bitstream without `vpcC`. Each test skips with a notice
//! without its fixtures, and the decode tests also without a GPU.

use numr::runtime::cuda::{CudaClient, CudaRuntime};
use vidr::bitstream::vp9_packet_color;
use vidr::decode::nvdec::NvdecDecoder;
use vidr::decode::{DecodedFrame, MATRIX_UNSPECIFIED};

use super::fixtures::{self, Stream};

const STREAMS: [&str; 4] = ["vp9_8bit", "vp9_bt470bg_pc", "vp9_untagged", "vp9_odd"];

/// Decodes every sample and returns the frames.
fn decode(stream: &Stream, client: &CudaClient) -> Vec<DecodedFrame<CudaRuntime>> {
    let name = stream.name;
    let mut decoder = NvdecDecoder::new(client, stream.codec(), &stream.config)
        .unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let mut frames = Vec::new();
    for (sample, pts) in stream.samples.iter().zip(&stream.pts) {
        if let Err(error) = decoder.send(sample, *pts) {
            panic!("{name}: send at pts {pts}: {error}");
        }
        while let Some(frame) = decoder.receive() {
            frames.push(frame);
        }
    }
    if let Err(error) = decoder.flush() {
        panic!("{name}: flush: {error}");
    }
    while let Some(frame) = decoder.receive() {
        frames.push(frame);
    }
    frames
}

/// The parser reads the first sample's header to the values ffprobe reports
/// on the raw IVF. This test needs no GPU.
#[test]
fn key_frame_header_matches_ffprobe_on_the_ivf() {
    for name in STREAMS {
        let Some(stream) = fixtures::load(name) else {
            continue;
        };
        let Some(first) = stream.samples.first() else {
            panic!("{name}: no samples");
        };
        let header = vp9_packet_color(first)
            .unwrap_or_else(|error| panic!("{name}: header: {error}"))
            .unwrap_or_else(|| panic!("{name}: the first sample has no colour"));
        let matrix = match header.color_space {
            1 => 5,
            2 => 1,
            _ => u32::from(MATRIX_UNSPECIFIED),
        };
        assert_eq!(matrix, stream.number("matrix"), "{name}: matrix");
        assert_eq!(
            header.full_range,
            stream.meta("range") == "pc",
            "{name}: range"
        );
        assert_eq!(header.bit_depth, 8, "{name}: bit depth");
    }
}

/// Every decoded frame reports the matrix and range ffprobe reads from the
/// bitstream, and unspecified transfer and primaries.
#[test]
fn decoded_frames_report_the_header_colour() {
    let Some(client) = fixtures::gpu() else {
        return;
    };
    for name in STREAMS {
        let Some(stream) = fixtures::load(name) else {
            continue;
        };
        let frames = decode(&stream, &client);
        assert!(!frames.is_empty(), "{name}: no frame decoded");
        for (index, frame) in frames.iter().enumerate() {
            let color = frame.color();
            assert_eq!(
                u32::from(color.matrix_coefficients),
                stream.number("matrix"),
                "{name} frame {index}: matrix"
            );
            assert_eq!(
                color.full_range,
                stream.meta("range") == "pc",
                "{name} frame {index}: range"
            );
            assert_eq!(
                color.transfer_characteristics, 2,
                "{name} frame {index}: transfer"
            );
            assert_eq!(color.colour_primaries, 2, "{name} frame {index}: primaries");
        }
    }
}

/// A same-size key frame with new colour keeps the colour of every earlier
/// picture and applies to its own.
#[test]
fn a_colour_change_applies_from_its_key_frame() {
    let Some(client) = fixtures::gpu() else {
        return;
    };
    let (Some(first), Some(second)) =
        (fixtures::load("vp9_8bit"), fixtures::load("vp9_bt470bg_pc"))
    else {
        return;
    };
    let mut decoder = NvdecDecoder::new(&client, first.codec(), &first.config)
        .unwrap_or_else(|error| panic!("open: {error}"));
    let mut frames = Vec::new();
    let span = first.pts.last().copied().unwrap_or(0) + 1;
    let joined = first.samples.iter().zip(first.pts.iter().copied()).chain(
        second
            .samples
            .iter()
            .zip(second.pts.iter().map(|pts| pts + span)),
    );
    for (sample, pts) in joined {
        decoder
            .send(sample, pts)
            .unwrap_or_else(|error| panic!("send at pts {pts}: {error}"));
        while let Some(frame) = decoder.receive() {
            frames.push(frame);
        }
    }
    decoder
        .flush()
        .unwrap_or_else(|error| panic!("flush: {error}"));
    while let Some(frame) = decoder.receive() {
        frames.push(frame);
    }
    assert_eq!(frames.len(), first.frames.len() + second.frames.len());
    let (head, tail) = frames.split_at(first.frames.len());
    assert_eq!(
        head[0].display(),
        tail[0].display(),
        "the streams share a size, so no new decoder hides the change"
    );
    for frame in head {
        assert_eq!(
            u32::from(frame.color().matrix_coefficients),
            first.number("matrix")
        );
        assert!(!frame.color().full_range);
    }
    for frame in tail {
        assert_eq!(
            u32::from(frame.color().matrix_coefficients),
            second.number("matrix")
        );
        assert!(frame.color().full_range);
    }
}

/// An untagged stream has no matrix, so conversion needs the caller's.
#[test]
fn untagged_stream_has_no_numr_matrix() {
    let Some(client) = fixtures::gpu() else {
        return;
    };
    let Some(stream) = fixtures::load("vp9_untagged") else {
        return;
    };
    for (index, frame) in decode(&stream, &client).iter().enumerate() {
        assert_eq!(
            frame.color().matrix_coefficients,
            MATRIX_UNSPECIFIED,
            "frame {index}: matrix"
        );
        assert_eq!(frame.color().matrix(), None, "frame {index}: numr matrix");
    }
}
