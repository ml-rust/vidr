//! The picture structure flags NVDEC reports, against what ffprobe reads
//! from the same streams.
//!
//! `DecodedFrame::progressive` is NVDEC's flag. A stream coded with
//! `frame_mbs_only_flag` 0 can disagree with the content, so the test prints
//! one line per stream for the cases it does not assert. Every test skips
//! with a notice when the NVIDIA library, a CUDA device, or a fixture is
//! missing.

use vidr::decode::nvdec::NvdecDecoder;

use super::fixtures::{self, Stream};

/// What NVDEC reported for the frames of one stream.
struct Observed {
    frames: usize,
    progressive: usize,
    top_field_first: usize,
}

fn observe(decoder: &mut NvdecDecoder, stream: &Stream) -> Observed {
    let mut observed = Observed {
        frames: 0,
        progressive: 0,
        top_field_first: 0,
    };
    let mut count = |decoder: &mut NvdecDecoder| {
        while let Some(frame) = decoder.receive() {
            observed.frames += 1;
            observed.progressive += usize::from(frame.progressive());
            observed.top_field_first += usize::from(frame.top_field_first());
        }
    };
    for (sample, pts) in stream.samples.iter().zip(&stream.pts) {
        if let Err(error) = decoder.send(sample, *pts) {
            panic!("{}: send at pts {pts}: {error}", stream.name);
        }
        count(decoder);
    }
    if let Err(error) = decoder.flush() {
        panic!("{}: flush: {error}", stream.name);
    }
    count(decoder);
    observed
}

/// Decodes `name` and returns its meta and what NVDEC reported.
fn decode(name: &'static str) -> Option<(Stream, Observed)> {
    let client = fixtures::gpu()?;
    let stream = fixtures::load(name)?;
    let mut decoder = NvdecDecoder::new(&client, stream.codec(), &stream.config)
        .unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let observed = observe(&mut decoder, &stream);
    assert_eq!(observed.frames, stream.frames.len(), "{name}: frame count");
    Some((stream, observed))
}

/// Prints the table row, and the verdict against ffprobe.
fn report(stream: &Stream, observed: &Observed) {
    let (Some(interlaced), Some(tff)) = (
        stream.meta_opt("interlaced"),
        stream.meta_opt("top_field_first"),
    ) else {
        eprintln!(
            "{}: the fixture has no picture structure meta. Regenerate the fixtures.",
            stream.name
        );
        return;
    };
    let expected_progressive = if interlaced == "0" {
        observed.frames
    } else {
        0
    };
    let verdict = if observed.progressive == expected_progressive {
        "agrees with ffprobe"
    } else {
        "DIFFERS from ffprobe"
    };
    eprintln!(
        "{}: ffprobe interlaced={interlaced} top_field_first={tff}. \
         NVDEC progressive on {} of {} frames, top_field_first on {}. {verdict}.",
        stream.name, observed.progressive, observed.frames, observed.top_field_first,
    );
}

/// Plain progressive H.264 reports `progressive` on every frame.
#[test]
fn progressive_h264_reports_progressive() {
    let Some((stream, observed)) = decode("h264_8bit") else {
        return;
    };
    report(&stream, &observed);
    assert_eq!(
        observed.progressive, observed.frames,
        "h264_8bit: every frame must report progressive"
    );
}

/// Interlaced content coded as MBAFF must not report `progressive`. A
/// refusal of interlaced sources relies on this.
#[test]
fn interlaced_mbaff_h264_reports_interlaced() {
    let Some((stream, observed)) = decode("h264_mbaff_interlaced") else {
        return;
    };
    report(&stream, &observed);
    assert_eq!(
        observed.progressive, 0,
        "h264_mbaff_interlaced: no frame may report progressive"
    );
}

/// Progressive content under `frame_mbs_only_flag` 0 is the case where the
/// flag and the content can disagree. The test prints what NVDEC reports and
/// asserts nothing about the flag.
#[test]
fn progressive_content_with_interlaced_tools_is_reported() {
    for name in ["h264_mbaff_progressive", "h264_frame_flagged"] {
        let Some((stream, observed)) = decode(name) else {
            return;
        };
        report(&stream, &observed);
    }
}
