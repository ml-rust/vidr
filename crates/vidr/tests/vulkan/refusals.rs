//! What the decoder refuses by name, and its trait contract.

use vidr::DecodeError;
use vidr::decode::{DecodeCodec, Decoder};
use vidr::vulkan::VulkanDecoder;

use super::support::{assert_no_validation_errors, device, samples};
use crate::decode_fixtures::load;

#[test]
fn field_coded_and_10_bit_streams_are_refused_at_open() {
    let Some((device, _guard)) = device() else {
        return;
    };
    for name in [
        "h264_mbaff_progressive",
        "h264_mbaff_interlaced",
        "h264_frame_flagged",
    ] {
        let Some(stream) = load(name) else {
            continue;
        };
        let error = VulkanDecoder::new(&device, DecodeCodec::H264, &stream.config)
            .err()
            .unwrap_or_else(|| panic!("{name}: opens"));
        assert!(
            matches!(error, DecodeError::H264Interlaced { .. }),
            "{name}: {error}"
        );
    }
    if let Some(stream) = load("h264_10bit") {
        let error = VulkanDecoder::new(&device, DecodeCodec::H264, &stream.config)
            .err()
            .unwrap_or_else(|| panic!("h264_10bit: opens"));
        assert!(
            matches!(error, DecodeError::H264BitDepth { luma: 10, .. }),
            "h264_10bit: {error}"
        );
    }
}

#[test]
fn other_codecs_are_not_supported_yet() {
    let Some((device, _guard)) = device() else {
        return;
    };
    for codec in [DecodeCodec::Hevc, DecodeCodec::Av1, DecodeCodec::Vp9] {
        let error = VulkanDecoder::new(&device, codec, &[])
            .err()
            .unwrap_or_else(|| panic!("{codec}: opens"));
        let text = error.to_string();
        assert!(matches!(error, DecodeError::VulkanRefused { .. }), "{text}");
        assert!(text.contains("not supported yet"), "{text}");
    }
}

#[test]
fn a_stream_that_starts_on_a_p_picture_is_refused() {
    let Some((device, _guard)) = device() else {
        return;
    };
    let name = "h264_baseline";
    let Some(stream) = load(name) else {
        return;
    };
    let mut decoder = VulkanDecoder::new(&device, DecodeCodec::H264, &stream.config)
        .unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let packets = samples(&stream);
    let (packet, pts) = &packets[1];
    let error = decoder.send(packet, *pts).expect_err("refused");
    assert!(
        matches!(error, DecodeError::H264References { .. }),
        "{name}: {error}"
    );
    assert_eq!(decoder.send(packet, *pts), Err(error), "the error repeats");
    assert_no_validation_errors(&device, name);
}

/// Baseline holds every frame as a reference, so a dropped frame leaves a gap.
#[test]
fn a_gap_in_frame_num_is_refused_by_name() {
    let Some((device, _guard)) = device() else {
        return;
    };
    let name = "h264_baseline";
    let Some(stream) = load(name) else {
        return;
    };
    let mut decoder = VulkanDecoder::new(&device, DecodeCodec::H264, &stream.config)
        .unwrap_or_else(|error| panic!("{name}: open: {error}"));
    let packets = samples(&stream);
    let (first, pts) = &packets[0];
    decoder
        .send(first, *pts)
        .unwrap_or_else(|error| panic!("{name}: send: {error}"));
    let (third, pts) = &packets[2];
    let error = decoder.send(third, *pts).expect_err("refused");
    assert!(error.to_string().contains("frame_num"), "{name}: {error}");
    assert_no_validation_errors(&device, name);
}

#[test]
fn the_trait_contract_holds() {
    let Some((device, _guard)) = device() else {
        return;
    };
    let mut decoder = VulkanDecoder::new(&device, DecodeCodec::H264, &[])
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        decoder.send(&[], 7),
        Err(DecodeError::EmptyPacket { pts: 7 })
    );
    assert_eq!(decoder.flush(), Ok(()));
    assert_eq!(decoder.flush(), Ok(()), "a second flush does nothing");
    assert!(decoder.receive().is_none());
    assert_eq!(
        decoder.send(&[0, 0, 1, 9], 8),
        Err(DecodeError::AfterFlush { pts: 8 })
    );
}
