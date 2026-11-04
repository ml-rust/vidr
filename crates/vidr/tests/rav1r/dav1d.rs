//! `Rav1rEncoder` output decoded by `Dav1dDecoder`, through `av1C`.

use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use vidr::decode::dav1d::Dav1dDecoder;
use vidr::decode::{ChromaLocation, DecodedFrame};
use vidr::encode::{ColorFacts, EncodedPacket, PixelFormat};

use super::support::{CLIPS, Clip, FRAMES, check_psnr, encode, pts_of};
use crate::decode_fixtures::cpu::decode_all;

fn decode_back(
    packets: &[EncodedPacket],
    record: &[u8],
    name: &str,
) -> Vec<DecodedFrame<CpuRuntime>> {
    let mut decoder = Dav1dDecoder::<CpuRuntime>::new(&CpuDevice::new(), record)
        .unwrap_or_else(|error| panic!("{name}: decoder open: {error}"));
    let units = packets
        .iter()
        .map(|packet| (packet.data.clone(), packet.pts));
    decode_all(&mut decoder, units, name)
}

/// One frame's samples in Y, U, V order.
fn samples(frame: &DecodedFrame<CpuRuntime>, name: &str) -> Vec<u16> {
    let read = |plane: &Tensor<CpuRuntime>| -> Vec<u16> {
        let bytes = plane
            .to_bytes()
            .unwrap_or_else(|error| panic!("{name}: reading a plane: {error}"));
        match frame.format() {
            PixelFormat::P010 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|sample| u16::from_le_bytes(*sample) >> 6)
                .collect(),
            _ => bytes.iter().map(|sample| u16::from(*sample)).collect(),
        }
    };
    let mut planar = read(frame.y());
    let chroma = read(frame.uv());
    planar.extend(chroma.iter().step_by(2));
    planar.extend(chroma.iter().skip(1).step_by(2));
    planar
}

fn check_decode(clip: &Clip, packets: &[EncodedPacket], record: &[u8], name: &str) {
    let frames = decode_back(packets, record, name);
    let decoded: Vec<Vec<u16>> = frames.iter().map(|frame| samples(frame, name)).collect();
    check_psnr(clip, &decoded, name);
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(frame.pts(), pts_of(index), "{name}: frame {index} pts");
        assert_eq!(frame.format(), clip.format, "{name}: frame {index} format");
        assert_eq!(
            (frame.display().width(), frame.display().height()),
            (clip.width as u32, clip.height as u32),
            "{name}: frame {index} size"
        );
    }
}

#[test]
fn decodes_back_at_each_depth_and_size() {
    for clip in &CLIPS {
        let name = format!("{clip:?}");
        let (packets, record) = encode(clip, clip.config(), FRAMES, &name);
        check_decode(clip, &packets, &record, &name);
    }
}

#[test]
fn reorder_decodes_back_in_display_order() {
    for clip in &CLIPS[..2] {
        let name = format!("reorder {clip:?}");
        let mut config = clip.config();
        config.b_frames = 3;
        let (packets, record) = encode(clip, config, FRAMES, &name);
        check_decode(clip, &packets, &record, &name);
    }
}

#[test]
fn colour_reads_back_from_the_decoder() {
    let name = "rav1r_dav1d_colour";
    let clip = &CLIPS[0];
    let expected = ColorFacts {
        matrix_coefficients: 1,
        transfer_characteristics: 1,
        colour_primaries: 1,
        full_range: true,
        chroma_location: Some(ChromaLocation::TopLeft),
    };
    let mut config = clip.config();
    config.color = expected;
    let (packets, record) = encode(clip, config, 3, name);
    for frame in decode_back(&packets, &record, name) {
        assert_eq!(*frame.color(), expected, "{name}: colour");
    }
}
