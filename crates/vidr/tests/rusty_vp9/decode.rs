//! VP9 output against ffmpeg's decode, byte for byte. The tolerance is zero: VP9
//! decoding is bit-exact by specification.

use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use vidr::decode::rusty_vp9::RustyVp9Decoder;
use vidr::decode::{
    DecodedFrame, Decoder, MATRIX_UNSPECIFIED, PRIMARIES_UNSPECIFIED, TRANSFER_UNSPECIFIED,
};
use vidr::{DecodeError, FrameRoute};

use crate::decode_fixtures::cpu::{check_bytes, decode_all, samples, stream_format};
use crate::decode_fixtures::{Stream, load};

/// A one-byte `show_existing_frame` header of profile 0: frame marker 2, profile
/// 0, show existing 1, slot 0.
const SHOW_SLOT_0: u8 = 0x88;

fn open() -> RustyVp9Decoder<CpuRuntime> {
    RustyVp9Decoder::new(&CpuDevice::new())
}

/// Checks the facts every frame of `stream` holds. Matrix and range come from the
/// frame header, which ffprobe reads from the raw IVF. The header states no
/// transfer or primaries, whatever `vpcC` says.
fn check_facts(stream: &Stream, frames: &[DecodedFrame<CpuRuntime>]) {
    let name = stream.name;
    let pts: Vec<i64> = frames.iter().map(DecodedFrame::pts).collect();
    assert_eq!(pts, stream.frames, "{name}: display-order pts");
    let format = stream_format(stream);
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(frame.format(), format, "{name} frame {index}: format");
        assert!(frame.progressive(), "{name} frame {index}: progressive");
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
            color.transfer_characteristics, TRANSFER_UNSPECIFIED,
            "{name} frame {index}: transfer"
        );
        assert_eq!(
            color.colour_primaries, PRIMARIES_UNSPECIFIED,
            "{name} frame {index}: primaries"
        );
        assert_eq!(
            color.chroma_location, None,
            "{name} frame {index}: chroma location"
        );
    }
}

/// Decodes every sample, and checks facts and bytes.
fn decode_stream(stream: &Stream) -> Vec<DecodedFrame<CpuRuntime>> {
    let mut decoder = open();
    assert_eq!(
        decoder.route(),
        FrameRoute::HostOnly,
        "{}: route",
        stream.name
    );
    let frames = decode_all(&mut decoder, samples(stream), stream.name);
    check_facts(stream, &frames);
    check_bytes(stream, &frames);
    frames
}

/// A fixed-size stream, with each frame at the fixture's size.
fn decodes_exactly(name: &'static str) {
    let Some(stream) = load(name) else {
        return;
    };
    let size = (stream.number("width"), stream.number("height"));
    for (index, frame) in decode_stream(&stream).iter().enumerate() {
        assert_eq!(
            (frame.display().width(), frame.display().height()),
            size,
            "{name} frame {index}: display size"
        );
    }
}

#[test]
fn vp9_8bit_equals_ffmpeg() {
    decodes_exactly("vp9_8bit");
}

#[test]
fn vp9_10bit_equals_ffmpeg() {
    decodes_exactly("vp9_10bit");
}

#[test]
fn vp9_odd_size_equals_ffmpeg() {
    decodes_exactly("vp9_odd");
}

#[test]
fn a_header_with_bt601_full_range_equals_ffmpeg() {
    decodes_exactly("vp9_bt470bg_pc");
}

#[test]
fn an_untagged_stream_equals_ffmpeg() {
    decodes_exactly("vp9_untagged");
    if let Some(stream) = load("vp9_untagged") {
        assert_eq!(stream.number("matrix"), u32::from(MATRIX_UNSPECIFIED));
    }
}

/// Spatial resize changes the frame size, and each frame keeps its coded size.
#[test]
fn a_stream_that_resizes_equals_ffmpeg() {
    let name = "vp9_resize";
    let Some(stream) = load(name) else {
        return;
    };
    let frames = decode_stream(&stream);
    let mut sizes: Vec<(u32, u32)> = frames
        .iter()
        .map(|frame| (frame.display().width(), frame.display().height()))
        .collect();
    sizes.dedup();
    assert!(
        sizes.len() >= 2,
        "{name}: the fixture never resizes: {sizes:?}. Regenerate it with a lower rate."
    );
}

/// Superframes hold hidden alt-ref frames. Each packet gives one frame, at its pts.
#[test]
fn hidden_frames_in_superframes_equal_ffmpeg() {
    let name = "vp9_altref";
    let Some(stream) = load(name) else {
        return;
    };
    let superframes = stream
        .samples
        .iter()
        .filter(|sample| sample.last().is_some_and(|byte| byte & 0xe0 == 0xc0))
        .count();
    assert!(superframes > 0, "{name}: the fixture holds no superframe");
    let frames = decode_stream(&stream);
    assert_eq!(
        frames.len(),
        stream.samples.len(),
        "{name}: one frame per packet"
    );
}

fn first_sample(stream: &Stream) -> &[u8] {
    match stream.samples.first() {
        Some(sample) => sample,
        None => panic!("{}: the fixture has no samples", stream.name),
    }
}

fn plane_bytes(frame: &DecodedFrame<CpuRuntime>) -> Vec<u8> {
    let bytes = |plane: &numr::tensor::Tensor<CpuRuntime>| {
        plane
            .to_bytes()
            .unwrap_or_else(|error| panic!("reading a plane: {error}"))
    };
    [bytes(frame.y()), bytes(frame.uv())].concat()
}

/// A key frame fills every slot, so showing slot 0 repeats it. The repeat takes
/// its own pts, and the colour of the frame in the slot.
#[test]
fn show_existing_repeats_a_slot_with_its_colour() {
    let name = "vp9_bt470bg_pc";
    let Some(stream) = load(name) else {
        return;
    };
    let packets = vec![
        (first_sample(&stream).to_vec(), 100),
        (vec![SHOW_SLOT_0], 200),
    ];
    let frames = decode_all(&mut open(), packets, name);
    assert_eq!(frames.len(), 2, "{name}: frame count");
    assert_eq!(
        (frames[0].pts(), frames[1].pts()),
        (100, 200),
        "{name}: pts"
    );
    assert_eq!(frames[0].color(), frames[1].color(), "{name}: colour");
    assert_eq!(frames[1].color().matrix_coefficients, 5, "{name}: matrix");
    assert!(frames[1].color().full_range, "{name}: range");
    assert_eq!(
        plane_bytes(&frames[0]),
        plane_bytes(&frames[1]),
        "{name}: bytes"
    );
}

/// A superframe of `frames`, with four-byte sizes in its index.
fn superframe(frames: &[&[u8]]) -> Vec<u8> {
    let marker = 0xc0 | (3 << 3) | (frames.len() as u8 - 1);
    let mut chunk = frames.concat();
    chunk.push(marker);
    for frame in frames {
        chunk.extend((frame.len() as u32).to_le_bytes());
    }
    chunk.push(marker);
    chunk
}

/// libvpx returns the last shown frame of a chunk, and so does the adapter.
#[test]
fn a_superframe_gives_its_last_shown_frame() {
    let name = "vp9_8bit";
    let Some(stream) = load(name) else {
        return;
    };
    let chunk = superframe(&[first_sample(&stream), &[SHOW_SLOT_0]]);
    let frames = decode_all(&mut open(), vec![(chunk, 40)], name);
    assert_eq!(frames.len(), 1, "{name}: frame count");
    assert_eq!(frames[0].pts(), 40, "{name}: pts");
}

/// `send` refuses the first frame by name, and the refusal stays.
fn refuses(name: &'static str, profile: u8, bit_depth: u8, x: bool, y: bool) {
    let Some(stream) = load(name) else {
        return;
    };
    let expected = DecodeError::Vp9Layout {
        decoder: "rusty_vp9",
        profile,
        bit_depth,
        subsampling_x: x,
        subsampling_y: y,
    };
    let mut decoder = open();
    let first = first_sample(&stream);
    assert_eq!(
        decoder.send(first, 0),
        Err(expected.clone()),
        "{name}: send"
    );
    assert_eq!(
        decoder.send(first, 1),
        Err(expected.clone()),
        "{name}: second send"
    );
    assert_eq!(decoder.flush(), Err(expected), "{name}: flush");
    assert!(
        decoder.receive().is_none(),
        "{name}: a frame after a refusal"
    );
}

#[test]
fn twelve_bits_are_refused() {
    refuses("vp9_12bit", 2, 12, true, true);
}

#[test]
fn profile_1_is_refused() {
    refuses("vp9_444", 1, 8, false, false);
}

#[test]
fn profile_3_is_refused() {
    refuses("vp9_422_10bit", 3, 10, true, false);
}

#[test]
fn calls_out_of_order_are_refused() {
    let mut decoder = open();
    assert_eq!(
        decoder.send(&[], 5),
        Err(DecodeError::EmptyPacket { pts: 5 })
    );
    assert_eq!(decoder.flush(), Ok(()));
    assert_eq!(decoder.flush(), Ok(()));
    assert_eq!(
        decoder.send(&[SHOW_SLOT_0], 6),
        Err(DecodeError::AfterFlush { pts: 6 })
    );
    assert!(decoder.receive().is_none());
}

/// A wrong frame marker stops at the header parser, with an error and no panic.
#[test]
fn a_bad_header_is_a_bitstream_error() {
    assert!(matches!(
        open().send(&[0x00, 0x00], 1),
        Err(DecodeError::Bitstream(_))
    ));
}

/// A size that runs past the index leaves the decoder usable.
#[test]
fn a_bad_superframe_index_leaves_the_decoder_usable() {
    let name = "vp9_8bit";
    let Some(stream) = load(name) else {
        return;
    };
    let mut decoder = open();
    let marker = 0xc0;
    assert!(matches!(
        decoder.send(&[0x00, marker, 200, marker], 1),
        Err(DecodeError::Bitstream(_))
    ));
    assert_eq!(
        decoder.send(first_sample(&stream), 2),
        Ok(()),
        "{name}: send"
    );
    assert!(decoder.receive().is_some(), "{name}: no frame");
}

/// An inter frame before any key frame has no reference, so the parser refuses it.
#[test]
fn an_inter_frame_needs_a_key_frame_first() {
    // Frame marker 2, profile 0, show existing 0, non-key frame, show frame 1,
    // error resilient 0, then reset_frame_context 0.
    let inter = [0x86, 0x00, 0x00, 0x00];
    assert!(matches!(
        open().send(&inter, 1),
        Err(DecodeError::Bitstream(_))
    ));
}
