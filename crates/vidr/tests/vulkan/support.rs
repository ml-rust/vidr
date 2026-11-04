//! The device, full decodes, and word planes back as ffmpeg's bytes.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use numr::runtime::wgpu::WgpuRuntime;
use numr::tensor::Tensor;
use vidr::bitstream::{AvcConfig, HevcNalType, StartCode, ToAnnexB, annexb};
use vidr::decode::{DecodeCodec, DecodedFrame, Decoder, PixelFormat};
use vidr::vulkan::{
    DpbPreference, VulkanDecoder, VulkanDevice, VulkanOptions, vulkan_video_available,
};
use vidr::{BitstreamError, DecodeError};

use crate::decode_fixtures::Stream;

/// Concurrent use loses a WebGPU device, so these tests run one at a time.
static BACKEND: Mutex<()> = Mutex::new(());

/// The device with validation on when the layer exists, and the guard for this test.
/// Prints a notice and returns `None` with no device.
pub(crate) fn device() -> Option<(Arc<VulkanDevice>, MutexGuard<'static, ()>)> {
    let guard = BACKEND.lock().unwrap_or_else(PoisonError::into_inner);
    if let Err(error) = vulkan_video_available() {
        eprintln!("skipping: {error}");
        return None;
    }
    let options = VulkanOptions { validation: true };
    let device = match VulkanDevice::open_shared(options) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping: {error}");
            return None;
        }
    };
    if !device.validation_active() {
        eprintln!(
            "notice: VK_LAYER_KHRONOS_validation is absent, so this test checks no validation \
             errors. Install the Vulkan validation layers to check them."
        );
    }
    Some((device, guard))
}

/// Fails on any validation error so far.
pub(crate) fn assert_no_validation_errors(device: &VulkanDevice, name: &str) {
    let errors = device.validation_errors();
    assert!(errors.is_empty(), "{name}: validation errors: {errors:#?}");
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

/// Every sample as an Annex-B access unit with its pts.
pub(crate) fn annexb_units(stream: &Stream) -> Vec<(Vec<u8>, i64)> {
    let name = stream.name;
    let config =
        AvcConfig::parse(&stream.config).unwrap_or_else(|error| panic!("{name}: avcC: {error}"));
    let mut filter = ToAnnexB::h264(&config);
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

/// Sends every packet, receiving after each, then flushes and drains.
pub(crate) fn decode_all(
    decoder: &mut dyn Decoder<WgpuRuntime>,
    packets: impl IntoIterator<Item = (Vec<u8>, i64)>,
    name: &str,
) -> Vec<DecodedFrame<WgpuRuntime>> {
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

/// One stream through each DPB mode the driver lists, as `(mode, frames)`. A mode the driver
/// lacks prints a notice, and a run with no mode fails.
pub(crate) fn decode_in_each_mode(
    device: &Arc<VulkanDevice>,
    codec: DecodeCodec,
    config: &[u8],
    packets: &[(Vec<u8>, i64)],
    name: &str,
) -> Vec<(DpbPreference, Vec<DecodedFrame<WgpuRuntime>>)> {
    let mut runs = Vec::new();
    for preference in [DpbPreference::Coincide, DpbPreference::Distinct] {
        let opened = VulkanDecoder::with_dpb_preference(device, codec, config, preference);
        let mut decoder = match opened {
            Ok(decoder) => decoder,
            Err(error) if lacks_mode(&error) => {
                eprintln!("notice: {name}: {preference:?} does not run here: {error}");
                continue;
            }
            Err(error) => panic!("{name}: open: {error}"),
        };
        let mut frames = Vec::new();
        let mut lacking = false;
        for (packet, pts) in packets {
            match decoder.send(packet, *pts) {
                Ok(()) => {}
                // A session opens at the first picture, when `config` holds no sequence.
                Err(error) if frames.is_empty() && lacks_mode(&error) => {
                    eprintln!("notice: {name}: {preference:?} does not run here: {error}");
                    lacking = true;
                    break;
                }
                Err(error) => panic!("{name}: {preference:?}: send at pts {pts}: {error}"),
            }
            while let Some(frame) = decoder.receive() {
                frames.push(frame);
            }
        }
        if lacking {
            continue;
        }
        decoder
            .flush()
            .unwrap_or_else(|error| panic!("{name}: {preference:?}: flush: {error}"));
        while let Some(frame) = decoder.receive() {
            frames.push(frame);
        }
        runs.push((preference, frames));
    }
    assert!(!runs.is_empty(), "{name}: no DPB mode ran");
    runs
}

/// Whether an error states that the driver lacks a DPB mode, or the image usage it needs.
fn lacks_mode(error: &DecodeError) -> bool {
    matches!(
        error,
        DecodeError::VulkanRefused { what, .. }
            if what.contains("DPB mode") || what.contains("images with usage")
    )
}

/// Whether the device decodes `codec`. Prints a notice when it does not.
pub(crate) fn codec_present(device: &Arc<VulkanDevice>, codec: DecodeCodec) -> bool {
    match VulkanDecoder::new(device, codec, &[]) {
        Ok(_) => true,
        Err(error) => {
            eprintln!("skipping: {error}");
            false
        }
    }
}

/// The bytes of a word plane without what pads each row. `row_bytes` is the width in
/// bytes that `-f rawvideo` holds.
fn dense_rows(plane: &Tensor<WgpuRuntime>, row_bytes: usize, name: &str) -> Vec<u8> {
    let words = plane
        .to_bytes()
        .unwrap_or_else(|error| panic!("{name}: reading a plane: {error}"));
    let [rows, columns] = plane.shape() else {
        panic!("{name}: plane shape {:?}", plane.shape());
    };
    let stride = columns * 4;
    let mut dense = Vec::with_capacity(rows * row_bytes);
    for row in words.chunks_exact(stride) {
        dense.extend_from_slice(&row[..row_bytes]);
        assert!(
            row[row_bytes..].iter().all(|&byte| byte == 0),
            "{name}: nonzero bytes past a row's data"
        );
    }
    dense
}

/// One frame's dense luma and interleaved chroma bytes, in `nv12` or `p010le` form.
fn dense_planes(frame: &DecodedFrame<WgpuRuntime>, name: &str) -> (Vec<u8>, Vec<u8>) {
    let width = frame.display().width() as usize;
    let sample = frame.format().bytes_per_sample();
    (
        dense_rows(frame.y(), width * sample, name),
        dense_rows(frame.uv(), width.div_ceil(2) * 2 * sample, name),
    )
}

/// Each frame's bytes, in `-f rawvideo` form: `nv12` or `p010le`.
pub(crate) fn frame_bytes(frames: &[DecodedFrame<WgpuRuntime>], name: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    for frame in frames {
        let (y, uv) = dense_planes(frame, name);
        bytes.extend(y);
        bytes.extend(uv);
    }
    bytes
}

/// One frame's planes as `yuv420p` or `yuv420p10le` holds them: Y, then U, then V, with a
/// 10-bit sample in two little-endian bytes holding its value.
pub(crate) fn planar_planes(
    frame: &DecodedFrame<WgpuRuntime>,
    name: &str,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (mut y, uv) = dense_planes(frame, name);
    let sample = frame.format().bytes_per_sample();
    let mut u = Vec::with_capacity(uv.len() / 2);
    let mut v = Vec::with_capacity(uv.len() / 2);
    for pair in uv.chunks_exact(2 * sample) {
        let (cb, cr) = pair.split_at(sample);
        u.extend_from_slice(cb);
        v.extend_from_slice(cr);
    }
    if frame.format() == PixelFormat::P010 {
        for plane in [&mut y, &mut u, &mut v] {
            for stored in plane.as_chunks_mut::<2>().0.iter_mut() {
                let value = u16::from_le_bytes([stored[0], stored[1]]) >> 6;
                stored.copy_from_slice(&value.to_le_bytes());
            }
        }
    }
    (y, u, v)
}

/// Each frame's bytes in `yuv420p` or `yuv420p10le` form.
pub(crate) fn planar_bytes(frames: &[DecodedFrame<WgpuRuntime>], name: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    for frame in frames {
        let (y, u, v) = planar_planes(frame, name);
        bytes.extend(y);
        bytes.extend(u);
        bytes.extend(v);
    }
    bytes
}

/// An Annex-B HEVC stream as packets of one access unit each. A packet's pts counts its
/// pictures. A VPS, SPS, PPS, AUD, or prefix SEI after a picture's slices starts the next
/// unit. The first slice segment of the next picture does too (7.4.2.4.4).
pub(crate) fn hevc_units(stream: &[u8], name: &str) -> Vec<(Vec<u8>, i64)> {
    try_hevc_units(stream).unwrap_or_else(|error| panic!("{name}: split: {error}"))
}

/// As [`hevc_units`], with the error of a stream that does not start at a start code.
pub(crate) fn try_hevc_units(stream: &[u8]) -> Result<Vec<(Vec<u8>, i64)>, BitstreamError> {
    let units = annexb::split(stream)?;
    let mut packets: Vec<(Vec<u8>, i64)> = Vec::new();
    let mut current = Vec::new();
    let mut has_slice = false;
    for nal in units {
        // An end of sequence unit is two bytes long, and it still reaches the decoder.
        let [first, second, rest @ ..] = nal else {
            continue;
        };
        let kind = HevcNalType::from_header(*first);
        let base_layer = first & 1 == 0 && second >> 3 == 0;
        let slice = base_layer && matches!(kind.value(), 0..=9 | 16..=21);
        let starts_picture = slice && rest.first().is_some_and(|byte| byte & 0x80 != 0);
        let prefix = kind == HevcNalType::VPS
            || kind == HevcNalType::SPS
            || kind == HevcNalType::PPS
            || kind == HevcNalType::AUD
            || kind == HevcNalType::SEI_PREFIX;
        if has_slice && (starts_picture || prefix) {
            packets.push((std::mem::take(&mut current), packets.len() as i64));
            has_slice = false;
        }
        has_slice |= slice;
        current.extend_from_slice(StartCode::Four.bytes());
        current.extend_from_slice(nal);
    }
    if !current.is_empty() {
        packets.push((current, packets.len() as i64));
    }
    Ok(packets)
}

/// Checks `decoded` against `expected`, naming the first byte that differs.
pub(crate) fn assert_same_bytes(decoded: &[u8], expected: &[u8], name: &str) {
    assert_eq!(decoded.len(), expected.len(), "{name}: decoded bytes");
    if let Some(offset) = decoded.iter().zip(expected).position(|(a, b)| a != b) {
        panic!(
            "{name}: byte {offset} differs from ffmpeg's decode: {} against {}",
            decoded[offset], expected[offset]
        );
    }
}
