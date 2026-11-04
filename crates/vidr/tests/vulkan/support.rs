//! The device, full decodes, and word planes back as ffmpeg's bytes.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use numr::runtime::wgpu::WgpuRuntime;
use numr::tensor::Tensor;
use vidr::bitstream::{AvcConfig, ToAnnexB};
use vidr::decode::{DecodedFrame, Decoder};
use vidr::vulkan::{VulkanDevice, VulkanOptions, vulkan_video_available};

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

/// Each frame's NV12 bytes, in `-f rawvideo` form.
pub(crate) fn nv12_bytes(frames: &[DecodedFrame<WgpuRuntime>], name: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    for frame in frames {
        let width = frame.display().width() as usize;
        bytes.extend(dense_rows(frame.y(), width, name));
        bytes.extend(dense_rows(frame.uv(), width.div_ceil(2) * 2, name));
    }
    bytes
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
