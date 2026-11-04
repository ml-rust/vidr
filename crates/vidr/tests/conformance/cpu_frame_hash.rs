//! One frame's hash in the layout of libvpx's and libaom's `.md5` files, on numr's CPU backend.

use numr::runtime::cpu::CpuRuntime;
use vidr::decode::{DecodedFrame, PixelFormat};

use super::md5::hex_digest;

/// One frame's hash. numr's CPU backend holds samples.
pub(super) fn frame_hash(frame: &DecodedFrame<CpuRuntime>) -> Result<String, String> {
    let read = |plane: &numr::tensor::Tensor<CpuRuntime>| {
        plane
            .to_bytes()
            .map_err(|error| format!("reading a plane: {error}"))
    };
    let (mut y, uv) = (read(frame.y())?, read(frame.uv())?);
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
            for stored in plane.as_chunks_mut::<2>().0 {
                *stored = (u16::from_le_bytes(*stored) >> 6).to_le_bytes();
            }
        }
    }
    Ok(hex_digest(&[&y, &u, &v]))
}
