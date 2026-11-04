//! Proves that [`Picture`] planes reach numr through `HostPlanes`. This test file
//! is the only one under `decode::cpu` that names a numr type.

use numr::runtime::cpu::{CpuDevice, CpuRuntime};

use super::dsp::sample::Sample;
use super::hevc::picture::{HostScratch, Picture};
use crate::decode::PixelFormat;
use crate::encode::EncodeFrame;
use crate::frame::{HostFrame, download_planes, upload_planes};

fn fill<S: Sample>(picture: &mut Picture<S>, value: fn(usize, usize, usize) -> S) {
    fill_rows(picture.luma_mut(), 0, value);
    fill_rows(picture.cb_mut(), 1, value);
    fill_rows(picture.cr_mut(), 2, value);
}

fn fill_rows<S: Sample>(
    plane: &mut super::hevc::picture::Plane<S>,
    index: usize,
    value: fn(usize, usize, usize) -> S,
) {
    for y in 0..plane.height() {
        for (x, sample) in plane.row_mut(y).unwrap().iter_mut().enumerate() {
            *sample = value(index, x, y);
        }
    }
}

/// Tight little-endian bytes of the plane's rows, as `HostFrame` holds them.
fn tight<S: Sample>(plane: &super::hevc::picture::Plane<S>) -> Vec<u8> {
    let mut bytes = Vec::new();
    for y in 0..plane.height() {
        for sample in plane.row(y).unwrap() {
            let value = sample.to_i32() as u16;
            bytes.extend_from_slice(&value.to_le_bytes()[..S::BYTES]);
        }
    }
    bytes
}

fn round_trip<S: Sample>(picture: &Picture<S>, format: PixelFormat) {
    let device = CpuDevice::new();
    let mut scratch = HostScratch::default();
    let planes = picture.host_planes(&mut scratch).unwrap();
    let (y, uv) = upload_planes::<CpuRuntime>(&device, planes, format).unwrap();
    let frame = EncodeFrame::new(&y, &uv);
    let mut out = HostFrame::new();
    download_planes(&frame, &mut out).unwrap();
    assert_eq!(out.format(), format);
    assert_eq!(out.y(), tight(picture.luma()));
    assert_eq!(out.u(), tight(picture.cb()));
    assert_eq!(out.v(), tight(picture.cr()));
}

#[test]
fn eight_bit_pictures_round_trip_through_i420() {
    let mut picture = Picture::<u8>::new(6, 4).unwrap();
    fill(&mut picture, |plane, x, y| (plane * 50 + y * 8 + x) as u8);
    round_trip(&picture, PixelFormat::Nv12);
}

#[test]
fn ten_bit_pictures_round_trip_through_i010() {
    let mut picture = Picture::<u16>::new(6, 4).unwrap();
    fill(&mut picture, |plane, x, y| (plane * 300 + y * 8 + x) as u16);
    round_trip(&picture, PixelFormat::P010);
}

#[test]
fn odd_size_pictures_round_trip() {
    let mut picture = Picture::<u8>::new(5, 3).unwrap();
    fill(&mut picture, |plane, x, y| (plane * 50 + y * 8 + x) as u8);
    round_trip(&picture, PixelFormat::Nv12);
}
