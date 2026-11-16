//! A Matroska pixel crop applied after decode, against ffmpeg.
//!
//! ffmpeg applies a container crop through its `crop` filter without `exact`,
//! which rounds an odd offset or size to the chroma grid. vidr crops exactly,
//! as `av_frame_apply_cropping` does with `AV_FRAME_CROP_UNALIGNED`. So odd
//! crops compare against `crop=...:exact=1` on the uncropped decode, and an
//! even crop also compares against ffmpeg's own reading of the file.
//!
//! ffmpeg with libvpx-vp9 writes each source under `CARGO_TARGET_TMPDIR`. A
//! missing ffmpeg or encoder skips with a printed notice.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;

use containr::mkv::{Demuxer, MkvOptions, Muxer, PacketIn, TrackSpec};
use containr::{Packet, TrackKind};
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use vidr::Dimensions;
use vidr::decode::rusty_vp9::RustyVp9Decoder;
use vidr::decode::{CropRect, Cropped, Decoder};
use vidr::media::{Input, read_stream_facts};

use super::fixtures::load;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
const FRAMES: u32 = 4;

fn scratch(name: &str) -> PathBuf {
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("matroska_crop");
    std::fs::create_dir_all(&directory)
        .unwrap_or_else(|error| panic!("creating {}: {error}", directory.display()));
    directory.join(name)
}

/// Runs ffmpeg with `args`, or prints a notice and returns `None` when it does
/// not run or fails.
fn ffmpeg(args: &[&str], what: &str) -> Option<Vec<u8>> {
    let output = match Command::new("ffmpeg")
        .args(["-v", "error"])
        .args(args)
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            eprintln!("skipping the crop check: ffmpeg does not run: {error}");
            return None;
        }
    };
    if !output.status.success() {
        eprintln!(
            "skipping the crop check: ffmpeg cannot {what}: {}",
            String::from_utf8_lossy(&output.stderr).trim_end()
        );
        return None;
    }
    Some(output.stdout)
}

/// A VP9 WebM of the test picture, or `None` after a notice.
fn vp9_source(name: &str) -> Option<PathBuf> {
    let path = scratch(name);
    let picture = format!("testsrc2=size={WIDTH}x{HEIGHT}:rate=25");
    let frames = FRAMES.to_string();
    let target = path.to_str().expect("a UTF-8 path");
    ffmpeg(
        &[
            "-y",
            "-f",
            "lavfi",
            "-i",
            &picture,
            "-frames:v",
            &frames,
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libvpx-vp9",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            target,
        ],
        "encode VP9 with libvpx-vp9",
    )?;
    Some(path)
}

/// The first video track of `path` and its packets, read through containr.
fn packets(path: &Path) -> (Demuxer<std::io::BufReader<File>>, u32, Vec<Packet>) {
    let demuxer = Demuxer::open_path(path).unwrap();
    let track = demuxer
        .tracks()
        .iter()
        .find(|track| track.kind == TrackKind::Video)
        .expect("a video track")
        .id;
    let packets = demuxer
        .track_packets(track)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    (demuxer, track, packets)
}

/// Copies the video of `source` into `output` with `crop` as its
/// `PixelCrop` values, through containr's muxer.
fn remux_with_crop(source: &Path, output: &Path, crop: CropRect) {
    let (mut demuxer, track_id, packets) = packets(source);
    let track = demuxer.track(track_id).unwrap().clone();
    let mkv = track.matroska.expect("a Matroska track");
    let mut video = mkv.video.expect("a Video element");
    video.crop = [
        u64::from(crop.top),
        u64::from(crop.bottom),
        u64::from(crop.left),
        u64::from(crop.right),
    ];
    let mut spec = TrackSpec::new(TrackKind::Video, mkv.codec_id, track.timebase);
    spec.matroska.codec_private = mkv.codec_private;
    spec.matroska.video = Some(video);
    let mut muxer = Muxer::new(File::create(output).unwrap(), MkvOptions::default()).unwrap();
    let id = muxer.add_track(spec).unwrap();
    for packet in &packets {
        let data = demuxer.read_packet(packet).unwrap();
        let written = PacketIn {
            data: &data,
            pts: packet.pts,
            duration: packet.duration,
            sync: packet.sync,
            discardable: packet.discardable,
            discard_padding: None,
        };
        muxer.write_packet(id, &written).unwrap();
    }
    muxer.finish().unwrap();
}

/// Decodes `path` on `RustyVp9Decoder` wrapped in `Cropped`, with the crop its
/// decoder config carries. Returns the crop and every plane byte.
fn decode_cropped(path: &Path) -> (CropRect, Vec<u8>) {
    let input = Input::open(path).unwrap();
    let track = input.video_track().unwrap().clone();
    let config = input.decode_config(&track).unwrap();
    let crop = config.crop.expect("the config carries the crop");
    let size = crop.fit(Dimensions::new(WIDTH, HEIGHT).unwrap()).unwrap();
    assert_eq!(read_stream_facts(&input).unwrap().geometry.coded, size);

    let device = CpuDevice::new();
    let mut decoder = Cropped::new(RustyVp9Decoder::<CpuRuntime>::new(&device), crop);
    let (mut demuxer, _, packets) = packets(path);
    let mut frames = Vec::new();
    for packet in &packets {
        let data = demuxer.read_packet(packet).unwrap();
        decoder.send(&data, packet.pts).unwrap();
        while let Some(frame) = decoder.receive() {
            frames.push(frame);
        }
    }
    decoder.flush().unwrap();
    while let Some(frame) = decoder.receive() {
        frames.push(frame);
    }
    assert!(!frames.is_empty(), "{}: no frame", path.display());
    let mut bytes = Vec::new();
    for frame in &frames {
        assert_eq!(frame.display(), size);
        bytes.extend(frame.y().to_vec::<u8>());
        bytes.extend(frame.uv().to_vec::<u8>());
    }
    (crop, bytes)
}

/// ffmpeg's decode of `source` cut by its exact `crop` filter, in NV12.
fn ffmpeg_exact(source: &Path, crop: CropRect) -> Option<Vec<u8>> {
    let size = crop.fit(Dimensions::new(WIDTH, HEIGHT).unwrap()).unwrap();
    let filter = format!(
        "crop={}:{}:{}:{}:exact=1",
        size.width(),
        size.height(),
        crop.left,
        crop.top
    );
    let path = source.to_str().expect("a UTF-8 path");
    ffmpeg(
        &[
            "-flags",
            "unaligned",
            "-i",
            path,
            "-vf",
            &filter,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "nv12",
            "pipe:1",
        ],
        "decode and crop the source",
    )
}

const fn rect(top: u32, bottom: u32, left: u32, right: u32) -> CropRect {
    CropRect {
        top,
        bottom,
        left,
        right,
    }
}

#[test]
fn odd_crops_decode_like_ffmpegs_exact_crop() {
    let Some(source) = vp9_source("odd_source.webm") else {
        return;
    };
    for (index, crop) in [rect(5, 1, 3, 2), rect(0, 3, 1, 0), rect(1, 2, 4, 6)]
        .into_iter()
        .enumerate()
    {
        let cropped = scratch(&format!("odd_{index}.mkv"));
        remux_with_crop(&source, &cropped, crop);
        let (carried, decoded) = decode_cropped(&cropped);
        assert_eq!(carried, crop);
        let Some(expected) = ffmpeg_exact(&source, crop) else {
            return;
        };
        assert_eq!(decoded.len(), expected.len(), "{crop}: bytes");
        assert!(
            decoded == expected,
            "{crop}: the planes differ from ffmpeg's"
        );
    }
}

#[test]
fn an_even_crop_decodes_like_ffmpeg_reading_the_crop() {
    let Some(source) = vp9_source("even_source.webm") else {
        return;
    };
    let crop = rect(4, 2, 6, 2);
    let cropped = scratch("even.mkv");
    remux_with_crop(&source, &cropped, crop);
    let (_, decoded) = decode_cropped(&cropped);
    let path = cropped.to_str().expect("a UTF-8 path");
    let Some(read) = ffmpeg(
        &[
            "-flags",
            "unaligned",
            "-i",
            path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "nv12",
            "pipe:1",
        ],
        "decode the cropped file",
    ) else {
        return;
    };
    let uncropped = (WIDTH * HEIGHT * 3 / 2 * FRAMES) as usize;
    if read.len() == uncropped {
        eprintln!(
            "skipping the check against ffmpeg's reading: this ffmpeg ignores a Matroska crop. \
             ffmpeg 7.1 or newer applies it."
        );
        return;
    }
    assert!(decoded == read, "{crop}: the planes differ from ffmpeg's");
}

/// `crop_vp9.mkv`, which `gen_media.py` writes with `mkvmerge --cropping`.
#[test]
fn an_mkvmerge_crop_decodes_like_ffmpegs_exact_crop() {
    let Some(fixture) = load("crop_vp9.mkv") else {
        return;
    };
    let Some(uncropped) = load("vp9.webm") else {
        return;
    };
    let sides: Vec<u32> = fixture
        .meta("crop")
        .split(',')
        .map(|side| side.parse().unwrap())
        .collect();
    let crop = rect(sides[0], sides[1], sides[2], sides[3]);
    let (carried, decoded) = decode_cropped(&fixture.path);
    assert_eq!(carried, crop);
    let Some(expected) = ffmpeg_exact(&uncropped.path, crop) else {
        return;
    };
    assert!(
        decoded == expected,
        "{crop}: the planes differ from ffmpeg's"
    );
}
