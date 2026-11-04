//! `Rav1rEncoder` output decoded by the `dav1d` command line tool and by ffmpeg,
//! and its tags read back by ffprobe.
//!
//! The stream reaches each tool as Section 5 low-overhead OBUs: each packet's
//! bytes after a temporal delimiter. That is codec framing, so no container is
//! written. Files go under Cargo's test directory, inside `target/`.

use std::path::PathBuf;
use std::process::{Command, Output};

use vidr::SampleAspectRatio;
use vidr::decode::ChromaLocation;
use vidr::encode::{ColorFacts, EncodedPacket, PixelFormat};

use super::support::{CLIPS, Clip, FRAMES, check_psnr, encode};

/// A temporal delimiter OBU with a size field.
const TEMPORAL_DELIMITER: [u8; 2] = [0x12, 0x00];

fn write_stream(packets: &[EncodedPacket], name: &str) -> PathBuf {
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("rav1r");
    std::fs::create_dir_all(&directory)
        .unwrap_or_else(|error| panic!("creating {}: {error}", directory.display()));
    let path = directory.join(format!("{name}.obu"));
    let mut stream = Vec::new();
    for packet in packets {
        stream.extend_from_slice(&TEMPORAL_DELIMITER);
        stream.extend_from_slice(&packet.data);
    }
    std::fs::write(&path, stream)
        .unwrap_or_else(|error| panic!("writing {}: {error}", path.display()));
    path
}

/// Raw I420 or I010 bytes as frames of samples in Y, U, V order.
fn split_frames(clip: &Clip, raw: &[u8], name: &str) -> Vec<Vec<u16>> {
    let sample = clip.format.bytes_per_sample();
    let frame_bytes = clip.samples() * sample;
    assert_eq!(raw.len() % frame_bytes, 0, "{name}: raw bytes");
    raw.chunks_exact(frame_bytes)
        .map(|frame| match clip.format {
            PixelFormat::P010 => frame
                .as_chunks::<2>()
                .0
                .iter()
                .map(|bytes| u16::from_le_bytes(*bytes))
                .collect(),
            _ => frame.iter().map(|byte| u16::from(*byte)).collect(),
        })
        .collect()
}

/// Runs a tool, or prints a notice and returns `None` when it does not start.
fn run_tool(command: &mut Command, name: &str) -> Option<Output> {
    match command.output() {
        Ok(output) => Some(output),
        Err(error) => {
            eprintln!("skipping the {name} check: {error}");
            None
        }
    }
}

fn check_status(output: &Output, name: &str) {
    assert!(
        output.status.success(),
        "{name}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Whether ffmpeg has an AV1 software decoder. Its native `av1` decoder needs a
/// hardware accelerator.
fn ffmpeg_decodes_av1(name: &str) -> bool {
    let probe = Command::new("ffmpeg")
        .args(["-hide_banner", "-decoders"])
        .output();
    let Ok(probe) = probe else {
        eprintln!("skipping the {name} check: ffmpeg does not start");
        return false;
    };
    let listing = String::from_utf8_lossy(&probe.stdout);
    let found = listing
        .split_whitespace()
        .any(|word| word == "libdav1d" || word == "libaom-av1");
    if !found {
        eprintln!("skipping the {name} check: ffmpeg has no AV1 software decoder");
    }
    found
}

/// Each case's packets through the `dav1d` tool, above the floor.
fn through_dav1d(clip: &Clip, packets: &[EncodedPacket], name: &str) {
    let input = write_stream(packets, name);
    let output_path = input.with_extension("yuv");
    let mut command = Command::new("dav1d");
    command
        .args(["--quiet", "--filmgrain", "0", "-i"])
        .arg(&input)
        .arg("-o")
        .arg(&output_path);
    let Some(output) = run_tool(&mut command, name) else {
        return;
    };
    check_status(&output, name);
    let raw = std::fs::read(&output_path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", output_path.display()));
    check_psnr(clip, &split_frames(clip, &raw, name), name);
}

/// Each case's packets through ffmpeg, above the floor.
fn through_ffmpeg(clip: &Clip, packets: &[EncodedPacket], name: &str) {
    if !ffmpeg_decodes_av1(name) {
        return;
    }
    let input = write_stream(packets, name);
    let pixel_format = match clip.format {
        PixelFormat::P010 => "yuv420p10le",
        _ => "yuv420p",
    };
    let mut command = Command::new("ffmpeg");
    command
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(&input)
        .args(["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt"])
        .args([pixel_format, "-"]);
    let Some(output) = run_tool(&mut command, name) else {
        return;
    };
    check_status(&output, name);
    check_psnr(clip, &split_frames(clip, &output.stdout, name), name);
}

#[test]
fn the_dav1d_tool_decodes_each_depth_and_size() {
    for (index, clip) in CLIPS.iter().enumerate() {
        let name = format!("dav1d_tool_{index}");
        let (packets, _) = encode(clip, clip.config(), FRAMES, &name);
        through_dav1d(clip, &packets, &name);
    }
}

#[test]
fn ffmpeg_decodes_each_depth_and_size() {
    for (index, clip) in CLIPS.iter().enumerate() {
        let name = format!("ffmpeg_{index}");
        let (packets, _) = encode(clip, clip.config(), FRAMES, &name);
        through_ffmpeg(clip, &packets, &name);
    }
}

#[test]
fn reorder_decodes_in_each_tool() {
    let clip = &CLIPS[0];
    let mut config = clip.config();
    config.b_frames = 3;
    let name = "reorder_tools";
    let (packets, _) = encode(clip, config, FRAMES, name);
    through_dav1d(clip, &packets, "reorder_dav1d");
    through_ffmpeg(clip, &packets, "reorder_ffmpeg");
}

/// ffprobe's stream field `key`.
fn probe_field<'a>(text: &'a str, key: &str, name: &str) -> &'a str {
    text.lines()
        .filter_map(|line| line.split_once('='))
        .find_map(|(found, value)| (found == key).then_some(value))
        .unwrap_or_else(|| panic!("{name}: ffprobe reports no `{key}`: {text}"))
}

/// Encodes with the tags, and compares what ffprobe reads from the bitstream.
fn check_tags(
    clip: &Clip,
    color: ColorFacts,
    ratio: (u32, u32),
    expected: &[(&str, &str)],
    name: &str,
) {
    if !ffmpeg_decodes_av1(name) {
        return;
    }
    let mut config = clip.config();
    config.color = color;
    config.sample_aspect_ratio = SampleAspectRatio::new(ratio.0, ratio.1);
    let (packets, _) = encode(clip, config, 3, name);
    let input = write_stream(&packets, name);
    let mut command = Command::new("ffprobe");
    command
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
        .args(["stream", "-of", "default=noprint_wrappers=1"])
        .arg(&input);
    let Some(output) = run_tool(&mut command, name) else {
        return;
    };
    check_status(&output, name);
    let text = String::from_utf8_lossy(&output.stdout);
    for (key, value) in expected {
        assert_eq!(probe_field(&text, key, name), *value, "{name}: {key}");
    }
}

#[test]
fn ffprobe_reads_bt709_full_range_and_the_aspect_ratio() {
    check_tags(
        &CLIPS[0],
        ColorFacts {
            matrix_coefficients: 1,
            transfer_characteristics: 1,
            colour_primaries: 1,
            full_range: true,
            chroma_location: Some(ChromaLocation::Left),
        },
        (4, 3),
        &[
            ("pix_fmt", "yuv420p"),
            ("color_space", "bt709"),
            ("color_transfer", "bt709"),
            ("color_primaries", "bt709"),
            ("color_range", "pc"),
            ("chroma_location", "left"),
            ("sample_aspect_ratio", "4:3"),
        ],
        "ffprobe_bt709",
    );
}

#[test]
fn ffprobe_reads_ten_bits_and_bt2020_tags() {
    check_tags(
        &CLIPS[1],
        ColorFacts {
            matrix_coefficients: 9,
            transfer_characteristics: 16,
            colour_primaries: 9,
            full_range: false,
            chroma_location: None,
        },
        (1, 1),
        &[
            ("pix_fmt", "yuv420p10le"),
            ("color_space", "bt2020nc"),
            ("color_transfer", "smpte2084"),
            ("color_primaries", "bt2020"),
            ("color_range", "tv"),
            ("sample_aspect_ratio", "1:1"),
        ],
        "ffprobe_bt2020",
    );
}
