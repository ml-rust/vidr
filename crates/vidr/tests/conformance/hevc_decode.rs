//! The JCT-VC HEVC_v1 zips through `HevcDecoder`, the public adapter over the core's
//! `StreamDecoder`, on numr's CPU backend.
//!
//! `scripts/conformance/fetch_hevc.sh` fetches the zips. Pass `--no-capture` to see
//! the table on a run that passes.
//!
//! | Variable | Meaning | Default |
//! | --- | --- | --- |
//! | `VIDR_HEVC_CONFORMANCE` | The directory of zips | Skip with a notice |
//! | `VIDR_HEVC_ONLY` | Run zips whose name holds this text | Every zip |
//! | `VIDR_HEVC_REPORT` | A file for one line per vector | No file |
//! | `VIDR_CONFORMANCE_JOBS` | Vectors decoded at once | Half the logical CPUs |
//! | `VIDR_HEVC_THREADS` | Worker threads per decoder. 1 decodes on the calling thread | 1 |
//!
//! Each picture checks against its hash SEI message, in the decoder. Each output
//! picture also feeds one MD5 of the whole stream, as the reference decoder writes its
//! file: Y, then U, then V, row by row, with no padding. An 8-bit sample is one byte. A
//! 10-bit sample is two little-endian bytes holding its value. No frame stays after its
//! hash. The tolerance is zero.
//!
//! A hash file's name often states neither side. A plain `.md5` that differs from the
//! bitstream's own MD5 counts as the decoded file's hash.
//!
//! | Status | Meaning |
//! | --- | --- |
//! | `pass` | The whole-stream MD5 matches, and no hash SEI message differs |
//! | `pass-sei` | The zip has no whole-stream MD5, and no hash SEI message differs |
//! | `unchecked` | The zip has no whole-stream MD5, and the stream no hash SEI message |
//! | `refused` | The decoder refuses the stream by name |
//! | `fail` | A hash differs, or the decoder returns an error or panics |
//!
//! Each report line holds name, status, frames, and first error text, split by tabs. The
//! lines come in completion order during a run, and in name order at its end.

use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use vidr::bitstream::{HevcNalType, annexb};
use vidr::decode::hevc::{HevcDecoder, HevcOptions};
use vidr::decode::{DecodedFrame, Decoder, PixelFormat};
use vidr::{BitstreamError, DecodeError};

use super::hevc::{BITSTREAMS, HashKind, md5_token, read_stream, selected_zips, unzip, zip_name};
use super::md5::{Md5, hex_digest};
use super::runner::{panic_text, run_each};
use crate::cpu_hevc::packets::nal_packets;

/// `payloadType` of the decoded picture hash SEI message (D.2.1).
const PICTURE_HASH_PAYLOAD: u8 = 132;

/// `VIDR_HEVC_THREADS`, or 1 when it is unset.
pub(super) fn hevc_threads() -> usize {
    std::env::var("VIDR_HEVC_THREADS").map_or(1, |text| {
        text.parse::<usize>()
            .ok()
            .filter(|&threads| threads > 0)
            .unwrap_or_else(|| {
                panic!("VIDR_HEVC_THREADS holds `{text}`. Set it to 1 or more, or unset it.")
            })
    })
}

/// What one vector did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Pass,
    PassSei,
    Unchecked,
    Refused,
    Fail,
}

impl Status {
    const fn label(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::PassSei => "pass-sei",
            Self::Unchecked => "unchecked",
            Self::Refused => "refused",
            Self::Fail => "fail",
        }
    }
}

/// One line of the table and the report.
#[derive(Debug)]
struct Row {
    name: String,
    status: Status,
    frames: usize,
    /// The first error text, or the refusal. Empty on a pass.
    detail: String,
}

impl Row {
    /// The report line, on one line whatever the error text holds.
    fn report_line(&self) -> String {
        let detail = self.detail.replace(['\t', '\n', '\r'], " ");
        format!(
            "{}\t{}\t{}\t{detail}",
            self.name,
            self.status.label(),
            self.frames
        )
    }
}

/// Why a decode stops early.
enum Stop {
    Refused(String),
    Fail(String),
}

/// The refusals of a stream outside what `HevcDecoder` takes.
fn is_refusal(error: &DecodeError) -> bool {
    match error {
        DecodeError::UnsupportedLayout { .. } | DecodeError::HevcTool { .. } => true,
        DecodeError::Bitstream(BitstreamError::InvalidField { allowed, .. }) => {
            allowed.contains("because this decoder") || allowed.contains("because vidr does not")
        }
        _ => false,
    }
}

fn stop_of(error: &DecodeError, step: &str) -> Stop {
    if is_refusal(error) {
        Stop::Refused(error.to_string())
    } else {
        Stop::Fail(format!("{step}: {error}"))
    }
}

/// True for a suffix SEI NAL unit of the base layer whose first message is a picture hash.
fn holds_picture_hash(nal: &[u8]) -> bool {
    let [first, second, payload, ..] = nal else {
        return false;
    };
    HevcNalType::from_header(*first) == HevcNalType::SEI_SUFFIX
        && first & 1 == 0
        && second >> 3 == 0
        && *payload == PICTURE_HASH_PAYLOAD
}

/// The whole-stream MD5 a hash file publishes. A line that names a `.yuv` file wins.
/// A file named for the bitstream states none. In a file with no such name, the first
/// digest wins whose line names no bitstream and that differs from the bitstream's own.
fn published_yuv_md5(kind: HashKind, text: &str, bitstream_md5: &str) -> Option<String> {
    let named = text
        .lines()
        .filter(|line| line.to_ascii_lowercase().contains("yuv"))
        .find_map(md5_token);
    if named.is_some() {
        return named;
    }
    match kind {
        HashKind::Bitstream => None,
        HashKind::Yuv => md5_token(text),
        HashKind::Unnamed => text
            .lines()
            .filter(|line| {
                let lower = line.to_ascii_lowercase();
                !BITSTREAMS.iter().any(|ending| lower.contains(ending))
            })
            .filter_map(md5_token)
            .find(|digest| digest != bitstream_md5),
    }
}

/// Takes frames from the decoder, feeds the MD5, and keeps none of them.
struct FrameSink {
    md5: Md5,
    u: Vec<u8>,
    v: Vec<u8>,
    frames: usize,
    pts: HashSet<i64>,
}

impl FrameSink {
    fn new() -> Self {
        Self {
            md5: Md5::new(),
            u: Vec::new(),
            v: Vec::new(),
            frames: 0,
            pts: HashSet::new(),
        }
    }

    fn drain(&mut self, decoder: &mut HevcDecoder<CpuRuntime>) -> Result<(), String> {
        while let Some(frame) = decoder.receive() {
            self.take(&frame)?;
        }
        Ok(())
    }

    /// Feeds one frame. numr's CPU backend holds samples, in rows with no padding.
    fn take(&mut self, frame: &DecodedFrame<CpuRuntime>) -> Result<(), String> {
        if !self.pts.insert(frame.pts()) {
            return Err(format!(
                "frame {}: pts {} repeats, so a picture holds the wrong packet's pts",
                self.frames,
                frame.pts()
            ));
        }
        let read = |plane: &numr::tensor::Tensor<CpuRuntime>| {
            plane
                .to_bytes()
                .map_err(|error| format!("reading a plane: {error}"))
        };
        let (mut y, uv) = (read(frame.y())?, read(frame.uv())?);
        let sample = frame.format().bytes_per_sample();
        self.u.clear();
        self.v.clear();
        for pair in uv.chunks_exact(2 * sample) {
            let (cb, cr) = pair.split_at(sample);
            self.u.extend_from_slice(cb);
            self.v.extend_from_slice(cr);
        }
        if frame.format() == PixelFormat::P010 {
            for plane in [&mut y, &mut self.u, &mut self.v] {
                for stored in plane.as_chunks_mut::<2>().0 {
                    *stored = (u16::from_le_bytes(*stored) >> 6).to_le_bytes();
                }
            }
        }
        self.md5.update(&y);
        self.md5.update(&self.u);
        self.md5.update(&self.v);
        self.frames += 1;
        Ok(())
    }
}

/// Sends every NAL unit, and drains the frames after each.
fn feed(stream: &[u8], sink: &mut FrameSink) -> Result<(), Stop> {
    let options = HevcOptions {
        threads: Some(hevc_threads()),
        check_picture_hash: true,
    };
    let mut decoder = HevcDecoder::<CpuRuntime>::with_options(&CpuDevice::new(), &[], options)
        .map_err(|error| stop_of(&error, "open"))?;
    let packets =
        nal_packets(stream).map_err(|error| Stop::Fail(format!("the bitstream: {error}")))?;
    for (index, (packet, pts)) in packets.enumerate() {
        let sent = decoder.send(&packet, pts);
        sink.drain(&mut decoder).map_err(Stop::Fail)?;
        sent.map_err(|error| stop_of(&error, &format!("NAL unit {index}")))?;
    }
    let flushed = decoder.flush();
    sink.drain(&mut decoder).map_err(Stop::Fail)?;
    flushed.map_err(|error| stop_of(&error, "flush"))
}

/// The bitstream, and the whole-stream MD5 with the member that states it.
type ZipStream = (Vec<u8>, Option<(String, String)>);

fn read_zip(zip: &Path) -> Result<ZipStream, String> {
    let (stream, hash) = read_stream(zip)?;
    let Some((hash, kind)) = hash else {
        return Ok((stream, None));
    };
    let text = unzip("-p", zip, Some(&hash))?;
    let published = published_yuv_md5(kind, &text, &hex_digest(&[&stream]));
    Ok((stream, published.map(|digest| (hash, digest))))
}

fn decode_zip(zip: &Path) -> Row {
    let name = zip_name(zip);
    let row = |status, frames, detail| Row {
        name: name.clone(),
        status,
        frames,
        detail,
    };
    let (stream, published) = match read_zip(zip) {
        Ok(read) => read,
        Err(reason) => return row(Status::Fail, 0, reason),
    };
    let mut sink = FrameSink::new();
    match feed(&stream, &mut sink) {
        Ok(()) => {}
        Err(Stop::Refused(reason)) => return row(Status::Refused, sink.frames, reason),
        Err(Stop::Fail(reason)) => return row(Status::Fail, sink.frames, reason),
    }
    let FrameSink { md5, frames, .. } = sink;
    if frames == 0 {
        return row(Status::Fail, 0, "the decoder gives no frame".to_owned());
    }
    let digest = md5.finish();
    if let Some((member, expected)) = published {
        if digest != expected {
            let detail = format!(
                "the {frames} frames have MD5 {digest}, and {member} states {expected}. No hash \
                 SEI message differs."
            );
            return row(Status::Fail, frames, detail);
        }
        return row(Status::Pass, frames, String::new());
    }
    let has_sei = annexb::split(&stream).is_ok_and(|mut units| units.any(holds_picture_hash));
    let status = if has_sei {
        Status::PassSei
    } else {
        Status::Unchecked
    };
    row(status, frames, String::new())
}

/// Decodes one zip. A panic in the decoder is a failure of this vector alone.
fn run_zip(zip: &Path, report: Option<&Mutex<File>>) -> Row {
    let decoded = catch_unwind(AssertUnwindSafe(|| decode_zip(zip)));
    let row = decoded.unwrap_or_else(|panic| Row {
        name: zip_name(zip),
        status: Status::Fail,
        frames: 0,
        detail: format!("panic: {}", panic_text(&*panic)),
    });
    if let Some(mut file) = report.and_then(|report| report.lock().ok()) {
        // The final rewrite reports a path it cannot write.
        let _ = writeln!(file, "{}", row.report_line());
    }
    row
}

#[test]
fn hevc_zips_decode_and_match() {
    let Some(zips) = selected_zips() else {
        return;
    };
    let report_path = std::env::var_os("VIDR_HEVC_REPORT").map(PathBuf::from);
    let report = report_path.as_ref().map(|path| {
        let file = File::create(path).unwrap_or_else(|error| {
            panic!(
                "creating {}: {error}. Set VIDR_HEVC_REPORT to a path you can write.",
                path.display()
            )
        });
        Mutex::new(file)
    });
    let rows = run_each(&zips, |zip| run_zip(zip, report.as_ref()));

    println!("| Vector | Status | Frames | Detail |");
    println!("| --- | --- | --- | --- |");
    let mut failed = Vec::new();
    for row in &rows {
        println!(
            "| {} | {} | {} | {} |",
            row.name,
            row.status.label(),
            row.frames,
            row.detail
        );
        if row.status == Status::Fail {
            failed.push(format!("{}: {}", row.name, row.detail));
        }
    }
    let count = |status: Status| rows.iter().filter(|row| row.status == status).count();
    println!(
        "{} vectors: {} pass, {} pass-sei, {} unchecked, {} refused by name, {} fail",
        rows.len(),
        count(Status::Pass),
        count(Status::PassSei),
        count(Status::Unchecked),
        count(Status::Refused),
        failed.len()
    );
    if let Some(path) = &report_path {
        let lines: String = rows.iter().map(|row| row.report_line() + "\n").collect();
        std::fs::write(path, lines)
            .unwrap_or_else(|error| panic!("writing {}: {error}", path.display()));
    }
    assert!(
        failed.is_empty(),
        "{} vectors fail:\n{}",
        failed.len(),
        failed.join("\n")
    );
}

#[test]
fn each_hash_file_form_gives_its_digest() {
    let bit = "ffc9e0afd09cb90dad7706b252f15dcd";
    let yuv = "277006085088cf68b9cafbbba0c19443";
    // `X.yuv.md5` with the bare digest in uppercase.
    assert_eq!(
        published_yuv_md5(HashKind::Yuv, &yuv.to_ascii_uppercase(), bit).as_deref(),
        Some(yuv)
    );
    // `X.md5` with the bare digest, which differs from the bitstream's.
    assert_eq!(
        published_yuv_md5(HashKind::Unnamed, yuv, bit).as_deref(),
        Some(yuv)
    );
    // `X.md5` holding the bitstream's own digest states no decoded hash.
    assert_eq!(published_yuv_md5(HashKind::Unnamed, bit, bit), None);
    // `X.bit.md5` states none.
    assert_eq!(published_yuv_md5(HashKind::Bitstream, bit, bit), None);
    // `X.md5` with a named file.
    let named = format!("{yuv} *BUMPING_A_ericsson.yuv\n");
    assert_eq!(
        published_yuv_md5(HashKind::Unnamed, &named, bit).as_deref(),
        Some(yuv)
    );
    // Two lines: the one that names the decoded file wins.
    let both = format!("{bit}  x.bit\n{yuv}  x.yuv\n");
    assert_eq!(
        published_yuv_md5(HashKind::Unnamed, &both, bit).as_deref(),
        Some(yuv)
    );
    assert_eq!(published_yuv_md5(HashKind::Unnamed, "none\n", bit), None);
    // MD5summer's comment lines, then a line that names the bitstream.
    let summer = format!("# Generated 4/16/2013\n{yuv} *x.bit\n");
    assert_eq!(published_yuv_md5(HashKind::Unnamed, &summer, bit), None);
}

#[test]
fn a_report_line_stays_on_one_line() {
    let row = Row {
        name: "AMP_A_Samsung_7".to_owned(),
        status: Status::Fail,
        frames: 3,
        detail: "first\nsecond\tthird".to_owned(),
    };
    assert_eq!(
        row.report_line(),
        "AMP_A_Samsung_7\tfail\t3\tfirst second third"
    );
}

#[test]
fn only_refusals_by_name_count_as_refused() {
    let tool = DecodeError::HevcTool {
        decoder: "HevcDecoder",
        field: "sps_3d_extension_flag",
        value: 1,
        supported: "0",
    };
    assert!(is_refusal(&tool));
    let corrupt = DecodeError::Software {
        decoder: "HevcDecoder",
        call: "StreamDecoder::decode_nal",
        input: "a NAL unit".to_owned(),
        reason: "the data ends".to_owned(),
    };
    assert!(!is_refusal(&corrupt));
}

#[test]
fn a_hash_sei_is_found_by_its_payload_type() {
    assert!(holds_picture_hash(&[0x50, 0x01, 132, 16]));
    // A prefix SEI, another payload type, another layer, and a short unit.
    assert!(!holds_picture_hash(&[0x4e, 0x01, 132, 16]));
    assert!(!holds_picture_hash(&[0x50, 0x01, 5, 16]));
    assert!(!holds_picture_hash(&[0x50, 0x09, 132, 16]));
    assert!(!holds_picture_hash(&[0x50, 0x01]));
}
