//! Generated decode fixtures: one stream's samples, facts, and ffmpeg's decode.
//!
//! `tests/fixtures/gen_nvdec.py` writes them. A test whose fixture is missing
//! prints a notice and passes without checks.

use std::collections::HashMap;
use std::path::PathBuf;

use vidr::decode::DecodeCodec;

use crate::framing::unframe;

#[cfg(any(
    feature = "dav1d",
    feature = "openh264",
    feature = "openh264-cisco",
    feature = "cpu-vp9"
))]
pub(crate) mod cpu;

/// One generated stream.
pub(crate) struct Stream {
    pub(crate) name: &'static str,
    /// The `avcC`, `hvcC`, `av1C`, or `vpcC` payload. VP9 decoders ignore it.
    #[cfg_attr(
        not(any(
            feature = "nvdec",
            feature = "dav1d",
            feature = "openh264",
            feature = "openh264-cisco"
        )),
        allow(dead_code)
    )]
    pub(crate) config: Vec<u8>,
    /// Every MP4 sample, in decode order.
    pub(crate) samples: Vec<Vec<u8>>,
    /// The pts of each sample, in decode order.
    pub(crate) pts: Vec<i64>,
    /// The pts of each frame ffmpeg decodes, in display order.
    pub(crate) frames: Vec<i64>,
    meta: HashMap<String, String>,
    /// ffmpeg's `nv12` or `p010le` decode, or `None` for a stream the GPU
    /// or vidr refuses.
    pub(crate) yuv: Option<Vec<u8>>,
}

#[allow(dead_code, reason = "each decoder's tests read their own fields")]
impl Stream {
    /// A `NAME.meta` value.
    pub(crate) fn meta(&self, key: &str) -> &str {
        match self.meta.get(key) {
            Some(value) => value,
            None => panic!("{}.meta has no `{key}`", self.name),
        }
    }

    /// A `NAME.meta` value, or `None` when the key is absent.
    pub(crate) fn meta_opt(&self, key: &str) -> Option<&str> {
        self.meta.get(key).map(String::as_str)
    }

    /// A numeric `NAME.meta` value.
    pub(crate) fn number(&self, key: &str) -> u32 {
        let value = self.meta(key);
        value
            .parse()
            .unwrap_or_else(|error| panic!("{}.meta `{key}={value}`: {error}", self.name))
    }

    /// The stream's codec.
    pub(crate) fn codec(&self) -> DecodeCodec {
        match self.meta("codec") {
            "h264" => DecodeCodec::H264,
            "hevc" => DecodeCodec::Hevc,
            "av1" => DecodeCodec::Av1,
            "vp9" => DecodeCodec::Vp9,
            other => panic!("{}.meta names codec `{other}`", self.name),
        }
    }
}

fn directory() -> PathBuf {
    std::env::var_os("VIDR_NVDEC_FIXTURES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/nvdec"),
        PathBuf::from,
    )
}

fn lines(text: &str, name: &str) -> Vec<i64> {
    text.lines()
        .map(|line| {
            line.parse()
                .unwrap_or_else(|error| panic!("{name}: pts `{line}`: {error}"))
        })
        .collect()
}

/// Loads stream `name`, or prints a notice and returns `None` when a file
/// is missing.
pub(crate) fn load(name: &'static str) -> Option<Stream> {
    let dir = directory();
    let read = |extension: &str| std::fs::read(dir.join(format!("{name}.{extension}"))).ok();
    let text =
        |extension: &str| std::fs::read_to_string(dir.join(format!("{name}.{extension}"))).ok();
    let (Some(config), Some(samples), Some(pts), Some(frames), Some(meta)) = (
        read("config"),
        read("samples"),
        text("pts"),
        text("frames"),
        text("meta"),
    ) else {
        eprintln!(
            "skipping: decode fixture `{name}` is missing in {}. \
             Run `python3 crates/vidr/tests/fixtures/gen_nvdec.py`, \
             or set VIDR_NVDEC_FIXTURES to its output directory.",
            dir.display()
        );
        return None;
    };
    let meta = meta
        .lines()
        .filter_map(|line| line.split_once('='))
        .map(|(key, value)| (key.to_owned(), value.to_owned()))
        .collect();
    let stream = Stream {
        name,
        config,
        samples: unframe(&samples),
        pts: lines(&pts, name),
        frames: lines(&frames, name),
        meta,
        yuv: read("yuv"),
    };
    assert_eq!(
        stream.samples.len(),
        stream.pts.len(),
        "{name}: sample and pts counts differ. Regenerate the fixtures."
    );
    Some(stream)
}
