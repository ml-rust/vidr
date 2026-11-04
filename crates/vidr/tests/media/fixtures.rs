//! Generated media fixtures and their ffprobe facts.

use std::collections::HashMap;
use std::path::PathBuf;

use vidr::media::{Input, Track};

/// One generated file and its `FILE.meta` lines.
pub(crate) struct Fixture {
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    meta: HashMap<String, String>,
}

impl Fixture {
    /// A `FILE.meta` value.
    pub(crate) fn meta(&self, key: &str) -> &str {
        match self.meta.get(key) {
            Some(value) => value,
            None => panic!("{}.meta has no `{key}`", self.name),
        }
    }

    /// A `FILE.meta` value, or `None` when the key is absent.
    pub(crate) fn meta_opt(&self, key: &str) -> Option<&str> {
        self.meta.get(key).map(String::as_str)
    }

    /// A numeric `FILE.meta` value.
    pub(crate) fn number(&self, key: &str) -> i64 {
        let value = self.meta(key);
        value
            .parse()
            .unwrap_or_else(|error| panic!("{}.meta `{key}={value}`: {error}", self.name))
    }

    /// An H.273 code from `FILE.meta`.
    pub(crate) fn code(&self, key: &str) -> u8 {
        let value = self.meta(key);
        value
            .parse()
            .unwrap_or_else(|error| panic!("{}.meta `{key}={value}`: {error}", self.name))
    }
}

fn directory() -> PathBuf {
    std::env::var_os("VIDR_MEDIA_FIXTURES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/media"),
        PathBuf::from,
    )
}

/// The container `input` opened as: `mp4` or `mkv`.
pub(crate) fn container(input: &Input) -> &'static str {
    match input {
        #[cfg(feature = "mp4")]
        Input::Mp4 { .. } => "mp4",
        #[cfg(feature = "mkv")]
        Input::Mkv { .. } => "mkv",
        _ => panic!("vidr has a container this test does not name"),
    }
}

/// Loads fixture `name`, or prints a notice and returns `None` when the file
/// is missing or its container feature is off.
pub(crate) fn load(name: &str) -> Option<Fixture> {
    let dir = directory();
    let path = dir.join(name);
    let meta = std::fs::read_to_string(dir.join(format!("{name}.meta"))).ok();
    let (true, Some(meta)) = (path.is_file(), meta) else {
        eprintln!(
            "skipping: media fixture `{name}` is missing in {}. \
             Run `python3 crates/vidr/tests/fixtures/gen_media.py`, \
             or set VIDR_MEDIA_FIXTURES to its output directory.",
            dir.display()
        );
        return None;
    };
    let meta = meta
        .lines()
        .filter_map(|line| line.split_once('='))
        .map(|(key, value)| (key.to_owned(), value.to_owned()))
        .collect();
    let fixture = Fixture {
        name: name.to_owned(),
        path,
        meta,
    };
    let enabled = match fixture.meta("container") {
        "mp4" => cfg!(feature = "mp4"),
        "mkv" => cfg!(feature = "mkv"),
        other => panic!("{name}.meta names container `{other}`"),
    };
    if !enabled {
        eprintln!("skipping: media fixture `{name}` needs a container feature that is off.");
        return None;
    }
    Some(fixture)
}

/// Loads and opens fixture `name`, and checks the container the bytes hold.
pub(crate) fn open(name: &str) -> Option<(Fixture, Input)> {
    let fixture = load(name)?;
    let input = Input::open(&fixture.path).unwrap_or_else(|error| panic!("{name}: {error}"));
    assert_eq!(container(&input), fixture.meta("container"), "{name}");
    Some((fixture, input))
}

/// The video track of an open fixture.
pub(crate) fn video(fixture: &Fixture, input: &Input) -> Track {
    input
        .video_track()
        .unwrap_or_else(|error| panic!("{}: {error}", fixture.name))
        .clone()
}

/// The coded width a track states.
pub(crate) fn width(track: &Track) -> u64 {
    match &track.matroska {
        Some(mkv) => mkv.video.as_ref().map_or(0, |video| video.pixel_width),
        None => track.video().map_or(0, |video| u64::from(video.width)),
    }
}

/// Writes `bytes` to a file under cargo's test directory.
pub(crate) fn scratch_file(name: &str, bytes: &[u8]) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::write(&path, bytes)
        .unwrap_or_else(|error| panic!("writing {}: {error}", path.display()));
    path
}
