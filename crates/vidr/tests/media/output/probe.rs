//! ffprobe's and ffmpeg's view of a file.

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

/// One file as `ffprobe -of flat` lists it.
pub(crate) struct Probe {
    fields: HashMap<String, String>,
    path: std::path::PathBuf,
}

/// Probes `path`, or prints a notice and returns `None` when ffprobe is missing.
pub(crate) fn probe(path: &Path) -> Option<Probe> {
    let output = Command::new("ffprobe")
        .args(["-v", "error", "-show_streams", "-show_format"])
        .args(["-show_chapters", "-of", "flat"])
        .arg(path)
        .output();
    let Ok(output) = output else {
        eprintln!("skipping: ffprobe is not on PATH.");
        return None;
    };
    assert!(
        output.status.success(),
        "ffprobe refuses {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    let fields = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.split_once('='))
        .map(|(key, value)| (key.to_owned(), value.trim_matches('"').to_owned()))
        .collect();
    Some(Probe {
        fields,
        path: path.to_path_buf(),
    })
}

impl Probe {
    fn field(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(String::as_str)
    }

    /// The indexes of the streams of `codec_type`. An attached picture is not a
    /// video stream here.
    pub(crate) fn streams(&self, codec_type: &str) -> Vec<usize> {
        (0..)
            .map_while(|index| {
                let found = self.stream(index, "codec_type")?;
                Some((index, found.to_owned()))
            })
            .filter(|(index, found)| {
                found == codec_type && self.stream(*index, "disposition.attached_pic") != Some("1")
            })
            .map(|(index, _)| index)
            .collect()
    }

    /// A stream field, such as `codec_name`.
    pub(crate) fn stream(&self, index: usize, field: &str) -> Option<&str> {
        self.field(&format!("streams.stream.{index}.{field}"))
    }

    /// A stream tag, in any case: MP4 and Matroska spell keys differently.
    pub(crate) fn stream_tag(&self, index: usize, key: &str) -> Option<&str> {
        self.tag(&format!("streams.stream.{index}.tags."), key)
    }

    /// A file tag, in any case.
    pub(crate) fn format_tag(&self, key: &str) -> Option<&str> {
        self.tag("format.tags.", key)
    }

    fn tag(&self, prefix: &str, key: &str) -> Option<&str> {
        self.fields.iter().find_map(|(name, value)| {
            name.strip_prefix(prefix)
                .filter(|tag| tag.eq_ignore_ascii_case(key))
                .map(|_| value.as_str())
        })
    }

    /// The side-data rotation of a stream.
    pub(crate) fn rotation(&self, index: usize) -> Option<&str> {
        self.stream(index, "side_data_list.side_data.0.rotation")
    }

    /// Every chapter: its start in nanoseconds, and its title.
    pub(crate) fn chapters(&self) -> Vec<(i128, String)> {
        (0..)
            .map_while(|index| {
                let start = self.field(&format!("chapters.chapter.{index}.start"))?;
                let base = self.field(&format!("chapters.chapter.{index}.time_base"))?;
                let title = self.field(&format!("chapters.chapter.{index}.tags.title"))?;
                Some((to_nanos(start.parse().ok()?, base), title.to_owned()))
            })
            .collect()
    }

    /// A stream's packets, in file order: pts in nanoseconds, and size.
    pub(crate) fn packets(&self, index: usize) -> Vec<(i128, u64)> {
        let output = Command::new("ffprobe")
            .args(["-v", "error", "-select_streams"])
            .arg(index.to_string())
            .args(["-show_entries", "packet=pts,size", "-of", "csv=p=0"])
            .arg(&self.path)
            .output()
            .unwrap();
        let base = self.stream(index, "time_base").unwrap();
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|line| {
                // A packet with side data prints a trailing empty field.
                let mut fields = line.split(',');
                let (pts, size) = (fields.next()?, fields.next()?);
                Some((to_nanos(pts.parse().ok()?, base), size.parse().ok()?))
            })
            .collect()
    }

    /// A stream's pts in presentation order, in nanoseconds.
    pub(crate) fn sorted_pts(&self, index: usize) -> Vec<i128> {
        let mut pts: Vec<i128> = self.packets(index).iter().map(|&(pts, _)| pts).collect();
        pts.sort_unstable();
        pts
    }

    /// Decodes the video and audio, and returns what ffmpeg logs.
    pub(crate) fn decode_log(&self) -> String {
        let output = Command::new("ffmpeg")
            .args(["-nostdin", "-v", "error", "-i"])
            .arg(&self.path)
            .args(["-map", "0:v", "-map", "0:a?", "-f", "null", "-"])
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", self.path.display());
        String::from_utf8_lossy(&output.stderr).into_owned()
    }
}

/// `ticks` of the time base `NUM/DEN`, in nanoseconds, rounded down.
fn to_nanos(ticks: i128, base: &str) -> i128 {
    let (num, den) = base.split_once('/').unwrap();
    let num: i128 = num.parse().unwrap();
    let den: i128 = den.parse().unwrap();
    (ticks * num * 1_000_000_000).div_euclid(den)
}

/// Fails unless two lists match within `tolerance` nanoseconds.
pub(crate) fn assert_near(what: &str, source: &[i128], output: &[i128], tolerance: i128) {
    assert_eq!(source.len(), output.len(), "{what}: count");
    for (index, (from, to)) in source.iter().zip(output).enumerate() {
        assert!(
            (from - to).abs() <= tolerance,
            "{what}: entry {index} is {to} ns and the source has {from} ns"
        );
    }
}
