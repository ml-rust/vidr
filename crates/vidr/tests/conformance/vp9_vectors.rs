//! libvpx's VP9 test vectors on disk, for each decoder's sweep.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Whether the vector's name states a layout the adapter refuses.
pub(super) fn refusal_expected(name: &str) -> bool {
    name.starts_with("vp91-") || name.starts_with("vp93-") || name.contains("12bit")
}

/// The vectors in `directory`: each file that has a `.md5` beside it, by name.
pub(super) fn vectors(directory: &Path, filter: Option<&str>) -> Vec<PathBuf> {
    let entries = std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("reading {}: {error}", directory.display()));
    let mut found: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension() == Some(OsStr::new("md5")))
        .map(|path| path.with_extension(""))
        .filter(|path| path.is_file())
        .filter(|path| filter.is_none_or(|text| vector_name(path).contains(text)))
        .collect();
    found.sort();
    found
}

pub(super) fn vector_name(path: &Path) -> String {
    path.file_name()
        .map_or_else(String::new, |name| name.to_string_lossy().into_owned())
}

/// The hash of each frame: the first word of each line.
pub(super) fn expected_hashes(vector: &Path) -> Result<Vec<String>, String> {
    let mut path = vector.as_os_str().to_owned();
    path.push(".md5");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| format!("reading {}: {error}", PathBuf::from(&path).display()))?;
    Ok(text
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .map(str::to_ascii_lowercase)
        .collect())
}

/// The Matroska file of a vector: the WebM file itself, or an IVF file's remux
/// under `work`. `None` means ffmpeg did not run.
pub(super) fn matroska_of(vector: &Path, work: &Path) -> Result<Option<PathBuf>, String> {
    if vector.extension() != Some(OsStr::new("ivf")) {
        return Ok(Some(vector.to_path_buf()));
    }
    std::fs::create_dir_all(work)
        .map_err(|error| format!("creating {}: {error}", work.display()))?;
    let remuxed = work.join(format!("{}.mkv", vector_name(vector)));
    let output = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(vector)
        .args(["-map", "0:v:0", "-c", "copy", "-f", "matroska"])
        .arg(&remuxed)
        .output();
    match output {
        Err(_) => Ok(None),
        Ok(output) if output.status.success() => Ok(Some(remuxed)),
        Ok(output) => Err(format!(
            "ffmpeg cannot remux the IVF file: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )),
    }
}
