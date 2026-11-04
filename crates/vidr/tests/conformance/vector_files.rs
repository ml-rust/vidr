//! Test vectors with a per-frame `.md5` file beside them, for libvpx's and libaom's sets.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

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
