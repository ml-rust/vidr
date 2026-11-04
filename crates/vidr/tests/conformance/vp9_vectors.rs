//! libvpx's VP9 test vectors on disk, for each decoder's sweep.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::vector_files::vector_name;

/// Whether the vector's name states a layout the adapter refuses.
pub(super) fn refusal_expected(name: &str) -> bool {
    name.starts_with("vp91-") || name.starts_with("vp93-") || name.contains("12bit")
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
