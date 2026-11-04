//! Path helpers shared by the request, provenance, and reuse modules.

use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{self, Write as _};
use std::path::{Component, Path, PathBuf};

/// Replaces a leading `~` with the `HOME` directory. Other paths pass through.
pub(crate) fn expand_user(path: &Path) -> PathBuf {
    if path.components().next() == Some(Component::Normal(OsStr::new("~")))
        && let Some(home) = env::var_os("HOME")
    {
        let mut expanded = PathBuf::from(home);
        expanded.extend(path.components().skip(1));
        return expanded;
    }
    path.to_path_buf()
}

/// Makes a path absolute and resolves symbolic links in its existing prefix.
///
/// A missing tail is kept, with `..` applied. The path itself need not exist.
pub(crate) fn resolve(path: &Path) -> io::Result<PathBuf> {
    let mut base = std::path::absolute(path)?;
    let mut tail: Vec<OsString> = Vec::new();
    loop {
        match fs::canonicalize(&base) {
            Ok(mut real) => {
                for name in tail.iter().rev() {
                    if name.as_os_str() == OsStr::new("..") {
                        real.pop();
                    } else {
                        real.push(name);
                    }
                }
                return Ok(real);
            }
            Err(error) => {
                let name = match base.components().next_back() {
                    Some(Component::Normal(name)) => name.to_os_string(),
                    Some(Component::ParentDir) => OsString::from(".."),
                    _ => return Err(error),
                };
                tail.push(name);
                base.pop();
            }
        }
    }
}

/// Writes `text` to `destination` atomically: a temporary file in the same
/// directory, a sync, then a rename. A reader sees the old file or the whole new
/// one. A failure removes the temporary file and leaves `destination` as it was.
/// The file takes the mode `fs::write` gives a new file, under the process's umask.
pub(crate) fn write_atomic(destination: &Path, prefix: &str, text: &str) -> io::Result<()> {
    let directory = destination.parent().unwrap_or_else(|| Path::new("."));
    let mut builder = tempfile::Builder::new();
    builder.prefix(prefix).suffix(".tmp");
    #[cfg(unix)]
    builder.permissions(std::os::unix::fs::PermissionsExt::from_mode(0o666));
    let mut file = builder.tempfile_in(directory)?;
    file.write_all(text.as_bytes())?;
    file.flush()?;
    file.as_file().sync_all()?;
    file.persist(destination).map_err(|error| error.error)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_keeps_a_missing_tail() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = fs::canonicalize(dir.path()).expect("canonicalize");
        let resolved =
            resolve(&root.join("a").join("b").join("..").join("c.mp4")).expect("resolve");
        assert_eq!(resolved, root.join("a").join("c.mp4"));
    }

    #[cfg(unix)]
    #[test]
    fn an_atomic_write_takes_the_mode_of_a_plain_write() {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = tempfile::tempdir().expect("tempdir");
        let plain = dir.path().join("plain");
        let atomic = dir.path().join("atomic");
        fs::write(&plain, "text").expect("write");
        write_atomic(&atomic, ".atomic-", "text").expect("atomic write");
        let mode = |path: &Path| fs::metadata(path).expect("metadata").permissions().mode();
        assert_eq!(mode(&atomic), mode(&plain));
        assert_eq!(fs::read_to_string(&atomic).expect("read"), "text");
    }

    #[cfg(unix)]
    #[test]
    fn resolve_follows_a_link_in_the_existing_prefix() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = fs::canonicalize(dir.path()).expect("canonicalize");
        fs::create_dir(root.join("real")).expect("mkdir");
        std::os::unix::fs::symlink(root.join("real"), root.join("link")).expect("symlink");
        let resolved = resolve(&root.join("link").join("missing.mp4")).expect("resolve");
        assert_eq!(resolved, root.join("real").join("missing.mp4"));
    }
}
