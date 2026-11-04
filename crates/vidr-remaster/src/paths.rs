//! Path helpers shared by the request, provenance, and reuse modules.

use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io;
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
    fn resolve_follows_a_link_in_the_existing_prefix() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = fs::canonicalize(dir.path()).expect("canonicalize");
        fs::create_dir(root.join("real")).expect("mkdir");
        std::os::unix::fs::symlink(root.join("real"), root.join("link")).expect("symlink");
        let resolved = resolve(&root.join("link").join("missing.mp4")).expect("resolve");
        assert_eq!(resolved, root.join("real").join("missing.mp4"));
    }
}
