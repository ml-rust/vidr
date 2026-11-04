//! The output's temporary home, and the step that makes it final.
//!
//! A run writes inside a temporary directory beside the output, so the output
//! path holds no partial file at any moment. Dropping [`TempOutput`] removes the
//! directory and everything in it.
//!
//! `process::exit` skips that drop. So each live directory also sits in a registry,
//! and [`remove_temp_directories`] removes them before the command exits on a second
//! signal.

use std::ffi::OsString;
use std::fmt;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use tempfile::TempDir;

use crate::provenance::record_path;

/// The directories of every live [`TempOutput`] in this process.
static LIVE_DIRECTORIES: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

fn live_directories() -> MutexGuard<'static, Vec<PathBuf>> {
    // A `Vec` of paths has no partial state, so a poisoned lock is still correct.
    LIVE_DIRECTORIES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Removes every live temporary directory, best effort, for a caller about to end
/// the process without running destructors.
///
/// This function takes a lock, allocates, and walks directories. So it is not
/// async-signal-safe, and it must run on a normal thread. The `ctrlc` crate calls
/// its handler on such a thread. A file that a writer creates during the walk can
/// leave its directory behind. A later drop of the [`TempOutput`] does no harm.
pub fn remove_temp_directories() {
    for directory in live_directories().drain(..) {
        // Best effort: the process is ending, and no caller can act on an error.
        let _ = fs::remove_dir_all(directory);
    }
}

/// An error making or publishing the temporary output.
#[derive(Debug)]
#[non_exhaustive]
pub enum PublishError {
    /// The output path has no file name.
    NoFileName {
        /// The output path.
        path: PathBuf,
    },
    /// The temporary directory beside the output does not create.
    TempDirectory {
        /// The output's directory.
        directory: PathBuf,
        /// The operating-system error.
        error: io::Error,
    },
    /// The output appears during the run, and the request does not overwrite.
    OutputExists {
        /// The output path.
        path: PathBuf,
    },
    /// The hard link or the rename fails.
    Move {
        /// The temporary file.
        from: PathBuf,
        /// The output path.
        to: PathBuf,
        /// The operating-system error.
        error: io::Error,
    },
}

impl fmt::Display for PublishError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoFileName { path } => write!(
                f,
                "Output path {} has no file name. Choose an output file.",
                path.display()
            ),
            Self::TempDirectory { directory, error } => write!(
                f,
                "Cannot create a temporary directory in {}: {error}. Check the directory \
                 permissions and free space.",
                directory.display()
            ),
            Self::OutputExists { path } => write!(
                f,
                "Output already exists: {}. Choose another filename or pass --overwrite.",
                path.display()
            ),
            Self::Move { from, to, error } => write!(
                f,
                "Cannot move {} to {}: {error}. Check the directory permissions, then run again.",
                from.display(),
                to.display()
            ),
        }
    }
}

impl std::error::Error for PublishError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TempDirectory { error, .. } | Self::Move { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// The output is final, and its provenance sidecar did not land.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MissingSidecar {
    /// The final output.
    pub output: PathBuf,
    /// The path the sidecar did not reach.
    pub sidecar: PathBuf,
    /// The operating-system error.
    pub reason: String,
}

impl fmt::Display for MissingSidecar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "The output {} is final, and its provenance sidecar {} did not land: {}. Reuse \
             refuses this output until it has a sidecar. Check the directory permissions, then \
             run again with --overwrite.",
            self.output.display(),
            self.sidecar.display(),
            self.reason
        )
    }
}

/// How [`TempOutput::publish_with_sidecar`] ends once the output is final.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Published {
    /// The output and its sidecar are both in place.
    Complete,
    /// The output is in place, and the sidecar is not.
    WithoutSidecar(MissingSidecar),
}

/// A temporary directory beside the output, and the file a run writes in it.
#[derive(Debug)]
pub struct TempOutput {
    directory: TempDir,
    prepared: PathBuf,
    destination: PathBuf,
}

impl TempOutput {
    /// Creates `.<stem>-<random>/` beside `destination`.
    ///
    /// The file inside keeps the output's name, so its extension still names the
    /// container. This function does not create the file.
    ///
    /// # Errors
    ///
    /// [`PublishError::NoFileName`] and [`PublishError::TempDirectory`].
    pub fn beside(destination: &Path) -> Result<Self, PublishError> {
        let no_name = || PublishError::NoFileName {
            path: destination.to_path_buf(),
        };
        let name = destination.file_name().ok_or_else(no_name)?;
        let stem = destination.file_stem().ok_or_else(no_name)?;
        let parent = destination
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let mut prefix = OsString::from(".");
        prefix.push(stem);
        prefix.push("-");
        let directory = tempfile::Builder::new()
            .prefix(&prefix)
            .tempdir_in(parent)
            .map_err(|error| PublishError::TempDirectory {
                directory: parent.to_path_buf(),
                error,
            })?;
        let prepared = directory.path().join(name);
        live_directories().push(directory.path().to_path_buf());
        Ok(Self {
            directory,
            prepared,
            destination: destination.to_path_buf(),
        })
    }

    /// The path a run writes.
    pub fn prepared(&self) -> &Path {
        &self.prepared
    }

    /// The temporary directory.
    pub fn directory(&self) -> &Path {
        self.directory.path()
    }

    /// Makes the prepared file the output, then removes the temporary directory.
    ///
    /// With `overwrite`, a rename replaces any existing output in one step.
    /// Without it, a hard link fails on an existing output, so a file that
    /// appears during the run stays. On a file system with no hard links, an
    /// exclusive create keeps that rule, and the rename then fills it.
    ///
    /// # Errors
    ///
    /// [`PublishError::OutputExists`] and [`PublishError::Move`]. The temporary
    /// directory goes in both cases, so no partial file remains.
    pub fn publish(self, overwrite: bool) -> Result<(), PublishError> {
        self.publish_with(overwrite, |from, to| fs::hard_link(from, to))
    }

    /// Publishes as [`TempOutput::publish`] does, then renames `sidecar`, a record
    /// written inside the temporary directory, to the output's record path.
    ///
    /// The record is complete before this call, so only that rename can fail after
    /// the output is final. The rename stays inside one file system, since the
    /// temporary directory sits beside the output.
    ///
    /// # Errors
    ///
    /// [`PublishError::OutputExists`] and [`PublishError::Move`] when the output does
    /// not publish. A failed sidecar rename is no error: it returns
    /// [`Published::WithoutSidecar`], since the output is final.
    pub fn publish_with_sidecar(
        self,
        overwrite: bool,
        sidecar: &Path,
    ) -> Result<Published, PublishError> {
        self.move_prepared(overwrite, |from, to| fs::hard_link(from, to))?;
        let landing = record_path(&self.destination);
        Ok(match fs::rename(sidecar, &landing) {
            Ok(()) => Published::Complete,
            Err(error) => Published::WithoutSidecar(MissingSidecar {
                output: self.destination.clone(),
                sidecar: landing,
                reason: error.to_string(),
            }),
        })
    }

    fn publish_with(
        self,
        overwrite: bool,
        link: impl FnOnce(&Path, &Path) -> io::Result<()>,
    ) -> Result<(), PublishError> {
        self.move_prepared(overwrite, link)
    }

    fn move_prepared(
        &self,
        overwrite: bool,
        link: impl FnOnce(&Path, &Path) -> io::Result<()>,
    ) -> Result<(), PublishError> {
        let moved = if overwrite {
            fs::rename(&self.prepared, &self.destination)
        } else {
            link(&self.prepared, &self.destination).or_else(|error| {
                if links_are_unsupported(&error) {
                    exclusive_rename(&self.prepared, &self.destination)
                } else {
                    Err(error)
                }
            })
        };
        moved.map_err(|error| {
            if !overwrite && error.kind() == io::ErrorKind::AlreadyExists {
                PublishError::OutputExists {
                    path: self.destination.clone(),
                }
            } else {
                PublishError::Move {
                    from: self.prepared.clone(),
                    to: self.destination.clone(),
                    error,
                }
            }
        })
    }
}

impl Drop for TempOutput {
    fn drop(&mut self) {
        // The field drop that follows removes the directory itself.
        let directory = self.directory.path();
        live_directories().retain(|live| live != directory);
    }
}

/// True for the errors a file system with no hard links returns: `EPERM` on
/// FAT and exFAT, and `ENOTSUP`.
fn links_are_unsupported(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::PermissionDenied | io::ErrorKind::Unsupported
    )
}

/// Claims `to` with an exclusive create, then renames `from` over it. The claim
/// holds no video data, and a failed rename removes it.
fn exclusive_rename(from: &Path, to: &Path) -> io::Result<()> {
    File::options().write(true).create_new(true).open(to)?;
    fs::rename(from, to).inspect_err(|_| {
        let _ = fs::remove_file(to);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::{check_record, record_upscaled};
    use crate::request::ModelId;

    fn entries(directory: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(directory)
            .expect("read_dir")
            .map(|entry| {
                entry
                    .expect("entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        names.sort();
        names
    }

    fn prepared(root: &Path, name: &str, bytes: &[u8]) -> TempOutput {
        let output = TempOutput::beside(&root.join(name)).expect("temp output");
        fs::write(output.prepared(), bytes).expect("write");
        output
    }

    #[test]
    fn the_temporary_directory_sits_beside_the_output() {
        let root = tempfile::tempdir().expect("tempdir");
        let output = TempOutput::beside(&root.path().join("clip.restored.mp4")).expect("temp");
        assert_eq!(output.directory().parent(), Some(root.path()));
        let name = output
            .directory()
            .file_name()
            .expect("name")
            .to_string_lossy()
            .into_owned();
        assert!(name.starts_with(".clip.restored-"), "{name}");
        assert_eq!(
            output.prepared().file_name().expect("name"),
            "clip.restored.mp4"
        );
    }

    #[test]
    fn a_hard_link_publishes_and_the_directory_goes() {
        let root = tempfile::tempdir().expect("tempdir");
        prepared(root.path(), "out.mp4", b"new")
            .publish(false)
            .expect("publish");
        assert_eq!(fs::read(root.path().join("out.mp4")).expect("read"), b"new");
        assert_eq!(entries(root.path()), ["out.mp4"]);
    }

    #[test]
    fn an_existing_output_stays_without_overwrite() {
        let root = tempfile::tempdir().expect("tempdir");
        let output = prepared(root.path(), "out.mp4", b"new");
        fs::write(root.path().join("out.mp4"), b"old").expect("write");
        let error = output.publish(false).expect_err("refused");
        assert!(
            matches!(error, PublishError::OutputExists { .. }),
            "{error}"
        );
        assert!(error.to_string().contains("--overwrite"), "{error}");
        assert_eq!(fs::read(root.path().join("out.mp4")).expect("read"), b"old");
        assert_eq!(entries(root.path()), ["out.mp4"]);
    }

    #[test]
    fn overwrite_replaces_an_existing_output() {
        let root = tempfile::tempdir().expect("tempdir");
        let output = prepared(root.path(), "out.mp4", b"new");
        fs::write(root.path().join("out.mp4"), b"old").expect("write");
        output.publish(true).expect("publish");
        assert_eq!(fs::read(root.path().join("out.mp4")).expect("read"), b"new");
        assert_eq!(entries(root.path()), ["out.mp4"]);
    }

    #[test]
    fn dropping_the_temporary_output_leaves_no_partial_file() {
        let root = tempfile::tempdir().expect("tempdir");
        drop(prepared(root.path(), "out.mp4", b"partial"));
        assert!(entries(root.path()).is_empty());
    }

    #[test]
    fn a_file_system_with_no_hard_links_still_publishes() {
        let root = tempfile::tempdir().expect("tempdir");
        prepared(root.path(), "out.mp4", b"new")
            .publish_with(false, |_, _| {
                Err(io::Error::from(io::ErrorKind::PermissionDenied))
            })
            .expect("publish");
        assert_eq!(fs::read(root.path().join("out.mp4")).expect("read"), b"new");
        assert_eq!(entries(root.path()), ["out.mp4"]);
    }

    #[test]
    fn the_exclusive_path_also_keeps_an_existing_output() {
        let root = tempfile::tempdir().expect("tempdir");
        let output = prepared(root.path(), "out.mp4", b"new");
        fs::write(root.path().join("out.mp4"), b"old").expect("write");
        let error = output
            .publish_with(false, |_, _| {
                Err(io::Error::from(io::ErrorKind::Unsupported))
            })
            .expect_err("refused");
        assert!(
            matches!(error, PublishError::OutputExists { .. }),
            "{error}"
        );
        assert_eq!(fs::read(root.path().join("out.mp4")).expect("read"), b"old");
    }

    #[test]
    fn a_live_temporary_directory_is_registered_until_it_drops() {
        let root = tempfile::tempdir().expect("tempdir");
        let output = TempOutput::beside(&root.path().join("out.mp4")).expect("temp");
        let directory = output.directory().to_path_buf();
        assert!(live_directories().contains(&directory));
        drop(output);
        assert!(!live_directories().contains(&directory));
    }

    #[test]
    fn a_publish_also_leaves_the_registry() {
        let root = tempfile::tempdir().expect("tempdir");
        let output = prepared(root.path(), "out.mp4", b"new");
        let directory = output.directory().to_path_buf();
        output.publish(false).expect("publish");
        assert!(!live_directories().contains(&directory));
    }

    /// A source file and a prepared output with its record inside the temporary
    /// directory.
    fn with_record(root: &Path) -> (PathBuf, TempOutput, PathBuf) {
        let source = root.join("source.mp4");
        fs::write(&source, b"abc").expect("write");
        let output = prepared(root, "out.mp4", b"new");
        let record = record_upscaled(&source, output.prepared(), ModelId::Fast).expect("record");
        (source, output, record)
    }

    #[test]
    fn the_sidecar_lands_beside_the_output_and_checks_at_the_final_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = fs::canonicalize(dir.path()).expect("canonicalize");
        let (source, output, record) = with_record(&root);
        assert_eq!(record.parent(), Some(output.directory()));
        let written = fs::read(&record).expect("read");
        let published = output
            .publish_with_sidecar(false, &record)
            .expect("publish");
        assert_eq!(published, Published::Complete);
        assert_eq!(
            entries(&root),
            ["out.mp4", "out.mp4.upscale.json", "source.mp4"]
        );
        let destination = root.join("out.mp4");
        check_record(&source, &destination, ModelId::Fast, None).expect("record checks");
        let landed = fs::read(record_path(&destination)).expect("read");
        assert_eq!(landed, written);
        // A record written after the publish holds the same bytes.
        let direct = record_upscaled(&source, &destination, ModelId::Fast).expect("record");
        assert_eq!(fs::read(direct).expect("read"), written);
    }

    #[test]
    fn a_sidecar_that_does_not_land_is_a_named_outcome_with_both_paths() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = fs::canonicalize(dir.path()).expect("canonicalize");
        let (_source, output, record) = with_record(&root);
        // A rename over a directory that holds a file fails.
        let landing = root.join("out.mp4.upscale.json");
        fs::create_dir(&landing).expect("mkdir");
        fs::write(landing.join("kept"), b"kept").expect("write");
        let published = output
            .publish_with_sidecar(false, &record)
            .expect("publish");
        let Published::WithoutSidecar(missing) = published else {
            panic!("the sidecar landed: {published:?}");
        };
        assert_eq!(missing.output, root.join("out.mp4"));
        assert_eq!(missing.sidecar, landing);
        let message = missing.to_string();
        assert!(message.contains(&missing.output.display().to_string()));
        assert!(message.contains(&missing.sidecar.display().to_string()));
        assert!(message.contains("--overwrite"), "{message}");
        assert_eq!(fs::read(root.join("out.mp4")).expect("read"), b"new");
        assert_eq!(
            entries(&root),
            ["out.mp4", "out.mp4.upscale.json", "source.mp4"]
        );
    }

    #[test]
    fn a_refused_output_moves_no_sidecar() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = fs::canonicalize(dir.path()).expect("canonicalize");
        let (_source, output, record) = with_record(&root);
        fs::write(root.join("out.mp4"), b"old").expect("write");
        let error = output
            .publish_with_sidecar(false, &record)
            .expect_err("refused");
        assert!(
            matches!(error, PublishError::OutputExists { .. }),
            "{error}"
        );
        assert_eq!(fs::read(root.join("out.mp4")).expect("read"), b"old");
        assert_eq!(entries(&root), ["out.mp4", "source.mp4"]);
    }

    #[test]
    fn a_failed_move_names_both_paths_and_leaves_no_file() {
        let root = tempfile::tempdir().expect("tempdir");
        let output = prepared(root.path(), "out.mp4", b"new");
        let error = output
            .publish_with(false, |_, _| Err(io::Error::other("disk gone")))
            .expect_err("refused");
        let message = error.to_string();
        assert!(message.contains("out.mp4"), "{message}");
        assert!(message.contains("disk gone"), "{message}");
        assert!(entries(root.path()).is_empty());
    }
}
