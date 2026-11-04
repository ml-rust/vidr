//! The run report: `report.json` in a directory for each run, under
//! `$XDG_STATE_HOME/vidr-remaster/upscale-<run id>/`.
//!
//! A run writes the report once, when it completes, stops, or fails. The write is
//! atomic, so no reader sees a partial report. Validation refusals come before the
//! report exists, so they write none.

use std::env;
use std::ffi::OsStr;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value};

use crate::catalog::licence_notice;
use crate::paths::write_atomic;
use crate::request::{Request, Validated};

/// The directory under the state home.
const APPLICATION: &str = "vidr-remaster";
/// The report's file name.
const REPORT_FILE: &str = "report.json";
/// The name start of the temporary file a report write uses.
const REPORT_TEMP_PREFIX: &str = ".report-";
/// The attempts to find a run directory no other run holds.
const DIRECTORY_ATTEMPTS: u32 = 16;

/// The last state of a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReportStatus {
    /// The output and its sidecar are written.
    Complete,
    /// The output is final, and its sidecar did not land.
    OutputWithoutSidecar,
    /// A stop request ended the run.
    Interrupted,
    /// An error ended the run.
    Blocked,
}

impl ReportStatus {
    /// The value of the report's `status` field.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::OutputWithoutSidecar => "output_without_sidecar",
            Self::Interrupted => "interrupted",
            Self::Blocked => "blocked",
        }
    }
}

/// An error creating or writing the run report.
#[derive(Debug)]
#[non_exhaustive]
pub enum ReportError {
    /// Neither `XDG_STATE_HOME` nor `HOME` names a state directory.
    NoStateHome,
    /// The run directory does not create.
    Directory {
        /// The directory.
        path: PathBuf,
        /// The operating-system error.
        error: io::Error,
    },
    /// The report does not serialize or does not write.
    Write {
        /// The report path.
        path: PathBuf,
        /// What went wrong.
        reason: String,
    },
}

impl fmt::Display for ReportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoStateHome => f.write_str(
                "XDG_STATE_HOME is not an absolute path and HOME is not set, so the run report \
                 has no directory. Set XDG_STATE_HOME to an absolute path.",
            ),
            Self::Directory { path, error } => write!(
                f,
                "Cannot create the run report directory {}: {error}. Set XDG_STATE_HOME to a \
                 writable directory.",
                path.display()
            ),
            Self::Write { path, reason } => write!(
                f,
                "Cannot write the run report {}: {reason}. Check the directory permissions and \
                 free space.",
                path.display()
            ),
        }
    }
}

impl std::error::Error for ReportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Directory { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// Returns the state home: an absolute `xdg`, else `home` joined with
/// `.local/state`. The XDG specification ignores a relative value.
///
/// # Errors
///
/// [`ReportError::NoStateHome`] when neither names a directory.
pub fn state_home(xdg: Option<&OsStr>, home: Option<&OsStr>) -> Result<PathBuf, ReportError> {
    if let Some(path) = xdg.map(Path::new).filter(|path| path.is_absolute()) {
        return Ok(path.to_path_buf());
    }
    home.filter(|value| !value.is_empty())
        .map(|value| Path::new(value).join(".local").join("state"))
        .ok_or(ReportError::NoStateHome)
}

/// Returns the state home from `XDG_STATE_HOME` and `HOME`.
///
/// # Errors
///
/// [`ReportError::NoStateHome`] when neither names a directory.
pub fn state_home_from_env() -> Result<PathBuf, ReportError> {
    let xdg = env::var_os("XDG_STATE_HOME");
    let home = env::var_os("HOME");
    state_home(xdg.as_deref(), home.as_deref())
}

/// Returns `YYYYMMDD-HHMMSS-ffffff` for `now`, in UTC. A time before 1970 gives
/// the start of 1970.
pub fn run_id(now: SystemTime) -> String {
    let since = now.duration_since(UNIX_EPOCH).unwrap_or_default();
    let seconds = since.as_secs();
    let days = seconds / 86_400;
    let rest = seconds % 86_400;
    let (year, month, day) = civil_date(days);
    format!(
        "{year:04}{month:02}{day:02}-{:02}{:02}{:02}-{:06}",
        rest / 3600,
        rest / 60 % 60,
        rest % 60,
        since.subsec_micros()
    )
}

/// The Gregorian date `days` after 1970-01-01, by Howard Hinnant's
/// `civil_from_days`, for days at or after the epoch.
fn civil_date(days: u64) -> (u64, u64, u64) {
    let shifted = days + 719_468;
    let era = shifted / 146_097;
    let day_of_era = shifted % 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + u64::from(month <= 2);
    (year, month, day)
}

/// The fields of one run's report, in the order the run sets them.
#[derive(Debug)]
pub struct RunReport {
    path: PathBuf,
    started: Instant,
    fields: Map<String, Value>,
}

impl RunReport {
    /// Creates the run directory under `state_home` and starts the fields with
    /// the status `running`, the request, and the model's licence notice.
    ///
    /// # Errors
    ///
    /// [`ReportError::Directory`] when the directory does not create.
    pub fn start(
        state_home: &Path,
        request: &Request,
        validated: &Validated,
    ) -> Result<Self, ReportError> {
        let directory = create_run_directory(state_home)?;
        let mut report = Self {
            path: directory.join(REPORT_FILE),
            started: Instant::now(),
            fields: Map::new(),
        };
        report.set("status", "running");
        report.set("input", validated.source.display().to_string());
        report.set("output", validated.destination.display().to_string());
        report.set("model", request.model.as_str());
        report.set("licence_notice", licence_notice(request.model));
        report.set("backend", "boostr on numr CUDA, F16");
        report.set("resolution", request.resolution.to_string());
        report.set("quality", request.quality);
        report.set("codec", request.codec.as_str());
        Ok(report)
    }

    /// Sets a field. A field set again keeps its place.
    pub fn set(&mut self, key: &str, value: impl Into<Value>) {
        self.fields.insert(key.to_owned(), value.into());
    }

    /// The report's path. No file exists there before [`RunReport::finish`].
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Sets the status, the error, and the elapsed time, then writes the file
    /// atomically: a temporary file in the same directory, a sync, then a rename.
    ///
    /// # Errors
    ///
    /// [`ReportError::Write`] when the file does not write. Any earlier report at
    /// this path stays, and no temporary file remains.
    pub fn finish(
        &mut self,
        status: ReportStatus,
        error: Option<&str>,
    ) -> Result<PathBuf, ReportError> {
        self.set("status", status.as_str());
        if let Some(error) = error {
            self.set("error", error);
        }
        let seconds = self.started.elapsed().as_secs_f64();
        self.set("wall_seconds", (seconds * 1000.0).round() / 1000.0);
        let write_error = |reason: String| ReportError::Write {
            path: self.path.clone(),
            reason,
        };
        let mut text = serde_json::to_string_pretty(&self.fields)
            .map_err(|error| write_error(error.to_string()))?;
        text.push('\n');
        write_atomic(&self.path, REPORT_TEMP_PREFIX, &text)
            .map_err(|error| write_error(error.to_string()))?;
        Ok(self.path.clone())
    }
}

/// Creates `<state home>/vidr-remaster/upscale-<run id>/`. A directory another
/// run holds gives a new id.
fn create_run_directory(state_home: &Path) -> Result<PathBuf, ReportError> {
    let parent = state_home.join(APPLICATION);
    fs::create_dir_all(&parent).map_err(|error| ReportError::Directory {
        path: parent.clone(),
        error,
    })?;
    let mut attempt = 0;
    loop {
        let path = parent.join(format!("upscale-{}", run_id(SystemTime::now())));
        attempt += 1;
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error)
                if error.kind() == io::ErrorKind::AlreadyExists && attempt < DIRECTORY_ATTEMPTS => {
            }
            Err(error) => return Err(ReportError::Directory { path, error }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::request::{Codec, ModelId};
    use crate::resolution::ResolutionMode;

    fn sample() -> (Request, Validated) {
        let mut request = Request::new("/videos/clip.mp4");
        request.quality = 26;
        request.codec = Codec::H264;
        request.resolution = ResolutionMode::Double;
        let validated = Validated {
            source: PathBuf::from("/videos/clip.mp4"),
            destination: PathBuf::from("/videos/clip.purevision.restored.mp4"),
        };
        (request, validated)
    }

    fn read(path: &Path) -> Value {
        let text = fs::read_to_string(path).expect("read report");
        assert!(text.ends_with("}\n"), "{text}");
        serde_json::from_str(&text).expect("parse report")
    }

    #[test]
    fn the_state_home_is_xdg_then_home() {
        let xdg = OsStr::new("/state");
        let home = OsStr::new("/home/user");
        assert_eq!(
            state_home(Some(xdg), Some(home)).expect("xdg"),
            Path::new("/state")
        );
        assert_eq!(
            state_home(None, Some(home)).expect("home"),
            Path::new("/home/user/.local/state")
        );
    }

    #[test]
    fn a_relative_xdg_value_is_ignored() {
        let home = OsStr::new("/home/user");
        assert_eq!(
            state_home(Some(OsStr::new("state")), Some(home)).expect("home"),
            Path::new("/home/user/.local/state")
        );
        let error = state_home(Some(OsStr::new("")), None).expect_err("refused");
        assert!(error.to_string().contains("XDG_STATE_HOME"), "{error}");
    }

    #[test]
    fn the_run_id_has_the_poc_form_in_utc() {
        assert_eq!(run_id(UNIX_EPOCH), "19700101-000000-000000");
        let moment = UNIX_EPOCH + Duration::new(1_000_000_000, 123_456_000);
        assert_eq!(run_id(moment), "20010909-014640-123456");
        let leap = UNIX_EPOCH + Duration::from_secs(1_709_164_800);
        assert_eq!(run_id(leap), "20240229-000000-000000");
    }

    #[test]
    fn a_completed_report_holds_the_request_and_its_facts_in_order() {
        let home = tempfile::tempdir().expect("tempdir");
        let (request, validated) = sample();
        let mut report = RunReport::start(home.path(), &request, &validated).expect("start");
        let directory = report.path().parent().expect("parent").to_path_buf();
        assert_eq!(
            directory.parent().expect("parent"),
            home.path().join("vidr-remaster")
        );
        let name = directory
            .file_name()
            .expect("name")
            .to_string_lossy()
            .into_owned();
        assert!(name.starts_with("upscale-"), "{name}");
        assert!(!report.path().exists());
        report.set("dropped_tracks", Value::Array(Vec::new()));
        report.set("max_rounding_error_ns", Some(500_000_u64));
        let path = report.finish(ReportStatus::Complete, None).expect("finish");
        let value = read(&path);
        assert_eq!(value["status"], "complete");
        assert_eq!(value["input"], "/videos/clip.mp4");
        assert_eq!(value["output"], "/videos/clip.purevision.restored.mp4");
        assert_eq!(value["model"], "purevision");
        assert_eq!(value["resolution"], "2x");
        assert_eq!(value["quality"], 26);
        assert_eq!(value["codec"], "h264");
        assert_eq!(value["max_rounding_error_ns"], 500_000);
        assert!(
            value["licence_notice"]
                .as_str()
                .expect("notice")
                .contains("CC BY-NC-SA 4.0")
        );
        assert!(value["wall_seconds"].as_f64().expect("seconds") >= 0.0);
        assert!(value.get("error").is_none());
        let keys: Vec<&str> = value
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys.first(), Some(&"status"));
        assert_eq!(keys.last(), Some(&"wall_seconds"));
    }

    #[test]
    fn a_failed_report_holds_the_error() {
        let home = tempfile::tempdir().expect("tempdir");
        let (request, validated) = sample();
        let mut report = RunReport::start(home.path(), &request, &validated).expect("start");
        let path = report
            .finish(ReportStatus::Blocked, Some("The scan fails."))
            .expect("finish");
        let value = read(&path);
        assert_eq!(value["status"], "blocked");
        assert_eq!(value["error"], "The scan fails.");
    }

    #[test]
    fn a_stopped_report_holds_no_error() {
        let home = tempfile::tempdir().expect("tempdir");
        let (mut request, validated) = sample();
        request.model = ModelId::Fast;
        let mut report = RunReport::start(home.path(), &request, &validated).expect("start");
        let path = report
            .finish(ReportStatus::Interrupted, None)
            .expect("finish");
        let value = read(&path);
        assert_eq!(value["status"], "interrupted");
        assert_eq!(value["licence_notice"], Value::Null);
        assert!(value.get("error").is_none());
    }

    #[test]
    fn the_report_arrives_by_rename_and_leaves_no_temporary_file() {
        let home = tempfile::tempdir().expect("tempdir");
        let (request, validated) = sample();
        let mut report = RunReport::start(home.path(), &request, &validated).expect("start");
        // A write in place changes every hard link of the file. A rename changes none.
        let linked = home.path().join("linked");
        fs::write(&linked, b"earlier").expect("write");
        fs::hard_link(&linked, report.path()).expect("link");
        let path = report.finish(ReportStatus::Complete, None).expect("finish");
        assert_eq!(fs::read(&linked).expect("read"), b"earlier");
        assert_eq!(read(&path)["status"], "complete");
        let names: Vec<String> = fs::read_dir(path.parent().expect("parent"))
            .expect("read_dir")
            .map(|entry| {
                entry
                    .expect("entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert_eq!(names, ["report.json"]);
    }

    #[test]
    fn a_failed_report_write_names_the_path_and_leaves_no_temporary_file() {
        let home = tempfile::tempdir().expect("tempdir");
        let (request, validated) = sample();
        let mut report = RunReport::start(home.path(), &request, &validated).expect("start");
        // A rename over a directory that holds a file fails.
        fs::create_dir(report.path()).expect("mkdir");
        fs::write(report.path().join("kept"), b"kept").expect("write");
        let error = report
            .finish(ReportStatus::Complete, None)
            .expect_err("refused");
        assert!(matches!(error, ReportError::Write { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains(&report.path().display().to_string()),
            "{error}"
        );
        assert_eq!(fs::read(report.path().join("kept")).expect("read"), b"kept");
        let entries = fs::read_dir(report.path().parent().expect("parent"))
            .expect("read_dir")
            .count();
        assert_eq!(entries, 1);
    }

    #[test]
    fn a_state_home_that_is_a_file_is_refused_by_name() {
        let home = tempfile::tempdir().expect("tempdir");
        let file = home.path().join("state");
        fs::write(&file, b"").expect("write");
        let (request, validated) = sample();
        let error = RunReport::start(&file, &request, &validated).expect_err("refused");
        assert!(matches!(error, ReportError::Directory { .. }), "{error}");
        assert!(error.to_string().contains("XDG_STATE_HOME"), "{error}");
    }

    #[test]
    fn a_run_directory_is_unique() {
        let home = tempfile::tempdir().expect("tempdir");
        let first = create_run_directory(home.path()).expect("first");
        let second = create_run_directory(home.path()).expect("second");
        assert_ne!(first, second);
    }
}
