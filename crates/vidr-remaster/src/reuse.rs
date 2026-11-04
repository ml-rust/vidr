//! Reuse of an earlier full-length upscale, without running a model again.
//!
//! Probing a video needs a demuxer, so this module takes the probe as a
//! [`VideoProbe`] and does no container parsing itself.

use std::cmp::Reverse;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use vidr::{DURATION_TOLERANCE, Dimensions, Rational, SampleAspectRatio, TimeError};

use crate::paths::{expand_user, resolve};
use crate::provenance::{ProvenanceError, RecordedCodecs, check_record_for, record_path};
use crate::request::ModelId;

/// The facts of a video that decide whether an upscale matches its source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoFacts {
    /// Number of video frames.
    pub frames: u64,
    /// Total duration in seconds.
    pub duration: Rational,
    /// Display size of the upright picture.
    pub dimensions: Dimensions,
    /// Sample aspect ratio of the upright picture.
    pub sample_aspect_ratio: SampleAspectRatio,
}

/// Reads the [`VideoFacts`] of a video file.
pub trait VideoProbe {
    /// The error a probe returns. Its text goes into the reuse messages.
    type Error: fmt::Display;

    /// Reads the facts of the video at `path`.
    ///
    /// # Errors
    ///
    /// `Self::Error` when the file is not a readable video.
    fn probe(&mut self, path: &Path) -> Result<VideoFacts, Self::Error>;
}

/// A candidate whose facts do not match the source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatchError {
    /// The frame counts differ.
    FrameCount {
        /// The candidate's frame count.
        candidate: u64,
        /// The source's frame count.
        source: u64,
    },
    /// The durations differ by more than [`DURATION_TOLERANCE`].
    Duration {
        /// The candidate's duration.
        candidate: Rational,
        /// The source's duration.
        source: Rational,
    },
    /// An automatic candidate that is not exactly twice the source size.
    Dimensions,
    /// The display aspect ratios differ by more than two pixels of rounding.
    AspectRatio,
    /// The comparison overflowed exact arithmetic.
    Time(TimeError),
}

impl fmt::Display for MatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FrameCount { candidate, source } => {
                write!(f, "Frame count differs: {candidate}/{source}.")
            }
            Self::Duration { candidate, source } => write!(
                f,
                "Video duration differs: {}/{} seconds.",
                seconds(*candidate),
                seconds(*source)
            ),
            Self::Dimensions => f.write_str("Dimensions do not match the original 2\u{d7} output."),
            Self::AspectRatio => {
                f.write_str("Display aspect ratio differs from the original video.")
            }
            Self::Time(error) => write!(f, "Cannot compare the videos: {error}."),
        }
    }
}

impl std::error::Error for MatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Time(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TimeError> for MatchError {
    fn from(error: TimeError) -> Self {
        Self::Time(error)
    }
}

fn seconds(value: Rational) -> f64 {
    value.num() as f64 / value.den() as f64
}

fn display_aspect(facts: &VideoFacts) -> Result<Rational, TimeError> {
    Rational::new(
        i64::from(facts.dimensions.width()),
        u64::from(facts.dimensions.height()),
    )?
    .checked_mul(facts.sample_aspect_ratio.ratio())
}

/// Checks a candidate against the source.
///
/// With `automatic`, the candidate must be exactly twice the source size. An
/// explicit candidate can have any size with the same display aspect ratio.
///
/// # Errors
///
/// The first [`MatchError`]: frames, duration, size, then aspect ratio.
pub fn check_match(
    candidate: &VideoFacts,
    source: &VideoFacts,
    automatic: bool,
) -> Result<(), MatchError> {
    if candidate.frames != source.frames {
        return Err(MatchError::FrameCount {
            candidate: candidate.frames,
            source: source.frames,
        });
    }
    let difference = candidate
        .duration
        .checked_sub(source.duration)?
        .checked_abs()?;
    if difference > DURATION_TOLERANCE {
        return Err(MatchError::Duration {
            candidate: candidate.duration,
            source: source.duration,
        });
    }
    if automatic {
        let doubled = (
            u64::from(source.dimensions.width()) * 2,
            u64::from(source.dimensions.height()) * 2,
        );
        let actual = (
            u64::from(candidate.dimensions.width()),
            u64::from(candidate.dimensions.height()),
        );
        if actual != doubled {
            return Err(MatchError::Dimensions);
        }
    }
    let original = display_aspect(source)?;
    let enhanced = display_aspect(candidate)?;
    let smallest_side = [source.dimensions, candidate.dimensions]
        .into_iter()
        .flat_map(|dimensions| [dimensions.width(), dimensions.height()])
        .min()
        .unwrap_or(1);
    let tolerance = original
        .max(enhanced)
        .checked_mul(Rational::new(2, u64::from(smallest_side))?)?;
    if original.checked_sub(enhanced)?.checked_abs()? > tolerance {
        return Err(MatchError::AspectRatio);
    }
    Ok(())
}

/// Why `find_upscaled` passed over a candidate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SkipReason {
    /// The probe failed or the facts do not match the source.
    Incompatible(String),
    /// The candidate is older than the source.
    Older,
}

/// A candidate that `find_upscaled` passed over.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skipped {
    /// The resolved candidate path.
    pub candidate: PathBuf,
    /// Why it was passed over.
    pub reason: SkipReason,
}

impl fmt::Display for Skipped {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.reason {
            SkipReason::Incompatible(reason) => write!(
                f,
                "Skipping incompatible enhanced video {}: {reason}",
                self.candidate.display()
            ),
            SkipReason::Older => write!(
                f,
                "Skipping older enhanced video: {}",
                self.candidate.display()
            ),
        }
    }
}

/// The outcome of `find_upscaled`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reuse {
    /// The enhanced video to reuse, or `None` when the model must run.
    pub selected: Option<PathBuf>,
    /// Candidates passed over, in the order tried. The caller reports them.
    pub skipped: Vec<Skipped>,
}

/// An error selecting an enhanced video.
#[derive(Debug)]
pub enum ReuseError {
    /// The explicit enhanced input is not a file.
    ExplicitMissing {
        /// The resolved path.
        path: PathBuf,
    },
    /// The explicit enhanced input is the source or the output.
    ExplicitCollides {
        /// The resolved path.
        path: PathBuf,
    },
    /// The explicit enhanced input does not match the source.
    ExplicitMismatch {
        /// The resolved path.
        path: PathBuf,
        /// The probe or match error text.
        reason: String,
    },
    /// A matching candidate has no provenance record.
    NoRecord {
        /// The resolved candidate.
        path: PathBuf,
        /// The command-line options that resolve the error.
        action: String,
    },
    /// A matching candidate fails its provenance check.
    Unverified {
        /// The resolved candidate.
        path: PathBuf,
        /// The provenance error.
        error: ProvenanceError,
        /// The command-line options that resolve the error.
        action: String,
    },
    /// A source path with no file name.
    NoFileName {
        /// The path passed.
        path: PathBuf,
    },
    /// A path that cannot be resolved or a file whose time cannot be read.
    Io {
        /// The path passed.
        path: PathBuf,
        /// The operating-system error.
        error: io::Error,
    },
}

impl fmt::Display for ReuseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExplicitMissing { path } => write!(
                f,
                "Enhanced input does not exist: {}. Supply a readable video file.",
                path.display()
            ),
            Self::ExplicitCollides { path } => write!(
                f,
                "Enhanced input matches the source or output: {}. Choose separate files.",
                path.display()
            ),
            Self::ExplicitMismatch { path, reason } => write!(
                f,
                "Enhanced input {} cannot be reused: {reason} Choose the matching full-length upscale.",
                path.display()
            ),
            Self::NoRecord { path, action } => write!(
                f,
                "Matching enhanced video has no provenance record: {}. {action}",
                path.display()
            ),
            Self::Unverified {
                path,
                error,
                action,
            } => write!(
                f,
                "Enhanced video association cannot be checked for {}: {error} {action}",
                path.display()
            ),
            Self::NoFileName { path } => write!(
                f,
                "Source path {} has no file name. Supply a path to a video file.",
                path.display()
            ),
            Self::Io { path, error } => write!(
                f,
                "Cannot read {}: {error}. Check that the path exists and is readable.",
                path.display()
            ),
        }
    }
}

impl std::error::Error for ReuseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Unverified { error, .. } => Some(error),
            Self::Io { error, .. } => Some(error),
            _ => None,
        }
    }
}

fn io_error(path: &Path) -> impl FnOnce(io::Error) -> ReuseError + '_ {
    move |error| ReuseError::Io {
        path: path.to_path_buf(),
        error,
    }
}

fn modified(path: &Path) -> Result<SystemTime, ReuseError> {
    fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .map_err(io_error(path))
}

/// The candidate names beside the source, in the order the PoC lists them.
fn candidate_paths(source: &Path, model: ModelId) -> Result<Vec<PathBuf>, ReuseError> {
    let stem = source.file_stem().ok_or_else(|| ReuseError::NoFileName {
        path: source.to_path_buf(),
    })?;
    let mut paths = Vec::new();
    for kind in ["2x", "restored"] {
        for suffix in [".mp4", ".mkv"] {
            let mut name = stem.to_os_string();
            name.push(format!(".{model}.{kind}{suffix}"));
            paths.push(source.with_file_name(name));
        }
    }
    Ok(paths)
}

/// [`find_upscaled_for`] for a run that encodes with NVENC.
///
/// # Errors
///
/// As [`find_upscaled_for`].
pub fn find_upscaled<P: VideoProbe>(
    probe: &mut P,
    source: &Path,
    model: ModelId,
    source_facts: &VideoFacts,
    destination: &Path,
    explicit: Option<&Path>,
) -> Result<Reuse, ReuseError> {
    let hardware = RecordedCodecs::HARDWARE.encoder;
    find_upscaled_for(
        probe,
        source,
        model,
        source_facts,
        destination,
        explicit,
        hardware,
    )
}

/// Selects a matching full-length upscale, or checks an explicit enhanced input.
///
/// With `explicit`, the input must differ from the source and the output and
/// match the source by frames, duration, and aspect ratio. Any size is
/// accepted, and no provenance record is needed.
///
/// Without `explicit`, the candidates beside the source are
/// `<stem>.<model>.<2x|restored>.<mp4|mkv>`, tried newest first. A candidate
/// that does not match is skipped. A matching candidate must pass
/// [`check_record_for`] with `encoder`, this run's encoder. One older than the
/// source is skipped. The result lists every candidate passed over, so the
/// caller can report them.
///
/// # Errors
///
/// A [`ReuseError`] when the explicit input is refused, when a matching
/// candidate has no record or fails its record, or on a file system error.
pub fn find_upscaled_for<P: VideoProbe>(
    probe: &mut P,
    source: &Path,
    model: ModelId,
    source_facts: &VideoFacts,
    destination: &Path,
    explicit: Option<&Path>,
    encoder: &str,
) -> Result<Reuse, ReuseError> {
    let source = resolve(source).map_err(io_error(source))?;
    let destination = resolve(destination).map_err(io_error(destination))?;
    if let Some(explicit) = explicit {
        let candidate = resolve(&expand_user(explicit)).map_err(io_error(explicit))?;
        if !candidate.is_file() {
            return Err(ReuseError::ExplicitMissing { path: candidate });
        }
        if candidate == source || candidate == destination {
            return Err(ReuseError::ExplicitCollides { path: candidate });
        }
        let mismatch = |reason: String| ReuseError::ExplicitMismatch {
            path: candidate.clone(),
            reason,
        };
        let facts = probe
            .probe(&candidate)
            .map_err(|error| mismatch(error.to_string()))?;
        check_match(&facts, source_facts, false).map_err(|error| mismatch(error.to_string()))?;
        return Ok(Reuse {
            selected: Some(candidate),
            skipped: Vec::new(),
        });
    }

    let mut candidates = Vec::new();
    for path in candidate_paths(&source, model)? {
        if path.is_file() {
            candidates.push((modified(&path)?, path));
        }
    }
    candidates.sort_by_key(|(modified, _)| Reverse(*modified));

    let mut skipped = Vec::new();
    let mut source_digest: Option<String> = None;
    for (_, path) in candidates {
        let candidate = resolve(&path).map_err(io_error(&path))?;
        if candidate == source || candidate == destination {
            continue;
        }
        let incompatible = probe
            .probe(&candidate)
            .map_err(|error| error.to_string())
            .and_then(|facts| {
                check_match(&facts, source_facts, true).map_err(|error| error.to_string())
            });
        if let Err(reason) = incompatible {
            skipped.push(Skipped {
                candidate,
                reason: SkipReason::Incompatible(reason),
            });
            continue;
        }
        let action = format!(
            "Use --from-upscaled '{}' to select it explicitly, or --force-ai to run AI again.",
            candidate.display()
        );
        if !record_path(&candidate).is_file() {
            return Err(ReuseError::NoRecord {
                path: candidate,
                action,
            });
        }
        let checked = check_record_for(
            &source,
            &candidate,
            model,
            Some(encoder),
            source_digest.as_deref(),
        );
        match checked {
            Ok(digest) => source_digest = Some(digest),
            Err(error) => {
                return Err(ReuseError::Unverified {
                    path: candidate,
                    error,
                    action,
                });
            }
        }
        if modified(&candidate)? < modified(&source)? {
            skipped.push(Skipped {
                candidate,
                reason: SkipReason::Older,
            });
            continue;
        }
        return Ok(Reuse {
            selected: Some(candidate),
            skipped,
        });
    }
    Ok(Reuse {
        selected: None,
        skipped,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use crate::provenance::{record_codecs, record_upscaled};

    use super::*;

    fn facts(frames: u64, width: u32, height: u32) -> VideoFacts {
        VideoFacts {
            frames,
            duration: Rational::new(i64::try_from(frames).expect("frames"), 25).expect("duration"),
            dimensions: Dimensions::new(width, height).expect("dimensions"),
            sample_aspect_ratio: SampleAspectRatio::SQUARE,
        }
    }

    /// Facts by file name. A name without facts fails to probe.
    struct Probe(HashMap<String, VideoFacts>);

    impl VideoProbe for Probe {
        type Error = String;

        fn probe(&mut self, path: &Path) -> Result<VideoFacts, String> {
            let name = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            self.0
                .get(&name)
                .copied()
                .ok_or_else(|| format!("No moving-video stream in {}.", path.display()))
        }
    }

    struct Fixture {
        _dir: tempfile::TempDir,
        root: PathBuf,
        source: PathBuf,
        probe: Probe,
    }

    fn source_facts() -> VideoFacts {
        facts(8, 64, 36)
    }

    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = fs::canonicalize(dir.path()).expect("canonicalize");
        let source = root.join("source.mp4");
        fs::write(&source, b"source").expect("write source");
        let probe = Probe(HashMap::from([("source.mp4".to_owned(), source_facts())]));
        Fixture {
            _dir: dir,
            root,
            source,
            probe,
        }
    }

    impl Fixture {
        /// Creates an enhanced video beside the source with the given facts.
        fn enhanced(&mut self, name: &str, facts: VideoFacts) -> PathBuf {
            let path = self.root.join(name);
            fs::write(&path, format!("enhanced {name}")).expect("write enhanced");
            self.probe.0.insert(name.to_owned(), facts);
            path
        }

        fn record(&self, enhanced: &Path, model: ModelId) {
            record_upscaled(&self.source, enhanced, model).expect("record");
        }

        fn find(&mut self, model: ModelId, explicit: Option<&Path>) -> Result<Reuse, ReuseError> {
            let destination = self.root.join("final.mp4");
            find_upscaled(
                &mut self.probe,
                &self.source,
                model,
                &source_facts(),
                &destination,
                explicit,
            )
        }
    }

    fn set_modified(path: &Path, time: SystemTime) {
        fs::File::options()
            .write(true)
            .open(path)
            .and_then(|file| file.set_modified(time))
            .expect("set mtime");
    }

    fn source_time(f: &Fixture) -> SystemTime {
        modified(&f.source).expect("source mtime")
    }

    #[test]
    fn auto_reuses_matching_full_length_output() {
        let mut f = fixture();
        let existing = f.enhanced("source.purevision.2x.mp4", facts(8, 128, 72));
        f.record(&existing, ModelId::PureVision);
        let reuse = f.find(ModelId::PureVision, None).expect("found");
        assert_eq!(reuse.selected, Some(existing));
        assert!(reuse.skipped.is_empty());
    }

    #[test]
    fn auto_skips_a_short_preview() {
        let mut f = fixture();
        f.enhanced("source.fast.2x.mp4", facts(2, 128, 72));
        let reuse = f.find(ModelId::Fast, None).expect("searched");
        assert_eq!(reuse.selected, None);
        assert_eq!(reuse.skipped.len(), 1);
        assert_eq!(
            reuse.skipped[0].reason,
            SkipReason::Incompatible("Frame count differs: 2/8.".to_owned())
        );
        assert!(
            reuse.skipped[0]
                .to_string()
                .starts_with("Skipping incompatible enhanced video ")
        );
    }

    #[test]
    fn auto_skips_an_output_older_than_the_source() {
        let mut f = fixture();
        let existing = f.enhanced("source.fast.2x.mp4", facts(8, 128, 72));
        f.record(&existing, ModelId::Fast);
        set_modified(&existing, source_time(&f) - Duration::from_secs(1));
        let reuse = f.find(ModelId::Fast, None).expect("searched");
        assert_eq!(reuse.selected, None);
        assert_eq!(reuse.skipped.len(), 1);
        assert_eq!(reuse.skipped[0].reason, SkipReason::Older);
        assert!(
            reuse.skipped[0]
                .to_string()
                .starts_with("Skipping older enhanced video: ")
        );
    }

    #[test]
    fn unlabelled_matching_output_requires_explicit_selection() {
        let mut f = fixture();
        let existing = f.enhanced("source.purevision.2x.mp4", facts(8, 128, 72));
        let error = f.find(ModelId::PureVision, None).expect_err("refused");
        assert!(matches!(error, ReuseError::NoRecord { .. }));
        assert!(error.to_string().contains("--from-upscaled"));
        assert!(error.to_string().contains("--force-ai"));
        let reuse = f
            .find(ModelId::PureVision, Some(&existing))
            .expect("explicit");
        assert_eq!(reuse.selected, Some(existing));
    }

    #[test]
    fn same_stem_sources_do_not_share_automatic_output() {
        let mut f = fixture();
        let existing = f.enhanced("source.fast.2x.mp4", facts(8, 128, 72));
        f.record(&existing, ModelId::Fast);
        let other = f.root.join("sub");
        fs::create_dir(&other).expect("mkdir");
        let moved = other.join("source.mp4");
        fs::copy(&f.source, &moved).expect("copy");
        fs::copy(&existing, other.join("source.fast.2x.mp4")).expect("copy enhanced");
        fs::copy(
            record_path(&existing),
            record_path(&other.join("source.fast.2x.mp4")),
        )
        .expect("copy record");
        let destination = f.root.join("final.mp4");
        let error = find_upscaled(
            &mut f.probe,
            &moved,
            ModelId::Fast,
            &source_facts(),
            &destination,
            None,
        )
        .expect_err("refused");
        assert!(
            error.to_string().contains("Recorded source path differs"),
            "{error}"
        );
    }

    #[test]
    fn source_content_change_is_rejected_with_original_timestamp() {
        let mut f = fixture();
        let existing = f.enhanced("source.fast.2x.mp4", facts(8, 128, 72));
        f.record(&existing, ModelId::Fast);
        let before = source_time(&f);
        fs::write(&f.source, b"replacement content").expect("modify source");
        set_modified(&f.source, before);
        let error = f.find(ModelId::Fast, None).expect_err("refused");
        assert!(
            error.to_string().contains("Source content differs"),
            "{error}"
        );
    }

    #[test]
    fn a_change_of_encoder_invalidates_a_reuse_and_names_both_flags() {
        let mut f = fixture();
        let existing = f.enhanced("source.fast.2x.mp4", facts(8, 128, 72));
        let codecs = RecordedCodecs {
            decoder: "dav1d",
            encoder: "rav1r",
        };
        record_codecs(&f.source, &existing, ModelId::Fast, codecs).expect("record");
        let destination = f.root.join("final.mp4");
        let source = f.source.clone();
        let mut find = |encoder: &str| {
            find_upscaled_for(
                &mut f.probe,
                &source,
                ModelId::Fast,
                &source_facts(),
                &destination,
                None,
                encoder,
            )
        };
        let reuse = find("rav1r").expect("same encoder");
        assert_eq!(reuse.selected, Some(existing));
        let error = find("nvenc").expect_err("refused");
        let text = error.to_string();
        assert!(text.contains("Recorded encoder differs"), "{text}");
        assert!(text.contains("--from-upscaled"), "{text}");
        assert!(text.contains("--force-ai"), "{text}");
    }

    #[test]
    fn enhanced_content_change_is_rejected() {
        let mut f = fixture();
        let existing = f.enhanced("source.fast.2x.mp4", facts(8, 128, 72));
        f.record(&existing, ModelId::Fast);
        fs::write(&existing, b"changed enhancement").expect("modify enhanced");
        let error = f.find(ModelId::Fast, None).expect_err("refused");
        assert!(
            error.to_string().contains("Enhanced video content differs"),
            "{error}"
        );
    }

    #[test]
    fn newest_candidate_wins() {
        let mut f = fixture();
        let older = f.enhanced("source.fast.2x.mp4", facts(8, 128, 72));
        let newer = f.enhanced("source.fast.restored.mkv", facts(8, 128, 72));
        f.record(&older, ModelId::Fast);
        f.record(&newer, ModelId::Fast);
        let base = source_time(&f);
        set_modified(&older, base + Duration::from_secs(10));
        set_modified(&newer, base + Duration::from_secs(20));
        assert_eq!(
            f.find(ModelId::Fast, None).expect("found").selected,
            Some(newer)
        );
        set_modified(&older, base + Duration::from_secs(30));
        assert_eq!(
            f.find(ModelId::Fast, None).expect("found").selected,
            Some(older)
        );
    }

    #[test]
    fn an_incompatible_newest_candidate_falls_through_to_the_next() {
        let mut f = fixture();
        let good = f.enhanced("source.fast.2x.mp4", facts(8, 128, 72));
        let preview = f.enhanced("source.fast.restored.mp4", facts(2, 128, 72));
        f.record(&good, ModelId::Fast);
        let base = source_time(&f);
        set_modified(&good, base + Duration::from_secs(10));
        set_modified(&preview, base + Duration::from_secs(20));
        let reuse = f.find(ModelId::Fast, None).expect("found");
        assert_eq!(reuse.selected, Some(good));
        assert_eq!(reuse.skipped.len(), 1);
        assert_eq!(reuse.skipped[0].candidate, preview);
    }

    #[test]
    fn a_candidate_for_another_model_is_not_listed() {
        let mut f = fixture();
        f.enhanced("source.fast.2x.mp4", facts(8, 128, 72));
        let reuse = f.find(ModelId::PureVision, None).expect("searched");
        assert_eq!(
            reuse,
            Reuse {
                selected: None,
                skipped: Vec::new()
            }
        );
    }

    #[test]
    fn a_candidate_equal_to_the_destination_is_ignored() {
        let mut f = fixture();
        let existing = f.enhanced("source.fast.2x.mp4", facts(8, 128, 72));
        let reuse = find_upscaled(
            &mut f.probe,
            &f.source,
            ModelId::Fast,
            &source_facts(),
            &existing,
            None,
        )
        .expect("searched");
        assert_eq!(
            reuse,
            Reuse {
                selected: None,
                skipped: Vec::new()
            }
        );
    }

    #[test]
    fn explicit_path_accepts_fourfold_enhancement() {
        let mut f = fixture();
        let existing = f.enhanced("custom.mp4", facts(8, 256, 144));
        let reuse = f.find(ModelId::Fast, Some(&existing)).expect("explicit");
        assert_eq!(reuse.selected, Some(existing));
    }

    #[test]
    fn explicit_mismatch_collision_and_missing_file_are_refused() {
        let mut f = fixture();
        let preview = f.enhanced("preview.mp4", facts(2, 128, 72));
        let error = f.find(ModelId::Fast, Some(&preview)).expect_err("refused");
        assert!(matches!(error, ReuseError::ExplicitMismatch { .. }));
        assert!(error.to_string().contains("Frame count differs"));
        assert!(
            error
                .to_string()
                .ends_with("Choose the matching full-length upscale.")
        );

        let source = f.source.clone();
        let error = f.find(ModelId::Fast, Some(&source)).expect_err("refused");
        assert!(matches!(error, ReuseError::ExplicitCollides { .. }));

        let missing = f.root.join("absent.mp4");
        let error = f.find(ModelId::Fast, Some(&missing)).expect_err("refused");
        assert!(matches!(error, ReuseError::ExplicitMissing { .. }));
    }

    #[test]
    fn match_checks_frames_duration_size_and_aspect() {
        let source = facts(8, 64, 36);
        assert_eq!(check_match(&facts(8, 128, 72), &source, true), Ok(()));
        assert_eq!(
            check_match(&facts(9, 128, 72), &source, true),
            Err(MatchError::FrameCount {
                candidate: 9,
                source: 8
            })
        );
        assert_eq!(
            check_match(&facts(8, 256, 144), &source, true),
            Err(MatchError::Dimensions)
        );
        assert_eq!(check_match(&facts(8, 256, 144), &source, false), Ok(()));
        assert_eq!(
            check_match(&facts(8, 128, 96), &source, false),
            Err(MatchError::AspectRatio)
        );
    }

    #[test]
    fn duration_tolerance_is_two_milliseconds() {
        let source = facts(8, 64, 36);
        let mut candidate = facts(8, 128, 72);
        candidate.duration = source
            .duration
            .checked_add(Rational::new(2, 1000).expect("ms"))
            .expect("sum");
        assert_eq!(check_match(&candidate, &source, true), Ok(()));
        candidate.duration = source
            .duration
            .checked_add(Rational::new(3, 1000).expect("ms"))
            .expect("sum");
        assert!(matches!(
            check_match(&candidate, &source, true),
            Err(MatchError::Duration { .. })
        ));
    }

    #[test]
    fn aspect_tolerates_rounding_and_counts_the_sample_aspect_ratio() {
        let source = facts(8, 1280, 720);
        assert_eq!(check_match(&facts(8, 854, 480), &source, false), Ok(()));
        let mut stretched = facts(8, 1280, 720);
        stretched.sample_aspect_ratio = SampleAspectRatio::new(2, 1);
        assert_eq!(
            check_match(&stretched, &source, false),
            Err(MatchError::AspectRatio)
        );
    }
}
