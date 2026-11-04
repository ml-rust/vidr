//! A remaster request: its fields, defaults, and the checks that run before any work.

use std::ffi::OsString;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::execution::{DecoderChoice, EncoderChoice, RuntimeChoice};
use crate::paths::{expand_user, resolve};
use crate::resolution::ResolutionMode;

/// Lowest encoder quality. NVENC refuses 0.
pub const MIN_QUALITY: u32 = 1;
/// Highest encoder quality.
pub const MAX_QUALITY: u32 = 51;
/// Encoder quality used when the request names none.
pub const DEFAULT_QUALITY: u32 = 24;

/// A restoration model, by name.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ModelId {
    /// The `fast` model.
    Fast,
    /// The `realesrgan` model.
    RealEsrgan,
    /// The `purevision` model.
    #[default]
    PureVision,
}

impl ModelId {
    /// Every model, in the order the help text lists them.
    pub const ALL: [Self; 3] = [Self::Fast, Self::RealEsrgan, Self::PureVision];

    /// The name used in file names, records, and the command line.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::RealEsrgan => "realesrgan",
            Self::PureVision => "purevision",
        }
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A model name that is not `fast`, `realesrgan`, or `purevision`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseModelError {
    name: String,
}

impl fmt::Display for ParseModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Unsupported model '{}'. Use fast, realesrgan, or purevision.",
            self.name
        )
    }
}

impl std::error::Error for ParseModelError {}

impl FromStr for ModelId {
    type Err = ParseModelError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|model| model.as_str() == text)
            .ok_or_else(|| ParseModelError {
                name: text.to_owned(),
            })
    }
}

/// The output video codec.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Codec {
    /// H.265.
    #[default]
    Hevc,
    /// H.264.
    H264,
    /// AV1.
    Av1,
}

impl Codec {
    /// The name used on the command line.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hevc => "hevc",
            Self::H264 => "h264",
            Self::Av1 => "av1",
        }
    }
}

impl fmt::Display for Codec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A codec name that is not `hevc`, `h264`, or `av1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseCodecError {
    name: String,
}

impl fmt::Display for ParseCodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Unsupported codec '{}'. Use hevc, h264, or av1.",
            self.name
        )
    }
}

impl std::error::Error for ParseCodecError {}

impl FromStr for Codec {
    type Err = ParseCodecError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "hevc" => Ok(Self::Hevc),
            "h264" => Ok(Self::H264),
            "av1" => Ok(Self::Av1),
            _ => Err(ParseCodecError {
                name: text.to_owned(),
            }),
        }
    }
}

/// An error in a remaster request.
#[derive(Debug)]
pub enum RequestError {
    /// The source is not a regular file.
    SourceMissing {
        /// The resolved source path.
        path: PathBuf,
    },
    /// The output resolves to the source.
    OutputIsSource,
    /// The output has an extension other than `.mp4` or `.mkv`.
    UnsupportedContainer {
        /// The extension with its dot, or empty when there is none.
        suffix: String,
    },
    /// The output exists and the request does not allow overwriting.
    OutputExists {
        /// The resolved output path.
        path: PathBuf,
    },
    /// The output exists and is not a regular file.
    OutputNotFile {
        /// The resolved output path.
        path: PathBuf,
    },
    /// The output's parent directory does not exist.
    OutputDirectoryMissing {
        /// The parent directory.
        path: PathBuf,
    },
    /// Quality 0, which NVENC refuses.
    QualityZero,
    /// A quality outside 1 to 51.
    QualityOutOfRange {
        /// The quality passed.
        quality: u32,
    },
    /// Both `--from-upscaled` and `--force-ai`.
    ExclusiveOptions,
    /// A source path with no file name.
    NoFileName {
        /// The path passed.
        path: PathBuf,
    },
    /// A path that cannot be resolved.
    Path {
        /// The path passed.
        path: PathBuf,
        /// The operating-system error.
        error: io::Error,
    },
}

impl fmt::Display for RequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceMissing { path } => write!(
                f,
                "Input video does not exist: {}. Supply a readable video file.",
                path.display()
            ),
            Self::OutputIsSource => {
                f.write_str("Output matches the source video. Choose a different output filename.")
            }
            Self::UnsupportedContainer { suffix } => write!(
                f,
                "Unsupported output container {suffix}. Use .mp4 or .mkv."
            ),
            Self::OutputExists { path } => write!(
                f,
                "Output already exists: {}. Choose another filename or pass --overwrite.",
                path.display()
            ),
            Self::OutputNotFile { path } => write!(
                f,
                "Output is not a regular file: {}. Choose another filename.",
                path.display()
            ),
            Self::OutputDirectoryMissing { path } => write!(
                f,
                "Output directory does not exist: {}. Create it or choose an existing directory.",
                path.display()
            ),
            Self::QualityZero => write!(
                f,
                "Encoder quality 0 is refused because NVENC rejects it. Use quality {MIN_QUALITY} to {MAX_QUALITY}."
            ),
            Self::QualityOutOfRange { quality } => write!(
                f,
                "Invalid encoder quality {quality}. Use quality {MIN_QUALITY} to {MAX_QUALITY}."
            ),
            Self::ExclusiveOptions => {
                f.write_str("Choose --from-upscaled or --force-ai, not both.")
            }
            Self::NoFileName { path } => write!(
                f,
                "Source path {} has no file name. Supply a path to a video file.",
                path.display()
            ),
            Self::Path { path, error } => write!(
                f,
                "Cannot resolve {}: {error}. Check the path and its permissions.",
                path.display()
            ),
        }
    }
}

impl std::error::Error for RequestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Path { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// The resolved paths of a request that passed [`Request::validate`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Validated {
    /// The resolved source file.
    pub source: PathBuf,
    /// The resolved output file. It need not exist yet.
    pub destination: PathBuf,
}

/// A remaster request as the command line states it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    /// The source video.
    pub source: PathBuf,
    /// The restoration model.
    pub model: ModelId,
    /// The output path, or `None` for the default name.
    pub output: Option<PathBuf>,
    /// Encoder quality, 1 to 51.
    pub quality: u32,
    /// Output codec.
    pub codec: Codec,
    /// The numr backend that holds the frames.
    pub runtime: RuntimeChoice,
    /// The decoder.
    pub decoder: DecoderChoice,
    /// The encoder.
    pub encoder: EncoderChoice,
    /// Cisco's OpenH264 binary, for a build with `openh264-cisco`.
    pub openh264_library: Option<PathBuf>,
    /// Replace an existing output.
    pub overwrite: bool,
    /// Output size mode.
    pub resolution: ResolutionMode,
    /// An enhanced video to reuse instead of running a model.
    pub from_upscaled: Option<PathBuf>,
    /// Run the model even when a matching upscale exists.
    pub force_ai: bool,
    /// The directory that holds the model directories. `None` reads
    /// `VIDR_MODELS_DIR`.
    pub models_dir: Option<PathBuf>,
}

impl Request {
    /// A request for `source` with every other field at its default.
    pub fn new(source: impl Into<PathBuf>) -> Self {
        Self {
            source: source.into(),
            model: ModelId::default(),
            output: None,
            quality: DEFAULT_QUALITY,
            codec: Codec::default(),
            runtime: RuntimeChoice::default(),
            decoder: DecoderChoice::default(),
            encoder: EncoderChoice::default(),
            openh264_library: None,
            overwrite: false,
            resolution: ResolutionMode::default(),
            from_upscaled: None,
            force_ai: false,
            models_dir: None,
        }
    }

    /// Checks the request against the file system and returns the resolved paths.
    ///
    /// # Errors
    ///
    /// A [`RequestError`] naming the first check that fails, in this order:
    /// source, output name, output state, output directory, quality, and
    /// the `--from-upscaled` and `--force-ai` pair.
    pub fn validate(&self) -> Result<Validated, RequestError> {
        let source = resolve_path(&self.source)?;
        if !source.is_file() {
            return Err(RequestError::SourceMissing { path: source });
        }
        let destination = output_path(&source, self.model, self.output.as_deref())?;
        let exists = destination.exists();
        if exists && !self.overwrite {
            return Err(RequestError::OutputExists { path: destination });
        }
        if exists && !destination.is_file() {
            return Err(RequestError::OutputNotFile { path: destination });
        }
        let parent = destination.parent().unwrap_or_else(|| Path::new("."));
        if !parent.is_dir() {
            return Err(RequestError::OutputDirectoryMissing {
                path: parent.to_path_buf(),
            });
        }
        match self.quality {
            0 => return Err(RequestError::QualityZero),
            quality if quality > MAX_QUALITY => {
                return Err(RequestError::QualityOutOfRange { quality });
            }
            _ => {}
        }
        if self.from_upscaled.is_some() && self.force_ai {
            return Err(RequestError::ExclusiveOptions);
        }
        Ok(Validated {
            source,
            destination,
        })
    }
}

fn resolve_path(path: &Path) -> Result<PathBuf, RequestError> {
    resolve(&expand_user(path)).map_err(|error| RequestError::Path {
        path: path.to_path_buf(),
        error,
    })
}

/// Returns the resolved output path.
///
/// Without `output`, the name is `<stem>.<model>.restored.mp4` beside the
/// source. An explicit `output` must end in `.mp4` or `.mkv` and differ from
/// the source.
///
/// # Errors
///
/// [`RequestError::OutputIsSource`] or [`RequestError::UnsupportedContainer`].
pub fn output_path(
    source: &Path,
    model: ModelId,
    output: Option<&Path>,
) -> Result<PathBuf, RequestError> {
    let source = resolve_path(source)?;
    let destination = match output {
        Some(path) if !path.as_os_str().is_empty() => resolve_path(path)?,
        _ => {
            let stem = source.file_stem().ok_or_else(|| RequestError::NoFileName {
                path: source.clone(),
            })?;
            let mut name: OsString = stem.to_os_string();
            name.push(".");
            name.push(model.as_str());
            name.push(".restored.mp4");
            source.with_file_name(name)
        }
    };
    if source == destination {
        return Err(RequestError::OutputIsSource);
    }
    let extension = destination
        .extension()
        .map(|extension| extension.to_string_lossy().into_owned());
    let supported = extension
        .as_deref()
        .is_some_and(|extension| matches!(extension.to_ascii_lowercase().as_str(), "mp4" | "mkv"));
    if !supported {
        let suffix = extension.map_or_else(String::new, |extension| format!(".{extension}"));
        return Err(RequestError::UnsupportedContainer { suffix });
    }
    Ok(destination)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    struct Fixture {
        _dir: tempfile::TempDir,
        root: PathBuf,
        source: PathBuf,
    }

    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = fs::canonicalize(dir.path()).expect("canonicalize");
        let source = root.join("clip.mp4");
        fs::write(&source, b"video").expect("write source");
        Fixture {
            _dir: dir,
            root,
            source,
        }
    }

    #[test]
    fn defaults_match_the_command_line() {
        let request = Request::new("clip.mp4");
        assert_eq!(request.model, ModelId::PureVision);
        assert_eq!(request.codec, Codec::Hevc);
        assert_eq!(request.quality, 24);
        assert_eq!(request.resolution, ResolutionMode::Auto);
        assert!(!request.overwrite && !request.force_ai);
        assert_eq!(request.output, None);
        assert_eq!(request.from_upscaled, None);
        assert_eq!(request.models_dir, None);
        assert_eq!(request.runtime, RuntimeChoice::Auto);
        assert_eq!(request.decoder, DecoderChoice::Auto);
        assert_eq!(request.encoder, EncoderChoice::Auto);
        assert_eq!(request.openh264_library, None);
    }

    #[test]
    fn default_output_name_uses_stem_and_model() {
        let f = fixture();
        let validated = Request::new(&f.source).validate().expect("valid");
        assert_eq!(validated.source, f.source);
        assert_eq!(
            validated.destination,
            f.root.join("clip.purevision.restored.mp4")
        );
        let path = output_path(&f.source, ModelId::Fast, None).expect("path");
        assert_eq!(path, f.root.join("clip.fast.restored.mp4"));
    }

    #[test]
    fn explicit_output_takes_mp4_or_mkv_in_any_case() {
        let f = fixture();
        for name in ["out.mp4", "out.mkv", "out.MKV"] {
            let path = output_path(&f.source, ModelId::Fast, Some(&f.root.join(name)));
            assert_eq!(path.expect("accepted"), f.root.join(name));
        }
    }

    #[test]
    fn unsupported_container_names_the_suffix() {
        let f = fixture();
        let error = output_path(&f.source, ModelId::Fast, Some(&f.root.join("out.avi")))
            .expect_err("refused");
        assert_eq!(
            error.to_string(),
            "Unsupported output container .avi. Use .mp4 or .mkv."
        );
        let error =
            output_path(&f.source, ModelId::Fast, Some(&f.root.join("out"))).expect_err("refused");
        assert!(matches!(
            error,
            RequestError::UnsupportedContainer { ref suffix } if suffix.is_empty()
        ));
    }

    #[test]
    fn output_equal_to_source_is_refused() {
        let f = fixture();
        let mut request = Request::new(&f.source);
        request.output = Some(f.source.clone());
        assert!(matches!(
            request.validate(),
            Err(RequestError::OutputIsSource)
        ));
    }

    #[test]
    fn missing_or_directory_source_is_refused() {
        let f = fixture();
        for path in [f.root.join("absent.mp4"), f.root.clone()] {
            let error = Request::new(&path).validate().expect_err("refused");
            assert!(
                matches!(error, RequestError::SourceMissing { .. }),
                "{error}"
            );
            assert!(error.to_string().starts_with("Input video does not exist:"));
        }
    }

    #[test]
    fn existing_output_needs_overwrite() {
        let f = fixture();
        let target = f.root.join("out.mp4");
        fs::write(&target, b"old").expect("write output");
        let mut request = Request::new(&f.source);
        request.output = Some(target.clone());
        assert!(matches!(
            request.validate(),
            Err(RequestError::OutputExists { .. })
        ));
        request.overwrite = true;
        assert_eq!(request.validate().expect("valid").destination, target);
    }

    #[test]
    fn existing_output_directory_is_refused_even_with_overwrite() {
        let f = fixture();
        let target = f.root.join("dir.mkv");
        fs::create_dir(&target).expect("mkdir");
        let mut request = Request::new(&f.source);
        request.output = Some(target);
        request.overwrite = true;
        assert!(matches!(
            request.validate(),
            Err(RequestError::OutputNotFile { .. })
        ));
    }

    #[test]
    fn missing_output_parent_is_refused() {
        let f = fixture();
        let mut request = Request::new(&f.source);
        request.output = Some(f.root.join("missing").join("out.mp4"));
        let error = request.validate().expect_err("refused");
        assert!(matches!(
            error,
            RequestError::OutputDirectoryMissing { ref path } if *path == f.root.join("missing")
        ));
    }

    #[test]
    fn quality_runs_from_1_to_51() {
        let f = fixture();
        let mut request = Request::new(&f.source);
        for quality in [1, 24, 51] {
            request.quality = quality;
            assert!(request.validate().is_ok(), "{quality}");
        }
        request.quality = 0;
        let error = request.validate().expect_err("refused");
        assert!(matches!(error, RequestError::QualityZero));
        assert!(error.to_string().contains("quality 0 is refused"));
        request.quality = 52;
        assert!(matches!(
            request.validate(),
            Err(RequestError::QualityOutOfRange { quality: 52 })
        ));
    }

    #[test]
    fn from_upscaled_and_force_ai_are_exclusive() {
        let f = fixture();
        let mut request = Request::new(&f.source);
        request.from_upscaled = Some(f.root.join("up.mp4"));
        request.force_ai = true;
        let error = request.validate().expect_err("refused");
        assert_eq!(
            error.to_string(),
            "Choose --from-upscaled or --force-ai, not both."
        );
    }

    #[test]
    fn names_parse_and_unknown_names_are_refused() {
        for model in ModelId::ALL {
            assert_eq!(model.as_str().parse::<ModelId>(), Ok(model));
        }
        assert!("tensorrt".parse::<ModelId>().is_err());
        assert_eq!("h264".parse::<Codec>(), Ok(Codec::H264));
        assert_eq!("av1".parse::<Codec>(), Ok(Codec::Av1));
        assert_eq!(
            "vp9".parse::<Codec>().expect_err("refused").to_string(),
            "Unsupported codec 'vp9'. Use hevc, h264, or av1."
        );
    }
}
