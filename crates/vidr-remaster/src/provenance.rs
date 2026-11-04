//! Provenance records tie an enhanced video to the exact source content.
//!
//! The record is a JSON sidecar named `<enhanced file name>.upscale.json`, schema 1.

use std::ffi::OsString;
use std::fmt::{self, Write as _};
use std::fs::File;
use std::io::{self, Read, Write as _};
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::paths::resolve;
use crate::request::ModelId;

/// Largest record `check_record` reads, in bytes.
const MAX_RECORD_BYTES: u64 = 65536;
/// The only schema this release reads and writes.
const SCHEMA: i64 = 1;

/// An error recording or checking provenance.
#[derive(Debug)]
pub enum ProvenanceError {
    /// The enhanced output and the source are one file.
    SameFile,
    /// A path that is not valid UTF-8 cannot go into the JSON record.
    NonUtf8Path {
        /// The path passed.
        path: PathBuf,
    },
    /// A path that cannot be resolved or a file that cannot be hashed.
    File {
        /// The path passed.
        path: PathBuf,
        /// The operating-system error.
        error: io::Error,
    },
    /// The record cannot be written.
    Write {
        /// The record path.
        path: PathBuf,
        /// The operating-system error.
        error: io::Error,
    },
    /// The record cannot be read or is not JSON.
    Unreadable {
        /// The record path.
        path: PathBuf,
        /// What went wrong.
        reason: String,
    },
    /// The record is larger than 64 KiB.
    TooLarge {
        /// The record path.
        path: PathBuf,
    },
    /// The record is not an object with `"schema": 1`.
    UnsupportedSchema {
        /// The record path.
        path: PathBuf,
    },
    /// The recorded source path differs from the source.
    SourcePathDiffers {
        /// The record path.
        path: PathBuf,
    },
    /// The recorded model differs from the requested model.
    ModelDiffers {
        /// The record path.
        path: PathBuf,
    },
    /// The source content differs from the record.
    SourceContentDiffers {
        /// The record path.
        path: PathBuf,
    },
    /// The enhanced content differs from the record.
    EnhancedContentDiffers {
        /// The record path.
        path: PathBuf,
    },
}

impl fmt::Display for ProvenanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SameFile => f.write_str(
                "Enhanced output matches its source. Choose separate files before recording provenance.",
            ),
            Self::NonUtf8Path { path } => write!(
                f,
                "Path {} is not valid UTF-8, so the record cannot hold it. Rename the file.",
                path.display()
            ),
            Self::File { path, error } => write!(
                f,
                "Cannot read {}: {error}. Check that the file exists and is readable.",
                path.display()
            ),
            Self::Write { path, error } => write!(
                f,
                "Cannot write provenance at {}: {error}. Check the directory permissions.",
                path.display()
            ),
            Self::Unreadable { path, reason } => {
                write!(f, "Cannot read provenance at {}: {reason}.", path.display())
            }
            Self::TooLarge { path } => {
                write!(f, "Provenance record exceeds 64 KiB at {}.", path.display())
            }
            Self::UnsupportedSchema { path } => {
                write!(f, "Unsupported provenance schema at {}.", path.display())
            }
            Self::SourcePathDiffers { path } => {
                write!(f, "Recorded source path differs at {}.", path.display())
            }
            Self::ModelDiffers { path } => {
                write!(f, "Recorded model differs at {}.", path.display())
            }
            Self::SourceContentDiffers { path } => write!(
                f,
                "Source content differs from the record at {}.",
                path.display()
            ),
            Self::EnhancedContentDiffers { path } => write!(
                f,
                "Enhanced video content differs from the record at {}.",
                path.display()
            ),
        }
    }
}

impl std::error::Error for ProvenanceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::File { error, .. } | Self::Write { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// The sidecar path for an enhanced video: its file name plus `.upscale.json`.
pub fn record_path(enhanced: &Path) -> PathBuf {
    let mut name: OsString = enhanced.file_name().map(OsString::from).unwrap_or_default();
    name.push(".upscale.json");
    enhanced.with_file_name(name)
}

fn resolve_file(path: &Path) -> Result<PathBuf, ProvenanceError> {
    resolve(path).map_err(|error| ProvenanceError::File {
        path: path.to_path_buf(),
        error,
    })
}

fn sha256_hex(path: &Path) -> Result<String, ProvenanceError> {
    let fail = |error| ProvenanceError::File {
        path: path.to_path_buf(),
        error,
    };
    let mut file = File::open(path).map_err(fail)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    loop {
        let count = match file.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(fail(error)),
        };
        hasher.update(&buffer[..count]);
    }
    let digest = hasher.finalize();
    let bytes: &[u8] = digest.as_ref();
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        // Writing to a String does not fail.
        let _ = write!(hex, "{byte:02x}");
    }
    Ok(hex)
}

/// Appends `text` as a JSON string with every non-ASCII character escaped.
fn push_json_string(out: &mut String, text: &str) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(character),
            _ => {
                let mut units = [0_u16; 2];
                for unit in character.encode_utf16(&mut units) {
                    // Writing to a String does not fail.
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}

/// The record text: two-space indent, fixed key order, final newline.
fn render_record(
    source: &str,
    model: ModelId,
    source_sha256: &str,
    enhanced_sha256: &str,
) -> String {
    let mut out = String::from("{\n  \"schema\": 1,\n  \"source\": ");
    push_json_string(&mut out, source);
    out.push_str(",\n  \"model\": ");
    push_json_string(&mut out, model.as_str());
    out.push_str(",\n  \"source_sha256\": ");
    push_json_string(&mut out, source_sha256);
    out.push_str(",\n  \"enhanced_sha256\": ");
    push_json_string(&mut out, enhanced_sha256);
    out.push_str("\n}\n");
    out
}

fn write_atomic(destination: &Path, text: &str) -> io::Result<()> {
    let directory = destination.parent().unwrap_or_else(|| Path::new("."));
    let mut file = tempfile::Builder::new()
        .prefix(".upscale-record-")
        .suffix(".tmp")
        .tempfile_in(directory)?;
    file.write_all(text.as_bytes())?;
    file.flush()?;
    file.as_file().sync_all()?;
    file.persist(destination).map_err(|error| error.error)?;
    Ok(())
}

/// Writes the sidecar for `enhanced` and returns its path.
///
/// The write is atomic: a temporary file in the same directory, then a rename.
///
/// # Errors
///
/// [`ProvenanceError::SameFile`] when the paths resolve to one file, and a
/// file or write error otherwise.
pub fn record_upscaled(
    source: &Path,
    enhanced: &Path,
    model: ModelId,
) -> Result<PathBuf, ProvenanceError> {
    let source = resolve_file(source)?;
    let enhanced = resolve_file(enhanced)?;
    if source == enhanced {
        return Err(ProvenanceError::SameFile);
    }
    let source_text = source
        .to_str()
        .ok_or_else(|| ProvenanceError::NonUtf8Path {
            path: source.clone(),
        })?;
    let text = render_record(
        source_text,
        model,
        &sha256_hex(&source)?,
        &sha256_hex(&enhanced)?,
    );
    let destination = record_path(&enhanced);
    write_atomic(&destination, &text).map_err(|error| ProvenanceError::Write {
        path: destination.clone(),
        error,
    })?;
    Ok(destination)
}

fn read_record(path: &Path) -> Result<Value, ProvenanceError> {
    let unreadable = |reason: String| ProvenanceError::Unreadable {
        path: path.to_path_buf(),
        reason,
    };
    let mut bytes = Vec::new();
    File::open(path)
        .and_then(|file| file.take(MAX_RECORD_BYTES + 1).read_to_end(&mut bytes))
        .map_err(|error| unreadable(error.to_string()))?;
    if bytes.len() as u64 > MAX_RECORD_BYTES {
        return Err(ProvenanceError::TooLarge {
            path: path.to_path_buf(),
        });
    }
    serde_json::from_slice(&bytes).map_err(|error| unreadable(error.to_string()))
}

/// Checks the recorded association and returns the source digest for reuse.
///
/// Pass the digest from an earlier call as `source_digest` to skip hashing the
/// source again.
///
/// # Errors
///
/// A [`ProvenanceError`] for the first check that fails: record size and
/// syntax, schema, source path, model, source content, enhanced content.
pub fn check_record(
    source: &Path,
    enhanced: &Path,
    model: ModelId,
    source_digest: Option<&str>,
) -> Result<String, ProvenanceError> {
    let source = resolve_file(source)?;
    let enhanced = resolve_file(enhanced)?;
    let path = record_path(&enhanced);
    let record = read_record(&path)?;
    let schema_ok = record
        .as_object()
        .and_then(|object| object.get("schema"))
        .and_then(Value::as_i64)
        == Some(SCHEMA);
    if !schema_ok {
        return Err(ProvenanceError::UnsupportedSchema { path });
    }
    let field = |name: &str| record.get(name).and_then(Value::as_str);
    if source.to_str().is_none() || field("source") != source.to_str() {
        return Err(ProvenanceError::SourcePathDiffers { path });
    }
    if field("model") != Some(model.as_str()) {
        return Err(ProvenanceError::ModelDiffers { path });
    }
    let source_digest = match source_digest {
        Some(digest) if !digest.is_empty() => digest.to_owned(),
        _ => sha256_hex(&source)?,
    };
    if field("source_sha256") != Some(source_digest.as_str()) {
        return Err(ProvenanceError::SourceContentDiffers { path });
    }
    if field("enhanced_sha256") != Some(sha256_hex(&enhanced)?.as_str()) {
        return Err(ProvenanceError::EnhancedContentDiffers { path });
    }
    Ok(source_digest)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    const SHA256_ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const SHA256_HELLO: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

    struct Fixture {
        _dir: tempfile::TempDir,
        root: PathBuf,
        source: PathBuf,
        enhanced: PathBuf,
    }

    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = fs::canonicalize(dir.path()).expect("canonicalize");
        let source = root.join("source.mp4");
        let enhanced = root.join("source.fast.2x.mp4");
        fs::write(&source, b"abc").expect("write source");
        fs::write(&enhanced, b"hello").expect("write enhanced");
        Fixture {
            _dir: dir,
            root,
            source,
            enhanced,
        }
    }

    /// A sidecar in the form the Python PoC writes with `json.dumps(record, indent=2)`.
    fn poc_sidecar(source: &Path, model: &str) -> String {
        format!(
            "{{\n  \"schema\": 1,\n  \"source\": \"{}\",\n  \"model\": \"{model}\",\n  \
             \"source_sha256\": \"{SHA256_ABC}\",\n  \"enhanced_sha256\": \"{SHA256_HELLO}\"\n}}\n",
            source.display()
        )
    }

    #[test]
    fn record_path_appends_the_suffix_to_the_file_name() {
        assert_eq!(
            record_path(Path::new("/v/a.fast.2x.mp4")),
            PathBuf::from("/v/a.fast.2x.mp4.upscale.json")
        );
    }

    #[test]
    fn written_record_is_byte_identical_to_the_poc_form() {
        let f = fixture();
        let path = record_upscaled(&f.source, &f.enhanced, ModelId::Fast).expect("record");
        assert_eq!(path, f.root.join("source.fast.2x.mp4.upscale.json"));
        let text = fs::read_to_string(&path).expect("read record");
        assert_eq!(text, poc_sidecar(&f.source, "fast"));
    }

    #[test]
    fn a_sidecar_in_the_exact_poc_form_checks() {
        let f = fixture();
        fs::write(
            record_path(&f.enhanced),
            poc_sidecar(&f.source, "purevision"),
        )
        .expect("write sidecar");
        let digest =
            check_record(&f.source, &f.enhanced, ModelId::PureVision, None).expect("record checks");
        assert_eq!(digest, SHA256_ABC);
        let again = check_record(&f.source, &f.enhanced, ModelId::PureVision, Some(&digest))
            .expect("record checks with a known digest");
        assert_eq!(again, SHA256_ABC);
    }

    #[test]
    fn a_poc_sidecar_with_escaped_non_ascii_checks() {
        let f = fixture();
        let source = f.root.join("caf\u{e9} \u{1f3ac}.mp4");
        fs::write(&source, b"abc").expect("write source");
        let path = record_upscaled(&source, &f.enhanced, ModelId::Fast).expect("record");
        let text = fs::read_to_string(&path).expect("read record");
        assert!(text.contains("caf\\u00e9 \\ud83c\\udfac.mp4"), "{text}");
        assert!(text.is_ascii());
        check_record(&source, &f.enhanced, ModelId::Fast, None).expect("record checks");
    }

    #[test]
    fn json_strings_escape_like_python() {
        let mut out = String::new();
        push_json_string(&mut out, "a\"b\\c\n\r\t\u{8}\u{c}\u{1}\u{7f}");
        assert_eq!(out, "\"a\\\"b\\\\c\\n\\r\\t\\b\\f\\u0001\\u007f\"");
    }

    #[test]
    fn same_file_is_refused() {
        let f = fixture();
        let error = record_upscaled(&f.source, &f.source, ModelId::Fast).expect_err("refused");
        assert!(matches!(error, ProvenanceError::SameFile));
    }

    #[test]
    fn writing_leaves_no_temporary_file() {
        let f = fixture();
        record_upscaled(&f.source, &f.enhanced, ModelId::Fast).expect("record");
        let mut names: Vec<String> = fs::read_dir(&f.root)
            .expect("read dir")
            .map(|entry| {
                entry
                    .expect("entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "source.fast.2x.mp4",
                "source.fast.2x.mp4.upscale.json",
                "source.mp4"
            ]
        );
    }

    #[test]
    fn round_trip_checks() {
        let f = fixture();
        record_upscaled(&f.source, &f.enhanced, ModelId::Fast).expect("record");
        let digest = check_record(&f.source, &f.enhanced, ModelId::Fast, None).expect("checks");
        assert_eq!(digest, SHA256_ABC);
    }

    #[test]
    fn missing_record_is_unreadable() {
        let f = fixture();
        let error = check_record(&f.source, &f.enhanced, ModelId::Fast, None).expect_err("refused");
        assert!(matches!(error, ProvenanceError::Unreadable { .. }));
        assert!(error.to_string().starts_with("Cannot read provenance at "));
    }

    #[test]
    fn malformed_record_is_unreadable() {
        let f = fixture();
        fs::write(record_path(&f.enhanced), b"{ not json").expect("write");
        let error = check_record(&f.source, &f.enhanced, ModelId::Fast, None).expect_err("refused");
        assert!(matches!(error, ProvenanceError::Unreadable { .. }));
    }

    #[test]
    fn record_over_64_kib_is_refused() {
        let f = fixture();
        fs::write(record_path(&f.enhanced), vec![b' '; 65537]).expect("write");
        let error = check_record(&f.source, &f.enhanced, ModelId::Fast, None).expect_err("refused");
        assert!(matches!(error, ProvenanceError::TooLarge { .. }));
        fs::write(record_path(&f.enhanced), vec![b' '; 65536]).expect("write");
        let error = check_record(&f.source, &f.enhanced, ModelId::Fast, None).expect_err("refused");
        assert!(matches!(error, ProvenanceError::Unreadable { .. }));
    }

    #[test]
    fn schema_must_be_the_integer_1() {
        let f = fixture();
        let good = poc_sidecar(&f.source, "fast");
        for replaced in [
            "\"schema\": 2",
            "\"schema\": 1.0",
            "\"schema\": true",
            "\"schema\": \"1\"",
        ] {
            fs::write(
                record_path(&f.enhanced),
                good.replace("\"schema\": 1", replaced),
            )
            .expect("write");
            let error =
                check_record(&f.source, &f.enhanced, ModelId::Fast, None).expect_err("refused");
            assert!(
                matches!(error, ProvenanceError::UnsupportedSchema { .. }),
                "{replaced}"
            );
        }
        fs::write(record_path(&f.enhanced), b"[1]").expect("write");
        let error = check_record(&f.source, &f.enhanced, ModelId::Fast, None).expect_err("refused");
        assert!(matches!(error, ProvenanceError::UnsupportedSchema { .. }));
    }

    #[test]
    fn source_path_model_and_content_changes_are_refused() {
        let f = fixture();
        record_upscaled(&f.source, &f.enhanced, ModelId::Fast).expect("record");

        let other = f.root.join("other.mp4");
        fs::copy(&f.source, &other).expect("copy");
        let error = check_record(&other, &f.enhanced, ModelId::Fast, None).expect_err("refused");
        assert!(matches!(error, ProvenanceError::SourcePathDiffers { .. }));

        let error =
            check_record(&f.source, &f.enhanced, ModelId::RealEsrgan, None).expect_err("refused");
        assert!(matches!(error, ProvenanceError::ModelDiffers { .. }));

        fs::write(&f.source, b"abcd").expect("modify source");
        let error = check_record(&f.source, &f.enhanced, ModelId::Fast, None).expect_err("refused");
        assert!(matches!(
            error,
            ProvenanceError::SourceContentDiffers { .. }
        ));
        assert!(
            error
                .to_string()
                .starts_with("Source content differs from the record at ")
        );

        fs::write(&f.source, b"abc").expect("restore source");
        fs::write(&f.enhanced, b"hello!").expect("modify enhanced");
        let error = check_record(&f.source, &f.enhanced, ModelId::Fast, None).expect_err("refused");
        assert!(matches!(
            error,
            ProvenanceError::EnhancedContentDiffers { .. }
        ));
    }
}
