//! The restoration models: where each one lives, where it comes from, and its
//! licence.
//!
//! This module reads the file system only. It holds no network code. A URL
//! appears in error text so the user can fetch the model.

use std::env;
use std::ffi::OsStr;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::paths::expand_user;
use crate::request::ModelId;

/// The environment variable that names the model root.
pub const MODELS_DIR_ENV: &str = "VIDR_MODELS_DIR";

/// Extension of a checkpoint file.
const CHECKPOINT_EXTENSION: &str = "pth";

/// The network a model's checkpoint holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Network {
    /// SRVGGNetCompact, which runs at x4.
    SrVggCompact,
    /// RRDBNet, which runs at x2.
    Rrdb,
}

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::SrVggCompact => "SRVGGNetCompact",
            Self::Rrdb => "RRDBNet",
        })
    }
}

/// One model of the catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelEntry {
    /// The model.
    pub model: ModelId,
    /// The directory name under the model root.
    pub directory: &'static str,
    /// The checkpoint's download URL. It appears in error text only.
    pub url: &'static str,
    /// The licence of the weights, as an SPDX name.
    pub licence: &'static str,
    /// The network the checkpoint holds.
    pub network: Network,
    /// The Cargo feature that builds the network.
    pub feature: &'static str,
    /// A notice a run's report and this model's errors must carry.
    pub notice: Option<&'static str>,
}

const FAST: ModelEntry = ModelEntry {
    model: ModelId::Fast,
    directory: "realesr-general-x4v3",
    url: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    licence: "BSD-3-Clause",
    network: Network::SrVggCompact,
    feature: "model-srvgg",
    notice: None,
};

const REALESRGAN: ModelEntry = ModelEntry {
    model: ModelId::RealEsrgan,
    directory: "RealESRGAN_x2plus",
    url: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    licence: "BSD-3-Clause",
    network: Network::Rrdb,
    feature: "model-rrdb",
    notice: None,
};

const PUREVISION: ModelEntry = ModelEntry {
    model: ModelId::PureVision,
    directory: "2x_PureVision",
    url: "https://github.com/limitlesslab/AI-upscaling-models/releases/download/PureVision/2x_PureVision.pth",
    licence: "CC-BY-NC-SA-4.0",
    network: Network::Rrdb,
    feature: "model-rrdb",
    notice: Some(
        "The purevision weights use CC BY-NC-SA 4.0, which forbids commercial use and requires \
         share-alike. See https://github.com/limitlesslab/AI-upscaling-models.",
    ),
};

/// Returns the catalog entry of `model`.
pub const fn catalog_entry(model: ModelId) -> &'static ModelEntry {
    match model {
        ModelId::Fast => &FAST,
        ModelId::RealEsrgan => &REALESRGAN,
        ModelId::PureVision => &PUREVISION,
    }
}

/// Returns the licence notice a run's report must state for `model`, or `None`
/// when the model needs none.
pub const fn licence_notice(model: ModelId) -> Option<&'static str> {
    catalog_entry(model).notice
}

/// An error finding a model.
#[derive(Debug)]
#[non_exhaustive]
pub enum CatalogError {
    /// Neither `--models-dir` nor `VIDR_MODELS_DIR` names a model root.
    NoModelRoot,
    /// The model's directory does not exist.
    MissingDirectory {
        /// The model.
        model: ModelId,
        /// The expected directory.
        path: PathBuf,
    },
    /// The model's directory holds no `.pth` file.
    MissingCheckpoint {
        /// The model.
        model: ModelId,
        /// The directory.
        path: PathBuf,
    },
    /// The model's directory does not list.
    Unreadable {
        /// The directory.
        path: PathBuf,
        /// The operating-system error.
        error: io::Error,
    },
    /// This build leaves out the model's network.
    NotBuilt {
        /// The model.
        model: ModelId,
    },
}

impl fmt::Display for CatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoModelRoot => write!(
                f,
                "No model root is set. Pass --models-dir or set {MODELS_DIR_ENV} to the directory \
                 that holds the model directories."
            ),
            Self::MissingDirectory { model, path } => {
                let entry = catalog_entry(*model);
                write!(
                    f,
                    "Model {model} needs the directory {}, which does not exist. Download {} into \
                     that directory.",
                    path.display(),
                    entry.url
                )?;
                write_notice(f, entry)
            }
            Self::MissingCheckpoint { model, path } => {
                let entry = catalog_entry(*model);
                write!(
                    f,
                    "Model {model} needs a .pth file in {}, and the directory holds none. \
                     Download {} into that directory.",
                    path.display(),
                    entry.url
                )?;
                write_notice(f, entry)
            }
            Self::Unreadable { path, error } => write!(
                f,
                "Cannot list the model directory {}: {error}. Check the path and its permissions.",
                path.display()
            ),
            Self::NotBuilt { model } => {
                let entry = catalog_entry(*model);
                write!(
                    f,
                    "Model {model} is not in this build, because the {feature} feature is off. \
                     Rebuild vidr-remaster with --features {feature}, or choose a model this \
                     build holds.",
                    feature = entry.feature
                )?;
                write_notice(f, entry)
            }
        }
    }
}

fn write_notice(f: &mut fmt::Formatter<'_>, entry: &ModelEntry) -> fmt::Result {
    match entry.notice {
        Some(notice) => write!(f, " {notice}"),
        None => Ok(()),
    }
}

impl std::error::Error for CatalogError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Unreadable { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// Returns the model root: `flag` when set, else `environment`.
///
/// An empty value counts as unset. This function reads no environment, so a
/// caller and a test pass both values.
///
/// # Errors
///
/// [`CatalogError::NoModelRoot`] when neither names a directory.
pub fn models_root(
    flag: Option<&Path>,
    environment: Option<&OsStr>,
) -> Result<PathBuf, CatalogError> {
    flag.filter(|path| !path.as_os_str().is_empty())
        .or_else(|| environment.filter(|value| !value.is_empty()).map(Path::new))
        .map(Path::to_path_buf)
        .ok_or(CatalogError::NoModelRoot)
}

/// Returns the model root: `--models-dir` when set, else `VIDR_MODELS_DIR`.
///
/// # Errors
///
/// [`CatalogError::NoModelRoot`] when neither names a directory.
pub fn models_root_from_env(flag: Option<&Path>) -> Result<PathBuf, CatalogError> {
    let environment = env::var_os(MODELS_DIR_ENV);
    models_root(flag, environment.as_deref())
}

/// Returns the directory of `model` under `root`. A leading `~` in `root`
/// expands to the home directory.
///
/// # Errors
///
/// - [`CatalogError::MissingDirectory`] names the expected path and the URL.
/// - [`CatalogError::MissingCheckpoint`] names the directory and the URL.
/// - [`CatalogError::Unreadable`] names the directory that does not list.
pub fn resolve_model_dir(root: &Path, model: ModelId) -> Result<PathBuf, CatalogError> {
    let path = expand_user(root).join(catalog_entry(model).directory);
    if !path.is_dir() {
        return Err(CatalogError::MissingDirectory { model, path });
    }
    let entries = match fs::read_dir(&path) {
        Ok(entries) => entries,
        Err(error) => return Err(CatalogError::Unreadable { path, error }),
    };
    for entry in entries {
        let file = match entry {
            Ok(entry) => entry.path(),
            Err(error) => return Err(CatalogError::Unreadable { path, error }),
        };
        let is_checkpoint = file
            .extension()
            .and_then(OsStr::to_str)
            .is_some_and(|extension| extension.eq_ignore_ascii_case(CHECKPOINT_EXTENSION));
        if is_checkpoint && file.is_file() {
            return Ok(path);
        }
    }
    Err(CatalogError::MissingCheckpoint { model, path })
}

/// Refuses `model` by name when this build leaves out its network.
///
/// # Errors
///
/// [`CatalogError::NotBuilt`] names the model and the Cargo feature to enable.
pub fn require_model(model: ModelId) -> Result<(), CatalogError> {
    let built = match catalog_entry(model).network {
        Network::SrVggCompact => cfg!(feature = "model-srvgg"),
        Network::Rrdb => cfg!(feature = "model-rrdb"),
    };
    if built {
        Ok(())
    } else {
        Err(CatalogError::NotBuilt { model })
    }
}
