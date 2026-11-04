//! Remaster pipeline: model selection, output resolution, provenance records,
//! run orchestration, and the `vidr-remaster` command, over `vidr` and boostr's
//! restoration models.
//!
//! The default build holds the pure logic: output resolution, request
//! validation, provenance records, reuse of an earlier upscale, the model
//! catalog, the run report, the publish rules, and the command line. It has no
//! GPU code and no model. [`remaster`] then validates, reports, and names the
//! features it lacks.
//!
//! A run needs `cuda` and a container feature, `mp4` or `mkv`. `transform` adds
//! the resize-only frame transform. `model-srvgg` and `model-rrdb` add the
//! restoration models. A run decodes with NVDEC and encodes with NVENC, so it
//! needs an NVIDIA GPU until a vendor-neutral path exists.

pub mod catalog;
pub mod cli;
mod error;
mod paths;
#[cfg(any(feature = "mp4", feature = "mkv"))]
pub mod plan;
pub mod provenance;
pub mod publish;
pub mod rate;
pub mod report;
pub mod request;
pub mod resolution;
pub mod reuse;
pub mod run;
#[cfg(all(feature = "cuda", any(feature = "mp4", feature = "mkv")))]
mod run_gpu;
#[cfg(any(feature = "mp4", feature = "mkv"))]
pub mod scan;
#[cfg(feature = "transform")]
pub mod transform;

pub use catalog::{
    CatalogError, MODELS_DIR_ENV, ModelEntry, Network, catalog_entry, licence_notice, models_root,
    models_root_from_env, require_model, resolve_model_dir,
};
pub use cli::main_with_args;
pub use error::RemasterError;
pub use provenance::{ProvenanceError, check_record, record_path, record_upscaled};
pub use publish::{PublishError, TempOutput};
pub use rate::frame_rate_hint;
pub use report::{ReportError, ReportStatus, RunReport, run_id, state_home, state_home_from_env};
pub use request::{
    Codec, DEFAULT_QUALITY, MAX_QUALITY, MIN_QUALITY, ModelId, ParseCodecError, ParseModelError,
    Request, RequestError, Validated, output_path,
};
pub use resolution::{
    ResolutionError, ResolutionMode, StandardResolution, choose_resolution, coded_output_size,
};
pub use reuse::{
    MatchError, Reuse, ReuseError, SkipReason, Skipped, VideoFacts, VideoProbe, check_match,
    find_upscaled,
};
pub use run::{Progress, RunOutcome, missing_features, progress_line, remaster, remaster_in};
#[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
pub use transform::remaster_execution;
#[cfg(feature = "transform")]
pub use transform::{RemasterTransform, TransformClient, TransformError};
