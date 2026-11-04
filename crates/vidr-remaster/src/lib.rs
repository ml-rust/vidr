//! Remaster pipeline: model selection, output resolution, provenance
//! records, and output checks over `vidr` and boostr's restoration models.
//!
//! The default build holds the pure logic: output resolution, request
//! validation, provenance records, reuse of an earlier upscale, and the model
//! catalog. It has no GPU code and no model, and it refuses a model by name.
//! The `transform` feature adds the resize-only frame transform.
//! `model-srvgg` and `model-rrdb` add the restoration models. The crate has no
//! command-line interface yet.

pub mod catalog;
mod paths;
pub mod provenance;
pub mod request;
pub mod resolution;
pub mod reuse;
#[cfg(feature = "transform")]
pub mod transform;

pub use catalog::{
    CatalogError, MODELS_DIR_ENV, ModelEntry, Network, catalog_entry, licence_notice, models_root,
    models_root_from_env, require_model, resolve_model_dir,
};
pub use provenance::{ProvenanceError, check_record, record_path, record_upscaled};
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
#[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
pub use transform::remaster_execution;
#[cfg(feature = "transform")]
pub use transform::{RemasterTransform, TransformClient, TransformError};
