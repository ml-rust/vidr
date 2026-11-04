//! Remaster pipeline: model selection, output resolution, provenance
//! records, and output checks over `vidr` and boostr's restoration models.
//!
//! This release holds the pure logic only: output resolution, request
//! validation, provenance records, and reuse of an earlier upscale. It has no
//! command-line interface, no GPU code, and no model.

mod paths;
pub mod provenance;
pub mod request;
pub mod resolution;
pub mod reuse;

pub use provenance::{ProvenanceError, check_record, record_path, record_upscaled};
pub use request::{
    Codec, DEFAULT_QUALITY, MAX_QUALITY, MIN_QUALITY, ModelId, ParseCodecError, ParseModelError,
    Request, RequestError, Validated, output_path,
};
pub use resolution::{ResolutionError, ResolutionMode, StandardResolution, choose_resolution};
pub use reuse::{
    MatchError, Reuse, ReuseError, SkipReason, Skipped, VideoFacts, VideoProbe, check_match,
    find_upscaled,
};
