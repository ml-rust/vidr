//! The steps before a runtime exists, and the choice of runtime.

use numr::runtime::cpu::CpuRuntime;
use serde_json::{Value, json};
use vidr::media::{DroppedTrack, Input, InputSource, OutputKind, plan_copy, read_stream_facts};

use crate::error::RemasterError;
use crate::execution::{Compiled, ExecutionRequest, Machine, RuntimeKind, Selector};
use crate::frame_path::FramePath;
use crate::publish::Published;
use crate::report::RunReport;
use crate::request::{Request, Validated};
use crate::run::Progress;

use super::pass::{Opening, RunContext, run_on};
use super::scanning::source_facts;

/// Runs every step after validation.
///
/// # Errors
///
/// The first [`RemasterError`], or [`RemasterError::Stopped`] after a stop
/// request. The temporary output drops on every return, so no partial file stays.
/// After the output publishes, the only thing left that can fail is the sidecar's
/// rename, and that returns [`Published::WithoutSidecar`].
pub(crate) fn execute<P: FnMut(Progress) + Send>(
    request: &Request,
    validated: &Validated,
    progress: &mut P,
    control: &vidr::pipeline::PumpControl,
    report: &mut RunReport,
) -> Result<Published, RemasterError> {
    progress(Progress::Analyzing {
        source: validated.source.clone(),
    });

    // Every stream refusal comes before the device work.
    let mut input = Input::open(&validated.source)?;
    let kind = OutputKind::from_path(&validated.destination)?;
    let plan = plan_copy(&mut input, kind)?;
    report.set("dropped_tracks", dropped_tracks(plan.dropped()));
    report.set("max_rounding_error_ns", plan.max_rounding_error_ns());
    let expected_tracks = plan.expected_tracks();
    let stated = read_stream_facts(&input)?;
    let track = input.video_track()?.clone();
    let facts = source_facts(&input.decode_config(&track)?)?;
    let timeline = InputSource::new(input)?.timing().timeline;

    let selector = Selector::new(
        ExecutionRequest::from(request),
        Compiled::this_build(),
        Machine::probe(request.runtime),
    );
    let execution = selector.select(&facts)?;
    let runtime = execution.runtime;
    let frame_path = FramePath::new(&execution);
    report.set("frame_path", frame_path.section());

    let run = RunContext {
        request,
        validated,
        progress,
        control,
        report,
        selector: &selector,
        execution,
        frame_path,
        opening: Opening {
            plan,
            expected_tracks,
            stated,
            timeline,
        },
    };
    match runtime {
        RuntimeKind::Cuda => on_cuda(run),
        RuntimeKind::Wgpu => on_wgpu(run),
        RuntimeKind::Cpu => run_on::<CpuRuntime, P>(run),
    }
}

#[cfg(feature = "cuda")]
fn on_cuda<P: FnMut(Progress) + Send>(run: RunContext<'_, P>) -> Result<Published, RemasterError> {
    run_on::<numr::runtime::cuda::CudaRuntime, P>(run)
}

/// `execution_for` chooses CUDA only in a build that holds it.
#[cfg(not(feature = "cuda"))]
fn on_cuda<P: FnMut(Progress) + Send>(_run: RunContext<'_, P>) -> Result<Published, RemasterError> {
    Err(RemasterError::MissingFeatures {
        missing: vec!["cuda"],
    })
}

#[cfg(feature = "wgpu")]
fn on_wgpu<P: FnMut(Progress) + Send>(run: RunContext<'_, P>) -> Result<Published, RemasterError> {
    run_on::<numr::runtime::wgpu::WgpuRuntime, P>(run)
}

/// `execution_for` chooses WebGPU only in a build that holds it.
#[cfg(not(feature = "wgpu"))]
fn on_wgpu<P: FnMut(Progress) + Send>(_run: RunContext<'_, P>) -> Result<Published, RemasterError> {
    Err(RemasterError::MissingFeatures {
        missing: vec!["wgpu"],
    })
}

fn dropped_tracks(tracks: &[DroppedTrack]) -> Value {
    let list = tracks
        .iter()
        .map(|track| {
            json!({
                "track_id": track.track_id,
                "kind": format!("{:?}", track.kind).to_lowercase(),
                "codec": track.codec,
            })
        })
        .collect();
    Value::Array(list)
}
