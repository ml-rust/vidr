//! The steps of a run on one runtime, in order: the pre-scan, the output sizes, reuse
//! of an earlier upscale, the encode pass into a temporary directory, the output
//! check, the provenance sidecar, and the publish.

use std::fs::{self, File};
use std::path::Path;
use std::time::Instant;

use numr::runtime::Device as _;
use serde_json::{Value, json};
use vidr::color::{ColorPlan, color_plan};
use vidr::decode::{Decoder, PixelFormat};
use vidr::encode::Encoder;
use vidr::media::{CopyPlan, Input, MuxSink, VideoOutput, read_stream_facts, read_track_facts};
use vidr::pipeline::{PumpControl, PumpReport, run};
use vidr::{EditTimeline, EncodeError, StreamFacts, TrackFacts, check_output, check_tracks};

use crate::catalog::{catalog_entry, require_model};
use crate::error::RemasterError;
use crate::execution::{EncoderBackend, Execution, Selector};
use crate::frame_path::{FramePath, HostCopies};
use crate::plan::{
    EncodePlan, OutputSizes, encode_codec, encode_config, output_sizes, start_offset, stream_rate,
};
use crate::provenance::{RecordedCodecs, record_codecs};
use crate::publish::{Published, TempOutput};
use crate::report::RunReport;
use crate::request::{ModelId, Request, Validated};
use crate::run::Progress;
use crate::scan::{decoded_stream, video_facts, video_section};
use crate::transform::{RemasterTransform, TransformClient};

use super::backends::BoxEncoder;
use super::runtime::{Opened, RunRuntime};
use super::scanning::{FrameSource, ReuseQuery, Scanner, open_frames, select_existing};
use super::watch::watch_frames;

/// What the steps before a runtime exists found.
pub(super) struct Opening {
    pub(super) plan: CopyPlan,
    pub(super) expected_tracks: TrackFacts,
    pub(super) stated: StreamFacts,
    pub(super) timeline: Option<EditTimeline>,
}

/// One run's inputs, with the choice of runtime and codecs.
pub(super) struct RunContext<'a, P> {
    pub(super) request: &'a Request,
    pub(super) validated: &'a Validated,
    pub(super) progress: &'a mut P,
    pub(super) control: &'a PumpControl,
    pub(super) report: &'a mut RunReport,
    pub(super) selector: &'a Selector,
    pub(super) execution: Execution,
    pub(super) frame_path: FramePath,
    pub(super) opening: Opening,
}

pub(super) fn stop_check(control: &PumpControl) -> Result<(), RemasterError> {
    if control.stop_requested() {
        Err(RemasterError::Stopped)
    } else {
        Ok(())
    }
}

fn file_size(path: &Path) -> Result<u64, RemasterError> {
    fs::metadata(path)
        .map(|metadata| metadata.len())
        .map_err(|error| RemasterError::File {
            path: path.to_path_buf(),
            error,
        })
}

/// The report's `backend` field for a resize of an existing enhancement.
fn reuse_backend(encoder: EncoderBackend) -> String {
    match encoder {
        EncoderBackend::Nvenc => "NVENC resize of existing enhancement".to_owned(),
        other => format!("{other} resize of existing enhancement"),
    }
}

fn processing_section<O>(
    outcome: &PumpReport<O>,
    started: Instant,
    reused: bool,
    frames: &FrameSource,
    encoder: EncoderBackend,
) -> Value {
    let mut section = json!({
        "wall_seconds": started.elapsed().as_secs_f64(),
        "decoded_frames": outcome.frames_in,
        "dropped_frames": outcome.frames_dropped,
        "encoded_frames": outcome.frames_out,
        "packets": outcome.packets,
        "ai_executed": !reused,
    });
    if let (true, Some(fields)) = (reused, section.as_object_mut()) {
        fields.insert("backend".to_owned(), reuse_backend(encoder).into());
        fields.insert(
            "reused_input".to_owned(),
            frames.path.display().to_string().into(),
        );
    }
    section
}

/// Runs every step on `R`.
///
/// # Errors
///
/// As `dispatch::execute`.
pub(super) fn run_on<R, P>(run_context: RunContext<'_, P>) -> Result<Published, RemasterError>
where
    R: RunRuntime,
    R::Client: TransformClient<R>,
    P: FnMut(Progress) + Send,
{
    let RunContext {
        request,
        validated,
        progress,
        control,
        report,
        selector,
        mut execution,
        mut frame_path,
        opening,
    } = run_context;
    let Opening {
        plan,
        expected_tracks,
        stated,
        timeline,
    } = opening;
    let source_path = validated.source.as_path();
    let destination = validated.destination.as_path();

    let opened = R::open(request.openh264_library.as_deref())?;
    report.set("backend", R::BACKEND);
    let mut scanner = Scanner::<R>::new(&opened, control, selector);
    let chosen = scanner.scan(execution.decoder, source_path, progress)?;
    if let Some(reason) = chosen.fell_through {
        execution.decoder = chosen.backend;
        execution.reasons.decoder = reason;
        selector.refresh_check(&mut execution)?;
        frame_path.choose(&execution);
        report.set("frame_path", frame_path.section());
    }
    let scan = chosen.scan;
    let source_stream = decoded_stream(stated, &scan);
    let rotation = source_stream.geometry.rotation;

    let sizes = output_sizes(&source_stream.geometry, request.resolution)?;
    report.set(
        "input_video",
        video_section(
            &source_stream,
            &scan,
            file_size(source_path)?,
            &expected_tracks,
            Some(sizes.display),
        )?,
    );

    let source_facts = video_facts(&source_stream)?;
    let query = ReuseQuery {
        request,
        validated,
        source_facts: &source_facts,
        rotation,
        runtime: execution.runtime,
        encoder: execution.encoder.name(),
    };
    let existing = select_existing(&query, &mut scanner, control, progress)?;
    stop_check(control)?;
    let reused = existing.is_some();
    let frames = match existing {
        Some(frames) => {
            progress(Progress::Reusing {
                path: frames.path.clone(),
            });
            report.set("reused_upscaled", frames.path.display().to_string());
            report.set("backend", reuse_backend(execution.encoder));
            execution.decoder = frames.decoder;
            execution.reasons.decoder = frames.decoder_reason.clone();
            selector.refresh_check(&mut execution)?;
            frame_path.choose(&execution);
            report.set("frame_path", frame_path.section());
            frames
        }
        None => FrameSource {
            path: source_path.to_path_buf(),
            stream: source_stream.clone(),
            scan,
            decoder: execution.decoder,
            decoder_reason: execution.reasons.decoder.clone(),
        },
    };
    let colors = color_plan(&frames.scan.color)?;
    let transform = build_transform::<R, P>(request, reused, sizes, colors, &opened, progress)?;
    stop_check(control)?;

    let total = frames.stream.frame_count() as u64;
    progress(Progress::Sizes {
        input: source_stream.geometry.display_dimensions(),
        output: sizes.display,
        frames: total,
        reused,
    });
    progress(Progress::Output {
        path: destination.to_path_buf(),
    });

    let opened_frames = open_frames(&frames.path)?;
    let mut timing = opened_frames.source.timing();
    timing.hints = frames.stream.hints;
    let encode_plan = EncodePlan {
        codec: request.codec,
        encoder: execution.encoder,
        quality: request.quality,
        coded: sizes.coded,
        format: frames.scan.format,
        frame_rate: stream_rate(&frames.stream)?,
        time_base: timing.time_base,
        sample_aspect_ratio: source_stream.geometry.sample_aspect_ratio,
        tags: colors.tags,
    };
    let encoder = open_encoder::<R>(&opened, selector, &mut execution, encode_plan)?;
    let decoder = R::open_decoder(&opened, frames.decoder, &opened_frames.decode, &frames.path)?;
    if reused {
        report.set("backend", reuse_backend(execution.encoder));
    }
    frame_path.choose(&execution);
    frame_path.set_routes(Decoder::route(&decoder), Encoder::route(&encoder));
    report.set("frame_path", frame_path.section());
    if let Some(line) = frame_path.line() {
        progress(Progress::FramePath { line });
    }

    let mut video = VideoOutput::new(
        encode_codec(request.codec),
        Vec::new(),
        sizes.coded.width(),
        sizes.coded.height(),
        timing.time_base,
    );
    video.sample_aspect_ratio = source_stream.geometry.sample_aspect_ratio;
    video.rotation = rotation;
    video.colour = colors.tags;
    video.start_offset = start_offset(&source_stream, timeline)?;

    let temporary = TempOutput::beside(destination)?;
    let file = File::options()
        .read(true)
        .write(true)
        .create_new(true)
        .open(temporary.prepared())
        .map_err(|error| RemasterError::File {
            path: temporary.prepared().to_path_buf(),
            error,
        })?;
    let sink = MuxSink::new(file, plan, video);
    let packets = opened_frames.source;

    let started = Instant::now();
    let outcome = watch_frames(progress, control, total, || {
        run::<R, _, _, _, _, _>(packets, decoder, transform, encoder, sink, timing, control)
    });
    let counted = control.progress();
    frame_path.set_encode_pass(HostCopies {
        frames: counted.host_copied_frames,
        bytes: counted.host_copied_bytes,
    });
    frame_path.set_scans(scanner.copies());
    report.set("frame_path", frame_path.section());
    let outcome = outcome?;
    report.set(
        "processing",
        processing_section(&outcome, started, reused, &frames, execution.encoder),
    );
    outcome
        .output
        .sync_all()
        .map_err(|error| RemasterError::File {
            path: temporary.prepared().to_path_buf(),
            error,
        })?;
    drop(outcome);

    progress(Progress::Checking);
    stop_check(control)?;
    let written = Input::open(temporary.prepared())?;
    let written_stream = read_stream_facts(&written)?;
    let written_tracks = read_track_facts(&written)?;
    drop(written);
    let check = scanner.scan(execution.check_decoder, temporary.prepared(), progress)?;
    if let Some(reason) = check.fell_through {
        execution.check_decoder = check.backend;
        execution.reasons.check_decoder = reason;
        frame_path.choose(&execution);
    }
    frame_path.set_scans(scanner.copies());
    report.set("frame_path", frame_path.section());
    let written_scan = check.scan;
    let written_stream = decoded_stream(written_stream, &written_scan);
    check_output(&source_stream, &written_stream, sizes.display)?;
    check_tracks(&expected_tracks, &written_tracks)?;
    report.set(
        "output_video",
        video_section(
            &written_stream,
            &written_scan,
            file_size(temporary.prepared())?,
            &written_tracks,
            None,
        )?,
    );

    stop_check(control)?;
    // The record holds the source path and two digests, and no output path. So it
    // stays correct after the move, and each way it can fail comes before the publish.
    let codecs = RecordedCodecs {
        decoder: frames.decoder.name(),
        encoder: execution.encoder.name(),
    };
    let sidecar = record_codecs(source_path, temporary.prepared(), request.model, codecs)?;
    stop_check(control)?;
    let published = temporary.publish_with_sidecar(request.overwrite, &sidecar)?;
    progress(Progress::Saved {
        path: destination.to_path_buf(),
    });
    Ok(published)
}

/// Whether the error states what the GPU cannot encode. Under `--encoder auto`, a CPU
/// encoder takes over for these.
const fn refused_by_hardware(error: &EncodeError) -> bool {
    matches!(
        error,
        EncodeError::CodecUnsupported { .. }
            | EncodeError::BitDepthUnsupported { .. }
            | EncodeError::SizeOutOfRange { .. }
            | EncodeError::VulkanRefused { .. }
            | EncodeError::VulkanUnavailable { .. }
            | EncodeError::VulkanNoMemoryType { .. }
    )
}

/// Opens the chosen encoder. A library or queue check says only that the encoder loads, so what
/// this GPU refuses comes here, and `--encoder auto` falls through to the CPU.
fn open_encoder<R: RunRuntime>(
    opened: &Opened<R>,
    selector: &Selector,
    execution: &mut Execution,
    plan: EncodePlan,
) -> Result<BoxEncoder<R>, RemasterError> {
    let config = encode_config(&plan)?;
    match R::open_encoder(opened, execution.encoder, config) {
        Err(RemasterError::Encode(error)) if refused_by_hardware(&error) => {
            let ten_bit = plan.format == PixelFormat::P010;
            let Some((backend, reason)) = selector.cpu_encoder_after(ten_bit, &error.to_string())
            else {
                return Err(RemasterError::Encode(error));
            };
            execution.encoder = backend;
            execution.reasons.encoder = reason;
            selector.refresh_check(execution)?;
            let config = encode_config(&EncodePlan {
                encoder: backend,
                ..plan
            })?;
            R::open_encoder(opened, backend, config)
        }
        other => other,
    }
}

#[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
fn load_model<R>(
    model: ModelId,
    models_dir: Option<&Path>,
    device: &R::Device,
    sizes: OutputSizes,
    colors: ColorPlan,
) -> Result<RemasterTransform<R>, RemasterError>
where
    R: RunRuntime,
    R::Client: TransformClient<R>,
{
    let root = crate::catalog::models_root_from_env(models_dir)?;
    let directory = crate::catalog::resolve_model_dir(&root, model)?;
    Ok(RemasterTransform::<R>::load_at(
        model,
        &directory,
        device,
        sizes.coded,
        colors,
        R::model_execution(),
    )?)
}

/// `require_model` already refused every model in this build, so no path reaches
/// this function with a model. It keeps the refusal in one type.
#[cfg(not(any(feature = "model-srvgg", feature = "model-rrdb")))]
fn load_model<R>(
    model: ModelId,
    _models_dir: Option<&Path>,
    _device: &R::Device,
    _sizes: OutputSizes,
    _colors: ColorPlan,
) -> Result<RemasterTransform<R>, RemasterError>
where
    R: RunRuntime,
    R::Client: TransformClient<R>,
{
    Err(crate::catalog::CatalogError::NotBuilt { model }.into())
}

fn build_transform<R, P>(
    request: &Request,
    reused: bool,
    sizes: OutputSizes,
    colors: ColorPlan,
    opened: &Opened<R>,
    progress: &mut P,
) -> Result<RemasterTransform<R>, RemasterError>
where
    R: RunRuntime,
    R::Client: TransformClient<R>,
    P: FnMut(Progress),
{
    if reused {
        return Ok(RemasterTransform::<R>::resize_only(sizes.coded, colors)?);
    }
    require_model(request.model)?;
    let transform = load_model::<R>(
        request.model,
        request.models_dir.as_deref(),
        &opened.device,
        sizes,
        colors,
    )?;
    progress(Progress::ModelLoaded {
        name: catalog_entry(request.model).directory,
        device: opened.device.name(),
        precision: R::PRECISION,
    });
    Ok(transform)
}
