//! The GPU path of a run, in the order of its steps.
//!
//! Every decision lives in plain functions in `scan`, `plan`, `publish`, and
//! `report`, so tests reach them with no GPU. This file holds the calls that need
//! one, and runs only with `cuda` and a container feature.

use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use numr::runtime::Device as _;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime, is_cuda_available};
use serde_json::{Value, json};
use vidr::color::{ColorPlan, color_plan};
use vidr::decode::DecodedFrame;
use vidr::decode::nvdec::{NvdecDecoder, nvdec_available};
use vidr::encode::nvenc::{NvencEncoder, nvenc_available};
use vidr::media::{
    DecodeConfig, DroppedTrack, Input, InputSource, MuxSink, OutputKind, VideoOutput, plan_copy,
    read_stream_facts, read_track_facts,
};
use vidr::pipeline::{PacketSource as _, PumpControl, PumpReport, run};
use vidr::{Rotation, StreamFacts, check_output, check_tracks};

use crate::catalog::{catalog_entry, require_model};
use crate::error::RemasterError;
use crate::plan::{
    EncodePlan, OutputSizes, encode_codec, encode_config, encoder_open_error, output_sizes,
    start_offset, stream_rate,
};
use crate::provenance::record_upscaled;
use crate::publish::{Published, TempOutput};
use crate::report::RunReport;
use crate::request::{ModelId, Request, Validated};
use crate::reuse::{Reuse, VideoFacts, VideoProbe, find_upscaled};
use crate::run::Progress;
use crate::scan::{
    ScanError, ScanFacts, ScanFrame, ScanState, candidate_facts, decoded_stream, video_facts,
    video_section,
};
use crate::transform::RemasterTransform;

/// Frames between `Scanned` events, as the PoC prints them.
const SCAN_NOTICE_INTERVAL: u64 = 10_000;
/// How often the watcher reads the pump's counters.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(250);
/// The report's `backend` field for a resize of an existing enhancement.
const REUSE_BACKEND: &str = "NVENC resize of existing enhancement";

/// The video that supplies the frames: the source, or an enhanced version of it.
struct FrameSource {
    path: PathBuf,
    stream: StreamFacts,
    scan: ScanFacts,
}

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
    control: &PumpControl,
    report: &mut RunReport,
) -> Result<Published, RemasterError> {
    let source_path = validated.source.as_path();
    let destination = validated.destination.as_path();
    progress(Progress::Analyzing {
        source: source_path.to_path_buf(),
    });

    // Every stream refusal comes before the GPU work.
    let mut input = Input::open(source_path)?;
    let kind = OutputKind::from_path(destination)?;
    let plan = plan_copy(&mut input, kind)?;
    report.set("dropped_tracks", dropped_tracks(plan.dropped()));
    report.set("max_rounding_error_ns", plan.max_rounding_error_ns());
    let expected_tracks = plan.expected_tracks();
    let stated = read_stream_facts(&input)?;
    let timeline = InputSource::new(input)?.timing().timeline;

    let client = open_gpu()?;
    let device = CudaDevice::new(0);
    let scan = scan_file(&client, source_path, control, progress)?;
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
    let existing = select_existing(
        request,
        validated,
        &source_facts,
        rotation,
        &client,
        progress,
        control,
    )?;
    stop_check(control)?;
    let reused = existing.is_some();
    let frames = match existing {
        Some(frames) => {
            progress(Progress::Reusing {
                path: frames.path.clone(),
            });
            report.set("reused_upscaled", frames.path.display().to_string());
            report.set("backend", REUSE_BACKEND);
            frames
        }
        None => FrameSource {
            path: source_path.to_path_buf(),
            stream: source_stream.clone(),
            scan,
        },
    };
    let colors = color_plan(&frames.scan.color)?;
    let transform = build_transform(request, reused, sizes, colors, &device, progress)?;
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

    let opened = open_frames(&frames.path)?;
    let mut timing = opened.source.timing();
    timing.hints = frames.stream.hints;
    let config = encode_config(&EncodePlan {
        codec: request.codec,
        quality: request.quality,
        coded: sizes.coded,
        format: frames.scan.format,
        frame_rate: stream_rate(&frames.stream)?,
        time_base: timing.time_base,
        sample_aspect_ratio: source_stream.geometry.sample_aspect_ratio,
        tags: colors.tags,
    })?;
    let encoder = NvencEncoder::new(&client, config).map_err(encoder_open_error)?;
    let decoder = NvdecDecoder::new(&client, opened.decode.codec, &opened.decode.config)
        .map_err(|error| decode_error(&frames.path, error))?;

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
    let packets = opened.source;

    let started = Instant::now();
    let outcome = watch_frames(progress, control, total, || {
        run::<CudaRuntime, _, _, _, _, _>(
            packets, decoder, transform, encoder, sink, timing, control,
        )
    })?;
    report.set(
        "processing",
        processing_section(&outcome, started, reused, &frames),
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
    let written_scan = scan_file(&client, temporary.prepared(), control, progress)?;
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
    let sidecar = record_upscaled(source_path, temporary.prepared(), request.model)?;
    stop_check(control)?;
    let published = temporary.publish_with_sidecar(request.overwrite, &sidecar)?;
    progress(Progress::Saved {
        path: destination.to_path_buf(),
    });
    Ok(published)
}

fn stop_check(control: &PumpControl) -> Result<(), RemasterError> {
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

fn decode_error(path: &Path, error: vidr::DecodeError) -> RemasterError {
    RemasterError::Decode {
        path: path.to_path_buf(),
        error,
    }
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

fn processing_section<O>(
    outcome: &PumpReport<O>,
    started: Instant,
    reused: bool,
    frames: &FrameSource,
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
        fields.insert("backend".to_owned(), REUSE_BACKEND.into());
        fields.insert(
            "reused_input".to_owned(),
            frames.path.display().to_string().into(),
        );
    }
    section
}

/// Opens CUDA device 0, with the two driver libraries a run needs.
fn open_gpu() -> Result<CudaClient, RemasterError> {
    let unavailable = |reason: String| RemasterError::Gpu { reason };
    nvdec_available().map_err(|error| unavailable(error.to_string()))?;
    nvenc_available().map_err(|error| unavailable(error.to_string()))?;
    if !is_cuda_available() {
        return Err(RemasterError::NoGpu);
    }
    CudaClient::new(CudaDevice::new(0)).map_err(|error| {
        unavailable(format!(
            "numr cannot open CUDA device 0: {error}. Check the driver and free GPU memory."
        ))
    })
}

/// A video's packets and what its decoder takes.
struct OpenedFrames {
    source: InputSource,
    decode: DecodeConfig,
    container: Option<vidr::media::ContainerColour>,
}

fn open_frames(path: &Path) -> Result<OpenedFrames, RemasterError> {
    let input = Input::open(path)?;
    let track = input.video_track()?.clone();
    let decode = input.decode_config(&track)?;
    let container = input.colour(&track);
    Ok(OpenedFrames {
        source: InputSource::new(input)?,
        decode,
        container,
    })
}

/// Decodes `path` once and drops each frame at once. See [`ScanState`].
fn scan_file<P: FnMut(Progress)>(
    client: &CudaClient,
    path: &Path,
    control: &PumpControl,
    progress: &mut P,
) -> Result<ScanFacts, RemasterError> {
    let opened = open_frames(path)?;
    let mut source = opened.source;
    let mut decoder = NvdecDecoder::new(client, opened.decode.codec, &opened.decode.config)
        .map_err(|error| decode_error(path, error))?;
    let mut state = ScanState::new(opened.container);
    let mut data = Vec::new();
    let mut packets = 0_u64;
    while let Some(packet) = source.next(&mut data)? {
        stop_check(control)?;
        packets += 1;
        decoder
            .send(&data, packet.pts)
            .map_err(|error| decode_error(path, error))?;
        observe_ready(&mut decoder, &mut state, path, progress)?;
    }
    decoder.flush().map_err(|error| decode_error(path, error))?;
    observe_ready(&mut decoder, &mut state, path, progress)?;
    state.finish(packets).map_err(|refusal| {
        RemasterError::Scan(ScanError {
            path: path.to_path_buf(),
            refusal,
        })
    })
}

fn observe_ready<P: FnMut(Progress)>(
    decoder: &mut NvdecDecoder,
    state: &mut ScanState,
    path: &Path,
    progress: &mut P,
) -> Result<(), RemasterError> {
    while let Some(frame) = decoder.receive() {
        let frames = state.observe(scan_frame(&frame)).map_err(|refusal| {
            RemasterError::Scan(ScanError {
                path: path.to_path_buf(),
                refusal,
            })
        })?;
        if frames.is_multiple_of(SCAN_NOTICE_INTERVAL) {
            progress(Progress::Scanned { frames });
        }
    }
    Ok(())
}

fn scan_frame(frame: &DecodedFrame<CudaRuntime>) -> ScanFrame {
    ScanFrame {
        pts: frame.pts(),
        display: frame.display(),
        format: frame.format(),
        color: *frame.color(),
        progressive: frame.progressive(),
    }
}

/// [`VideoProbe`] over `read_stream_facts` and a full scan of the candidate.
///
/// The pump decodes the selected candidate, so it must pass what the source
/// passes. The probe keeps each result, so the selected one needs no second scan.
struct ScanProbe<'a, P> {
    client: &'a CudaClient,
    control: &'a PumpControl,
    progress: &'a mut P,
    rotation: Rotation,
    scanned: HashMap<PathBuf, FrameSource>,
}

impl<P: FnMut(Progress)> ScanProbe<'_, P> {
    fn read(&mut self, path: &Path) -> Result<(StreamFacts, ScanFacts), RemasterError> {
        let input = Input::open(path)?;
        let stated = read_stream_facts(&input)?;
        drop(input);
        let scan = scan_file(self.client, path, self.control, self.progress)?;
        Ok((decoded_stream(stated, &scan), scan))
    }
}

impl<P: FnMut(Progress)> VideoProbe for ScanProbe<'_, P> {
    type Error = String;

    fn probe(&mut self, path: &Path) -> Result<VideoFacts, String> {
        let (stream, scan) = self.read(path).map_err(|error| error.to_string())?;
        let facts = candidate_facts(&stream, self.rotation)?;
        self.scanned.insert(
            path.to_path_buf(),
            FrameSource {
                path: path.to_path_buf(),
                stream,
                scan,
            },
        );
        Ok(facts)
    }
}

/// Selects an enhanced video, unless the request runs the model regardless.
fn select_existing<P: FnMut(Progress)>(
    request: &Request,
    validated: &Validated,
    source_facts: &VideoFacts,
    rotation: Rotation,
    client: &CudaClient,
    progress: &mut P,
    control: &PumpControl,
) -> Result<Option<FrameSource>, RemasterError> {
    if request.force_ai {
        return Ok(None);
    }
    let mut probe = ScanProbe {
        client,
        control,
        progress: &mut *progress,
        rotation,
        scanned: HashMap::new(),
    };
    let found = find_upscaled(
        &mut probe,
        &validated.source,
        request.model,
        source_facts,
        &validated.destination,
        request.from_upscaled.as_deref(),
    );
    let mut scanned = probe.scanned;
    // A stop makes a probe fail, and that failure is no statement on the candidate.
    stop_check(control)?;
    let Reuse { selected, skipped } = found?;
    for candidate in skipped {
        progress(Progress::Skipped(candidate));
    }
    Ok(selected.and_then(|path| scanned.remove(&path)))
}

#[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
fn load_model(
    model: ModelId,
    models_dir: Option<&Path>,
    device: &CudaDevice,
    sizes: OutputSizes,
    colors: ColorPlan,
) -> Result<RemasterTransform<CudaRuntime>, RemasterError> {
    let root = crate::catalog::models_root_from_env(models_dir)?;
    let directory = crate::catalog::resolve_model_dir(&root, model)?;
    Ok(RemasterTransform::<CudaRuntime>::load(
        model,
        &directory,
        device,
        sizes.coded,
        colors,
    )?)
}

/// `require_model` already refused every model in this build, so no path reaches
/// this function with a model. It keeps the refusal in one type.
#[cfg(not(any(feature = "model-srvgg", feature = "model-rrdb")))]
fn load_model(
    model: ModelId,
    _models_dir: Option<&Path>,
    _device: &CudaDevice,
    _sizes: OutputSizes,
    _colors: ColorPlan,
) -> Result<RemasterTransform<CudaRuntime>, RemasterError> {
    Err(crate::catalog::CatalogError::NotBuilt { model }.into())
}

fn build_transform<P: FnMut(Progress)>(
    request: &Request,
    reused: bool,
    sizes: OutputSizes,
    colors: ColorPlan,
    device: &CudaDevice,
    progress: &mut P,
) -> Result<RemasterTransform<CudaRuntime>, RemasterError> {
    if reused {
        return Ok(RemasterTransform::<CudaRuntime>::resize_only(
            sizes.coded,
            colors,
        )?);
    }
    require_model(request.model)?;
    let transform = load_model(
        request.model,
        request.models_dir.as_deref(),
        device,
        sizes,
        colors,
    )?;
    progress(Progress::ModelLoaded {
        name: catalog_entry(request.model).directory,
        device: device.name(),
    });
    Ok(transform)
}

/// Runs `pump` on this thread while a watcher thread reports the frames it
/// encodes. A run with no watcher thread still completes, with no frame events.
fn watch_frames<P, T, E>(
    progress: &mut P,
    control: &PumpControl,
    total: u64,
    pump: impl FnOnce() -> Result<T, E>,
) -> Result<T, RemasterError>
where
    P: FnMut(Progress) + Send,
    RemasterError: From<E>,
{
    let finished = AtomicBool::new(false);
    let started = Instant::now();
    let result = thread::scope(|scope| {
        let watcher = thread::Builder::new()
            .name("vidr-progress".to_owned())
            .spawn_scoped(scope, || {
                let mut shown = 0;
                loop {
                    let done = finished.load(Ordering::Acquire);
                    let count = control.progress().frames_out;
                    if count != shown {
                        shown = count;
                        progress(Progress::Frames {
                            count,
                            total,
                            elapsed: started.elapsed(),
                        });
                    }
                    if done {
                        return;
                    }
                    thread::park_timeout(PROGRESS_INTERVAL);
                }
            })
            .ok();
        let result = pump();
        finished.store(true, Ordering::Release);
        if let Some(watcher) = watcher {
            watcher.thread().unpark();
            // A panic in the caller's callback does not change the run's result.
            let _ = watcher.join();
        }
        result
    });
    Ok(result?)
}
