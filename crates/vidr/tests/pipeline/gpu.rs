//! Real NVDEC, `to_rgb`, identity, `from_rgb`, NVENC, and `OutputMux`, into MP4
//! and MKV. ffmpeg generates the source under cargo's test directory. The test
//! skips with a notice without a GPU, the NVIDIA libraries, or ffmpeg.

use std::collections::HashSet;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use numr::dtype::DType;
use numr::ops::ConvLayout;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime, is_cuda_available};
use vidr::color::{ColorPlan, color_plan, merge_container};
use vidr::decode::nvdec::{NvdecDecoder, nvdec_available};
use vidr::decode::{
    ColorFacts, DecodedFrame, Decoder, MATRIX_UNSPECIFIED, PRIMARIES_UNSPECIFIED, PixelFormat,
    TRANSFER_UNSPECIFIED,
};
use vidr::encode::nvenc::{NvencEncoder, nvenc_available};
use vidr::encode::{EncodeCodec, EncodeConfig, EncodeFrame, EncodedPacket, Encoder, RateControl};
use vidr::frame::{OutputPlanes, from_rgb, to_rgb};
use vidr::media::{
    Input, InputSource, MuxSink, OutputKind, VideoOutput, plan_copy, read_stream_facts,
    read_track_facts,
};
use vidr::pipeline::{
    FrameTransform, MAX_IN_FLIGHT_FRAMES, PacketSource, PipelineError, PumpControl, run,
};
use vidr::{
    DecodeError, Dimensions, EncodeError, FrameRoute, Rational, Rounding, StartOffset, TrackTiming,
    check_output, check_tracks,
};

use super::fakes::watched;

/// numr's free list keeps one bucket per byte size. No other test uses this
/// size, so only this test's planes pass through their buckets.
const WIDTH: u32 = 656;
const HEIGHT: u32 = 368;
/// 8.008 s at 30000/1001.
const FRAMES: u64 = 240;
/// Luma PSNR floor at cq 20 against the source's own decode. It leaves room
/// for the sharp detail of `testsrc2`, so a failure means frames arrive wrong.
const FLOOR_DB: f64 = 30.0;
/// Luma-size buffers a run can touch. The frames in flight count, with one more
/// while the transform holds a frame and its output. A full H.264 decoder queue
/// at the flush adds the last term. numr's free list reuses each buffer, so a
/// leak grows the count with the frame count, far past this bound.
const LUMA_BUFFERS: usize = MAX_IN_FLIGHT_FRAMES + 1 + 16;
/// RGB buffers a run can touch. One RGB tensor lives at a time, and numr's
/// conversions can add temporaries of the same size.
const RGB_BUFFERS: usize = 4;

fn gpu() -> Option<CudaClient> {
    if let Err(error) = nvdec_available() {
        eprintln!("skipping: {error}");
        return None;
    }
    if let Err(error) = nvenc_available() {
        eprintln!("skipping: {error}");
        return None;
    }
    if !is_cuda_available() {
        eprintln!("skipping: no CUDA device 0. Run the pipeline tests on a machine with a GPU.");
        return None;
    }
    match CudaClient::new(CudaDevice::new(0)) {
        Ok(client) => Some(client),
        Err(error) => {
            eprintln!("skipping: numr cannot open CUDA device 0: {error}");
            None
        }
    }
}

/// An H.264 source with B-frames and an edit list, and AAC audio to copy.
/// Tests run in parallel processes, so each writes its own file, then renames it.
fn source() -> Option<PathBuf> {
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let path = directory.join("pipeline_source.mp4");
    if path.exists() {
        return Some(path);
    }
    let partial = directory.join(format!("pipeline_source_{}.mp4", std::process::id()));
    let generated = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-f", "lavfi", "-i"])
        .arg(format!("testsrc2=size={WIDTH}x{HEIGHT}:rate=30000/1001"))
        .args(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"])
        .args(["-t", "8.008", "-map", "0:v", "-map", "1:a"])
        .args(["-c:v", "libx264", "-preset", "veryfast", "-bf", "2"])
        .args(["-pix_fmt", "yuv420p", "-video_track_timescale", "30000"])
        .args(["-c:a", "aac"])
        .arg(&partial)
        .output();
    match generated {
        Ok(output) if output.status.success() => {
            fs::rename(&partial, &path)
                .unwrap_or_else(|error| panic!("renaming {}: {error}", partial.display()));
            Some(path)
        }
        Ok(output) => {
            eprintln!(
                "skipping: ffmpeg cannot generate the source: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            None
        }
        Err(error) => {
            eprintln!("skipping: ffmpeg does not run: {error}");
            None
        }
    }
}

/// Converts to RGB and back, with no model.
struct Identity {
    client: CudaClient,
    colors: ColorPlan,
    measures: Arc<Measures>,
}

impl FrameTransform<CudaRuntime> for Identity {
    fn apply(
        &mut self,
        frame: DecodedFrame<CudaRuntime>,
    ) -> Result<OutputPlanes<CudaRuntime>, PipelineError> {
        let layout = ConvLayout::Nchw;
        let rgb = to_rgb(
            &self.client,
            &frame,
            &self.colors.decode,
            DType::F32,
            layout,
        )?;
        Measures::note(&self.measures.rgb_buffers, rgb.ptr());
        Ok(from_rgb(
            &self.client,
            &rgb,
            &self.colors.encode,
            frame.format(),
            layout,
        )?)
    }
}

/// Frames in flight and the device buffers a run touches. numr's allocator keeps
/// no statistics, and `cuMemGetInfo` counts every process on the GPU. So the
/// test counts distinct addresses from numr's free list. That list reuses a
/// freed buffer before it asks the driver for a new one.
#[derive(Default)]
struct Measures {
    emitted: AtomicU64,
    sent: AtomicU64,
    high_water: AtomicU64,
    luma_buffers: Mutex<HashSet<u64>>,
    rgb_buffers: Mutex<HashSet<u64>>,
}

impl Measures {
    fn note(buffers: &Mutex<HashSet<u64>>, address: u64) {
        buffers
            .lock()
            .unwrap_or_else(|error| panic!("the buffer set: {error}"))
            .insert(address);
    }

    fn count(buffers: &Mutex<HashSet<u64>>) -> usize {
        buffers
            .lock()
            .unwrap_or_else(|error| panic!("the buffer set: {error}"))
            .len()
    }
}

struct CountingDecoder {
    inner: NvdecDecoder,
    measures: Arc<Measures>,
}

impl Decoder<CudaRuntime> for CountingDecoder {
    fn send(&mut self, packet: &[u8], pts: i64) -> Result<(), DecodeError> {
        self.inner.send(packet, pts)
    }

    fn flush(&mut self) -> Result<(), DecodeError> {
        self.inner.flush()
    }

    fn receive(&mut self) -> Option<DecodedFrame<CudaRuntime>> {
        let frame = self.inner.receive()?;
        Measures::note(&self.measures.luma_buffers, frame.y().ptr());
        let emitted = self.measures.emitted.fetch_add(1, Ordering::SeqCst) + 1;
        let in_flight = emitted - self.measures.sent.load(Ordering::SeqCst);
        self.measures
            .high_water
            .fetch_max(in_flight, Ordering::SeqCst);
        Some(frame)
    }

    fn route(&self) -> FrameRoute {
        Decoder::route(&self.inner)
    }
}

struct CountingEncoder {
    inner: NvencEncoder,
    measures: Arc<Measures>,
}

impl Encoder<CudaRuntime> for CountingEncoder {
    fn send(
        &mut self,
        frame: &EncodeFrame<'_, CudaRuntime>,
        pts: i64,
        duration: i64,
    ) -> Result<(), EncodeError> {
        self.inner.send(frame, pts, duration)?;
        self.measures.sent.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn flush(&mut self) -> Result<(), EncodeError> {
        self.inner.flush()
    }

    fn receive(&mut self) -> Option<EncodedPacket> {
        self.inner.receive()
    }

    fn config_record(&self) -> Option<&[u8]> {
        self.inner.config_record()
    }

    fn route(&self) -> FrameRoute {
        Encoder::route(&self.inner)
    }
}

/// The luma of each frame of `path`, in display order, read back to the host.
fn decoded_luma(client: &CudaClient, path: &Path) -> Vec<Vec<u8>> {
    let input = Input::open(path).unwrap_or_else(|error| panic!("{error}"));
    let track = input
        .video_track()
        .unwrap_or_else(|error| panic!("{error}"))
        .clone();
    let config = input
        .decode_config(&track)
        .unwrap_or_else(|error| panic!("{error}"));
    let mut decoder = NvdecDecoder::new(client, config.codec, &config.config)
        .unwrap_or_else(|error| panic!("{error}"));
    let mut source = InputSource::new(input).unwrap_or_else(|error| panic!("{error}"));
    let mut data = Vec::new();
    let mut frames = Vec::new();
    let luma = |frame: DecodedFrame<CudaRuntime>| {
        frame
            .y()
            .try_to_vec::<u8>()
            .unwrap_or_else(|error| panic!("reading luma: {error}"))
    };
    while let Some(packet) = source
        .next(&mut data)
        .unwrap_or_else(|error| panic!("{error}"))
    {
        decoder
            .send(&data, packet.pts)
            .unwrap_or_else(|error| panic!("{error}"));
        while let Some(frame) = decoder.receive() {
            frames.push(luma(frame));
        }
    }
    decoder.flush().unwrap_or_else(|error| panic!("{error}"));
    while let Some(frame) = decoder.receive() {
        frames.push(luma(frame));
    }
    frames
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "luma sizes differ");
    let squares: f64 = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    let mean = squares / a.len() as f64;
    if mean == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mean).log10()
    }
}

fn round_trip(kind: OutputKind) {
    let Some(client) = gpu() else {
        return;
    };
    let Some(source_path) = source() else {
        return;
    };
    let output_path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("pipeline_output.{}", kind.extension()));
    let name = kind.to_string();

    let mut input = Input::open(&source_path).unwrap_or_else(|error| panic!("{name}: {error}"));
    let track = input
        .video_track()
        .unwrap_or_else(|error| panic!("{name}: {error}"))
        .clone();
    let plan = plan_copy(&mut input, kind).unwrap_or_else(|error| panic!("{name}: {error}"));
    let expected_tracks = plan.expected_tracks();
    let source_facts = read_stream_facts(&input).unwrap_or_else(|error| panic!("{name}: {error}"));
    let unspecified = ColorFacts {
        matrix_coefficients: MATRIX_UNSPECIFIED,
        transfer_characteristics: TRANSFER_UNSPECIFIED,
        colour_primaries: PRIMARIES_UNSPECIFIED,
        full_range: false,
        chroma_location: None,
    };
    let colors = color_plan(&merge_container(unspecified, input.colour(&track)))
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    let decode = input
        .decode_config(&track)
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    let source = InputSource::new(input).unwrap_or_else(|error| panic!("{name}: {error}"));
    let timing = source.timing();

    let decoder = NvdecDecoder::new(&client, decode.codec, &decode.config)
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    let rate = Rational::new(30000, 1001).unwrap_or_else(|error| panic!("{name}: {error}"));
    let mut config = EncodeConfig::new(
        EncodeCodec::H264,
        WIDTH,
        HEIGHT,
        PixelFormat::Nv12,
        rate,
        timing.time_base,
    );
    config.rate_control = RateControl::VbrQuality { cq: 20 };
    config.color = colors.tags;
    let encoder =
        NvencEncoder::new(&client, config).unwrap_or_else(|error| panic!("{name}: {error}"));

    let start = StartOffset {
        edit: timing
            .timeline
            .map_or(Rational::ZERO, |timeline| timeline.start_offset()),
        container_start: Rational::ZERO,
    };
    let source_timing = TrackTiming::new(
        source_facts.time_base,
        start,
        source_facts.frames.clone(),
        source_facts.hints,
    )
    .unwrap_or_else(|error| panic!("{name}: {error}"));
    let mut video = VideoOutput::new(
        EncodeCodec::H264,
        Vec::new(),
        WIDTH,
        HEIGHT,
        timing.time_base,
    );
    video.colour = colors.tags;
    video.start_offset = source_timing
        .video_offset()
        .unwrap_or_else(|error| panic!("{name}: {error}"));

    let file = File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&output_path)
        .unwrap_or_else(|error| panic!("creating {}: {error}", output_path.display()));
    let sink = MuxSink::new(file, plan, video);

    let measures = Arc::new(Measures::default());
    let decoder = CountingDecoder {
        inner: decoder,
        measures: Arc::clone(&measures),
    };
    let encoder = CountingEncoder {
        inner: encoder,
        measures: Arc::clone(&measures),
    };
    let transform = Identity {
        client: client.clone(),
        colors,
        measures: Arc::clone(&measures),
    };
    let report = watched(move || {
        let control = PumpControl::new();
        run::<CudaRuntime, _, _, _, _, _>(
            source, decoder, transform, encoder, sink, timing, &control,
        )
    })
    .wait()
    .unwrap_or_else(|error| panic!("{name}: {error}"));
    assert_eq!(report.frames_in, FRAMES, "{name}: frames in");
    assert_eq!(report.frames_dropped, 0, "{name}: frames dropped");
    assert_eq!(report.frames_out, FRAMES, "{name}: frames out");
    assert_eq!(report.packets, FRAMES, "{name}: packets");
    assert_eq!(report.host_copied_frames, 0, "{name}: host copied frames");
    assert_eq!(report.host_copied_bytes, 0, "{name}: host copied bytes");

    // Frame count, pts, and total duration, the last-frame duration included.
    let output = Input::open(&output_path).unwrap_or_else(|error| panic!("{name}: {error}"));
    let output_facts = read_stream_facts(&output).unwrap_or_else(|error| panic!("{name}: {error}"));
    let target = Dimensions::new(WIDTH, HEIGHT).unwrap_or_else(|error| panic!("{name}: {error}"));
    check_output(&source_facts, &output_facts, target)
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    if kind == OutputKind::Mp4 {
        let last = |facts: &vidr::StreamFacts| facts.frames.last().map(|frame| frame.duration);
        assert_eq!(
            last(&output_facts),
            last(&source_facts),
            "{name}: last-frame duration"
        );
        let source_ticks = source_timing
            .last_duration_ticks(Rounding::HalfEven)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(last(&output_facts), Some(source_ticks), "{name}");
    }
    let copied = read_track_facts(&output).unwrap_or_else(|error| panic!("{name}: {error}"));
    check_tracks(&expected_tracks, &copied).unwrap_or_else(|error| panic!("{name}: {error}"));

    let original = decoded_luma(&client, &source_path);
    let written = decoded_luma(&client, &output_path);
    assert_eq!(written.len(), original.len(), "{name}: decoded frames");
    for (index, (a, b)) in original.iter().zip(&written).enumerate() {
        let db = psnr(a, b);
        assert!(
            db > FLOOR_DB,
            "{name}: frame {index} luma PSNR {db:.2} dB, floor {FLOOR_DB} dB"
        );
    }

    let high_water = measures.high_water.load(Ordering::SeqCst);
    assert!(
        high_water <= MAX_IN_FLIGHT_FRAMES as u64,
        "{name}: {high_water} frames in flight"
    );
    let luma_buffers = Measures::count(&measures.luma_buffers);
    assert!(
        luma_buffers <= LUMA_BUFFERS,
        "{name}: {luma_buffers} luma buffers over {FRAMES} frames"
    );
    let rgb_buffers = Measures::count(&measures.rgb_buffers);
    assert!(
        rgb_buffers <= RGB_BUFFERS,
        "{name}: {rgb_buffers} RGB buffers over {FRAMES} frames"
    );
}

/// Both outputs run in one test, one after the other. They never share a
/// free-list bucket at the same time.
#[test]
fn nvdec_to_nvenc_keeps_every_frame_and_its_timing() {
    round_trip(OutputKind::Mp4);
    round_trip(OutputKind::Mkv);
}
