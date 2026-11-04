//! The three threads of a run and the channels between them.
//!
//! # Threads and capacities
//!
//! | Thread | Work | Sends to | Capacity |
//! | --- | --- | --- | --- |
//! | T1 | demux and decode | T2 | 2 |
//! | T2 | edit window, then transform | T3 | 1 |
//! | T3 | encode and mux | | |
//!
//! # Ending a run
//!
//! `None` on a channel marks a clean end of stream. A channel that closes with
//! no `None` means the sender stopped early, so the receiver stops too and
//! flushes nothing. A stage whose channel closes reports nothing, so the first
//! real error stays the root cause. A panic drops its stage's channel ends
//! while it unwinds, so it wakes the other stages like an error.
//!
//! # numr's stream
//!
//! This file makes no driver call, holds no permit, and binds no context.
//! `NvdecDecoder` and `NvencEncoder` bind numr's context at the start of each
//! call that reaches the driver. Every numr entry point binds its own.
//!
//! No thread waits on a channel while it holds a permit. Decoder, encoder, and
//! numr calls return before the pump touches a channel. T3's host wait in
//! NVENC `send` holds no permit and waits on one event. T1's and T2's later
//! work cannot extend that wait. It ends when the GPU reaches work already
//! queued, which no host lock delays. So the wait-for graph has no cycle: T1
//! waits on T2, T2 on T3, T3 on the GPU.
//!
//! A transform that captures a CUDA graph holds the write side of the capture
//! lock. T1 and T3 then block at their next permit, and they hold nothing while
//! they block. The capture ends without them, so this adds no cycle either.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Mutex, PoisonError};
use std::thread::{self, Scope, ScopedJoinHandle};

use numr::runtime::Runtime;
#[cfg(any(feature = "nvdec", feature = "nvenc"))]
use numr::runtime::cuda::CudaRuntime;
#[cfg(any(feature = "nvdec", feature = "nvenc"))]
use numr::tensor::Tensor;

use super::control::{PumpControl, PumpReport, count};
use super::present::{Presented, Presenter, Pushed};
use super::traits::{FrameTransform, PacketSink, PacketSource, PumpTiming};
use super::{PipelineError, PumpStage};
use crate::decode::{DecodedFrame, Decoder};
use crate::encode::{EncodeFrame, Encoder};
use crate::frame::OutputPlanes;
use crate::time::{DurationHints, FrameTiming};

/// Frames T1 can queue for T2.
const DECODED_CAPACITY: usize = 2;
/// Frames T2 can queue for T3.
const PLANES_CAPACITY: usize = 1;

/// The most frames between the decoder's `receive` and the end of the
/// encoder's `send`.
///
/// One waits in T1's blocked send. T2 holds one back for its timing and blocks
/// on the planes of another. T3 encodes one. The rest sit in the channels.
pub const MAX_IN_FLIGHT_FRAMES: usize = 1 + DECODED_CAPACITY + 2 + PLANES_CAPACITY + 1;

/// `NvdecDecoder` and `NvencEncoder` declare `Send`, each with its invariant.
/// Each owns its driver objects, binds numr's context on every call, and takes
/// `&mut self`. So the caller builds each on any thread, and the pump moves it.
#[cfg(any(feature = "nvdec", feature = "nvenc"))]
const fn assert_send<T: Send>() {}

#[cfg(any(feature = "nvdec", feature = "nvenc"))]
const _: () = {
    assert_send::<Tensor<CudaRuntime>>();
    assert_send::<DecodedFrame<CudaRuntime>>();
    assert_send::<OutputPlanes<CudaRuntime>>();
};
#[cfg(feature = "nvdec")]
const _: () = assert_send::<crate::decode::nvdec::NvdecDecoder>();
#[cfg(feature = "nvenc")]
const _: () = assert_send::<crate::encode::nvenc::NvencEncoder>();

/// A decoded frame and the duration of the packet that carried it.
struct DecodedItem<R: Runtime> {
    frame: DecodedFrame<R>,
    packet_duration: i64,
}

/// Planes to encode and their output timing.
struct PlanesItem<R: Runtime> {
    planes: OutputPlanes<R>,
    pts: i64,
    duration: i64,
}

/// What the three threads share.
struct Shared<'a> {
    control: &'a PumpControl,
    first_error: Mutex<Option<PipelineError>>,
    /// The last packet's duration in decode order. T1 stores it before its end
    /// marker, and T2 reads it after that marker.
    last_packet_duration: AtomicI64,
}

impl Shared<'_> {
    /// Keeps the first error and asks the other stages to stop.
    fn fail(&self, error: PipelineError) {
        // An `Option` has no partial state, so a poisoned lock is still correct.
        let mut first = self
            .first_error
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        first.get_or_insert(error);
        drop(first);
        self.control.request_stop();
    }
}

/// Runs a whole video through three threads, and returns after all three end.
///
/// T1 reads `source` and runs `decoder`. T2 applies the edit window and runs
/// `transform`. T3 runs `encoder` and writes to `sink`. Each component stays on
/// its own thread and drops there.
///
/// # Timing
///
/// T2 holds one decoded frame back, before the transform. The next frame's pts
/// decides whether the held frame stays in the edit window, and sets its
/// duration: the next output pts minus its own. The last frame's duration comes
/// from the chain in `time/chain.rs`: its packet's duration, the last packet's
/// duration, the frame rate in `timing.hints`, then the gap between the last
/// two pts. These are the rules and inputs of `read_stream_facts` and
/// [`TrackTiming`](crate::time::TrackTiming), so `check_output` sees the same
/// values on both sides. Output pts equal [`EditTimeline::present`].
///
/// [`EditTimeline::present`]: crate::time::EditTimeline::present
///
/// # Memory
///
/// At most [`MAX_IN_FLIGHT_FRAMES`] frames sit between the decoder's `receive`
/// and the end of the encoder's `send`. A decoder queue adds what one packet or
/// the final flush returns. T3 blocked in a slow sink stops T2 from
/// queuing kernels, and then stops T1 from reading packets. The pump allocates
/// a tensor for no frame and copies no frame to the host. Per frame it
/// allocates nothing itself. `EncodedPacket::data` allocates in the encoder.
///
/// # Errors
///
/// - The first error of any stage, its root cause. The other stages stop, and
///   their later errors are not reported.
/// - [`PipelineError::Panic`] for a panic in a stage, and
///   [`PipelineError::Spawn`] when a thread does not start.
/// - [`PipelineError::Stopped`] after [`PumpControl::request_stop`]. A stop that
///   arrives after the sink finishes does not change the result.
/// - [`PipelineError::NoFrames`] when the decoder returns no frame.
/// - [`PipelineError::EditList`] when the edit window keeps no frame.
/// - [`PipelineError::Timing`] for pts that do not increase, or a last frame
///   with no duration source.
/// - [`PipelineError::CountMismatch`] when the encoder's counts differ from the
///   frames decoded and kept.
///
/// On every error the sink drops without [`PacketSink::finish`], so the output
/// is not final. The caller removes it.
///
/// # Hangs
///
/// The pump cannot interrupt a call into a component. A source, decoder,
/// transform, encoder, or sink that never returns holds the run open.
pub fn run<R, S, D, T, E, K>(
    source: S,
    decoder: D,
    transform: T,
    encoder: E,
    sink: K,
    timing: PumpTiming,
    control: &PumpControl,
) -> Result<PumpReport<K::Output>, PipelineError>
where
    R: Runtime,
    S: PacketSource,
    D: Decoder<R> + Send,
    T: FrameTransform<R>,
    E: Encoder<R> + Send,
    K: PacketSink,
{
    let shared = Shared {
        control,
        first_error: Mutex::new(None),
        last_packet_duration: AtomicI64::new(0),
    };
    let shared = &shared;
    let (decoded_in, decoded_out) = sync_channel::<Option<DecodedItem<R>>>(DECODED_CAPACITY);
    let (planes_in, planes_out) = sync_channel::<Option<PlanesItem<R>>>(PLANES_CAPACITY);

    let output = thread::scope(|scope| {
        let decode = spawn(scope, PumpStage::Decode, shared, move || {
            decode_stage(source, decoder, &decoded_in, shared)
        });
        let work = spawn(scope, PumpStage::Transform, shared, move || {
            transform_stage(transform, &decoded_out, &planes_in, timing, shared)
        });
        let encode = spawn(scope, PumpStage::Encode, shared, move || {
            encode_stage(encoder, sink, &planes_out, shared)
        });
        join(decode, PumpStage::Decode, shared);
        join(work, PumpStage::Transform, shared);
        join(encode, PumpStage::Encode, shared).flatten()
    });

    let first_error = shared
        .first_error
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take();
    if let Some(error) = first_error {
        return Err(error);
    }
    let progress = control.progress();
    match output {
        Some(output) => Ok(PumpReport {
            frames_in: progress.frames_in,
            frames_dropped: progress.frames_dropped,
            frames_out: progress.frames_out,
            packets: progress.packets,
            output,
        }),
        None if control.stop_requested() => Err(PipelineError::Stopped {
            frames_in: progress.frames_in,
            packets: progress.packets,
        }),
        None => Err(PipelineError::Incomplete),
    }
}

/// Starts one stage. A thread that does not start drops `work`, which closes
/// its channels, so the other stages end.
fn spawn<'scope, F, O>(
    scope: &'scope Scope<'scope, '_>,
    stage: PumpStage,
    shared: &'scope Shared<'_>,
    work: F,
) -> Option<ScopedJoinHandle<'scope, Option<O>>>
where
    F: FnOnce() -> Result<O, PipelineError> + Send + 'scope,
    O: Send + 'scope,
{
    let spawned = thread::Builder::new()
        .name(stage.thread_name().to_owned())
        .spawn_scoped(scope, move || guarded(stage, shared, work));
    match spawned {
        Ok(handle) => Some(handle),
        Err(error) => {
            shared.fail(PipelineError::Spawn {
                stage,
                reason: error.to_string(),
            });
            None
        }
    }
}

/// Runs a stage, and turns an error or a panic into the shared first error.
fn guarded<F, O>(stage: PumpStage, shared: &Shared<'_>, work: F) -> Option<O>
where
    F: FnOnce() -> Result<O, PipelineError>,
{
    // AssertUnwindSafe: after a panic the stage's own state is dropped unread.
    // The shared state is atomics and a lock that `fail` recovers.
    match catch_unwind(AssertUnwindSafe(work)) {
        Ok(Ok(value)) => Some(value),
        Ok(Err(error)) => {
            shared.fail(error);
            None
        }
        Err(payload) => {
            shared.fail(panic_error(stage, payload.as_ref()));
            None
        }
    }
}

/// Joins a stage. `guarded` catches a stage's panics. This covers a panic that
/// escapes it, such as one from dropping the panic payload.
fn join<O>(
    handle: Option<ScopedJoinHandle<'_, Option<O>>>,
    stage: PumpStage,
    shared: &Shared<'_>,
) -> Option<O> {
    match handle?.join() {
        Ok(value) => value,
        Err(payload) => {
            shared.fail(panic_error(stage, payload.as_ref()));
            None
        }
    }
}

fn panic_error(stage: PumpStage, payload: &(dyn Any + Send)) -> PipelineError {
    let message = payload
        .downcast_ref::<&str>()
        .map(|text| (*text).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a payload that is not text".to_owned());
    PipelineError::Panic { stage, message }
}

/// The durations of packets whose frames the decoder has not returned yet.
///
/// A decoder returns only a frame's pts, and reorders frames. The list holds
/// at most the decoder's reorder depth, so its capacity settles early.
#[derive(Default)]
struct PacketDurations {
    entries: Vec<(i64, i64)>,
}

impl PacketDurations {
    fn push(&mut self, pts: i64, duration: i64) {
        self.entries.push((pts, duration));
    }

    /// The duration of the packet at `pts`, or 0 for none. Frames come in pts
    /// order, so this also forgets each earlier packet. A VP9 packet that shows
    /// no frame leaves this way.
    fn take(&mut self, pts: i64) -> i64 {
        let duration = self
            .entries
            .iter()
            .find(|entry| entry.0 == pts)
            .map_or(0, |entry| entry.1);
        self.entries.retain(|entry| entry.0 > pts);
        duration
    }
}

/// T1. A return without the end marker tells T2 the stream is not complete.
fn decode_stage<R, S, D>(
    mut source: S,
    mut decoder: D,
    frames: &SyncSender<Option<DecodedItem<R>>>,
    shared: &Shared<'_>,
) -> Result<(), PipelineError>
where
    R: Runtime,
    S: PacketSource,
    D: Decoder<R>,
{
    let control = shared.control;
    let mut data = Vec::new();
    let mut durations = PacketDurations::default();
    let mut last_duration = 0;
    while !control.stop_requested() {
        let Some(packet) = source.next(&mut data)? else {
            decoder.flush()?;
            if !send_frames(&mut decoder, frames, &mut durations, control) {
                return Ok(());
            }
            shared
                .last_packet_duration
                .store(last_duration, Ordering::Relaxed);
            // An error means T2 stopped, and T2 reports its own cause.
            let _ = frames.send(None);
            return Ok(());
        };
        count(&control.packets_read);
        last_duration = packet.duration;
        durations.push(packet.pts, packet.duration);
        decoder.send(&data, packet.pts)?;
        if !send_frames(&mut decoder, frames, &mut durations, control) {
            return Ok(());
        }
    }
    Ok(())
}

/// Sends every frame the decoder has ready. Returns `false` when T2 stopped.
fn send_frames<R, D>(
    decoder: &mut D,
    frames: &SyncSender<Option<DecodedItem<R>>>,
    durations: &mut PacketDurations,
    control: &PumpControl,
) -> bool
where
    R: Runtime,
    D: Decoder<R>,
{
    while let Some(frame) = decoder.receive() {
        count(&control.frames_in);
        let packet_duration = durations.take(frame.pts());
        let item = DecodedItem {
            frame,
            packet_duration,
        };
        if frames.send(Some(item)).is_err() {
            return false;
        }
    }
    true
}

/// T2.
fn transform_stage<R, T>(
    mut transform: T,
    frames: &Receiver<Option<DecodedItem<R>>>,
    planes: &SyncSender<Option<PlanesItem<R>>>,
    timing: PumpTiming,
    shared: &Shared<'_>,
) -> Result<(), PipelineError>
where
    R: Runtime,
    T: FrameTransform<R>,
{
    let control = shared.control;
    let mut presenter = Presenter::new(timing.time_base, timing.timeline);
    loop {
        // A closed channel means T1 stopped early, and T1 reports its own cause.
        let Ok(item) = frames.recv() else {
            return Ok(());
        };
        if control.stop_requested() {
            return Ok(());
        }
        let Some(item) = item else {
            break;
        };
        let frame_timing = FrameTiming::new(item.frame.pts(), None, item.packet_duration);
        match presenter.push(frame_timing, item.frame)? {
            Pushed::Held => {}
            Pushed::Dropped => count(&control.frames_dropped),
            Pushed::Emit(presented) => {
                if !send_planes(&mut transform, presented, planes)? {
                    return Ok(());
                }
            }
        }
    }
    let progress = control.progress();
    if progress.frames_in == 0 {
        return Err(PipelineError::NoFrames {
            packets_read: progress.packets_read,
        });
    }
    let last = presenter.finish(last_hints(timing.hints, shared))?;
    if !send_planes(&mut transform, last, planes)? {
        return Ok(());
    }
    // An error means T3 stopped, and T3 reports its own cause.
    let _ = planes.send(None);
    Ok(())
}

/// The caller's hints with the last packet's duration from T1.
fn last_hints(mut hints: DurationHints, shared: &Shared<'_>) -> DurationHints {
    let duration = shared.last_packet_duration.load(Ordering::Relaxed);
    if duration > 0 {
        hints.packet_duration = Some(duration);
    }
    hints
}

/// Runs the transform on a kept frame and sends its planes. Returns `false` when
/// T3 stopped.
fn send_planes<R, T>(
    transform: &mut T,
    presented: Presented<DecodedFrame<R>>,
    planes: &SyncSender<Option<PlanesItem<R>>>,
) -> Result<bool, PipelineError>
where
    R: Runtime,
    T: FrameTransform<R>,
{
    let item = PlanesItem {
        planes: transform.apply(presented.item)?,
        pts: presented.pts,
        duration: presented.duration,
    };
    Ok(planes.send(Some(item)).is_ok())
}

/// T3. It returns `None` when the run stopped short of the sink's finish.
fn encode_stage<R, E, K>(
    mut encoder: E,
    mut sink: K,
    planes: &Receiver<Option<PlanesItem<R>>>,
    shared: &Shared<'_>,
) -> Result<Option<K::Output>, PipelineError>
where
    R: Runtime,
    E: Encoder<R>,
    K: PacketSink,
{
    let control = shared.control;
    let mut started = false;
    loop {
        // A closed channel means T2 stopped early, and T2 or T1 reports the cause.
        let Ok(item) = planes.recv() else {
            return Ok(None);
        };
        if control.stop_requested() {
            return Ok(None);
        }
        let Some(item) = item else {
            break;
        };
        // The planes stay alive until `send` returns, past the copy that reads them.
        encoder.send(&EncodeFrame::from(&item.planes), item.pts, item.duration)?;
        count(&control.frames_out);
        drop(item);
        write_packets(&mut encoder, &mut sink, &mut started, control)?;
    }
    encoder.flush()?;
    write_packets(&mut encoder, &mut sink, &mut started, control)?;
    let progress = control.progress();
    let kept = progress.frames_in.saturating_sub(progress.frames_dropped);
    if progress.frames_out != kept || progress.packets != kept {
        return Err(PipelineError::CountMismatch {
            frames_in: progress.frames_in,
            frames_dropped: progress.frames_dropped,
            frames_out: progress.frames_out,
            packets: progress.packets,
        });
    }
    if control.stop_requested() {
        return Ok(None);
    }
    sink.finish().map(Some)
}

/// Passes every packet the encoder has ready to the sink. The first packet
/// starts the sink with the encoder's config record.
fn write_packets<R, E, K>(
    encoder: &mut E,
    sink: &mut K,
    started: &mut bool,
    control: &PumpControl,
) -> Result<(), PipelineError>
where
    R: Runtime,
    E: Encoder<R>,
    K: PacketSink,
{
    while let Some(packet) = encoder.receive() {
        if !*started {
            let Some(record) = encoder.config_record() else {
                return Err(PipelineError::NoConfigRecord { pts: packet.pts });
            };
            sink.start(record)?;
            *started = true;
        }
        sink.write(packet)?;
        count(&control.packets);
    }
    Ok(())
}
