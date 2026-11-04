//! A fake source, decoder, transform, encoder, and sink over CPU tensors, the
//! counters they share, and the watchdog.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use numr::dtype::DType;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use vidr::decode::{
    ColorFacts, DecodedFrame, Decoder, MATRIX_UNSPECIFIED, PRIMARIES_UNSPECIFIED, PixelFormat,
    TRANSFER_UNSPECIFIED,
};
use vidr::encode::{EncodeFrame, EncodedPacket, Encoder};
use vidr::frame::OutputPlanes;
use vidr::pipeline::{
    FrameTransform, PacketSink, PacketSource, PipelineError, PumpControl, PumpReport, PumpStage,
    PumpTiming, SourcePacket, run,
};
use vidr::{DecodeError, Dimensions, EncodeError, FrameRoute, Rational};

const WATCHDOG: Duration = Duration::from_secs(120);

/// A step that can fail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Point {
    Source,
    DecoderSend,
    DecoderFlush,
    Transform,
    EncoderSend,
    EncoderFlush,
    SinkStart,
    SinkWrite,
    SinkFinish,
}

impl Point {
    pub(crate) const ALL: [Self; 9] = [
        Self::Source,
        Self::DecoderSend,
        Self::DecoderFlush,
        Self::Transform,
        Self::EncoderSend,
        Self::EncoderFlush,
        Self::SinkStart,
        Self::SinkWrite,
        Self::SinkFinish,
    ];

    /// The thread the step runs on.
    pub(crate) const fn stage(self) -> PumpStage {
        match self {
            Self::Source | Self::DecoderSend | Self::DecoderFlush => PumpStage::Decode,
            Self::Transform => PumpStage::Transform,
            Self::EncoderSend
            | Self::EncoderFlush
            | Self::SinkStart
            | Self::SinkWrite
            | Self::SinkFinish => PumpStage::Encode,
        }
    }

    /// The call the step fails on: a step that runs once, or the sixth.
    pub(crate) const fn call(self) -> u64 {
        match self {
            Self::DecoderFlush | Self::EncoderFlush | Self::SinkStart | Self::SinkFinish => 0,
            _ => 5,
        }
    }

    /// The error the step returns.
    pub(crate) fn error(self) -> PipelineError {
        match self {
            Self::DecoderSend => DecodeError::EmptyPacket { pts: -1 }.into(),
            Self::DecoderFlush => DecodeError::AfterFlush { pts: -1 }.into(),
            Self::EncoderSend => EncodeError::AfterFlush { pts: -1 }.into(),
            Self::EncoderFlush => EncodeError::AfterFlush { pts: -2 }.into(),
            other => PipelineError::failed(format!("{other:?}"), "injected"),
        }
    }
}

/// What a step does when it fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fault {
    Error,
    Panic,
    PanicWithNoText,
}

/// A gate a sink waits on.
#[derive(Default)]
pub(crate) struct Gate {
    open: Mutex<bool>,
    changed: Condvar,
}

impl Gate {
    pub(crate) fn open(&self) {
        *self.open.lock().unwrap() = true;
        self.changed.notify_all();
    }

    fn wait(&self) {
        let mut open = self.open.lock().unwrap();
        while !*open {
            open = self.changed.wait(open).unwrap();
        }
    }
}

/// What a run does.
#[derive(Default)]
pub(crate) struct Script {
    /// `(pts, duration)` of each packet in decode order.
    pub(crate) packets: Vec<(i64, i64)>,
    /// Frames the decoder holds back to reorder them into pts order.
    pub(crate) reorder: usize,
    /// The decoder returns no frame.
    pub(crate) decoder_returns_nothing: bool,
    pub(crate) fault: Option<(Point, Fault)>,
    /// The encoder loses the frame at this index.
    pub(crate) encoder_loses: Option<u64>,
    /// The encoder returns no config record.
    pub(crate) no_config_record: bool,
    pub(crate) sink_delay: Duration,
    pub(crate) sink_gate: Option<Arc<Gate>>,
    /// The sink asks for a stop before this write.
    pub(crate) stop_at_write: Option<(Arc<PumpControl>, u64)>,
    /// The decoder's route. `None` means `FrameRoute::Device`.
    pub(crate) decoder_route: Option<FrameRoute>,
    /// The encoder's route. `None` means `FrameRoute::Device`.
    pub(crate) encoder_route: Option<FrameRoute>,
    /// The transform returns planes of `[4, 4]` and `[2, 4]`, not `[2, 2]` and
    /// `[1, 2]`.
    pub(crate) transform_doubles: bool,
}

/// What the fakes measure.
#[derive(Default)]
pub(crate) struct Probe {
    pub(crate) reads: AtomicU64,
    pub(crate) emitted: AtomicU64,
    pub(crate) applied: AtomicU64,
    pub(crate) sent: AtomicU64,
    pub(crate) written: AtomicU64,
    /// The largest count of frames between the decoder's `receive` and the
    /// end of the sink's `write`.
    pub(crate) high_water: AtomicU64,
    pub(crate) decoder_flushed: AtomicU64,
    pub(crate) encoder_flushed: AtomicU64,
    pub(crate) sink_started: AtomicU64,
    pub(crate) sink_finished: AtomicU64,
    /// Fakes dropped. All five dropped means every stage ended.
    pub(crate) drops: AtomicU64,
    /// Media pts of each frame the transform takes.
    pub(crate) applied_pts: Mutex<Vec<i64>>,
    /// `(pts, duration)` of each frame the encoder takes.
    pub(crate) encoded: Mutex<Vec<(i64, i64)>>,
    pub(crate) written_pts: Mutex<Vec<i64>>,
}

impl Probe {
    pub(crate) fn read(counter: &AtomicU64) -> u64 {
        counter.load(Ordering::SeqCst)
    }

    fn add(counter: &AtomicU64) -> u64 {
        counter.fetch_add(1, Ordering::SeqCst) + 1
    }

    pub(crate) fn encoded(&self) -> Vec<(i64, i64)> {
        self.encoded.lock().unwrap().clone()
    }

    pub(crate) fn applied_pts(&self) -> Vec<i64> {
        self.applied_pts.lock().unwrap().clone()
    }

    pub(crate) fn written_pts(&self) -> Vec<i64> {
        self.written_pts.lock().unwrap().clone()
    }
}

/// One step's fault, shared by the fakes that run it.
#[derive(Clone, Copy)]
struct Faults(Option<(Point, Fault)>);

impl Faults {
    /// Fails when `point` is the scripted step and this is its scripted call.
    fn check(self, point: Point, call: u64) -> Result<(), PipelineError> {
        match self.0 {
            Some((scripted, fault)) if scripted == point && call == point.call() => match fault {
                Fault::Error => Err(point.error()),
                Fault::Panic => panic!("injected panic at {point:?}"),
                Fault::PanicWithNoText => std::panic::panic_any(7_u8),
            },
            _ => Ok(()),
        }
    }
}

struct FakeSource {
    packets: VecDeque<(i64, i64)>,
    faults: Faults,
    probe: Arc<Probe>,
}

impl PacketSource for FakeSource {
    fn next(&mut self, data: &mut Vec<u8>) -> Result<Option<SourcePacket>, PipelineError> {
        self.faults
            .check(Point::Source, Probe::read(&self.probe.reads))?;
        let Some((pts, duration)) = self.packets.pop_front() else {
            return Ok(None);
        };
        Probe::add(&self.probe.reads);
        data.clear();
        data.push(1);
        Ok(Some(SourcePacket { pts, duration }))
    }
}

struct FakeDecoder {
    held: BinaryHeap<Reverse<i64>>,
    ready: VecDeque<i64>,
    reorder: usize,
    returns_nothing: bool,
    sends: u64,
    route: FrameRoute,
    faults: Faults,
    probe: Arc<Probe>,
}

fn plane(shape: &[usize]) -> Tensor<CpuRuntime> {
    Tensor::empty(shape, DType::U8, &CpuDevice::new()).unwrap()
}

fn frame(pts: i64) -> DecodedFrame<CpuRuntime> {
    DecodedFrame::from_planes(
        plane(&[2, 2]),
        plane(&[1, 2]),
        PixelFormat::Nv12,
        pts,
        Dimensions::new(2, 2).unwrap(),
        ColorFacts {
            matrix_coefficients: MATRIX_UNSPECIFIED,
            transfer_characteristics: TRANSFER_UNSPECIFIED,
            colour_primaries: PRIMARIES_UNSPECIFIED,
            full_range: false,
            chroma_location: None,
        },
    )
}

impl FakeDecoder {
    fn fail(error: PipelineError) -> DecodeError {
        match error {
            PipelineError::Decode(error) => error,
            other => panic!("a decoder step gives {other}"),
        }
    }
}

impl Decoder<CpuRuntime> for FakeDecoder {
    fn send(&mut self, _packet: &[u8], pts: i64) -> Result<(), DecodeError> {
        self.faults
            .check(Point::DecoderSend, self.sends)
            .map_err(Self::fail)?;
        self.sends += 1;
        if self.returns_nothing {
            return Ok(());
        }
        self.held.push(Reverse(pts));
        while self.held.len() > self.reorder {
            if let Some(Reverse(next)) = self.held.pop() {
                self.ready.push_back(next);
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), DecodeError> {
        self.faults
            .check(Point::DecoderFlush, 0)
            .map_err(Self::fail)?;
        Probe::add(&self.probe.decoder_flushed);
        while let Some(Reverse(next)) = self.held.pop() {
            self.ready.push_back(next);
        }
        Ok(())
    }

    fn receive(&mut self) -> Option<DecodedFrame<CpuRuntime>> {
        let pts = self.ready.pop_front()?;
        let emitted = Probe::add(&self.probe.emitted);
        let in_flight = emitted - Probe::read(&self.probe.written);
        self.probe.high_water.fetch_max(in_flight, Ordering::SeqCst);
        Some(frame(pts))
    }

    fn route(&self) -> FrameRoute {
        self.route
    }
}

struct FakeTransform {
    calls: u64,
    doubles: bool,
    faults: Faults,
    probe: Arc<Probe>,
}

impl FrameTransform<CpuRuntime> for FakeTransform {
    fn apply(
        &mut self,
        frame: DecodedFrame<CpuRuntime>,
    ) -> Result<OutputPlanes<CpuRuntime>, PipelineError> {
        self.faults.check(Point::Transform, self.calls)?;
        self.calls += 1;
        self.probe.applied_pts.lock().unwrap().push(frame.pts());
        Probe::add(&self.probe.applied);
        if self.doubles {
            return Ok(OutputPlanes::new(plane(&[4, 4]), plane(&[2, 4])));
        }
        let (y, uv) = frame.into_planes();
        Ok(OutputPlanes::new(y, uv))
    }
}

struct FakeEncoder {
    queue: VecDeque<EncodedPacket>,
    sends: u64,
    loses: Option<u64>,
    no_config_record: bool,
    luma: [usize; 2],
    route: FrameRoute,
    faults: Faults,
    probe: Arc<Probe>,
}

impl FakeEncoder {
    fn fail(error: PipelineError) -> EncodeError {
        match error {
            PipelineError::Encode(error) => error,
            other => panic!("an encoder step gives {other}"),
        }
    }
}

impl Encoder<CpuRuntime> for FakeEncoder {
    fn send(
        &mut self,
        frame: &EncodeFrame<'_, CpuRuntime>,
        pts: i64,
        duration: i64,
    ) -> Result<(), EncodeError> {
        assert_eq!(
            frame.y().shape(),
            self.luma,
            "luma of the frame at pts {pts}"
        );
        self.faults
            .check(Point::EncoderSend, self.sends)
            .map_err(Self::fail)?;
        let index = self.sends;
        self.sends += 1;
        self.probe.encoded.lock().unwrap().push((pts, duration));
        Probe::add(&self.probe.sent);
        if self.loses != Some(index) {
            self.queue.push_back(EncodedPacket {
                data: vec![1],
                pts,
                dts: pts,
                duration,
                keyframe: true,
            });
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), EncodeError> {
        self.faults
            .check(Point::EncoderFlush, 0)
            .map_err(Self::fail)?;
        Probe::add(&self.probe.encoder_flushed);
        Ok(())
    }

    fn receive(&mut self) -> Option<EncodedPacket> {
        self.queue.pop_front()
    }

    fn config_record(&self) -> Option<&[u8]> {
        (!self.no_config_record).then_some(b"record".as_slice())
    }

    fn route(&self) -> FrameRoute {
        self.route
    }
}

struct FakeSink {
    writes: u64,
    delay: Duration,
    gate: Option<Arc<Gate>>,
    stop_at_write: Option<(Arc<PumpControl>, u64)>,
    faults: Faults,
    probe: Arc<Probe>,
}

impl PacketSink for FakeSink {
    type Output = u64;

    fn start(&mut self, config_record: &[u8]) -> Result<(), PipelineError> {
        assert_eq!(config_record, b"record");
        self.faults.check(Point::SinkStart, 0)?;
        Probe::add(&self.probe.sink_started);
        Ok(())
    }

    fn write(&mut self, packet: EncodedPacket) -> Result<(), PipelineError> {
        assert_eq!(
            Probe::read(&self.probe.sink_started),
            1,
            "a write before `start`"
        );
        if let Some(gate) = &self.gate {
            gate.wait();
        }
        if let Some((control, at)) = &self.stop_at_write
            && *at == self.writes
        {
            control.request_stop();
        }
        if !self.delay.is_zero() {
            thread::sleep(self.delay);
        }
        self.faults.check(Point::SinkWrite, self.writes)?;
        self.writes += 1;
        self.probe.written_pts.lock().unwrap().push(packet.pts);
        Probe::add(&self.probe.written);
        Ok(())
    }

    fn finish(self) -> Result<u64, PipelineError> {
        self.faults.check(Point::SinkFinish, 0)?;
        Probe::add(&self.probe.sink_finished);
        Ok(self.writes)
    }
}

/// Counts drops, one impl per fake.
macro_rules! count_drop {
    ($($fake:ty),+) => {
        $(impl Drop for $fake {
            fn drop(&mut self) {
                Probe::add(&self.probe.drops);
            }
        })+
    };
}
count_drop!(
    FakeSource,
    FakeDecoder,
    FakeTransform,
    FakeEncoder,
    FakeSink
);

/// A result that arrives on another thread, under the watchdog.
pub(crate) struct Watched<T>(Receiver<T>);

impl<T> Watched<T> {
    /// The result. A run that passes the watchdog's limit fails the test.
    pub(crate) fn wait(self) -> T {
        match self.0.recv_timeout(WATCHDOG) {
            Ok(value) => value,
            Err(RecvTimeoutError::Timeout) => panic!("the pump hangs: no result in {WATCHDOG:?}"),
            Err(RecvTimeoutError::Disconnected) => panic!("the thread that runs the pump panics"),
        }
    }
}

/// Runs `work` on a new thread.
pub(crate) fn watched<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> Watched<T> {
    let (done, result) = mpsc::channel();
    thread::spawn(move || {
        // The receiver is gone only after a test already failed.
        let _ = done.send(work());
    });
    Watched(result)
}

pub(crate) type RunResult = Result<PumpReport<u64>, PipelineError>;

/// The five fakes of one run.
struct Fakes {
    source: FakeSource,
    decoder: FakeDecoder,
    transform: FakeTransform,
    encoder: FakeEncoder,
    sink: FakeSink,
    probe: Arc<Probe>,
}

fn fakes(script: Script) -> Fakes {
    let probe = Arc::new(Probe::default());
    let faults = Faults(script.fault);
    let source = FakeSource {
        packets: script.packets.into_iter().collect(),
        faults,
        probe: Arc::clone(&probe),
    };
    let decoder = FakeDecoder {
        held: BinaryHeap::new(),
        ready: VecDeque::new(),
        reorder: script.reorder,
        returns_nothing: script.decoder_returns_nothing,
        sends: 0,
        route: script.decoder_route.unwrap_or(FrameRoute::Device),
        faults,
        probe: Arc::clone(&probe),
    };
    let transform = FakeTransform {
        calls: 0,
        doubles: script.transform_doubles,
        faults,
        probe: Arc::clone(&probe),
    };
    let encoder = FakeEncoder {
        queue: VecDeque::new(),
        sends: 0,
        loses: script.encoder_loses,
        no_config_record: script.no_config_record,
        luma: if script.transform_doubles {
            [4, 4]
        } else {
            [2, 2]
        },
        route: script.encoder_route.unwrap_or(FrameRoute::Device),
        faults,
        probe: Arc::clone(&probe),
    };
    let sink = FakeSink {
        writes: 0,
        delay: script.sink_delay,
        gate: script.sink_gate,
        stop_at_write: script.stop_at_write,
        faults,
        probe: Arc::clone(&probe),
    };
    Fakes {
        source,
        decoder,
        transform,
        encoder,
        sink,
        probe,
    }
}

/// Starts a run of `script`.
pub(crate) fn launch(
    script: Script,
    timing: PumpTiming,
    control: &Arc<PumpControl>,
) -> (Arc<Probe>, Watched<RunResult>) {
    let Fakes {
        source,
        decoder,
        transform,
        encoder,
        sink,
        probe,
    } = fakes(script);
    let control = Arc::clone(control);
    let result = watched(move || {
        run::<CpuRuntime, _, _, _, _, _>(
            source, decoder, transform, encoder, sink, timing, &control,
        )
    });
    (probe, result)
}

/// Runs `script` to its end with the decoder and encoder as boxed trait objects.
pub(crate) fn pump_boxed(script: Script, timing: PumpTiming) -> RunResult {
    let Fakes {
        source,
        decoder,
        transform,
        encoder,
        sink,
        probe: _,
    } = fakes(script);
    let decoder: Box<dyn Decoder<CpuRuntime> + Send> = Box::new(decoder);
    let encoder: Box<dyn Encoder<CpuRuntime> + Send> = Box::new(encoder);
    let control = PumpControl::new();
    watched(move || {
        run::<CpuRuntime, _, _, _, _, _>(
            source, decoder, transform, encoder, sink, timing, &control,
        )
    })
    .wait()
}

/// Runs `script` to its end.
pub(crate) fn pump(script: Script, timing: PumpTiming) -> (RunResult, Arc<Probe>) {
    let control = Arc::new(PumpControl::new());
    let (probe, result) = launch(script, timing, &control);
    (result.wait(), probe)
}

/// A tick of 1/1000 s.
pub(crate) fn millis() -> PumpTiming {
    PumpTiming::new(Rational::new(1, 1000).unwrap())
}

/// `count` packets 10 ticks apart, each of 10 ticks.
pub(crate) fn steady(count: i64) -> Vec<(i64, i64)> {
    (0..count).map(|index| (index * 10, 10)).collect()
}
