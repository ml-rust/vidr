//! `HevcDecoder` over the worker pool. The calling thread walks the NAL units as the
//! single-thread path does, and keeps its actions in one queue: each picture's end and each
//! output, in that path's order. It applies them as the pictures finish. So an error of a
//! picture comes where that path reports it, and the frames before it are the same.

use std::collections::VecDeque;
use std::sync::Arc;

use numr::dtype::DType;
use numr::runtime::Runtime;

use super::super::cpu::dsp::sample::Sample;
use super::super::cpu::hevc::dpb::buffer::Frame;
use super::super::cpu::hevc::error::HevcError;
use super::super::cpu::hevc::picture::DisplayWindow;
use super::super::cpu::hevc::stage::UnitSite;
use super::super::cpu::hevc::stream::{DecodeOptions, DecodedPicture, StreamDecoder};
use super::super::cpu::hevc::threads::completion::PictureFailure;
use super::super::cpu::hevc::threads::handle::PictureHandle;
use super::super::cpu::hevc::threads::pool::WorkerPool;
use super::super::cpu::hevc::threads::stage::{StageEvent, ThreadedPictures};
use super::super::{DecodedFrame, PixelFormat};
use super::decoder::{
    DECODER, Engine, HevcDecoder, PendingPicture, check_sps, core_error, end_error, to_frame,
};
use crate::error::DecodeError;

/// `MaxDpbSize` at any level (A-2). A corrupt SPS cannot raise the cap past this.
const MAX_DPB_PICTURES: usize = 16;

type ThreadedStream = StreamDecoder<ThreadedPictures<u8>, ThreadedPictures<u16>>;
type Ticket = DecodedPicture<PictureHandle<u8>, PictureHandle<u16>>;

/// One action of the single-thread path, in its order.
enum Step {
    /// The end of a picture: its result is the picture's error, if any.
    Retire(Ticket),
    /// A frame to deliver. The picture's `Retire` comes first, so it has finished.
    Output {
        picture: Ticket,
        pending: PendingPicture,
    },
}

/// The core on the pool, and the actions not applied yet.
pub(super) struct ThreadedEngine {
    stream: ThreadedStream,
    /// Joined on drop, after `stream` drops its handles.
    pool: WorkerPool,
    threads: usize,
    steps: VecDeque<Step>,
    /// The `Retire` steps in `steps`: pictures with buffers that the queue keeps alive.
    in_flight: usize,
    /// The picture buffer size of the latest SPS.
    dpb_size: usize,
    /// The most pictures in flight so far.
    #[cfg(test)]
    peak_in_flight: usize,
}

impl ThreadedEngine {
    /// Starts `threads` workers.
    pub(super) fn start(threads: usize, options: DecodeOptions) -> Result<Self, HevcError> {
        let pool = WorkerPool::start(threads)?;
        let stream = StreamDecoder::with_stages(
            options,
            ThreadedPictures::new(pool.spawner()),
            ThreadedPictures::new(pool.spawner()),
        );
        Ok(Self {
            stream,
            pool,
            threads,
            steps: VecDeque::new(),
            in_flight: 0,
            dpb_size: 1,
            #[cfg(test)]
            peak_in_flight: 0,
        })
    }

    /// Takes one unit of the `hvcC` record.
    pub(super) fn config_unit(&mut self, nal: &[u8]) -> Result<(), HevcError> {
        self.stream.decode_nal(nal)
    }

    /// Asks each task to end, and joins the workers.
    pub(super) fn stop(&mut self) {
        self.pool.shutdown();
    }

    /// The most pictures in flight. Each holds a frame, maps, and parsed CTUs, so this cap
    /// bounds memory on any stream.
    const fn cap(&self) -> usize {
        self.threads + self.dpb_size
    }
}

impl<R> HevcDecoder<R>
where
    R: Runtime<DType = DType>,
{
    /// [`Self::decode_unit`] on the pool. An error on this thread first applies every earlier
    /// action, because an earlier picture's error comes first.
    pub(super) fn threaded_unit(&mut self, nal: &[u8], pts: i64) -> Result<(), DecodeError> {
        if let Err(error) = check_sps(nal) {
            return Err(self.settle(error));
        }
        let Engine::Threaded(engine) = &mut self.engine else {
            return Ok(());
        };
        engine.stream.set_pts(pts);
        let result = engine.stream.decode_nal(nal);
        let started = engine.stream.take_started();
        if let Some(picture) = started {
            engine.dpb_size = picture.dpb_size.clamp(1, MAX_DPB_PICTURES);
        }
        self.note_started(started, pts);
        let queued = self.take_events();
        if queued.is_ok() {
            self.drop_dropped(started);
        }
        let failed = queued.err().or_else(|| {
            result.err().map(|error| {
                let input = format!("the {}-byte NAL unit at pts {pts}", nal.len());
                core_error("StreamDecoder::decode_nal", &input, error)
            })
        });
        if let Some(error) = failed {
            return Err(self.settle(error));
        }
        self.advance_ready()?;
        self.hold_cap()
    }

    /// [`Self::finish`] on the pool: the end of the stream, then every action.
    pub(super) fn threaded_finish(&mut self) -> Result<(), DecodeError> {
        let Engine::Threaded(engine) = &mut self.engine else {
            return Ok(());
        };
        let ended = engine.stream.flush();
        let queued = self.take_events();
        let failed = queued.err().or_else(|| ended.err().map(end_error));
        if let Some(error) = failed {
            return Err(self.settle(error));
        }
        while self.step(true)? {}
        Ok(())
    }

    /// Moves the core's events to the action queue. The send budget and the pts of each
    /// output apply here, because this is where the single-thread path outputs it.
    fn take_events(&mut self) -> Result<(), DecodeError> {
        loop {
            let Engine::Threaded(engine) = &mut self.engine else {
                return Ok(());
            };
            let Some(event) = engine.stream.next_output() else {
                return Ok(());
            };
            let (ticket, is_output) = match event {
                DecodedPicture::Eight(StageEvent::Retire(picture)) => {
                    (DecodedPicture::Eight(picture), false)
                }
                DecodedPicture::Eight(StageEvent::Output(picture)) => {
                    (DecodedPicture::Eight(picture), true)
                }
                DecodedPicture::Ten(StageEvent::Retire(picture)) => {
                    (DecodedPicture::Ten(picture), false)
                }
                DecodedPicture::Ten(StageEvent::Output(picture)) => {
                    (DecodedPicture::Ten(picture), true)
                }
            };
            if !is_output {
                engine.steps.push_back(Step::Retire(ticket));
                engine.in_flight += 1;
                #[cfg(test)]
                {
                    engine.peak_in_flight = engine.peak_in_flight.max(engine.in_flight);
                }
                continue;
            }
            self.charge_send(window_of(&ticket))?;
            let pending = self
                .pending
                .remove(&decode_order_of(&ticket))
                .ok_or(DecodeError::UnmatchedPicture { decoder: DECODER })?;
            if let Engine::Threaded(engine) = &mut self.engine {
                engine.steps.push_back(Step::Output {
                    picture: ticket,
                    pending,
                });
            }
        }
    }

    /// Applies each action whose picture has finished, with no wait.
    pub(super) fn advance_ready(&mut self) -> Result<(), DecodeError> {
        while self.step(false)? {}
        Ok(())
    }

    /// Blocks while the pictures in flight reach their cap.
    fn hold_cap(&mut self) -> Result<(), DecodeError> {
        while let Engine::Threaded(engine) = &self.engine
            && engine.in_flight >= engine.cap()
        {
            if !self.step(true)? {
                break;
            }
        }
        Ok(())
    }

    /// Applies every action in order, and returns the first error among them, or `error`.
    fn settle(&mut self, error: DecodeError) -> DecodeError {
        loop {
            match self.step(true) {
                Ok(true) => {}
                Ok(false) => return error,
                Err(first) => return first,
            }
        }
    }

    /// Applies the first action. Returns false when none is left, or when `block` is false
    /// and its picture is running.
    fn step(&mut self, block: bool) -> Result<bool, DecodeError> {
        let Engine::Threaded(engine) = &mut self.engine else {
            return Ok(false);
        };
        match engine.steps.front() {
            None => Ok(false),
            Some(Step::Retire(ticket)) => {
                let Some(result) = settled(ticket, block) else {
                    return Ok(false);
                };
                let order = decode_order_of(ticket);
                engine.steps.pop_front();
                engine.in_flight -= 1;
                match result {
                    Ok(()) | Err(PictureFailure::Abandoned) => Ok(true),
                    Err(failure) => Err(self.failed_picture(failure, order)),
                }
            }
            Some(Step::Output { .. }) => {
                if let Some(Step::Output { picture, pending }) = engine.steps.pop_front() {
                    self.deliver(&picture, pending)?;
                }
                Ok(true)
            }
        }
    }

    /// The error of the picture at `order`. A picture that fails at the end of the stream is
    /// the last one. On one thread, the waiting pictures still leave before its error.
    fn failed_picture(&mut self, failure: PictureFailure, order: u64) -> DecodeError {
        let at_end = matches!(
            failure,
            PictureFailure::Decode {
                site: UnitSite::End,
                ..
            }
        );
        if at_end {
            while let Engine::Threaded(engine) = &mut self.engine
                && let Some(step) = engine.steps.pop_front()
            {
                if let Step::Output { picture, pending } = step
                    && decode_order_of(&picture) != order
                    && let Err(error) = self.deliver(&picture, pending)
                {
                    return error;
                }
            }
        }
        failure_error(failure)
    }

    /// Queues the frame of a finished picture.
    fn deliver(&mut self, picture: &Ticket, pending: PendingPicture) -> Result<(), DecodeError> {
        let frame = match picture {
            DecodedPicture::Eight(handle) => {
                let frame = handle.frame().map_err(failure_error)?;
                self.upload(&frame, pending, PixelFormat::Nv12)?
            }
            DecodedPicture::Ten(handle) => {
                let frame = handle.frame().map_err(failure_error)?;
                self.upload(&frame, pending, PixelFormat::P010)?
            }
        };
        self.frames.push_back(frame);
        Ok(())
    }

    fn upload<S: Sample>(
        &mut self,
        frame: &Arc<Frame<S>>,
        pending: PendingPicture,
        format: PixelFormat,
    ) -> Result<DecodedFrame<R>, DecodeError> {
        to_frame::<R, S>(&self.device, frame, pending, format, &mut self.scratch)
    }
}

fn settled(ticket: &Ticket, block: bool) -> Option<Result<(), PictureFailure>> {
    match ticket {
        DecodedPicture::Eight(handle) => handle.settled(block),
        DecodedPicture::Ten(handle) => handle.settled(block),
    }
}

fn window_of(ticket: &Ticket) -> DisplayWindow {
    match ticket {
        DecodedPicture::Eight(handle) => handle.window(),
        DecodedPicture::Ten(handle) => handle.window(),
    }
}

fn decode_order_of(ticket: &Ticket) -> u64 {
    match ticket {
        DecodedPicture::Eight(handle) => handle.decode_order(),
        DecodedPicture::Ten(handle) => handle.decode_order(),
    }
}

/// Names a picture's failure as the single-thread path names the error at that NAL unit.
fn failure_error(failure: PictureFailure) -> DecodeError {
    match failure {
        PictureFailure::Decode {
            error,
            site: UnitSite::Nal { bytes, pts },
        } => {
            let input = format!("the {bytes}-byte NAL unit at pts {pts}");
            core_error("StreamDecoder::decode_nal", &input, error)
        }
        PictureFailure::Decode {
            error,
            site: UnitSite::End,
        } => end_error(error),
        PictureFailure::Abandoned => end_error(HevcError::WorkerStopped),
    }
}

#[cfg(test)]
#[path = "threaded_tests.rs"]
mod tests;
