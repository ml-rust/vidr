//! The stages of one picture on a worker. A stage holds no lock while it works, and waits
//! for no other picture, so a stage that runs always ends.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};

use crate::decode::cpu::dsp::sample::Sample;
use crate::decode::cpu::hevc::dpb::buffer::Frame;
use crate::decode::cpu::hevc::error::HevcError;

use super::completion::{Gate, PictureFailure, Resolver};
use super::handle::PictureHandle;
use super::job::{PictureJob, PicturePools};
use super::pool::Spawner;

/// The name of a stage in a panic's error.
const PARSE_STAGE: &str = "parse";
/// The name of a stage in a panic's error.
const RECONSTRUCT_STAGE: &str = "reconstruct";

type PictureResolver<S> = Resolver<Arc<Frame<S>>>;

/// One picture between its stages.
struct PictureRun<S> {
    /// The job and its resolver. Empty while a stage runs and after an early end. A run that
    /// drops with them here stores a named failure through the resolver.
    slot: Mutex<Option<(PictureJob<S>, PictureResolver<S>)>>,
    pools: Arc<PicturePools<S>>,
    spawner: Spawner,
}

impl<S: Sample> PictureRun<S> {
    /// The job, or `None` when it has ended. A poisoned lock drops the job's resolver, which
    /// names the failure.
    fn take(&self) -> Option<(PictureJob<S>, PictureResolver<S>)> {
        self.slot.lock().ok()?.take()
    }

    fn put(&self, job: PictureJob<S>, resolver: PictureResolver<S>) {
        if let Ok(mut slot) = self.slot.lock() {
            *slot = Some((job, resolver));
        }
    }
}

/// Runs `work`, and turns a panic into a named error. After a panic, the stage's state is
/// dropped unread.
pub(super) fn guarded<T>(
    stage: &'static str,
    work: impl FnOnce() -> Result<T, HevcError>,
) -> Result<T, HevcError> {
    match catch_unwind(AssertUnwindSafe(work)) {
        Ok(result) => result,
        Err(payload) => Err(HevcError::WorkerPanic {
            stage,
            message: panic_text(payload.as_ref()),
        }),
    }
}

fn panic_text(payload: &(dyn Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|text| (*text).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a payload that is not text".to_owned())
}

fn parse_stage<S: Sample>(run: &PictureRun<S>) {
    let Some((mut job, resolver)) = run.take() else {
        return;
    };
    if run.spawner.stopped() {
        return;
    }
    let site = job.finish_site;
    let parsed = guarded(PARSE_STAGE, || {
        job.parse_pending(&run.spawner);
        Ok(())
    });
    if let Err(error) = parsed {
        resolver.resolve(Err(PictureFailure::Decode { error, site }));
        return;
    }
    match job.parse_outcome() {
        Some(failure) => resolver.resolve(Err(failure)),
        None => run.put(job, resolver),
    }
}

fn reconstruct_stage<S: Sample>(run: &PictureRun<S>) {
    let Some((mut job, resolver)) = run.take() else {
        return;
    };
    if run.spawner.stopped() {
        return;
    }
    let site = job.finish_site;
    let done = guarded(RECONSTRUCT_STAGE, || {
        job.reconstruct(&run.pools, &run.spawner)
    });
    match done {
        Ok(()) => {
            let frame = job.into_frame(&run.pools);
            resolver.resolve(Ok(Arc::new(frame)));
        }
        Err(error) => resolver.resolve(Err(PictureFailure::Decode { error, site })),
    }
}

/// Starts a picture. The parse queues now. The reconstruct stage queues when the parse and
/// every picture of `references` have ended. So it holds no worker while it waits.
pub(crate) fn start_picture<S: Sample>(
    job: PictureJob<S>,
    resolver: PictureResolver<S>,
    references: &[PictureHandle<S>],
    pools: &Arc<PicturePools<S>>,
    spawner: &Spawner,
) {
    let run = Arc::new(PictureRun {
        slot: Mutex::new(Some((job, resolver))),
        pools: Arc::clone(pools),
        spawner: spawner.clone(),
    });
    let opening = {
        let (run, spawner) = (Arc::clone(&run), spawner.clone());
        move || spawner.spawn(move || reconstruct_stage(&run))
    };
    let gate = Gate::new(references.len() + 1, Box::new(opening));
    for reference in references {
        if !reference.subscribe(&gate) {
            gate.arrive();
        }
    }
    spawner.spawn(move || {
        parse_stage(&run);
        gate.arrive();
    });
}

/// Parses an abandoned picture, for the error of an earlier slice. No reconstruct follows.
pub(crate) fn start_abandoned<S: Sample>(
    mut job: PictureJob<S>,
    resolver: PictureResolver<S>,
    pools: &Arc<PicturePools<S>>,
    spawner: &Spawner,
) {
    job.abandoned = true;
    start_picture(job, resolver, &[], pools, spawner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::cpu::hevc::stage::UnitSite;
    use crate::decode::cpu::hevc::threads::completion::Completion;
    use crate::decode::cpu::hevc::threads::pool::WorkerPool;

    #[test]
    fn a_panic_is_a_named_error_with_its_message() {
        let error = guarded::<()>(PARSE_STAGE, || panic!("the text")).unwrap_err();
        assert_eq!(
            error,
            HevcError::WorkerPanic {
                stage: "parse",
                message: "the text".to_owned(),
            }
        );
    }

    #[test]
    fn a_panic_on_a_worker_reaches_the_waiter() {
        let pool = WorkerPool::start(2).unwrap();
        let (cell, resolver) = Completion::<u32>::pending(UnitSite::End);
        pool.spawner().spawn(move || {
            let result = guarded(RECONSTRUCT_STAGE, || -> Result<u32, HevcError> {
                panic!("no value")
            });
            resolver.resolve(result.map_err(|error| PictureFailure::Decode {
                error,
                site: UnitSite::End,
            }));
        });
        assert!(matches!(
            cell.wait(),
            Err(PictureFailure::Decode {
                error: HevcError::WorkerPanic {
                    stage: "reconstruct",
                    ..
                },
                ..
            })
        ));
    }
}
