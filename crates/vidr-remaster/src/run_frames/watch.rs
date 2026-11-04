//! The watcher thread that reports the frames of the encode pass.

use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use vidr::pipeline::PumpControl;

use crate::error::RemasterError;
use crate::run::Progress;

/// How often the watcher reads the pump's counters.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(250);

/// Runs `pump` on this thread while a watcher thread reports the frames it
/// encodes. A run with no watcher thread still completes, with no frame events.
pub(super) fn watch_frames<P, T, E>(
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
