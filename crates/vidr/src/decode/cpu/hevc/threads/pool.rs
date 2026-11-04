//! The fixed worker threads of one decoder: a rayon pool on threads this type joins.
//!
//! rayon is the safe way to lend borrowed picture rows to a fixed pool. Its `ThreadPool`
//! ends its threads with no join on drop, so a spawn handler keeps their handles here.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::thread::{self, JoinHandle};

use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::decode::cpu::hevc::error::HevcError;

/// Stack of one worker. A worker that waits for rows of a filter pass can run another
/// picture's stage on the same stack. The cap on pictures in flight bounds that depth.
const WORKER_STACK_BYTES: usize = 8 << 20;

/// Starts tasks on the pool, from any thread. It keeps no thread alive after shutdown:
/// a task that comes later drops, and its `Resolver` names the failure.
#[derive(Clone, Debug)]
pub(crate) struct Spawner {
    pool: Weak<ThreadPool>,
    stop: Arc<AtomicBool>,
}

impl Spawner {
    /// Queues `task` behind the tasks already queued. A task never waits for a picture:
    /// a gate holds it back until its inputs are ready.
    pub(crate) fn spawn(&self, task: impl FnOnce() + Send + 'static) {
        if let Some(pool) = self.pool.upgrade() {
            pool.spawn_fifo(task);
        }
    }

    /// True after a shutdown request. A task that sees it ends at once.
    pub(crate) fn stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }
}

/// A fixed set of threads, started at open and joined on drop.
#[derive(Debug)]
pub(crate) struct WorkerPool {
    pool: Option<Arc<ThreadPool>>,
    threads: Vec<JoinHandle<()>>,
    stop: Arc<AtomicBool>,
}

impl WorkerPool {
    /// Starts `threads` workers.
    ///
    /// # Errors
    ///
    /// [`HevcError::WorkerStart`] when a thread does not start. The threads that did start
    /// end before this returns.
    pub(crate) fn start(threads: usize) -> Result<Self, HevcError> {
        let mut handles = Vec::new();
        let built = ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("hevc-worker-{index}"))
            .stack_size(WORKER_STACK_BYTES)
            // Each stage catches its own panics. This handler stops rayon's abort for one
            // that escapes a stage.
            .panic_handler(|_| {})
            .spawn_handler(|thread| {
                let mut builder = thread::Builder::new();
                if let Some(name) = thread.name() {
                    builder = builder.name(name.to_owned());
                }
                if let Some(size) = thread.stack_size() {
                    builder = builder.stack_size(size);
                }
                handles.push(builder.spawn(move || thread.run())?);
                Ok(())
            })
            .build();
        match built {
            Ok(pool) => Ok(Self {
                pool: Some(Arc::new(pool)),
                threads: handles,
                stop: Arc::new(AtomicBool::new(false)),
            }),
            Err(error) => {
                join_all(handles);
                Err(HevcError::WorkerStart {
                    threads,
                    reason: error.to_string(),
                })
            }
        }
    }

    /// A handle for starting tasks.
    pub(crate) fn spawner(&self) -> Spawner {
        Spawner {
            pool: self.pool.as_ref().map_or_else(Weak::new, Arc::downgrade),
            stop: Arc::clone(&self.stop),
        }
    }

    /// Asks each task to end, and joins every thread. rayon runs each queued task before a
    /// thread exits. No task waits for a picture, so the join ends. A later call does
    /// nothing.
    pub(crate) fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.pool = None;
        join_all(std::mem::take(&mut self.threads));
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn join_all(threads: Vec<JoinHandle<()>>) {
    for thread in threads {
        // rayon's worker loop holds no panic of a task, so this result names nothing new.
        let _ = thread.join();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::mpsc;

    use super::*;
    use crate::decode::cpu::hevc::stage::UnitSite;
    use crate::decode::cpu::hevc::threads::completion::{Completion, Gate, PictureFailure};

    #[test]
    fn tasks_run_on_every_pool_size() {
        for threads in [1, 2, 8] {
            let pool = WorkerPool::start(threads).unwrap();
            let spawner = pool.spawner();
            let (sender, receiver) = mpsc::channel();
            for value in 0..32 {
                let sender = sender.clone();
                spawner.spawn(move || sender.send(value).unwrap());
            }
            let mut seen: Vec<i32> = (0..32).map(|_| receiver.recv().unwrap()).collect();
            seen.sort_unstable();
            assert_eq!(seen, (0..32).collect::<Vec<_>>());
        }
    }

    #[test]
    fn drop_before_any_task_joins_every_thread() {
        drop(WorkerPool::start(4).unwrap());
    }

    /// Every queued task runs or sees the stop flag, before the drop returns.
    #[test]
    fn drop_with_queued_tasks_returns_after_they_end() {
        let ran = Arc::new(AtomicUsize::new(0));
        let pool = WorkerPool::start(2).unwrap();
        let spawner = pool.spawner();
        for _ in 0..64 {
            let (ran, inner) = (Arc::clone(&ran), spawner.clone());
            spawner.spawn(move || {
                if !inner.stopped() {
                    ran.fetch_add(1, Ordering::SeqCst);
                }
            });
        }
        drop(pool);
        let left = ran.load(Ordering::SeqCst);
        assert!(left <= 64);
        assert_eq!(Arc::strong_count(&ran), 1, "a task outlives the pool");
    }

    /// A task with a reference that never comes holds no thread, so the drop returns. Its
    /// work then drops with the input, and its result names the failure.
    #[test]
    fn a_task_blocked_on_a_reference_that_never_comes_does_not_stop_the_drop() {
        let site = UnitSite::End;
        let (reference, resolver) = Completion::<u32>::pending(site);
        let (picture, picture_resolver) = Completion::<u32>::pending(site);
        let pool = WorkerPool::start(2).unwrap();
        let spawner = pool.spawner();
        let gate = Gate::new(
            1,
            Box::new(move || spawner.spawn(move || picture_resolver.resolve(Ok(1)))),
        );
        assert!(reference.subscribe(&gate));
        drop(gate);
        drop(pool);
        assert_eq!(picture.poll(), None);
        drop(resolver);
        assert!(matches!(
            picture.wait(),
            Err(PictureFailure::Decode {
                error: HevcError::WorkerStopped,
                ..
            })
        ));
    }

    /// A chain of pictures on one worker runs in dependency order, whatever the queue order:
    /// a later picture waits in its gate and holds no thread.
    #[test]
    fn one_worker_runs_a_chain_queued_backwards() {
        let pool = WorkerPool::start(1).unwrap();
        let spawner = pool.spawner();
        let site = UnitSite::End;
        let (first, first_resolver) = Completion::<u32>::pending(site);
        let (second, second_resolver) = Completion::<u32>::pending(site);
        let (input, sink) = (Arc::clone(&first), spawner.clone());
        let gate = Gate::new(
            1,
            Box::new(move || {
                sink.spawn(move || {
                    let value = input.wait().map(|value| value + 1);
                    second_resolver.resolve(value);
                });
            }),
        );
        assert!(first.subscribe(&gate));
        spawner.spawn(move || first_resolver.resolve(Ok(1)));
        assert_eq!(second.wait(), Ok(2));
    }

    #[test]
    fn a_task_after_shutdown_drops() {
        let mut pool = WorkerPool::start(2).unwrap();
        let spawner = pool.spawner();
        pool.shutdown();
        pool.shutdown();
        assert!(spawner.stopped());
        let (cell, resolver) = Completion::<u32>::pending(UnitSite::End);
        spawner.spawn(move || resolver.resolve(Ok(1)));
        assert!(matches!(cell.wait(), Err(PictureFailure::Decode { .. })));
    }
}
