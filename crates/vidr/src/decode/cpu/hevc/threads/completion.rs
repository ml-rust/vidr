//! The result of a picture that finishes later, and the gate that starts work when its
//! inputs are ready. A waiter holds no worker: a gate keeps its work out of the pool until
//! every input arrives.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};

use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::stage::UnitSite;

/// Why a picture gives no frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PictureFailure {
    /// An error, and the NAL unit it belongs to.
    Decode {
        /// The error.
        error: HevcError,
        /// The NAL unit the calling-thread path reports it at.
        site: UnitSite,
    },
    /// An error on the calling thread ended the picture, and its slices parse without one.
    Abandoned,
}

impl PictureFailure {
    fn poisoned() -> Self {
        Self::Decode {
            error: HevcError::LockPoisoned {
                what: "a picture's completion cell",
            },
            site: UnitSite::End,
        }
    }
}

/// A value that arrives once, or its failure.
enum CellState<T> {
    Pending(Vec<Arc<Gate>>),
    Done(Result<T, PictureFailure>),
}

/// The cell of one picture's result. Locks stay per picture.
pub(crate) struct Completion<T> {
    state: Mutex<CellState<T>>,
    changed: Condvar,
}

impl<T> fmt::Debug for Completion<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Completion").finish_non_exhaustive()
    }
}

impl<T> Completion<T> {
    /// A cell with no result, and the only way to set one.
    pub(crate) fn pending(site: UnitSite) -> (Arc<Self>, Resolver<T>) {
        let cell = Arc::new(Self {
            state: Mutex::new(CellState::Pending(Vec::new())),
            changed: Condvar::new(),
        });
        let resolver = Resolver {
            cell: Arc::clone(&cell),
            site,
            sent: false,
        };
        (cell, resolver)
    }

    /// A cell that holds `value` now.
    pub(crate) fn ready(value: T) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(CellState::Done(Ok(value))),
            changed: Condvar::new(),
        })
    }

    /// Registers `gate`, which takes one arrival when the result arrives. Returns false
    /// when the result is here, and the caller counts the arrival itself.
    pub(crate) fn subscribe(&self, gate: &Arc<Gate>) -> bool {
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        match &mut *state {
            CellState::Pending(waiters) => {
                waiters.push(Arc::clone(gate));
                true
            }
            CellState::Done(_) => false,
        }
    }

    /// Stores the result, wakes waiters, and opens gates. A second result does nothing.
    fn resolve(&self, result: Result<T, PictureFailure>) {
        let waiters = {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            if matches!(&*state, CellState::Done(_)) {
                return;
            }
            match std::mem::replace(&mut *state, CellState::Done(result)) {
                CellState::Pending(waiters) => waiters,
                CellState::Done(_) => Vec::new(),
            }
        };
        self.changed.notify_all();
        for gate in waiters {
            gate.arrive();
        }
    }

    /// The result when it is here. A poisoned lock is a named failure.
    pub(crate) fn poll(&self) -> Option<Result<T, PictureFailure>>
    where
        T: Clone,
    {
        let Ok(state) = self.state.lock() else {
            return Some(Err(PictureFailure::poisoned()));
        };
        match &*state {
            CellState::Pending(_) => None,
            CellState::Done(result) => Some(result.clone()),
        }
    }

    /// Blocks until the result is here. A poisoned lock is a named failure. A dropped
    /// [`Resolver`] gives a failure, so a result always comes.
    pub(crate) fn wait(&self) -> Result<T, PictureFailure>
    where
        T: Clone,
    {
        let Ok(mut state) = self.state.lock() else {
            return Err(PictureFailure::poisoned());
        };
        loop {
            if let CellState::Done(result) = &*state {
                return result.clone();
            }
            state = match self.changed.wait(state) {
                Ok(state) => state,
                Err(_) => return Err(PictureFailure::poisoned()),
            };
        }
    }

    /// The result, for the cell's last holder.
    pub(crate) fn into_result(self) -> Option<Result<T, PictureFailure>> {
        match self.state.into_inner() {
            Ok(CellState::Done(result)) => Some(result),
            _ => None,
        }
    }
}

/// Sets a cell's result. Dropped with no result, it stores [`HevcError::WorkerStopped`],
/// so a waiter never hangs on a task the pool drops.
pub(crate) struct Resolver<T> {
    cell: Arc<Completion<T>>,
    site: UnitSite,
    sent: bool,
}

impl<T> Resolver<T> {
    /// Stores the result.
    pub(crate) fn resolve(mut self, result: Result<T, PictureFailure>) {
        self.sent = true;
        self.cell.resolve(result);
    }
}

impl<T> Drop for Resolver<T> {
    fn drop(&mut self) {
        if !self.sent {
            self.cell.resolve(Err(PictureFailure::Decode {
                error: HevcError::WorkerStopped,
                site: self.site,
            }));
        }
    }
}

/// Runs its work once when a count of arrivals is reached. Dropped first, it drops the
/// work, and the work's [`Resolver`] names the failure.
pub(crate) struct Gate {
    remaining: AtomicUsize,
    open: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

impl Gate {
    /// A gate that runs `open` at the arrival number `count`. A `count` of 0 never opens.
    pub(crate) fn new(count: usize, open: Box<dyn FnOnce() + Send>) -> Arc<Self> {
        Arc::new(Self {
            remaining: AtomicUsize::new(count),
            open: Mutex::new(Some(open)),
        })
    }

    /// Counts one arrival.
    pub(crate) fn arrive(&self) {
        if self.remaining.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        let open = self
            .open
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(open) = open {
            open();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;
    use std::thread;

    use super::*;

    const SITE: UnitSite = UnitSite::Nal { bytes: 3, pts: 4 };

    #[test]
    fn a_result_wakes_a_blocked_waiter() {
        let (cell, resolver) = Completion::<u32>::pending(SITE);
        let waiter = {
            let cell = Arc::clone(&cell);
            thread::spawn(move || cell.wait())
        };
        assert_eq!(cell.poll(), None);
        resolver.resolve(Ok(7));
        assert_eq!(waiter.join().unwrap(), Ok(7));
        assert_eq!(cell.poll(), Some(Ok(7)));
    }

    /// A task the pool drops cannot leave a waiter blocked.
    #[test]
    fn a_dropped_resolver_names_the_failure() {
        let (cell, resolver) = Completion::<u32>::pending(SITE);
        let waiter = {
            let cell = Arc::clone(&cell);
            thread::spawn(move || cell.wait())
        };
        drop(resolver);
        assert_eq!(
            waiter.join().unwrap(),
            Err(PictureFailure::Decode {
                error: HevcError::WorkerStopped,
                site: SITE,
            })
        );
    }

    #[test]
    fn a_failure_reaches_every_reader() {
        let (cell, resolver) = Completion::<u32>::pending(SITE);
        let failure = PictureFailure::Decode {
            error: HevcError::MissingReference { poc: 2 },
            site: UnitSite::End,
        };
        resolver.resolve(Err(failure.clone()));
        assert_eq!(cell.wait(), Err(failure.clone()));
        assert_eq!(cell.poll(), Some(Err(failure)));
    }

    #[test]
    fn a_ready_cell_gives_its_value_to_its_last_holder() {
        let cell = Completion::ready(5u32);
        assert_eq!(cell.poll(), Some(Ok(5)));
        let only = Arc::try_unwrap(cell).unwrap();
        assert_eq!(only.into_result(), Some(Ok(5)));
    }

    /// A lock poisoned by a panic is a named failure, never a hang.
    #[test]
    fn a_poisoned_cell_is_a_named_failure() {
        let (cell, resolver) = Completion::<u32>::pending(SITE);
        let poison = Arc::clone(&cell);
        let _ = thread::spawn(move || {
            let _guard = poison.state.lock().unwrap();
            panic!("poison");
        })
        .join();
        assert_eq!(cell.wait(), Err(PictureFailure::poisoned()));
        assert_eq!(cell.poll(), Some(Err(PictureFailure::poisoned())));
        resolver.resolve(Ok(1));
    }

    #[test]
    fn a_gate_opens_once_at_its_last_arrival() {
        let opened = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&opened);
        let gate = Gate::new(
            2,
            Box::new(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            }),
        );
        gate.arrive();
        assert_eq!(opened.load(Ordering::SeqCst), 0);
        gate.arrive();
        assert_eq!(opened.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_result_counts_one_arrival_for_each_gate() {
        let opened = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&opened);
        let gate = Gate::new(1, Box::new(move || flag.store(true, Ordering::SeqCst)));
        let (cell, resolver) = Completion::<u32>::pending(SITE);
        assert!(cell.subscribe(&gate));
        assert!(!opened.load(Ordering::SeqCst));
        resolver.resolve(Ok(1));
        assert!(opened.load(Ordering::SeqCst));
        let late = Gate::new(1, Box::new(|| {}));
        assert!(!cell.subscribe(&late));
    }

    /// A gate that never opens drops its work, and the work's resolver names the failure.
    #[test]
    fn a_dropped_gate_drops_its_work() {
        let (cell, resolver) = Completion::<u32>::pending(SITE);
        let gate = Gate::new(1, Box::new(move || resolver.resolve(Ok(1))));
        drop(gate);
        assert!(matches!(
            cell.wait(),
            Err(PictureFailure::Decode {
                error: HevcError::WorkerStopped,
                ..
            })
        ));
    }
}
