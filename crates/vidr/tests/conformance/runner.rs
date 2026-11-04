//! The worker pool of the conformance tests.

use std::any::Any;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

/// `VIDR_CONFORMANCE_JOBS`, or half the logical CPUs.
fn job_count() -> usize {
    let half = thread::available_parallelism().map_or(1, |count| count.get() / 2);
    std::env::var("VIDR_CONFORMANCE_JOBS")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(half)
        .max(1)
}

/// Runs `run` on every item, each worker taking the next one, and keeps the input
/// order in the results.
pub(super) fn run_each<T, O>(items: &[T], run: impl Fn(&T) -> O + Sync) -> Vec<O>
where
    T: Sync,
    O: Send,
{
    let next = AtomicUsize::new(0);
    let results = Mutex::new(Vec::with_capacity(items.len()));
    thread::scope(|scope| {
        for _ in 0..job_count() {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(item) = items.get(index) else {
                        break;
                    };
                    let outcome = run(item);
                    if let Ok(mut results) = results.lock() {
                        results.push((index, outcome));
                    }
                }
            });
        }
    });
    let mut results = results
        .into_inner()
        .unwrap_or_else(|error| error.into_inner());
    results.sort_by_key(|(index, _)| *index);
    results.into_iter().map(|(_, outcome)| outcome).collect()
}

/// The message of a caught panic.
pub(super) fn panic_text(panic: &(dyn Any + Send)) -> &str {
    panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .unwrap_or("no message")
}
