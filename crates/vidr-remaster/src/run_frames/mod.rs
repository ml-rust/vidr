//! The frame path of a run, in the order of its steps, on any numr runtime.
//!
//! Every decision lives in plain functions in `execution`, `scan`, `plan`, `publish`,
//! and `report`, so tests reach them with no device. This directory holds the calls
//! that open a device or a codec.
//!
//! | Module | Holds |
//! | --- | --- |
//! | `dispatch` | The steps before a runtime exists, and the choice of runtime |
//! | `pass` | The steps on one runtime, generic over it |
//! | `runtime` | `RunRuntime`, one implementation per numr backend |
//! | `backends` | The CPU codecs, on any runtime |
//! | `scanning` | A pre-scan on the chosen decoder, and the reuse probe |
//! | `watch` | The progress watcher of the encode pass |

mod backends;
mod dispatch;
mod pass;
mod runtime;
mod scanning;
mod watch;

pub(crate) use dispatch::execute;
