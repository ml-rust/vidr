//! The stages of a picture on a fixed worker pool, with the output of the calling-thread
//! path.
//!
//! Each picture is one job that owns its buffers, and nothing shares a picture while a
//! worker writes it.
//!
//! | Module | Holds |
//! | --- | --- |
//! | `completion` | The cell a later picture reads |
//! | `handle` | The form the decoded picture buffer keeps |
//! | `job`, `run` | The worker stages |
//! | `pool` | The fixed threads |
//! | `stage` | The part on the calling thread |

pub(crate) mod completion;
pub(crate) mod handle;
pub(crate) mod job;
pub(crate) mod pool;
pub(crate) mod run;
pub(crate) mod stage;
