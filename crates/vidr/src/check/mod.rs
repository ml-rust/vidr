//! Output checks: the output keeps the source's frame count, frame timing,
//! last-frame duration, rotation, and display aspect ratio, at the target
//! size.
//!
//! [`check_output`] runs every check and reports every mismatch, never only
//! the first. A mismatch is an error, never a warning.
//!
//! Each check is exact unless rounding the pipeline cannot avoid makes exact
//! equality wrong:
//!
//! | Check | Comparison | Reason |
//! | --- | --- | --- |
//! | Frame count | exact | The pipeline neither drops nor adds frames. |
//! | Display dimensions | exact, against the target | The scale step writes the target size. |
//! | Rotation | exact | The output copies the source rotation. |
//! | Pts order | exact | Strictly increasing pts are a stream invariant. |
//! | Pts values | exact, after a half-even rescale | Each output pts is the source pts rescaled with [`Rounding::HalfEven`](crate::time::Rounding::HalfEven), as the PoC writes `round(timestamp / clock)`. In the same time base the rescale is the identity. |
//! | Total duration, same time base | exact | The pipeline copies every tick, and writes the last-frame duration rounded half-even to a whole tick. |
//! | Total duration, different time bases | [`DURATION_TOLERANCE`] | A coarser output clock rounds the last relative pts and the last-frame duration by up to half a tick each, so the total can move by up to one output tick. |
//! | Display aspect ratio | 2 pixels on the shortest side | Pixel dimensions are integers and codecs store the sample aspect ratio in 16-bit fields, so the output cannot always hold the source ratio exactly. The PoC uses the same bound (`reuse_upscaled.py`). |

mod deviation;
mod output;
mod shape;
mod timing;

pub use deviation::{CheckError, Deviation, Side};
pub use output::{DURATION_TOLERANCE, StreamFacts, check_output};
