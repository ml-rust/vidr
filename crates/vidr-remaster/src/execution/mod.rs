//! Which runtime, decoder, and encoder a run uses. Everything here is pure, so a
//! test reaches every request on any machine.
//!
//! | Module | Holds |
//! | --- | --- |
//! | `choice` | The `--runtime`, `--decoder`, and `--encoder` values, and what they resolve to |
//! | `availability` | What this build holds and what this machine answers |
//! | `error` | The refusals |
//! | `select` | [`execution_for`] and the fall-through of `auto` |

mod availability;
mod choice;
mod error;
mod select;

pub use availability::{Compiled, Machine, Probe};
pub use choice::{
    DecoderBackend, DecoderChoice, EncoderBackend, EncoderChoice, Openh264Build, ParseChoiceError,
    RuntimeChoice, RuntimeKind,
};
pub use error::ExecutionError;
pub use select::{
    Execution, ExecutionRequest, Reasons, Selector, SourceCodec, SourceFacts, execution_for,
};
