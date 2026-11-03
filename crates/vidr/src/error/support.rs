//! Display helpers shared by the error types.

use std::fmt;

/// Displays an optional value, or `missing` for `None`.
pub(super) struct Shown<T>(pub(super) Option<T>);

impl<T: fmt::Display> fmt::Display for Shown<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Some(value) => fmt::Display::fmt(value, f),
            None => f.write_str("missing"),
        }
    }
}
