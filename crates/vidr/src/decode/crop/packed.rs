//! Cutting a packed word plane to a byte window on the device.

use numr::dtype::DataType as _;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::window::{Window, copy_error};
use crate::error::DecodeError;

/// A packed plane cut to `window`, whose columns count bytes. The result holds
/// `ceil(columns / 4)` words per row, from the start of each row's first word.
/// The bytes past `columns` in a row's last word come from the source row or
/// the next one, and no reader looks at them.
///
/// A window that starts on a word takes a narrowed view and one copy. Any
/// other start takes numr's strided copy byte by byte, which runs on the
/// device on every backend.
pub(super) fn word_window<R: Runtime>(
    plane: &Tensor<R>,
    window: Window,
    name: &'static str,
) -> Result<Tensor<R>, DecodeError> {
    let failed = |reason: String| copy_error(name, plane, window, reason);
    if plane.dtype().size_in_bytes() != 4 {
        return Err(failed("a packed plane holds 4-byte words".to_owned()));
    }
    let words = window.columns.div_ceil(4);
    if window.column.is_multiple_of(4) {
        return plane
            .narrow(0, window.row, window.rows)
            .and_then(|view| view.narrow(1, window.column / 4, words))
            .and_then(|view| view.compact())
            .map_err(|error| failed(error.to_string()));
    }
    let span = byte_span(plane, window).map_err(failed)?;
    let device = plane.device();
    let out = Tensor::<R>::empty(&[window.rows, words], plane.dtype(), device)
        .map_err(|error| failed(error.to_string()))?;
    let wide = words * 4;
    let rows = window.rows;
    // Each row reads `wide` bytes from its start. The last row's read can
    // pass the end of the storage, so that case first packs the rows tight
    // into a buffer with room for the overrun.
    if span.wide_end(wide) <= plane.storage().size_in_bytes() {
        let source = Bytes::at(plane, span.start, span.pitch);
        byte_copy(source, &out, [rows, wide]).map_err(failed)?;
    } else {
        let packed = window.rows * window.columns;
        let staging = Tensor::<R>::empty(&[(packed + 3).div_ceil(4)], plane.dtype(), device)
            .map_err(|error| failed(error.to_string()))?;
        let source = Bytes::at(plane, span.start, span.pitch);
        byte_copy(source, &staging, [rows, window.columns]).map_err(failed)?;
        let tight = Bytes::at(&staging, 0, window.columns);
        byte_copy(tight, &out, [rows, wide]).map_err(failed)?;
    }
    Ok(out)
}

/// Rows of bytes in a tensor's storage: the first byte and the row pitch.
struct Bytes<'a, R: Runtime> {
    tensor: &'a Tensor<R>,
    start: usize,
    pitch: usize,
}

impl<'a, R: Runtime> Bytes<'a, R> {
    const fn at(tensor: &'a Tensor<R>, start: usize, pitch: usize) -> Self {
        Self {
            tensor,
            start,
            pitch,
        }
    }
}

/// Copies `shape` (`[rows, bytes per row]`) from `source` into the dense
/// storage of `target`, on the source's device. The caller checks that each
/// read lies inside the source storage and that `target` holds every byte.
fn byte_copy<R: Runtime>(
    source: Bytes<'_, R>,
    target: &Tensor<R>,
    shape: [usize; 2],
) -> Result<(), String> {
    let pitch = isize::try_from(source.pitch)
        .map_err(|_| format!("a row pitch of {} bytes", source.pitch))?;
    R::copy_strided(
        source.tensor.storage().ptr(),
        source.start,
        target.storage().ptr(),
        &shape,
        &[pitch, 1],
        1,
        source.tensor.device(),
    )
    .map_err(|error| error.to_string())
}

/// Where a byte window of a packed plane sits in its storage.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct ByteSpan {
    /// The window's first byte.
    start: usize,
    /// Bytes from one row's start to the next.
    pitch: usize,
    /// Rows in the window.
    rows: usize,
}

impl ByteSpan {
    /// The byte after a read of `wide` bytes from the start of each row.
    pub(super) const fn wide_end(&self, wide: usize) -> usize {
        self.start + (self.rows - 1) * self.pitch + wide
    }
}

/// The span of `window` in the storage of `plane`, checked to lie inside the
/// plane and its storage.
pub(super) fn byte_span<R: Runtime>(plane: &Tensor<R>, window: Window) -> Result<ByteSpan, String> {
    let (&[rows, columns], &[row_stride, 1]) = (plane.shape(), plane.strides()) else {
        return Err("a packed plane is a 2-D view with column stride 1".to_owned());
    };
    let row_stride = usize::try_from(row_stride)
        .ok()
        .filter(|stride| *stride >= columns)
        .ok_or_else(|| format!("a packed plane's row stride covers its row, not {row_stride}"))?;
    let inside = window.rows > 0
        && window.row + window.rows <= rows
        && window.column + window.columns <= columns * 4;
    if !inside {
        return Err("the window lies outside the plane".to_owned());
    }
    match span_of(plane.offset(), row_stride, window) {
        Some((span, end)) if end <= plane.storage().size_in_bytes() => Ok(span),
        _ => Err("the window lies outside the plane's storage".to_owned()),
    }
}

/// The span of `window` in a plane that starts `offset` words into its
/// storage, and the byte after the window's last byte. `None` on overflow.
fn span_of(offset: usize, row_stride: usize, window: Window) -> Option<(ByteSpan, usize)> {
    let pitch = row_stride.checked_mul(4)?;
    let first_row = offset.checked_add(window.row.checked_mul(row_stride)?)?;
    let start = first_row.checked_mul(4)?.checked_add(window.column)?;
    let end = start
        .checked_add(window.rows.checked_sub(1)?.checked_mul(pitch)?)?
        .checked_add(window.columns)?;
    let span = ByteSpan {
        start,
        pitch,
        rows: window.rows,
    };
    Some((span, end))
}
