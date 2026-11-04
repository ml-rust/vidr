//! All-zero values of plain C structures.

/// A plain C structure for which all-zero bytes are a valid value.
///
/// # Safety
///
/// Every field of the type accepts all-zero bytes: integers, bitfields, and raw pointers.
pub(super) unsafe trait AllZero: Sized {}

/// The all-zero value of `T`.
pub(super) fn zeroed<T: AllZero>() -> T {
    // SAFETY: `AllZero` states that all-zero bytes are a valid `T`.
    unsafe { std::mem::zeroed() }
}
