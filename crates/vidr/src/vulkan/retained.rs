//! Stable heap storage for what a session parameters object points to.
//!
//! NVIDIA's driver reads through the pointers of a Std structure after
//! `vkCreateVideoSessionParametersKHR` returns. So each pointee keeps one address for the life
//! of the parameters. `Arc::as_ptr` gives that address, and a move of the owner leaves it alone.
//! The values never change after construction, so a list never reallocates. The encoder's profile
//! and rate control chains use the same storage, so a move of their owner leaves each chain valid.

use std::sync::Arc;

/// One value at a stable heap address.
pub(super) struct Retained<T>(Arc<T>);

impl<T: Copy> Retained<T> {
    pub(super) fn new(value: T) -> Self {
        Self(Arc::new(value))
    }

    /// The value's address, valid while any owner of it lives.
    pub(super) fn as_ptr(&self) -> *const T {
        Arc::as_ptr(&self.0)
    }

    /// A copy of the value. A copy of a Std structure holds the same pointers.
    pub(super) fn get(&self) -> T {
        *self.0
    }
}

/// A fixed list at a stable heap address.
pub(super) struct RetainedList<T>(Arc<[T]>);

impl<T: Copy> RetainedList<T> {
    pub(super) fn new(items: Vec<T>) -> Self {
        Self(Arc::from(items))
    }

    /// The first item's address, valid while any owner of the list lives. An empty list gives a
    /// dangling address, so a Std structure states null for it.
    pub(super) fn as_ptr(&self) -> *const T {
        Arc::as_ptr(&self.0).cast()
    }

    pub(super) fn as_slice(&self) -> &[T] {
        &self.0
    }

    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

// SAFETY: `T` is `Copy`, so it holds no owned resource. A retained value never changes after
// construction, and Rust code here never reads through a pointer inside it. The driver reads
// through those pointers only inside calls its owner makes, and an owner runs on one thread at a
// time.
unsafe impl<T: Copy> Send for Retained<T> {}
// SAFETY: as above, a shared reference gives no way to change the value.
unsafe impl<T: Copy> Sync for Retained<T> {}
// SAFETY: as for `Retained`.
unsafe impl<T: Copy> Send for RetainedList<T> {}
// SAFETY: as for `Retained`.
unsafe impl<T: Copy> Sync for RetainedList<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    const fn assert_send_sync<T: Send + Sync>() {}
    /// A raw pointer is neither `Send` nor `Sync`, and a Std structure holds some.
    const _: () = assert_send_sync::<Retained<*const u8>>();
    const _: () = assert_send_sync::<RetainedList<*const u8>>();

    #[test]
    fn a_move_keeps_the_value_at_its_address() {
        let value = Retained::new(7_u32);
        let before = value.as_ptr();
        let moved = [value];
        assert_eq!(moved[0].as_ptr(), before);
        assert_eq!(moved[0].get(), 7);
        // SAFETY: `moved` owns the value, and it still lives.
        assert_eq!(unsafe { *before }, 7);
    }

    #[test]
    fn a_move_keeps_the_list_at_its_address() {
        let list = RetainedList::new(vec![1_i32, 2, 3]);
        let before = list.as_ptr();
        let moved = Some(list);
        let moved = moved.as_ref().expect("the list");
        assert_eq!(moved.as_ptr(), before);
        assert_eq!(moved.as_slice(), [1, 2, 3]);
        assert!(!moved.is_empty());
        assert!(RetainedList::<i32>::new(Vec::new()).is_empty());
    }
}
