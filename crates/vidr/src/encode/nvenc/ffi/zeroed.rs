//! The all-zero constructor of the FFI mirrors.

/// Gives each listed type a `zeroed` constructor.
macro_rules! impl_zeroed {
    ($($name:ident),+ $(,)?) => {$(
        impl $name {
            /// Every field zero and every pointer null, as the header requires
            /// of reserved fields.
            pub(crate) fn zeroed() -> Self {
                // SAFETY: the type holds only integers, raw pointers, `Option`s
                // of function pointers, and arrays and unions of those. All
                // zero bytes are a valid value of each.
                unsafe { std::mem::zeroed() }
            }
        }
    )+};
}
