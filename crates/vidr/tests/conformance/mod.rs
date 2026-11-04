//! Conformance sets the repository does not hold. Each test reads a directory
//! from an environment variable, and prints a skip notice without it.

mod hevc;
#[cfg(feature = "cpu-hevc")]
mod hevc_decode;
#[cfg(feature = "cpu-hevc")]
mod hevc_robust;
#[cfg(any(feature = "cpu-hevc", all(feature = "cpu-vp9", feature = "mkv")))]
mod md5;
#[cfg(any(feature = "cpu-hevc", all(feature = "cpu-vp9", feature = "mkv")))]
mod runner;
#[cfg(all(feature = "cpu-vp9", feature = "mkv"))]
mod vp9;
