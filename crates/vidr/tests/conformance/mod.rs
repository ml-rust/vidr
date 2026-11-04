//! Conformance sets the repository does not hold. Each test reads a directory
//! from an environment variable, and prints a skip notice without it.

mod hevc;
#[cfg(all(feature = "cpu-vp9", feature = "mkv"))]
mod md5;
#[cfg(all(feature = "cpu-vp9", feature = "mkv"))]
mod vp9;
