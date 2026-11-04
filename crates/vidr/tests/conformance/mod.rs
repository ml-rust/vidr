//! Conformance sets the repository does not hold. Each test reads a directory
//! from an environment variable, and prints a skip notice without it.

mod hevc;
#[cfg(feature = "cpu-hevc")]
mod hevc_decode;
#[cfg(any(feature = "cpu-hevc", feature = "vulkan-video"))]
mod hevc_hash;
#[cfg(feature = "cpu-hevc")]
mod hevc_robust;
#[cfg(any(
    feature = "cpu-hevc",
    feature = "vulkan-video",
    all(feature = "cpu-vp9", feature = "mkv")
))]
mod md5;
#[cfg(any(feature = "cpu-hevc", feature = "vulkan-video"))]
mod mutate;
#[cfg(any(
    feature = "cpu-hevc",
    feature = "vulkan-video",
    all(feature = "cpu-vp9", feature = "mkv")
))]
mod runner;
#[cfg(all(feature = "cpu-vp9", feature = "mkv"))]
mod vp9;
#[cfg(all(feature = "mkv", any(feature = "cpu-vp9", feature = "vulkan-video")))]
mod vp9_vectors;
#[cfg(feature = "vulkan-video")]
mod vulkan_hevc;
#[cfg(feature = "vulkan-video")]
mod vulkan_robust;
#[cfg(all(feature = "vulkan-video", feature = "mkv"))]
mod vulkan_vp9;
