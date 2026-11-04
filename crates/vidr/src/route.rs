//! How a codec backend's frames move between host and device memory.
//!
//! Every `Decoder` and `Encoder` declares its route, and has no default. The
//! pump counts a host copy from the route, so no backend copies silently.

use std::fmt;

/// How one codec backend moves each frame.
///
/// | Route | Each frame | Host copies |
/// | --- | --- | --- |
/// | `Device` | stays in device memory | 0 |
/// | `HostUpload` | goes from host memory to a device tensor | 1 |
/// | `HostDownload` | goes from a device tensor to host memory | 1 |
/// | `HostOnly` | stays in host memory, between codec buffers and CPU tensors | 1 |
///
/// A hardware codec returns `Device`. A CPU decoder on a GPU backend returns
/// `HostUpload`, and a CPU encoder on a GPU backend returns `HostDownload`. A
/// CPU codec on numr's CPU backend returns `HostOnly`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FrameRoute {
    /// Frames stay in device memory.
    Device,
    /// The backend works in host memory and uploads each frame.
    HostUpload,
    /// The backend works in host memory and downloads each frame.
    HostDownload,
    /// The backend and the frame tensors both sit in host memory.
    HostOnly,
}

impl FrameRoute {
    /// Whether each frame takes one copy through host memory.
    pub const fn copies_through_host(self) -> bool {
        !matches!(self, Self::Device)
    }

    /// The route's name in a run's output and report.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Device => "device",
            Self::HostUpload => "host upload",
            Self::HostDownload => "host download",
            Self::HostOnly => "host only",
        }
    }
}

impl fmt::Display for FrameRoute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_device_route_avoids_a_host_copy() {
        assert!(!FrameRoute::Device.copies_through_host());
        assert!(FrameRoute::HostUpload.copies_through_host());
        assert!(FrameRoute::HostDownload.copies_through_host());
        assert!(FrameRoute::HostOnly.copies_through_host());
    }

    #[test]
    fn each_route_has_its_own_name() {
        assert_eq!(FrameRoute::Device.to_string(), "device");
        assert_eq!(FrameRoute::HostUpload.to_string(), "host upload");
        assert_eq!(FrameRoute::HostDownload.to_string(), "host download");
        assert_eq!(FrameRoute::HostOnly.to_string(), "host only");
    }
}
