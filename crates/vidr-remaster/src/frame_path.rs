//! How a run's frames move: the `frame_path` section of the report, and the one line
//! a run with a host copy prints.
//!
//! The report's counts come from `PumpReport` for the encode pass. A scan decodes on
//! its own decoder, so its copies count apart.

use serde_json::{Value, json};
use vidr::FrameRoute;

use crate::execution::{DecoderBackend, EncoderBackend, Execution, RuntimeKind};
use crate::notices::backend_notices;

/// Frame copies through host memory, and their bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostCopies {
    /// Frame copies.
    pub frames: u64,
    /// Bytes of those copies.
    pub bytes: u64,
}

/// What the report states about a run's frames.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FramePath {
    execution: Execution,
    decoder_route: FrameRoute,
    encoder_route: FrameRoute,
    encode_pass: HostCopies,
    scans: HostCopies,
}

impl FramePath {
    /// The path of `execution`, with each route its backend declares on the runtime.
    pub fn new(execution: &Execution) -> Self {
        Self {
            execution: execution.clone(),
            decoder_route: execution.decoder.route_on(execution.runtime),
            encoder_route: execution.encoder.route_on(execution.runtime),
            encode_pass: HostCopies::default(),
            scans: HostCopies::default(),
        }
    }

    /// Takes a later choice: a reused video's decoder, or a fall-through of `auto`.
    pub fn choose(&mut self, execution: &Execution) {
        let (encode_pass, scans) = (self.encode_pass, self.scans);
        *self = Self::new(execution);
        self.encode_pass = encode_pass;
        self.scans = scans;
    }

    /// Takes the routes the opened decoder and encoder declare.
    pub const fn set_routes(&mut self, decoder: FrameRoute, encoder: FrameRoute) {
        self.decoder_route = decoder;
        self.encoder_route = encoder;
    }

    /// Takes `PumpReport`'s host copies.
    pub const fn set_encode_pass(&mut self, copies: HostCopies) {
        self.encode_pass = copies;
    }

    /// Takes the host copies of every scan so far.
    pub const fn set_scans(&mut self, copies: HostCopies) {
        self.scans = copies;
    }

    /// The runtime.
    pub const fn runtime(&self) -> RuntimeKind {
        self.execution.runtime
    }

    /// The decoder of the encode pass.
    pub const fn decoder(&self) -> DecoderBackend {
        self.execution.decoder
    }

    /// The encoder.
    pub const fn encoder(&self) -> EncoderBackend {
        self.execution.encoder
    }

    /// Host copies per frame of the encode pass: one for each host route.
    pub const fn copies_per_frame(&self) -> u32 {
        self.decoder_route.copies_through_host() as u32
            + self.encoder_route.copies_through_host() as u32
    }

    /// The line a run prints before its encode pass, or `None` for a run with no
    /// host copy.
    pub fn line(&self) -> Option<String> {
        let copies = self.copies_per_frame();
        (copies > 0).then(|| {
            format!(
                "Frame path: {} runtime, {} decode ({}), {} encode ({}), {copies} host {} per \
                 frame",
                self.execution.runtime.label(),
                self.execution.decoder,
                self.decoder_route,
                self.execution.encoder,
                self.encoder_route,
                if copies == 1 { "copy" } else { "copies" },
            )
        })
    }

    /// The report's `frame_path` section.
    pub fn section(&self) -> Value {
        let reasons = &self.execution.reasons;
        let mut notices = backend_notices(self.execution.decoder.name());
        notices.extend(backend_notices(self.execution.encoder.name()));
        notices.dedup();
        json!({
            "runtime": self.execution.runtime.name(),
            "decoder": {
                "name": self.execution.decoder.name(),
                "route": self.decoder_route.name(),
            },
            "encoder": {
                "name": self.execution.encoder.name(),
                "route": self.encoder_route.name(),
            },
            "output_check_decoder": self.execution.check_decoder.name(),
            "host_copied_frames": self.encode_pass.frames,
            "host_copied_bytes": self.encode_pass.bytes,
            "scan_host_copied_frames": self.scans.frames,
            "scan_host_copied_bytes": self.scans.bytes,
            "why": {
                "runtime": reasons.runtime,
                "decoder": reasons.decoder,
                "encoder": reasons.encoder,
                "output_check_decoder": reasons.check_decoder,
            },
            "notices": notices,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{DecoderBackend, EncoderBackend, Reasons};

    fn execution(
        runtime: RuntimeKind,
        decoder: DecoderBackend,
        encoder: EncoderBackend,
    ) -> Execution {
        Execution {
            runtime,
            decoder,
            encoder,
            check_decoder: decoder,
            reasons: Reasons {
                runtime: "runtime reason".to_owned(),
                decoder: "decoder reason".to_owned(),
                encoder: "encoder reason".to_owned(),
                check_decoder: "check reason".to_owned(),
            },
        }
    }

    #[test]
    fn a_hardware_run_prints_no_line_and_counts_nothing() {
        let path = FramePath::new(&execution(
            RuntimeKind::Cuda,
            DecoderBackend::Nvdec,
            EncoderBackend::Nvenc,
        ));
        assert_eq!(path.copies_per_frame(), 0);
        assert_eq!(path.line(), None);
        let section = path.section();
        assert_eq!(section["runtime"], "cuda");
        assert_eq!(section["decoder"]["route"], "device");
        assert_eq!(section["encoder"]["route"], "device");
        assert_eq!(section["host_copied_frames"], 0);
        assert_eq!(section["host_copied_bytes"], 0);
        assert_eq!(section["notices"], json!([]));
    }

    #[test]
    fn a_cpu_run_has_the_report_shape_and_one_line() {
        let mut path = FramePath::new(&execution(
            RuntimeKind::Cpu,
            DecoderBackend::Dav1d,
            EncoderBackend::Rav1r,
        ));
        path.set_encode_pass(HostCopies {
            frames: 12,
            bytes: 864,
        });
        path.set_scans(HostCopies {
            frames: 6,
            bytes: 432,
        });
        assert_eq!(
            path.line().as_deref(),
            Some(
                "Frame path: CPU runtime, dav1d decode (host only), rav1r encode (host only), 2 \
                 host copies per frame"
            )
        );
        let section = path.section();
        assert_eq!(section["runtime"], "cpu");
        assert_eq!(section["decoder"]["name"], "dav1d");
        assert_eq!(section["decoder"]["route"], "host only");
        assert_eq!(section["encoder"]["name"], "rav1r");
        assert_eq!(section["encoder"]["route"], "host only");
        assert_eq!(section["output_check_decoder"], "dav1d");
        assert_eq!(section["host_copied_frames"], 12);
        assert_eq!(section["host_copied_bytes"], 864);
        assert_eq!(section["scan_host_copied_frames"], 6);
        assert_eq!(section["scan_host_copied_bytes"], 432);
        assert_eq!(section["why"]["decoder"], "decoder reason");
        let notices = section["notices"].as_array().expect("notices");
        assert!(
            notices
                .iter()
                .any(|notice| notice.as_str().is_some_and(|text| text.contains("rav1e"))),
            "{notices:?}"
        );
    }

    #[test]
    fn a_gpu_runtime_with_cpu_codecs_names_upload_and_download() {
        let path = FramePath::new(&execution(
            RuntimeKind::Cuda,
            DecoderBackend::RustyVp9,
            EncoderBackend::Rav1r,
        ));
        let section = path.section();
        assert_eq!(section["decoder"]["route"], "host upload");
        assert_eq!(section["encoder"]["route"], "host download");
    }

    #[test]
    fn a_later_choice_keeps_the_counts() {
        let mut path = FramePath::new(&execution(
            RuntimeKind::Cpu,
            DecoderBackend::Dav1d,
            EncoderBackend::Rav1r,
        ));
        path.set_scans(HostCopies {
            frames: 3,
            bytes: 9,
        });
        path.choose(&execution(
            RuntimeKind::Cpu,
            DecoderBackend::Hevc,
            EncoderBackend::Rav1r,
        ));
        assert_eq!(path.decoder(), DecoderBackend::Hevc);
        assert_eq!(path.section()["scan_host_copied_frames"], 3);
    }
}
