//! [`execution_for`]: the one place that chooses a run's runtime and codecs.
//!
//! | Runtime | Decoders | Encoders |
//! | --- | --- | --- |
//! | CUDA | NVDEC, CPU | NVENC, CPU |
//! | WebGPU | Vulkan Video for H.264, HEVC, VP9, and AV1, CPU | CPU |
//! | CPU | CPU | CPU |
//!
//! `auto` prefers the hardware path, then the CPU. A library check says only that
//! the driver loads, so a capability refusal comes from the open or the first
//! packet. [`Selector::cpu_decoder_after`] and [`Selector::cpu_encoder_after`] take
//! the fall-through from there.

use crate::request::{Codec, Request};

use super::availability::{Compiled, Machine};
use super::choice::{
    DecoderBackend, DecoderChoice, EncoderBackend, EncoderChoice, Openh264Build, RuntimeChoice,
    RuntimeKind,
};
use super::error::ExecutionError;

/// `profile_idc` of H.264 Baseline.
const BASELINE: u8 = 66;
/// Why Vulkan Video and another runtime do not fit.
const VULKAN_RUNTIME: &str = "Vulkan Video frames live on WebGPU tensors";

/// The part of a request that picks the execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecutionRequest {
    /// `--runtime`.
    pub runtime: RuntimeChoice,
    /// `--decoder`.
    pub decoder: DecoderChoice,
    /// `--encoder`.
    pub encoder: EncoderChoice,
    /// `--codec`.
    pub codec: Codec,
    /// Whether `--openh264-library` names a binary.
    pub openh264_library: bool,
}

impl From<&Request> for ExecutionRequest {
    fn from(request: &Request) -> Self {
        Self {
            runtime: request.runtime,
            decoder: request.decoder,
            encoder: request.encoder,
            codec: request.codec,
            openh264_library: request.openh264_library.is_some(),
        }
    }
}

/// The codec of a video a decoder reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SourceCodec {
    /// H.264.
    H264,
    /// HEVC.
    Hevc,
    /// AV1.
    Av1,
    /// VP9.
    Vp9,
}

impl SourceCodec {
    /// The name in a sentence.
    pub const fn name(self) -> &'static str {
        match self {
            Self::H264 => "H.264",
            Self::Hevc => "HEVC",
            Self::Av1 => "AV1",
            Self::Vp9 => "VP9",
        }
    }
}

/// What the decoder choice reads from a video.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceFacts {
    /// The codec.
    pub codec: SourceCodec,
    /// `profile_idc` of an H.264 track, when the build reads it.
    pub h264_profile: Option<u8>,
}

/// Why each part of an execution was chosen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reasons {
    /// Why this runtime.
    pub runtime: String,
    /// Why this decoder.
    pub decoder: String,
    /// Why this encoder.
    pub encoder: String,
    /// Why this decoder reads the output back.
    pub check_decoder: String,
}

/// What a run uses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Execution {
    /// The numr backend that holds the frames.
    pub runtime: RuntimeKind,
    /// The decoder of the encode pass and the source's pre-scan.
    pub decoder: DecoderBackend,
    /// The encoder.
    pub encoder: EncoderBackend,
    /// The decoder that reads the output back for the output check.
    pub check_decoder: DecoderBackend,
    /// Why each part was chosen.
    pub reasons: Reasons,
}

/// Chooses `(runtime, decoder, encoder)` from the request, the source's codec, what
/// this build holds, and what this machine answers.
///
/// # Errors
///
/// An [`ExecutionError`] that names the sides, the feature, or the probe's text,
/// and the next action.
pub fn execution_for(
    request: &ExecutionRequest,
    source: &SourceFacts,
    compiled: &Compiled,
    machine: &Machine,
) -> Result<Execution, ExecutionError> {
    Selector::new(*request, *compiled, machine.clone()).select(source)
}

/// A request and its availability, for the first choice and each later one a run
/// makes: a reused video's decoder, and the fall-through of `auto`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selector {
    request: ExecutionRequest,
    compiled: Compiled,
    machine: Machine,
}

impl Selector {
    /// A selector over `request`, `compiled`, and `machine`.
    pub fn new(request: ExecutionRequest, compiled: Compiled, machine: Machine) -> Self {
        Self {
            request,
            compiled,
            machine,
        }
    }

    /// The choice for a run whose source is `source`.
    ///
    /// # Errors
    ///
    /// As [`execution_for`].
    pub fn select(&self, source: &SourceFacts) -> Result<Execution, ExecutionError> {
        let (runtime, runtime_reason) = self.runtime()?;
        let (decoder, decoder_reason) = self.decoder_for(runtime, source)?;
        let (encoder, encoder_reason) = self.encoder_on(runtime)?;
        let (check_decoder, check_reason) = self.check_decoder(runtime, decoder, encoder)?;
        Ok(Execution {
            runtime,
            decoder,
            encoder,
            check_decoder,
            reasons: Reasons {
                runtime: runtime_reason,
                decoder: decoder_reason,
                encoder: encoder_reason,
                check_decoder: check_reason,
            },
        })
    }

    /// Whether `--decoder vulkan` holds.
    fn wants_vulkan(&self) -> bool {
        self.request.decoder == DecoderChoice::Vulkan
    }

    /// Refuses `--decoder vulkan` on a runtime with no path for it.
    fn refuse_vulkan_on(&self, runtime: &str) -> Result<(), ExecutionError> {
        if self.wants_vulkan() {
            return Err(ExecutionError::Pair {
                first: runtime.to_owned(),
                second: "--decoder vulkan".to_owned(),
                reason: VULKAN_RUNTIME,
                working: "--runtime wgpu with that flag, or --decoder cpu",
            });
        }
        Ok(())
    }

    fn require_wgpu(&self, what: &str) -> Result<(), ExecutionError> {
        if !self.compiled.wgpu {
            return Err(ExecutionError::NotBuilt {
                what: what.to_owned(),
                feature: "wgpu",
            });
        }
        if let Err(reason) = &self.machine.wgpu {
            return Err(ExecutionError::Unavailable {
                what: what.to_owned(),
                reason: reason.clone(),
                next: "Install a WebGPU driver, or pass --runtime cpu.",
            });
        }
        Ok(())
    }

    /// The CUDA flag that names a hardware path, when one does.
    fn hardware_flag(&self) -> Option<&'static str> {
        if self.request.decoder == DecoderChoice::Nvdec {
            Some("--decoder nvdec")
        } else if self.request.encoder == EncoderChoice::Nvenc {
            Some("--encoder nvenc")
        } else {
            None
        }
    }

    fn require_cuda(&self, what: &str) -> Result<(), ExecutionError> {
        if !self.compiled.cuda {
            return Err(ExecutionError::NotBuilt {
                what: what.to_owned(),
                feature: "cuda",
            });
        }
        self.machine
            .cuda
            .as_ref()
            .map_err(|reason| ExecutionError::Unavailable {
                what: what.to_owned(),
                reason: reason.clone(),
                next: "Run from a terminal with NVIDIA GPU access, or pass --runtime cpu.",
            })
            .copied()
    }

    /// Refuses a hardware flag on a runtime with no hardware path for it.
    fn refuse_hardware_on(
        &self,
        runtime: &str,
        working: &'static str,
    ) -> Result<(), ExecutionError> {
        match self.hardware_flag() {
            Some(flag) => Err(ExecutionError::Pair {
                first: runtime.to_owned(),
                second: flag.to_owned(),
                reason: "NVDEC and NVENC read and write CUDA tensors only",
                working,
            }),
            None => Ok(()),
        }
    }

    fn runtime(&self) -> Result<(RuntimeKind, String), ExecutionError> {
        match self.request.runtime {
            RuntimeChoice::Auto => self.auto_runtime(),
            RuntimeChoice::Cuda => {
                self.refuse_vulkan_on("--runtime cuda")?;
                self.require_cuda("--runtime cuda")?;
                Ok((RuntimeKind::Cuda, "--runtime cuda".to_owned()))
            }
            RuntimeChoice::Wgpu => {
                self.refuse_hardware_on(
                    "--runtime wgpu",
                    "--runtime cuda with that flag, or --runtime wgpu --decoder cpu --encoder cpu",
                )?;
                self.require_wgpu("--runtime wgpu")?;
                Ok((RuntimeKind::Wgpu, "--runtime wgpu".to_owned()))
            }
            RuntimeChoice::Cpu => {
                self.refuse_hardware_on(
                    "--runtime cpu",
                    "--runtime cuda with that flag, or --runtime cpu --decoder cpu --encoder cpu",
                )?;
                self.refuse_vulkan_on("--runtime cpu")?;
                Ok((RuntimeKind::Cpu, "--runtime cpu".to_owned()))
            }
        }
    }

    fn auto_runtime(&self) -> Result<(RuntimeKind, String), ExecutionError> {
        if self.wants_vulkan() {
            if let Some(flag) = self.hardware_flag() {
                return Err(ExecutionError::Pair {
                    first: "--decoder vulkan".to_owned(),
                    second: flag.to_owned(),
                    reason: VULKAN_RUNTIME,
                    working: "--decoder vulkan --encoder cpu",
                });
            }
            self.require_wgpu("--decoder vulkan")?;
            return Ok((
                RuntimeKind::Wgpu,
                "auto: --decoder vulkan runs on WebGPU.".to_owned(),
            ));
        }
        if let Some(flag) = self.hardware_flag() {
            self.require_cuda(flag)?;
            return Ok((RuntimeKind::Cuda, format!("auto: {flag} runs on CUDA.")));
        }
        let cuda = if self.compiled.cuda {
            self.machine.cuda.clone()
        } else {
            Err("the cuda feature is off".to_owned())
        };
        let Err(cuda) = cuda else {
            return Ok((RuntimeKind::Cuda, "auto: CUDA device 0 answers.".to_owned()));
        };
        let wgpu = if self.compiled.wgpu {
            self.machine.wgpu.clone()
        } else {
            Err("the wgpu feature is off".to_owned())
        };
        let Err(wgpu) = wgpu else {
            return Ok((
                RuntimeKind::Wgpu,
                format!("auto: CUDA is out: {cuda}. A WebGPU adapter answers."),
            ));
        };
        Ok((
            RuntimeKind::Cpu,
            format!("auto: CUDA is out: {cuda}. WebGPU is out: {wgpu}. The CPU needs no device."),
        ))
    }

    /// The hardware decoder of `codec` that `runtime` holds, or why none fits.
    fn hardware_decoder(&self, runtime: RuntimeKind) -> Result<DecoderBackend, String> {
        match runtime {
            RuntimeKind::Cuda => self
                .machine
                .nvdec
                .clone()
                .map(|()| DecoderBackend::Nvdec)
                .map_err(|reason| format!("NVDEC is out: {reason}")),
            RuntimeKind::Wgpu => self
                .machine
                .vulkan
                .clone()
                .map(|()| DecoderBackend::Vulkan)
                .map_err(|reason| format!("Vulkan Video is out: {reason}")),
            RuntimeKind::Cpu => Err("the CPU runtime has no hardware decoder".to_owned()),
        }
    }

    /// NVENC when `runtime` holds it, or why not.
    fn hardware_encoder(&self, runtime: RuntimeKind) -> Result<EncoderBackend, String> {
        match runtime {
            RuntimeKind::Cuda => self
                .machine
                .nvenc
                .clone()
                .map(|()| EncoderBackend::Nvenc)
                .map_err(|reason| format!("NVENC is out: {reason}")),
            RuntimeKind::Wgpu => {
                Err("the WebGPU runtime has no hardware encoder in this release".to_owned())
            }
            RuntimeKind::Cpu => Err("the CPU runtime has no hardware encoder".to_owned()),
        }
    }

    fn openh264_build(&self) -> Result<Openh264Build, ExecutionError> {
        if self.request.openh264_library {
            if self.compiled.openh264_cisco {
                return Ok(Openh264Build::Cisco);
            }
            return Err(ExecutionError::NotBuilt {
                what: "--openh264-library".to_owned(),
                feature: "openh264-cisco",
            });
        }
        if self.compiled.openh264 {
            return Ok(Openh264Build::Source);
        }
        if self.compiled.openh264_cisco {
            return Err(ExecutionError::CiscoLibraryMissing);
        }
        Err(ExecutionError::NotBuilt {
            what: "H.264 CPU decode and encode".to_owned(),
            feature: "openh264",
        })
    }

    /// The CPU decoder of `video`, or the refusal that names what is missing.
    fn cpu_decoder(&self, video: &SourceFacts) -> Result<DecoderBackend, ExecutionError> {
        let codec = video.codec;
        let built = |on: bool, feature, backend| {
            if on {
                Ok(backend)
            } else {
                Err(ExecutionError::not_built_decoder(codec, feature))
            }
        };
        match codec {
            SourceCodec::Av1 => built(self.compiled.dav1d, "dav1d", DecoderBackend::Dav1d),
            SourceCodec::Vp9 => built(self.compiled.cpu_vp9, "cpu-vp9", DecoderBackend::RustyVp9),
            SourceCodec::Hevc => built(self.compiled.cpu_hevc, "cpu-hevc", DecoderBackend::Hevc),
            SourceCodec::H264 => {
                let build = self.openh264_build()?;
                match video.h264_profile {
                    Some(profile_idc) if profile_idc != BASELINE => {
                        Err(ExecutionError::H264NotBaseline { profile_idc })
                    }
                    _ => Ok(DecoderBackend::Openh264(build)),
                }
            }
        }
    }

    /// The decoder of `video` on `runtime`, and why.
    ///
    /// # Errors
    ///
    /// As [`execution_for`].
    pub fn decoder_for(
        &self,
        runtime: RuntimeKind,
        video: &SourceFacts,
    ) -> Result<(DecoderBackend, String), ExecutionError> {
        let cpu_reason = |backend: DecoderBackend| {
            format!("{backend} decodes {} on the CPU.", video.codec.name())
        };
        match self.request.decoder {
            DecoderChoice::Nvdec => self
                .machine
                .nvdec
                .clone()
                .map(|()| (DecoderBackend::Nvdec, "--decoder nvdec".to_owned()))
                .map_err(|reason| ExecutionError::Unavailable {
                    what: "--decoder nvdec".to_owned(),
                    reason,
                    next: "Install the NVIDIA driver, or pass --decoder cpu.",
                }),
            DecoderChoice::Vulkan => {
                if !self.compiled.vulkan_video {
                    return Err(ExecutionError::NotBuilt {
                        what: "--decoder vulkan".to_owned(),
                        feature: "vulkan-video",
                    });
                }
                self.machine
                    .vulkan
                    .clone()
                    .map(|()| (DecoderBackend::Vulkan, "--decoder vulkan".to_owned()))
                    .map_err(|reason| ExecutionError::Unavailable {
                        what: "--decoder vulkan".to_owned(),
                        reason,
                        next: "Install a driver with VK_KHR_video_decode_h264, or pass --decoder \
                               cpu.",
                    })
            }
            DecoderChoice::Cpu => {
                let backend = self.cpu_decoder(video)?;
                Ok((backend, format!("--decoder cpu: {}", cpu_reason(backend))))
            }
            DecoderChoice::Auto => match self.hardware_decoder(runtime) {
                Ok(backend) => Ok((
                    backend,
                    match backend {
                        DecoderBackend::Vulkan => "auto: Vulkan Video is present.".to_owned(),
                        _ => "auto: NVDEC is present.".to_owned(),
                    },
                )),
                Err(hardware) => match self.cpu_decoder(video) {
                    Ok(backend) => Ok((
                        backend,
                        format!("auto: {hardware}. {}", cpu_reason(backend)),
                    )),
                    Err(cpu) => Err(ExecutionError::NoPath {
                        side: "decoder",
                        hardware,
                        cpu: Box::new(cpu),
                    }),
                },
            },
        }
    }

    fn cpu_encoder(&self) -> Result<EncoderBackend, ExecutionError> {
        match self.request.codec {
            Codec::Hevc => Err(ExecutionError::HevcCpuEncode),
            Codec::H264 => self.openh264_build().map(EncoderBackend::Openh264),
            Codec::Av1 if self.compiled.cpu_av1 => Ok(EncoderBackend::Rav1r),
            Codec::Av1 => Err(ExecutionError::NotBuilt {
                what: "AV1 CPU encode".to_owned(),
                feature: "cpu-av1",
            }),
        }
    }

    fn encoder_on(&self, runtime: RuntimeKind) -> Result<(EncoderBackend, String), ExecutionError> {
        let codec = self.request.codec;
        let cpu_reason = |backend: EncoderBackend| format!("{backend} encodes {codec} on the CPU.");
        match self.request.encoder {
            EncoderChoice::Nvenc => self
                .machine
                .nvenc
                .clone()
                .map(|()| (EncoderBackend::Nvenc, "--encoder nvenc".to_owned()))
                .map_err(|reason| ExecutionError::Unavailable {
                    what: "--encoder nvenc".to_owned(),
                    reason,
                    next: "Install the NVIDIA driver, or pass --encoder cpu.",
                }),
            EncoderChoice::Cpu => {
                let backend = self.cpu_encoder()?;
                Ok((backend, format!("--encoder cpu: {}", cpu_reason(backend))))
            }
            EncoderChoice::Auto => match self.hardware_encoder(runtime) {
                Ok(backend) => Ok((backend, "auto: NVENC is present.".to_owned())),
                Err(hardware) => match self.cpu_encoder() {
                    Ok(backend) => Ok((
                        backend,
                        format!("auto: {hardware}. {}", cpu_reason(backend)),
                    )),
                    Err(cpu) => Err(ExecutionError::NoPath {
                        side: "encoder",
                        hardware,
                        cpu: Box::new(cpu),
                    }),
                },
            },
        }
    }

    /// Whether `decoder` reads what `encoder` writes at the request's codec.
    /// NVENC writes H.264 High, which OpenH264 refuses.
    fn reads_output(&self, decoder: DecoderBackend, encoder: EncoderBackend) -> bool {
        match decoder {
            DecoderBackend::Nvdec | DecoderBackend::Vulkan => true,
            DecoderBackend::Dav1d => self.request.codec == Codec::Av1,
            DecoderBackend::Hevc => self.request.codec == Codec::Hevc,
            DecoderBackend::RustyVp9 => false,
            DecoderBackend::Openh264(_) => {
                self.request.codec == Codec::H264 && matches!(encoder, EncoderBackend::Openh264(_))
            }
        }
    }

    /// The CPU decoder that reads this run's output, when the build holds one.
    fn cpu_check_decoder(&self, encoder: EncoderBackend) -> Option<DecoderBackend> {
        match (self.request.codec, encoder) {
            (Codec::Av1, _) if self.compiled.dav1d => Some(DecoderBackend::Dav1d),
            (Codec::Hevc, _) if self.compiled.cpu_hevc => Some(DecoderBackend::Hevc),
            (Codec::H264, EncoderBackend::Openh264(build)) => Some(DecoderBackend::Openh264(build)),
            _ => None,
        }
    }

    fn check_decoder(
        &self,
        runtime: RuntimeKind,
        decoder: DecoderBackend,
        encoder: EncoderBackend,
    ) -> Result<(DecoderBackend, String), ExecutionError> {
        if self.reads_output(decoder, encoder) {
            return Ok((decoder, "The run's decoder reads the output.".to_owned()));
        }
        let hardware = self.hardware_decoder(runtime).ok();
        let cpu = self.cpu_check_decoder(encoder);
        let first = if self.request.decoder == DecoderChoice::Cpu {
            cpu.or(hardware)
        } else {
            hardware.or(cpu)
        };
        match first {
            Some(backend) => Ok((
                backend,
                format!(
                    "{decoder} does not read {} output, so {backend} checks it.",
                    self.request.codec
                ),
            )),
            None => Err(ExecutionError::NoOutputCheck {
                codec: self.request.codec,
                reason: match self.request.codec {
                    Codec::Av1 => {
                        "Rebuild with --features dav1d, or use NVDEC on a GPU that lists AV1."
                    }
                    Codec::Hevc => "Rebuild with --features cpu-hevc, or use NVDEC.",
                    Codec::H264 => {
                        "NVENC writes High, and OpenH264 reads Baseline only. Use NVDEC, or pass \
                         --encoder cpu."
                    }
                },
            }),
        }
    }

    /// Chooses the output check's decoder again, after a run's decoder or encoder
    /// changed.
    ///
    /// # Errors
    ///
    /// [`ExecutionError::NoOutputCheck`].
    pub fn refresh_check(&self, execution: &mut Execution) -> Result<(), ExecutionError> {
        let (check_decoder, reason) =
            self.check_decoder(execution.runtime, execution.decoder, execution.encoder)?;
        execution.check_decoder = check_decoder;
        execution.reasons.check_decoder = reason;
        Ok(())
    }

    /// Whether `--decoder auto` holds, so a capability refusal can fall through.
    pub const fn decoder_is_auto(&self) -> bool {
        matches!(self.request.decoder, DecoderChoice::Auto)
    }

    /// The CPU decoder of `video`, after a hardware decoder refused it for `reason`.
    ///
    /// # Errors
    ///
    /// [`ExecutionError::NoPath`] names `reason` and what the CPU path lacks.
    pub fn cpu_decoder_after(
        &self,
        video: &SourceFacts,
        reason: &str,
    ) -> Result<(DecoderBackend, String), ExecutionError> {
        match self.cpu_decoder(video) {
            Ok(backend) => Ok((
                backend,
                format!(
                    "auto: the hardware decoder refused the video: {reason} {backend} decodes {} \
                     on the CPU.",
                    video.codec.name()
                ),
            )),
            Err(cpu) => Err(ExecutionError::NoPath {
                side: "decoder",
                hardware: format!("the hardware decoder refused the video: {reason}"),
                cpu: Box::new(cpu),
            }),
        }
    }

    /// The CPU encoder that replaces NVENC after its refusal for `reason`, or `None`
    /// when `--encoder auto` does not hold or no CPU encoder takes the frames.
    /// OpenH264 takes no 10-bit frames.
    pub fn cpu_encoder_after(
        &self,
        ten_bit: bool,
        reason: &str,
    ) -> Option<(EncoderBackend, String)> {
        if self.request.encoder != EncoderChoice::Auto {
            return None;
        }
        let backend = self.cpu_encoder().ok()?;
        if ten_bit && matches!(backend, EncoderBackend::Openh264(_)) {
            return None;
        }
        Some((
            backend,
            format!(
                "auto: NVENC refused the output: {reason} {backend} encodes {} on the CPU.",
                self.request.codec
            ),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn av1() -> SourceFacts {
        SourceFacts {
            codec: SourceCodec::Av1,
            h264_profile: None,
        }
    }

    fn machine_without_gpu() -> Machine {
        let out = || Err("absent".to_owned());
        Machine {
            cuda: out(),
            nvdec: out(),
            nvenc: out(),
            wgpu: out(),
            vulkan: out(),
        }
    }

    fn cpu_build() -> Compiled {
        Compiled {
            dav1d: true,
            cpu_av1: true,
            ..Compiled::default()
        }
    }

    fn selector(request: ExecutionRequest, compiled: Compiled) -> Selector {
        Selector::new(request, compiled, machine_without_gpu())
    }

    #[test]
    fn the_output_check_falls_back_when_the_encoder_changes() {
        let request = ExecutionRequest {
            codec: Codec::Av1,
            ..ExecutionRequest::default()
        };
        let selector = selector(request, cpu_build());
        let mut execution = selector.select(&av1()).expect("execution");
        assert_eq!(execution.check_decoder, DecoderBackend::Dav1d);
        execution.decoder = DecoderBackend::RustyVp9;
        selector.refresh_check(&mut execution).expect("check");
        assert_eq!(execution.check_decoder, DecoderBackend::Dav1d);
        assert!(execution.reasons.check_decoder.contains("rusty_vp9"));
    }

    #[test]
    fn nvdec_falls_through_to_the_cpu_decoder_and_keeps_the_reason() {
        let selector = selector(ExecutionRequest::default(), cpu_build());
        let (backend, reason) = selector
            .cpu_decoder_after(&av1(), "No AV1 here.")
            .expect("cpu");
        assert_eq!(backend, DecoderBackend::Dav1d);
        assert!(reason.contains("No AV1 here."), "{reason}");
        let vp9 = SourceFacts {
            codec: SourceCodec::Vp9,
            h264_profile: None,
        };
        let error = selector
            .cpu_decoder_after(&vp9, "No VP9 here.")
            .expect_err("refused");
        let text = error.to_string();
        assert!(text.contains("No VP9 here."), "{text}");
        assert!(text.contains("cpu-vp9"), "{text}");
    }

    #[test]
    fn nvenc_falls_through_only_for_auto_and_for_frames_the_encoder_takes() {
        let av1_output = ExecutionRequest {
            codec: Codec::Av1,
            ..ExecutionRequest::default()
        };
        let auto = selector(av1_output, cpu_build());
        let (backend, reason) = auto.cpu_encoder_after(true, "No AV1.").expect("rav1r");
        assert_eq!(backend, EncoderBackend::Rav1r);
        assert!(reason.contains("No AV1."), "{reason}");

        let forced = ExecutionRequest {
            encoder: EncoderChoice::Nvenc,
            ..av1_output
        };
        assert_eq!(
            selector(forced, cpu_build()).cpu_encoder_after(false, "x"),
            None
        );

        let h264_output = ExecutionRequest {
            codec: Codec::H264,
            ..ExecutionRequest::default()
        };
        let openh264 = Compiled {
            openh264: true,
            ..Compiled::default()
        };
        let h264 = selector(h264_output, openh264);
        assert_eq!(h264.cpu_encoder_after(true, "x"), None);
        assert!(h264.cpu_encoder_after(false, "x").is_some());
    }
}
