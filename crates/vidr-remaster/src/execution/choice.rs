//! The values of `--runtime`, `--decoder`, and `--encoder`, and what `auto` and
//! `cpu` resolve to.
//!
//! The WebGPU runtime is Vulkan Video's home, and `Selector` holds one match per
//! runtime for the hardware side.

use std::fmt;
use std::str::FromStr;

use vidr::FrameRoute;

/// A value of a choice flag that the flag does not take.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseChoiceError {
    flag: &'static str,
    name: String,
    choices: &'static str,
}

impl fmt::Display for ParseChoiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Unsupported {} '{}'. Use {}.",
            self.flag, self.name, self.choices
        )
    }
}

impl std::error::Error for ParseChoiceError {}

/// One command-line choice: its type, flag, values, and help names.
macro_rules! choice {
    (
        $(#[$meta:meta])*
        $name:ident, $flag:literal, $choices:literal,
        { $($(#[$value_meta:meta])* $value:ident => $text:literal,)+ }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        pub enum $name {
            /// Prefers the hardware path, then the CPU.
            #[default]
            Auto,
            $($(#[$value_meta])* $value,)+
        }

        impl $name {
            /// Every value, in the order the help text lists them.
            pub const ALL: &[Self] = &[Self::Auto, $(Self::$value,)+];

            /// The name on the command line.
            pub const fn as_str(self) -> &'static str {
                match self {
                    Self::Auto => "auto",
                    $(Self::$value => $text,)+
                }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl FromStr for $name {
            type Err = ParseChoiceError;

            fn from_str(text: &str) -> Result<Self, Self::Err> {
                Self::ALL
                    .iter()
                    .copied()
                    .find(|value| value.as_str() == text)
                    .ok_or_else(|| ParseChoiceError {
                        flag: $flag,
                        name: text.to_owned(),
                        choices: $choices,
                    })
            }
        }
    };
}

choice! {
    /// `--runtime`: the numr backend that holds the frames.
    RuntimeChoice, "--runtime", "auto, cuda, wgpu, or cpu",
    {
        /// numr's CUDA backend.
        Cuda => "cuda",
        /// numr's WebGPU backend.
        Wgpu => "wgpu",
        /// numr's CPU backend.
        Cpu => "cpu",
    }
}

choice! {
    /// `--decoder`.
    DecoderChoice, "--decoder", "auto, nvdec, vulkan, or cpu",
    {
        /// NVDEC, on the CUDA runtime.
        Nvdec => "nvdec",
        /// Vulkan Video, on the WebGPU runtime.
        Vulkan => "vulkan",
        /// The CPU decoder of the source's codec.
        Cpu => "cpu",
    }
}

choice! {
    /// `--encoder`.
    EncoderChoice, "--encoder", "auto, nvenc, or cpu",
    {
        /// NVENC, on the CUDA runtime.
        Nvenc => "nvenc",
        /// The CPU encoder of the output's codec.
        Cpu => "cpu",
    }
}

/// The numr backend a run uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RuntimeKind {
    /// numr's CUDA backend.
    Cuda,
    /// numr's WebGPU backend.
    Wgpu,
    /// numr's CPU backend.
    Cpu,
}

impl RuntimeKind {
    /// The name in a report.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
            Self::Cpu => "cpu",
        }
    }

    /// The name in a sentence.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Wgpu => "WebGPU",
            Self::Cpu => "CPU",
        }
    }
}

impl fmt::Display for RuntimeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// How the OpenH264 code reaches the process.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Openh264Build {
    /// Compiled from source, with no Cisco patent cover.
    Source,
    /// Cisco's binary, loaded from `--openh264-library`.
    Cisco,
}

/// A decoder a run uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DecoderBackend {
    /// NVDEC. Frames stay in device memory.
    Nvdec,
    /// Vulkan Video, H.264, HEVC, VP9, or AV1. Frames stay in device memory.
    Vulkan,
    /// `Dav1dDecoder`, AV1.
    Dav1d,
    /// `RustyVp9Decoder`, VP9.
    RustyVp9,
    /// `HevcDecoder`, HEVC.
    Hevc,
    /// `Openh264Decoder`, H.264 Baseline.
    Openh264(Openh264Build),
}

impl DecoderBackend {
    /// The name in a report and a provenance record.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Nvdec => "nvdec",
            Self::Vulkan => "vulkan",
            Self::Dav1d => "dav1d",
            Self::RustyVp9 => "rusty_vp9",
            Self::Hevc => "hevc",
            Self::Openh264(Openh264Build::Source) => "openh264",
            Self::Openh264(Openh264Build::Cisco) => "openh264-cisco",
        }
    }

    /// The route the backend declares on `runtime`.
    pub const fn route_on(self, runtime: RuntimeKind) -> FrameRoute {
        match (self, runtime) {
            (Self::Nvdec | Self::Vulkan, _) => FrameRoute::Device,
            (_, RuntimeKind::Cpu) => FrameRoute::HostOnly,
            (_, RuntimeKind::Cuda | RuntimeKind::Wgpu) => FrameRoute::HostUpload,
        }
    }
}

impl fmt::Display for DecoderBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// An encoder a run uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EncoderBackend {
    /// NVENC. Frames stay in device memory.
    Nvenc,
    /// `Rav1rEncoder`, AV1.
    Rav1r,
    /// `Openh264Encoder`, H.264 Constrained Baseline.
    Openh264(Openh264Build),
}

impl EncoderBackend {
    /// The name in a report and a provenance record.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Nvenc => "nvenc",
            Self::Rav1r => "rav1r",
            Self::Openh264(Openh264Build::Source) => "openh264",
            Self::Openh264(Openh264Build::Cisco) => "openh264-cisco",
        }
    }

    /// The route the backend declares on `runtime`.
    pub const fn route_on(self, runtime: RuntimeKind) -> FrameRoute {
        match (self, runtime) {
            (Self::Nvenc, _) => FrameRoute::Device,
            (_, RuntimeKind::Cpu) => FrameRoute::HostOnly,
            (_, RuntimeKind::Cuda | RuntimeKind::Wgpu) => FrameRoute::HostDownload,
        }
    }
}

impl fmt::Display for EncoderBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_choice_parses_every_name_it_prints() {
        for value in RuntimeChoice::ALL {
            assert_eq!(value.as_str().parse::<RuntimeChoice>().as_ref(), Ok(value));
        }
        for value in DecoderChoice::ALL {
            assert_eq!(value.as_str().parse::<DecoderChoice>().as_ref(), Ok(value));
        }
        for value in EncoderChoice::ALL {
            assert_eq!(value.as_str().parse::<EncoderChoice>().as_ref(), Ok(value));
        }
    }

    #[test]
    fn an_unknown_value_names_the_flag_and_the_choices() {
        let error = "vaapi".parse::<DecoderChoice>().expect_err("refused");
        assert_eq!(
            error.to_string(),
            "Unsupported --decoder 'vaapi'. Use auto, nvdec, vulkan, or cpu."
        );
    }

    #[test]
    fn a_hardware_backend_stays_on_the_device_and_a_cpu_backend_copies() {
        assert_eq!(
            DecoderBackend::Nvdec.route_on(RuntimeKind::Cuda),
            FrameRoute::Device
        );
        assert_eq!(
            DecoderBackend::Vulkan.route_on(RuntimeKind::Wgpu),
            FrameRoute::Device
        );
        assert_eq!(
            EncoderBackend::Nvenc.route_on(RuntimeKind::Cuda),
            FrameRoute::Device
        );
        assert_eq!(
            DecoderBackend::Dav1d.route_on(RuntimeKind::Cuda),
            FrameRoute::HostUpload
        );
        assert_eq!(
            EncoderBackend::Rav1r.route_on(RuntimeKind::Wgpu),
            FrameRoute::HostDownload
        );
        assert_eq!(
            DecoderBackend::Hevc.route_on(RuntimeKind::Cpu),
            FrameRoute::HostOnly
        );
        assert_eq!(
            EncoderBackend::Openh264(Openh264Build::Source).route_on(RuntimeKind::Cpu),
            FrameRoute::HostOnly
        );
    }
}
