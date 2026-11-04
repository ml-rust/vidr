//! What this build holds and what this machine answers.

use super::choice::RuntimeChoice;

/// What one probe says: nothing, or why the thing is missing.
pub type Probe = Result<(), String>;

/// The features this build holds. A test states any combination.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Compiled {
    /// `cuda`: the CUDA runtime, NVDEC, and NVENC.
    pub cuda: bool,
    /// `wgpu`: the WebGPU runtime.
    pub wgpu: bool,
    /// `vulkan-video`: decode and encode on Vulkan Video, on the WebGPU runtime.
    pub vulkan_video: bool,
    /// `dav1d`: AV1 decode.
    pub dav1d: bool,
    /// `cpu-vp9`: VP9 decode.
    pub cpu_vp9: bool,
    /// `cpu-hevc`: HEVC decode.
    pub cpu_hevc: bool,
    /// `openh264`: H.264 Baseline decode and encode, compiled from source.
    pub openh264: bool,
    /// `openh264-cisco`: the same on Cisco's binary.
    pub openh264_cisco: bool,
    /// `cpu-av1`: AV1 encode.
    pub cpu_av1: bool,
}

impl Compiled {
    /// The features of this build.
    pub const fn this_build() -> Self {
        Self {
            cuda: cfg!(feature = "cuda"),
            wgpu: cfg!(feature = "wgpu"),
            vulkan_video: cfg!(feature = "vulkan-video"),
            dav1d: cfg!(feature = "dav1d"),
            cpu_vp9: cfg!(feature = "cpu-vp9"),
            cpu_hevc: cfg!(feature = "cpu-hevc"),
            openh264: cfg!(feature = "openh264"),
            openh264_cisco: cfg!(feature = "openh264-cisco"),
            cpu_av1: cfg!(feature = "cpu-av1"),
        }
    }
}

/// What this machine answers. A feature that is off answers with its name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Machine {
    /// CUDA device 0.
    pub cuda: Probe,
    /// `libnvcuvid.so.1` and each function vidr calls.
    pub nvdec: Probe,
    /// `libnvidia-encode.so.1`, each function vidr calls, and its API version.
    pub nvenc: Probe,
    /// A WebGPU adapter.
    pub wgpu: Probe,
    /// A device with an H.264 Vulkan Video decode queue.
    pub vulkan: Probe,
    /// A device with a Vulkan Video encode queue and a codec vidr names.
    pub vulkan_encode: Probe,
}

impl Machine {
    /// Asks this machine. `--runtime cpu` asks nothing, so a run on the CPU never
    /// touches a GPU driver.
    pub fn probe(runtime: RuntimeChoice) -> Self {
        if runtime == RuntimeChoice::Cpu {
            let skipped = || Err("--runtime cpu asks no device".to_owned());
            return Self {
                cuda: skipped(),
                nvdec: skipped(),
                nvenc: skipped(),
                wgpu: skipped(),
                vulkan: skipped(),
                vulkan_encode: skipped(),
            };
        }
        let cuda = probe_cuda();
        Self {
            nvdec: probe_nvdec(cuda.as_ref().err()),
            nvenc: probe_nvenc(cuda.as_ref().err()),
            cuda,
            wgpu: probe_wgpu(),
            vulkan: probe_vulkan(),
            vulkan_encode: probe_vulkan_encode(),
        }
    }
}

#[cfg(feature = "cuda")]
fn probe_cuda() -> Probe {
    if numr::runtime::cuda::is_cuda_available() {
        Ok(())
    } else {
        Err("CUDA device 0 does not answer".to_owned())
    }
}

#[cfg(not(feature = "cuda"))]
fn probe_cuda() -> Probe {
    Err("the cuda feature is off".to_owned())
}

/// The library check needs no device. `cuda` still answers first, because NVDEC
/// runs on numr's CUDA context.
#[cfg(feature = "cuda")]
fn probe_nvdec(_cuda: Option<&String>) -> Probe {
    vidr::decode::nvdec::nvdec_available().map_err(|error| error.to_string())
}

#[cfg(not(feature = "cuda"))]
fn probe_nvdec(cuda: Option<&String>) -> Probe {
    Err(cuda.cloned().unwrap_or_default())
}

#[cfg(feature = "cuda")]
fn probe_nvenc(_cuda: Option<&String>) -> Probe {
    vidr::encode::nvenc::nvenc_available().map_err(|error| error.to_string())
}

#[cfg(not(feature = "cuda"))]
fn probe_nvenc(cuda: Option<&String>) -> Probe {
    Err(cuda.cloned().unwrap_or_default())
}

#[cfg(feature = "wgpu")]
fn probe_wgpu() -> Probe {
    if numr::runtime::wgpu::is_wgpu_available() {
        Ok(())
    } else {
        Err("no WebGPU adapter answers".to_owned())
    }
}

#[cfg(not(feature = "wgpu"))]
fn probe_wgpu() -> Probe {
    Err("the wgpu feature is off".to_owned())
}

/// This opens no device, so numr can still open its own.
#[cfg(feature = "vulkan-video")]
fn probe_vulkan() -> Probe {
    vidr::vulkan::vulkan_video_available().map_err(|error| error.to_string())
}

#[cfg(not(feature = "vulkan-video"))]
fn probe_vulkan() -> Probe {
    Err("the vulkan-video feature is off".to_owned())
}

/// This opens no device, so numr can still open its own.
#[cfg(feature = "vulkan-video")]
fn probe_vulkan_encode() -> Probe {
    vidr::vulkan::vulkan_encode_available().map_err(|error| error.to_string())
}

#[cfg(not(feature = "vulkan-video"))]
fn probe_vulkan_encode() -> Probe {
    Err("the vulkan-video feature is off".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_feature_that_is_off_answers_with_its_name() {
        let machine = Machine::probe(RuntimeChoice::Auto);
        if !cfg!(feature = "cuda") {
            for probe in [&machine.cuda, &machine.nvdec, &machine.nvenc] {
                assert!(probe.as_ref().is_err_and(|text| text.contains("cuda")));
            }
        }
        if !cfg!(feature = "wgpu") {
            assert!(
                machine
                    .wgpu
                    .as_ref()
                    .is_err_and(|text| text.contains("wgpu"))
            );
        }
        if !cfg!(feature = "vulkan-video") {
            assert!(
                machine
                    .vulkan
                    .as_ref()
                    .is_err_and(|text| text.contains("vulkan-video"))
            );
            assert!(
                machine
                    .vulkan_encode
                    .as_ref()
                    .is_err_and(|text| text.contains("vulkan-video"))
            );
        }
    }

    #[test]
    fn the_cpu_runtime_asks_no_device() {
        let machine = Machine::probe(RuntimeChoice::Cpu);
        assert!(machine.cuda.is_err() && machine.nvdec.is_err());
        assert!(machine.nvenc.is_err() && machine.wgpu.is_err() && machine.vulkan.is_err());
        assert!(machine.vulkan_encode.is_err());
    }
}
