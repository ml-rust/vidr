//! What differs per numr backend: how its device opens, which hardware codecs it
//! has, and how a model runs on it.

use std::path::{Path, PathBuf};

#[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
use boostr::model::super_resolution::SrExecution;
use numr::dtype::DType;
#[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
use numr::ops::ConvLayout;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use vidr::encode::EncodeConfig;
use vidr::media::DecodeConfig;

use crate::error::RemasterError;
use crate::execution::{DecoderBackend, EncoderBackend};

use super::backends::{BoxDecoder, BoxEncoder, open_cpu_decoder, open_cpu_encoder};

/// A runtime's open device.
pub(super) struct Opened<R: Runtime> {
    /// The client a hardware codec opens on.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(super) client: R::Client,
    /// The device frames live on.
    pub(super) device: R::Device,
    /// `--openh264-library`.
    pub(super) library: Option<PathBuf>,
}

/// One numr backend a run works on.
pub(super) trait RunRuntime: Runtime<DType = DType> {
    /// The report's `backend` field for a model run.
    const BACKEND: &'static str;
    /// The runtime and dtype in the model's progress line.
    const PRECISION: &'static str;

    /// Opens the device.
    ///
    /// # Errors
    ///
    /// [`RemasterError::NoGpu`] or [`RemasterError::Gpu`] when no device answers.
    fn open(library: Option<&Path>) -> Result<Opened<Self>, RemasterError>;

    /// How a model runs here.
    #[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
    fn model_execution() -> SrExecution;

    /// Opens `backend` on the track that `decode` describes, of the file at `path`.
    ///
    /// # Errors
    ///
    /// [`RemasterError::Decode`] for the decoder's refusal, and
    /// [`RemasterError::Execution`] for a backend this runtime has no path for.
    fn open_decoder(
        opened: &Opened<Self>,
        backend: DecoderBackend,
        decode: &DecodeConfig,
        path: &Path,
    ) -> Result<BoxDecoder<Self>, RemasterError> {
        open_cpu_decoder::<Self>(
            &opened.device,
            opened.library.as_deref(),
            backend,
            decode,
            path,
        )
    }

    /// Opens `backend` at `config`.
    ///
    /// # Errors
    ///
    /// The encoder's refusal, and [`RemasterError::Execution`] for a backend this
    /// runtime has no path for.
    fn open_encoder(
        opened: &Opened<Self>,
        backend: EncoderBackend,
        config: EncodeConfig,
    ) -> Result<BoxEncoder<Self>, RemasterError> {
        open_cpu_encoder::<Self>(opened.library.as_deref(), backend, config)
    }
}

impl RunRuntime for CpuRuntime {
    const BACKEND: &'static str = "boostr on numr CPU, F32";
    const PRECISION: &'static str = "CPU F32";

    fn open(library: Option<&Path>) -> Result<Opened<Self>, RemasterError> {
        let device = CpuDevice::new();
        Ok(Opened {
            client: Self::default_client(&device),
            device,
            library: library.map(Path::to_path_buf),
        })
    }

    #[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
    fn model_execution() -> SrExecution {
        SrExecution::new(DType::F32, ConvLayout::Nchw)
    }
}

#[cfg(feature = "wgpu")]
mod webgpu {
    use numr::runtime::wgpu::{WgpuDevice, WgpuRuntime, is_wgpu_available};
    #[cfg(feature = "vulkan-video")]
    use vidr::vulkan::{VulkanDecoder, VulkanDevice, VulkanOptions};

    #[cfg(feature = "vulkan-video")]
    use super::super::backends::decode_error;
    use super::*;

    impl RunRuntime for WgpuRuntime {
        const BACKEND: &'static str = "boostr on numr WebGPU, F32";
        const PRECISION: &'static str = "WebGPU F32";

        /// Under `vulkan-video`, the device with the video queue becomes numr's device.
        /// A machine with no such device opens numr's own, and a run on it takes CPU
        /// decoders.
        fn open(library: Option<&Path>) -> Result<Opened<Self>, RemasterError> {
            #[cfg(feature = "vulkan-video")]
            if let Ok(vulkan) = VulkanDevice::open_shared(VulkanOptions::default()) {
                return Ok(Opened {
                    client: vulkan.client().clone(),
                    device: vulkan.numr_device().clone(),
                    library: library.map(Path::to_path_buf),
                });
            }
            if !is_wgpu_available() {
                return Err(RemasterError::Gpu {
                    reason: "No WebGPU adapter answers. Install a WebGPU driver, or pass \
                             --runtime cpu."
                        .to_owned(),
                });
            }
            let device = WgpuDevice::new(0);
            Ok(Opened {
                client: Self::default_client(&device),
                device,
                library: library.map(Path::to_path_buf),
            })
        }

        /// boostr documents F32 as the only float dtype of numr's WebGPU backend.
        #[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
        fn model_execution() -> SrExecution {
            SrExecution::new(DType::F32, ConvLayout::Nchw)
        }

        #[cfg(feature = "vulkan-video")]
        fn open_decoder(
            opened: &Opened<Self>,
            backend: DecoderBackend,
            decode: &DecodeConfig,
            path: &Path,
        ) -> Result<BoxDecoder<Self>, RemasterError> {
            if backend != DecoderBackend::Vulkan {
                return open_cpu_decoder::<Self>(
                    &opened.device,
                    opened.library.as_deref(),
                    backend,
                    decode,
                    path,
                );
            }
            let failed = |error| decode_error(path, error);
            let device = VulkanDevice::open_shared(VulkanOptions::default()).map_err(failed)?;
            let decoder =
                VulkanDecoder::new(&device, decode.codec, &decode.config).map_err(failed)?;
            Ok(Box::new(decoder))
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime, is_cuda_available};
    use vidr::decode::nvdec::NvdecDecoder;
    use vidr::encode::nvenc::NvencEncoder;

    use super::super::backends::decode_error;
    use super::*;
    use crate::plan::encoder_open_error;

    impl RunRuntime for CudaRuntime {
        const BACKEND: &'static str = "boostr on numr CUDA, F16";
        const PRECISION: &'static str = "CUDA FP16";

        fn open(library: Option<&Path>) -> Result<Opened<Self>, RemasterError> {
            if !is_cuda_available() {
                return Err(RemasterError::NoGpu);
            }
            let client =
                CudaClient::new(CudaDevice::new(0)).map_err(|error| RemasterError::Gpu {
                    reason: format!(
                        "numr cannot open CUDA device 0: {error}. Check the driver and free GPU \
                         memory."
                    ),
                })?;
            Ok(Opened {
                client,
                device: CudaDevice::new(0),
                library: library.map(Path::to_path_buf),
            })
        }

        #[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
        fn model_execution() -> SrExecution {
            crate::transform::remaster_execution()
        }

        fn open_decoder(
            opened: &Opened<Self>,
            backend: DecoderBackend,
            decode: &DecodeConfig,
            path: &Path,
        ) -> Result<BoxDecoder<Self>, RemasterError> {
            if backend != DecoderBackend::Nvdec {
                return open_cpu_decoder::<Self>(
                    &opened.device,
                    opened.library.as_deref(),
                    backend,
                    decode,
                    path,
                );
            }
            let decoder = NvdecDecoder::new(&opened.client, decode.codec, &decode.config)
                .map_err(|error| decode_error(path, error))?;
            Ok(Box::new(decoder))
        }

        fn open_encoder(
            opened: &Opened<Self>,
            backend: EncoderBackend,
            config: EncodeConfig,
        ) -> Result<BoxEncoder<Self>, RemasterError> {
            if backend != EncoderBackend::Nvenc {
                return open_cpu_encoder::<Self>(opened.library.as_deref(), backend, config);
            }
            let encoder = NvencEncoder::new(&opened.client, config).map_err(encoder_open_error)?;
            Ok(Box::new(encoder))
        }
    }
}
