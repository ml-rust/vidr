//! Opens either OpenH264 path, and builds frames on numr's CPU backend.

use numr::dtype::DType;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use vidr::decode::openh264::Openh264Decoder;
use vidr::encode::EncodeConfig;
use vidr::encode::openh264::Openh264Encoder;
use vidr::{DecodeError, EncodeError};

/// The path of Cisco's binary, or a notice and `None`.
#[cfg(feature = "openh264-cisco")]
pub(crate) fn cisco_binary() -> Option<std::path::PathBuf> {
    let path = std::env::var_os("VIDR_OPENH264_CISCO").map(std::path::PathBuf::from);
    if path.is_none() {
        eprintln!("skipping: VIDR_OPENH264_CISCO is unset. Set it to Cisco's binary.");
    }
    path
}

/// A decoder on the source build.
#[cfg(feature = "openh264")]
pub(crate) fn decoder(config: &[u8]) -> Option<Result<Openh264Decoder<CpuRuntime>, DecodeError>> {
    Some(Openh264Decoder::new(&CpuDevice::new(), config))
}

/// A decoder on Cisco's binary, in a build with no source path.
#[cfg(not(feature = "openh264"))]
pub(crate) fn decoder(config: &[u8]) -> Option<Result<Openh264Decoder<CpuRuntime>, DecodeError>> {
    let path = cisco_binary()?;
    Some(Openh264Decoder::with_cisco_binary(
        &CpuDevice::new(),
        config,
        path,
    ))
}

/// An encoder on the source build.
#[cfg(feature = "openh264")]
pub(crate) fn encoder(
    config: EncodeConfig,
) -> Option<Result<Openh264Encoder<CpuRuntime>, EncodeError>> {
    Some(Openh264Encoder::new(config))
}

/// An encoder on Cisco's binary, in a build with no source path.
#[cfg(not(feature = "openh264"))]
pub(crate) fn encoder(
    config: EncodeConfig,
) -> Option<Result<Openh264Encoder<CpuRuntime>, EncodeError>> {
    let path = cisco_binary()?;
    Some(Openh264Encoder::with_cisco_binary(config, path))
}

/// A U8 or U16 plane from its bytes.
pub(crate) fn plane(bytes: &[u8], shape: [usize; 2], dtype: DType) -> Tensor<CpuRuntime> {
    Tensor::from_bytes(bytes, &shape, dtype, &CpuDevice::new())
        .unwrap_or_else(|error| panic!("plane of shape {shape:?}: {error}"))
}

/// PSNR of 8-bit samples, peak 255.
pub(crate) fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "sample counts differ");
    let squares: f64 = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    let mean = squares / a.len() as f64;
    if mean == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mean).log10()
    }
}
