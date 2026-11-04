//! Generated NVDEC fixtures and the GPU they run on.
//!
//! `tests/fixtures/gen_nvdec.py` writes the fixtures. A test whose fixture,
//! NVIDIA library, or CUDA device is missing prints a notice and passes
//! without checks.

use numr::runtime::cuda::{CudaClient, CudaDevice, is_cuda_available};
use vidr::decode::nvdec::nvdec_available;

pub(crate) use crate::decode_fixtures::{Stream, load};

/// numr's client on CUDA device 0, or a notice and `None` when the NVIDIA
/// library or a CUDA device is missing.
pub(crate) fn gpu() -> Option<CudaClient> {
    if let Err(error) = nvdec_available() {
        eprintln!("skipping: {error}");
        return None;
    }
    if !is_cuda_available() {
        eprintln!("skipping: no CUDA device 0. Run the NVDEC tests on a machine with a GPU.");
        return None;
    }
    match CudaClient::new(CudaDevice::new(0)) {
        Ok(client) => Some(client),
        Err(error) => {
            eprintln!("skipping: numr cannot open CUDA device 0: {error}");
            None
        }
    }
}
