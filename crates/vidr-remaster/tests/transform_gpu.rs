//! `RemasterTransform` on CUDA with the real checkpoints.
//!
//! Run with `cargo nextest run -p vidr-remaster --features cuda,model-srvgg,model-rrdb`.
//! Each model is `$BOOSTR_MODELS_DIR/<catalog directory>`. A missing model, or no
//! GPU, skips with a printed notice.
//!
//! The reference runs boostr's parity-checked calls at F32 NCHW with F32
//! accumulation. `forward_resized` is the reference for the compact network, and
//! `forward` plus `resize` for RRDB. The transform runs F16 NHWC with half
//! accumulation allowed, so the two differ by rounding only. `PSNR_FLOOR_DB`
//! bounds that difference on the luma and chroma planes.

#![cfg(all(feature = "cuda", feature = "model-srvgg", feature = "model-rrdb"))]

mod common;

use std::env;
use std::path::PathBuf;

use boostr::model::super_resolution::{SrExecution, SrModel, load_sr_dir};
use common::{FORMATS, dims, peak, plan, psnr, ramp_frame, samples};
use numr::dtype::DType;
use numr::ops::{ConvLayout, ImageOps, ResizeConfig, ResizeFilter};
use numr::runtime::Runtime;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime, is_cuda_available};
use vidr::decode::DecodedFrame;
use vidr::frame::{OutputPlanes, from_rgb, to_rgb};
use vidr::pipeline::FrameTransform;
use vidr_remaster::{ModelId, RemasterTransform, resolve_model_dir};

const SOURCE: (usize, usize) = (96, 64);
/// Twice the source, then a size only the resize reaches.
const TARGETS: [(usize, usize); 2] = [(192, 128), (144, 96)];
/// Lowest plane PSNR against the F32 reference, at each format's peak.
const PSNR_FLOOR_DB: f64 = 50.0;

fn gpu() -> Option<(CudaClient, CudaDevice)> {
    if !is_cuda_available() {
        eprintln!("skipping: no CUDA device 0. Run this test on a machine with a GPU.");
        return None;
    }
    let device = CudaDevice::new(0);
    Some((CudaRuntime::default_client(&device), device))
}

fn model_dir(model: ModelId) -> Option<PathBuf> {
    let Some(root) = env::var_os("BOOSTR_MODELS_DIR") else {
        eprintln!("skipping {model}: BOOSTR_MODELS_DIR is not set.");
        return None;
    };
    match resolve_model_dir(PathBuf::from(root).as_path(), model) {
        Ok(dir) => Some(dir),
        Err(error) => {
            eprintln!("skipping {model}: {error}");
            None
        }
    }
}

fn reference(
    client: &CudaClient,
    model: &SrModel<CudaRuntime>,
    frame: &DecodedFrame<CudaRuntime>,
    (width, height): (usize, usize),
) -> OutputPlanes<CudaRuntime> {
    let colors = plan();
    let layout = ConvLayout::Nchw;
    let rgb = to_rgb(client, frame, &colors.decode, DType::F32, layout).expect("to_rgb");
    let output = match model {
        SrModel::Compact(net) => net
            .forward_resized(client, &rgb, (height, width))
            .expect("forward_resized"),
        SrModel::Rrdb(net) => {
            let raw = net.forward(client, &rgb).expect("forward");
            if raw.shape() == [1, 3, height, width] {
                raw
            } else {
                let config =
                    ResizeConfig::new((height, width), ResizeFilter::Bicubic).with_antialias(true);
                client.resize(&raw, &config, layout).expect("resize")
            }
        }
    };
    from_rgb(client, &output, &colors.encode, frame.format(), layout).expect("from_rgb")
}

fn check(id: ModelId) {
    let Some((client, device)) = gpu() else {
        return;
    };
    let Some(dir) = model_dir(id) else {
        return;
    };
    let exec = SrExecution::new(DType::F32, ConvLayout::Nchw);
    let reference_model = load_sr_dir::<CudaRuntime>(&dir, &device, exec).expect("reference load");
    for target in TARGETS {
        for format in FORMATS {
            let label = format!("{id} {format} to {}x{}", target.0, target.1);
            let frame = ramp_frame::<CudaRuntime>(SOURCE, format, &device);
            let want = reference(&client, &reference_model, &frame, target);
            let mut transform = RemasterTransform::<CudaRuntime>::load(
                id,
                &dir,
                &device,
                dims(target.0, target.1),
                plan(),
            )
            .unwrap_or_else(|error| panic!("{label}: {error}"));
            let got = transform
                .apply(frame)
                .unwrap_or_else(|error| panic!("{label}: {error}"));
            assert_eq!(got.y.shape(), [target.1, target.0], "{label}: luma shape");
            assert_eq!(
                got.uv.shape(),
                [target.1 / 2, target.0],
                "{label}: chroma shape"
            );
            assert_eq!(got.y.dtype(), format.sample_dtype(), "{label}: dtype");
            let luma = psnr(
                &samples(&got.y, format),
                &samples(&want.y, format),
                peak(format),
            );
            let chroma = psnr(
                &samples(&got.uv, format),
                &samples(&want.uv, format),
                peak(format),
            );
            eprintln!("{label}: luma {luma:.2} dB, chroma {chroma:.2} dB");
            assert!(luma >= PSNR_FLOOR_DB, "{label}: luma {luma:.2} dB");
            assert!(chroma >= PSNR_FLOOR_DB, "{label}: chroma {chroma:.2} dB");
        }
    }
}

#[test]
fn fast_matches_the_reference() {
    check(ModelId::Fast);
}

#[test]
fn realesrgan_matches_the_reference() {
    check(ModelId::RealEsrgan);
}

#[test]
fn purevision_matches_the_reference() {
    check(ModelId::PureVision);
}
