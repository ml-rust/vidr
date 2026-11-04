//! Configs and frames the encoder refuses, by name.

use numr::dtype::DType;
use numr::runtime::RuntimeClient;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;
use vidr::EncodeError;
use vidr::encode::nvenc::NvencEncoder;
use vidr::encode::{EncodeCodec, EncodeConfig, EncodeFrame, PixelFormat};

use super::support::{self, DURATION};

fn refusal(client: &CudaClient, config: EncodeConfig) -> Option<EncodeError> {
    NvencEncoder::new(client, config).err()
}

/// A GPU can gain what this one lacks, so a success prints a notice.
#[test]
fn av1_is_refused_by_name_on_a_gpu_without_it() {
    let Some(client) = support::gpu() else {
        return;
    };
    let config = support::config(EncodeCodec::Av1, PixelFormat::Nv12);
    match refusal(&client, config) {
        Some(error @ EncodeError::CodecUnsupported { .. }) => {
            let message = error.to_string();
            assert!(message.contains("no AV1 encoder"), "{message}");
            assert!(message.contains("H.264"), "{message}");
        }
        Some(other) => panic!("expected CodecUnsupported, got {other:?}: {other}"),
        None => eprintln!("notice: this GPU encodes AV1, so the refusal does not apply."),
    }
}

#[test]
fn h264_10_bit_is_refused_by_name_on_a_gpu_without_it() {
    let Some(client) = support::gpu() else {
        return;
    };
    let config = support::config(EncodeCodec::H264, PixelFormat::P010);
    match refusal(&client, config) {
        Some(EncodeError::BitDepthUnsupported {
            codec: EncodeCodec::H264,
            format: PixelFormat::P010,
        }) => {}
        Some(other) => panic!("expected BitDepthUnsupported, got {other:?}: {other}"),
        None => eprintln!("notice: this GPU encodes 10-bit H.264, so the refusal does not apply."),
    }
}

#[test]
fn odd_sizes_are_refused_for_h264_and_hevc() {
    let Some(client) = support::gpu() else {
        return;
    };
    for codec in [EncodeCodec::H264, EncodeCodec::Hevc] {
        let mut config = support::config(codec, PixelFormat::Nv12);
        config.width = 641;
        assert_eq!(
            refusal(&client, config),
            Some(EncodeError::OddSize {
                codec,
                width: 641,
                height: 360,
            })
        );
    }
}

#[test]
fn too_many_b_frames_are_refused() {
    let Some(client) = support::gpu() else {
        return;
    };
    let mut config = support::config(EncodeCodec::H264, PixelFormat::Nv12);
    config.b_frames = 64;
    assert!(matches!(
        refusal(&client, config),
        Some(EncodeError::BFramesUnsupported { requested: 64, .. })
    ));
}

#[test]
fn a_size_below_the_gpu_minimum_is_refused() {
    let Some(client) = support::gpu() else {
        return;
    };
    let mut config = support::config(EncodeCodec::H264, PixelFormat::Nv12);
    config.width = 16;
    config.height = 16;
    assert!(matches!(
        refusal(&client, config),
        Some(EncodeError::SizeOutOfRange { .. })
    ));
}

fn plane(client: &CudaClient, shape: [usize; 2], dtype: DType) -> Tensor<CudaRuntime> {
    Tensor::empty(&shape, dtype, client.device())
        .unwrap_or_else(|error| panic!("plane allocation: {error}"))
}

/// A refused frame costs nothing: the encoder still takes a good one after it.
#[test]
fn bad_frames_and_calls_are_refused_and_the_encoder_goes_on() {
    let Some(client) = support::gpu() else {
        return;
    };
    let format = PixelFormat::Nv12;
    let config = support::config(EncodeCodec::H264, format);
    let mut encoder =
        NvencEncoder::new(&client, config).unwrap_or_else(|error| panic!("open: {error}"));

    // P010 samples into an NV12 encoder.
    let y16 = plane(&client, [360, 640], DType::U16);
    let uv16 = plane(&client, [180, 640], DType::U16);
    let error = encoder.send(&EncodeFrame::new(&y16, &uv16), 0, DURATION);
    assert!(matches!(
        error,
        Err(EncodeError::SurfaceMismatch { plane: "luma", .. })
    ));

    // A chroma plane of the wrong shape.
    let good = support::frame_planes(&client, format, 0, 0);
    let small = plane(&client, [90, 320], DType::U8);
    let error = encoder.send(&EncodeFrame::new(&good.y, &small), 0, DURATION);
    assert!(matches!(
        error,
        Err(EncodeError::SurfaceMismatch {
            plane: "chroma",
            ..
        })
    ));

    let frame = EncodeFrame::new(&good.y, &good.uv);
    assert_eq!(
        encoder.send(&frame, 0, 0),
        Err(EncodeError::NonPositiveDuration {
            pts: 0,
            duration: 0,
        })
    );
    assert_eq!(encoder.send(&frame, 100, DURATION), Ok(()));
    assert_eq!(
        encoder.send(&frame, 100, DURATION),
        Err(EncodeError::PtsNotIncreasing {
            pts: 100,
            previous: 100,
        })
    );
    assert_eq!(encoder.flush(), Ok(()));
    assert_eq!(
        encoder.send(&frame, 200, DURATION),
        Err(EncodeError::AfterFlush { pts: 200 })
    );
    assert_eq!(encoder.receive().map(|packet| packet.pts), Some(100));
    assert_eq!(encoder.receive(), None);
}
