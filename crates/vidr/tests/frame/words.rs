//! Word-form frames on the CPU backend, which takes both plane forms. These run
//! with no GPU, and prove the word path against the sample path.

use numr::dtype::DType;
use numr::ops::{ChromaSiting, ColorMatrix, ColorRange, ConvLayout, YuvPlanes, YuvSpec};
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;
use vidr::decode::{DecodedFrame, PixelFormat, PlaneForm};
use vidr::encode::EncodeFrame;
use vidr::frame::{OutputPlanes, from_rgb, to_rgb, to_rgb_reusing};
use vidr::{Dimensions, FrameError};

use super::support::{
    FORMATS, LAYOUTS, ODD_SIZES, Samples, cpu, pattern, planes_frame, rgb_tensor, specs,
};

const EVEN_SIZES: [(usize, usize); 3] = [(32, 18), (34, 20), (6, 4)];

/// F32 RGB of a word frame has the bits of the sample frame at every size. An
/// odd size runs `pad_packed_luma`, and sentinels past each row prove that no
/// byte past the width reaches the output.
#[test]
fn word_frames_equal_sample_frames_bit_for_bit() {
    let (client, device) = cpu();
    for size in EVEN_SIZES.into_iter().chain(ODD_SIZES) {
        for format in FORMATS {
            let samples = Samples::smooth(size, format);
            let words = samples.words_frame::<CpuRuntime>(&device);
            let plain = samples.frame::<CpuRuntime>(&device);
            assert_eq!(words.plane_form(), PlaneForm::Words);
            assert_eq!(plain.plane_form(), PlaneForm::Samples);
            for spec in specs() {
                for layout in LAYOUTS {
                    let label = format!("{size:?} {format} {spec:?} {layout:?}");
                    let got = to_rgb(&client, &words, &spec, DType::F32, layout)
                        .unwrap_or_else(|error| panic!("{label}: {error}"));
                    let want = to_rgb(&client, &plain, &spec, DType::F32, layout)
                        .unwrap_or_else(|error| panic!("{label}: {error}"));
                    assert_eq!(got.shape(), want.shape(), "{label}");
                    assert_eq!(
                        got.to_bytes().unwrap_or_else(|error| panic!("{error}")),
                        want.to_bytes().unwrap_or_else(|error| panic!("{error}")),
                        "{label}"
                    );
                }
            }
        }
    }
}

/// `to_rgb_reusing` writes the word path's bits into its buffer at an even size.
#[test]
fn reusing_a_buffer_gives_the_same_bits_for_a_word_frame() {
    let (client, device) = cpu();
    let spec = specs()[2];
    for format in FORMATS {
        let samples = Samples::smooth((32, 18), format);
        let words = samples.words_frame::<CpuRuntime>(&device);
        let want = to_rgb(&client, &words, &spec, DType::F32, ConvLayout::Nchw)
            .unwrap_or_else(|error| panic!("{error}"));
        let mut buffer = None;
        let got = to_rgb_reusing(
            &client,
            &words,
            &spec,
            DType::F32,
            ConvLayout::Nchw,
            &mut buffer,
        )
        .unwrap_or_else(|error| panic!("{error}"));
        assert!(buffer.is_some());
        assert_eq!(
            got.to_bytes().unwrap_or_else(|error| panic!("{error}")),
            want.to_bytes().unwrap_or_else(|error| panic!("{error}"))
        );
    }
}

/// `planes` gives numr's packed variant for words, and the sample variant for
/// samples.
#[test]
fn planes_follow_the_form() {
    let (_, device) = cpu();
    let samples = Samples::smooth((6, 4), PixelFormat::Nv12);
    assert!(matches!(
        samples.words_frame::<CpuRuntime>(&device).planes(),
        YuvPlanes::Nv12Packed { .. }
    ));
    assert!(matches!(
        samples.frame::<CpuRuntime>(&device).planes(),
        YuvPlanes::Nv12 { .. }
    ));
    let samples = Samples::smooth((6, 4), PixelFormat::P010);
    assert!(matches!(
        samples.words_frame::<CpuRuntime>(&device).planes(),
        YuvPlanes::P010Packed { .. }
    ));
    assert!(matches!(
        samples.frame::<CpuRuntime>(&device).planes(),
        YuvPlanes::P010 { .. }
    ));
}

/// A word frame needs U32 words in the documented shapes. Sample planes under
/// the word form, and word planes under the sample form, are named refusals.
#[test]
fn a_plane_of_the_wrong_form_is_refused_by_name() {
    let (client, device) = cpu();
    let spec = specs()[0];
    let samples = Samples::smooth((6, 4), PixelFormat::Nv12);
    let (y, uv) = samples.frame::<CpuRuntime>(&device).into_planes();
    let plain = planes_frame(y.clone(), uv.clone(), PixelFormat::Nv12, 6, 4);
    let mislabelled = DecodedFrame::from_packed_planes(
        y,
        uv,
        PixelFormat::Nv12,
        0,
        plain.display(),
        *plain.color(),
    );
    let error = to_rgb(&client, &mislabelled, &spec, DType::F32, ConvLayout::Nhwc)
        .expect_err("sample planes pass as words");
    assert_eq!(
        error,
        FrameError::PlaneMismatch {
            plane: "luma",
            expected: "[4, 2] in U32".to_owned(),
            found: "[4, 6] in U8".to_owned(),
        }
    );

    let (y, uv) = samples.words_frame::<CpuRuntime>(&device).into_planes();
    let mislabelled = planes_frame(y, uv, PixelFormat::Nv12, 6, 4);
    let error = to_rgb(&client, &mislabelled, &spec, DType::F32, ConvLayout::Nhwc)
        .expect_err("word planes pass as samples");
    assert_eq!(
        error,
        FrameError::PlaneMismatch {
            plane: "luma",
            expected: "[4, 6] in U8".to_owned(),
            found: "[4, 2] in U32".to_owned(),
        }
    );
}

/// The CPU backend keeps sample planes from `from_rgb`.
#[test]
fn from_rgb_writes_samples_on_a_sample_backend() {
    let (client, device) = cpu();
    let values = pattern(6, 4, 0);
    let rgb = rgb_tensor::<CpuRuntime>(&values, 6, 4, ConvLayout::Nhwc, &device);
    let spec = YuvSpec::new(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left);
    let planes = from_rgb(&client, &rgb, &spec, PixelFormat::Nv12, ConvLayout::Nhwc)
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(planes.plane_form(), PlaneForm::Samples);
    assert_eq!(planes.word_layout(), None);
    assert_eq!(planes.y.shape(), [4, 6]);
    assert_eq!(planes.uv.shape(), [2, 6]);
    assert_eq!(EncodeFrame::from(&planes).plane_form(), PlaneForm::Samples);
}

/// The form and layout survive each step from a decoded frame or output planes
/// to the frame an encoder takes.
#[test]
fn encode_frames_keep_the_form_and_layout() {
    let (_, device) = cpu();
    let samples = Samples::smooth((7, 5), PixelFormat::P010);
    let words = samples.words_frame::<CpuRuntime>(&device);
    let display = Dimensions::new(7, 5).unwrap_or_else(|error| panic!("{error}"));
    let expected = Some((PixelFormat::P010, display));

    let frame = EncodeFrame::from(&words);
    assert_eq!(frame.plane_form(), PlaneForm::Words);
    assert_eq!(frame.word_layout(), expected);

    let cloned = words.clone();
    assert_eq!(cloned.plane_form(), PlaneForm::Words);

    let (y, uv): (Tensor<CpuRuntime>, Tensor<CpuRuntime>) = words.into_planes();
    let planes = OutputPlanes::from_packed_planes(y, uv, PixelFormat::P010, display);
    assert_eq!(planes.plane_form(), PlaneForm::Words);
    let copy = planes.clone();
    let frame = EncodeFrame::from(&copy);
    assert_eq!(frame.plane_form(), PlaneForm::Words);
    assert_eq!(frame.word_layout(), expected);
}
