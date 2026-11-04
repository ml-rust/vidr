//! `RemasterTransform` on the CPU backend, with tiny networks the tests
//! synthesize. No weights are stored.
//!
//! - The hand chains call `to_rgb`, the network, the cast, `resize`, and
//!   `from_rgb` with no reused buffer. `apply` must match them exactly.
//! - A zero compact network returns the nearest-upsampled input. So a flat
//!   frame keeps its codes, within 1 code.
//! - A zero RRDB network returns the bias of `conv_last`. So its planes follow
//!   the BT.709 formula, within 1 code.

#![cfg(feature = "transform")]

mod common;

use common::{FORMATS, dims, flat_frame, plan, ramp_frame, samples};
use numr::dtype::DType;
use numr::ops::{ConvLayout, ImageOps, ResizeConfig, ResizeFilter, TypeConversionOps};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use vidr::Rotation;
use vidr::decode::{DecodedFrame, PixelFormat};
use vidr::frame::{OutputPlanes, from_rgb, to_rgb};
use vidr::pipeline::FrameTransform;
use vidr_remaster::{RemasterTransform, TransformError, coded_output_size};

/// A flat frame's 8-bit codes, in gamut in BT.709.
const FLAT: (u16, u16, u16) = (100, 110, 150);
/// A side whose RGB cannot allocate on any machine.
const TOO_LARGE: usize = 1 << 28;

/// A source `(width, height)` and a target `(width, height)`.
type SizeCase = ((usize, usize), (usize, usize));

fn cpu() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    (CpuRuntime::default_client(&device), device)
}

/// The 8-bit code `code` at the depth of `format`.
fn scaled(code: u16, format: PixelFormat) -> u16 {
    code << (format.bit_depth() - 8)
}

fn assert_shape_and_format(
    planes: &OutputPlanes<CpuRuntime>,
    (width, height): (usize, usize),
    format: PixelFormat,
) {
    assert_eq!(planes.y.shape(), [height, width], "{format} luma shape");
    assert_eq!(
        planes.uv.shape(),
        [height / 2, width],
        "{format} chroma shape"
    );
    assert_eq!(
        planes.y.dtype(),
        format.sample_dtype(),
        "{format} luma dtype"
    );
    assert_eq!(
        planes.uv.dtype(),
        format.sample_dtype(),
        "{format} chroma dtype"
    );
    assert_eq!(width % 2, 0, "odd width");
    assert_eq!(height % 2, 0, "odd height");
}

fn assert_same(
    got: &OutputPlanes<CpuRuntime>,
    want: &OutputPlanes<CpuRuntime>,
    format: PixelFormat,
) {
    assert_eq!(
        samples(&got.y, format),
        samples(&want.y, format),
        "{format} luma"
    );
    assert_eq!(
        samples(&got.uv, format),
        samples(&want.uv, format),
        "{format} chroma"
    );
}

/// Every sample of the plane lies within 1 code of `want`.
fn assert_near(values: &[u16], want: u16, label: &str) {
    for value in values {
        assert!(value.abs_diff(want) <= 1, "{label}: {value} against {want}");
    }
}

/// `image` resized in F32, or passed through at the target size.
fn resized(
    client: &CpuClient,
    image: numr::tensor::Tensor<CpuRuntime>,
    (width, height): (usize, usize),
    layout: ConvLayout,
) -> numr::tensor::Tensor<CpuRuntime> {
    let shape = match layout {
        ConvLayout::Nchw => [1, 3, height, width],
        ConvLayout::Nhwc => [1, height, width, 3],
    };
    if image.shape() == shape {
        return image;
    }
    let wide = client.cast(&image, DType::F32).expect("cast");
    let config = ResizeConfig::new((height, width), ResizeFilter::Bicubic).with_antialias(true);
    client.resize(&wide, &config, layout).expect("resize")
}

mod resize_only {
    use super::*;

    fn hand_chain(
        frame: &DecodedFrame<CpuRuntime>,
        target: (usize, usize),
    ) -> OutputPlanes<CpuRuntime> {
        let (client, _) = cpu();
        let (colors, layout) = (plan(), ConvLayout::Nhwc);
        let rgb = to_rgb(&client, frame, &colors.decode, DType::F32, layout).expect("to_rgb");
        let output = resized(&client, rgb, target, layout);
        from_rgb(&client, &output, &colors.encode, frame.format(), layout).expect("from_rgb")
    }

    fn transform(target: (usize, usize)) -> RemasterTransform<CpuRuntime> {
        RemasterTransform::resize_only(dims(target.0, target.1), plan()).expect("transform")
    }

    #[test]
    fn apply_matches_the_hand_chain_at_every_size() {
        let (_, device) = cpu();
        let cases = [
            ((32, 18), (64, 36)),
            ((64, 36), (32, 18)),
            ((32, 18), (32, 18)),
            ((13, 9), (26, 18)),
        ];
        for (source, target) in cases {
            for format in FORMATS {
                let frame = ramp_frame::<CpuRuntime>(source, format, &device);
                let want = hand_chain(&frame, target);
                let got = transform(target).apply(frame).expect("apply");
                assert_shape_and_format(&got, target, format);
                assert_same(&got, &want, format);
            }
        }
    }

    #[test]
    fn a_flat_frame_keeps_its_codes() {
        let (_, device) = cpu();
        for format in FORMATS {
            let frame = flat_frame::<CpuRuntime>((32, 18), format, FLAT, &device);
            let planes = transform((64, 36)).apply(frame).expect("apply");
            assert_near(&samples(&planes.y, format), scaled(FLAT.0, format), "luma");
            let chroma = samples(&planes.uv, format);
            for pair in chroma.as_chunks::<2>().0 {
                assert_near(&pair[..1], scaled(FLAT.1, format), "U");
                assert_near(&pair[1..], scaled(FLAT.2, format), "V");
            }
        }
    }

    #[test]
    fn repeated_frames_give_the_same_planes() {
        let (_, device) = cpu();
        let format = FORMATS[0];
        let mut transform = transform((64, 36));
        let frame = || ramp_frame::<CpuRuntime>((32, 18), format, &device);
        let first = transform.apply(frame()).expect("first");
        let second = transform.apply(frame()).expect("second");
        assert_same(&second, &first, format);
        assert_ne!(first.y.ptr(), second.y.ptr(), "output planes must be new");
    }

    #[test]
    fn an_odd_target_is_refused() {
        for target in [(33, 18), (32, 19)] {
            let error =
                RemasterTransform::<CpuRuntime>::resize_only(dims(target.0, target.1), plan())
                    .err()
                    .expect("refused");
            assert!(matches!(error, TransformError::OddTarget { .. }), "{error}");
            assert!(error.to_string().contains("even"), "{error}");
        }
    }

    #[test]
    fn a_rotated_source_takes_the_transposed_target() {
        let (_, device) = cpu();
        let target = coded_output_size(dims(18, 32), Rotation::Clockwise90);
        let format = FORMATS[0];
        let frame = ramp_frame::<CpuRuntime>((32, 18), format, &device);
        let planes = RemasterTransform::resize_only(target, plan())
            .expect("transform")
            .apply(frame)
            .expect("apply");
        assert_shape_and_format(&planes, (32, 18), format);
    }

    #[test]
    fn out_of_memory_names_the_frame_and_the_next_action() {
        let (_, device) = cpu();
        let frame = ramp_frame::<CpuRuntime>((32, 18), FORMATS[0], &device);
        let message = transform((TOO_LARGE, TOO_LARGE))
            .apply(frame)
            .expect_err("the allocation fails")
            .to_string();
        assert!(message.contains("32x18"), "{message}");
        assert!(message.contains("resize-only"), "{message}");
        assert!(message.contains("out of memory"), "{message}");
        assert!(message.contains("smaller source"), "{message}");
    }

    #[test]
    fn the_transform_crosses_threads() {
        fn assert_send<T: Send>() {}
        assert_send::<RemasterTransform<CpuRuntime>>();
    }
}

#[cfg(any(feature = "model-srvgg", feature = "model-rrdb"))]
mod networks {
    #[cfg(feature = "model-rrdb")]
    use boostr::model::super_resolution::{RrdbConfig, RrdbNet};
    use boostr::model::super_resolution::{SrExecution, SrModel};
    #[cfg(feature = "model-srvgg")]
    use boostr::model::super_resolution::{SrVggCompact, SrVggCompactConfig};
    use boostr::nn::{VarBuilder, VarMap};
    use numr::tensor::Tensor;
    use vidr_remaster::{ModelId, remaster_execution};

    use super::*;

    /// A conv's weight and bias, with values from `fill`.
    fn insert_conv(
        map: &mut VarMap<CpuRuntime>,
        name: &str,
        (inputs, outputs): (usize, usize),
        fill: &mut dyn FnMut() -> f32,
        device: &CpuDevice,
    ) {
        let mut insert = |key: String, shape: &[usize]| {
            let values: Vec<f32> = (0..shape.iter().product()).map(|_| fill()).collect();
            let tensor = Tensor::<CpuRuntime>::from_slice(&values, shape, device).expect("weight");
            map.insert(key, tensor);
        };
        insert(format!("{name}.weight"), &[outputs, inputs, 3, 3]);
        insert(format!("{name}.bias"), &[outputs]);
    }

    /// A deterministic small value per weight.
    fn pattern() -> impl FnMut() -> f32 {
        let mut index = 0_usize;
        move || {
            index += 1;
            ((index % 11) as f32 - 5.0) * 0.003
        }
    }

    #[cfg(feature = "model-srvgg")]
    fn compact(exec: SrExecution, fill: &mut dyn FnMut() -> f32) -> SrModel<CpuRuntime> {
        let (client, device) = cpu();
        let config = SrVggCompactConfig {
            num_in_ch: 3,
            num_out_ch: 3,
            num_feat: 4,
            num_conv: 1,
            upscale: 4,
        };
        let mut map = VarMap::<CpuRuntime>::new();
        for (i, channels) in [(3, 4), (4, 4), (4, 48)].into_iter().enumerate() {
            insert_conv(
                &mut map,
                &format!("body.{}", 2 * i),
                channels,
                fill,
                &device,
            );
            if i < 2 {
                let slope =
                    Tensor::<CpuRuntime>::from_slice(&[0.0_f32; 4], &[4], &device).expect("slope");
                map.insert(format!("body.{}.weight", 2 * i + 1), slope);
            }
        }
        let mut vb = VarBuilder::new(&mut map, &device);
        let net = SrVggCompact::from_varbuilder(&client, &mut vb, &config, exec).expect("compact");
        SrModel::Compact(Box::new(net))
    }

    /// A one-block RRDB network. `conv_last` takes `last_bias`.
    #[cfg(feature = "model-rrdb")]
    fn rrdb(
        exec: SrExecution,
        fill: &mut dyn FnMut() -> f32,
        last_bias: Option<[f32; 3]>,
    ) -> SrModel<CpuRuntime> {
        let (client, device) = cpu();
        let config = RrdbConfig {
            num_in_ch: 3,
            num_out_ch: 3,
            num_feat: 4,
            num_block: 1,
            num_grow_ch: 2,
            scale: 2,
        };
        let feat = config.num_feat;
        let mut map = VarMap::<CpuRuntime>::new();
        let first = (config.first_conv_in_channels(), feat);
        insert_conv(&mut map, "conv_first", first, fill, &device);
        for dense in 1..=3 {
            for conv in 1..=5 {
                let outputs = if conv < 5 { config.num_grow_ch } else { feat };
                insert_conv(
                    &mut map,
                    &format!("body.0.rdb{dense}.conv{conv}"),
                    (config.dense_conv_in_channels(conv), outputs),
                    fill,
                    &device,
                );
            }
        }
        for name in ["conv_body", "conv_up1", "conv_up2", "conv_hr"] {
            insert_conv(&mut map, name, (feat, feat), fill, &device);
        }
        insert_conv(&mut map, "conv_last", (feat, 3), fill, &device);
        if let Some(bias) = last_bias {
            let bias = Tensor::<CpuRuntime>::from_slice(&bias, &[3], &device).expect("bias");
            map.insert("conv_last.bias".to_string(), bias);
        }
        let mut vb = VarBuilder::new(&mut map, &device);
        let net = RrdbNet::from_varbuilder(&client, &mut vb, &config, exec).expect("rrdb");
        SrModel::Rrdb(Box::new(net))
    }

    fn hand_chain(
        model: &SrModel<CpuRuntime>,
        frame: &DecodedFrame<CpuRuntime>,
        target: (usize, usize),
    ) -> OutputPlanes<CpuRuntime> {
        let (client, _) = cpu();
        let colors = plan();
        let exec = *model.execution();
        let rgb = to_rgb(&client, frame, &colors.decode, exec.dtype, exec.layout).expect("to_rgb");
        let raw = model.forward_native(&client, &rgb).expect("forward");
        let output = resized(&client, raw, target, exec.layout);
        from_rgb(
            &client,
            &output,
            &colors.encode,
            frame.format(),
            exec.layout,
        )
        .expect("from_rgb")
    }

    /// `apply` equals the hand chain, on F16 NHWC and on F32 NCHW.
    fn check_chain(
        id: ModelId,
        build: impl Fn(SrExecution) -> SrModel<CpuRuntime>,
        cases: &[SizeCase],
    ) {
        let (_, device) = cpu();
        let executions = [
            remaster_execution(),
            SrExecution::new(DType::F32, ConvLayout::Nchw),
        ];
        for exec in executions {
            for &(source, target) in cases {
                for format in FORMATS {
                    let frame = ramp_frame::<CpuRuntime>(source, format, &device);
                    let want = hand_chain(&build(exec), &frame, target);
                    let mut transform = RemasterTransform::from_model(
                        id,
                        build(exec),
                        dims(target.0, target.1),
                        plan(),
                    )
                    .expect("transform");
                    assert_eq!(transform.model(), Some(id));
                    let got = transform.apply(frame).expect("apply");
                    assert_shape_and_format(&got, target, format);
                    assert_same(&got, &want, format);
                }
            }
        }
    }

    /// The BT.709 limited-range codes of one RGB colour.
    #[cfg(feature = "model-rrdb")]
    fn bt709_codes([r, g, b]: [f64; 3], format: PixelFormat) -> (u16, u16, u16) {
        let scale = f64::from(1_u32 << (format.bit_depth() - 8));
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let cb = (b - y) / 1.8556;
        let cr = (r - y) / 1.5748;
        let code = |value: f64| (value * scale).round() as u16;
        (
            code(16.0 + 219.0 * y),
            code(128.0 + 224.0 * cb),
            code(128.0 + 224.0 * cr),
        )
    }

    #[cfg(feature = "model-srvgg")]
    mod fast {
        use super::*;

        fn zero(exec: SrExecution) -> SrModel<CpuRuntime> {
            compact(exec, &mut || 0.0)
        }

        #[test]
        fn apply_matches_the_hand_chain() {
            let build = |exec| compact(exec, &mut pattern());
            let cases = [
                ((16, 10), (32, 20)),
                ((16, 10), (64, 40)),
                ((13, 9), (26, 18)),
                ((16, 10), (20, 12)),
            ];
            check_chain(ModelId::Fast, build, &cases);
        }

        #[test]
        fn a_flat_frame_keeps_its_codes_through_a_zero_network() {
            let (_, device) = cpu();
            for format in FORMATS {
                let frame = flat_frame::<CpuRuntime>((16, 10), format, FLAT, &device);
                let model = zero(remaster_execution());
                let planes =
                    RemasterTransform::from_model(ModelId::Fast, model, dims(32, 20), plan())
                        .expect("transform")
                        .apply(frame)
                        .expect("apply");
                assert_near(&samples(&planes.y, format), scaled(FLAT.0, format), "luma");
                for pair in samples(&planes.uv, format).as_chunks::<2>().0 {
                    assert_near(&pair[..1], scaled(FLAT.1, format), "U");
                    assert_near(&pair[1..], scaled(FLAT.2, format), "V");
                }
            }
        }

        #[test]
        fn out_of_memory_names_the_model_and_a_smaller_source() {
            let (_, device) = cpu();
            let frame = ramp_frame::<CpuRuntime>((16, 10), FORMATS[0], &device);
            let target = dims(TOO_LARGE, TOO_LARGE);
            let message = RemasterTransform::from_model(
                ModelId::Fast,
                zero(remaster_execution()),
                target,
                plan(),
            )
            .expect("transform")
            .apply(frame)
            .expect_err("the allocation fails")
            .to_string();
            assert!(message.contains("Model fast"), "{message}");
            assert!(message.contains("16x10"), "{message}");
            assert!(message.contains("smaller source"), "{message}");
            assert!(!message.contains("--model fast"), "{message}");
        }
    }

    #[cfg(feature = "model-rrdb")]
    mod rrdb_models {
        use super::*;

        const BIAS: [f32; 3] = [0.25, 0.5, 0.75];

        #[test]
        fn apply_matches_the_hand_chain() {
            let build = |exec| rrdb(exec, &mut pattern(), None);
            let cases = [
                ((16, 10), (32, 20)),
                ((13, 9), (26, 18)),
                ((16, 10), (20, 12)),
            ];
            check_chain(ModelId::RealEsrgan, build, &cases);
        }

        #[test]
        fn a_zero_network_returns_the_bias_at_an_equal_size() {
            let (_, device) = cpu();
            for format in FORMATS {
                for (source, target) in [((16, 10), (32, 20)), ((13, 9), (26, 18))] {
                    let frame = ramp_frame::<CpuRuntime>(source, format, &device);
                    let model = rrdb(remaster_execution(), &mut || 0.0, Some(BIAS));
                    let planes = RemasterTransform::from_model(
                        ModelId::PureVision,
                        model,
                        dims(target.0, target.1),
                        plan(),
                    )
                    .expect("transform")
                    .apply(frame)
                    .expect("apply");
                    assert_shape_and_format(&planes, target, format);
                    let (y, u, v) = bt709_codes(BIAS.map(f64::from), format);
                    assert_near(&samples(&planes.y, format), y, "luma");
                    for pair in samples(&planes.uv, format).as_chunks::<2>().0 {
                        assert_near(&pair[..1], u, "U");
                        assert_near(&pair[1..], v, "V");
                    }
                }
            }
        }

        #[test]
        fn out_of_memory_on_a_full_model_names_the_fast_model() {
            let (_, device) = cpu();
            let frame = ramp_frame::<CpuRuntime>((16, 10), FORMATS[0], &device);
            let model = rrdb(remaster_execution(), &mut || 0.0, None);
            let target = dims(TOO_LARGE, TOO_LARGE);
            let message = RemasterTransform::from_model(ModelId::PureVision, model, target, plan())
                .expect("transform")
                .apply(frame)
                .expect_err("the allocation fails")
                .to_string();
            assert!(message.contains("Model purevision"), "{message}");
            assert!(message.contains("16x10"), "{message}");
            assert!(
                message.contains("--model fast or a smaller source"),
                "{message}"
            );
        }

        #[cfg(feature = "model-srvgg")]
        #[test]
        fn a_checkpoint_of_another_network_is_refused() {
            let model = rrdb(remaster_execution(), &mut || 0.0, None);
            let error = RemasterTransform::from_model(ModelId::Fast, model, dims(32, 20), plan())
                .err()
                .expect("refused");
            assert!(
                matches!(error, TransformError::WrongNetwork { .. }),
                "{error}"
            );
            let message = error.to_string();
            assert!(message.contains("SRVGGNetCompact"), "{message}");
            assert!(message.contains("RRDBNet"), "{message}");
        }
    }
}
