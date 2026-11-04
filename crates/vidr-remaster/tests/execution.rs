//! `execution_for` over every request, build, and machine, against the rules written
//! out a second time here. Named rows then pin each refusal's text.

use vidr_remaster::{
    Codec, Compiled, DecoderBackend, DecoderChoice, EncoderBackend, EncoderChoice, Execution,
    ExecutionError, ExecutionRequest, Machine, Openh264Build, Probe, RuntimeChoice, RuntimeKind,
    SourceCodec, SourceFacts, execution_for,
};

const OUTPUT_CODECS: [Codec; 3] = [Codec::Hevc, Codec::H264, Codec::Av1];
/// `profile_idc` of H.264 High.
const HIGH: u8 = 100;
/// `profile_idc` of H.264 Baseline.
const BASELINE: u8 = 66;

fn up() -> Probe {
    Ok(())
}

fn down(what: &str) -> Probe {
    Err(format!("{what} is absent"))
}

fn source(codec: SourceCodec) -> SourceFacts {
    SourceFacts {
        codec,
        h264_profile: None,
    }
}

fn h264(profile: Option<u8>) -> SourceFacts {
    SourceFacts {
        codec: SourceCodec::H264,
        h264_profile: profile,
    }
}

fn sources() -> [SourceFacts; 6] {
    [
        source(SourceCodec::Av1),
        source(SourceCodec::Vp9),
        source(SourceCodec::Hevc),
        h264(None),
        h264(Some(BASELINE)),
        h264(Some(HIGH)),
    ]
}

/// Every combination of the build's features, with each OpenH264 form and whether
/// `--openh264-library` names a binary.
fn builds() -> Vec<(Compiled, bool)> {
    let forms = [
        (false, false, false),
        (false, false, true),
        (true, false, false),
        (true, false, true),
        (false, true, false),
        (false, true, true),
        (true, true, false),
        (true, true, true),
    ];
    let mut all = Vec::new();
    for bits in 0..128_u32 {
        for (openh264, openh264_cisco, library) in forms {
            all.push((
                Compiled {
                    cuda: bits & 1 != 0,
                    wgpu: bits & 2 != 0,
                    dav1d: bits & 4 != 0,
                    cpu_vp9: bits & 8 != 0,
                    cpu_hevc: bits & 16 != 0,
                    cpu_av1: bits & 32 != 0,
                    vulkan_video: bits & 64 != 0,
                    openh264,
                    openh264_cisco,
                },
                library,
            ));
        }
    }
    all
}

/// Every answer `compiled` can see. A feature that is off answers one way.
fn machines(compiled: &Compiled) -> Vec<Machine> {
    let probes = |on: bool| if on { vec![true, false] } else { vec![false] };
    let answer = |on: bool, what: &str| if on { up() } else { down(what) };
    let mut all = Vec::new();
    for cuda in probes(compiled.cuda) {
        for nvdec in probes(compiled.cuda) {
            for nvenc in probes(compiled.cuda) {
                for wgpu in probes(compiled.wgpu) {
                    for vulkan in probes(compiled.vulkan_video) {
                        all.push(Machine {
                            cuda: answer(cuda, "cuda"),
                            nvdec: answer(nvdec, "nvdec"),
                            nvenc: answer(nvenc, "nvenc"),
                            wgpu: answer(wgpu, "wgpu"),
                            vulkan: answer(vulkan, "vulkan"),
                        });
                    }
                }
            }
        }
    }
    all
}

fn expected_runtime(
    request: &ExecutionRequest,
    compiled: &Compiled,
    machine: &Machine,
) -> Option<RuntimeKind> {
    let hardware =
        request.decoder == DecoderChoice::Nvdec || request.encoder == EncoderChoice::Nvenc;
    let vulkan = request.decoder == DecoderChoice::Vulkan;
    let cuda = compiled.cuda && machine.cuda.is_ok();
    let wgpu = compiled.wgpu && machine.wgpu.is_ok();
    match request.runtime {
        RuntimeChoice::Auto if vulkan => (wgpu && !hardware).then_some(RuntimeKind::Wgpu),
        RuntimeChoice::Auto if hardware => cuda.then_some(RuntimeKind::Cuda),
        RuntimeChoice::Auto if cuda => Some(RuntimeKind::Cuda),
        RuntimeChoice::Auto if wgpu => Some(RuntimeKind::Wgpu),
        RuntimeChoice::Auto => Some(RuntimeKind::Cpu),
        RuntimeChoice::Cuda => (cuda && !vulkan).then_some(RuntimeKind::Cuda),
        RuntimeChoice::Wgpu => (wgpu && !hardware).then_some(RuntimeKind::Wgpu),
        RuntimeChoice::Cpu => (!hardware && !vulkan).then_some(RuntimeKind::Cpu),
    }
}

fn expected_openh264(compiled: &Compiled, library: bool) -> Option<Openh264Build> {
    if library {
        compiled.openh264_cisco.then_some(Openh264Build::Cisco)
    } else {
        compiled.openh264.then_some(Openh264Build::Source)
    }
}

fn expected_cpu_decoder(
    video: &SourceFacts,
    compiled: &Compiled,
    library: bool,
) -> Option<DecoderBackend> {
    match video.codec {
        SourceCodec::Av1 => compiled.dav1d.then_some(DecoderBackend::Dav1d),
        SourceCodec::Vp9 => compiled.cpu_vp9.then_some(DecoderBackend::RustyVp9),
        SourceCodec::Hevc => compiled.cpu_hevc.then_some(DecoderBackend::Hevc),
        SourceCodec::H264 => {
            let baseline = video.h264_profile.is_none_or(|profile| profile == BASELINE);
            expected_openh264(compiled, library)
                .filter(|_| baseline)
                .map(DecoderBackend::Openh264)
        }
    }
}

fn expected_cpu_encoder(
    codec: Codec,
    compiled: &Compiled,
    library: bool,
) -> Option<EncoderBackend> {
    match codec {
        Codec::Hevc => None,
        Codec::H264 => expected_openh264(compiled, library).map(EncoderBackend::Openh264),
        Codec::Av1 => compiled.cpu_av1.then_some(EncoderBackend::Rav1r),
    }
}

/// The decoder of the output check: the run's own when it reads the output, then the
/// runtime's hardware decoder and the CPU decoder, in the order `--decoder` states.
fn expected_check(
    request: &ExecutionRequest,
    runtime: RuntimeKind,
    decoder: DecoderBackend,
    encoder: EncoderBackend,
    compiled: &Compiled,
    machine: &Machine,
) -> Option<DecoderBackend> {
    let codec = request.codec;
    let same = match decoder {
        DecoderBackend::Nvdec => true,
        DecoderBackend::Vulkan => codec != Codec::Av1,
        DecoderBackend::Dav1d => codec == Codec::Av1,
        DecoderBackend::Hevc => codec == Codec::Hevc,
        DecoderBackend::RustyVp9 => false,
        DecoderBackend::Openh264(_) => {
            codec == Codec::H264 && matches!(encoder, EncoderBackend::Openh264(_))
        }
    };
    if same {
        return Some(decoder);
    }
    let hardware = match runtime {
        RuntimeKind::Cuda => machine.nvdec.is_ok().then_some(DecoderBackend::Nvdec),
        RuntimeKind::Wgpu => {
            (machine.vulkan.is_ok() && codec != Codec::Av1).then_some(DecoderBackend::Vulkan)
        }
        RuntimeKind::Cpu => None,
    };
    let cpu = match (codec, encoder) {
        (Codec::Av1, _) => compiled.dav1d.then_some(DecoderBackend::Dav1d),
        (Codec::Hevc, _) => compiled.cpu_hevc.then_some(DecoderBackend::Hevc),
        (Codec::H264, EncoderBackend::Openh264(build)) => Some(DecoderBackend::Openh264(build)),
        (Codec::H264, _) => None,
    };
    if request.decoder == DecoderChoice::Cpu {
        cpu.or(hardware)
    } else {
        hardware.or(cpu)
    }
}

fn expected(
    request: &ExecutionRequest,
    video: &SourceFacts,
    compiled: &Compiled,
    machine: &Machine,
) -> Option<(RuntimeKind, DecoderBackend, EncoderBackend, DecoderBackend)> {
    let library = request.openh264_library;
    let runtime = expected_runtime(request, compiled, machine)?;
    let hardware = runtime == RuntimeKind::Cuda;
    let vulkan = machine.vulkan.is_ok() && video.codec != SourceCodec::Av1;
    let decoder = match request.decoder {
        DecoderChoice::Nvdec => machine.nvdec.is_ok().then_some(DecoderBackend::Nvdec)?,
        DecoderChoice::Vulkan => vulkan.then_some(DecoderBackend::Vulkan)?,
        DecoderChoice::Cpu => expected_cpu_decoder(video, compiled, library)?,
        DecoderChoice::Auto if hardware && machine.nvdec.is_ok() => DecoderBackend::Nvdec,
        DecoderChoice::Auto if runtime == RuntimeKind::Wgpu && vulkan => DecoderBackend::Vulkan,
        DecoderChoice::Auto => expected_cpu_decoder(video, compiled, library)?,
    };
    let encoder = match request.encoder {
        EncoderChoice::Nvenc => machine.nvenc.is_ok().then_some(EncoderBackend::Nvenc)?,
        EncoderChoice::Cpu => expected_cpu_encoder(request.codec, compiled, library)?,
        EncoderChoice::Auto if hardware && machine.nvenc.is_ok() => EncoderBackend::Nvenc,
        EncoderChoice::Auto => expected_cpu_encoder(request.codec, compiled, library)?,
    };
    let check = expected_check(request, runtime, decoder, encoder, compiled, machine)?;
    Some((runtime, decoder, encoder, check))
}

/// The allowed pairs, stated apart from the rules above.
fn assert_allowed(execution: &Execution, name: &impl Fn() -> String) {
    if execution.runtime != RuntimeKind::Cuda {
        assert_ne!(execution.decoder, DecoderBackend::Nvdec, "{}", name());
        assert_ne!(execution.encoder, EncoderBackend::Nvenc, "{}", name());
        assert_ne!(execution.check_decoder, DecoderBackend::Nvdec, "{}", name());
    }
    if execution.runtime != RuntimeKind::Wgpu {
        assert_ne!(execution.decoder, DecoderBackend::Vulkan, "{}", name());
        assert_ne!(
            execution.check_decoder,
            DecoderBackend::Vulkan,
            "{}",
            name()
        );
    }
}

/// Every refusal states a next action, and each sentence stays free of hedges.
fn assert_actionable(error: &ExecutionError, name: &impl Fn() -> String) {
    let text = error.to_string();
    let acts = ["--", "Pass", "Rebuild", "Install", "Run from", "Use"];
    assert!(
        acts.iter().any(|act| text.contains(act)),
        "{}: no next action in: {text}",
        name()
    );
    for hedge in ["should", "might", "could", " may "] {
        assert!(!text.contains(hedge), "{}: `{hedge}` in: {text}", name());
    }
}

#[test]
fn every_combination_follows_the_rules() {
    for (compiled, library) in builds() {
        for machine in machines(&compiled) {
            for &runtime in RuntimeChoice::ALL {
                for &decoder in DecoderChoice::ALL {
                    for &encoder in EncoderChoice::ALL {
                        for codec in OUTPUT_CODECS {
                            let request = ExecutionRequest {
                                runtime,
                                decoder,
                                encoder,
                                codec,
                                openh264_library: library,
                            };
                            for video in sources() {
                                check_one(&request, &video, &compiled, &machine);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn check_one(
    request: &ExecutionRequest,
    video: &SourceFacts,
    compiled: &Compiled,
    machine: &Machine,
) {
    let name = || format!("{request:?} {video:?} {compiled:?} {machine:?}");
    let chosen = execution_for(request, video, compiled, machine);
    match (chosen, expected(request, video, compiled, machine)) {
        (Ok(execution), Some((runtime, decoder, encoder, check))) => {
            assert_eq!(execution.runtime, runtime, "{}", name());
            assert_eq!(execution.decoder, decoder, "{}", name());
            assert_eq!(execution.encoder, encoder, "{}", name());
            assert_eq!(execution.check_decoder, check, "{}", name());
            assert_allowed(&execution, &name);
            let reasons = &execution.reasons;
            assert!(!reasons.runtime.is_empty(), "{}", name());
            assert!(!reasons.decoder.is_empty(), "{}", name());
            assert!(!reasons.encoder.is_empty(), "{}", name());
            assert!(!reasons.check_decoder.is_empty(), "{}", name());
        }
        (Err(error), None) => assert_actionable(&error, &name),
        (chosen, wanted) => panic!(
            "{}: chose {chosen:?}, and the rules give {wanted:?}",
            name()
        ),
    }
}

fn no_gpu() -> Machine {
    Machine {
        cuda: down("CUDA device 0"),
        nvdec: down("libnvcuvid.so.1"),
        nvenc: down("libnvidia-encode.so.1"),
        wgpu: down("a WebGPU adapter"),
        vulkan: down("an H.264 Vulkan Video queue"),
    }
}

fn all_up() -> Machine {
    Machine {
        cuda: up(),
        nvdec: up(),
        nvenc: up(),
        wgpu: up(),
        vulkan: up(),
    }
}

fn everything() -> Compiled {
    Compiled {
        cuda: true,
        wgpu: true,
        vulkan_video: true,
        dav1d: true,
        cpu_vp9: true,
        cpu_hevc: true,
        openh264: true,
        openh264_cisco: true,
        cpu_av1: true,
    }
}

fn cpu_only() -> Compiled {
    Compiled {
        cuda: false,
        wgpu: false,
        vulkan_video: false,
        ..everything()
    }
}

fn request(codec: Codec) -> ExecutionRequest {
    ExecutionRequest {
        codec,
        ..ExecutionRequest::default()
    }
}

fn refused(
    request: &ExecutionRequest,
    video: &SourceFacts,
    compiled: &Compiled,
    machine: &Machine,
) -> String {
    execution_for(request, video, compiled, machine)
        .expect_err("refused")
        .to_string()
}

#[test]
fn auto_prefers_the_hardware_path_and_says_so() {
    let chosen = execution_for(
        &request(Codec::Hevc),
        &h264(Some(HIGH)),
        &everything(),
        &all_up(),
    )
    .expect("execution");
    assert_eq!(chosen.runtime, RuntimeKind::Cuda);
    assert_eq!(chosen.decoder, DecoderBackend::Nvdec);
    assert_eq!(chosen.encoder, EncoderBackend::Nvenc);
    assert_eq!(chosen.check_decoder, DecoderBackend::Nvdec);
    assert!(chosen.reasons.decoder.contains("NVDEC"));
    assert!(chosen.reasons.encoder.contains("NVENC"));
}

#[test]
fn auto_falls_through_with_the_probe_in_the_reason() {
    let chosen = execution_for(
        &request(Codec::Av1),
        &source(SourceCodec::Av1),
        &everything(),
        &no_gpu(),
    )
    .expect("execution");
    assert_eq!(chosen.runtime, RuntimeKind::Cpu);
    assert_eq!(chosen.decoder, DecoderBackend::Dav1d);
    assert_eq!(chosen.encoder, EncoderBackend::Rav1r);
    assert!(chosen.reasons.runtime.contains("CUDA device 0 is absent"));
    assert!(chosen.reasons.runtime.contains("WebGPU adapter is absent"));
    assert!(chosen.reasons.decoder.contains("dav1d"));
    assert!(chosen.reasons.encoder.contains("rav1r"));
}

/// A build on the WebGPU runtime with Vulkan Video and every CPU codec.
fn webgpu_build() -> Compiled {
    Compiled {
        cuda: false,
        ..everything()
    }
}

fn webgpu_request(decoder: DecoderChoice, codec: Codec) -> ExecutionRequest {
    ExecutionRequest {
        runtime: RuntimeChoice::Wgpu,
        decoder,
        ..request(codec)
    }
}

#[test]
fn auto_on_webgpu_picks_vulkan_and_cpu_for_av1() {
    let chosen = execution_for(
        &webgpu_request(DecoderChoice::Auto, Codec::Av1),
        &h264(Some(HIGH)),
        &webgpu_build(),
        &all_up(),
    )
    .expect("execution");
    assert_eq!(chosen.decoder, DecoderBackend::Vulkan);
    assert_eq!(chosen.encoder, EncoderBackend::Rav1r);
    assert_eq!(chosen.check_decoder, DecoderBackend::Dav1d);
    assert!(chosen.reasons.decoder.contains("Vulkan Video"));

    for codec in [SourceCodec::Hevc, SourceCodec::Vp9] {
        let chosen = execution_for(
            &webgpu_request(DecoderChoice::Auto, Codec::Av1),
            &source(codec),
            &webgpu_build(),
            &all_up(),
        )
        .expect("execution");
        assert_eq!(chosen.decoder, DecoderBackend::Vulkan, "{}", codec.name());
        assert!(chosen.reasons.decoder.contains("Vulkan Video"));
    }

    let chosen = execution_for(
        &webgpu_request(DecoderChoice::Auto, Codec::Av1),
        &source(SourceCodec::Av1),
        &webgpu_build(),
        &all_up(),
    )
    .expect("execution");
    assert_eq!(chosen.decoder, DecoderBackend::Dav1d);
    assert!(chosen.reasons.decoder.contains("not supported yet"));
}

#[test]
fn vulkan_reads_back_h264_output_and_auto_falls_to_the_cpu_without_it() {
    let chosen = execution_for(
        &webgpu_request(DecoderChoice::Vulkan, Codec::H264),
        &h264(Some(HIGH)),
        &webgpu_build(),
        &all_up(),
    )
    .expect("execution");
    assert_eq!(chosen.decoder, DecoderBackend::Vulkan);
    assert_eq!(chosen.check_decoder, DecoderBackend::Vulkan);
    assert!(matches!(chosen.encoder, EncoderBackend::Openh264(_)));

    let machine = Machine {
        vulkan: down("an H.264 Vulkan Video queue"),
        ..all_up()
    };
    let chosen = execution_for(
        &webgpu_request(DecoderChoice::Auto, Codec::Av1),
        &h264(Some(BASELINE)),
        &webgpu_build(),
        &machine,
    )
    .expect("execution");
    assert!(matches!(chosen.decoder, DecoderBackend::Openh264(_)));
    assert!(chosen.reasons.decoder.contains("H.264 Vulkan Video queue"));
}

#[test]
fn an_explicit_vulkan_request_names_each_refusal() {
    let codec = SourceCodec::Av1;
    let text = refused(
        &webgpu_request(DecoderChoice::Vulkan, Codec::Av1),
        &source(codec),
        &webgpu_build(),
        &all_up(),
    );
    assert!(text.contains(codec.name()), "{text}");
    assert!(text.contains("not supported yet"), "{text}");
    assert!(text.contains("--decoder cpu"), "{text}");

    for codec in [SourceCodec::Hevc, SourceCodec::Vp9] {
        let chosen = execution_for(
            &webgpu_request(DecoderChoice::Vulkan, Codec::Av1),
            &source(codec),
            &webgpu_build(),
            &all_up(),
        )
        .expect("execution");
        assert_eq!(chosen.decoder, DecoderBackend::Vulkan, "{}", codec.name());
    }

    let plain = Compiled {
        vulkan_video: false,
        ..webgpu_build()
    };
    let text = refused(
        &webgpu_request(DecoderChoice::Vulkan, Codec::Av1),
        &h264(None),
        &plain,
        &all_up(),
    );
    assert!(text.contains("--features vulkan-video"), "{text}");

    let machine = Machine {
        vulkan: down("an H.264 Vulkan Video queue"),
        ..all_up()
    };
    let text = refused(
        &webgpu_request(DecoderChoice::Vulkan, Codec::Av1),
        &h264(None),
        &webgpu_build(),
        &machine,
    );
    assert!(text.contains("H.264 Vulkan Video queue"), "{text}");
    assert!(text.contains("--decoder cpu"), "{text}");
}

#[test]
fn vulkan_with_cuda_or_cpu_is_refused_and_names_both_sides() {
    for (runtime, name) in [
        (RuntimeChoice::Cuda, "--runtime cuda"),
        (RuntimeChoice::Cpu, "--runtime cpu"),
    ] {
        let asked = ExecutionRequest {
            runtime,
            decoder: DecoderChoice::Vulkan,
            ..request(Codec::Av1)
        };
        let text = refused(&asked, &h264(None), &everything(), &all_up());
        assert!(text.contains(name), "{text}");
        assert!(text.contains("--decoder vulkan"), "{text}");
        assert!(text.contains("--runtime wgpu"), "{text}");
    }
    let asked = ExecutionRequest {
        decoder: DecoderChoice::Vulkan,
        encoder: EncoderChoice::Nvenc,
        ..request(Codec::Hevc)
    };
    let text = refused(&asked, &h264(None), &everything(), &all_up());
    assert!(text.contains("--decoder vulkan"), "{text}");
    assert!(text.contains("--encoder nvenc"), "{text}");
}

#[test]
fn an_auto_runtime_takes_webgpu_for_a_vulkan_request() {
    let asked = ExecutionRequest {
        decoder: DecoderChoice::Vulkan,
        ..request(Codec::Av1)
    };
    let chosen = execution_for(&asked, &h264(None), &everything(), &all_up()).expect("execution");
    assert_eq!(chosen.runtime, RuntimeKind::Wgpu);
    assert_eq!(chosen.decoder, DecoderBackend::Vulkan);
}

#[test]
fn a_missing_library_falls_through_on_cuda_and_keeps_the_gpu_runtime() {
    let machine = Machine {
        nvdec: down("libnvcuvid.so.1"),
        nvenc: down("libnvidia-encode.so.1"),
        ..all_up()
    };
    let chosen = execution_for(
        &request(Codec::Av1),
        &source(SourceCodec::Vp9),
        &everything(),
        &machine,
    )
    .expect("execution");
    assert_eq!(chosen.runtime, RuntimeKind::Cuda);
    assert_eq!(chosen.decoder, DecoderBackend::RustyVp9);
    assert_eq!(chosen.encoder, EncoderBackend::Rav1r);
    assert!(chosen.reasons.decoder.contains("libnvcuvid.so.1"));
    assert!(chosen.reasons.encoder.contains("libnvidia-encode.so.1"));
}

#[test]
fn an_explicit_hardware_codec_names_what_is_missing() {
    let forced = ExecutionRequest {
        runtime: RuntimeChoice::Cuda,
        decoder: DecoderChoice::Nvdec,
        ..request(Codec::Hevc)
    };
    let machine = Machine {
        nvdec: down("libnvcuvid.so.1"),
        ..all_up()
    };
    let text = refused(&forced, &h264(None), &everything(), &machine);
    assert!(text.contains("--decoder nvdec"), "{text}");
    assert!(text.contains("libnvcuvid.so.1"), "{text}");
    assert!(text.contains("--decoder cpu"), "{text}");

    let forced = ExecutionRequest {
        encoder: EncoderChoice::Nvenc,
        ..request(Codec::Hevc)
    };
    let machine = Machine {
        nvenc: down("libnvidia-encode.so.1"),
        ..all_up()
    };
    let text = refused(&forced, &h264(None), &everything(), &machine);
    assert!(text.contains("--encoder nvenc"), "{text}");
    assert!(text.contains("libnvidia-encode.so.1"), "{text}");
}

#[test]
fn a_refused_pair_names_both_sides_and_the_pair_that_works() {
    let cases = [
        (
            ExecutionRequest {
                runtime: RuntimeChoice::Cpu,
                decoder: DecoderChoice::Nvdec,
                ..request(Codec::Hevc)
            },
            [
                "--runtime cpu",
                "--decoder nvdec",
                "--runtime cuda",
                "--decoder cpu",
            ],
        ),
        (
            ExecutionRequest {
                runtime: RuntimeChoice::Wgpu,
                encoder: EncoderChoice::Nvenc,
                ..request(Codec::Hevc)
            },
            [
                "--runtime wgpu",
                "--encoder nvenc",
                "--runtime cuda",
                "--encoder cpu",
            ],
        ),
    ];
    for (asked, needles) in cases {
        let text = refused(&asked, &h264(None), &everything(), &all_up());
        for needle in needles {
            assert!(text.contains(needle), "{needle} is not in: {text}");
        }
    }
}

#[test]
fn each_codec_not_compiled_names_its_feature() {
    let none = Compiled::default();
    let cpu = ExecutionRequest {
        decoder: DecoderChoice::Cpu,
        encoder: EncoderChoice::Cpu,
        ..request(Codec::Av1)
    };
    let with_encoder = Compiled {
        cpu_av1: true,
        dav1d: true,
        ..none
    };
    for (video, feature) in [
        (source(SourceCodec::Vp9), "cpu-vp9"),
        (source(SourceCodec::Hevc), "cpu-hevc"),
        (h264(Some(BASELINE)), "openh264"),
    ] {
        let text = refused(&cpu, &video, &with_encoder, &no_gpu());
        assert!(text.contains(feature), "{feature} is not in: {text}");
    }
    let text = refused(&cpu, &source(SourceCodec::Av1), &none, &no_gpu());
    assert!(text.contains("dav1d"), "{text}");
    let with_decoder = Compiled {
        dav1d: true,
        ..none
    };
    let text = refused(&cpu, &source(SourceCodec::Av1), &with_decoder, &no_gpu());
    assert!(text.contains("cpu-av1"), "{text}");
    let forced = ExecutionRequest {
        runtime: RuntimeChoice::Cuda,
        ..request(Codec::Av1)
    };
    let text = refused(&forced, &source(SourceCodec::Av1), &with_decoder, &no_gpu());
    assert!(text.contains("--features cuda"), "{text}");
    let forced = ExecutionRequest {
        runtime: RuntimeChoice::Wgpu,
        ..request(Codec::Av1)
    };
    let text = refused(&forced, &source(SourceCodec::Av1), &with_decoder, &no_gpu());
    assert!(text.contains("--features wgpu"), "{text}");
}

#[test]
fn a_source_that_is_not_baseline_is_refused_by_name_on_openh264() {
    let cpu = ExecutionRequest {
        decoder: DecoderChoice::Cpu,
        ..request(Codec::H264)
    };
    let text = refused(&cpu, &h264(Some(HIGH)), &cpu_only(), &no_gpu());
    assert!(text.contains("OpenH264"), "{text}");
    assert!(text.contains("profile_idc 100"), "{text}");
    assert!(text.contains("B-frames"), "{text}");
    assert!(text.contains("reference decode"), "{text}");
    // NVDEC takes the same source, so `auto` has a path.
    let chosen = execution_for(
        &request(Codec::H264),
        &h264(Some(HIGH)),
        &everything(),
        &all_up(),
    );
    assert_eq!(chosen.expect("execution").decoder, DecoderBackend::Nvdec);
}

#[test]
fn hevc_cpu_encode_is_refused_and_names_av1() {
    let cpu = ExecutionRequest {
        encoder: EncoderChoice::Cpu,
        ..request(Codec::Hevc)
    };
    let text = refused(&cpu, &source(SourceCodec::Av1), &cpu_only(), &no_gpu());
    assert!(text.contains("--codec av1"), "{text}");
    // `auto` with no NVENC reaches the same refusal, with the hardware reason.
    let text = refused(
        &request(Codec::Hevc),
        &source(SourceCodec::Av1),
        &cpu_only(),
        &no_gpu(),
    );
    assert!(text.contains("--codec av1"), "{text}");
    assert!(text.contains("no hardware encoder"), "{text}");
}

#[test]
fn cisco_openh264_needs_its_binary_path() {
    let cisco_only = Compiled {
        openh264_cisco: true,
        ..Compiled::default()
    };
    let cpu = ExecutionRequest {
        decoder: DecoderChoice::Cpu,
        encoder: EncoderChoice::Cpu,
        ..request(Codec::H264)
    };
    let text = refused(&cpu, &h264(Some(BASELINE)), &cisco_only, &no_gpu());
    assert!(text.contains("--openh264-library"), "{text}");
    let with_path = ExecutionRequest {
        openh264_library: true,
        ..cpu
    };
    let chosen = execution_for(&with_path, &h264(Some(BASELINE)), &cisco_only, &no_gpu())
        .expect("execution");
    assert_eq!(
        chosen.decoder,
        DecoderBackend::Openh264(Openh264Build::Cisco)
    );
    assert_eq!(
        chosen.encoder,
        EncoderBackend::Openh264(Openh264Build::Cisco)
    );
    let source_only = Compiled {
        openh264: true,
        ..Compiled::default()
    };
    let text = refused(&with_path, &h264(Some(BASELINE)), &source_only, &no_gpu());
    assert!(text.contains("--features openh264-cisco"), "{text}");
}

#[test]
fn the_output_check_needs_a_decoder_for_the_output() {
    let no_dav1d = Compiled {
        dav1d: false,
        ..cpu_only()
    };
    let cpu = request(Codec::Av1);
    let text = refused(&cpu, &source(SourceCodec::Vp9), &no_dav1d, &no_gpu());
    assert!(text.contains("output check"), "{text}");
    assert!(text.contains("dav1d"), "{text}");

    // NVENC writes H.264 High, which OpenH264 cannot read back.
    let machine = Machine {
        nvdec: down("libnvcuvid.so.1"),
        ..all_up()
    };
    let nvenc = ExecutionRequest {
        encoder: EncoderChoice::Nvenc,
        ..request(Codec::H264)
    };
    let text = refused(&nvenc, &h264(Some(BASELINE)), &everything(), &machine);
    assert!(text.contains("output check"), "{text}");
    assert!(text.contains("OpenH264 reads Baseline only"), "{text}");

    // `--decoder cpu` prefers a CPU check, and `auto` prefers NVDEC.
    let with_cpu_decoder = ExecutionRequest {
        decoder: DecoderChoice::Cpu,
        encoder: EncoderChoice::Nvenc,
        ..request(Codec::Hevc)
    };
    let chosen = execution_for(
        &with_cpu_decoder,
        &source(SourceCodec::Vp9),
        &everything(),
        &all_up(),
    )
    .expect("execution");
    assert_eq!(chosen.decoder, DecoderBackend::RustyVp9);
    assert_eq!(chosen.check_decoder, DecoderBackend::Hevc);
    assert!(chosen.reasons.check_decoder.contains("rusty_vp9"));
}
