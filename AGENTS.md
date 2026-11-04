# vidr — Agent Guidelines

Video for Rust. With voxr (audio), it replaces the `ffmpeg` command line and TensorRT-style inference as embeddable libraries.
The PoC model set (Real-ESRGAN, PureVision) is the first use case, NOT the scope.

## Placement

- `numr`: kernels, meaning tensor operations. Conv, resize, colour conversion, pixel shuffle, activations. Block-level DSP inside a CPU codec stays in `crates/vidr/src/decode/cpu/dsp`.
- `boostr`: model networks, tensor-only, one feature gate per architecture.
- `crates/vidr`: video codecs, hardware decode and encode, frame timing, output checks, colour, filter graph.
- `crates/vidr-remaster`: CLI, model selection, resolution, provenance. It calls the vidr output checks.
- voxr: audio codecs and audio processing. NEVER add audio decoding to vidr.
- containr: MP4/MOV and MKV/WebM demux, mux and edit lists, shared with voxr. NEVER parse or write a container in this repo.
- NEVER put a kernel or a model network in this repo. Add it upstream, then depend on it.

## Invariants

- Library first. The CLI is a thin layer over the public API. NEVER put logic in a binary.
- Every codec, container, hardware API, and model is a Cargo feature. The default build is minimal.
- No vendor lock-in. Hardware APIs sit behind their own features: `cuda`, `vulkan-video`, `vaapi`. A vendor-neutral path exists for each operation.
- Frames are numr tensors. On a hardware path they stay in GPU memory from decode to encode, and a device-to-device copy is fine. A host round trip there is a defect.
- A CPU codec is the one exception: it copies each frame through host memory. It declares this through `FrameRoute`, and the run states it in its output and report. A silent host copy is a defect.
- CI has no GPU. Hardware features stay out of CI. Both crates build and test without them.
- Model weights load from a directory at run time. NEVER compile in or commit weights or media.
- Output keeps the source's frame count, frame timing, last-frame duration, and aspect ratio. A mismatch is an error, never a warning.
- Rotation, audio, text subtitles, chapters, and metadata copy to the output. Refuse a stream the output container cannot hold, and name the container that can.
- Model output matches the PyTorch reference within a stated tolerance. A parity test proves it.
- A dependency must have a release or commit in the last year. Otherwise write our own.

## Implemented vs roadmap

This table is the single source of truth on what works. NEVER describe a roadmap item as working in code, docs, or replies.

| Capability | Status |
| --- | --- |
| Workspace, CI, release workflows | Implemented |
| Frame timing model and output checks | Implemented |
| Bitstream filters (`h264`, `hevc`, `av1`): length-prefixed and Annex-B NAL units, OBU framing, `avcC` / `hvcC` / `av1C` parse and write, MP4 samples to NVDEC input, NVENC output to MP4 samples | Implemented |
| NVDEC hardware decode (`nvdec`): H.264, HEVC, AV1, VP9, 4:2:0 at 8 or 10 bits, into numr CUDA tensors on numr's stream | Implemented |
| NVENC hardware encode (`nvenc`): H.264, HEVC, AV1, NV12 and P010, from numr CUDA tensors on numr's stream, to MP4 samples and config records. AV1 runs on GPUs that list it, and no test covers it on the development GPU | Implemented |
| Frame conversion (`frame`): NV12 and P010 to F32 or F16 RGB at any display size, even-size RGB back to planes, without leaving the device | Implemented |
| Colour policy (`decode`, `merge_container` behind `mp4` or `mkv`): container fill of bitstream colour, SDR check, conversion plan with output tags | Implemented |
| Media input (`mp4`, `mkv`): open by file bytes through containr, first video track, edit timeline, rotation, container sample aspect ratio, colour tags, decoder config | Implemented |
| Media output (`mp4`, `mkv`): one-pass `.mp4` faststart or `.mkv`, encoded H.264, HEVC, or AV1 with audio, text subtitles, chapters, and metadata copied, MKV attachments, named refusals before GPU work, bounded mux buffer | Implemented |
| Copied-track checks: `read_stream_facts`, `read_track_facts`, `check_tracks` | Implemented |
| Plane forms, frame routes, host planes: `PlaneForm::Words` frames through `to_rgb` and `from_rgb` on numr's WebGPU backend (`wgpu`), `FrameRoute` on every `Decoder` and `Encoder` with host copies in `PumpReport`, `upload_planes` and `download_planes` for I420 and I010. | Implemented |
| CPU codecs through host memory: `Dav1dDecoder` (`dav1d`), AV1, 4:2:0 at 8 or 10 bits. `Openh264Decoder` (`openh264` source build, `openh264-cisco` Cisco binary), Baseline only, because OpenH264's B-slice output differs from the reference. `Openh264Encoder`, 8-bit Constrained Baseline. Each declares its copy through `FrameRoute`. `vidr-remaster` selects them | Implemented |
| AV1 CPU encode (`cpu-av1`, `cpu-av1-asm` for assembly): `Rav1rEncoder` on rav1r, 4:2:0 at 8 or 10 bits, odd sizes, `av1C`, `dts` equal to `pts`, keyframes on `gop_length` only, `b_frames` 0 or 3. Declares its copy through `FrameRoute`. `vidr-remaster` selects it. rav1r is not on crates.io yet, so this resolves only through the local path patch | Implemented |
| VP9 CPU decode (`cpu-vp9`): `RustyVp9Decoder` on `rusty_vp9`, profiles 0 and 2, 4:2:0 at 8 or 10 bits, odd sizes, resize, superframes, show-existing. Declares its copy through `FrameRoute`. `vidr-remaster` selects it. A chunk gives its last shown frame, as libvpx does. Adoption depends on the libvpx conformance run | Implemented |
| HEVC CPU decode (`cpu-hevc`): `HevcDecoder` on this crate's own core, written from ITU-T H.265, Main and Main 10, 4:2:0 at 8 or 10 bits, `hvcC` samples or Annex-B, conformance window, picture hash check as a constructor option, a fixed worker pool with the thread count as a constructor option and output equal at every count. Declares its copy through `FrameRoute`. Its kernels use AVX2 on x86-64 at run time, NEON on AArch64, and a scalar form elsewhere, through `fearless_simd`, with `forbid(unsafe_code)` on the tree. `vidr-remaster` selects it. It passes the whole-stream MD5 on every JCT-VC HEVC_v1 vector except `TSUNEQBD_A_MAIN10_Technicolor_2`, which holds unequal luma and chroma depths and is refused by name | Implemented |
| Conformance harness: fetch scripts, libvpx per-frame MD5 against `RustyVp9Decoder` (`VIDR_VP9_VECTORS`), libaom per-frame MD5 (`VIDR_AV1_VECTORS`) against `Dav1dDecoder`, 240 pass, 0 fail, and 2 monochrome refused by name, and against `VulkanDecoder`, 204 pass, 0 fail, and 38 refused by name, 36 for coded sizes below the driver's minimum and 2 monochrome, HEVC_v1 zip layout and hash lookup, and with `cpu-hevc` each picture's hash SEI and the whole-stream MD5 against `HevcDecoder` (`VIDR_HEVC_CONFORMANCE`), plus the mutation harnesses `hevc_mutations_stay_safe` and `hevc_vulkan_mutations_stay_safe` under `VIDR_HEVC_CONFORMANCE`, `vp9_vulkan_mutations_stay_safe` under `VIDR_VP9_VECTORS`, and `av1_vulkan_mutations_stay_safe` under `VIDR_AV1_VECTORS` | Implemented |
| Threaded pump (`pipeline`): demux and decode, edit window and transform, encode and mux on three threads with bounded channels, first-error and panic handling, stop requests, last-frame duration, frame count check. `InputSource` and `MuxSink` behind `mp4` or `mkv` | Implemented |
| `vidr-remaster` run and command (`mp4` or `mkv`, a decoder, and an encoder, models behind `model-srvgg` and `model-rrdb`): validation, copy plan, the choice of runtime and codecs, pre-scan on the chosen decoder, reuse, encode pass into a temporary directory, output check, publish, provenance sidecar, `report.json` with `frame_path`, licence notices, exit codes 0, 1, and 130. Generic over the numr runtime: CUDA with NVDEC, NVENC, or CPU codecs, WebGPU with Vulkan Video for H.264, HEVC, VP9, and AV1 decode, Vulkan Video for H.264, HEVC, and AV1 encode, or CPU codecs, CPU with CPU codecs. HEVC CPU encode is refused. `--encoder vulkan` and `auto` on WebGPU take Vulkan Video encode, and a capability refusal under `auto` falls to the CPU encoder with the reason in the report. A real run on the development RTX 3060 covers `fast`, `realesrgan`, and `purevision` on CUDA on a real clip. On WebGPU, on that GPU with NVIDIA 615.71.09, `fast` ran on a real clip between Vulkan decode and Vulkan encode. That run took no host copy over 150 frames, and copied audio, subtitles, and attachments. `realesrgan` and `purevision` have not run on WebGPU. `--from-upscaled` also ran there with Vulkan H.264, HEVC, VP9, or AV1 decode and a CPU encoder. 226 `vidr-remaster` tests pass on that GPU. The WebGPU path ran in the coincident DPB mode only, on NVIDIA only. No AMD or Intel GPU has run it, and no validation layer has run on it | Implemented |
| Vulkan Video decode (`vulkan-video`): `VulkanDecoder` on numr's WebGPU device, opened through wgpu-hal, frames only, one device-to-device plane copy per frame, `FrameRoute::Device`. H.264: 8-bit 4:2:0 Baseline, Main, and High. HEVC: Main, Main 10, and Main Still Picture, NV12 and P010, on `decode::hevc_order`. VP9: profiles 0 and 2, NV12 and P010, hidden frames, `show_existing_frame`, last shown frame per chunk. AV1: Main, NV12 and P010, hidden frames, `show_existing_frame`, tile groups, film grain in its own output image, last shown frame per temporal unit, operating point 0, on the parser in `bitstream::av1`. Annex B, tile lists, and a frame that reads a reference its `ref_order_hint` states as lost are refused by name. `vidr-remaster` takes all four on `--runtime wgpu`. A picture past the session's active references, a slot named twice, a VP9 frame with no tile bytes, an AV1 tile outside its frame, and a lost device each return a named error. Real-GPU tests of all four codecs pass on NVIDIA 615.71.09 on the development RTX 3060, in its coincident DPB mode only. On that driver, the HEVC_v1 sweep gives 129 pass, 0 fail, and 18 refused by name. The libvpx VP9 sweep gives 233 pass, 0 fail, and 81 refused by name. Those refusals name driver limits on coded size, level, DPB slots, and profile. The libaom AV1 sweep gives 204 pass, 0 fail, and 38 refused by name: 36 for coded sizes below the driver's minimum, and 2 monochrome. 1022 `vidr` tests pass on that driver. A real clip ran Vulkan decode, the `fast` model on numr WebGPU, and Vulkan encode with no host copy. That driver reads session parameter pointers after create, so each pointee keeps a stable address. It refuses some small H.264 decode sessions, so each decode session opens at a floor on each side. NVIDIA only, in the coincident DPB mode only. No AMD or Intel GPU has run any of it, and no validation layer has run on it | Implemented |
| Vulkan Video encode (`vulkan-video`): `VulkanEncoder` on the shared `VulkanDevice`, `Encoder<WgpuRuntime>`, word-form frames, one device-to-device copy per frame into an input image, `FrameRoute::Device`. H.264 High at 8 bits, HEVC Main, and Main 10 where the driver lists it. One previous frame as the reference, an IDR picture at `gop_length`, `b_frames` above 0 refused by name, constant quantiser where the driver lists its disabled mode and else a mapped VBR or CBR rate. `avcC` and `hvcC` from the driver's parameter sets. Real-GPU tests pass on NVIDIA 615.71.09 on the development RTX 3060, 1022 in `vidr`. They cover H.264 and HEVC at 8 bits and HEVC Main 10, I and P pictures, device-to-device input. Round-trip luma PSNR at quantiser 20 sits above the floors in `tests/vulkan/encode.rs`. A real clip ran Vulkan decode, the `fast` model on numr WebGPU, and Vulkan encode. That run took no host copy over 150 frames, and copied audio, subtitles, and attachments. That driver refuses a rate-controlled encode with no codec layer structure, so each layer states one. It refuses VBR with an equal maximum, so an equal maximum takes CBR where the driver lists it. NVIDIA only, in the coincident DPB mode only. No AMD or Intel GPU has run it, and no validation layer has run on it. AV1 through `VK_KHR_video_encode_av1`, on the same core, is written and has never run on any GPU. That driver lists no such extension, and the encoder refuses AV1 by name | Implemented |
| Everything else | Roadmap |

Update this table in the same change that implements a capability.

## Commands

- `cargo clippy -p <crate> --all-targets -- -D warnings`
- `cargo nextest run -p <crate> --no-fail-fast`
- `cargo fmt --all --check`
- `cargo deny --exclude-dev check`
- `shellcheck scripts/ci/*.sh`
- When adding a non-CUDA feature, add a clippy row for it in `.github/workflows/test.yml`.

## Releases

- vidr and vidr-remaster release together at one version.
- Each release needs a non-empty `## [X.Y.Z]` section in `CHANGELOG.md`.
- Pushing a `v*` tag runs the gate. Publishing is the manual `Release` workflow.
