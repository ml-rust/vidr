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
- No vendor lock-in. Hardware APIs sit behind their own features: `cuda` and `vulkan-video` today, `vaapi` when written. A vendor-neutral path exists for each operation.
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

| Capability | Features | Scope and limits | Status |
| --- | --- | --- | --- |
| Workspace, CI, release workflows | | | Implemented |
| Frame timing, output checks | | Frame count, timing, last-frame duration | Implemented |
| Copied-track checks | | `read_stream_facts`, `read_track_facts`, `check_tracks` | Implemented |
| Bitstream filters | `h264`, `hevc`, `av1` | Length-prefixed and Annex-B NAL units, OBU framing, `avcC`/`hvcC`/`av1C` parse and write, MP4 samples to NVDEC input, NVENC output to MP4 samples | Implemented |
| NVDEC decode | `nvdec` | H.264, HEVC, AV1, VP9. 4:2:0, 8 or 10 bits. Into numr CUDA tensors on numr's stream | Implemented |
| NVENC encode | `nvenc` | H.264, HEVC, AV1. NV12, and P010 where the GPU lists it. From numr CUDA tensors, to MP4 samples and config records. AV1 runs on GPUs that list it, and no test covers it on the development GPU | Implemented |
| Frame conversion | `frame` | NV12 and P010 to F32 or F16 RGB at any display size. Even-size RGB back to planes. Stays on the device | Implemented |
| Plane forms, frame routes, host planes | `frame`, `wgpu` | `PlaneForm::Words` through `to_rgb` and `from_rgb` on WebGPU. `FrameRoute` on every `Decoder` and `Encoder`. Host copies in `PumpReport`. `upload_planes` and `download_planes` for I420 and I010 | Implemented |
| Colour policy | `decode`, `merge_container` behind `mp4` or `mkv` | Container fill of bitstream colour, SDR check, conversion plan with output tags | Implemented |
| Media input | `mp4`, `mkv` | Open by file bytes through containr. First video track, edit timeline, rotation, sample aspect ratio, colour tags, decoder config | Implemented |
| Media output | `mp4`, `mkv` | One-pass `.mp4` faststart or `.mkv`. H.264, HEVC, or AV1. Copies audio, text subtitles, chapters, metadata, MKV attachments. Named refusals before GPU work. Bounded mux buffer | Implemented |
| Threaded pump | `pipeline` | Three threads with bounded channels: demux and decode, edit window and transform, encode and mux. First-error and panic handling, stop requests, last-frame duration, frame count check. `InputSource` and `MuxSink` behind `mp4` or `mkv` | Implemented |
| AV1 CPU decode | `dav1d` | `Dav1dDecoder`. 4:2:0, 8 or 10 bits | Implemented |
| H.264 CPU decode and encode | `openh264`, `openh264-cisco` | `Openh264Decoder`, Baseline only, because OpenH264's B-slice output differs from the reference. `Openh264Encoder`, 8-bit Constrained Baseline | Implemented |
| AV1 CPU encode | `cpu-av1`, `cpu-av1-asm` | `Rav1rEncoder`. 4:2:0, 8 or 10 bits, odd sizes, `av1C`, `dts` equal to `pts`, keyframes on `gop_length` only, `b_frames` 0 or 3 | Implemented |
| VP9 CPU decode | `cpu-vp9` | `RustyVp9Decoder`. Profiles 0 and 2, 4:2:0, 8 or 10 bits, odd sizes, resize, superframes, show-existing. A chunk gives its last shown frame, as libvpx does. Passes the libvpx per-frame MD5 with no failure, and refuses the rest by name | Implemented |
| HEVC CPU decode | `cpu-hevc` | `HevcDecoder`, this crate's own core, written from ITU-T H.265. Main and Main 10, 4:2:0, 8 or 10 bits. `hvcC` samples or Annex-B, conformance window. Picture hash check and thread count are constructor options, with output equal at every count. AVX2, NEON, or scalar kernels through `fearless_simd`, with `forbid(unsafe_code)`. Passes the whole-stream MD5 on every JCT-VC HEVC_v1 vector except `TSUNEQBD_A_MAIN10_Technicolor_2`, refused by name for unequal luma and chroma depths | Implemented |
| Vulkan Video decode | `vulkan-video` | `VulkanDecoder` on numr's WebGPU device, opened through wgpu-hal. Frames only. One device-to-device plane copy per frame, `FrameRoute::Device`. H.264: 8-bit 4:2:0 Baseline, Main, High. HEVC: Main, Main 10, Main Still Picture, NV12 and P010, on `decode::hevc_order`. VP9: profiles 0 and 2, NV12 and P010, hidden frames, `show_existing_frame`, last shown frame per chunk. AV1: Main, NV12 and P010, hidden frames, `show_existing_frame`, tile groups, film grain in its own output image, last shown frame per temporal unit, operating point 0, on `bitstream::av1`. `vidr-remaster` takes all four on `--runtime wgpu`. See Vulkan Video facts | Implemented |
| Vulkan Video encode | `vulkan-video` | `VulkanEncoder` on the shared `VulkanDevice`, `Encoder<WgpuRuntime>`, word-form frames, one device-to-device copy per frame into an input image, `FrameRoute::Device`. H.264 High 8-bit, HEVC Main, Main 10 where listed. One previous frame as reference, IDR at `gop_length`, `b_frames` above 0 refused by name. Constant quantiser where the driver lists its disabled mode, else mapped VBR or CBR. `avcC` and `hvcC` from driver parameter sets. Round-trip luma PSNR at quantiser 20 sits above the floors in `tests/vulkan/encode.rs` | Implemented |
| Vulkan Video AV1 encode | `vulkan-video` | Written on the same core through `VK_KHR_video_encode_av1`. Never run on any GPU. The encoder refuses AV1 by name when the driver lacks the extension | Roadmap until run |
| `vidr-remaster` | `mp4` or `mkv`, a decoder, an encoder, `model-srvgg`, `model-rrdb` | Validation, copy plan, runtime and codec choice, pre-scan on the chosen decoder, reuse, encode pass into a temporary directory, output check, publish, provenance sidecar, `report.json` with `frame_path`, licence notices, exit codes 0, 1, 130. Generic over the numr runtime. HEVC CPU encode is refused. `--encoder vulkan` and `auto` on WebGPU take Vulkan Video encode. A capability refusal under `auto` falls to the CPU encoder, with the reason in the report | Implemented |
| Conformance harness | `cpu-vp9`, `dav1d`, `cpu-hevc`, `vulkan-video` | Fetch scripts. libvpx per-frame MD5 against `RustyVp9Decoder` (`VIDR_VP9_VECTORS`). libaom per-frame MD5 (`VIDR_AV1_VECTORS`) against `Dav1dDecoder`, with no failure and monochrome vectors refused by name, and against `VulkanDecoder`. HEVC_v1 zip layout and hash lookup. With `cpu-hevc`, each picture's hash SEI and the whole-stream MD5 against `HevcDecoder` (`VIDR_HEVC_CONFORMANCE`). Mutation harnesses `hevc_mutations_stay_safe`, `hevc_vulkan_mutations_stay_safe`, `vp9_vulkan_mutations_stay_safe`, `av1_vulkan_mutations_stay_safe` | Implemented |
| Everything else | | | Roadmap |

Update this table in the same change that implements a capability.

### Runtime paths of `vidr-remaster`

| Runtime | Decode | Encode | Models run on a real clip |
| --- | --- | --- | --- |
| CUDA | NVDEC, CPU | NVENC, CPU | `fast`, `realesrgan`, `purevision` |
| WebGPU | Vulkan Video, CPU | Vulkan Video, CPU | `fast` only |
| CPU | CPU | CPU | |

- CUDA run: a real clip on the development RTX 3060.
- WebGPU run: `fast` on that GPU, between Vulkan decode and Vulkan encode. It took no host copy and copied audio, subtitles, and attachments.
- `--from-upscaled` also ran on WebGPU, with Vulkan H.264, HEVC, VP9, or AV1 decode and a CPU encoder.

### Vulkan Video facts

- Test scope: NVIDIA on the development RTX 3060, driver 615.71.09, coincident DPB mode only.
- Never run: any AMD or Intel GPU, any validation layer.
- Sweeps: HEVC_v1, libvpx VP9, and libaom AV1 give no failure on that driver. Real-GPU tests of all four decode codecs and of H.264 and HEVC encode pass.
- Named refusals: a picture past the session's active references, a slot named twice, a VP9 frame with no tile bytes, an AV1 tile outside its frame, a lost device.
- AV1 refusals: Annex B, tile lists, a frame that reads a reference its `ref_order_hint` states as lost, monochrome, coded sizes below the driver's minimum.
- Sweep refusals name driver limits on coded size, level, DPB slots, and profile.
- That driver reads session parameter pointers after create, so each pointee keeps a stable address.
- That driver refuses some small H.264 decode sessions, so each decode session opens at a floor on each side.
- That driver refuses a rate-controlled encode with no codec layer structure, so each layer states one.
- That driver refuses VBR with an equal maximum, so an equal maximum takes CBR where listed.

## Commands

- `cargo clippy -p <crate> --features <set> --all-targets -- -D warnings`, for each `<set>` in the `clippy` matrix of `.github/workflows/test.yml`
- `cargo nextest run -p <crate> --features <set> --no-fail-fast`, for each `<set>` in the `test` matrix
- `cargo test -p <crate> --features <set> --doc`, because nextest skips doctests
- `RUSTDOCFLAGS="-D warnings" cargo doc -p <crate> --no-deps`
- `cargo fmt --all --check`
- `cargo deny --exclude-dev check`
- `shellcheck scripts/ci/*.sh scripts/conformance/*.sh`
- When adding a feature that builds with no GPU SDK, add a clippy row for it in `.github/workflows/test.yml`. `cuda`, `nvdec`, `nvenc`, and `vulkan-video` stay out of CI.

## Releases

- vidr and vidr-remaster release together at one version.
- Each release needs a non-empty `## [X.Y.Z]` section in `CHANGELOG.md`.
- Pushing a `v*` tag runs the gate. Publishing is the manual `Release` workflow.
