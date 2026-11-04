# vidr — Agent Guidelines

Video for Rust. With voxr (audio), it replaces the `ffmpeg` command line and TensorRT-style inference as embeddable libraries.
The PoC model set (Real-ESRGAN, PureVision) is the first use case, NOT the scope.

## Placement

- `numr`: kernels. Conv, resize, colour conversion, pixel shuffle, activations.
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
- Frames are numr tensors. Frames stay in GPU memory from decode to encode. A host round trip on the frame path is a defect.
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
