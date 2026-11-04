<div align="center">

# vidr

<h3>Video in Rust. A library first, on any GPU.</h3>

<p>
Decode, filter, run models, and encode video in one process, with frames in GPU memory throughout. Every codec, container, and filter is a Cargo feature, so an application links only what it uses. Model weights load at run time from a directory, and no weights are compiled in.
</p>

<p>
  <a href="#status"><strong>Status</strong></a>
  ·
  <a href="#crates"><strong>Crates</strong></a>
  ·
  <a href="#vidr-remaster"><strong>vidr-remaster</strong></a>
  ·
  <a href="#cpu-codecs"><strong>CPU codecs</strong></a>
  ·
  <a href="#scope"><strong>Scope</strong></a>
  ·
  <a href="CONTRIBUTING.md"><strong>Contributing</strong></a>
</p>

<p>
  <a href="https://discord.gg/jBhFk9kHPg">
    <img src="https://img.shields.io/discord/1453357769720594543?label=Discord&logo=discord&logoColor=white&color=5865F2" alt="Join the Discord">
  </a>
</p>

<p>
  <a href="https://github.com/ml-rust/vidr/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/ml-rust/vidr/ci.yml?branch=main&label=ci" alt="CI status">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/github/license/ml-rust/vidr" alt="License">
  </a>
  <a href="https://github.com/ml-rust/vidr/stargazers">
    <img src="https://img.shields.io/github/stars/ml-rust/vidr?style=social" alt="GitHub stars">
  </a>
</p>

</div>

vidr is a Cargo workspace for video on top of [numr](https://github.com/ml-rust/numr) tensors and [boostr](https://github.com/ml-rust/boostr) models. A frame is a numr tensor. A codec, a filter, and a neural network all read and write that buffer. [voxr](https://github.com/ml-rust/voxr) is its audio counterpart. Together they cover the work an application hands to the `ffmpeg` command line and to an inference engine. An application links them as libraries.

```
decode (hardware or software) → filters and models on the GPU → encode → mux
```

## Status

In development. No release exists yet. [AGENTS.md](AGENTS.md) holds the implemented-vs-roadmap table. These work today:

- Frame timing model and output checks
- H.264, HEVC, and AV1 bitstream filters
- NVDEC hardware decode and NVENC hardware encode
- Vulkan Video decode on numr's WebGPU device, frames only: H.264 8-bit 4:2:0 Baseline, Main, and High. HEVC Main, Main 10, and Main Still Picture at 8 or 10 bits. VP9 profiles 0 and 2 at 8 or 10 bits. Real-GPU tests pass on NVIDIA's driver only, in its coincident DPB mode. The HEVC_v1 and libvpx VP9 sweeps show no failure there, and refuse some streams by name at driver limits. No AMD or Intel GPU has run any of it, and no validation layer has run on it
- CPU codecs, each with one host copy per frame: AV1 decode on dav1d, H.264 Baseline decode and encode on OpenH264. A source build of OpenH264 (`openh264`) gets no Cisco patent cover. Cisco's cover applies only to its own binary, which `openh264-cisco` loads from a path the caller supplies
- AV1 encode on rav1r, VP9 decode on `rusty_vp9`, and HEVC Main and Main 10 decode on this crate's own core, each with one host copy per frame. See [CPU codecs](#cpu-codecs)
- Conversion between NV12 or P010 planes and float RGB
- SDR colour policy
- MP4, MOV, Matroska, and WebM input facts through containr
- One-pass MP4 and MKV output, with every other stream copied, and the copied-track checks
- Threaded decode, transform, and encode pump
- The `vidr-remaster` command, on NVDEC and NVENC or on the CPU codecs

Vulkan Video decode is the only hardware path off NVIDIA, and no GPU off NVIDIA has run it. On WebGPU, a real GPU has run `vidr-remaster` only with `--from-upscaled`, Vulkan H.264, HEVC, or VP9 decode, and a CPU encoder. No model has run on WebGPU on a real GPU. The rest of [Scope](#scope) does not work yet.

## Crates

| Crate                                   | What it holds                                                                                           |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| [`vidr`](crates/vidr)                   | Video codecs, hardware decode and encode, frame timing, colour management, and the filter graph         |
| [`vidr-remaster`](crates/vidr-remaster) | The `vidr-remaster` command: restoration, upscaling, scaling, and encoding of a whole video in one pass |

## vidr-remaster

`vidr-remaster` runs a restoration model over a whole video, resizes it, and encodes it. One command writes one output.

```
vidr-remaster clip.mp4 --model fast --models-dir ~/models
```

The output is `clip.fast.restored.mp4` beside the source. It keeps the source's frame count, frame timing, last-frame duration, rotation, and aspect ratio. Audio, text subtitles, chapters, and metadata copy with no transcode. A mismatch in the output check is an error, and no output publishes.

### Requirements

- A decoder and an encoder: the `cuda` build on an NVIDIA GPU, a CPU codec for each side, one of each, or the `vulkan-video` build for H.264, HEVC, or VP9 decode with a CPU encoder. NVENC needs driver support for API 13.1. numr's CUDA backend needs `nvcc` at build time.
- A CPU build holds only the codecs its features name. See [CPU codecs](#cpu-codecs).
- A model directory that holds the catalog's directory names. Pass it with `--models-dir`, or set `VIDR_MODELS_DIR`.

To build against sibling checkouts, copy `.cargo/config.toml.example` to `.cargo/config.toml` first.

```
cargo build --release -p vidr-remaster --features cuda,mp4,mkv,model-srvgg,model-rrdb
```

A CPU build needs no GPU:

```
cargo build --release -p vidr-remaster --features mp4,dav1d,cpu-av1,model-srvgg
vidr-remaster clip.mp4 --model fast --models-dir ~/models --codec av1
```

A build without a container, a decoder, and an encoder reports which features it lacks.

### Flags

| Flag                          | Value                                                     | Default                         |
| ----------------------------- | --------------------------------------------------------- | ------------------------------- |
| `INPUT`                       | The source video                                          | Required                        |
| `--model`                     | `fast`, `realesrgan`, `purevision`                        | `purevision`                    |
| `-o`, `--output`              | The output path, `.mp4` or `.mkv`                         | `<stem>.<model>.restored.mp4`   |
| `--models-dir`                | The model root                                            | `$VIDR_MODELS_DIR`              |
| `--quality`                   | Encoder quality, 1 to 51, lower is better                 | `24`                            |
| `--codec`                     | `hevc`, `h264`, `av1`                                     | `hevc`                          |
| `--runtime`                   | `auto`, `cuda`, `wgpu`, `cpu`                             | `auto`                          |
| `--decoder`                   | `auto`, `nvdec`, `vulkan`, `cpu`                          | `auto`                          |
| `--encoder`                   | `auto`, `nvenc`, `cpu`                                    | `auto`                          |
| `--openh264-library`          | Cisco's OpenH264 binary, with `openh264-cisco`            | None                            |
| `--resolution`                | `auto`, `original`, `2x`, `480`, `720`, `1080`, `1440`, `2160`, `4320` | `auto`             |
| `--overwrite`                 | Replaces an existing output                               | Off                             |
| `--from-upscaled`             | Resizes this enhanced video, with no model pass           | None                            |
| `--force-ai`                  | Runs the model even when a matching 2x output exists      | Off                             |
| `--notices`                   | Prints the licence notices of this build                  | Off                             |

`--resolution` sets the shorter edge. `auto` takes the largest standard edge at or below the source's. The shorter edge never falls below 480.

Without `--force-ai` or `--from-upscaled`, a run looks beside the source for `<stem>.<model>.2x` and `<stem>.<model>.restored`, `.mp4` or `.mkv`. A candidate exactly twice the source's size, with a matching `.upscale.json` sidecar, replaces the model. A candidate with a missing or mismatched sidecar stops the run, and the error names both flags. A sidecar names the encoder that wrote its video, so a candidate from another encoder stops the run the same way.

### Execution

| Runtime | Decoders | Encoders |
| ------- | -------- | -------- |
| `cuda` | NVDEC, CPU | NVENC, CPU |
| `wgpu` | Vulkan Video for H.264, HEVC, and VP9, CPU | CPU |
| `cpu` | CPU | CPU |

`auto` prefers CUDA, then WebGPU, then the CPU, and NVDEC and NVENC over CPU codecs. When a library or device is missing, or a hardware decoder refuses the stream, `auto` falls through. Refusals of H.264 slice groups, data partitioning, a `frame_num` gap, field coding, and bit depth count. A stalled GPU queue does not fall through. The run ends with the error that names the stall. An explicit `--decoder nvdec`, `--decoder vulkan`, or `--encoder nvenc` then ends in an error naming what is missing. Under `auto` on `wgpu`, AV1 takes its CPU decoder, and `--decoder vulkan` on AV1 states that the codec is not supported yet. A pair outside the table is refused, and the error names both sides and the pair that works.

The output check decodes the output. The run's decoder does this when it reads the output codec, and otherwise NVDEC or the CPU decoder of that codec does. With no such decoder the run stops before any work.

Vulkan Video copies each plane from a video image to a numr buffer on the device, with no host copy. A CPU codec copies each frame through host memory. The run states this on one line of stdout. `report.json` holds `frame_path`: the runtime, each codec and route, the output check's decoder, the host copies of the encode pass and of the scans, and why each part was chosen.

A model on the CPU or WebGPU runtime runs at F32. A model on CUDA runs at F16.

### Models

Each directory under the model root holds one `.pth` file.

| `--model`    | Directory              | Network         | Licence of the weights | Cargo feature |
| ------------ | ---------------------- | --------------- | ---------------------- | ------------- |
| `fast`       | `realesr-general-x4v3` | SRVGGNetCompact | BSD-3-Clause           | `model-srvgg` |
| `realesrgan` | `RealESRGAN_x2plus`    | RRDBNet         | BSD-3-Clause           | `model-rrdb`  |
| `purevision` | `2x_PureVision`        | RRDBNet         | CC-BY-NC-SA-4.0        | `model-rrdb`  |

The `purevision` licence forbids commercial use and requires share-alike. The default model is `purevision`, so pass `--model` for commercial work. A run's report states this notice. A missing model's error names the expected path and the download URL. The command downloads nothing.

### Refusals

Each refusal ends the run with exit code 1 and leaves no output.

| Refused                                                 | Why                                                         | Next action                               |
| ------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------- |
| HDR, in the bitstream or the container tags             | The colour policy covers SDR only                           | Use an SDR source, or tone-map it first   |
| Interlaced or MBAFF-coded frames                        | The frame transform needs progressive frames                | Deinterlace the source first              |
| Fragmented MP4                                          | containr does not read it                                   | `ffmpeg -i <input> -c copy <output.mp4>`  |
| A size, pixel format, or colour change inside the video | A run holds one conversion and one encoder session          | Split the source at the change            |
| Nonincreasing pts, or a frame count unlike the packets  | The output check needs one frame per packet, in order       | Repair or remux the source                |
| A 10-bit source with `--codec h264`                     | NVENC and OpenH264 do not encode H.264 at 10 bits           | Pass `--codec hevc`, or `--codec av1` on the CPU |
| HEVC with a CPU encoder                                 | No CPU HEVC encoder exists here, by decision                | Pass `--codec av1`                        |
| An H.264 source that is not Baseline, on OpenH264       | OpenH264's B-slice output differs from the reference        | Use NVDEC or Vulkan Video, or re-encode as Baseline |
| Vulkan Video with `--runtime cuda` or `--encoder nvenc` | Its frames live on WebGPU tensors                           | Pass `--runtime wgpu`, with `--encoder cpu` |
| H.264 slice groups, data partitioning, or a `frame_num` gap on Vulkan Video | Vulkan Video has no form for the first two, and vidr does not infer missing frames | Use another decoder, or re-encode |
| An HEVC picture that binds a reference the DPB lacks, on Vulkan Video | Vulkan Video activates a DPB slot only through a decode | Use another decoder, or start at an earlier IRAP picture |
| A VP9 frame that grows past its session and keeps references, on Vulkan Video | A new session holds none of the references | Use another decoder, or call `VulkanDecoder::with_max_coded_size` in library code |
| An H.264 IDR picture that drops waiting frames on Vulkan Video: `no_output_of_prior_pics_flag` 1, or a new size or `max_dec_frame_buffering` | ITU-T H.264 C.4.4 removes them, so frames no longer match packets and the scan refuses the count | Use another decoder, or re-encode |
| A codec, a runtime, or `--openh264-library` not in the build | The feature is off                                     | Rebuild with the feature the error names  |
| A candidate from another encoder                        | The sidecar names another encoder                           | Pass `--from-upscaled` or `--force-ai`    |
| Attachments or ASS subtitles into `.mp4`                | MP4 has no form for them                                    | Write `.mkv`                              |
| A Matroska crop or `ContentEncoding`                    | NVDEC applies neither                                       | Remux or re-encode the source             |
| More than one MP4 sample description                    | A decoder takes one config record                           | Split or transcode the source             |
| An existing output                                      | A run replaces a file only on request                       | Pass `--overwrite`, or choose another path |

### Results

| Exit code | Meaning                                                                                       |
| --------- | --------------------------------------------------------------------------------------------- |
| 0         | The output and its sidecar are in place. A report that does not write gives a warning         |
| 1         | A refusal, a failure, or a final output whose sidecar did not land. The error names both paths |
| 130       | A signal stopped the run. The temporary output is gone                                         |

The first Ctrl-C, SIGTERM, or SIGHUP asks the run to stop at the next frame. A second one removes the temporary output and ends the process.

`report.json` goes under `$XDG_STATE_HOME/vidr-remaster/upscale-<run id>/`, else `~/.local/state`. Every run past validation writes one.

## CPU codecs

A CPU codec works in host memory, so each frame takes one copy through it. The codec declares the copy through `FrameRoute`. Every feature here is off by default. `vidr-remaster` has a feature of the same name for each codec, and selects the codec with `--decoder cpu` or `--encoder cpu`, or through `auto`.

| Codec | Feature | Direction | Bit depth, chroma | Crate | Licence | Patent note |
| ----- | ------- | --------- | ----------------- | ----- | ------- | ----------- |
| AV1 | `dav1d` | Decode | 8, 10, 4:2:0 | libdav1d through `dav1d` | BSD-2-Clause | AV1: see below |
| AV1 | `cpu-av1`, `cpu-av1-asm` | Encode | 8, 10, 4:2:0 | `rav1r` | BSD-2-Clause | AOM Patent License 1.0: see below |
| VP9 | `cpu-vp9` | Decode | 8, 10, 4:2:0 | `rusty_vp9` | Apache-2.0 | VP9 has a patent pool: see below |
| HEVC | `cpu-hevc` | Decode | 8, 10, 4:2:0 | This crate | Apache-2.0 | HEVC has patent pools: see below |
| H.264 | `openh264`, `openh264-cisco` | Decode, encode | 8, 4:2:0 | OpenH264 through `openh264` | BSD-2-Clause | Cisco's cover applies only to its own binary, which `openh264-cisco` loads |

These notes state what a binary distribution needs. They are not legal advice.

- **rav1r.** rav1r is a fork of rav1e, under the BSD-2-Clause licence and the Alliance for Open Media Patent License 1.0.
- **Notices.** `vidr-remaster --notices` prints the text of each codec in the build. `crates/vidr-remaster/notices/` holds each text unchanged, and the crate package includes that directory.
- **AV1.** A binary built with `cpu-av1` must reproduce the Alliance for Open Media Patent License 1.0 in its documentation, legal notices, or other written materials, with the BSD licence text. rav1r's `PATENTS` file holds the patent licence. Section 1.3 of that licence ends it for a licensee that sues over an Implementation, with two stated exceptions.
- **VP9.** VP9 has a patent pool. Shipping a binary with `cpu-vp9` on needs legal review.
- **HEVC.** HEVC has patent pools. Shipping a binary with `cpu-hevc` on needs legal review.

`rav1r` is not on crates.io yet. Until it is, `cpu-av1` builds only through the `[patch.crates-io]` path in `.cargo/config.toml`, and a CI run on a clean checkout does not resolve it.

`rusty_vp9` stays pinned to one release, and its adoption depends on the conformance run below.

### Conformance

The conformance sets stay outside the repository. Their licence terms are unconfirmed, and nothing here redistributes them.

```
scripts/conformance/fetch_vp9.sh DIR
VIDR_VP9_VECTORS=DIR cargo test -p vidr --features cpu-vp9,mkv libvpx_vectors -- --nocapture
```

The test compares every frame with libvpx's `.md5` file, prints a pass and fail table by vector, and names each failure. `VIDR_VP9_FILTER` picks vectors by name, and `VIDR_CONFORMANCE_JOBS` sets how many run at once. ffmpeg remuxes the `.ivf` vectors.

```
scripts/conformance/fetch_hevc.sh DIR
VIDR_HEVC_CONFORMANCE=DIR cargo test -p vidr --release --features cpu-hevc hevc_zips_decode -- --nocapture
```

The test decodes each HEVC_v1 zip through `HevcDecoder`. It checks each picture's hash SEI message and the zip's MD5 of the whole decoded stream, keeps no frame, and needs `unzip`. It prints a table by vector and names each failure. `VIDR_HEVC_ONLY` picks zips by name, `VIDR_HEVC_REPORT` names a file for one line per vector, and `VIDR_HEVC_THREADS` sets the worker threads per decoder, 1 by default. Without `cpu-hevc`, the same variable reads each zip's layout and decodes nothing.

## Scope

| Area            | Planned                                                                                       |
| --------------- | --------------------------------------------------------------------------------------------- |
| Hardware codecs | NVDEC and NVENC, Vulkan Video, VA-API                                                         |
| Software codecs | AV1, VP9, H.264, and HEVC decode. AV1 encode                                                  |
| Containers      | MP4, Matroska, and WebM read and write, with stream copy for audio, subtitles, and chapters   |
| Colour          | BT.601, BT.709, and BT.2020 matrices, full and limited range, HDR10 and HLG, tone mapping     |
| Filters         | Scaling, cropping, deinterlacing, frame-rate conversion, denoising, and deblocking            |
| Models          | Super-resolution, temporal restoration, frame interpolation, and any vision model boostr runs |
| Distribution    | Multi-GPU processing over [nexar](https://github.com/ml-rust/nexar)                           |

## Minimum Rust version

1.95.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions and checks.

Questions and design discussion happen on [Discord](https://discord.gg/jBhFk9kHPg).

## License

Licensed under Apache-2.0. See [LICENSE](LICENSE).
