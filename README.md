# vidr

Video decode, encode, and neural-network frame processing for Rust, on GPU or CPU.

vidr decodes video, runs models on the frames, and encodes the result in one process. It builds on [numr](https://github.com/ml-rust/numr) tensors and [boostr](https://github.com/ml-rust/boostr) models. Each frame is a numr tensor. Codecs and models read and write that same buffer. vidr needs no Python.

```
decode (hardware or CPU) → models → encode → mux
```

- On a hardware path, frames stay in GPU memory from decode to encode.
- Every codec, container, and model is a Cargo feature. An application links only what it uses.
- Model weights load from a directory at run time. vidr compiles in no weights.
- [voxr](https://github.com/ml-rust/voxr) handles audio. Together, vidr and voxr cover the work an application hands to the `ffmpeg` command line and an inference engine.

The library is the main product. [`vidr-remaster`](#vidr-remaster), a command that restores and upscales a whole video, is its first use case.

vidr is in development.

## What vidr supports

### Supported codecs

All decode and encode paths are 4:2:0.

Decode:

| Codec | NVIDIA (`nvdec`) | Vulkan Video (`vulkan-video`) | CPU |
| ----- | ---------------- | ----------------------------- | --- |
| H.264 | 8 or 10 bits | 8 bits. Baseline, Main, High | 8 bits, Baseline only (`openh264`, `openh264-cisco`) |
| HEVC | 8 or 10 bits | 8 or 10 bits. Main, Main 10, Main Still Picture | 8 or 10 bits. Main, Main 10 (`cpu-hevc`) |
| VP9 | 8 or 10 bits | 8 or 10 bits. Profiles 0 and 2 | 8 or 10 bits. Profiles 0 and 2 (`cpu-vp9`) |
| AV1 | 8 or 10 bits | 8 or 10 bits. Main | 8 or 10 bits (`dav1d`) |

Encode:

| Codec | NVIDIA (`nvenc`) | Vulkan Video (`vulkan-video`) | CPU |
| ----- | ---------------- | ----------------------------- | --- |
| H.264 | 8 bits. 10 bits on a GPU that supports it | 8 bits, High | 8 bits, Constrained Baseline (`openh264`, `openh264-cisco`) |
| HEVC | 8 bits. 10 bits on a GPU that supports it | Main. Main 10 where the driver supports it | Not planned |
| VP9 | None | None | None |
| AV1 | 8 bits. 10 bits on a GPU that supports it | Written, never run on a GPU (roadmap) | 8 or 10 bits (`cpu-av1`, `cpu-av1-asm`) |

Vulkan Video encode limits:

- Each frame uses the previous frame as its only reference.
- An IDR picture starts every `gop_length` frames.
- A request for B-frames stops with an error.
- AV1 encode stops with an error when the driver lacks `VK_KHR_video_encode_av1`.

### Codec backends

| Backend | Codecs | Implementation |
| ------- | ------ | -------------- |
| NVDEC and NVENC | NVIDIA GPUs | numr's CUDA backend |
| Vulkan Video | GPUs whose driver supports the Vulkan video extensions | numr's WebGPU backend |
| OpenH264 | CPU H.264 decode and encode | C++ library, called from Rust |
| libdav1d | CPU AV1 decode | C library, called from Rust |
| Rust | CPU HEVC decode, CPU VP9 decode, CPU AV1 encode through [rav1r](https://github.com/ml-rust/rav1r) | Rust |

[containr](https://github.com/ml-rust/containr) reads and writes MP4, MOV, Matroska, and WebM.

### Other library features

| Capability | Cargo feature |
| ---------- | ------------- |
| Frame timing model and output checks | Always built |
| Bitstream filters | `h264`, `hevc`, `av1` |
| Conversion between NV12 or P010 planes and float RGB | `frame` |
| SDR colour handling | `decode` |
| MP4, MOV, Matroska, and WebM input | `mp4`, `mkv` |
| One-pass MP4 and MKV output. Other streams copy over, and vidr checks them | `mp4`, `mkv` |
| Threaded decode, transform, and encode pipeline | `pipeline` |
| The `vidr-remaster` command | See [Build](#build) |

[AGENTS.md](AGENTS.md) holds the full table of what works and what is roadmap.

### Tested hardware

vidr is tested on one NVIDIA GPU with NVIDIA's driver. No AMD or Intel GPU has run any of it.

| Path | Tested with |
| ---- | ----------- |
| NVDEC, NVENC | A real clip with all three models. No test covers NVENC AV1 |
| Vulkan Video decode | A real clip, plus the HEVC_v1, libvpx VP9, and libaom AV1 test vectors. All decode with no error |
| Vulkan Video encode | A real clip, H.264 and HEVC |
| `dav1d` | The libaom AV1 test vectors. Monochrome vectors stop with an error |
| `cpu-vp9` | The libvpx VP9 test vectors |
| `cpu-hevc` | The HEVC_v1 test vectors. One vector with unequal luma and chroma bit depths stops with an error |
| `cpu-av1`, `openh264` | Tests in this repository. No conformance test set exists for these |

Vulkan Video test limits:

- Only tested with the driver mode that keeps decoded and reference pictures in one image (coincident DPB mode).
- Not tested with the Vulkan validation layers.
- A stream past a driver limit (size, level, AV1 monochrome) stops with an error naming the limit.

Models run by `vidr-remaster`, per runtime:

| Runtime | Models run on a real clip |
| ------- | ------------------------- |
| CUDA | `fast`, `realesrgan`, `purevision` |
| WebGPU | `fast` only, between Vulkan Video decode and encode, with no host copy |

`--from-upscaled` also ran on WebGPU, with Vulkan Video decode and a CPU encoder.

## Not supported yet

- VA-API.
- Vulkan Video on AMD or Intel GPUs. It is untested there.
- CPU H.264 above Baseline.
- CPU HEVC encode. It is not planned.
- VP9 encode, on any path.
- Vulkan Video AV1 encode. It is written and has never run on a GPU.
- Fragmented MP4 input, HDR, interlaced video, and a filter graph.

### Roadmap

None of these works yet.

| Area            | Planned                                                                     |
| --------------- | --------------------------------------------------------------------------- |
| Hardware codecs | VA-API                                                                      |
| Software codecs | H.264 above Baseline                                                        |
| Containers      | Fragmented MP4 input, through containr                                      |
| Colour          | BT.2020 matrices, HDR10 and HLG, tone mapping                               |
| Filters         | A filter graph: scaling, cropping, deinterlacing, frame-rate conversion, denoising, and deblocking |
| Models          | Temporal restoration, frame interpolation, and any vision model boostr runs |
| Distribution    | Multi-GPU processing over [nexar](https://github.com/ml-rust/nexar)         |

## Install

```bash
cargo add vidr --features dav1d,mp4
```

The `dav1d` feature needs libdav1d 1.3.0 or newer, found through pkg-config.

Minimum Rust version: 1.95.

## Quick start

`crates/vidr/examples/hevc_decode.rs` decodes an HEVC file on the CPU. Run it:

```bash
cargo run --release -p vidr --features cpu-hevc --example hevc_decode -- FILE.h265
```

`FILE.h265` is an Annex-B HEVC stream, Main or Main 10 at 4:2:0. The example prints the frame count:

```text
25 frames from FILE.h265
```

## Project layout

| Crate                                   | What it holds                                                                                           |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| [`vidr`](crates/vidr)                   | Video codecs, hardware decode and encode, frame timing, and colour management                           |
| [`vidr-remaster`](crates/vidr-remaster) | The `vidr-remaster` command: restoration, upscaling, scaling, and encoding of a whole video in one pass |

## vidr-remaster

`vidr-remaster` runs a restoration model over a whole video, resizes it, and encodes it. One command writes one output file.

### Build

Cargo fetches numr, boostr, containr, and rav1r from crates.io.

Pick the row for your machine, then run `cargo build --release -p vidr-remaster --features <features>`.

| Machine | `<features>` | Also needs | Reads |
| ------- | ------------ | ---------- | ----- |
| NVIDIA GPU | `cuda,mp4,mkv,model-srvgg,model-rrdb` | `nvcc` at build time, and a driver that supports NVENC API 13.1 | H.264, HEVC, VP9, AV1 |
| GPU with Vulkan Video | `vulkan-video,mp4,mkv,model-srvgg,model-rrdb` | A driver that supports the Vulkan video extensions. Tested on NVIDIA only | H.264, HEVC, VP9, AV1 |
| No GPU | `mp4,dav1d,cpu-av1,model-srvgg` | libdav1d 1.3.0 or newer, through pkg-config | AV1 only |

- **CPU codecs.** A CPU build holds only the codecs its features name. Add `cpu-hevc`, `cpu-vp9`, or `openh264` to read other codecs. See [CPU codecs](#cpu-codecs).
- **Mixed builds.** Add a CPU codec to a hardware build to cover what the hardware lacks.
- **Incomplete builds.** A build with no container, decoder, or encoder reports which features it lacks.

### Run

Put the model directories under one root, named as in [Models](#models). Pass the root with `--models-dir`, or set `VIDR_MODELS_DIR`.

```bash
# NVIDIA or Vulkan Video build
target/release/vidr-remaster clip.mp4 --model fast --models-dir ~/models

# No-GPU build: no CPU HEVC encoder exists, so pass --codec av1
target/release/vidr-remaster clip.mp4 --model fast --models-dir ~/models --codec av1
```

The default `--model` is `purevision`. Its licence forbids commercial use. See [Models](#models).

The run writes `clip.fast.restored.mp4` beside the source. The output keeps the source's:

- frame count and frame timing
- last-frame duration
- rotation and aspect ratio

Audio, text subtitles, chapters, and metadata copy over with no transcode.

vidr decodes the output and compares it with the source. On a mismatch, the run stops with an error and writes no output.

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
| `--encoder`                   | `auto`, `nvenc`, `vulkan`, `cpu`                          | `auto`                          |
| `--openh264-library`          | Path to Cisco's OpenH264 binary, with `openh264-cisco`    | None                            |
| `--resolution`                | `auto`, `original`, `2x`, `480`, `720`, `1080`, `1440`, `2160`, `4320` | `auto`             |
| `--overwrite`                 | Replaces an existing output                               | Off                             |
| `--from-upscaled`             | Resizes this already-enhanced video, with no model pass   | None                            |
| `--force-ai`                  | Runs the model even when a matching 2x output exists      | Off                             |
| `--notices`                   | Prints the licence notices of this build                  | Off                             |

`--resolution` sets the shorter edge of the output:

- `auto` picks the largest standard edge at or below the source's.
- The shorter edge never falls below 480.

### Reusing an earlier output

Without `--force-ai` or `--from-upscaled`, a run looks beside the source for an earlier model output. It checks `<stem>.<model>.2x` and `<stem>.<model>.restored`, as `.mp4` or `.mkv`.

Each output has a `.upscale.json` sidecar file beside it. The sidecar records how vidr made the output.

| Earlier output | What the run does |
| -------------- | ----------------- |
| Exactly twice the source's size, with a matching sidecar | Uses it and skips the model pass |
| Sidecar missing or not matching | Stops with an error that names both flags |
| Sidecar names another encoder | Stops with an error that names both flags |

### Choosing runtime, decoder, and encoder

| Runtime | Decoders | Encoders |
| ------- | -------- | -------- |
| `cuda` | NVDEC, CPU | NVENC, CPU |
| `wgpu` | Vulkan Video for H.264, HEVC, VP9, and AV1, CPU | Vulkan Video for H.264, HEVC, and AV1, CPU |
| `cpu` | CPU | CPU |

How `auto` picks:

- **Order.** CUDA, then WebGPU, then the CPU. A hardware codec comes before a CPU codec.
- **Missing library or device.** `auto` moves to the next choice and records the reason in the report.
- **Stream the hardware does not support.** `auto` moves to the next choice. Examples: H.264 slice groups, data partitioning, a `frame_num` gap, field coding, and bit depth.
- **Stalled GPU queue.** The run stops with an error that names the stall.

Explicit choices:

- **Named hardware codec.** `--decoder nvdec`, `--decoder vulkan`, `--encoder nvenc`, or `--encoder vulkan` never falls back. The error names what is missing.
- **Decoder and encoder pair not in the table.** The run stops with an error. The error names both sides and a pair that works.

### Output check

The output check decodes the output with the first decoder available from this list:

1. The run's decoder, when it reads the output codec.
2. The runtime's hardware decoder.
3. The CPU decoder of that codec.

With no such decoder, the run stops before any work.

### Frame copies and model precision

| Runtime | Frame copies | Model precision |
| ------- | ------------ | --------------- |
| `cuda` | No host copy with NVDEC and NVENC | F16 |
| `wgpu` | One GPU-to-GPU copy per plane with Vulkan Video, no host copy | F32 |
| `cpu` | One host copy per frame | F32 |

A CPU codec on any runtime copies each frame through host memory. Each run prints its frame copies on one line of stdout.

### Run report

The run writes `report.json` under `$XDG_STATE_HOME/vidr-remaster/upscale-<run id>/`. Without `XDG_STATE_HOME`, it uses `~/.local/state`. Every run that passes the input checks writes one.

The report's `frame_path` field holds:

- the runtime
- each codec, and the path frames take through it
- the decoder the output check used
- the host copies of the encode pass and of the scans
- why vidr picked each part

### Models

Each directory under the model root holds one `.pth` file.

| `--model`    | Directory              | Network         | Licence of the weights | Cargo feature |
| ------------ | ---------------------- | --------------- | ---------------------- | ------------- |
| `fast`       | `realesr-general-x4v3` | SRVGGNetCompact | BSD-3-Clause           | `model-srvgg` |
| `realesrgan` | `RealESRGAN_x2plus`    | RRDBNet         | BSD-3-Clause           | `model-rrdb`  |
| `purevision` | `2x_PureVision`        | RRDBNet         | CC-BY-NC-SA-4.0        | `model-rrdb`  |

- **Commercial use.** The `purevision` licence forbids commercial use and requires share-alike. It is the default, so pass `--model` for commercial work. The run report repeats this notice.
- **Missing model.** The error names the expected path and the download URL. `vidr-remaster` downloads nothing.

### Exit codes

| Exit code | Meaning                                                                                       |
| --------- | --------------------------------------------------------------------------------------------- |
| 0         | The output and its sidecar are in place. If the report cannot be written, the run prints a warning |
| 1         | An error. This includes an output whose sidecar was not written. That error names both paths |
| 130       | A signal stopped the run. The temporary output is removed                                     |

Stopping a run:

- The first Ctrl-C, SIGTERM, or SIGHUP stops the run at the next frame.
- A second signal removes the temporary output and ends the process.

### Unsupported input and errors

Each error below stops the run with exit code 1 and writes no output.

**Source video**

| Input | Why vidr stops | What to do |
| ----- | -------------- | ---------- |
| HDR, in the bitstream or the container tags | vidr handles SDR colour only | Use an SDR source, or tone-map it first |
| Interlaced or MBAFF-coded frames | The frame transform needs progressive frames | Deinterlace the source first |
| Fragmented MP4 | containr does not read it | Run `ffmpeg -i <input> -c copy <output.mp4>` |
| A change of size, pixel format, or colour inside the video | A run uses one colour conversion and one encoder session | Split the source at the change |
| Timestamps (pts) that do not increase, or a frame count that differs from the packet count | The output check needs one frame per packet, in order | Repair or remux the source |
| A Matroska `ContentEncoding` other than header stripping or zlib | vidr cannot undo that encoding | Remux or re-encode the source. Decrypt an encrypted source first |
| More than one MP4 sample description | A decoder takes one config record | Split or transcode the source |
| H.264 above Baseline, on OpenH264 | OpenH264's B-slice output differs from the reference decoder | Use NVDEC or Vulkan Video, or re-encode as Baseline |

**Output**

| Input | Why vidr stops | What to do |
| ----- | -------------- | ---------- |
| A 10-bit source with `--codec h264` | OpenH264 and Vulkan Video encode H.264 at 8 bits only. NVENC stops when the GPU lacks 10-bit support | Pass `--codec hevc`, or `--codec av1` on the CPU |
| HEVC with a CPU encoder | vidr has no CPU HEVC encoder, and none is planned | Pass `--codec av1` |
| Attachments or ASS subtitles into `.mp4` | MP4 cannot hold them | Write `.mkv` |
| An existing output file | A run replaces a file only on request | Pass `--overwrite`, or choose another path |
| An earlier output from another encoder | Its sidecar names another encoder | Pass `--from-upscaled` or `--force-ai` |
| A codec, runtime, or `--openh264-library` missing from the build | The build lacks that feature | Rebuild with the feature the error names |

**Vulkan Video encode**

| Input | Why vidr stops | What to do |
| ----- | -------------- | ---------- |
| `--runtime cuda`, or a pair with NVDEC or NVENC | Vulkan Video frames live in WebGPU tensors | Pass `--runtime wgpu`, with Vulkan Video or a CPU codec on each side |
| B-frames | The encoder uses only the previous frame as a reference | Pick another encoder |
| An odd frame size | H.264 and HEVC crop 4:2:0 frames in steps of two pixels, so they cannot state an odd size | Scale or pad to an even size, or pass `--encoder cpu --codec av1` with `cpu-av1` |
| A frame size outside the driver's range | The driver sets the supported range | Scale the frames, or pick another encoder |
| AV1, on a driver without `VK_KHR_video_encode_av1` | The driver does not support AV1 encode | Pass `--encoder cpu` with `cpu-av1`, or `--codec hevc` |

**Vulkan Video decode**

Another decoder reads every stream in this table. The last column lists other fixes.

| Input | Why vidr stops | Other fix |
| ----- | -------------- | --------- |
| `--runtime cuda`, or a pair with NVDEC or NVENC | Vulkan Video frames live in WebGPU tensors | Pass `--runtime wgpu` |
| H.264 slice groups or data partitioning | Vulkan Video does not support them | Re-encode |
| An H.264 `frame_num` gap | vidr does not fill in missing frames | Re-encode |
| An H.264 IDR picture with `no_output_of_prior_pics_flag` set to 1, or with a new size or `max_dec_frame_buffering` | The H.264 standard (ITU-T H.264, C.4.4) drops frames waiting for output here, so output frames no longer match input packets | Re-encode |
| An HEVC picture that refers to a reference picture the decoder has not decoded | Vulkan Video holds a reference picture only after decoding it | Start at an earlier random-access (IRAP) picture |
| A VP9 frame that grows past the decode session's size and keeps its references | A new session holds none of the references | Call `VulkanDecoder::with_max_coded_size` in library code |
| An AV1 frame smaller than the driver's minimum size | The driver does not support it | None |
| AV1 film grain, on a driver without a `filmGrainSupport` profile | The driver does not support it | None |
| An AV1 frame that reads a reference its `ref_order_hint` marks as lost | No stored picture can replace the lost frame | Start at a key frame |
| AV1 in Annex B framing | vidr reads only the low-overhead OBU format | Run `ffmpeg -i INPUT -c copy -f obu OUTPUT` |
| AV1 large-scale tile coding | vidr does not decode tile lists | None |

## CPU codecs

A CPU codec works in host memory, so each frame takes one copy through it. The run report states the copy. In library code, `FrameRoute` on the decoder or encoder declares it.

Every feature here is off by default. `vidr-remaster` has a feature of the same name for each codec. Select a CPU codec with `--decoder cpu`, `--encoder cpu`, or `auto`.

| Codec | Feature | Direction | Bit depth, chroma | Crate | Licence |
| ----- | ------- | --------- | ----------------- | ----- | ------- |
| AV1 | `dav1d` | Decode | 8, 10, 4:2:0 | libdav1d through `dav1d` | BSD-2-Clause |
| AV1 | `cpu-av1`, `cpu-av1-asm` | Encode | 8, 10, 4:2:0 | `rav1r` | BSD-2-Clause |
| VP9 | `cpu-vp9` | Decode | 8, 10, 4:2:0 | `rusty_vp9` | Apache-2.0 |
| HEVC | `cpu-hevc` | Decode | 8, 10, 4:2:0 | This crate | Apache-2.0 |
| H.264 | `openh264`, `openh264-cisco` | Decode, encode | 8, 4:2:0 | OpenH264 through `openh264` | BSD-2-Clause |

- **`cpu-av1-asm`.** Needs NASM 2.15 or newer on x86_64, or a C compiler on aarch64.
- **`rusty_vp9`.** vidr pins it to one release. [CONTRIBUTING.md](CONTRIBUTING.md#conformance) describes the libvpx conformance run.

### Licences and patents for distributed builds

Building vidr for your own use needs nothing here. If you distribute a program built with these features, each row lists what you must do.

| Feature | What a distributor does |
| ------- | ----------------------- |
| `cpu-av1`, `cpu-av1-asm` | Include the BSD-2-Clause licence text and the Alliance for Open Media Patent License 1.0 in your documentation or legal notices. rav1r, a fork of rav1e, carries both. The patent licence ends for a licensee who sues over an AV1 implementation, with two stated exceptions. rav1r's `PATENTS` file holds its full text |
| `dav1d` | Include the BSD-2-Clause licence text. AV1 also carries patent terms. Check them before you ship |
| `cpu-vp9` | Include the Apache-2.0 licence text. VP9 carries third-party patent pools, and vidr holds no patent licence. Whoever sells a product with `cpu-vp9` answers for royalties. Google's VP9 patent grant covers only Google's libvpx, not `rusty_vp9` |
| `cpu-hevc` | Include the Apache-2.0 licence text. HEVC carries third-party patent pools, and vidr holds no patent licence. Whoever sells a product with `cpu-hevc` answers for royalties |
| `openh264` | Include the BSD-2-Clause licence text. OpenH264 built from source has no Cisco patent licence |
| `openh264-cisco` | This feature loads Cisco's own OpenH264 binary from the path in `--openh264-library`. Cisco's patent licence covers that binary only |

`cpu-hevc` and `cpu-vp9` are off by default, like every CPU codec.

`vidr-remaster --notices` prints the licence text of each codec in the build. `crates/vidr-remaster/notices/` holds each text unchanged, and the crate package includes that directory.

## Related projects

- [numr](https://github.com/ml-rust/numr): tensors and kernels
- [boostr](https://github.com/ml-rust/boostr): model networks
- [voxr](https://github.com/ml-rust/voxr): audio
- [containr](https://github.com/ml-rust/containr): MP4, MOV, Matroska, and WebM
- [rav1r](https://github.com/ml-rust/rav1r): AV1 encoder
- [nexar](https://github.com/ml-rust/nexar): distributed runtime

## Contributing

Contributions are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) lists the conventions and checks.

Questions and design discussion happen on [Discord](https://discord.gg/jBhFk9kHPg).

## Licence

Licensed under Apache-2.0. See [LICENSE](LICENSE).
