<div align="center">

# vidr

<h3>Video in Rust. A library first, with GPU and CPU paths.</h3>

<p>
Decode, run models, and encode video in one process. On a hardware path, frames stay in GPU memory. Every codec, container, and model is a Cargo feature, so an application links only what it uses. Model weights load at run time from a directory, and no weights are compiled in.
</p>

<p>
  <a href="#status"><strong>Status</strong></a>
  ·
  <a href="#build"><strong>Build</strong></a>
  ·
  <a href="#run"><strong>Run</strong></a>
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

vidr is a video library for Rust: decode, encode, colour handling, and neural-network frame processing, built on [numr](https://github.com/ml-rust/numr) tensors and [boostr](https://github.com/ml-rust/boostr) models. A frame is a numr tensor. A codec and a neural network both read and write that buffer. It needs no Python.

[voxr](https://github.com/ml-rust/voxr) is the audio counterpart. Together they target the work an application hands to the `ffmpeg` command line and an inference engine. [Scope](#scope) lists what is still planned.

```
decode (hardware or CPU) → models → encode → mux
```

The library is the main product. [`vidr-remaster`](#vidr-remaster), a command that restores and upscales a whole video, is its first use case.

### Where the work happens

- Hardware codecs: NVDEC and NVENC on NVIDIA, and Vulkan Video, through numr's CUDA and WebGPU backends.
- Native libraries: CPU H.264 wraps OpenH264, and CPU AV1 decode wraps libdav1d. Both are C or C++ code that Rust calls.
- Rust: CPU HEVC decode, CPU VP9 decode, and CPU AV1 encode (through [rav1r](https://github.com/ml-rust/rav1r)) run in Rust.
- Containers: MP4, MOV, Matroska, and WebM go through [containr](https://github.com/ml-rust/containr).

### Not supported yet

- VA-API, and any AMD or Intel GPU run of Vulkan Video.
- H.264 on the CPU above Baseline, and any CPU HEVC encode.
- VP9 encode, on any path.
- Vulkan Video AV1 encode. It is written and has never run on a GPU.
- Fragmented MP4 input, HDR, interlaced video, and a filter graph.

[Scope](#scope) and [Refusals](#refusals) hold the full lists.

### Install

```bash
cargo add vidr --features dav1d,mp4
```

`dav1d` needs libdav1d 1.3.0 or newer, found through pkg-config. For a runnable library example, see `crates/vidr/examples/hevc_decode.rs`. Run it with `cargo run --release -p vidr --features cpu-hevc --example hevc_decode -- FILE.h265`.

## Status

In development. See [Build](#build).

### Decode

All paths are 4:2:0.

| Codec | NVIDIA (`nvdec`) | Vulkan Video (`vulkan-video`) | CPU |
| ----- | ---------------- | ----------------------------- | --- |
| H.264 | 8 or 10 bits | 8 bits. Baseline, Main, High | 8 bits, Baseline only (`openh264`, `openh264-cisco`) |
| HEVC | 8 or 10 bits | 8 or 10 bits. Main, Main 10, Main Still Picture | 8 or 10 bits. Main, Main 10 (`cpu-hevc`) |
| VP9 | 8 or 10 bits | 8 or 10 bits. Profiles 0 and 2 | 8 or 10 bits. Profiles 0 and 2 (`cpu-vp9`) |
| AV1 | 8 or 10 bits | 8 or 10 bits. Main | 8 or 10 bits (`dav1d`) |

### Encode

All paths are 4:2:0.

| Codec | NVIDIA (`nvenc`) | Vulkan Video (`vulkan-video`) | CPU |
| ----- | ---------------- | ----------------------------- | --- |
| H.264 | 8 bits. 10 bits on a GPU that lists it | 8 bits, High | 8 bits, Constrained Baseline (`openh264`, `openh264-cisco`) |
| HEVC | 8 bits. 10 bits on a GPU that lists it | Main. Main 10 where the driver lists it | None, by decision |
| VP9 | None | None | None |
| AV1 | 8 bits. 10 bits on a GPU that lists it | Never run. Refused by name with no `VK_KHR_video_encode_av1` | 8 or 10 bits (`cpu-av1`, `cpu-av1-asm`) |

Vulkan Video encode uses one previous frame as the reference, and an IDR picture at `gop_length`. It refuses B-frames by name.

### What has run

All of this ran on one NVIDIA GPU with NVIDIA's driver. No AMD or Intel GPU has run any of it.

| Path | Checked with |
| ---- | ------------ |
| NVDEC, NVENC | A real clip with all three models. No test covers NVENC AV1 |
| Vulkan Video decode | A real clip, plus the HEVC_v1, libvpx VP9, and libaom AV1 test vectors, with no decode error |
| Vulkan Video encode | A real clip, H.264 and HEVC |
| `dav1d` | The libaom AV1 test vectors |
| `cpu-vp9` | The libvpx VP9 test vectors |
| `cpu-hevc` | The HEVC_v1 test vectors |
| `cpu-av1`, `openh264` | Tests in this repository, with no conformance set |

Limits of the Vulkan Video runs:

- Streams past a driver limit, and AV1 monochrome, are refused by name.
- Only one driver mode has run: reference and output pictures sharing one image.
- No validation layer has run.

| Model | CUDA, per 1080p frame | WebGPU, per 1080p frame |
| ----- | --------------------- | ----------------------- |
| `fast` | About 0.22 s | About 4 s, with no host copy |
| `realesrgan` | Ran, not timed here | Has not run |
| `purevision` | Ran, not timed here | Has not run |

Times are wall-clock. `--from-upscaled` also ran on WebGPU, with Vulkan decode and a CPU encoder.

### Also working

| Capability | Feature |
| ---------- | ------- |
| Frame timing model and output checks | None |
| Bitstream filters | `h264`, `hevc`, `av1` |
| Conversion between NV12 or P010 planes and float RGB | `frame` |
| SDR colour policy | `decode` |
| MP4, MOV, Matroska, and WebM input | `mp4`, `mkv` |
| One-pass MP4 and MKV output, with other streams copied and checked | `mp4`, `mkv` |
| Threaded decode, transform, and encode pump | `pipeline` |
| The `vidr-remaster` command | See [Build](#build) |

[AGENTS.md](AGENTS.md) holds the full implemented-vs-roadmap table.

## Crates

| Crate                                   | What it holds                                                                                           |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| [`vidr`](crates/vidr)                   | Video codecs, hardware decode and encode, frame timing, and colour management                           |
| [`vidr-remaster`](crates/vidr-remaster) | The `vidr-remaster` command: restoration, upscaling, scaling, and encoding of a whole video in one pass |

## vidr-remaster

`vidr-remaster` runs a restoration model over a whole video, resizes it, and encodes it. One command writes one output.

### Build

Cargo fetches numr, boostr, containr, and rav1r from crates.io.

Pick one row, then run `cargo build --release -p vidr-remaster --features <features>`.

| Machine | `<features>` | Also needs | Reads |
| ------- | ------------ | ---------- | ----- |
| NVIDIA GPU | `cuda,mp4,mkv,model-srvgg,model-rrdb` | `nvcc` at build time, and driver support for NVENC API 13.1 | H.264, HEVC, VP9, AV1 |
| GPU with Vulkan Video | `vulkan-video,mp4,mkv,model-srvgg,model-rrdb` | A driver that lists the video extensions. Tested on NVIDIA only | H.264, HEVC, VP9, AV1 |
| No GPU | `mp4,dav1d,cpu-av1,model-srvgg` | libdav1d 1.3.0 or newer, through pkg-config | AV1 only |

- **CPU codecs.** A CPU build holds only the codecs its features name. Add `cpu-hevc`, `cpu-vp9`, or `openh264` to read other sources. See [CPU codecs](#cpu-codecs).
- **Mixed builds.** Add a CPU codec to a hardware build to cover what the hardware leaves out.
- **Incomplete builds.** A build with no container, decoder, or encoder reports which features it lacks.

### Run

Put the model directories under one root, named as in [Models](#models). Pass the root with `--models-dir`, or set `VIDR_MODELS_DIR`.

```bash
# NVIDIA or Vulkan Video build
target/release/vidr-remaster clip.mp4 --model fast --models-dir ~/models

# No-GPU build: no CPU HEVC encoder exists, so pass --codec av1
target/release/vidr-remaster clip.mp4 --model fast --models-dir ~/models --codec av1
```

The default `--model` is `purevision`, and its licence forbids commercial use. See [Models](#models).

The output is `clip.fast.restored.mp4` beside the source. It keeps the source's frame count, frame timing, last-frame duration, rotation, and aspect ratio. Audio, text subtitles, chapters, and metadata copy with no transcode. A mismatch in the output check is an error, and no output publishes.

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
| `--openh264-library`          | Cisco's OpenH264 binary, with `openh264-cisco`            | None                            |
| `--resolution`                | `auto`, `original`, `2x`, `480`, `720`, `1080`, `1440`, `2160`, `4320` | `auto`             |
| `--overwrite`                 | Replaces an existing output                               | Off                             |
| `--from-upscaled`             | Resizes this enhanced video, with no model pass           | None                            |
| `--force-ai`                  | Runs the model even when a matching 2x output exists      | Off                             |
| `--notices`                   | Prints the licence notices of this build                  | Off                             |

`--resolution` sets the shorter edge. `auto` takes the largest standard edge at or below the source's. The shorter edge never falls below 480.

### Reusing an earlier output

Without `--force-ai` or `--from-upscaled`, a run looks beside the source for `<stem>.<model>.2x` and `<stem>.<model>.restored`, `.mp4` or `.mkv`.

| Candidate | Result |
| --------- | ------ |
| Exactly twice the source's size, with a matching `.upscale.json` sidecar | Replaces the model pass |
| Missing or mismatched sidecar | Stops the run. The error names both flags |
| Sidecar names another encoder | Stops the run. The error names both flags |

### Execution

| Runtime | Decoders | Encoders |
| ------- | -------- | -------- |
| `cuda` | NVDEC, CPU | NVENC, CPU |
| `wgpu` | Vulkan Video for H.264, HEVC, VP9, and AV1, CPU | Vulkan Video for H.264, HEVC, and AV1, CPU |
| `cpu` | CPU | CPU |

How `auto` chooses:

- **Order.** CUDA, then WebGPU, then the CPU. A hardware codec comes before a CPU codec.
- **Missing library or device.** `auto` falls through, with the reason in the report.
- **Hardware refusal.** `auto` falls through. This covers H.264 slice groups, data partitioning, a `frame_num` gap, field coding, and bit depth.
- **Stalled GPU queue.** The run ends. The error names the stall.

Other choices:

- **Explicit hardware codec.** `--decoder nvdec`, `--decoder vulkan`, `--encoder nvenc`, or `--encoder vulkan` never falls through. The error names what is missing.
- **Pair outside the table.** The run is refused. The error names both sides and the pair that works.

The output check decodes the output, in this order:

1. The run's decoder, when it reads the output codec.
2. The runtime's hardware decoder.
3. The CPU decoder of that codec.

With no such decoder, the run stops before any work.

| Runtime | Frame copies | Model precision |
| ------- | ------------ | --------------- |
| `cuda` | No host copy on NVDEC or NVENC | F16 |
| `wgpu` | One device-to-device copy per plane on Vulkan Video, with no host copy | F32 |
| `cpu` | One host copy per frame | F32 |

A CPU codec on any runtime copies each frame through host memory. A run states its copies on one line of stdout.

`report.json` holds `frame_path`:

- the runtime
- each codec and route
- the output check's decoder
- the host copies of the encode pass and of the scans
- why each part was chosen

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

**The source**

| Refused | Why | Next action |
| ------- | --- | ----------- |
| HDR, in the bitstream or the container tags | The colour policy covers SDR only | Use an SDR source, or tone-map it first |
| Interlaced or MBAFF-coded frames | The frame transform needs progressive frames | Deinterlace the source first |
| Fragmented MP4 | containr does not read it | Run `ffmpeg -i <input> -c copy <output.mp4>` |
| A size, pixel format, or colour change inside the video | A run holds one conversion and one encoder session | Split the source at the change |
| Nonincreasing pts, or a frame count unlike the packets | The output check needs one frame per packet, in order | Repair or remux the source |
| A Matroska crop or `ContentEncoding` | NVDEC applies neither | Remux or re-encode the source |
| More than one MP4 sample description | A decoder takes one config record | Split or transcode the source |
| An H.264 source that is not Baseline, on OpenH264 | OpenH264's B-slice output differs from the reference | Use NVDEC or Vulkan Video, or re-encode as Baseline |

**The output**

| Refused | Why | Next action |
| ------- | --- | ----------- |
| A 10-bit source with `--codec h264` | OpenH264 and Vulkan Video encode H.264 at 8 bits. NVENC refuses it when the GPU lists no 10-bit support | Pass `--codec hevc`, or `--codec av1` on the CPU |
| HEVC with a CPU encoder | No CPU HEVC encoder exists here, by decision | Pass `--codec av1` |
| Attachments or ASS subtitles into `.mp4` | MP4 has no form for them | Write `.mkv` |
| An existing output | A run replaces a file only on request | Pass `--overwrite`, or choose another path |
| A candidate from another encoder | The sidecar names another encoder | Pass `--from-upscaled` or `--force-ai` |
| A codec, a runtime, or `--openh264-library` not in the build | The feature is off | Rebuild with the feature the error names |

**Vulkan Video encode**

| Refused | Why | Next action |
| ------- | --- | ----------- |
| `--runtime cuda`, or a pairing with NVDEC or NVENC | Its frames live on WebGPU tensors | Pass `--runtime wgpu`, with Vulkan Video or a CPU codec on each side |
| B-frames | One previous frame is the reference | Pick another encoder |
| An odd size | 4:2:0 needs even sides | Scale the frames, or pick another encoder |
| A size outside the driver's range | The driver sets the range | Scale the frames, or pick another encoder |
| AV1, on a driver with no `VK_KHR_video_encode_av1` | The driver lists no such queue operation | Pass `--encoder cpu` with `cpu-av1`, or `--codec hevc` |

**Vulkan Video decode**

Use another decoder for every row. A further option appears where one exists.

| Refused | Why | Further option |
| ------- | --- | -------------- |
| `--runtime cuda`, or a pairing with NVDEC or NVENC | Its frames live on WebGPU tensors | Pass `--runtime wgpu` |
| H.264 slice groups or data partitioning | Vulkan Video has no form for them | Re-encode |
| An H.264 `frame_num` gap | vidr does not infer missing frames | Re-encode |
| An H.264 IDR picture with `no_output_of_prior_pics_flag` 1, or a new size or `max_dec_frame_buffering` | ITU-T H.264 C.4.4 removes waiting frames, so frames no longer match packets | Re-encode |
| An HEVC picture that binds a reference the DPB lacks | Vulkan Video activates a DPB slot only through a decode | Start at an earlier IRAP picture |
| A VP9 frame that grows past its session and keeps references | A new session holds none of the references | Call `VulkanDecoder::with_max_coded_size` in library code |
| An AV1 frame under the driver's smallest coded size | Each frame checks against what the driver lists | None |
| AV1 film grain with no `filmGrainSupport` profile | Each frame checks against what the driver lists | None |
| An AV1 frame that reads a reference its `ref_order_hint` states as lost | No slot can stand in for the lost frame | Start at a key frame |
| AV1 in Annex B framing | vidr reads the low-overhead format | Run `ffmpeg -i INPUT -c copy -f obu OUTPUT` |
| AV1 large-scale tile coding | vidr has no tile list decode | None |

### Results

| Exit code | Meaning                                                                                       |
| --------- | --------------------------------------------------------------------------------------------- |
| 0         | The output and its sidecar are in place. A report that does not write gives a warning         |
| 1         | A refusal, an error, or a final output whose sidecar did not land. The error names both paths |
| 130       | A signal stopped the run. The temporary output is gone                                         |

The first Ctrl-C, SIGTERM, or SIGHUP asks the run to stop at the next frame. A second one removes the temporary output and ends the process.

`report.json` goes under `$XDG_STATE_HOME/vidr-remaster/upscale-<run id>/`, else `~/.local/state`. Every run past validation writes one.

## CPU codecs

A CPU codec works in host memory, so each frame takes one copy through it. The run report states the copy. In library code, `FrameRoute` on the decoder or encoder declares it.

Every feature here is off by default. `vidr-remaster` has a feature of the same name for each codec. Select the codec with `--decoder cpu`, `--encoder cpu`, or `auto`.

| Codec | Feature | Direction | Bit depth, chroma | Crate | Licence |
| ----- | ------- | --------- | ----------------- | ----- | ------- |
| AV1 | `dav1d` | Decode | 8, 10, 4:2:0 | libdav1d through `dav1d` | BSD-2-Clause |
| AV1 | `cpu-av1`, `cpu-av1-asm` | Encode | 8, 10, 4:2:0 | `rav1r` | BSD-2-Clause |
| VP9 | `cpu-vp9` | Decode | 8, 10, 4:2:0 | `rusty_vp9` | Apache-2.0 |
| HEVC | `cpu-hevc` | Decode | 8, 10, 4:2:0 | This crate | Apache-2.0 |
| H.264 | `openh264`, `openh264-cisco` | Decode, encode | 8, 4:2:0 | OpenH264 through `openh264` | BSD-2-Clause |

`cpu-av1-asm` needs NASM 2.15 or newer on x86_64, or a C compiler on aarch64.

### Shipping a binary

These notes state what a binary distribution needs. They are not legal advice.

| Feature | Condition |
| ------- | --------- |
| `cpu-av1` | Reproduce the Alliance for Open Media Patent License 1.0 and the BSD licence text in documentation, legal notices, or other written materials |
| `dav1d` | AV1 has patent terms. Get legal review |
| `cpu-vp9` | VP9 has a patent pool. Get legal review |
| `cpu-hevc` | HEVC has patent pools. Get legal review |
| `openh264` | A source build gets no Cisco patent cover |
| `openh264-cisco` | Cisco's cover applies only to its own binary. This feature loads it from a path the caller supplies |

- **rav1r.** rav1r is a fork of rav1e, under the BSD-2-Clause licence and the Alliance for Open Media Patent License 1.0. rav1r's `PATENTS` file holds the patent licence. Section 1.3 ends it for a licensee that sues over an Implementation, with two stated exceptions.
- **Notices.** `vidr-remaster --notices` prints the text of each codec in the build. `crates/vidr-remaster/notices/` holds each text unchanged, and the crate package includes that directory.
- **`rusty_vp9`.** It stays pinned to one release. [CONTRIBUTING.md](CONTRIBUTING.md#conformance) holds the libvpx conformance run.

## Scope

These are planned, and none of them works yet.

| Area            | Planned                                                                     |
| --------------- | --------------------------------------------------------------------------- |
| Hardware codecs | VA-API                                                                      |
| Software codecs | H.264 above Baseline                                                        |
| Containers      | Fragmented MP4 input, through containr                                      |
| Colour          | BT.2020 matrices, HDR10 and HLG, tone mapping                               |
| Filters         | A filter graph: scaling, cropping, deinterlacing, frame-rate conversion, denoising, and deblocking |
| Models          | Temporal restoration, frame interpolation, and any vision model boostr runs |
| Distribution    | Multi-GPU processing over [nexar](https://github.com/ml-rust/nexar)         |

## Minimum Rust version

1.95.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions and checks.

Questions and design discussion happen on [Discord](https://discord.gg/jBhFk9kHPg).

## License

Licensed under Apache-2.0. See [LICENSE](LICENSE).
