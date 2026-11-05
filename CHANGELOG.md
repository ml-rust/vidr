# Changelog

All notable changes to this project are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-05

First release. `vidr` and `vidr-remaster` share one version.

### Added

`vidr` features:

- `nvdec`: decode H.264, HEVC, VP9, and AV1 on NVIDIA GPUs, at 8 or 10 bits.
- `nvenc`: encode H.264, HEVC, and AV1 on NVIDIA GPUs that list them.
- `vulkan-video`: decode H.264, HEVC, VP9, and AV1, and encode H.264 and HEVC, on Vulkan GPUs. It has run on NVIDIA only. Encode refuses B-frames.
- `dav1d`: decode AV1 on the CPU.
- `cpu-av1`: encode AV1 on the CPU.
- `openh264`, `openh264-cisco`: decode and encode H.264 Baseline on the CPU.
- `cpu-vp9`: decode VP9 on the CPU.
- `cpu-hevc`: decode HEVC Main and Main 10 on the CPU.
- `mp4`, `mkv`: read MP4, MOV, Matroska, and WebM, and write MP4 and MKV. Output copies audio, text subtitles, chapters, and metadata.
- `frame`: convert NV12 and P010 to RGB and back on the GPU.
- `pipeline`: run decode, transform, and encode on three threads.

`vidr-remaster` command:

- Models: `fast`, `realesrgan`, and `purevision`, on CUDA or the CPU. On WebGPU, only `fast` has run.
- Output check: the output keeps the source's frame count and timing, or the run fails.
- Records: each output gets a provenance sidecar, and each run writes `report.json`.
