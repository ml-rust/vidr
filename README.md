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

vidr is a Cargo workspace for video on top of [numr](https://github.com/ml-rust/numr) tensors and [boostr](https://github.com/ml-rust/boostr) models. A frame is a numr tensor, so a codec, a filter, and a neural network all read and write the same buffer. [voxr](https://github.com/ml-rust/voxr) is its audio counterpart. Together they cover the work an application hands to the `ffmpeg` command line and to an inference engine today, as libraries it links.

```
decode (hardware or software) → filters and models on the GPU → encode → mux
```

## Status

In development. No release exists yet. The frame timing model, output checks, and the H.264, HEVC, and AV1 bitstream filters work today. Nothing in [Scope](#scope) works yet. [AGENTS.md](AGENTS.md) holds the implemented-vs-roadmap table.

## Crates

| Crate                                   | What it holds                                                                                           |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| [`vidr`](crates/vidr)                   | Video codecs, hardware decode and encode, frame timing, colour management, and the filter graph         |
| [`vidr-remaster`](crates/vidr-remaster) | The `vidr-remaster` command: restoration, upscaling, scaling, and encoding of a whole video in one pass |

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
