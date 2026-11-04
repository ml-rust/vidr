# Contributing to vidr

## Before you start

Open an issue before you start anything non-trivial. Discuss design on [Discord](https://discord.gg/jBhFk9kHPg).

## Prerequisites

- Rust 1.95 or newer.
- `cargo-deny` and `shellcheck` for the checks.
- `libdav1d-dev` for `dav1d` rows. `nasm` for `cpu-av1-asm` rows. `ffmpeg` for the `vidr-remaster` CPU rows.
- Checkouts of numr, boostr, containr, and rav1r beside this repository.

Copy the config before any build:

```bash
cp .cargo/config.toml.example .cargo/config.toml
```

The patch is required, because `rav1r` is not on crates.io and cargo resolves optional dependencies for every feature set. CI fails to resolve `rav1r` until it is published.

## Build and test

Run these from the repository root.

```bash
cargo fmt --all --check
RUSTDOCFLAGS="-D warnings" cargo doc -p vidr --no-deps
RUSTDOCFLAGS="-D warnings" cargo doc -p vidr-remaster --no-deps
cargo deny --exclude-dev check
shellcheck scripts/ci/*.sh scripts/conformance/*.sh
```

The widest clippy rows:

```bash
cargo clippy -p vidr --features cpu-hevc,cpu-vp9,mkv --all-targets -- -D warnings
cargo clippy -p vidr --features cpu-av1,cpu-vp9,dav1d,mkv --all-targets -- -D warnings
cargo clippy -p vidr-remaster --features mp4,mkv,dav1d,cpu-vp9,cpu-hevc,openh264,openh264-cisco,cpu-av1,model-srvgg,model-rrdb,f16 --all-targets -- -D warnings
```

CI runs each feature alone as well. The `clippy` matrix in `.github/workflows/test.yml` lists every row. Add a row for each new feature that builds with no GPU SDK.

The CI test rows:

```bash
cargo test -p vidr --features h264,hevc,av1,vp9,frame,mp4,mkv,pipeline,cpu-hevc --no-fail-fast
cargo test -p vidr --features dav1d,openh264,openh264-cisco --no-fail-fast
cargo test -p vidr --features cpu-av1,cpu-vp9,dav1d,mkv --no-fail-fast
cargo test -p vidr --features cpu-vp9,pipeline --no-fail-fast
cargo test -p vidr-remaster --no-fail-fast
cargo test -p vidr-remaster --features mp4,mkv,h264,hevc,model-srvgg,model-rrdb,f16 --no-fail-fast
cargo test -p vidr-remaster --features mp4,mkv,dav1d,cpu-vp9,cpu-hevc,openh264,cpu-av1,model-srvgg,model-rrdb,f16 --no-fail-fast
```

- **Hardware.** CI has no GPU and no CUDA toolkit. Put every hardware path behind its own feature, such as `cuda` or `vulkan-video`. Both crates must build and test without those features.
- **Skipped tests.** A test that needs a GPU, a model checkpoint, or a media fixture skips with a notice when its environment variable is unset.
- **Weights and media.** They never enter the repository. Load them at run time from a directory.

## Conformance

The vector sets stay outside the repository. Their licence terms are unconfirmed, and nothing here redistributes them.

| Codec | Fetch | Run | Filter by name |
| ----- | ----- | --- | -------------- |
| VP9 | `scripts/conformance/fetch_vp9.sh DIR` | `VIDR_VP9_VECTORS=DIR cargo test -p vidr --features cpu-vp9,mkv libvpx_vectors -- --nocapture` | `VIDR_VP9_FILTER` |
| AV1 | `scripts/conformance/fetch_av1.sh DIR` | `VIDR_AV1_VECTORS=DIR cargo test -p vidr --features dav1d,mkv libaom_vectors -- --nocapture` | `VIDR_AV1_FILTER` |
| HEVC | `scripts/conformance/fetch_hevc.sh DIR` | `VIDR_HEVC_CONFORMANCE=DIR cargo test -p vidr --release --features cpu-hevc hevc_zips_decode -- --nocapture` | `VIDR_HEVC_ONLY` |

- **VP9.** The test compares every frame with libvpx's `.md5` file. ffmpeg remuxes the `.ivf` vectors.
- **AV1.** The test compares every frame with libaom's `.md5` file, with zero tolerance. A `.mkv` vector needs `mkv`.
- **HEVC.** The test checks each picture's hash SEI message and the zip's MD5 of the whole stream. It keeps no frame and needs `unzip`. Without `cpu-hevc`, the variable reads each zip's layout and decodes nothing.
- **Output.** Each test prints a pass and fail table by vector and names each error.
- **`VIDR_CONFORMANCE_JOBS`.** Sets how many VP9 or AV1 vectors run at once.
- **`VIDR_HEVC_REPORT`.** Names a file for one line per vector.
- **`VIDR_HEVC_THREADS`.** Sets the worker threads per decoder, 1 by default.

Vulkan Video and mutation tests read the same variables. Add the feature in the second column to the `cargo test -p vidr` line.

| Test | Feature | Needs |
| ---- | ------- | ----- |
| `libvpx_vectors_match_on_vulkan_video`, `vp9_vulkan_mutations_stay_safe` | `vulkan-video,mkv` | `VIDR_VP9_VECTORS`, a GPU |
| `libaom_vectors_match_on_vulkan_video`, `av1_vulkan_mutations_stay_safe` | `vulkan-video`, plus `mkv` for `.mkv` vectors | `VIDR_AV1_VECTORS`, a GPU |
| `hevc_v1_vectors_match_on_vulkan_video`, `hevc_vulkan_mutations_stay_safe` | `vulkan-video` | `VIDR_HEVC_CONFORMANCE`, a GPU |
| `hevc_mutations_stay_safe` | `cpu-hevc` | `VIDR_HEVC_CONFORMANCE` |

The HEVC mutation tests also read `VIDR_HEVC_ROBUST_SEEDS`, `VIDR_HEVC_ROBUST_DEADLINE_SECS`, and `VIDR_HEVC_ROBUST_TRACE`.

## Conventions

| Work | Goes in |
| ---- | ------- |
| Kernels | [numr](https://github.com/ml-rust/numr) |
| Model networks | [boostr](https://github.com/ml-rust/boostr) |
| Audio | [voxr](https://github.com/ml-rust/voxr) |
| Containers | [containr](https://github.com/ml-rust/containr) |
| Video codecs, frames, and filters | `vidr` |
| The command line | `vidr-remaster` |

- Make every codec, container, hardware API, and model a Cargo feature. Keep the default build minimal.
- Keep the MSRV at the `rust-version` in the manifests. CI checks it.

## Pull requests

- Add tests for every behavior change.
- Add a line to `CHANGELOG.md` for every user-visible change.
- Run the commands in [Build and test](#build-and-test) before you open the pull request.

## Releases

- vidr and vidr-remaster release together at one version.
- Each release needs a non-empty `## [X.Y.Z]` section in `CHANGELOG.md`.
- `.github/workflows/release-validate.yml` refuses a tag with no section. The GitHub Release body is that section.
- A pushed `v*` tag runs the gate. The manual `Release` workflow publishes.
