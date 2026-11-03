# Contributing to vidr

## Before you start

Open an issue before you start anything non-trivial. Discuss design on [Discord](https://discord.gg/jBhFk9kHPg).

## Build and test

Run these commands from the repository root. They match `.github/workflows/test.yml`.

```bash
cargo fmt --all --check
cargo clippy -p vidr --all-targets -- -D warnings
cargo clippy -p vidr-remaster --all-targets -- -D warnings
cargo test -p vidr --no-fail-fast
cargo test -p vidr-remaster --no-fail-fast
cargo deny --exclude-dev check
shellcheck scripts/ci/*.sh
```

CI has no GPU and no CUDA toolkit. Put every hardware path behind its own feature, such as `cuda`, `vulkan-video`, or `vaapi`. The crates must build and test without them.

Tests that need a GPU, a model checkpoint, or a media fixture skip with a notice when their environment variable is unset.

Model weights and media files never enter the repository. Load them at run time from a directory.

To build against sibling checkouts of numr and boostr, copy `.cargo/config.toml.example` to `.cargo/config.toml`. The CI builds resolve from crates.io.

## Conventions

- Put kernels in [numr](https://github.com/ml-rust/numr) and model networks in [boostr](https://github.com/ml-rust/boostr). Put audio in [voxr](https://github.com/ml-rust/voxr). Put video codecs, frames, and filters in `vidr`. Put the command line in `vidr-remaster`.
- Make every codec, container, hardware API, and model a Cargo feature. Keep the default build minimal.
- Keep the MSRV at the `rust-version` in the manifests. CI checks it.

## Pull requests

- Add tests for every behavior change.
- Add a line to `CHANGELOG.md` for every user-visible change.
- Run the commands in [Build and test](#build-and-test) before you open the pull request.
