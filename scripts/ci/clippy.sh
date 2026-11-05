#!/usr/bin/env bash
# Run clippy on every feature set of one crate.
#
#   scripts/ci/clippy.sh vidr
#   scripts/ci/clippy.sh vidr-remaster
#
# Each feature gate compiles alone, so a gate that leans on another one's import
# fails here. `--all-targets` also checks the tests and examples each gate
# enables. Add a set for every new feature that builds with no GPU SDK.
#
# `dav1d` sets need libdav1d-dev. `cpu-av1-asm` sets need nasm.
#
# Every set runs. The run prints each failed set, then exits 1.

set -uo pipefail

crate="${1:-}"

case "$crate" in
    vidr)
        feature_sets=(
            ""
            "h264"
            "hevc"
            "av1"
            "vp9"
            "decode"
            "encode"
            "frame"
            "mp4"
            "mkv"
            "pipeline"
            "pipeline,mp4,mkv"
            "wgpu"
            "frame,wgpu"
            "dav1d"
            "openh264"
            "openh264-cisco"
            "dav1d,openh264,openh264-cisco"
            "cpu-av1"
            "cpu-av1-asm"
            "cpu-vp9"
            "cpu-hevc"
            "cpu-vp9,pipeline"
            "cpu-hevc,cpu-vp9,mkv"
            "cpu-av1,cpu-vp9,dav1d,mkv"
        )
        ;;
    vidr-remaster)
        feature_sets=(
            ""
            "mp4"
            "mkv"
            "h264"
            "hevc"
            "f16"
            "transform"
            "model-srvgg"
            "model-rrdb"
            "mp4,mkv,h264,hevc,model-srvgg,model-rrdb,f16"
            "wgpu"
            "dav1d"
            "cpu-vp9"
            "cpu-hevc"
            "openh264"
            "openh264-cisco"
            "cpu-av1"
            "cpu-av1-asm"
            "mp4,dav1d,cpu-av1"
            "mp4,openh264"
            "mp4,mkv,wgpu,dav1d,cpu-av1,model-srvgg,model-rrdb"
            "mp4,mkv,dav1d,cpu-vp9,cpu-hevc,openh264,openh264-cisco,cpu-av1,model-srvgg,model-rrdb,f16"
        )
        ;;
    *)
        echo "error: unknown crate '$crate': pass vidr or vidr-remaster" >&2
        exit 2
        ;;
esac

failed=()
for features in "${feature_sets[@]}"; do
    label="${features:-no features}"
    echo "::group::clippy $crate ($label)"
    cargo clippy -p "$crate" --features "$features" --all-targets -- -D warnings
    status=$?
    echo "::endgroup::"
    if [[ $status -ne 0 ]]; then
        failed+=("$label")
        echo "::error title=Clippy ($crate)::clippy failed on '$label'. Run: cargo clippy -p $crate --features \"$features\" --all-targets -- -D warnings"
    fi
done

echo "clippy $crate: $((${#feature_sets[@]} - ${#failed[@]})) of ${#feature_sets[@]} feature sets pass"
if [[ ${#failed[@]} -gt 0 ]]; then
    printf 'failed: %s\n' "${failed[@]}" >&2
    exit 1
fi
