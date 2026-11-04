#!/usr/bin/env bash
#
# Fetches libvpx's VP9 test vectors and per-frame .md5 files to a directory you
# name, for the conformance test in crates/vidr/tests/conformance/vp9.rs.
#
#   scripts/conformance/fetch_vp9.sh DIR [NAME_TEXT]
#
# NAME_TEXT fetches only the vectors whose name holds that text.
#
# vp9_vectors.sha1 lists the names and hashes. It comes from libvpx's
# test/test_vectors.cc and test/test-data.sha1. libvpx is BSD-licensed.
#
# The licence terms of the vectors themselves are unconfirmed. Nothing here
# redistributes them. Keep them out of the repository and out of any release.

set -euo pipefail

readonly BASE_URL="https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx"
readonly CLIENT="conformance-fetch"

fail() {
  echo "fetch_vp9.sh: $1" >&2
  exit 1
}

[[ $# -ge 1 && $# -le 2 ]] || fail "expected DIR [NAME_TEXT], got $# arguments. Run: fetch_vp9.sh DIR [NAME_TEXT]"
for tool in curl sha1sum; do
  command -v "$tool" >/dev/null || fail "$tool is not on PATH. Install it, then retry."
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd -- "$script_dir/../.." && pwd -P)
readonly manifest="$script_dir/vp9_vectors.sha1"
mkdir -p -- "$1" || fail "cannot create $1. Name a directory you can write."
dest=$(cd -- "$1" && pwd -P)
case "$dest/" in
  "$repo"/*) fail "$dest is inside the repository at $repo. Name a directory outside it." ;;
esac
filter=${2:-}

echo "The licence terms of these vectors are unconfirmed. Do not redistribute them."

fetched=0
kept=0
failed=()
while read -r hash name; do
  [[ -z $filter || $name == *"$filter"* ]] || continue
  target="$dest/$name"
  if [[ -f $target ]] && echo "$hash  $target" | sha1sum --check --status; then
    kept=$((kept + 1))
    continue
  fi
  partial="$target.partial"
  if ! curl --fail --silent --show-error --location --retry 3 \
    --user-agent "$CLIENT" --output "$partial" "$BASE_URL/$name"; then
    rm -f -- "$partial"
    failed+=("$name: the download fails")
    continue
  fi
  if ! echo "$hash  $partial" | sha1sum --check --status; then
    rm -f -- "$partial"
    failed+=("$name: the SHA-1 differs from $hash")
    continue
  fi
  mv -- "$partial" "$target"
  fetched=$((fetched + 1))
done <"$manifest"

echo "$fetched files fetched, $kept already in $dest, ${#failed[@]} failed."
if [[ ${#failed[@]} -gt 0 ]]; then
  printf '%s\n' "${failed[@]}" >&2
  fail "rerun this script to retry the files above."
fi
echo "Run the check with: VIDR_VP9_VECTORS=$dest cargo test -p vidr --features cpu-vp9,mkv libvpx_vectors -- --nocapture"
