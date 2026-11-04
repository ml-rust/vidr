#!/usr/bin/env bash
#
# Fetches the JCT-VC HEVC_v1 conformance zips to a directory you name, for the
# layout test in crates/vidr/tests/conformance/hevc.rs.
#
#   scripts/conformance/fetch_hevc.sh DIR [NAME_TEXT]
#
# NAME_TEXT fetches only the zips whose name holds that text. The site's index
# lists the zips, and each zip's own test checks it. The site states no hashes.
#
# The licence terms of this set are unconfirmed. Nothing here redistributes it.
# Keep it out of the repository and out of any release.

set -euo pipefail

readonly BASE_URL="https://www.itu.int/wftp3/av-arch/jctvc-site/bitstream_exchange/draft_conformance/HEVC_v1"
readonly CLIENT="conformance-fetch"

fail() {
  echo "fetch_hevc.sh: $1" >&2
  exit 1
}

[[ $# -ge 1 && $# -le 2 ]] || fail "expected DIR [NAME_TEXT], got $# arguments. Run: fetch_hevc.sh DIR [NAME_TEXT]"
for tool in curl unzip; do
  command -v "$tool" >/dev/null || fail "$tool is not on PATH. Install it, then retry."
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd -- "$script_dir/../.." && pwd -P)
mkdir -p -- "$1" || fail "cannot create $1. Name a directory you can write."
dest=$(cd -- "$1" && pwd -P)
case "$dest/" in
  "$repo"/*) fail "$dest is inside the repository at $repo. Name a directory outside it." ;;
esac
filter=${2:-}

echo "The licence terms of this set are unconfirmed. Do not redistribute it."

index=$(curl --fail --silent --show-error --location --retry 3 \
  --user-agent "$CLIENT" "$BASE_URL/") ||
  fail "the index at $BASE_URL/ does not download. Check the network, then retry."
names=$(grep -oiE 'href="[^"]+\.zip"' <<<"$index" | sed -E 's#.*/##; s/"$//' | sort -u) ||
  fail "the index at $BASE_URL/ lists no zip. The site changed, so update this script."

fetched=0
kept=0
failed=()
while read -r name; do
  [[ -z $filter || $name == *"$filter"* ]] || continue
  target="$dest/$name"
  if [[ -f $target ]] && unzip -tqq "$target" >/dev/null 2>&1; then
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
  if ! unzip -tqq "$partial" >/dev/null 2>&1; then
    rm -f -- "$partial"
    failed+=("$name: the zip does not test clean")
    continue
  fi
  mv -- "$partial" "$target"
  fetched=$((fetched + 1))
done <<<"$names"

echo "$fetched zips fetched, $kept already in $dest, ${#failed[@]} failed."
if [[ ${#failed[@]} -gt 0 ]]; then
  printf '%s\n' "${failed[@]}" >&2
  fail "rerun this script to retry the zips above."
fi
echo "Run the check with: VIDR_HEVC_CONFORMANCE=$dest cargo test -p vidr hevc_zips -- --nocapture"
