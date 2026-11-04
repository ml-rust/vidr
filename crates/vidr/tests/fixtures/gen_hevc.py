#!/usr/bin/env python3
"""Generate the CPU HEVC decoder fixtures for the tests in `decode/cpu/hevc/stream_tests.rs`.

Run from any directory:

    python3 crates/vidr/tests/fixtures/gen_hevc.py [--out DIR] [--only NAME ...]

ffmpeg and the x265 command line tool, with 8-bit and 10-bit support, must be on
PATH. The script exits with a notice when either is missing. The default output
directory is `crates/vidr/tests/fixtures/hevc/`, which git ignores. Never commit
its contents.

For each stream NAME the script writes:

- `NAME.hevc`: the x265 Annex-B stream.
- `NAME.yuv`: ffmpeg's decode as `-f rawvideo`, in `yuv420p` or `yuv420p10le`.
  An inter stream that tests expect the decoder to refuse still gets a `.yuv`.

The tests read `VIDR_HEVC_FIXTURES` first, then the default directory.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

QUIET = ["-hide_banner", "-loglevel", "error", "-y"]
# All-intra pictures, one frame thread, and a stable lookahead.
INTRA = ["--keyint", "1", "--min-keyint", "1", "--frame-threads", "1"]
NO_FILTERS = ["--no-deblock", "--no-sao"]
NO_WPP = ["--no-wpp"]
# A moving test pattern with noise, so the stream uses many modes and sizes.
BUSY = "testsrc2=size={size}:rate=25,noise=alls=12:allf=t"
# Low contrast gradients, so large blocks and strong smoothing occur.
SMOOTH = "gradients=size={size}:rate=25:speed=0.01"

# (name, size, bit depth, frames, source, x265 arguments).
STREAMS = [
    # H8: no in-loop filters.
    ("intra_8", "128x96", 8, 3, BUSY, [*INTRA, *NO_FILTERS, *NO_WPP, "--crf", "30"]),
    ("intra_10", "128x96", 10, 3, BUSY, [*INTRA, *NO_FILTERS, *NO_WPP, "--crf", "30"]),
    ("cra_8", "128x96", 8, 3, BUSY, [*INTRA, *NO_FILTERS, *NO_WPP, "--crf", "30", "--cra-nal"]),
    ("window_8", "122x90", 8, 2, BUSY, [*INTRA, *NO_FILTERS, *NO_WPP, "--crf", "30"]),
    ("window_10", "122x90", 10, 2, BUSY, [*INTRA, *NO_FILTERS, *NO_WPP, "--crf", "30"]),
    ("ctu16_8", "96x64", 8, 2, BUSY,
     [*INTRA, *NO_FILTERS, *NO_WPP, "--crf", "30", "--ctu", "16"]),
    ("ctu32_min16_8", "128x96", 8, 2, BUSY,
     [*INTRA, *NO_FILTERS, *NO_WPP, "--crf", "30", "--ctu", "32", "--min-cu-size", "16"]),
    ("wavefront_8", "256x192", 8, 1, BUSY,
     [*INTRA, *NO_FILTERS, "--crf", "34", "--wpp", "--ctu", "16"]),
    ("slices_8", "256x192", 8, 2, BUSY,
     [*INTRA, *NO_FILTERS, "--crf", "34", "--wpp", "--slices", "3", "--ctu", "16"]),
    ("deep_tu_8", "128x96", 8, 2, BUSY,
     [*INTRA, *NO_FILTERS, *NO_WPP, "--crf", "26", "--tu-intra-depth", "4", "--rd", "6"]),
    ("cu_qp_8", "128x96", 8, 2, BUSY,
     [*INTRA, *NO_FILTERS, *NO_WPP, "--crf", "28", "--aq-mode", "2", "--aq-strength", "2",
      "--qg-size", "8"]),
    ("chroma_offsets_8", "96x64", 8, 2, BUSY,
     [*INTRA, *NO_FILTERS, *NO_WPP, "--crf", "30", "--cbqpoffs", "5", "--crqpoffs", "-5"]),
    ("tskip_8", "96x64", 8, 2, BUSY,
     [*INTRA, *NO_FILTERS, *NO_WPP, "--crf", "24", "--tskip", "--rd", "6"]),
    ("lossless_8", "64x64", 8, 1, BUSY, [*INTRA, *NO_WPP, "--lossless"]),
    ("cu_lossless_8", "64x64", 8, 1, BUSY, [*INTRA, *NO_WPP, "--cu-lossless", "--crf", "30"]),
    ("smooth_8", "300x280", 8, 1, SMOOTH, [*INTRA, *NO_WPP, "--crf", "34", "--hash", "3"]),
    ("smooth_10", "300x280", 10, 1, SMOOTH, [*INTRA, *NO_WPP, "--crf", "34", "--hash", "3"]),
    # H9: each hash method. x265's chroma CRC disagrees with (D-24) above one CTU
    # row, so the CRC streams hold one row.
    ("hash_md5_8", "128x96", 8, 2, BUSY, [*INTRA, *NO_WPP, "--crf", "32", "--hash", "1"]),
    ("hash_md5_10", "128x96", 10, 2, BUSY, [*INTRA, *NO_WPP, "--crf", "32", "--hash", "1"]),
    ("hash_crc_8", "128x64", 8, 2, BUSY, [*INTRA, *NO_WPP, "--crf", "32", "--hash", "2"]),
    ("hash_crc_10", "128x64", 10, 2, BUSY, [*INTRA, *NO_WPP, "--crf", "32", "--hash", "2"]),
    ("hash_checksum_8", "128x96", 8, 2, BUSY, [*INTRA, *NO_WPP, "--crf", "32", "--hash", "3"]),
    ("hash_checksum_10", "128x96", 10, 2, BUSY, [*INTRA, *NO_WPP, "--crf", "32", "--hash", "3"]),
    # H10: deblocking on, SAO off.
    ("deblock_8", "128x96", 8, 2, BUSY,
     [*INTRA, *NO_WPP, "--no-sao", "--crf", "34", "--hash", "1"]),
    ("deblock_10", "128x96", 10, 2, BUSY,
     [*INTRA, *NO_WPP, "--no-sao", "--crf", "34", "--hash", "1"]),
    ("deblock_offsets_8", "96x64", 8, 2, BUSY,
     [*INTRA, *NO_WPP, "--no-sao", "--crf", "34", "--deblock=-3:3", "--hash", "1"]),
    ("deblock_chroma_offsets_8", "96x64", 8, 2, BUSY,
     [*INTRA, *NO_WPP, "--no-sao", "--crf", "36", "--cbqpoffs", "6", "--crqpoffs", "-6",
      "--hash", "1"]),
    # H11: SAO on.
    ("sao_8", "128x96", 8, 2, BUSY, [*INTRA, *NO_WPP, "--crf", "32", "--hash", "1"]),
    ("sao_10", "128x96", 10, 2, BUSY, [*INTRA, *NO_WPP, "--crf", "32", "--hash", "1"]),
    ("sao_only_8", "128x96", 8, 2, BUSY,
     [*INTRA, *NO_WPP, "--no-deblock", "--crf", "32", "--hash", "1"]),
    ("sao_window_8", "122x90", 8, 2, BUSY, [*INTRA, *NO_WPP, "--crf", "30", "--hash", "1"]),
    ("sao_slices_8", "256x192", 8, 2, BUSY,
     ["--keyint", "1", "--min-keyint", "1", "--crf", "34", "--wpp", "--slices", "3",
      "--ctu", "16", "--hash", "1"]),
    # An inter stream, which the tests expect the decoder to refuse.
    ("inter_8", "128x96", 8, 4, BUSY, [*NO_WPP, "--crf", "34", "--keyint", "4"]),
]


def run(tool, args):
    """Runs a tool, and exits with its error text when it fails."""
    result = subprocess.run([tool, *args], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"{tool} {' '.join(args)} failed: {result.stderr}")


def generate(out, stream):
    """Writes one stream and ffmpeg's decode of it."""
    name, size, depth, frames, source, x265_args = stream
    pix_fmt = "yuv420p" if depth == 8 else "yuv420p10le"
    raw = out / f"{name}.source.raw"
    hevc = out / f"{name}.hevc"
    run("ffmpeg", [*QUIET, "-f", "lavfi", "-i", f"{source.format(size=size)},format={pix_fmt}",
                   "-frames:v", str(frames), "-f", "rawvideo", str(raw)])
    run("x265", ["--input", str(raw), "--input-res", size, "--fps", "25",
                 "--input-depth", str(depth), "--input-csp", "i420", "--output-depth", str(depth),
                 "--frames", str(frames), "--log-level", "error", *x265_args, "-o", str(hevc)])
    raw.unlink()
    run("ffmpeg", [*QUIET, "-i", str(hevc), "-f", "rawvideo", "-pix_fmt", pix_fmt,
                   str(out / f"{name}.yuv")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "hevc")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    missing = [tool for tool in ("ffmpeg", "x265") if shutil.which(tool) is None]
    if missing:
        sys.exit(f"skipping: {', '.join(missing)} not on PATH. Install it to generate the "
                 "HEVC fixtures.")
    args.out.mkdir(parents=True, exist_ok=True)
    for stream in STREAMS:
        if args.only is None or stream[0] in args.only:
            generate(args.out, stream)


if __name__ == "__main__":
    main()
