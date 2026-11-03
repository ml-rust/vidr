#!/usr/bin/env python3
"""Generate the bitstream filter fixtures for `tests/bitstream/`.

Run from any directory:

    python3 crates/vidr/tests/fixtures/gen_bitstream.py [--out DIR]

ffmpeg with libx264, libx265 and libaom-av1 must be on PATH. The default
output directory is `crates/vidr/tests/fixtures/bitstream/`, which git
ignores. Never commit its contents.

For each stream NAME the script writes:

- `NAME.mp4`: the MP4 form.
- `NAME.raw`: the Annex-B (`-f h264`, `-f hevc`) or low-overhead OBU
  (`-f obu`) form that ffmpeg makes from `NAME.mp4`. For H.264 and HEVC it
  is the output of ffmpeg's `h264_mp4toannexb` / `hevc_mp4toannexb`. For
  AV1 it is the output of ffmpeg's `obu` muxer, which inserts temporal
  delimiters.
- `NAME.config`: the `avcC`, `hvcC` or `av1C` box payload from `NAME.mp4`.
  `gen_nvdec.py` also reads `vpcC` through `extract`.
- `NAME.samples`: every sample of the video track, in decode order, each as
  a 4-byte big-endian size and then the sample bytes.

The tests read `VIDR_BITSTREAM_FIXTURES` first, then the default directory.
"""

import argparse
import shutil
import struct
import subprocess
import sys
from pathlib import Path

SOURCE = [
    "-f", "lavfi",
    "-i", "testsrc2=size=128x96:rate=25",
    "-frames:v", "12",
    "-pix_fmt", "yuv420p",
]
QUIET = ["-hide_banner", "-loglevel", "error", "-y"]


def run(args):
    """Runs ffmpeg and exits with its stderr on an error."""
    result = subprocess.run(["ffmpeg", *QUIET, *args], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"ffmpeg {' '.join(args)} failed:\n{result.stderr}")


def encode_mp4(path, codec_args):
    """Encodes the test source straight into an MP4 file."""
    run([*SOURCE, *codec_args, str(path)])


def encode_raw_then_mp4(path, raw_format, codec_args, tag_args):
    """Encodes to a raw elementary stream, then remuxes it into MP4.

    A raw encode has no global header, so the encoder writes its parameter
    sets in band. The remux keeps them in the MP4 samples.
    """
    raw = path.with_suffix(".encoder")
    run([*SOURCE, *codec_args, "-f", raw_format, str(raw)])
    run(["-fflags", "+genpts", "-r", "25", "-f", raw_format, "-i", str(raw),
         "-c", "copy", *tag_args, str(path)])
    raw.unlink()


def boxes(data, start, end):
    """Yields (type, payload start, payload end) for each box in a range."""
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            sys.exit(f"box {kind!r} at {pos} has size {size}, past the parent end {end}")
        yield kind.decode("latin-1"), pos + header, pos + size
        pos += size


def child(data, start, end, kind):
    """Returns the payload range of the first child box of a type."""
    for name, body, stop in boxes(data, start, end):
        if name == kind:
            return body, stop
    sys.exit(f"box {kind!r} is missing")


def video_track(data):
    """Returns the `stbl` payload range of the first video track."""
    moov = child(data, 0, len(data), "moov")
    for name, body, stop in boxes(data, *moov):
        if name != "trak":
            continue
        mdia = child(data, body, stop, "mdia")
        hdlr = child(data, *mdia, "hdlr")
        if data[hdlr[0] + 8:hdlr[0] + 12] == b"vide":
            minf = child(data, *mdia, "minf")
            return child(data, *minf, "stbl")
    sys.exit("no video track")


def config_record(data, stbl):
    """Returns the codec config box payload from the first sample entry."""
    stsd = child(data, *stbl, "stsd")
    # Full box header (4) and entry count (4), then the first entry.
    entry = next(boxes(data, stsd[0] + 8, stsd[1]))
    # SampleEntry (8) and VisualSampleEntry (70) fields precede the children.
    for name, body, stop in boxes(data, entry[1] + 78, entry[2]):
        if name in ("avcC", "hvcC", "av1C", "vpcC"):
            return data[body:stop]
    sys.exit(f"sample entry {entry[0]!r} has no avcC, hvcC, av1C or vpcC box")


def samples(data, stbl):
    """Returns every sample of a track, in decode order."""
    stsz = child(data, *stbl, "stsz")
    uniform, count = struct.unpack_from(">II", data, stsz[0] + 4)
    if uniform:
        sizes = [uniform] * count
    else:
        sizes = list(struct.unpack_from(f">{count}I", data, stsz[0] + 12))

    names = {name for name, _, _ in boxes(data, *stbl)}
    if "stco" in names:
        stco = child(data, *stbl, "stco")
        chunks = struct.unpack_from(">I", data, stco[0] + 4)[0]
        offsets = list(struct.unpack_from(f">{chunks}I", data, stco[0] + 8))
    else:
        co64 = child(data, *stbl, "co64")
        chunks = struct.unpack_from(">I", data, co64[0] + 4)[0]
        offsets = list(struct.unpack_from(f">{chunks}Q", data, co64[0] + 8))

    stsc = child(data, *stbl, "stsc")
    runs = struct.unpack_from(">I", data, stsc[0] + 4)[0]
    table = [struct.unpack_from(">III", data, stsc[0] + 8 + 12 * i) for i in range(runs)]

    out = []
    index = 0
    for run_index, (first, per_chunk, _) in enumerate(table):
        last = table[run_index + 1][0] if run_index + 1 < runs else chunks + 1
        for chunk in range(first, last):
            pos = offsets[chunk - 1]
            for _ in range(per_chunk):
                if index == count:
                    break
                out.append(data[pos:pos + sizes[index]])
                pos += sizes[index]
                index += 1
    if index != count:
        sys.exit(f"sample table maps {index} of {count} samples")
    return out


def extract(mp4, out_dir, name):
    """Writes NAME.config and NAME.samples from an MP4 file."""
    data = mp4.read_bytes()
    stbl = video_track(data)
    (out_dir / f"{name}.config").write_bytes(config_record(data, stbl))
    framed = b"".join(struct.pack(">I", len(s)) + s for s in samples(data, stbl))
    (out_dir / f"{name}.samples").write_bytes(framed)


def write_raw(mp4, raw, raw_format, bsf):
    """Writes ffmpeg's own conversion of the MP4 samples to the raw form."""
    args = ["-i", str(mp4), "-map", "0:v:0", "-c", "copy"]
    if bsf:
        args += ["-bsf:v", bsf]
    run([*args, "-f", raw_format, str(raw)])


X264 = ["-c:v", "libx264", "-g", "5", "-bf", "2"]
X265 = "keyint=5:min-keyint=5:bframes=2:log-level=error"
AOM = ["-c:v", "libaom-av1", "-cpu-used", "8", "-g", "5", "-row-mt", "0"]

STREAMS = [
    # Parameter sets in avcC only, SEI in the first sample, IDR every 5.
    ("h264", "h264", "h264_mp4toannexb",
     lambda p: encode_mp4(p, X264)),
    # Parameter sets in band before every IDR, and in avcC.
    ("h264_inband", "h264", "h264_mp4toannexb",
     lambda p: encode_raw_then_mp4(p, "h264", X264, [])),
    # VPS, SPS, PPS and a prefix SEI in hvcC, CRA pictures as IRAPs.
    ("hevc", "hevc", "hevc_mp4toannexb",
     lambda p: encode_mp4(p, ["-c:v", "libx265", "-x265-params", X265, "-tag:v", "hvc1"])),
    # As `hevc`, without the info SEI, so hvcC holds parameter sets only.
    ("hevc_plain", "hevc", "hevc_mp4toannexb",
     lambda p: encode_mp4(p, ["-c:v", "libx265", "-x265-params", f"{X265}:info=0",
                              "-tag:v", "hvc1"])),
    # Parameter sets in band before every IRAP, and in hvcC.
    ("hevc_inband", "hevc", "hevc_mp4toannexb",
     lambda p: encode_raw_then_mp4(
         p, "hevc", ["-c:v", "libx265", "-x265-params", f"{X265}:repeat-headers=1"],
         ["-tag:v", "hev1"])),
    # Sequence header in av1C and in band on key frames.
    ("av1", "obu", None,
     lambda p: encode_mp4(p, AOM)),
]


def main():
    default = Path(__file__).resolve().parent / "bitstream"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=default,
                        help=f"output directory (default: {default})")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is not on PATH. Install ffmpeg with libx264, libx265 and libaom.")

    args.out.mkdir(parents=True, exist_ok=True)
    for name, raw_format, bsf, make in STREAMS:
        mp4 = args.out / f"{name}.mp4"
        make(mp4)
        write_raw(mp4, args.out / f"{name}.raw", raw_format, bsf)
        extract(mp4, args.out, name)
        print(f"{name}: {mp4}")


if __name__ == "__main__":
    main()
