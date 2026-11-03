#!/usr/bin/env python3
"""Generate the NVDEC decode fixtures for `tests/nvdec/`.

Run from any directory:

    python3 crates/vidr/tests/fixtures/gen_nvdec.py [--out DIR]

ffmpeg and ffprobe with libx264, libx265 (8 and 10 bit), libaom-av1 and
libvpx-vp9 must be on PATH. A stream whose encoder is missing is skipped
with a notice. The default output directory is
`crates/vidr/tests/fixtures/nvdec/`, which git ignores. Never commit its
contents.

For each stream NAME the script writes:

- `NAME.config`: the `avcC`, `hvcC`, `av1C` or `vpcC` payload.
- `NAME.samples`: every video sample in decode order, each as a 4-byte
  big-endian size and then the sample bytes.
- `NAME.pts`: the pts of each sample in decode order, one per line, in the
  track time base.
- `NAME.frames`: the pts of each frame ffmpeg decodes, in display order.
- `NAME.meta`: `key=value` lines: `codec`, `width`, `height`, `format`
  (`nv12` or `p010`), and the ffprobe colour fields `matrix`, `transfer`,
  `primaries` as H.273 code points and `range` (`tv` or `pc`). A stream
  whose SPS signals a chroma sample location also has `chroma_location`,
  the H.273 `ChromaSampleLocType` number.
- `NAME.yuv`: ffmpeg's software decode as `-f rawvideo` in `nv12` or
  `p010le`, every frame in display order. Streams the tests expect NVDEC
  to refuse have no `.yuv`.

The tests read `VIDR_NVDEC_FIXTURES` first, then the default directory.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_bitstream import QUIET, extract  # noqa: E402

SIZE = "320x180"
# An odd display size: the chroma plane is ceil(W/2) x ceil(H/2) samples.
# testsrc2 rounds an odd size down, so the source is one pixel larger on
# each axis and `crop` cuts it to the odd size.
ODD_SOURCE = "322x182"
ODD_CROP = "crop=321:181:exact=1,"
FRAMES = "12"
# The frames carry the colour description, so every encoder writes it into
# the sequence header.
COLOR_FILTER = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"

# H.273 code points of the ffprobe names the streams use.
H273 = {
    "bt709": 1,
    "unknown": 2,
    "bt470bg": 5,
    "smpte170m": 6,
    "bt2020nc": 9,
}

# ChromaSampleLocType the VUI of a stream signals, and the name ffprobe
# reports for it. The encoder writes both fields from the same value.
CHROMA_LOCATIONS = {"center": (1, "center")}
CENTER = ["-chroma_sample_location", "center"]

X264 = ["-c:v", "libx264", "-g", "5", "-bf", "2"]
X265 = "keyint=5:min-keyint=5:bframes=2:log-level=error"
AOM = ["-c:v", "libaom-av1", "-cpu-used", "8", "-g", "5", "-row-mt", "0"]
VPX = ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-g", "5"]

# (name, codec, source pix_fmt, encoder args, decoded pix_fmt or None)
STREAMS = [
    ("h264_8bit", "h264", "yuv420p", X264, "nv12"),
    ("hevc_8bit", "hevc", "yuv420p",
     ["-c:v", "libx265", "-x265-params", X265, "-tag:v", "hvc1"], "nv12"),
    ("hevc_10bit", "hevc", "yuv420p10le",
     ["-c:v", "libx265", "-profile:v", "main10", "-x265-params", X265, "-tag:v", "hvc1"],
     "p010le"),
    ("av1_8bit", "av1", "yuv420p", AOM, "nv12"),
    ("vp9_8bit", "vp9", "yuv420p", VPX, "nv12"),
    # The SPS VUI signals chroma sample location type 1.
    ("h264_chroma_center", "h264", "yuv420p", [*X264, *CENTER], "nv12"),
    ("hevc_chroma_center", "hevc", "yuv420p",
     ["-c:v", "libx265", "-x265-params", X265, "-tag:v", "hvc1", *CENTER], "nv12"),
    # Odd display size. H.264 and HEVC crop in units of two samples at 4:2:0,
    # so only AV1 and VP9 carry an odd display size.
    ("av1_odd", "av1", "yuv420p", AOM, "nv12"),
    ("vp9_odd", "vp9", "yuv420p", VPX, "nv12"),
    # NVDEC decodes no H.264 High 10. The test expects `Unsupported`.
    ("h264_10bit", "h264", "yuv420p10le", [*X264, "-profile:v", "high10"], None),
    # 4:4:4 has no vidr frame layout. The test expects `UnsupportedLayout`.
    ("hevc_444", "hevc", "yuv444p",
     ["-c:v", "libx265", "-x265-params", X265, "-tag:v", "hvc1"], None),
]


def tool(name, args):
    """Runs ffmpeg or ffprobe and returns stdout, or None on an error."""
    result = subprocess.run([name, *args], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"{name} {' '.join(args)} failed:\n{result.stderr}", file=sys.stderr)
        return None
    return result.stdout


def probe(path, section, entries):
    """Returns the ffprobe entries of the first video stream."""
    out = tool("ffprobe", ["-v", "error", "-select_streams", "v:0",
                           f"-show_{section}", "-show_entries", entries,
                           "-of", "json", str(path)])
    if out is None:
        sys.exit(f"ffprobe cannot read {path}")
    return json.loads(out)


def encode(path, pix_fmt, codec_args, size=SIZE, crop=""):
    """Encodes the test source into an MP4 file. Returns False on an error."""
    source = ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25",
              "-frames:v", FRAMES, "-pix_fmt", pix_fmt]
    return tool("ffmpeg", [*QUIET, *source, *codec_args, "-vf", crop + COLOR_FILTER, str(path)]) is not None


def write_meta(mp4, out, name, codec, decoded, chroma_location):
    """Writes NAME.pts, NAME.frames and NAME.meta. Checks the frame count.

    `chroma_location` is the `CHROMA_LOCATIONS` key the encoder wrote, or
    None. ffprobe must report the same location.
    """
    packets = probe(mp4, "packets", "packet=pts")["packets"]
    pts = [int(packet["pts"]) for packet in packets]
    frames = probe(mp4, "frames", "frame=pts")["frames"]
    shown = [int(frame["pts"]) for frame in frames]
    if sorted(pts) != shown:
        sys.exit(f"{name}: ffmpeg shows pts {shown}, but the packets carry {sorted(pts)}")
    (out / f"{name}.pts").write_text("".join(f"{value}\n" for value in pts))
    (out / f"{name}.frames").write_text("".join(f"{value}\n" for value in shown))

    stream = probe(mp4, "streams",
                   "stream=width,height,color_space,color_transfer,"
                   "color_primaries,color_range,chroma_location")["streams"][0]

    def code(key):
        value = stream.get(key, "unknown")
        if value not in H273:
            sys.exit(f"{name}: {key} {value} has no H.273 code in this script")
        return H273[value]

    meta = {
        "codec": codec,
        "width": stream["width"],
        "height": stream["height"],
        "format": "p010" if decoded == "p010le" else "nv12",
        "matrix": code("color_space"),
        "transfer": code("color_transfer"),
        "primaries": code("color_primaries"),
        "range": stream.get("color_range", "tv"),
    }
    if chroma_location is not None:
        number, probed = CHROMA_LOCATIONS[chroma_location]
        if stream.get("chroma_location") != probed:
            sys.exit(f"{name}: ffprobe reports chroma_location "
                     f"{stream.get('chroma_location')}, expected {probed}")
        meta["chroma_location"] = number
    (out / f"{name}.meta").write_text("".join(f"{k}={v}\n" for k, v in meta.items()))


def decode(mp4, out, name, decoded):
    """Writes ffmpeg's software decode as raw planes in display order."""
    args = [*QUIET, "-i", str(mp4), "-map", "0:v:0", "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", decoded, str(out / f"{name}.yuv")]
    if tool("ffmpeg", args) is None:
        sys.exit(f"{name}: ffmpeg cannot decode {mp4}")


def main():
    default = Path(__file__).resolve().parent / "nvdec"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=default,
                        help=f"output directory (default: {default})")
    args = parser.parse_args()

    for name in ("ffmpeg", "ffprobe"):
        if shutil.which(name) is None:
            sys.exit(f"{name} is not on PATH. Install ffmpeg with libx264, libx265, "
                     "libaom and libvpx.")

    args.out.mkdir(parents=True, exist_ok=True)
    for name, codec, pix_fmt, codec_args, decoded in STREAMS:
        mp4 = args.out / f"{name}.mp4"
        odd = name.endswith("_odd")
        if not encode(mp4, pix_fmt, codec_args,
                      ODD_SOURCE if odd else SIZE, ODD_CROP if odd else ""):
            print(f"{name}: skipped, the encoder is missing or refuses {pix_fmt}")
            continue
        extract(mp4, args.out, name)
        write_meta(mp4, args.out, name, codec, decoded,
                   "center" if "-chroma_sample_location" in codec_args else None)
        if decoded is not None:
            decode(mp4, args.out, name, decoded)
        print(f"{name}: {mp4}")


if __name__ == "__main__":
    main()
