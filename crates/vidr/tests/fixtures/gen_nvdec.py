#!/usr/bin/env python3
"""Generate the decode fixtures for `tests/nvdec/`, `tests/dav1d/`, `tests/openh264/`, and `tests/rusty_vp9/`.

Run from any directory:

    python3 crates/vidr/tests/fixtures/gen_nvdec.py [--out DIR] [--only NAME ...]

ffmpeg and ffprobe with libx264, libx265 (8 and 10 bit), libaom-av1 and
libvpx-vp9 must be on PATH, and vpxenc for `vp9_resize`. A stream whose
encoder is missing is skipped with a notice. The default output directory is
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
  `primaries` as H.273 code points and `range` (`tv` or `pc`). For VP9,
  `matrix` and `range` come from ffprobe on the raw IVF, which reads the
  frame header. ffprobe on the MP4 reads `vpcC`, a copy the decoder
  ignores. `transfer` and `primaries` still come from the MP4, because a
  VP9 frame header carries neither. A stream
  whose SPS signals a chroma sample location also has `chroma_location`,
  the H.273 `ChromaSampleLocType` number. `interlaced` and `top_field_first`
  (0 or 1) are what ffprobe reports for every frame. The script exits when
  frames disagree, so each stream holds one structure.
- `NAME.yuv`: ffmpeg's software decode as `-f rawvideo` in `nv12` or
  `p010le`, every frame in display order, each at its own size. Streams the
  tests expect a decoder to refuse have no `.yuv`.

`vp9_two_shown` is a stream no encoder writes: each sample is a superframe of two
shown frames. Its `NAME.frames` holds one pts per sample, and its `NAME.yuv` holds the
second frame of each sample, from ffmpeg's decode of the single-frame stream. libvpx
returns the last shown frame of a chunk, and ffmpeg splits a superframe and returns
both. The meta file holds `codec`, `width`, `height`, and `format`.

The tests read `VIDR_NVDEC_FIXTURES` first, then the default directory.
"""

import argparse
import json
import shutil
import struct
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
# Streams whose colour comes from encoder flags, not from frame properties.
# The frames stay untagged, so the encoder writes only what the flags say.
FLAG_COLOR = {"vp9_bt470bg_pc", "vp9_untagged"}

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
# libx264 has no PAFF mode. `+ildct+ilme` turns on MBAFF, which sets
# `frame_mbs_only_flag` 0 and `mb_adaptive_frame_field_flag` 1.
MBAFF = ["-flags", "+ildct+ilme", "-x264-params", "tff=1"]
# Source filters in front of the colour filter, by stream name.
SOURCE_FILTERS = {"h264_mbaff_interlaced": "interlace=scan=tff,"}
X265 = "keyint=5:min-keyint=5:bframes=2:log-level=error"
AOM = ["-c:v", "libaom-av1", "-cpu-used", "8", "-g", "5", "-row-mt", "0"]
VPX = ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-g", "5"]

# (name, codec, source pix_fmt, encoder args, decoded pix_fmt or None). A stream
# in `CUSTOM_ENCODES` has no encoder args.
STREAMS = [
    ("h264_8bit", "h264", "yuv420p", X264, "nv12"),
    ("hevc_8bit", "hevc", "yuv420p",
     ["-c:v", "libx265", "-x265-params", X265, "-tag:v", "hvc1"], "nv12"),
    ("hevc_10bit", "hevc", "yuv420p10le",
     ["-c:v", "libx265", "-profile:v", "main10", "-x265-params", X265, "-tag:v", "hvc1"],
     "p010le"),
    ("av1_8bit", "av1", "yuv420p", AOM, "nv12"),
    ("av1_10bit", "av1", "yuv420p10le", AOM, "p010le"),
    ("vp9_8bit", "vp9", "yuv420p", VPX, "nv12"),
    # VP9 states its colour in the key frame header only. These streams
    # write `color_space` 1 with full range, and `color_space` 0 with the
    # default limited range.
    ("vp9_bt470bg_pc", "vp9", "yuv420p",
     [*VPX, "-colorspace", "bt470bg", "-color_range", "pc"], "nv12"),
    ("vp9_untagged", "vp9", "yuv420p", VPX, "nv12"),
    ("vp9_10bit", "vp9", "yuv420p10le", [*VPX, "-profile:v", "2"], "p010le"),
    # Spatial resize changes the frame size mid-stream.
    ("vp9_resize", "vp9", "yuv420p", None, "nv12"),
    # Hidden alt-ref frames in superframes.
    ("vp9_altref", "vp9", "yuv420p", None, "nv12"),
    # VP9 layouts the rusty_vp9 tests expect `Vp9Layout` for.
    ("vp9_12bit", "vp9", "yuv420p12le", [*VPX, "-profile:v", "2"], None),
    ("vp9_444", "vp9", "yuv444p", [*VPX, "-profile:v", "1"], None),
    ("vp9_422_10bit", "vp9", "yuv422p10le", [*VPX, "-profile:v", "3"], None),
    # The SPS VUI signals chroma sample location type 1.
    ("h264_chroma_center", "h264", "yuv420p", [*X264, *CENTER], "nv12"),
    ("hevc_chroma_center", "hevc", "yuv420p",
     ["-c:v", "libx265", "-x265-params", X265, "-tag:v", "hvc1", *CENTER], "nv12"),
    # Picture structure. Progressive content coded as MBAFF, interlaced
    # content coded as MBAFF, and progressive content flagged with
    # `frame_mbs_only_flag` 0 but coded as plain frames (no MBAFF).
    ("h264_mbaff_progressive", "h264", "yuv420p", [*X264, *MBAFF], "nv12"),
    ("h264_mbaff_interlaced", "h264", "yuv420p", [*X264, *MBAFF], "nv12"),
    ("h264_frame_flagged", "h264", "yuv420p",
     [*X264, "-x264-params", "fake-interlaced=1"], "nv12"),
    # Odd display size. H.264 and HEVC crop in units of two samples at 4:2:0,
    # so only AV1 and VP9 carry an odd display size.
    ("av1_odd", "av1", "yuv420p", AOM, "nv12"),
    ("vp9_odd", "vp9", "yuv420p", VPX, "nv12"),
    # NVDEC decodes no H.264 High 10. The test expects `Unsupported`.
    ("h264_10bit", "h264", "yuv420p10le", [*X264, "-profile:v", "high10"], None),
    # 4:4:4 has no vidr frame layout. The test expects `UnsupportedLayout`.
    ("hevc_444", "hevc", "yuv444p",
     ["-c:v", "libx265", "-x265-params", X265, "-tag:v", "hvc1"], None),
    # OpenH264's profile vidr passes. The other H.264 streams are its refusals.
    ("h264_baseline", "h264", "yuv420p",
     ["-c:v", "libx264", "-profile:v", "baseline", "-g", "5"], "nv12"),
    # AV1 layouts with no vidr frame. dav1d's tests expect `UnsupportedLayout`.
    ("av1_12bit", "av1", "yuv420p12le", AOM, None),
    ("av1_422", "av1", "yuv422p10le", AOM, None),
    ("av1_444", "av1", "yuv444p", AOM, None),
    ("av1_mono", "av1", "gray", AOM, None),
]


def tool(name, args):
    """Runs ffmpeg or ffprobe and returns stdout, or None on an error."""
    result = subprocess.run([name, *args], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"{name} {' '.join(args)} failed:\n{result.stderr}", file=sys.stderr)
        return None
    return result.stdout


def picture_structure(mp4, name):
    """Returns (interlaced, top_field_first) as ffprobe reports every frame."""
    frames = probe(mp4, "frames", "frame=interlaced_frame,top_field_first")["frames"]
    seen = {(int(f["interlaced_frame"]), int(f["top_field_first"])) for f in frames}
    if len(seen) != 1:
        sys.exit(f"{name}: ffprobe reports mixed picture structures {sorted(seen)}")
    return seen.pop()


def probe(path, section, entries):
    """Returns the ffprobe entries of the first video stream."""
    out = tool("ffprobe", ["-v", "error", "-select_streams", "v:0",
                           f"-show_{section}", "-show_entries", entries,
                           "-of", "json", str(path)])
    if out is None:
        sys.exit(f"ffprobe cannot read {path}")
    return json.loads(out)


def encode(path, pix_fmt, codec_args, size=SIZE, crop="", color_filter=COLOR_FILTER):
    """Encodes the test source into an MP4 file. Returns False on an error."""
    source = ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25",
              "-frames:v", FRAMES, "-pix_fmt", pix_fmt]
    chain = crop + color_filter if color_filter else crop.rstrip(",") or "null"
    return tool("ffmpeg", [*QUIET, *source, *codec_args, "-vf", chain, str(path)]) is not None


def encode_vp9_resize(path, out):
    """Encodes through vpxenc at a low rate, so spatial resize changes the size."""
    raw = out / "vp9_resize.source.yuv"
    webm = out / "vp9_resize.source.webm"
    source = [*QUIET, "-f", "lavfi", "-i", "testsrc2=size=640x480:rate=25", "-frames:v", "60",
              "-pix_fmt", "yuv420p", "-f", "rawvideo", str(raw)]
    if shutil.which("vpxenc") is None or tool("ffmpeg", source) is None:
        return False
    encoded = subprocess.run(
        ["vpxenc", "--codec=vp9", "--disable-warning-prompt", "--width=640", "--height=480",
         "--fps=25/1", "--rt", "--cpu-used=8", "--end-usage=cbr", "--resize-allowed=1",
         "--resize-up=60", "--resize-down=30", "--target-bitrate=15", "--buf-sz=200",
         "--buf-initial-sz=100", "--buf-optimal-sz=100", "--limit=60", "--webm",
         "-o", str(webm), str(raw)], capture_output=True, text=True)
    raw.unlink()
    if encoded.returncode != 0:
        print(f"vpxenc failed:\n{encoded.stderr}", file=sys.stderr)
        return False
    remuxed = tool("ffmpeg", [*QUIET, "-i", str(webm), "-c", "copy", str(path)])
    webm.unlink()
    return remuxed is not None


def encode_vp9_altref(path, out):
    """Encodes in two passes, the mode that writes hidden alt-ref frames."""
    log = out / "vp9_altref.pass"
    common = ["-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25", "-frames:v", "30",
              "-pix_fmt", "yuv420p", "-vf", COLOR_FILTER, "-c:v", "libvpx-vp9",
              "-deadline", "good", "-cpu-used", "4", "-b:v", "300k", "-g", "30",
              "-auto-alt-ref", "1", "-lag-in-frames", "16", "-row-mt", "0",
              "-passlogfile", str(log)]
    first = tool("ffmpeg", [*QUIET, *common, "-pass", "1", "-f", "null", "-"])
    second = None
    if first is not None:
        second = tool("ffmpeg", [*QUIET, *common, "-pass", "2", str(path)])
    for leftover in out.glob("vp9_altref.pass*"):
        leftover.unlink()
    return second is not None


CUSTOM_ENCODES = {"vp9_resize": encode_vp9_resize, "vp9_altref": encode_vp9_altref}


def vp9_header_color(mp4, out, name):
    """Returns (matrix, range) that ffprobe reads from the raw IVF of a VP9 stream.

    The IVF holds the bitstream without `vpcC`, so ffprobe reports the
    `color_space` and `color_range` of the key frame header.
    """
    ivf = out / f"{name}.ivf"
    args = [*QUIET, "-i", str(mp4), "-map", "0:v:0", "-c", "copy", "-f", "ivf", str(ivf)]
    if tool("ffmpeg", args) is None:
        sys.exit(f"{name}: ffmpeg cannot write the raw IVF")
    try:
        stream = probe(ivf, "streams", "stream=color_space,color_range")["streams"][0]
    finally:
        ivf.unlink(missing_ok=True)
    space = stream.get("color_space", "unknown")
    if space not in H273:
        sys.exit(f"{name}: color_space {space} has no H.273 code in this script")
    return H273[space], stream.get("color_range", "tv")


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
    if codec == "vp9":
        meta["matrix"], meta["range"] = vp9_header_color(mp4, out, name)
    meta["interlaced"], meta["top_field_first"] = picture_structure(mp4, name)
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
            "-noautoscale", "-f", "rawvideo", "-pix_fmt", decoded, str(out / f"{name}.yuv")]
    if tool("ffmpeg", args) is None:
        sys.exit(f"{name}: ffmpeg cannot decode {mp4}")


def ivf_frames(data):
    """Returns the frame payloads of an IVF file."""
    frames = []
    offset = struct.unpack_from("<H", data, 6)[0]
    while offset < len(data):
        size = struct.unpack_from("<I", data, offset)[0]
        offset += 12
        frames.append(data[offset:offset + size])
        offset += size
    return frames


def superframe(frames):
    """Joins `frames` with the VP9 superframe index of Annex B, 4-byte sizes."""
    marker = bytes([0b11011000 | (len(frames) - 1)])
    sizes = b"".join(struct.pack("<I", len(frame)) for frame in frames)
    return b"".join(frames) + marker + sizes + marker


def write_vp9_two_shown(out):
    """Writes `vp9_two_shown`. Returns False when ffmpeg has no libvpx-vp9."""
    name = "vp9_two_shown"
    width, height = SIZE.split("x")
    ivf = out / f"{name}.source.ivf"
    raw = out / f"{name}.source.yuv"
    source = ["-f", "lavfi", "-i", f"testsrc2=size={SIZE}:rate=25", "-frames:v", FRAMES,
              "-pix_fmt", "yuv420p", "-vf", COLOR_FILTER]
    encoded = tool("ffmpeg", [*QUIET, *source, *VPX, "-auto-alt-ref", "0", "-lag-in-frames", "0",
                              "-f", "ivf", str(ivf)])
    if encoded is None:
        return False
    if tool("ffmpeg", [*QUIET, "-i", str(ivf), "-f", "rawvideo", "-pix_fmt", "nv12",
                       str(raw)]) is None:
        sys.exit(f"{name}: ffmpeg cannot decode {ivf}")
    frames = ivf_frames(ivf.read_bytes())
    decoded = raw.read_bytes()
    ivf.unlink()
    raw.unlink()
    frame_bytes = int(width) * int(height) * 3 // 2
    if len(frames) % 2 != 0 or len(decoded) != len(frames) * frame_bytes:
        sys.exit(f"{name}: {len(frames)} frames, {len(decoded)} decoded bytes")
    pairs = range(0, len(frames), 2)
    samples = [superframe(frames[index:index + 2]) for index in pairs]
    framed = b"".join(struct.pack(">I", len(sample)) + sample for sample in samples)
    pts = "".join(f"{index}\n" for index in pairs)
    second = b"".join(decoded[(index + 1) * frame_bytes:(index + 2) * frame_bytes]
                      for index in pairs)
    (out / f"{name}.config").write_bytes(b"")
    (out / f"{name}.samples").write_bytes(framed)
    (out / f"{name}.pts").write_text(pts)
    (out / f"{name}.frames").write_text(pts)
    (out / f"{name}.yuv").write_bytes(second)
    meta = {"codec": "vp9", "width": width, "height": height, "format": "nv12"}
    (out / f"{name}.meta").write_text("".join(f"{k}={v}\n" for k, v in meta.items()))
    return True


def main():
    default = Path(__file__).resolve().parent / "nvdec"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=default,
                        help=f"output directory (default: {default})")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="write these streams only (default: every stream)")
    args = parser.parse_args()

    for name in ("ffmpeg", "ffprobe"):
        if shutil.which(name) is None:
            sys.exit(f"{name} is not on PATH. Install ffmpeg with libx264, libx265, "
                     "libaom and libvpx.")

    args.out.mkdir(parents=True, exist_ok=True)
    for name, codec, pix_fmt, codec_args, decoded in STREAMS:
        if args.only and name not in args.only:
            continue
        mp4 = args.out / f"{name}.mp4"
        odd = name.endswith("_odd")
        prefix = ODD_CROP if odd else SOURCE_FILTERS.get(name, "")
        color_filter = "" if name in FLAG_COLOR else COLOR_FILTER
        if name in CUSTOM_ENCODES:
            written = CUSTOM_ENCODES[name](mp4, args.out)
        else:
            written = encode(mp4, pix_fmt, codec_args, ODD_SOURCE if odd else SIZE, prefix,
                             color_filter)
        if not written:
            print(f"{name}: skipped, the encoder is missing or refuses {pix_fmt}")
            continue
        extract(mp4, args.out, name)
        write_meta(mp4, args.out, name, codec, decoded,
                   "center" if "-chroma_sample_location" in (codec_args or []) else None)
        if decoded is not None:
            decode(mp4, args.out, name, decoded)
        print(f"{name}: {mp4}")
    if not args.only or "vp9_two_shown" in args.only:
        written = write_vp9_two_shown(args.out)
        print("vp9_two_shown: " + ("written" if written else "skipped, libvpx-vp9 is missing"))


if __name__ == "__main__":
    main()
