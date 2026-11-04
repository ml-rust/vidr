#!/usr/bin/env python3
"""Generate the decode fixtures for `tests/nvdec/`, `tests/dav1d/`, `tests/openh264/`, `tests/rusty_vp9/`, and `tests/vulkan/`.

Run from any directory:

    python3 crates/vidr/tests/fixtures/gen_nvdec.py [--out DIR] [--only NAME ...]

ffmpeg and ffprobe with libx264, libx265 (8 and 10 bit), libaom-av1 and
libvpx-vp9 must be on PATH, and vpxenc for `vp9_resize`. The AV1 streams
for the Vulkan Video tests also take aomenc, libsvtav1, and librav1e. A stream whose
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

`h264_no_output_prior` is `h264_8bit`'s encode with `no_output_of_prior_pics_flag` set to 1
on every IDR picture but the first. ffmpeg's H.264 decoder ignores the flag, so its decode
gives the bytes of each frame, and ITU-T H.264 C.4.4 gives which frames stay. `NAME.frames`
and `NAME.yuv` hold those frames only. Where ffmpeg has `h264_cuvid`, the script exits when
that decoder keeps other frames. The stream has no `NAME.mp4`, which would not hold the edit.

`h264_small_*` and `hevc_small_*` hold the sizes `STREAM_SIZES` states, for the small-session
tests of `tests/vulkan/small.rs`.

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
from gen_av1_headers import trace_blocks  # noqa: E402
from gen_bitstream import QUIET, extract  # noqa: E402
from gen_hevc import Bits, escape, unescape  # noqa: E402

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
SVT = ["-c:v", "libsvtav1", "-preset", "10", "-g", "12"]

# name: (aomenc arguments, size, source pix_fmt, syntax elements a frame header must set).
AOMENC = {
    "av1_altref": ([], "320x180", "yuv420p", ["show_existing_frame", "showable_frame"]),
    "av1_altref_10bit": (["--bit-depth=10", "--profile=0"], "320x180", "yuv420p10le",
                         ["show_existing_frame", "showable_frame"]),
    "av1_forward_key": (["--enable-fwd-kf=1", "--kf-max-dist=10"], "320x180", "yuv420p",
                        ["show_existing_frame", "showable_frame"]),
    "av1_tiles": (["--tile-columns=1", "--tile-rows=1"], "640x360", "yuv420p",
                  ["tile_cols_log2", "tile_rows_log2"]),
    "av1_tile_groups": (["--tile-columns=1", "--tile-rows=1", "--num-tile-groups=3"], "640x360",
                        "yuv420p", ["tile_cols_log2", "tile_start_and_end_present_flag"]),
    "av1_restoration": (["--cpu-used=1", "--enable-restoration=1", "--enable-cdef=1"], "320x180",
                        "yuv420p", ["cdef_bits", "lr_type"]),
    "av1_superres": (["--superres-mode=1", "--superres-denominator=12"], "320x180", "yuv420p",
                     ["use_superres"]),
    "av1_resize": (["--resize-mode=2"], "640x360", "yuv420p",
                   ["frame_size_override_flag", "frame_width_minus_1"]),
    "av1_film_grain": (["--film-grain-test=1"], "320x180", "yuv420p",
                       ["apply_grain", "show_existing_frame"]),
    "av1_film_grain_10bit": (["--film-grain-test=2", "--bit-depth=10", "--profile=0"], "320x180",
                             "yuv420p10le", ["apply_grain"]),
    "av1_screen": (["--tune-content=screen", "--enable-intrabc=1", "--enable-palette=1",
                    "--cpu-used=4"], "320x180", "yuv420p", ["allow_screen_content_tools"]),
    "av1_sb128": (["--sb-size=128"], "320x180", "yuv420p", ["use_128x128_superblock"]),
    "av1_segmentation": (["--aq-mode=1"], "320x180", "yuv420p", ["segmentation_enabled"]),
    "av1_qm": (["--enable-qm=1"], "320x180", "yuv420p", ["using_qmatrix"]),
    "av1_error_resilient": (["--error-resilient=1"], "320x180", "yuv420p",
                            ["error_resilient_mode"]),
    "av1_sframe": (["--sframe-dist=5", "--sframe-mode=1"], "320x180", "yuv420p", []),
}

# Streams at their own size. They cover small video sessions on a Vulkan driver.
STREAM_SIZES = {
    "h264_small_64": "64x64",
    "h264_small_128": "128x128",
    "hevc_small_64": "64x64",
    "hevc_small_128": "128x128",
    "hevc_small_144": "144x144",
}

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
    # What the Vulkan Video tests add: Main, a B pyramid, reference marking commands
    # from a short reference list, several slices per picture, and scaling lists.
    ("h264_main", "h264", "yuv420p",
     ["-c:v", "libx264", "-profile:v", "main", "-g", "5", "-bf", "2"], "nv12"),
    ("h264_bpyramid", "h264", "yuv420p",
     ["-c:v", "libx264", "-x264-params", "bframes=3:b-pyramid=normal:ref=4:keyint=8"], "nv12"),
    ("h264_mmco", "h264", "yuv420p",
     ["-c:v", "libx264", "-x264-params", "bframes=3:b-pyramid=normal:ref=2:keyint=8"], "nv12"),
    ("h264_slices", "h264", "yuv420p", [*X264, "-slices", "4"], "nv12"),
    ("h264_cqm", "h264", "yuv420p", [*X264, "-x264-params", "cqm=jvt"], "nv12"),
    # `h264_8bit`'s encode, then `SAMPLE_EDITS` sets a flag in its IDR pictures.
    ("h264_no_output_prior", "h264", "yuv420p", X264, "nv12"),
    # AV1 layouts with no vidr frame. dav1d's tests expect `UnsupportedLayout`.
    ("av1_12bit", "av1", "yuv420p12le", AOM, None),
    ("av1_422", "av1", "yuv422p10le", AOM, None),
    ("av1_444", "av1", "yuv444p", AOM, None),
    ("av1_mono", "av1", "gray", AOM, None),
    # What the Vulkan Video tests add. An entry of `AOMENC` states how aomenc writes each stream,
    # and the tool the script checks it holds.
    *[(name, "av1", stream[2], None, "p010le" if "10bit" in name else "nv12")
      for name, stream in AOMENC.items()],
    ("av1_svt", "av1", "yuv420p", SVT, "nv12"),
    ("av1_svt_10bit", "av1", "yuv420p10le", SVT, "p010le"),
    ("av1_svt_film_grain", "av1", "yuv420p", [*SVT, "-svtav1-params", "film-grain=8"], "nv12"),
    ("av1_rav1e", "av1", "yuv420p", ["-c:v", "librav1e", "-speed", "10", "-g", "12"], "nv12"),
    # What the small-session tests read. `STREAM_SIZES` states each size.
    ("h264_small_64", "h264", "yuv420p", X264, "nv12"),
    ("h264_small_128", "h264", "yuv420p", X264, "nv12"),
    ("hevc_small_64", "hevc", "yuv420p",
     ["-c:v", "libx265", "-x265-params", X265, "-tag:v", "hvc1"], "nv12"),
    ("hevc_small_128", "hevc", "yuv420p",
     ["-c:v", "libx265", "-x265-params", X265, "-tag:v", "hvc1"], "nv12"),
    ("hevc_small_144", "hevc", "yuv420p",
     ["-c:v", "libx265", "-x265-params", X265, "-tag:v", "hvc1"], "nv12"),
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


def encode_aomenc(name, path, out):
    """Encodes through aomenc, checks the tools the stream must hold, and remuxes to MP4."""
    if shutil.which("aomenc") is None:
        return False
    args, size, pix_fmt, required = AOMENC[name]
    source = out / f"{name}.source.y4m"
    obu = out / f"{name}.source.obu"
    made = tool("ffmpeg", [*QUIET, "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25",
                           "-frames:v", "24", "-pix_fmt", pix_fmt, "-strict", "-1", str(source)])
    if made is None:
        return False
    encoded = subprocess.run(
        ["aomenc", "--cpu-used=8", "--limit=24", "--threads=1", "--passes=1",
         "--lag-in-frames=16", "--obu", *args, "-o", str(obu), str(source)],
        capture_output=True, text=True)
    source.unlink()
    if encoded.returncode != 0:
        print(f"aomenc failed:\n{encoded.stderr}", file=sys.stderr)
        return False
    check_elements(name, obu, required)
    remuxed = tool("ffmpeg", [*QUIET, "-f", "obu", "-i", str(obu), "-c", "copy", str(path)])
    obu.unlink()
    return remuxed is not None


def check_elements(name, obu, required):
    """Exits when no frame header of `obu` sets a required syntax element."""
    seen = set()
    for block in trace_blocks(obu):
        for _, elements in block:
            for element, value, _, _ in elements:
                if value:
                    seen.add(element.split("[")[0])
    missing = [element for element in required if element not in seen]
    if missing:
        sys.exit(f"{name}: the encoder wrote no frame with {missing}. Change its arguments.")


CUSTOM_ENCODES = {
    "vp9_resize": encode_vp9_resize,
    "vp9_altref": encode_vp9_altref,
    **{name: (lambda path, out, name=name: encode_aomenc(name, path, out)) for name in AOMENC},
}


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


def avcc_sets(config):
    """Returns the length field size, the SPS list and the PPS list of an `avcC` payload."""
    size = (config[4] & 3) + 1
    at = 6
    lists = []
    for count in (config[5] & 0x1F, None):
        if count is None:
            count = config[at]
            at += 1
        units = []
        for _ in range(count):
            length = struct.unpack_from(">H", config, at)[0]
            units.append(config[at + 2:at + 2 + length])
            at += 2 + length
        lists.append(units)
    return size, lists[0], lists[1]


def idr_header_facts(sps, pps):
    """Returns what an IDR I slice header needs: `frame_num` bits, POC type and LSB bits,
    the PPS's bottom-field flag, and its redundant picture flag."""
    reader = Bits(unescape(sps[1:]))
    profile_idc = reader.take(8)
    reader.take(16)  # constraint flags, level_idc
    reader.ue()  # seq_parameter_set_id
    if profile_idc in (100, 110, 122, 244, 44, 83, 86, 118, 128):
        if reader.ue() == 3:  # chroma_format_idc
            reader.take()  # separate_colour_plane_flag
        reader.ue()  # bit_depth_luma_minus8
        reader.ue()  # bit_depth_chroma_minus8
        reader.take()  # qpprime_y_zero_transform_bypass_flag
        if reader.take():  # seq_scaling_matrix_present_flag
            sys.exit("the IDR flag edit takes a stream with no SPS scaling matrix")
    frame_num_bits = reader.ue() + 4
    poc_type = reader.ue()
    if poc_type == 1:
        sys.exit("the IDR flag edit takes POC types 0 and 2")
    lsb_bits = reader.ue() + 4 if poc_type == 0 else 0

    reader = Bits(unescape(pps[1:]))
    reader.ue()  # pic_parameter_set_id
    reader.ue()  # seq_parameter_set_id
    reader.take()  # entropy_coding_mode_flag
    bottom_field = reader.take()
    if reader.ue():  # num_slice_groups_minus1
        sys.exit("the IDR flag edit takes a stream with no slice groups")
    reader.ue()  # num_ref_idx_l0_default_active_minus1
    reader.ue()  # num_ref_idx_l1_default_active_minus1
    reader.take(3)  # weighted_pred_flag, weighted_bipred_idc
    reader.ue()  # pic_init_qp_minus26, in se(v) form
    reader.ue()  # pic_init_qs_minus26, in se(v) form
    reader.ue()  # chroma_qp_index_offset, in se(v) form
    reader.take(2)  # deblocking_filter_control_present_flag, constrained_intra_pred_flag
    return frame_num_bits, poc_type, lsb_bits, bottom_field, reader.take()


def set_no_output_flag(unit, facts):
    """Sets `no_output_of_prior_pics_flag` to 1 in an IDR I slice NAL unit (7.3.3)."""
    frame_num_bits, poc_type, lsb_bits, bottom_field, redundant = facts
    body = bytearray(unescape(unit[1:]))
    reader = Bits(bytes(body))
    reader.ue()  # first_mb_in_slice
    if reader.ue() % 5 != 2:  # slice_type
        sys.exit("the IDR flag edit takes I slices only")
    reader.ue()  # pic_parameter_set_id
    reader.take(frame_num_bits)  # frame_num
    reader.ue()  # idr_pic_id
    if poc_type == 0:
        reader.take(lsb_bits)  # pic_order_cnt_lsb
        if bottom_field:
            reader.ue()  # delta_pic_order_cnt_bottom, in se(v) form
    if redundant:
        reader.ue()  # redundant_pic_cnt
    body[reader.at // 8] |= 0x80 >> (reader.at % 8)
    return unit[:1] + escape(bytes(body))


def read_framed(path):
    """Returns the samples of a NAME.samples file."""
    data = path.read_bytes()
    samples = []
    at = 0
    while at < len(data):
        size = struct.unpack_from(">I", data, at)[0]
        samples.append(data[at + 4:at + 4 + size])
        at += 4 + size
    return samples


def sample_units(sample, size):
    """Returns the NAL units of one length-prefixed sample."""
    units = []
    at = 0
    while at < len(sample):
        length = int.from_bytes(sample[at:at + size], "big")
        units.append(sample[at + size:at + size + length])
        at += size + length
    return units


def is_idr(unit):
    """Whether a NAL unit is a slice of an IDR picture."""
    return unit[0] & 0x1F == 5


def kept_by_c44(decode_pts, flagged, reorder):
    """Returns the pts of each frame that C.4.4 outputs, in output order.

    `flagged` holds the decode-order index of each IDR picture with the flag set, and every
    frame then waiting goes with no output. Otherwise a frame leaves once more than `reorder`
    frames wait. `h264_8bit` never fills its DPB first, so `reorder` alone decides.
    """
    waiting = []
    kept = []
    for index, pts in enumerate(decode_pts):
        if index in flagged:
            waiting.clear()
        waiting.append(pts)
        if len(waiting) > reorder:
            waiting.sort()
            kept.append(waiting.pop(0))
    return kept + sorted(waiting)


def check_with_cuvid(annexb, out, name, expected):
    """Exits when `h264_cuvid` keeps other bytes. A missing decoder or GPU skips the check."""
    raw = out / f"{name}.cuvid.yuv"
    args = [*QUIET, "-c:v", "h264_cuvid", "-i", str(annexb), "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "nv12", str(raw)]
    result = subprocess.run(["ffmpeg", *args], capture_output=True)
    if result.returncode != 0:
        print(f"{name}: skipped the h264_cuvid check, ffmpeg cannot run it")
        return
    decoded = raw.read_bytes()
    raw.unlink()
    if decoded != expected:
        sys.exit(f"{name}: h264_cuvid gives {len(decoded)} bytes that differ from the "
                 f"{len(expected)} bytes C.4.4 keeps")


def edit_no_output_prior(out, name):
    """Sets the flag in NAME.samples, then rewrites NAME.frames and NAME.yuv to what stays."""
    size, sps, pps = avcc_sets((out / f"{name}.config").read_bytes())
    facts = idr_header_facts(sps[0], pps[0])
    decode_pts = [int(line) for line in (out / f"{name}.pts").read_text().split()]
    frames = [int(line) for line in (out / f"{name}.frames").read_text().split()]
    decoded = (out / f"{name}.yuv").read_bytes()
    frame_bytes = len(decoded) // len(frames)

    samples = []
    flagged = set()
    seen_idr = False
    for index, sample in enumerate(read_framed(out / f"{name}.samples")):
        units = sample_units(sample, size)
        if any(is_idr(unit) for unit in units):
            if seen_idr:
                flagged.add(index)
                units = [set_no_output_flag(unit, facts) if is_idr(unit) else unit
                         for unit in units]
            seen_idr = True
        samples.append(units)
    (out / f"{name}.samples").write_bytes(b"".join(
        struct.pack(">I", len(sample)) + sample
        for sample in (b"".join(len(unit).to_bytes(size, "big") + unit for unit in units)
                       for units in samples)))

    reorder = probe(out / f"{name}.mp4", "streams", "stream=has_b_frames")["streams"][0]
    kept = kept_by_c44(decode_pts, flagged, int(reorder["has_b_frames"]))
    if len(kept) >= len(frames):
        sys.exit(f"{name}: the flag drops no frame, so the test proves nothing")
    expected = b"".join(decoded[frames.index(pts) * frame_bytes:
                                (frames.index(pts) + 1) * frame_bytes] for pts in kept)
    annexb = out / f"{name}.h264"
    annexb.write_bytes(b"".join(b"\x00\x00\x00\x01" + unit for unit in [
        *sps, *pps, *(unit for units in samples for unit in units)]))
    check_with_cuvid(annexb, out, name, expected)
    annexb.unlink()
    (out / f"{name}.mp4").unlink()
    (out / f"{name}.frames").write_text("".join(f"{value}\n" for value in kept))
    (out / f"{name}.yuv").write_bytes(expected)


# The streams an edit changes after the script writes their files.
SAMPLE_EDITS = {"h264_no_output_prior": edit_no_output_prior}


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
            size = ODD_SOURCE if odd else STREAM_SIZES.get(name, SIZE)
            written = encode(mp4, pix_fmt, codec_args, size, prefix, color_filter)
        if not written:
            print(f"{name}: skipped, the encoder is missing or refuses {pix_fmt}")
            continue
        extract(mp4, args.out, name)
        write_meta(mp4, args.out, name, codec, decoded,
                   "center" if "-chroma_sample_location" in (codec_args or []) else None)
        if decoded is not None:
            decode(mp4, args.out, name, decoded)
        if name in SAMPLE_EDITS:
            SAMPLE_EDITS[name](args.out, name)
        print(f"{name}: {mp4}")
    if not args.only or "vp9_two_shown" in args.only:
        written = write_vp9_two_shown(args.out)
        print("vp9_two_shown: " + ("written" if written else "skipped, libvpx-vp9 is missing"))


if __name__ == "__main__":
    main()
