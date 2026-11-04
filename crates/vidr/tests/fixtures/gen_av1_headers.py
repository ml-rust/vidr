"""Generate AV1 header parser fixtures with ffmpeg's `trace_headers` as the oracle.

`gen_headers.py` calls `generate`. Each stream NAME writes `NAME.headers` in the format that
module states, with these sections, in stream order:

    @ av1_unit HEX      one temporal unit, whole, with no elements
    @ av1_seq HEX       one sequence header OBU payload, and its elements
    @ av1_frame         one frame header, and its elements
    @ av1_group         one tile group, and its elements

An `av1_frame` section ends with `frame_header_bits`, the bits from its first element to the end
of its last one. ffmpeg logs `gm_params` only as subexp codes, so this length checks them.

Encoder streams come from aomenc, SvtAv1EncApp, and rav1e, through ffmpeg or alone. The synthetic
stream holds syntax no encoder here writes: short signaling, timing and decoder model info, two
operating points, and a load of film grain. Its tile data is filler, because `trace_headers` and
the parser read headers only.

A missing encoder skips its stream with a printed notice.
"""

import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from gen_headers import FFMPEG_LOG, QUIET, SyntaxWriter, write_sidecar

OBU_SEQUENCE = 1
OBU_TD = 2
OBU_FRAME_HEADER = 3
OBU_TILE_GROUP = 4
OBU_FRAME = 6
# Elements past the end of a header, or that the parser checks and does not keep.
AFTER_HEADER = {"trailing_one_bit", "trailing_zero_bit", "zero_bit"}


def leb128(value):
    """Encodes a minimal `leb128()` value."""
    out = bytearray()
    while True:
        byte = value & 0x7f
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def read_leb128(data, position):
    """Returns a `leb128()` value and the position after it."""
    value = 0
    for index in range(8):
        byte = data[position + index]
        value |= (byte & 0x7f) << (7 * index)
        if not byte & 0x80:
            return value, position + index + 1
    sys.exit("a leb128() value is longer than 8 bytes")


def split_obus(data):
    """Returns (type, whole OBU, payload) of each OBU in low-overhead data."""
    obus = []
    position = 0
    while position < len(data):
        start = position
        first = data[position]
        position += 2 if first & 0x04 else 1
        size, position = read_leb128(data, position)
        obus.append(((first >> 3) & 0x0f, data[start:position + size], data[position:position + size]))
        position += size
    return obus


def obu(kind, payload):
    """Builds an OBU with a size field and no extension."""
    return bytes([(kind << 3) | 0x02]) + leb128(len(payload)) + payload


def trace_blocks(path):
    """Runs `trace_headers`, and returns one block per OBU outside the extradata.

    A block is a list of (title, elements). An element is (name, value, position, width)."""
    result = subprocess.run(
        ["ffmpeg", *FFMPEG_LOG, "-f", "obu", "-i", str(path), "-c", "copy", "-bsf:v",
         "trace_headers", "-f", "null", "-"],
        capture_output=True, text=True, errors="replace")
    lines = [line for line in result.stderr.splitlines() if "[trace_headers @" in line]
    if result.returncode != 0 or "Failed to read unit" in result.stderr:
        tail = "\n".join(lines[-8:] or result.stderr.splitlines()[-8:])
        sys.exit(f"ffmpeg cannot read {path}:\n{tail}")
    blocks = []
    section = None
    in_extradata = False
    for line in lines:
        text = line.split("] ", 1)[1].strip()
        if text == "Extradata":
            in_extradata = True
        elif text.startswith("Packet:"):
            in_extradata = False
        elif text == "OBU header":
            section = None
            if not in_extradata:
                blocks.append([])
        elif " = " in text and text[0].isdigit():
            head, value = text.rsplit(" = ", 1)
            fields = head.split()
            if section is not None:
                width = len(fields[2]) if len(fields) > 2 else 0
                section.append((fields[1], int(value), int(fields[0]), width))
        elif text.startswith("Frame ") and "size" in text:
            pass
        elif not in_extradata and blocks:
            section = []
            blocks[-1].append((text, section))
    return blocks


def plain_elements(elements):
    """The elements as (name, value)."""
    return [(name, value) for name, value, _, _ in elements]


def frame_elements(elements):
    """The elements of a frame header, without trailing bits, with `frame_header_bits`."""
    kept = [element for element in elements if element[0] not in AFTER_HEADER]
    pairs = plain_elements(kept)
    first = kept[0][2]
    last = kept[-1][2] + kept[-1][3]
    return [*pairs, ("frame_header_bits", last - first)]


def sidecar_pairs(path):
    """Pairs each OBU of a low-overhead file with its trace."""
    data = path.read_bytes()
    obus = split_obus(data)
    blocks = trace_blocks(path)
    if len(obus) != len(blocks):
        sys.exit(f"{path}: {len(obus)} OBUs but {len(blocks)} trace blocks")
    pairs = []
    # One packet per temporal unit: from each delimiter to the next.
    units = []
    for kind, whole, _ in obus:
        if kind == OBU_TD or not units:
            units.append(bytearray())
        units[-1] += whole
    unit_index = -1
    for (kind, _, payload), block in zip(obus, blocks):
        if kind == OBU_TD or unit_index < 0:
            unit_index += 1
            pairs.append(("av1_unit", bytes(units[unit_index]), []))
        for title, elements in block:
            if title == "Sequence Header":
                pairs.append(("av1_seq", payload, plain_elements(elements)))
            elif title == "Frame Header":
                pairs.append(("av1_frame", b"", frame_elements(elements)))
            elif title == "Tile Group":
                kept = [element for element in elements if element[0] not in AFTER_HEADER]
                pairs.append(("av1_group", b"", plain_elements(kept)))
    return pairs


def write_source(path, size, frames, pix_fmt="yuv420p"):
    """Writes a moving test source as Y4M, for a standalone encoder."""
    subprocess.run(
        ["ffmpeg", *QUIET, "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25", "-frames:v",
         str(frames), "-pix_fmt", pix_fmt, "-strict", "-1", str(path)],
        check=True, capture_output=True)


def encode_aomenc(path, work, args, size="320x180", frames=24, pix_fmt="yuv420p"):
    """Encodes through aomenc."""
    source = work / f"source-{size}-{frames}-{pix_fmt}.y4m"
    if not source.exists():
        write_source(source, size, frames, pix_fmt)
    subprocess.run(
        ["aomenc", "--cpu-used=8", f"--limit={frames}", "--threads=1", "--passes=1",
         "--lag-in-frames=16", "--obu", *args, "-o", str(path), str(source)],
        check=True, capture_output=True)


def encode_ffmpeg(path, args, size="320x180", frames=24, pix_fmt="yuv420p"):
    """Encodes through an ffmpeg wrapper into the `obu` muxer."""
    subprocess.run(
        ["ffmpeg", *QUIET, "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25", "-frames:v",
         str(frames), "-pix_fmt", pix_fmt, *args, "-f", "obu", str(path)],
        check=True, capture_output=True)


def write_av1_sequence():
    """A sequence header with every optional part, 4 bits of `order_hint`, and frame ids."""
    w = SyntaxWriter()
    w.u(3, 0).flag(False).flag(False)
    w.flag(True).u(32, 1001).u(32, 30000).flag(False)
    w.flag(True).u(5, 9).u(32, 500).u(5, 7).u(5, 15)
    w.flag(True).u(5, 1)
    # Point 0: temporal 0 and spatial 0. Point 1: the same at a level that reads a tier.
    w.u(12, 0x101).u(5, 4)
    w.flag(True).u(10, 77).u(10, 88).flag(True)
    w.flag(True).u(4, 3)
    w.u(12, 0x101).u(5, 9).flag(True)
    w.flag(False)
    w.flag(False)
    # 12 bits of width and height, 320x180.
    w.u(4, 11).u(4, 11).u(12, 319).u(12, 179)
    # Frame ids of 9 bits, deltas of 4 bits.
    w.flag(True).u(4, 2).u(3, 4)
    w.flag(False).flag(True).flag(True)
    w.flag(True).flag(True).flag(True).flag(True)
    w.flag(True).flag(True).flag(True)
    # Screen content tools off, which reads no integer MV syntax.
    w.flag(False).u(1, 0)
    w.u(3, 3)
    w.flag(True).flag(True).flag(True)
    # 10-bit, a colour description, chroma position 2, separate chroma deltas.
    w.flag(True).flag(False).flag(True).u(8, 9).u(8, 16).u(8, 9)
    w.flag(False).u(2, 2).flag(True)
    w.flag(True)
    w.trailing()
    return obu(OBU_SEQUENCE, w.to_bytes())


def write_common_tail(w):
    """From `disable_frame_end_update_cdf` to `tx_mode`, for 320x180 at 64x64 superblocks."""
    w.flag(False)
    # Uniform: one column increment, one row increment, id 1, 2 size bytes.
    w.flag(True).flag(True).flag(False).flag(True).flag(False)
    w.u(2, 1).u(2, 1)
    # Quantization: chroma deltas differ, and a quantizer matrix per plane.
    w.u(8, 90)
    w.flag(True).su_av1(7, -3)
    w.flag(True)
    w.flag(True).su_av1(7, 2)
    w.flag(False)
    w.flag(False)
    w.flag(True).su_av1(7, -1)
    w.flag(True).u(4, 3).u(4, 5).u(4, 7)
    return w


def write_segmentation_update(w):
    """Features of segment 1 and segment 6, with every other one off."""
    for segment in range(8):
        for feature in range(8):
            if segment == 1 and feature == 0:
                w.flag(True).su_av1(9, -20)
            elif segment == 1 and feature == 5:
                w.flag(True).u(3, 2)
            elif segment == 6 and feature == 7:
                w.flag(True)
            elif segment == 6 and feature == 2:
                w.flag(True).su_av1(7, 11)
            else:
                w.flag(False)


def write_filters(w):
    """Loop filter with an update, CDEF with 4 strengths, and restoration on every plane."""
    w.u(6, 10).u(6, 11).u(6, 12).u(6, 13).u(3, 2)
    w.flag(True).flag(True)
    for index in range(8):
        w.flag(index in (0, 5))
        if index in (0, 5):
            w.su_av1(7, 4 - index)
    w.flag(True).su_av1(7, -2).flag(False)
    w.u(2, 1).u(2, 2)
    for index in range(4):
        w.u(4, index).u(2, 3).u(4, index + 4).u(2, index)
    w.u(2, 1).u(2, 2).u(2, 3)
    w.flag(True).flag(True).flag(True)
    # `tx_mode`.
    w.flag(True)


def write_grain(w, frame_is_inter):
    """Film grain with every optional part."""
    w.flag(True).u(16, 4242)
    if frame_is_inter:
        w.flag(True)
    w.u(4, 2).u(8, 10).u(8, 20).u(8, 200).u(8, 40)
    w.flag(False)
    w.u(4, 1).u(8, 30).u(8, 31)
    w.u(4, 1).u(8, 50).u(8, 51)
    w.u(2, 3).u(2, 1)
    for value in range(4):
        w.u(8, 128 + value)
    for value in range(5):
        w.u(8, 100 + value)
    for value in range(5):
        w.u(8, 60 + value)
    w.u(2, 2).u(2, 1)
    w.u(8, 128).u(8, 192).u(9, 256)
    w.u(8, 127).u(8, 191).u(9, 255)
    w.flag(True).flag(True)


def write_subexp_zero(w):
    """A `decode_subexp()` of 0, for a parameter with 8193 symbols."""
    w.flag(False).u(3, 0)


def synth_key_frame():
    """A shown key frame with segmentation, delta q, delta lf, and film grain."""
    w = SyntaxWriter()
    # frame_presentation_time follows `show_frame`, as `equal_picture_interval` is 0.
    w.flag(False).u(2, 0).flag(True).u(16, 1234)
    w.flag(False)
    # current_frame_id, no override, order_hint.
    w.u(9, 5).flag(False).u(4, 0)
    # buffer_removal_time_present_flag, then one time for point 0.
    w.flag(True).u(8, 33)
    # Superres off, render size differs.
    w.flag(False).flag(True).u(16, 639).u(16, 359)
    write_common_tail(w)
    w.flag(True)
    write_segmentation_update(w)
    w.flag(True).u(2, 2).flag(True).u(2, 1).flag(True)
    write_filters(w)
    w.flag(False)
    write_grain(w, False)
    w.align()
    return obu(OBU_FRAME, w.to_bytes() + b"\x00" * 12)


def synth_inter_frame():
    """A hidden inter frame with short signaling, global motion, and a load of film grain."""
    w = SyntaxWriter()
    w.flag(False).u(2, 1).flag(False).flag(True).flag(False)
    w.flag(False).u(9, 6).flag(False).u(4, 2).u(3, 0)
    w.flag(False)
    w.u(8, 0x0f)
    w.flag(True).u(3, 0).u(3, 1)
    for _ in range(7):
        w.u(4, 0)
    # Superres at denominator 12, no render difference.
    w.flag(True).u(3, 3).flag(False)
    w.flag(True)
    w.flag(False).u(2, 1)
    w.flag(True)
    w.flag(True)
    write_common_tail(w)
    # Segmentation on with a new map and no new data: the features load.
    w.flag(True).flag(True).flag(True).flag(False)
    w.flag(True).u(2, 1).flag(False)
    write_filters(w)
    # reference_select, allow_warped_motion, reduced_tx_set. No hint allows skip mode.
    w.flag(True)
    w.flag(True)
    w.flag(False)
    # LAST rotzoom, LAST2 translation, LAST3 affine, the rest off.
    w.flag(True).flag(True)
    for _ in range(4):
        write_subexp_zero(w)
    w.flag(True).flag(False).flag(True)
    for _ in range(2):
        # `GM_ABS_TRANS_ONLY_BITS` at high precision: 1025 symbols.
        write_subexp_zero(w)
    w.flag(True).flag(False).flag(False)
    for _ in range(6):
        write_subexp_zero(w)
    w.u(4, 0)
    # Film grain from the name of the key frame, with a new seed.
    w.flag(True).u(16, 999).flag(False).u(3, 0)
    w.align()
    return obu(OBU_FRAME, w.to_bytes() + b"\x00" * 12)


def write_synthetic(path):
    """Writes the synthetic stream. ffmpeg reads the syntax or the generator exits."""
    delimiter = obu(OBU_TD, b"")
    path.write_bytes(delimiter + write_av1_sequence() + synth_key_frame()
                     + delimiter + synth_inter_frame()
                     + delimiter + obu(OBU_FRAME_HEADER, show_existing_bytes(1)))


def show_existing_bytes(name):
    """A `show_existing_frame` header of `name`, with its presentation time and id."""
    w = SyntaxWriter()
    w.flag(True).u(3, name).u(16, 77).u(9, 6)
    w.trailing()
    return w.to_bytes()


AOM_8 = {
    "headers_av1_default": ([], "320x180", 24),
    "headers_av1_tiles": (["--tile-columns=1", "--tile-rows=1"], "640x360", 12),
    "headers_av1_tile_groups":
        (["--tile-columns=1", "--tile-rows=1", "--num-tile-groups=3"], "640x360", 12),
    "headers_av1_film_grain": (["--film-grain-test=1"], "320x180", 24),
    "headers_av1_superres":
        (["--superres-mode=1", "--superres-denominator=12", "--enable-restoration=1"],
         "320x180", 24),
    "headers_av1_resize": (["--resize-mode=2", "--enable-restoration=1"], "320x180", 24),
    "headers_av1_error_resilient": (["--error-resilient=1"], "320x180", 24),
    "headers_av1_sframe": (["--sframe-dist=5", "--sframe-mode=1"], "320x180", 24),
    "headers_av1_forward_key": (["--enable-fwd-kf=1", "--kf-max-dist=10"], "320x180", 24),
    "headers_av1_sb128": (["--sb-size=128", "--enable-restoration=1"], "320x180", 24),
    "headers_av1_qm": (["--enable-qm=1"], "320x180", 24),
    "headers_av1_segmentation": (["--aq-mode=1"], "320x180", 24),
    "headers_av1_screen":
        (["--tune-content=screen", "--enable-intrabc=1"], "320x180", 12),
    "headers_av1_still": (["--enable-order-hint=1"], "320x180", 1),
    "headers_av1_no_order_hint": (["--enable-order-hint=0"], "320x180", 12),
}


def encoder_streams(work):
    """Returns (name, encode function) of each encoder stream."""
    streams = []
    for name, (args, size, frames) in AOM_8.items():
        streams.append((name, lambda path, args=args, size=size, frames=frames: encode_aomenc(
            path, work, args, size, frames)))
    streams.append(("headers_av1_10bit", lambda path: encode_aomenc(
        path, work, ["--bit-depth=10", "--profile=0"], "320x180", 12, "yuv420p10le")))
    streams.append(("headers_av1_mono", lambda path: encode_aomenc(
        path, work, ["--monochrome"], "320x180", 12)))
    streams.append(("headers_av1_svt", lambda path: encode_ffmpeg(
        path, ["-c:v", "libsvtav1", "-preset", "10", "-g", "12"])))
    streams.append(("headers_av1_svt_grain", lambda path: encode_ffmpeg(
        path, ["-c:v", "libsvtav1", "-preset", "10", "-g", "12", "-svtav1-params",
               "film-grain=8"])))
    streams.append(("headers_av1_rav1e", lambda path: encode_ffmpeg(
        path, ["-c:v", "librav1e", "-speed", "10", "-g", "12"])))
    return streams


def generate(out_dir, work):
    """Writes every AV1 `NAME.headers` sidecar into `out_dir`."""
    if shutil.which("ffmpeg") is None:
        print("skipping the AV1 header streams: ffmpeg is not on PATH")
        return
    synthetic = work / "headers_synth_av1.obu"
    write_synthetic(synthetic)
    write_sidecar(out_dir / "headers_synth_av1.headers", sidecar_pairs(synthetic))
    print("headers_synth_av1: synthetic AV1")
    for name, encode in encoder_streams(work):
        path = work / f"{name}.obu"
        try:
            encode(path)
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            print(f"skipping {name}: its encoder is missing or refused it: {error}")
            continue
        write_sidecar(out_dir / f"{name}.headers", sidecar_pairs(path))
        print(f"{name}: encoder stream")


if __name__ == "__main__":
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=arguments.out, prefix="headers-work-") as temporary:
        generate(arguments.out, Path(temporary))
