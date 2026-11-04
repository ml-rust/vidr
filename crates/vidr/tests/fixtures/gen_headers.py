"""Generate header parser fixtures with ffmpeg's `trace_headers` as the oracle.

`gen_bitstream.py` calls `generate`. Each stream NAME writes one text file,
`NAME.headers`, into the bitstream fixture directory, which git ignores. A
source stream is removed once its sidecar exists, so the directory holds no
extra media from this module.

Sidecar format, one section per parsed header, in stream order:

    @ vps|sps|pps|slice|vp9 HEX
    name value

`gen_av1_headers.py` writes the AV1 sidecars and states their sections.

HEX is the NAL unit or VP9 frame the section describes, emulation prevention
bytes included. Each following line is one syntax element as `trace_headers`
logs it. A name holds its indices, for example `used_by_curr_pic_flag[2]`.

Two stream sources exist:

- Encoder streams: libx265, libvpx-vp9, and vpxenc, through ffmpeg or alone.
- Synthetic streams: `SyntaxWriter` writes headers with syntax no encoder here
  emits, such as tiles, PCM, long-term pictures, dependent slice segments, and
  VP9 frames of every profile. Their slice data is filler, because
  `trace_headers` and the parsers read headers only. ffmpeg stays the oracle.

An encoder or tool that is missing skips its stream with a printed notice.
"""

import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

FFMPEG_LOG = ["-hide_banner", "-loglevel", "trace", "-y"]
QUIET = ["-hide_banner", "-loglevel", "error", "-y"]
SECTIONS = {
    "Video Parameter Set": "vps",
    "Sequence Parameter Set": "sps",
    "Picture Parameter Set": "pps",
    "Slice Segment Header": "slice",
    "Frame": "vp9",
}
HEVC_TYPES = {"vps": {32}, "sps": {33}, "pps": {34}, "slice": set(range(0, 32))}
START_CODE = b"\x00\x00\x00\x01"


class SyntaxWriter:
    """Writes syntax elements, most significant bit first."""

    def __init__(self):
        self.bits = []

    def u(self, count, value):
        """Writes `value` in `count` bits."""
        if value < 0 or value >= (1 << count):
            sys.exit(f"value {value} does not fit {count} bits")
        for shift in range(count - 1, -1, -1):
            self.bits.append((value >> shift) & 1)
        return self

    def flag(self, value):
        """Writes one bit."""
        return self.u(1, 1 if value else 0)

    def ue(self, value):
        """Writes an unsigned Exp-Golomb code."""
        code = value + 1
        length = code.bit_length()
        return self.u(length - 1, 0).u(length, code)

    def se(self, value):
        """Writes a signed Exp-Golomb code."""
        return self.ue(2 * value - 1 if value > 0 else -2 * value)

    def su(self, count, value):
        """Writes a VP9 `s(n)`: magnitude, then sign."""
        return self.u(count, abs(value)).flag(value < 0)

    def su_av1(self, count, value):
        """Writes an AV1 `su(n)`: two's complement in `count` bits."""
        return self.u(count, value & ((1 << count) - 1))

    def trailing(self):
        """Writes `rbsp_trailing_bits()` or `byte_alignment()`."""
        self.bits.append(1)
        return self.align()

    def align(self):
        """Writes zero bits up to a byte boundary."""
        while len(self.bits) % 8:
            self.bits.append(0)
        return self

    def to_bytes(self):
        """Returns the bits as bytes, zero-padded."""
        padded = self.bits + [0] * (-len(self.bits) % 8)
        out = bytearray()
        for start in range(0, len(padded), 8):
            byte = 0
            for bit in padded[start:start + 8]:
                byte = (byte << 1) | bit
            out.append(byte)
        return bytes(out)


def escape(rbsp):
    """Adds emulation prevention bytes."""
    out = bytearray()
    zeros = 0
    for byte in rbsp:
        if zeros >= 2 and byte <= 3:
            out.append(3)
            zeros = 0
        out.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    return bytes(out)


def hevc_nal(nal_type, writer, filler=b"", temporal_id=0):
    """Builds a NAL unit from a finished writer and optional slice data."""
    header = struct.pack(">H", (nal_type << 9) | (temporal_id + 1))
    return header + escape(writer.to_bytes() + filler)


def write_ptl(w, profile_idc, sub_layers):
    """Writes `profile_tier_level()` (7.3.3). `sub_layers` holds one pair per
    lower sub-layer: profile present and level present."""
    w.u(2, 0).flag(False).u(5, profile_idc)
    w.u(32, 1 << (31 - profile_idc))
    w.flag(True).flag(False).flag(False).flag(True)
    w.u(43, 0).flag(False).u(8, 120)
    for profile, level in sub_layers:
        w.flag(profile).flag(level)
    if sub_layers:
        for _ in range(len(sub_layers), 8):
            w.u(2, 0)
    for index, (profile, level) in enumerate(sub_layers):
        if profile:
            w.u(2, 0).flag(False).u(5, profile_idc)
            w.u(32, 1 << (31 - profile_idc))
            w.flag(True).flag(False).flag(False).flag(True)
            w.u(43, 0).flag(False)
        if level:
            w.u(8, 90 + 3 * index)


def write_hrd(w, common, sub_layers):
    """Writes `hrd_parameters()` (E.2.2) with NAL and VCL parameters."""
    if common:
        w.flag(True).flag(True)
        w.flag(True).u(8, 10).u(5, 3).flag(True).u(5, 4)
        w.u(4, 2).u(4, 3).u(4, 1)
        w.u(5, 20).u(5, 21).u(5, 22)
    for _ in range(sub_layers + 1):
        w.flag(False).flag(True).ue(2)
        w.ue(1)
        for _ in range(2):
            for _ in range(2):
                w.ue(100).ue(200).ue(10).ue(20).flag(True)


def vps_nal(sub_layers=1, timing=True, ordering=True, layer_sets=0):
    """Builds a VPS with timing, HRD, layer sets, and an extension."""
    w = SyntaxWriter()
    w.u(4, 0).flag(True).flag(True).u(6, 0).u(3, sub_layers)
    w.flag(sub_layers == 0).u(16, 0xffff)
    write_ptl(w, 1, [(True, True)] * sub_layers)
    w.flag(ordering)
    for index in range(0 if ordering else sub_layers, sub_layers + 1):
        w.ue(4 + index).ue(1 + index).ue(3 * index)
    w.u(6, 1 if layer_sets else 0).ue(layer_sets)
    for _ in range(layer_sets):
        w.flag(True).flag(False)
    w.flag(timing)
    if timing:
        w.u(32, 1001).u(32, 30000).flag(True).ue(2)
        w.ue(layer_sets + 1)
        for index in range(layer_sets + 1):
            w.ue(index)
            if index > 0:
                # ffmpeg reads no sub-layer HRD when this flag is 0, so a stream
                # with 0 is no oracle for the inheritance 7.4.3.1 states.
                w.flag(True)
            write_hrd(w, True, sub_layers)
    w.flag(False).trailing()
    return hevc_nal(32, w)


def write_scaling(w, chroma_444=False):
    """Writes `scaling_list_data()` (7.3.4), with every prediction mode.

    The matrix loop runs on 3 steps at size 3 whatever the chroma format, as
    the V11 syntax states. `chroma_444` changes only the delta codes used."""
    for size_id in range(4):
        for matrix_id in range(0, 6, 3 if size_id == 3 else 1):
            if matrix_id == 0:
                # Explicit list: a ramp that stays above 0.
                w.flag(True)
                count = min(64, 1 << (4 + (size_id << 1)))
                if size_id > 1:
                    w.se(4)
                for _ in range(count):
                    w.se(1)
            elif matrix_id == 1:
                # Prediction from the default list.
                w.flag(False).ue(0)
            elif matrix_id == 3 and size_id == 3 and chroma_444:
                w.flag(False).ue(0)
            elif matrix_id == 3 and size_id == 3:
                # Prediction from matrix 0 at size 3: delta 1 means 3 steps.
                w.flag(False).ue(1)
            else:
                # Prediction from the previous list.
                w.flag(False).ue(1)


def write_vui(w):
    """Writes `vui_parameters()` (E.2.1) with every presence flag set."""
    w.flag(True).u(8, 255).u(16, 7).u(16, 5)
    w.flag(True).flag(True)
    w.flag(True).u(3, 5).flag(True).flag(True).u(8, 9).u(8, 16).u(8, 9)
    w.flag(True).ue(1).ue(2)
    w.flag(False).flag(False).flag(True)
    w.flag(True).ue(1).ue(1).ue(0).ue(0)
    w.flag(False)
    w.flag(True).flag(True).flag(True).flag(False)
    w.ue(4).ue(2).ue(1).ue(14).ue(15)


def write_rps_explicit(w, negative, positive, first):
    """Writes `st_ref_pic_set()` with explicit lists of `(delta_minus1, used)`."""
    if not first:
        w.flag(False)
    w.ue(len(negative)).ue(len(positive))
    for delta, used in [*negative, *positive]:
        w.ue(delta).flag(used)


def write_rps_inter(w, in_slice, delta_idx, sign, abs_minus1, flags):
    """Writes an inter-predicted `st_ref_pic_set()`. `flags` holds one pair of
    `used_by_curr_pic_flag` and `use_delta_flag` per source entry."""
    w.flag(True)
    if in_slice:
        w.ue(delta_idx)
    w.flag(sign).ue(abs_minus1)
    for used, use_delta in flags:
        w.flag(used)
        if not used:
            w.flag(use_delta)


def sps_full(extension=True):
    """An SPS with every optional structure, for `synth_full`."""
    w = SyntaxWriter()
    w.u(4, 0).u(3, 1).flag(False)
    write_ptl(w, 1, [(True, True)])
    w.ue(0).ue(1)
    w.ue(416).ue(240)
    w.flag(True).ue(1).ue(2).ue(0).ue(3)
    w.ue(0).ue(0).ue(4)
    w.flag(True)
    for index in range(2):
        w.ue(6 + index).ue(1 + index).ue(3 * index)
    w.ue(0).ue(3).ue(0).ue(3).ue(2).ue(1)
    w.flag(True).flag(True)
    write_scaling(w)
    w.flag(True).flag(True)
    w.flag(True).u(4, 6).u(4, 5).ue(0).ue(2).flag(True)
    w.ue(3)
    write_rps_explicit(w, [(0, True), (1, False)], [(0, True)], first=True)
    # NumDeltaPocs 3: four pairs. Entry 3 uses delta 0 for the new picture.
    write_rps_inter(w, False, 0, True, 0,
                    [(True, True), (False, True), (False, False), (True, True)])
    write_rps_explicit(w, [], [], first=False)
    w.flag(True).ue(3)
    for lsb, used in [(17, True), (200, False), (5, True)]:
        w.u(8, lsb).flag(used)
    w.flag(True).flag(True)
    w.flag(True)
    write_vui(w)
    w.flag(extension)
    if extension:
        w.flag(True).flag(True).flag(False).flag(False).u(4, 3)
        for value in (1, 1, 1, 1, 0, 1, 0, 1, 0):
            w.flag(value)
        # `sps_multilayer_extension()`: `inter_view_mv_vert_constraint_flag`.
        w.flag(True)
        # `sps_extension_data_flag` bits, which a decoder ignores.
        w.u(5, 0b10110)
    w.trailing()
    return hevc_nal(33, w)


def sps_plain(chroma, separate=False, depth=0, high_precision=False, poc_bits=8,
              num_sets=2, long_term=0, tmvp=True, sao=True, width=416, height=240,
              scaling=False, sub_layer_info=False):
    """A shorter SPS, for the chroma format and bit depth variants."""
    w = SyntaxWriter()
    w.u(4, 0).u(3, 0).flag(True)
    write_ptl(w, 4 if chroma != 1 or depth else 1, [])
    w.ue(1).ue(chroma)
    if chroma == 3:
        w.flag(separate)
    w.ue(width).ue(height).flag(False)
    w.ue(depth).ue(depth).ue(poc_bits - 4)
    w.flag(sub_layer_info).ue(5).ue(2).ue(0)
    w.ue(0).ue(2).ue(0).ue(3).ue(1).ue(1)
    w.flag(scaling)
    if scaling:
        w.flag(True)
        write_scaling(w, chroma_444=chroma == 3)
    w.flag(False).flag(sao).flag(False)
    w.ue(num_sets)
    for index in range(num_sets):
        write_rps_explicit(w, [(0, True), (0, True)], [(1, True)], first=index == 0)
    w.flag(long_term > 0)
    if long_term:
        w.ue(long_term)
        for index in range(long_term):
            w.u(poc_bits, index + 3).flag(index % 2 == 0)
    w.flag(tmvp).flag(False).flag(False)
    w.flag(high_precision)
    if high_precision:
        w.flag(True).flag(False).flag(False).flag(False).u(4, 0)
        for value in (0, 0, 0, 0, 0, 0, 1, 0, 0):
            w.flag(value)
    w.trailing()
    return hevc_nal(33, w)


def pps_full(pps_id=0, sps_id=0):
    """A PPS with tiles, wavefront, deblock control, scaling, and extensions."""
    w = SyntaxWriter()
    w.ue(pps_id).ue(sps_id)
    w.flag(True).flag(True).u(3, 2).flag(True).flag(True)
    w.ue(2).ue(1).se(-3)
    w.flag(True).flag(True).flag(True).ue(2)
    w.se(3).se(-2).flag(True)
    w.flag(True).flag(True).flag(True)
    w.flag(True).flag(True)
    w.ue(2).ue(1).flag(False).ue(1).ue(0).ue(0).flag(False)
    w.flag(True)
    w.flag(True).flag(True).flag(False).se(2).se(-1)
    w.flag(True)
    write_scaling(w)
    w.flag(True).ue(2).flag(True)
    w.flag(True)
    w.flag(True).flag(False).flag(False).flag(False).u(4, 0)
    w.ue(2).flag(False).flag(True).ue(1).ue(1)
    for cb, cr in [(3, -3), (2, -1)]:
        w.se(cb).se(cr)
    w.ue(0).ue(0)
    w.trailing()
    return hevc_nal(34, w)


def pps_plain(pps_id, sps_id, uniform_tiles=False, wavefront=False, weighted=True,
              lists=False, chroma_offsets=False, deblock_override=False):
    """A shorter PPS for the chroma format and bit depth variants."""
    w = SyntaxWriter()
    w.ue(pps_id).ue(sps_id)
    w.flag(False).flag(False).u(3, 0).flag(False).flag(False)
    w.ue(0).ue(0).se(0)
    w.flag(False).flag(False).flag(False)
    w.se(0).se(0).flag(chroma_offsets)
    w.flag(weighted).flag(weighted).flag(False)
    w.flag(uniform_tiles).flag(wavefront)
    if uniform_tiles:
        w.ue(1).ue(1).flag(True).flag(True)
    w.flag(True)
    w.flag(deblock_override)
    if deblock_override:
        w.flag(True).flag(True)
    w.flag(False).flag(lists).ue(0).flag(False).flag(False)
    w.trailing()
    return hevc_nal(34, w)


def slice_full_idr():
    """An IDR slice under `pps_full`: extra bits, entry points, an extension."""
    w = SyntaxWriter()
    w.flag(True).flag(True).ue(0)
    w.u(2, 1).ue(2).flag(True)
    w.flag(True).flag(True)
    w.se(4).se(1).se(-1).flag(True)
    w.flag(True).flag(False).se(3).se(-3)
    w.flag(True)
    # Two tile columns with wavefront rows: at most 3 * 4 - 1 entry points.
    w.ue(3).ue(9)
    for offset in (40, 1000, 3):
        w.u(10, offset)
    w.ue(3)
    for byte in (0xaa, 0x00, 0x55):
        w.u(8, byte)
    w.trailing()
    return hevc_nal(19, w, filler=b"\x80" * 24)


def slice_full_dependent(address):
    """A dependent slice segment under `pps_full`, with no entry points."""
    w = SyntaxWriter()
    w.flag(False).flag(True).ue(0)
    w.flag(True).u(5, address)
    w.ue(0)
    w.ue(0)
    w.trailing()
    return hevc_nal(19, w, filler=b"\x80" * 8)


def slice_full_b():
    """A B slice under `pps_full`: an inter-predicted RPS, long-term pictures,
    list modification, and a weight table."""
    w = SyntaxWriter()
    w.flag(True).ue(0)
    w.u(2, 0).ue(0).flag(False)
    w.u(8, 37).flag(False)
    # Slice RPS predicted from SPS set 0 (delta_idx_minus1 2): NumDeltaPocs 3.
    write_rps_inter(w, True, 2, True, 1,
                    [(True, True), (True, True), (False, False), (True, True)])
    # Long-term: two from the SPS, one in the slice.
    w.ue(2).ue(1)
    w.u(2, 0).flag(True).ue(1)
    w.u(2, 2).flag(False)
    w.u(8, 99).flag(True).flag(True).ue(2)
    w.flag(True)
    w.flag(True).flag(True)
    # num_ref_idx override: 3 in list 0, 2 in list 1.
    w.flag(True).ue(2).ue(1)
    # NumPicTotalCurr 6 entries: a 3-bit list entry.
    w.flag(True)
    for entry in (0, 3, 5):
        w.u(3, entry)
    w.flag(True)
    for entry in (2, 1):
        w.u(3, entry)
    w.flag(True).flag(True)
    # Collocated from list 1, index 1.
    w.flag(False).ue(1)
    # pred_weight_table(): denominators, flags, then values.
    w.ue(6).se(-1)
    for flag in (True, False, True):
        w.flag(flag)
    for flag in (False, True, True):
        w.flag(flag)
    w.se(-20).se(10)
    w.se(5).se(-40).se(-5).se(40)
    w.se(3).se(-3)
    w.se(1).se(1).se(2).se(2)
    for flag in (True, False):
        w.flag(flag)
    for flag in (False, True):
        w.flag(flag)
    w.se(7).se(-7)
    w.se(-1).se(-2).se(-3).se(-4)
    w.ue(1)
    w.se(-1).se(2).se(-2).flag(True)
    w.flag(True).flag(False).se(1).se(-1)
    w.flag(False)
    w.ue(0)
    w.ue(0)
    w.trailing()
    return hevc_nal(1, w, filler=b"\x80" * 16)


def slice_plain(nal_type, slice_type, chroma=True, poc_bits=8, tmvp=True, sao=True,
                entry=None, weighted=True, separate=None, set_idx=None, set_bits=0,
                bypass_rps=False, address_bits=0, address=None, lt=None):
    """A slice under `pps_plain` for the chroma format and bit depth variants."""
    w = SyntaxWriter()
    first = address is None
    w.flag(first)
    if 16 <= nal_type <= 23:
        w.flag(False)
    w.ue(0)
    if not first:
        w.u(address_bits, address)
    w.ue(slice_type)
    if separate is not None:
        w.u(2, separate)
    idr = nal_type in (19, 20)
    if not idr:
        w.u(poc_bits, 6)
        if bypass_rps:
            w.flag(False)
            write_rps_explicit(w, [(0, True)], [(0, True)], first=True)
        else:
            w.flag(True)
            if set_bits:
                w.u(set_bits, set_idx)
        if lt is not None:
            w.ue(lt).ue(0)
            for _ in range(lt):
                w.flag(False)
        if tmvp:
            w.flag(True)
    if sao:
        w.flag(True)
        if chroma:
            w.flag(True)
    if slice_type != 2:
        w.flag(False)
        if slice_type == 0:
            w.flag(False)
        if tmvp and not idr:
            if slice_type == 0:
                w.flag(True)
        if weighted:
            w.ue(5)
            if chroma:
                w.se(1)
            for lists in range(2 if slice_type == 0 else 1):
                w.flag(True)
                if chroma:
                    w.flag(True)
                w.se(-2).se(-50 if chroma else 50)
                if chroma:
                    w.se(4).se(-100).se(-4).se(100)
        w.ue(0)
    w.se(1)
    w.flag(True)
    if entry is not None:
        w.ue(len(entry))
        if entry:
            w.ue(7)
            for offset in entry:
                w.u(8, offset)
    w.trailing()
    return hevc_nal(nal_type, w, filler=b"\x80" * 16)


def synth_full():
    """Tiles with wavefront, PCM, scaling lists, RPS, long-term, range flags."""
    return [vps_nal(sub_layers=1, layer_sets=1), sps_full(), pps_full(),
            slice_full_idr(), slice_full_dependent(14), slice_full_b()]


def synth_vps_no_ordering():
    """A VPS and SPS with sub-layer ordering info only for the top layer."""
    return [vps_nal(sub_layers=2, ordering=False), vps_nal(sub_layers=0, timing=False),
            sps_plain(1), pps_plain(0, 1), slice_plain(19, 2)]


def synth_mono():
    """Monochrome: no chroma SAO flag and no chroma weights."""
    return [vps_nal(sub_layers=0), sps_plain(0), pps_plain(0, 1),
            slice_plain(19, 2, chroma=False),
            slice_plain(1, 1, chroma=False, set_idx=1, set_bits=1)]


def synth_separate_planes():
    """4:4:4 with `separate_colour_plane_flag`: `colour_plane_id` in slices."""
    return [vps_nal(sub_layers=0), sps_plain(3, separate=True), pps_plain(0, 1, weighted=False),
            slice_plain(19, 2, chroma=False, separate=0),
            slice_plain(19, 2, chroma=False, separate=2)]


def synth_444_scaling():
    """4:4:4 with scaling lists and uniform tiles."""
    return [vps_nal(sub_layers=0), sps_plain(3, scaling=True),
            pps_plain(0, 1, uniform_tiles=True, weighted=False),
            slice_plain(19, 2, entry=[9, 200, 4])]


def synth_high_precision():
    """10-bit with `high_precision_offsets_enabled_flag`: wide offset ranges."""
    return [vps_nal(sub_layers=0), sps_plain(1, depth=2, high_precision=True), pps_plain(0, 1),
            slice_plain(19, 2), slice_plain(1, 0, set_idx=0, set_bits=1)]


def synth_short_poc():
    """A 4-bit POC, one RPS in the slice, one long-term candidate in the SPS."""
    return [vps_nal(sub_layers=0), sps_plain(1, poc_bits=4, num_sets=0, long_term=1),
            pps_plain(0, 1), slice_plain(19, 2),
            slice_plain(1, 1, poc_bits=4, bypass_rps=True, lt=1)]


SYNTH_HEVC = {
    "headers_synth_full": synth_full,
    "headers_synth_vps": synth_vps_no_ordering,
    "headers_synth_mono": synth_mono,
    "headers_synth_planes": synth_separate_planes,
    "headers_synth_444_scaling": synth_444_scaling,
    "headers_synth_high_precision": synth_high_precision,
    "headers_synth_short_poc": synth_short_poc,
}


def vp9_prefix(w, profile):
    """Writes `frame_marker` and the profile bits."""
    w.u(2, 2).flag(profile & 1).flag(profile >> 1)
    if profile == 3:
        w.flag(False)


def vp9_sync(w):
    """Writes `frame_sync_code()`."""
    w.u(8, 0x49).u(8, 0x83).u(8, 0x42)


def vp9_color(w, profile, space=2, full=False, twelve=False, ss=(1, 0)):
    """Writes `color_config()`."""
    if profile >= 2:
        w.flag(twelve)
    w.u(3, space)
    if space != 7:
        w.flag(full)
        if profile & 1:
            w.flag(ss[0]).flag(ss[1]).flag(False)
    elif profile & 1:
        w.flag(False)


def vp9_common(w, width, level=20, resilient=False, seg=None, deltas=None, q=(60, 0, 0, 0),
               tile_cols=0, tile_rows=0, size=9):
    """Writes the header tail from `refresh_frame_context` to its end."""
    if not resilient:
        w.flag(True).flag(False)
    w.u(2, 2)
    w.u(6, level).u(3, 2).flag(deltas is not None)
    if deltas is not None:
        w.flag(True)
        for value in deltas[0]:
            w.flag(value is not None)
            if value is not None:
                w.su(6, value)
        for value in deltas[1]:
            w.flag(value is not None)
            if value is not None:
                w.su(6, value)
    w.u(8, q[0])
    for delta in q[1:]:
        w.flag(delta != 0)
        if delta:
            w.su(4, delta)
    w.flag(seg is not None)
    if seg is not None:
        seg(w)
    sb_cols = (((width + 7) >> 3) + 7) >> 3
    minimum = 0
    while (64 << minimum) < sb_cols:
        minimum += 1
    maximum = 1
    while (sb_cols >> maximum) >= 4:
        maximum += 1
    maximum -= 1
    cols = max(tile_cols, minimum)
    for _ in range(minimum, cols):
        w.flag(True)
    if cols < maximum:
        w.flag(False)
    if tile_rows == 0:
        w.flag(False)
    else:
        w.flag(True).flag(tile_rows == 2)
    w.u(16, size)
    w.align()
    return w.to_bytes() + b"\x80" * size


def vp9_seg_full(w):
    """Writes `segmentation_params()` with a map update and new feature data."""
    w.flag(True)
    for prob in (10, None, 30, None, 50, None, 70):
        w.flag(prob is not None)
        if prob is not None:
            w.u(8, prob)
    w.flag(True)
    for prob in (5, None, 15):
        w.flag(prob is not None)
        if prob is not None:
            w.u(8, prob)
    w.flag(True).flag(False)
    for segment in range(8):
        w.flag(True).u(8, segment * 3).flag(segment % 2 == 1)
        w.flag(segment > 3)
        if segment > 3:
            w.u(6, segment).flag(True)
        w.flag(segment == 2)
        if segment == 2:
            w.u(2, 1)
        w.flag(segment == 5)


def vp9_seg_keep(w):
    """Writes `segmentation_params()` with no updates: state persists."""
    w.flag(False).flag(False)


def vp9_key(profile, width=64, height=48, **kwargs):
    """A key frame with a render size."""
    color = {key: kwargs.pop(key) for key in ("space", "full", "twelve", "ss") if key in kwargs}
    w = SyntaxWriter()
    vp9_prefix(w, profile)
    w.flag(False).flag(False).flag(True).flag(kwargs.get("resilient", False))
    vp9_sync(w)
    vp9_color(w, profile, **color)
    w.u(16, width - 1).u(16, height - 1)
    w.flag(True).u(16, width * 2 - 1).u(16, height * 2 - 1)
    return vp9_common(w, width, **kwargs)


def vp9_intra_only(profile, width=64, height=48, refresh=0x0f, **kwargs):
    """A hidden intra-only frame, with `color_config()` above profile 0."""
    color = {key: kwargs.pop(key) for key in ("space", "full", "twelve", "ss") if key in kwargs}
    w = SyntaxWriter()
    vp9_prefix(w, profile)
    w.flag(False).flag(True).flag(False).flag(False)
    w.flag(True).u(2, 2)
    vp9_sync(w)
    if profile > 0:
        vp9_color(w, profile, **color)
    w.u(8, refresh)
    w.u(16, width - 1).u(16, height - 1).flag(False)
    return vp9_common(w, width, **kwargs)


def vp9_inter(profile, width=64, found=0, explicit=None, filter_literal=None, show=True,
              hidden_intra_bit=True, refs=(0, 1, 2), sign=(False, True, False),
              refresh=1, **kwargs):
    """An inter frame. `found` is the first `found_ref` index, or None for an
    explicit size in `explicit`."""
    w = SyntaxWriter()
    vp9_prefix(w, profile)
    w.flag(False).flag(True).flag(show).flag(False)
    if not show and hidden_intra_bit:
        w.flag(False)
    w.u(2, 1)
    w.u(8, refresh)
    for ref, bias in zip(refs, sign):
        w.u(3, ref).flag(bias)
    if found is None:
        w.flag(False).flag(False).flag(False)
        w.u(16, explicit[0] - 1).u(16, explicit[1] - 1)
        width = explicit[0]
    else:
        for _ in range(found):
            w.flag(False)
        w.flag(True)
    w.flag(False)
    w.flag(True)
    w.flag(filter_literal is None)
    if filter_literal is not None:
        w.u(2, filter_literal)
    return vp9_common(w, width, **kwargs)


def vp9_show_existing(profile, index):
    """A `show_existing_frame` header."""
    w = SyntaxWriter()
    vp9_prefix(w, profile)
    w.flag(True).u(3, index)
    w.align()
    return w.to_bytes()


def synth_vp9_profile0():
    """Segmentation and loop filter persistence, resize through refs, tiles."""
    deltas = ((2, None, -3, None), (None, -1))
    return [
        vp9_key(0, full=True, seg=vp9_seg_full, deltas=deltas, q=(50, 3, -4, 2)),
        vp9_inter(0, seg=vp9_seg_keep, deltas=((None, 5, None, None), (4, None)), filter_literal=0),
        vp9_show_existing(0, 0),
        vp9_inter(0, found=None, explicit=(96, 72), refresh=2, filter_literal=1),
        vp9_inter(0, width=96, found=1, refs=(3, 1, 0), filter_literal=3, show=False, refresh=8),
        vp9_intra_only(0, width=64, seg=vp9_seg_full),
        vp9_key(0, width=4160, height=48, tile_cols=2, tile_rows=2, resilient=False),
        vp9_inter(0, width=4160, tile_cols=1, tile_rows=1),
    ]


def synth_vp9_profiles():
    """Colour syntax of profiles 1 to 3, RGB, and error-resilient frames."""
    return [
        vp9_key(1, space=7),
        vp9_key(1, space=2, full=True, ss=(1, 0)),
        vp9_intra_only(1, space=5, full=True, ss=(0, 1)),
    ]


def synth_vp9_high():
    """Profile 2 at 12 bits with an intra-only frame, and profile 3 at 10 bits."""
    return [vp9_key(2, twelve=True, space=5), vp9_intra_only(2, twelve=True, space=5),
            vp9_inter(2)]


def synth_vp9_444():
    """Profile 3 with 4:4:4 at 10 bits and RGB at 12 bits."""
    return [vp9_key(3, ss=(0, 0), space=2), vp9_key(3, space=7, twelve=True)]


def vp9_resilient():
    """An error-resilient inter frame: no reset or context bits."""
    w = SyntaxWriter()
    vp9_prefix(w, 0)
    w.flag(False).flag(True).flag(True).flag(True)
    w.u(8, 1)
    for ref in range(3):
        w.u(3, ref).flag(False)
    w.flag(True).flag(False)
    w.flag(False).flag(True)
    return vp9_common(w, 64, resilient=True)


def synth_vp9_resilient():
    """Key frame, then an error-resilient inter frame that resets state."""
    deltas = ((1, 1, 1, 1), (1, 1))
    return [vp9_key(0, seg=vp9_seg_full, deltas=deltas), vp9_resilient()]


SYNTH_VP9 = {
    "headers_synth_vp9_p0": synth_vp9_profile0,
    "headers_synth_vp9_p1": synth_vp9_profiles,
    "headers_synth_vp9_p2": synth_vp9_high,
    "headers_synth_vp9_p3": synth_vp9_444,
    "headers_synth_vp9_resilient": synth_vp9_resilient,
}


def trace_sections(args):
    """Runs `trace_headers` and returns (extradata, kind, elements) per section."""
    result = subprocess.run(
        ["ffmpeg", *FFMPEG_LOG, *args, "-c", "copy", "-bsf:v", "trace_headers", "-f", "null", "-"],
        capture_output=True, text=True, errors="replace")
    if result.returncode != 0 or "Failed to read unit" in result.stderr:
        lines = [line for line in result.stderr.splitlines() if "trace_headers" in line]
        tail = "\n".join(lines[-8:] or result.stderr.splitlines()[-8:])
        sys.exit(f"ffmpeg {' '.join(args)} failed:\n{tail}")
    sections = []
    current = None
    in_extradata = False
    for line in result.stderr.splitlines():
        if "[trace_headers @" not in line:
            continue
        text = line.split("] ", 1)[1].strip()
        if text == "Extradata":
            in_extradata = True
        elif text.startswith("Packet:"):
            in_extradata = False
            current = None
        elif text in SECTIONS:
            current = (in_extradata, SECTIONS[text], [])
            sections.append(current)
        elif " = " in text and current is not None and text[0].isdigit():
            # `BITPOS NAME BITS = VALUE`. BITS is empty for a zero-width element.
            fields = text.split(" = ")
            head = fields[0].split()
            current[2].append((head[1], int(fields[1])))
        elif " = " not in text and text.split()[0] not in ("Frame:", "nal_unit_type:"):
            # A section title outside the five headers, such as an SEI.
            current = None
    return sections


def write_sidecar(path, pairs):
    """Writes the sections with the raw bytes each one describes."""
    with path.open("w") as out:
        for kind, raw, elements in pairs:
            out.write(f"@ {kind} {raw.hex()}\n")
            for name, value in elements:
                out.write(f"{name} {value}\n")


def split_annexb(data):
    """Splits a 4-byte start code stream of generator-made NAL units."""
    parts = data.split(START_CODE)
    return [part for part in parts if part]


def hevc_pairs(nals, sections):
    """Pairs NAL units with the header sections of the same types, in order."""
    wanted = [nal for nal in nals if (nal[0] >> 1) & 0x3f in {32, 33, 34, *range(0, 32)}]
    if len(wanted) != len(sections):
        sys.exit(f"{len(wanted)} NAL units but {len(sections)} trace sections")
    pairs = []
    for nal, (_, kind, elements) in zip(wanted, sections):
        nal_type = (nal[0] >> 1) & 0x3f
        if nal_type not in HEVC_TYPES[kind]:
            sys.exit(f"NAL type {nal_type} faces trace section `{kind}`")
        pairs.append((kind, nal, elements))
    return pairs


def hvcc_nals(config):
    """Returns the NAL units of an `hvcC` payload, in array order."""
    nals = []
    arrays = config[22]
    position = 23
    for _ in range(arrays):
        position += 1
        count = struct.unpack_from(">H", config, position)[0]
        position += 2
        for _ in range(count):
            size = struct.unpack_from(">H", config, position)[0]
            position += 2
            nals.append(config[position:position + size])
            position += size
    return nals


def sample_nals(sample):
    """Returns the NAL units of a 4-byte length-prefixed sample."""
    nals = []
    position = 0
    while position < len(sample):
        size = struct.unpack_from(">I", sample, position)[0]
        nals.append(sample[position + 4:position + 4 + size])
        position += 4 + size
    return nals


def superframe_split(packet):
    """Splits a VP9 chunk into frames, as Annex B of the VP9 specification."""
    marker = packet[-1]
    if marker >> 5 != 0b110:
        return [packet]
    frames = (marker & 7) + 1
    size_bytes = ((marker >> 3) & 3) + 1
    index = 2 + frames * size_bytes
    if len(packet) < index or packet[-index] != marker:
        return [packet]
    out = []
    position = 0
    cursor = len(packet) - index + 1
    for _ in range(frames):
        size = int.from_bytes(packet[cursor:cursor + size_bytes], "little")
        cursor += size_bytes
        if size:
            out.append(packet[position:position + size])
        position += size
    return out


def ivf_write(path, frames):
    """Writes frames into an IVF file, one packet each."""
    with path.open("wb") as out:
        out.write(b"DKIF" + struct.pack("<HH4sHHIIII", 0, 32, b"VP90", 64, 48, 25, 1,
                                        len(frames), 0))
        for index, frame in enumerate(frames):
            out.write(struct.pack("<IQ", len(frame), index) + frame)


def from_mp4_hevc(mp4, extract, work):
    """Pairs an MP4 HEVC stream with its trace."""
    name = mp4.stem
    extract(mp4, work, name)
    config = (work / f"{name}.config").read_bytes()
    framed = (work / f"{name}.samples").read_bytes()
    nals = hvcc_nals(config)
    position = 0
    while position < len(framed):
        size = struct.unpack_from(">I", framed, position)[0]
        nals += sample_nals(framed[position + 4:position + 4 + size])
        position += 4 + size
    nals = [nal for nal in nals if (nal[0] >> 1) & 0x3f not in range(35, 64)]
    return hevc_pairs(nals, trace_sections(["-i", str(mp4)]))


def from_mp4_vp9(mp4, extract, work):
    """Pairs an MP4 VP9 stream with its trace."""
    name = mp4.stem
    extract(mp4, work, name)
    framed = (work / f"{name}.samples").read_bytes()
    frames = []
    position = 0
    while position < len(framed):
        size = struct.unpack_from(">I", framed, position)[0]
        frames += superframe_split(framed[position + 4:position + 4 + size])
        position += 4 + size
    sections = trace_sections(["-i", str(mp4)])
    if len(frames) != len(sections):
        sys.exit(f"{mp4}: {len(frames)} frames but {len(sections)} trace sections")
    return [("vp9", frame, elements) for frame, (_, _, elements) in zip(frames, sections)]


def synth_hevc(name, build, work):
    """Writes an Annex-B stream and pairs its NAL units with the trace."""
    raw = work / f"{name}.hevc"
    nals = build()
    raw.write_bytes(b"".join(START_CODE + nal for nal in nals))
    sections = trace_sections(["-f", "hevc", "-i", str(raw)])
    # A raw demuxer lifts parameter sets into extradata as well as the packets.
    packets = [section for section in sections if not section[0]]
    return hevc_pairs(nals, packets)


def synth_vp9(name, build, work):
    """Writes an IVF stream and pairs its frames with the trace."""
    ivf = work / f"{name}.ivf"
    frames = build()
    ivf_write(ivf, frames)
    sections = trace_sections(["-f", "ivf", "-i", str(ivf)])
    if len(frames) != len(sections):
        sys.exit(f"{name}: {len(frames)} frames but {len(sections)} trace sections")
    return [("vp9", frame, elements) for frame, (_, _, elements) in zip(frames, sections)]


SCALING_FILE = "\n".join(
    [f"{name}=\n" + "\n".join(", ".join(str(16 + (i * j) % 40) for i in range(side))
                              for j in range(side)) + (f"\n{name}_DC=\n20\n" if dc else "\n")
     for name, side, dc in [
         *[(f"INTRA4X4_{c}", 4, False) for c in ("LUMA", "CHROMAU", "CHROMAV")],
         *[(f"INTER4X4_{c}", 4, False) for c in ("LUMA", "CHROMAU", "CHROMAV")],
         *[(f"INTRA8X8_{c}", 8, False) for c in ("LUMA", "CHROMAU", "CHROMAV")],
         *[(f"INTER8X8_{c}", 8, False) for c in ("LUMA", "CHROMAU", "CHROMAV")],
         *[(f"INTRA16X16_{c}", 8, True) for c in ("LUMA", "CHROMAU", "CHROMAV")],
         *[(f"INTER16X16_{c}", 8, True) for c in ("LUMA", "CHROMAU", "CHROMAV")],
         ("INTRA32X32_LUMA", 8, True),
         ("INTER32X32_LUMA", 8, True),
     ]])


def x265_streams(work):
    """Returns (name, pixel format, size, x265 params) of each libx265 stream."""
    scaling = work / "scaling.txt"
    scaling.write_text(SCALING_FILE)
    base = "keyint=8:min-keyint=8:bframes=3:log-level=error"
    return [
        ("headers_x265_default", "yuv420p", "128x96", base),
        ("headers_x265_wpp", "yuv420p", "640x480", f"{base}:wpp=1:pools=4:frame-threads=1"),
        ("headers_x265_weighted", "yuv420p", "128x96", f"{base}:weightp=1:weightb=1"),
        ("headers_x265_scaling_default", "yuv420p", "128x96", f"{base}:scaling-list=default"),
        ("headers_x265_scaling_file", "yuv420p", "128x96", f"{base}:scaling-list={scaling}"),
        ("headers_x265_amp", "yuv420p", "128x96", f"{base}:rect=1:amp=1"),
        ("headers_x265_refs", "yuv420p", "128x96", f"{base}:ref=4:b-pyramid=1:bframes=4"),
        ("headers_x265_temporal", "yuv420p", "128x96", f"{base}:temporal-layers=3"),
        ("headers_x265_slices", "yuv420p", "640x480", f"{base}:slices=3"),
        ("headers_x265_hrd", "yuv420p", "128x96",
         f"{base}:hrd=1:vbv-maxrate=500:vbv-bufsize=500"),
        ("headers_x265_lossless", "yuv420p", "128x96", f"{base}:cu-lossless=1:tskip=1"),
        ("headers_x265_qp_deblock", "yuv420p", "128x96",
         f"{base}:cbqpoffs=2:crqpoffs=-2:deblock=1,-1"),
        ("headers_x265_10bit", "yuv420p10le", "128x96", base),
        ("headers_x265_422", "yuv422p10le", "128x96", base),
    ]


def encode_x265(path, pixel_format, size, params):
    """Encodes an MP4 through ffmpeg's libx265 wrapper. A fade gives x265 a
    reason to send weights."""
    subprocess.run(
        ["ffmpeg", *QUIET, "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25",
         "-frames:v", "24", "-vf", "fade=in:0:24", "-pix_fmt", pixel_format, "-c:v", "libx265",
         "-x265-params", params, "-tag:v", "hvc1", str(path)],
        check=True, capture_output=True)


def encode_vpx(path, size, frames, args):
    """Encodes an MP4 through ffmpeg's libvpx wrapper."""
    subprocess.run(
        ["ffmpeg", *QUIET, "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25",
         "-frames:v", str(frames), "-pix_fmt", "yuv420p", "-c:v", "libvpx-vp9", *args, str(path)],
        check=True, capture_output=True)


def encode_vpx_two_pass(path, work):
    """Encodes with alternate reference frames: hidden frames in superframes."""
    common = ["-deadline", "good", "-cpu-used", "4", "-g", "30", "-b:v", "300k",
              "-auto-alt-ref", "1", "-lag-in-frames", "16", "-row-mt", "0",
              "-passlogfile", str(work / "vpxlog")]
    encode_vpx(work / "pass1.webm", "320x240", 40, [*common, "-pass", "1"])
    encode_vpx(path, "320x240", 40, [*common, "-pass", "2"])


def encode_vpxenc_resize(path, work):
    """Encodes through vpxenc at a low rate, so spatial resize changes the size."""
    raw = work / "resize.yuv"
    webm = work / "resize.webm"
    subprocess.run(
        ["ffmpeg", *QUIET, "-f", "lavfi", "-i", "testsrc2=size=640x480:rate=25",
         "-frames:v", "150", "-pix_fmt", "yuv420p", "-f", "rawvideo", str(raw)],
        check=True, capture_output=True)
    subprocess.run(
        ["vpxenc", "--codec=vp9", "--disable-warning-prompt", "--width=640", "--height=480",
         "--fps=25/1", "--rt", "--cpu-used=8", "--end-usage=cbr", "--resize-allowed=1",
         "--resize-up=60", "--resize-down=30", "--target-bitrate=15", "--buf-sz=200",
         "--buf-initial-sz=100", "--buf-optimal-sz=100", "--limit=150", "--webm",
         "-o", str(webm), str(raw)],
        check=True, capture_output=True)
    subprocess.run(["ffmpeg", *QUIET, "-i", str(webm), "-c", "copy", str(path)],
                   check=True, capture_output=True)
    raw.unlink()


def vp9_streams(work):
    """Returns (name, encode) of each libvpx stream."""
    realtime = ["-deadline", "realtime", "-cpu-used", "8"]
    return [
        ("headers_vp9_default", lambda p: encode_vpx(p, "128x96", 12, [*realtime, "-g", "5"])),
        ("headers_vp9_tiles", lambda p: encode_vpx(
            p, "1280x480", 6, [*realtime, "-tile-columns", "2", "-tile-rows", "1"])),
        ("headers_vp9_segmentation", lambda p: encode_vpx(
            p, "640x480", 30, ["-deadline", "good", "-cpu-used", "4", "-g", "12", "-b:v", "500k",
                               "-aq-mode", "3", "-row-mt", "0"])),
        ("headers_vp9_hidden", lambda p: encode_vpx_two_pass(p, work)),
        ("headers_vp9_resize", lambda p: encode_vpxenc_resize(p, work)),
        ("headers_vp9_error_resilient", lambda p: encode_vpx(
            p, "128x96", 8, [*realtime, "-error-resilient", "default"])),
    ]


def generate(out_dir, extract):
    """Writes every `NAME.headers` sidecar into `out_dir`."""
    with tempfile.TemporaryDirectory(dir=out_dir, prefix="headers-work-") as temporary:
        work = Path(temporary)
        for name, build in SYNTH_HEVC.items():
            write_sidecar(out_dir / f"{name}.headers", synth_hevc(name, build, work))
            print(f"{name}: synthetic HEVC")
        for name, build in SYNTH_VP9.items():
            write_sidecar(out_dir / f"{name}.headers", synth_vp9(name, build, work))
            print(f"{name}: synthetic VP9")
        for name, pixel_format, size, params in x265_streams(work):
            mp4 = work / f"{name}.mp4"
            try:
                encode_x265(mp4, pixel_format, size, params)
            except subprocess.CalledProcessError as error:
                print(f"skipping {name}: libx265 refused it: {error.stderr[-200:]!r}")
                continue
            write_sidecar(out_dir / f"{name}.headers", from_mp4_hevc(mp4, extract, work))
            print(f"{name}: libx265")
        for name, encode in vp9_streams(work):
            mp4 = work / f"{name}.mp4"
            try:
                encode(mp4)
            except (subprocess.CalledProcessError, FileNotFoundError) as error:
                print(f"skipping {name}: its encoder is missing or refused it: {error}")
                continue
            write_sidecar(out_dir / f"{name}.headers", from_mp4_vp9(mp4, extract, work))
            print(f"{name}: libvpx")
        from gen_av1_headers import generate as generate_av1
        generate_av1(out_dir, work)
    if shutil.which("vpxenc") is None:
        print("notice: vpxenc is not on PATH, so headers_vp9_resize is absent")
