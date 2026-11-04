#!/usr/bin/env python3
"""Generate the CPU HEVC decoder fixtures for the tests in `decode/cpu/hevc/stream_tests.rs`.

Run from any directory:

    python3 crates/vidr/tests/fixtures/gen_hevc.py [--out DIR] [--only NAME ...]

ffmpeg and the x265 command line tool, with 8-bit and 10-bit support, must be on
PATH. The script exits with a notice when either is missing. The default output
directory is `crates/vidr/tests/fixtures/hevc/`, which git ignores. Never commit
its contents.

For each stream NAME the script writes:

- `NAME.hevc`: the x265 Annex-B stream. Some entries edit it afterwards. `EDITS` names
  those entries.
- `NAME.yuv`: ffmpeg's decode as `-f rawvideo`, in `yuv420p` or `yuv420p10le`, in output
  order with no frame dropped or repeated. `RECON_REFERENCE` names the entries that use
  x265's reconstruction instead.

The tests read `VIDR_HEVC_FIXTURES` first, then the default directory.
"""

import argparse
import re
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
# A moving test pattern with no noise, so merge and skip blocks occur.
MOVING = "testsrc2=size={size}:rate=25"
# A fade from black, so x265 picks explicit weights.
FADE = "testsrc2=size={size}:rate=25,fade=t=in:st=0:d=1"
# One frame thread and the hash SEI message, for every inter stream.
INTER = ["--frame-threads", "1", "--hash", "1", "--crf", "30"]

# (name, size, bit depth, frames, source, x265 arguments).
STREAMS = [
    # The default scaling lists of Tables 7-5 and 7-6.
    ("scaling_default_8", "128x96", 8, 8, MOVING, [*INTER, "--scaling-list", "default"]),
    ("scaling_default_10", "128x96", 10, 8, MOVING, [*INTER, "--scaling-list", "default"]),
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
    # H12 to H17: inter pictures. Each stream checks its hash SEI messages.
    ("inter_8", "128x96", 8, 4, BUSY, [*NO_WPP, "--crf", "34", "--keyint", "4", "--hash", "1"]),
    ("p_only_8", "128x96", 8, 12, MOVING, [*INTER, *NO_WPP, "--bframes", "0"]),
    ("p_only_10", "128x96", 10, 12, MOVING, [*INTER, *NO_WPP, "--bframes", "0"]),
    ("p_noisy_8", "128x96", 8, 8, BUSY, [*INTER, *NO_WPP, "--bframes", "0"]),
    ("p_no_filters_8", "128x96", 8, 12, MOVING,
     [*INTER, *NO_WPP, *NO_FILTERS, "--bframes", "0"]),
    ("b_no_filters_8", "128x96", 8, 12, MOVING,
     [*INTER, *NO_WPP, *NO_FILTERS, "--bframes", "3", "--no-b-pyramid"]),
    ("deblock_only_8", "128x96", 8, 12, MOVING,
     [*INTER, *NO_WPP, "--no-sao", "--bframes", "3"]),
    ("sao_only_inter_8", "128x96", 8, 12, MOVING,
     [*INTER, *NO_WPP, "--no-deblock", "--bframes", "3"]),
    ("bframes_8", "128x96", 8, 16, MOVING, [*INTER, *NO_WPP, "--bframes", "3", "--no-b-pyramid"]),
    ("bframes_10", "128x96", 10, 16, MOVING,
     [*INTER, *NO_WPP, "--bframes", "3", "--no-b-pyramid"]),
    ("b_pyramid_8", "128x96", 8, 24, MOVING, [*INTER, *NO_WPP, "--bframes", "7", "--b-pyramid"]),
    ("b_pyramid_10", "128x96", 10, 24, BUSY, [*INTER, *NO_WPP, "--bframes", "7", "--b-pyramid"]),
    ("refs_4_8", "128x96", 8, 20, MOVING,
     [*INTER, *NO_WPP, "--ref", "4", "--limit-refs", "0", "--bframes", "3"]),
    ("weightp_8", "128x96", 8, 20, FADE, [*INTER, *NO_WPP, "--weightp", "--bframes", "0"]),
    ("weightp_10", "128x96", 10, 20, FADE, [*INTER, *NO_WPP, "--weightp", "--bframes", "0"]),
    ("weightb_8", "128x96", 8, 20, FADE, [*INTER, *NO_WPP, "--weightb", "--bframes", "3"]),
    ("weightb_10", "128x96", 10, 20, FADE, [*INTER, *NO_WPP, "--weightb", "--bframes", "3"]),
    ("merange_8", "128x96", 8, 12, MOVING, [*INTER, *NO_WPP, "--merange", "16", "--me", "star"]),
    ("merange_wide_8", "256x192", 8, 8, MOVING,
     [*INTER, *NO_WPP, "--merange", "128", "--me", "full"]),
    ("rect_8", "128x96", 8, 12, BUSY, [*INTER, *NO_WPP, "--rect", "--no-amp", "--rd", "4"]),
    ("amp_8", "128x96", 8, 12, BUSY, [*INTER, *NO_WPP, "--rect", "--amp", "--rd", "4"]),
    ("amp_10", "128x96", 10, 12, BUSY, [*INTER, *NO_WPP, "--rect", "--amp", "--rd", "4"]),
    ("tmvp_on_8", "128x96", 8, 16, MOVING, [*INTER, *NO_WPP, "--temporal-mvp", "--bframes", "3"]),
    ("tmvp_off_8", "128x96", 8, 16, MOVING,
     [*INTER, *NO_WPP, "--no-temporal-mvp", "--bframes", "3"]),
    ("merge_5_8", "128x96", 8, 12, MOVING, [*INTER, *NO_WPP, "--max-merge", "5", "--bframes", "3"]),
    ("merge_1_8", "128x96", 8, 12, MOVING, [*INTER, *NO_WPP, "--max-merge", "1", "--bframes", "3"]),
    ("tu_depth_8", "128x96", 8, 12, BUSY,
     [*INTER, *NO_WPP, "--tu-inter-depth", "3", "--rd", "4", "--rect"]),
    ("tu_depth_1_8", "128x96", 8, 12, BUSY,
     [*INTER, *NO_WPP, "--tu-inter-depth", "1", "--rd", "4", "--rect"]),
    ("cu_qp_inter_8", "128x96", 8, 12, BUSY,
     [*INTER, *NO_WPP, "--aq-mode", "2", "--aq-strength", "2", "--qg-size", "8"]),
    ("ctu16_inter_8", "128x96", 8, 12, MOVING, [*INTER, *NO_WPP, "--ctu", "16"]),
    ("ctu32_inter_8", "128x96", 8, 12, MOVING, [*INTER, *NO_WPP, "--ctu", "32"]),
    ("window_inter_8", "122x90", 8, 12, MOVING, [*INTER, *NO_WPP, "--bframes", "3"]),
    # At 64x64 this source gives a stream whose hash and x265 reconstruction differ from
    # ffmpeg's and dec265's decode. At this size x265, ffmpeg, dec265, and NVDEC agree.
    ("lossless_inter_8", "256x192", 8, 12, BUSY, [*NO_WPP, "--frame-threads", "1", "--hash", "1",
                                                  "--lossless"]),
    ("cu_lossless_inter_8", "64x64", 8, 8, MOVING, [*INTER, *NO_WPP, "--cu-lossless"]),
    ("wavefront_inter_8", "256x192", 8, 8, MOVING, [*INTER, "--wpp", "--ctu", "16"]),
    ("slices_inter_8", "256x192", 8, 8, MOVING, [*INTER, "--wpp", "--slices", "3", "--ctu", "16"]),
    # Decode order and output order.
    ("scene_cut_8", "128x96", 8, 24, MOVING,
     [*INTER, *NO_WPP, "--keyint", "6", "--min-keyint", "6", "--no-open-gop", "--bframes", "2"]),
    ("open_gop_8", "128x96", 8, 40, MOVING,
     [*INTER, *NO_WPP, "--open-gop", "--keyint", "12", "--min-keyint", "12", "--no-scenecut",
      "--bframes", "3"]),
    ("long_gop_8", "96x64", 8, 90, MOVING,
     [*INTER, *NO_WPP, "--keyint", "250", "--min-keyint", "250", "--no-scenecut",
      "--bframes", "4", "--ref", "4", "--limit-refs", "0"]),
    ("poc_wrap_8", "96x64", 8, 40, MOVING,
     [*INTER, *NO_WPP, "--log2-max-poc-lsb", "4", "--keyint", "250", "--min-keyint", "250",
      "--no-scenecut", "--bframes", "3"]),
    ("rasl_cut_8", "128x96", 8, 40, MOVING,
     [*INTER, *NO_WPP, "--open-gop", "--keyint", "12", "--min-keyint", "12", "--no-scenecut",
      "--bframes", "3"]),
    ("end_of_sequence_8", "128x96", 8, 40, MOVING,
     [*INTER, *NO_WPP, "--open-gop", "--keyint", "12", "--min-keyint", "12", "--no-scenecut",
      "--bframes", "3"]),
    ("no_output_prior_8", "128x96", 8, 24, MOVING,
     [*INTER, *NO_WPP, "--keyint", "12", "--min-keyint", "12", "--no-open-gop", "--no-scenecut",
      "--bframes", "3"]),
    # The stream drifts from its encoder here, and carries no hash for that reason.
    ("parallel_merge_8", "128x96", 8, 12, MOVING,
     [*NO_WPP, "--frame-threads", "1", "--crf", "30", "--bframes", "3", "--ctu", "32"]),
]

NAL_CRA = 21
NAL_IDR_W_RADL = 19
NAL_IDR_N_LP = 20
NAL_VPS = 32
NAL_SPS = 33
NAL_PPS = 34
NAL_END_OF_SEQUENCE = 36


def split_nals(data):
    """The NAL units of an Annex-B stream, without start codes."""
    starts = [match.start() for match in re.finditer(b"\x00\x00\x01", data)]
    units = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(data)
        unit = data[start + 3:end]
        units.append(unit.rstrip(b"\x00") if index + 1 < len(starts) else unit)
    return units


def join_nals(units):
    """An Annex-B stream of these NAL units."""
    return b"".join(b"\x00\x00\x00\x01" + unit for unit in units)


def nal_type(unit):
    """`nal_unit_type` of a NAL unit."""
    return unit[0] >> 1


def from_first_cra(units):
    """The parameter sets, then every unit from the first CRA picture on. The RASL pictures
    after that CRA picture lose their references."""
    first = next(index for index, unit in enumerate(units) if nal_type(unit) == NAL_CRA)
    sets = [unit for unit in units[:first] if nal_type(unit) in (NAL_VPS, NAL_SPS, NAL_PPS)]
    return first, sets


def edit_rasl_cut(units):
    """A stream that starts at a CRA picture."""
    first, sets = from_first_cra(units)
    return sets + units[first:]


def edit_end_of_sequence(units):
    """An end of sequence NAL unit, then the stream again from its CRA picture."""
    first, sets = from_first_cra(units)
    return units[:first] + [bytes([NAL_END_OF_SEQUENCE << 1, 1])] + sets + units[first:]


def edit_no_output_prior(units):
    """Sets `no_output_of_prior_pics_flag` to 1 in every IDR picture but the first."""
    edited = []
    seen_idr = False
    for unit in units:
        is_idr = nal_type(unit) in (NAL_IDR_W_RADL, NAL_IDR_N_LP)
        # The flag is the second bit after the NAL unit header, and the first is
        # first_slice_segment_in_pic_flag.
        if is_idr and seen_idr:
            unit = unit[:2] + bytes([unit[2] | 0x40]) + unit[3:]
        seen_idr = seen_idr or is_idr
        edited.append(unit)
    return edited


class Bits:
    """Reads an RBSP as a string of bits."""

    def __init__(self, payload):
        self.bits = "".join(f"{byte:08b}" for byte in payload)
        self.at = 0

    def take(self, count=1):
        """The next `count` bits as an integer."""
        value = int(self.bits[self.at:self.at + count], 2)
        self.at += count
        return value

    def ue(self):
        """One Exp-Golomb value."""
        zeros = 0
        while self.take() == 0:
            zeros += 1
        return (1 << zeros) - 1 + (self.take(zeros) if zeros else 0)


def unescape(payload):
    """Removes emulation prevention bytes."""
    return payload.replace(b"\x00\x00\x03", b"\x00\x00")


def escape(payload):
    """Adds emulation prevention bytes."""
    out = bytearray()
    zeros = 0
    for byte in payload:
        if zeros >= 2 and byte <= 3:
            out.append(3)
            zeros = 0
        out.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    return bytes(out)


def edit_parallel_merge(units, level_minus2=2):
    """Rewrites `log2_parallel_merge_level_minus2` in each PPS (7.3.2.3). x265 has no such
    option. The merge lists change, the syntax stays parseable, and ffmpeg still decodes it.
    """
    edited = []
    for unit in units:
        if nal_type(unit) != NAL_PPS:
            edited.append(unit)
            continue
        reader = Bits(unescape(unit[2:]))
        reader.ue()  # pps_pic_parameter_set_id
        reader.ue()  # pps_seq_parameter_set_id
        reader.take(2)  # dependent_slice_segments_enabled_flag, output_flag_present_flag
        reader.take(3)  # num_extra_slice_header_bits
        reader.take(2)  # sign_data_hiding_enabled_flag, cabac_init_present_flag
        reader.ue()  # num_ref_idx_l0_default_active_minus1
        reader.ue()  # num_ref_idx_l1_default_active_minus1
        reader.ue()  # init_qp_minus26, in se(v) form
        reader.take(2)  # constrained_intra_pred_flag, transform_skip_enabled_flag
        if reader.take():  # cu_qp_delta_enabled_flag
            reader.ue()  # diff_cu_qp_delta_depth
        reader.ue()  # pps_cb_qp_offset
        reader.ue()  # pps_cr_qp_offset
        reader.take(4)  # chroma offsets present, weighted_pred, weighted_bipred, bypass
        if reader.take():  # tiles_enabled_flag
            sys.exit("the parallel merge edit takes a stream with no tiles")
        reader.take(2)  # entropy_coding_sync_enabled_flag, across slices
        if reader.take():  # deblocking_filter_control_present_flag
            reader.take()  # deblocking_filter_override_enabled_flag
            if not reader.take():  # pps_deblocking_filter_disabled_flag
                reader.ue()  # pps_beta_offset_div2
                reader.ue()  # pps_tc_offset_div2
        if reader.take():  # pps_scaling_list_data_present_flag
            sys.exit("the parallel merge edit takes a stream with no scaling list")
        reader.take()  # lists_modification_present_flag
        start = reader.at
        reader.ue()  # log2_parallel_merge_level_minus2
        code = f"{level_minus2 + 1:b}"
        tail = reader.bits[reader.at:].rstrip("0")
        bits = reader.bits[:start] + "0" * (len(code) - 1) + code + tail
        bits += "0" * (-len(bits) % 8)
        body = int(bits, 2).to_bytes(len(bits) // 8, "big")
        edited.append(unit[:2] + escape(body))
    return edited


# The streams an edit changes after x265 writes them.
EDITS = {
    "rasl_cut_8": edit_rasl_cut,
    "end_of_sequence_8": edit_end_of_sequence,
    "no_output_prior_8": edit_no_output_prior,
    "parallel_merge_8": edit_parallel_merge,
}

# The streams ffmpeg 9.0.2 decodes wrongly in chroma. The hash SEI message, NVDEC, and dec265
# agree with x265's reconstruction, so that reconstruction is the reference.
RECON_REFERENCE = {
    "wavefront_inter_8", "slices_inter_8", "ctu16_inter_8_vk", "cu_lossless_inter_8_vk",
}

# The streams Vulkan Video tests read. Drivers state a minimum coded size, so each gets a
# copy named `NAME_vk` with both sides scaled to reach this shorter side. x265 labels short
# all-intra streams with a range extensions profile, and Vulkan Video refuses that, so those
# stay out.
VULKAN_MIN_SIDE = 144
VULKAN_STREAMS = {
    "inter_8", "bframes_8", "bframes_10", "b_pyramid_8", "b_pyramid_10", "p_only_8", "p_only_10",
    "slices_inter_8", "wavefront_8", "wavefront_inter_8", "weightp_8", "weightp_10", "weightb_8",
    "weightb_10", "scaling_default_8", "scaling_default_10", "window_inter_8", "open_gop_8",
    "end_of_sequence_8", "poc_wrap_8", "long_gop_8", "refs_4_8", "amp_8", "amp_10", "tmvp_off_8",
    "cu_lossless_inter_8", "rect_8", "ctu16_inter_8", "rasl_cut_8", "no_output_prior_8",
}


def vulkan_copy(stream):
    """The same stream at a size every driver takes."""
    name, size, depth, frames, source, x265_args = stream
    width, height = (int(side) for side in size.split("x"))
    scale = -(-VULKAN_MIN_SIDE // min(width, height))
    return (f"{name}_vk", f"{width * scale}x{height * scale}", depth, frames, source, x265_args)


def run(tool, args):
    """Runs a tool, and exits with its error text when it fails."""
    result = subprocess.run([tool, *args], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"{tool} {' '.join(args)} failed: {result.stderr}")


def generate(out, stream):
    """Writes one stream and ffmpeg's decode of it."""
    name, size, depth, frames, source, x265_args = stream
    base = name.removesuffix("_vk")
    recon = name in RECON_REFERENCE or base in RECON_REFERENCE
    pix_fmt = "yuv420p" if depth == 8 else "yuv420p10le"
    raw = out / f"{name}.source.raw"
    hevc = out / f"{name}.hevc"
    run("ffmpeg", [*QUIET, "-f", "lavfi", "-i", f"{source.format(size=size)},format={pix_fmt}",
                   "-frames:v", str(frames), "-f", "rawvideo", str(raw)])
    yuv = out / f"{name}.yuv"
    recon_args = ["--recon", str(yuv), "--recon-depth", str(depth)] if recon else []
    run("x265", ["--input", str(raw), "--input-res", size, "--fps", "25",
                 "--input-depth", str(depth), "--input-csp", "i420", "--output-depth", str(depth),
                 "--frames", str(frames), "--log-level", "error", *x265_args, *recon_args,
                 "-o", str(hevc)])
    raw.unlink()
    if base in EDITS:
        hevc.write_bytes(join_nals(EDITS[base](split_nals(hevc.read_bytes()))))
    if not recon:
        run("ffmpeg", [*QUIET, "-i", str(hevc), "-fps_mode", "passthrough", "-f", "rawvideo",
                       "-pix_fmt", pix_fmt, str(yuv)])


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
    copies = [vulkan_copy(stream) for stream in STREAMS if stream[0] in VULKAN_STREAMS]
    for stream in [*STREAMS, *copies]:
        if args.only is None or stream[0] in args.only:
            generate(args.out, stream)


if __name__ == "__main__":
    main()
