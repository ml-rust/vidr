#!/usr/bin/env python3
"""Generate the media input fixtures for `tests/media/`.

Run from any directory:

    python3 crates/vidr/tests/fixtures/gen_media.py [--out DIR]

ffmpeg and ffprobe with libx264 and the `h264_metadata` filter must be on
PATH. libx265, libvpx-vp9 and an AV1 encoder are optional. A stream whose
encoder is missing is skipped with a notice. The default output directory is
`crates/vidr/tests/fixtures/media/`, which git ignores. Never commit its
contents.

For each fixture FILE (`NAME.EXT`) the script writes FILE and `FILE.meta`.
Every `key=value` line of the meta file comes from ffprobe or from the recipe:

- `container`: `mp4` or `mkv`, the bytes the recipe wrote.
- `index`, `width`, `height`, `codec`: the stream the PoC picks, which is the
  first video stream without the `attached_pic` disposition.
- `first_width`: the width of the stream `-map 0:v:0` picks.
- `video_streams`: how many video streams ffprobe lists, attached pictures
  included.
- `default_index`: the index of the first video stream with the default flag.
- `attached_pictures`: how many streams hold the `attached_pic` disposition.
- `rotation`: the side-data `rotation`, counterclockwise. `0` when absent.
- `sar`: ffprobe's `sample_aspect_ratio`, when it reports one.
- `primaries`, `transfer`, `matrix`, `range`: the stream fields ffprobe
  reports, as H.273 code points and `tv`, `pc` or `unknown`.
- `frame_primaries`, `frame_transfer`, `frame_matrix`, `frame_range`: the
  same fields of the first decoded frame.
- `raw_primaries`, `raw_transfer`, `raw_matrix`, `raw_range`: the same fields
  from the H.264 stream outside its container, so the SPS alone sets them.
- `time_base`: the stream time base, `NUM/DEN`.
- `edited_pts`: the pts of each packet in presentation order, comma
  separated, with the edit list applied.
- `extradata_adler32`: the Adler-32 of the codec config, in hexadecimal.

The tests read `VIDR_MEDIA_FIXTURES` first, then the default directory.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

FFMPEG = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error"]
FFPROBE = ["ffprobe", "-v", "error", "-of", "json"]
MAIN_SIZE = "64x48"
SMALL_SIZE = "32x24"
COVER_SIZE = "16x12"
SOURCE = ["-f", "lavfi", "-i", f"testsrc=size={MAIN_SIZE}:rate=25:duration=0.48"]
SMALL_SOURCE = ["-f", "lavfi", "-i", f"testsrc=size={SMALL_SIZE}:rate=25:duration=0.48"]
AUDIO_SOURCE = ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.2"]
X264 = ["-c:v", "libx264", "-preset", "veryfast", "-bf", "2", "-g", "6", "-pix_fmt", "yuv420p"]

PRIMARIES = {"unknown": 2, "bt709": 1, "bt470bg": 5, "smpte170m": 6, "bt2020": 9}
TRANSFER = {
    "unknown": 2,
    "bt709": 1,
    "bt470bg": 5,
    "smpte170m": 6,
    "smpte2084": 16,
    "arib-std-b67": 18,
}
MATRIX = {
    "unknown": 2,
    "bt709": 1,
    "bt470bg": 5,
    "smpte170m": 6,
    "bt2020nc": 9,
    "bt2020c": 10,
}

# Each AV1 encoder with the flags that keep it fast on a small source.
AV1_ENCODERS = [
    ("libsvtav1", ["-preset", "12"]),
    ("libaom-av1", ["-cpu-used", "8", "-row-mt", "1"]),
    ("librav1e", ["-speed", "10"]),
]


def run(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"command failed ({result.returncode}): {' '.join(map(str, args))}\n{result.stderr}")
    return result.stdout


def ffmpeg(*args):
    run([*FFMPEG, *map(str, args)])


def probe(path, *args):
    return json.loads(run([*FFPROBE, *args, str(path)]))


def encoders():
    listing = run(["ffmpeg", "-hide_banner", "-encoders"])
    return {line.split()[1] for line in listing.splitlines() if len(line.split()) > 1}


def tags(primaries, transfer, matrix, range_name):
    """The options that tag the SPS and the container with the same values."""
    frame = f"color_primaries={primaries}:color_trc={transfer}:colorspace={matrix}:range={range_name}"
    return [
        "-vf", f"setparams={frame}",
        "-color_primaries", primaries,
        "-color_trc", transfer,
        "-colorspace", matrix,
        "-color_range", range_name,
    ]


def colour_fields(entry, prefix):
    """The `prefix`-named meta lines for one ffprobe stream or frame."""
    def code(table, field):
        return table[entry.get(field, "unknown")]

    return {
        f"{prefix}primaries": code(PRIMARIES, "color_primaries"),
        f"{prefix}transfer": code(TRANSFER, "color_transfer"),
        f"{prefix}matrix": code(MATRIX, "color_space"),
        f"{prefix}range": entry.get("color_range", "unknown"),
    }


def rotation_of(stream):
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            return side["rotation"]
    return 0


def write_meta(path, container, scratch):
    """Writes `FILE.meta` from ffprobe's view of `path`."""
    streams = probe(path, "-show_streams", "-show_data_hash", "adler32")["streams"]
    videos = [stream for stream in streams if stream["codec_type"] == "video"]
    meta = {"container": container}
    moving = [v for v in videos if not v.get("disposition", {}).get("attached_pic")]
    if moving:
        video = moving[0]
        meta |= {
            "index": video["index"],
            "width": video["width"],
            "height": video["height"],
            "first_width": videos[0]["width"],
            "video_streams": len(videos),
            "default_index": next(
                (v["index"] for v in videos if v.get("disposition", {}).get("default")), "none"
            ),
            "attached_pictures": len(videos) - len(moving),
            "codec": video["codec_name"],
            "rotation": rotation_of(video),
            "time_base": video["time_base"],
        }
        sar = video.get("sample_aspect_ratio", "N/A")
        if ":" in sar:
            meta["sar"] = sar
        if "extradata_hash" in video:
            meta["extradata_adler32"] = video["extradata_hash"].split(":")[1]
        meta |= colour_fields(video, "")
        selector = ["-select_streams", str(video["index"])]
        frame = probe(path, *selector, "-read_intervals", "%+#1", "-show_frames")["frames"][0]
        meta |= colour_fields(frame, "frame_")
        packets = probe(path, *selector, "-show_entries", "packet=pts")["packets"]
        meta["edited_pts"] = ",".join(map(str, sorted(int(p["pts"]) for p in packets)))
        if video["codec_name"] == "h264":
            raw = scratch / "raw.h264"
            ffmpeg("-i", path, "-map", f"0:{video['index']}", "-c", "copy",
                   "-bsf:v", "h264_mp4toannexb", "-f", "h264", raw)
            meta |= colour_fields(probe(raw, "-show_streams")["streams"][0], "raw_")
    text = "".join(f"{key}={value}\n" for key, value in meta.items())
    path.with_name(path.name + ".meta").write_text(text)


class Writer:
    """Runs recipes and writes each result with its meta file."""

    def __init__(self, out, scratch):
        self.out = out
        self.scratch = scratch

    def make(self, name, container, *args):
        path = self.out / name
        ffmpeg(*args, path)
        write_meta(path, container, self.scratch)
        return path

    def copy(self, source, name, container, *args):
        return self.make(name, container, "-i", source, "-c", "copy", *args)

    def rotated(self, source, name, container, *rotation):
        return self.make(name, container, *rotation, "-i", source, "-c", "copy")


def basic_fixtures(writer, plain):
    for ext, container in (("mp4", "mp4"), ("mkv", "mkv")):
        source = plain[ext]
        writer.copy(source, f"plain.{ext}", container)
        writer.rotated(source, f"rotate_90.{ext}", container, "-display_rotation", "90")
        writer.rotated(source, f"rotate_180.{ext}", container, "-display_rotation", "180")
        writer.rotated(source, f"rotate_270.{ext}", container, "-display_rotation", "-90")
        writer.rotated(source, f"rotate_45.{ext}", container, "-display_rotation", "45")
        writer.rotated(source, f"mirror.{ext}", container, "-display_rotation", "90", "-display_hflip")
        writer.make(f"anamorphic.{ext}", container, *SOURCE, *X264, "-vf", "setsar=8/9")
    writer.copy(plain["mp4"], "plain.mov", "mp4")
    writer.make("edit_offset.mp4", "mp4", "-itsoffset", "0.5", "-i", plain["mp4"], "-c", "copy")
    writer.make("fragmented.mp4", "mp4", "-i", plain["mp4"], "-c", "copy",
                "-movflags", "frag_keyframe+empty_moov")


def wrong_extension_fixtures(writer, plain):
    """MP4 bytes named `.mkv` and Matroska bytes named `.mp4`."""
    for source, name, container in (
        (plain["mp4"], "mp4_bytes.mkv", "mp4"),
        (plain["mkv"], "mkv_bytes.mp4", "mkv"),
    ):
        path = writer.out / name
        shutil.copyfile(source, path)
        write_meta(path, container, writer.scratch)


def track_choice_fixtures(writer, small):
    """A second video track flagged default, an attached picture, no video."""
    for ext, container in (("mp4", "mp4"), ("mkv", "mkv")):
        writer.make(f"second_default.{ext}", container, *SOURCE, *SMALL_SOURCE, *X264,
                    "-map", "0", "-map", "1",
                    "-disposition:v:0", "0", "-disposition:v:1", "default")
    cover = writer.scratch / "cover.png"
    ffmpeg("-f", "lavfi", "-i", f"testsrc=size={COVER_SIZE}", "-frames:v", "1", cover)
    writer.make("attached_picture.mp4", "mp4", "-i", small, "-i", cover,
                "-map", "1", "-map", "0", "-c", "copy", "-disposition:v:0", "attached_pic")
    writer.make("attached_picture.mkv", "mkv", "-i", small, "-attach", cover,
                "-metadata:s:t", "mimetype=image/png", "-c", "copy")
    writer.make("audio_only.mp4", "mp4", *AUDIO_SOURCE, "-c:a", "aac", "-b:a", "48k")
    writer.make("audio_only.mkv", "mkv", *AUDIO_SOURCE, "-c:a", "aac", "-b:a", "48k")


def colour_fixtures(writer):
    """Every way the SPS and the container can agree or disagree on colour."""
    bt709 = ("bt709", "bt709", "bt709")
    for ext, container in (("mp4", "mp4"), ("mkv", "mkv")):
        limited = writer.make(f"colour_limited.{ext}", container, *SOURCE, *X264, *tags(*bt709, "tv"))
        full = writer.make(f"colour_full.{ext}", container, *SOURCE, *X264, *tags(*bt709, "pc"))
        untagged = writer.out / f"plain.{ext}"
        writer.copy(limited, f"colour_disagree.{ext}", container, "-bsf:v",
                    "h264_metadata=colour_primaries=5:transfer_characteristics=6:matrix_coefficients=5")
        writer.copy(limited, f"colour_sps_unspecified.{ext}", container, "-bsf:v",
                    "h264_metadata=colour_primaries=2:transfer_characteristics=2:matrix_coefficients=2")
        writer.copy(untagged, f"colour_container_only.{ext}", container,
                    "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-colorspace", "bt709", "-color_range", "pc")
        writer.copy(limited, f"colour_sps_full.{ext}", container, "-bsf:v",
                    "h264_metadata=video_full_range_flag=1")
        writer.copy(full, f"colour_sps_limited.{ext}", container, "-bsf:v",
                    "h264_metadata=video_full_range_flag=0")
        writer.make(f"colour_pq.{ext}", container, *SOURCE, *X264,
                    *tags("bt2020", "smpte2084", "bt2020nc", "tv"))
        writer.make(f"colour_hlg.{ext}", container, *SOURCE, *X264,
                    *tags("bt2020", "arib-std-b67", "bt2020nc", "tv"))
    # QuickTime `nclc` carries no range. ffmpeg writes it in a `.mov` file.
    writer.make("colour_nclc.mov", "mp4", *SOURCE, *X264, *tags(*bt709, "pc"))


def codec_fixtures(writer, available):
    """One stream per codec NVDEC takes, in each container that holds it."""
    if "libx265" in available:
        x265 = ["-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "log-level=none",
                "-pix_fmt", "yuv420p"]
        writer.make("hevc.mp4", "mp4", *SOURCE, *x265, "-tag:v", "hvc1")
        writer.make("hevc.mkv", "mkv", *SOURCE, *x265)
    else:
        print("skipping hevc fixtures: ffmpeg has no libx265")
    if "libvpx-vp9" in available:
        vp9 = ["-c:v", "libvpx-vp9", "-cpu-used", "8", "-deadline", "realtime",
               "-b:v", "100k", "-pix_fmt", "yuv420p"]
        writer.make("vp9.mp4", "mp4", *SOURCE, *vp9)
        writer.make("vp9.webm", "mkv", *SOURCE, *vp9)
    else:
        print("skipping vp9 fixtures: ffmpeg has no libvpx-vp9")
    for encoder, flags in AV1_ENCODERS:
        if encoder in available:
            av1 = ["-c:v", encoder, *flags, "-pix_fmt", "yuv420p"]
            writer.make("av1.mp4", "mp4", *SOURCE, *av1)
            writer.make("av1.mkv", "mkv", *SOURCE, *av1)
            break
    else:
        print("skipping av1 fixtures: ffmpeg has no AV1 encoder")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parent / "media"
    parser.add_argument("--out", type=Path, default=default)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / ".scratch"
    scratch.mkdir(exist_ok=True)

    available = encoders()
    if "libx264" not in available:
        sys.exit("ffmpeg has no libx264. Install an ffmpeg build with libx264.")

    plain = {}
    for ext in ("mp4", "mkv"):
        plain[ext] = scratch / f"plain_source.{ext}"
        ffmpeg(*SOURCE, *X264, plain[ext])
    small = scratch / "small_source.mp4"
    ffmpeg(*SMALL_SOURCE, *X264, small)

    writer = Writer(out, scratch)
    basic_fixtures(writer, plain)
    wrong_extension_fixtures(writer, plain)
    track_choice_fixtures(writer, small)
    colour_fixtures(writer)
    codec_fixtures(writer, available)
    shutil.rmtree(scratch)
    print(f"wrote fixtures to {out}")


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
