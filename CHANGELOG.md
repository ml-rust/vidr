# Changelog

All notable changes to this project are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each release must have a non-empty section here before it can be tagged — `.github/workflows/release-validate.yml` refuses a tag whose version has no entry, and the GitHub Release body is this file's section for that version. vidr and vidr-remaster release together at one version.

## [Unreleased]

### Added

- `vidr`: exact frame timing model. `Rational` time values with checked arithmetic and named rounding modes, `rescale`, `TrackTiming` with the last-frame duration chain and total duration, and `EditTimeline` for MP4-style edit lists at rate 1.
- `vidr`: frame geometry. `Rotation` in multiples of 90 degrees from degrees or a `tkhd` display matrix, sample and display aspect ratios.
- `vidr`: bitstream filters behind the `h264`, `hevc`, and `av1` features.
  - `ToAnnexB` turns MP4 and Matroska samples into the Annex-B access units NVDEC takes. Its output equals ffmpeg's `h264_mp4toannexb` and `hevc_mp4toannexb` byte for byte. One exception: an HEVC sample with in-band VPS, SPS, and PPS gets no duplicate copy.
  - `FromAnnexB` turns NVENC Annex-B access units into samples, and builds `avcC` or `hvcC` from the first access unit.
  - `Av1ToDecoder` and `Av1FromEncoder` convert AV1 samples and temporal units, and build `av1C`.
  - `AvcConfig`, `HevcConfig`, and `Av1Config` parse and write the config records. The `annexb`, `length_prefixed`, and `av1::obu` modules split and join NAL units and OBUs.
- `vidr`: NVDEC hardware decode behind the `nvdec` feature, and the backend-neutral `Decoder` trait and `DecodedFrame` behind `decode`.
  - `NvdecDecoder` decodes H.264, HEVC, AV1, and VP9, 4:2:0 at 8 or 10 bits, on numr's CUDA context and stream. `libnvcuvid.so.1` loads at run time.
  - `DecodedFrame` holds dense NV12 or P010 planes as numr tensors, the pts, the display size, and the colour facts. The planes go straight to `ColorOps::yuv_to_rgb`.
  - `ColorFacts::chroma_location` comes from `av1C` for AV1 and from the SPS VUI for H.264 and HEVC, through `h264_sps_vui` and `hevc_sps_vui`. numr models the horizontal siting only, so the vertical offset of H.273 types 2 to 5 is kept in `ChromaLocation` and not applied.
- `vidr`: NVENC hardware encode behind the `nvenc` feature, and the backend-neutral `Encoder` trait, `EncodeConfig`, `EncodeFrame`, and `EncodedPacket` behind `encode`.
  - `NvencEncoder` encodes H.264, HEVC, and AV1 as far as the GPU lists them, from NV12 or P010 planes on numr's CUDA context and stream. `libnvidia-encode.so.1` loads at run time, and the driver must support NVENC API 13.1.
  - Packets come out in decode order as MP4 samples, with exact pts, dts, and duration. `config_record` returns `avcC`, `hvcC`, or `av1C` after the first packet.
  - A codec, bit depth, B-frame count, or size the GPU refuses returns a named `EncodeError`. So does an odd H.264 or HEVC size, and a plane that does not match the config.
- `vidr-remaster`: the pure logic of the Python proof of concept, as a library. The crate has no command-line interface yet.
  - `choose_resolution` picks even output dimensions for the modes `auto`, `original`, `2x`, and the standard shorter edges 480 to 4320, with a 480-pixel minimum.
  - `Request::validate` and `output_path` check the source, the output name and container, an existing output, the output directory, the encoder quality, and the `--from-upscaled` and `--force-ai` pair. Quality runs from 1 to 51, because NVENC refuses 0.
  - `record_upscaled` and `check_record` write and check the `.upscale.json` sidecar, schema 1. The text equals the PoC's byte for byte. The write is atomic and the read stops at 64 KiB.
  - `find_upscaled` selects a matching full-length upscale from the candidates beside the source, newest first. A `VideoProbe` supplies the video facts, and `check_match` compares them.
- `vidr`: `ChromaLocation::to_h273` and `ChromaLocation::to_av1`.
- `vidr`: `DecodedFrame::progressive` and `DecodedFrame::top_field_first`, from NVDEC's `CUVIDPARSERDISPINFO`. The flags describe how the picture was coded, not the content. To decide whether a stream is interlaced, read the SPS.
- `vidr`: `VuiSignal::sample_aspect_ratio` for H.264 and HEVC. `aspect_ratio_idc` 1 to 16 map through the standards' table, and idc 255 gives the sent `sar_width` and `sar_height`. Idc 0, the reserved values, and an extended ratio with a zero part give `None`.
- `vidr`: `MATRIX_FCC` and `MATRIX_SMPTE240M`. `ColorFacts::matrix` maps them to numr's `ColorMatrix::Fcc` and `ColorMatrix::Smpte240m`, so `yuv_spec` now returns a spec for these two codes.
- `vidr`: frame conversion behind the `frame` feature, on any numr backend whose `ColorOps` takes U8 and U16 planes.
  - `frame::to_rgb` converts a `DecodedFrame` in NV12 or P010 to float RGB in F32 or F16, in `ConvLayout::Nchw` or `ConvLayout::Nhwc`. An odd AV1 or VP9 display size pads luma by replication on the odd side, converts, and cuts the RGB back to `[H, W]` on the device.
  - `frame::from_rgb` converts even-size float RGB to `OutputPlanes`, and `EncodeFrame::from(&planes)` borrows them for an encoder. An odd size returns `FrameError::OddRgbSize`.
  - `Error::Frame` holds the new `FrameError`. `DecodedFrame::from_planes` builds a frame from planes outside a decoder. `PixelFormat::sample_dtype` names a format's numr dtype.
  - The feature enables `numr/f16`, which F16 RGB needs on the CPU backend.
- `vidr`: colour policy in `color`, behind `decode`.
  - `check_sdr` refuses transfer 16 or 18, matrix 9 or 10, and primaries 9, and names the code. Transfers 14 and 15 alone pass.
  - `color_plan` runs `check_sdr`, then returns one `YuvSpec` for decode and encode, and the output tags. An unspecified matrix converts as BT.601 and stays untagged. A matrix outside BT.709, the BT.601 variants, FCC, SMPTE 240M, and unspecified returns `ColorError::UnsupportedMatrix`. The chroma location passes through.
  - `merge_container`, behind `mp4` or `mkv`, fills unspecified bitstream fields from the container. A specified bitstream value wins, which matches ffprobe's frame fields and differs from its stream fields. `full_range` takes full from either side, because the flag has no unspecified state.
- `vidr`: media input over containr in `media`, behind the `mp4` and `mkv` features. Neither needs a GPU.
  - `Input::open` picks the container from the file bytes, never the extension. Unknown bytes return `MediaError::UnknownFormat`, and fragmented MP4 returns `MediaError::Fragmented` with containr's message.
  - `video_track` returns the first video track in file order, whichever track holds the default flag.
  - `edit_timeline` converts the raw edit list with the movie timescale. `rotation` reads the display matrix or `ProjectionPoseRoll`, and refuses a mirror, another angle, yaw, pitch, or a roll on a spherical projection. `sample_aspect_ratio` and `colour` return what the container states.
  - `decode_config` returns the codec and config record `NvdecDecoder::new` takes. It refuses a Matroska `ContentEncoding`, more than one MP4 sample description, and a missing config record.
- `vidr`: media output over containr in `media`, behind the `mp4` and `mkv` features. Neither needs a GPU, and both now enable `encode`.
  - `OutputKind::from_path` takes `.mp4` and `.mkv`, in any case, and refuses other extensions by naming those two.
  - `plan_copy` runs every `StreamCopy` plan and a dry run of the whole mux before GPU work. Audio, text subtitles, chapters, global metadata, and copied track metadata copy, with no transcode. `CopyPlan::dropped` lists extra video, data, and `tmcd` tracks. Attachments and any stream or item the output cannot hold are refused, with the output kind that holds them.
  - MP4 to MKV rounds copied tracks to a 1 ms `TimestampScale`. `CopyPlan::max_rounding_error_ns` reports the error, at most half a tick. A chapter time with no whole nanosecond is refused and names MP4, never rounded. Matroska to Matroska copies exactly.
  - `OutputMux` writes in one pass, starting after the encoder's first packet, with `moov` first in MP4. `VideoOutput` states the codec, config record, size, sample aspect ratio, rotation, time base, colour tags, language, and start offset. The offset is an empty edit in MP4 and a pts shift in MKV. HEVC uses `hvc1`.
  - MKV video pts round half-even from the first frame, in integers, so `check_output` sees the exact rescale it compares. A pts on the absolute time line is within one tick, and the total duration within half a tick. Two frames on one tick are refused and name MP4.
  - Copied packets demux one at a time by decode time. `OutputMux::buffered_bytes` and `CopyPlan::with_max_buffer` state and set the memory bound. MKV holds its attachments in memory from `finish` until it writes them.
- `vidr`: `read_stream_facts` fills `StreamFacts` from a demuxed file with the edit list applied. `read_track_facts` and `check_tracks` compare audio, subtitle, chapter, and copied packet counts against `CopyPlan::expected_tracks`, with new `Deviation` variants. A text subtitle's count leaves out empty gap samples.
- `vidr`: `Rotation::display_matrix`, the inverse of `Rotation::from_display_matrix`.
- `vidr`: `SampleAspectRatio::from_fraction`, which reduces wide parts. `Error::Color` and `Error::Media` hold the new `ColorError` and `MediaError`.
- `vidr`: `check_output`, which compares an output stream with its source and reports every mismatch in frame count, display size, rotation, display aspect ratio, pts order, pts values, and total duration.
- `vidr`: the `vp9` feature, which `nvdec` enables. `vp9_color` reads `color_space` and `color_range` from one VP9 frame header, and `vp9_packet_color` reads every frame of a chunk, superframes included. `Vp9Color` holds the values and the bit depth.

### Fixed

- `vidr`: NVDEC decode of VP9 reported matrix coefficients 0 (identity) and limited range for every stream, because NVDEC returns an all-zero video signal description for VP9. `ColorFacts::yuv_spec` returned `None`, so frame conversion refused a BT.709 VP9 stream. `NvdecDecoder` now reads `color_space` and `color_range` from the latest key frame header, maps `color_space` to H.273 `MatrixCoefficients`, and reports transfer and primaries as unspecified, because VP9 carries neither. A stream with no key frame header yet reports matrix 2, never 0.
