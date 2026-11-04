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
- `vidr-remaster`: the pure logic of the Python proof of concept, as a library.
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
- `vidr`: the threaded pump behind the `pipeline` feature, which needs no GPU. `pipeline::run` takes a `PacketSource`, a `Decoder`, a `FrameTransform`, an `Encoder`, and a `PacketSink`.
  - T1 demuxes and decodes, T2 applies the edit window and the transform, and T3 encodes and writes, over channels of capacity 2 and 1. A blocked sink stops T2 from queuing kernels, then stops T1 from parsing. `MAX_IN_FLIGHT_FRAMES` states the bound.
  - The first error stops every stage and returns as the root cause. A panic returns `PipelineError::Panic`, which names the stage. `PumpControl::request_stop` ends a run with `PipelineError::Stopped`, and the sink never finishes.
  - T2 holds one frame back before the transform, so a frame outside the edit window costs no work, and each frame's duration is the next output pts minus its own. The last frame's duration comes from the chain `TrackTiming` uses, from the same inputs as `read_stream_facts`. A frame count mismatch between the decoder and the encoder is `PipelineError::CountMismatch`.
  - `media::InputSource` and `media::MuxSink` connect `Input` and `OutputMux` to the pump. `MuxSink` starts the output when the encoder's first packet supplies the config record.
  - `Error::Pipeline` holds the new `PipelineError`.
- `vidr`: `Rotation::display_matrix`, the inverse of `Rotation::from_display_matrix`.
- `vidr`: `SampleAspectRatio::from_fraction`, which reduces wide parts. `Error::Color` and `Error::Media` hold the new `ColorError` and `MediaError`.
- `vidr`: `check_output`, which compares an output stream with its source and reports every mismatch in frame count, display size, rotation, display aspect ratio, pts order, pts values, and total duration.
- `vidr`: the `vp9` feature, which `nvdec` enables. `vp9_color` reads `color_space` and `color_range` from one VP9 frame header, and `vp9_packet_color` reads every frame of a chunk, superframes included. `Vp9Color` holds the values and the bit depth.
- `vidr-remaster`: `remaster`, one whole run, and the `vidr-remaster` command through `main_with_args`. A run needs `cuda` and `mp4` or `mkv`, so it needs an NVIDIA GPU until a vendor-neutral path lands. A build without them validates the request, writes a report, and names the missing features.
  - A run plans the stream copy, then pre-scans the source with one full NVDEC decode. The scan refuses interlaced or MBAFF-coded frames, a mid-stream change of size, pixel format, or colour, pts that do not increase, HDR, and a decoded frame count that differs from the packet count.
  - `find_upscaled` selects an earlier upscale with a probe over `read_stream_facts` and the same scan. It also refuses a candidate whose rotation differs from the source's.
  - The encode pass writes inside a temporary directory beside the output, with `Preset::P5`, `VbrQuality`, and no B-frames. The frame rate is a hint reduced to 32-bit parts by continued fraction. NVENC's refusal of 10-bit H.264 becomes an error that names `--codec hevc`.
  - The output check re-decodes the output, then runs `check_output` and `check_tracks`. A mismatch is an error, and the temporary output never publishes.
  - `TempOutput` publishes with a hard link, or with a rename under `--overwrite`. On a file system with no hard links, an exclusive create keeps the no-overwrite rule.
  - `report.json` goes under `$XDG_STATE_HOME/vidr-remaster/upscale-<run id>/`, else `~/.local/state`, on success, failure, and stop. It adds the model's licence notice, dropped tracks, and the rounding error of copied tracks.
  - The command takes the proof of concept's flags without `--backend`, and `--models-dir`. It exits 0, 1, or 130. Ctrl-C, SIGTERM, and SIGHUP ask the run to stop, which removes the temporary output. A second signal ends the process at once.
- `vidr`: `Input::bitstream_sample_aspect_ratio` reads the VUI of the first SPS in an H.264 or HEVC config record. `read_stream_facts` falls back to it when the container states no ratio.
- `vidr`: `pipeline::PumpControl`, `PumpProgress`, and `PumpReport` are in every build, so a caller holds a stop request with no `pipeline` feature.
- `vidr-remaster`: end-to-end tests in `tests/e2e/` with `tests/e2e.rs` as the root, behind `cuda`, `mp4`, `mkv`, `model-srvgg`, and `model-rrdb`. The nextest `nvenc` group caps them. A missing GPU, model, or ffmpeg skips with a notice.
- `vidr-remaster`: `RunOutcome::OutputWithoutSidecar`, `ReportStatus::OutputWithoutSidecar`, `MissingSidecar`, `Published`, `TempOutput::publish_with_sidecar`, and `remove_temp_directories`.
- README: a `vidr-remaster` usage section.
- `vidr`: plane forms. `PlaneForm::Samples` holds U8 or U16 planes and `PlaneForm::Words` holds numr's packed U32 words, the only form numr's WebGPU kernels take.
  - `DecodedFrame::from_packed_planes`, `EncodeFrame::from_packed_planes`, and `OutputPlanes::from_packed_planes` build a word-form frame. `plane_form` reads the form on each type.
  - `to_rgb`, `to_rgb_reusing`, `DecodedFrame::planes`, and the plane checks follow the form. An odd word frame pads luma with `ColorOps::pad_packed_luma`. `from_rgb` allocates words when `Runtime::packed_color_planes` is true.
  - The `wgpu` feature passes through to `numr/wgpu`. Tests in `tests/frame/wgpu.rs` compare the WebGPU path with the CPU path bit for bit, and skip with a notice without an adapter.
- `vidr`: `FrameRoute` states how a decoder or encoder moves each frame: `Device`, `HostUpload`, `HostDownload`, or `HostOnly`. `PumpReport` and `PumpProgress` gain `host_copied_frames` and `host_copied_bytes`, which the pump computes from each route and the frame sizes.
- `vidr`: `Decoder` and `Encoder` impls for `Box<T>`, with `T` unsized, so a run takes `Box<dyn Decoder<R> + Send>`.
- `vidr`: CPU codecs, each with one host copy per frame that `FrameRoute` declares. `Dav1dDecoder` behind `dav1d` decodes AV1 at 4:2:0, 8 or 10 bits, on libdav1d 1.3.0 or newer. `Openh264Decoder` and `Openh264Encoder` run on OpenH264 compiled from source behind `openh264`, or on Cisco's binary at a caller's path behind `openh264-cisco`. The decoder takes Baseline only and refuses 10 bits, interlaced streams, and other profiles by name. The encoder writes 8-bit Constrained Baseline and refuses P010 and B-frames by name. A source build gets no Cisco patent cover, and Cisco's cover applies only to its own binary. `h264_sps_facts`, `av1_sequence_facts`, `upload_route`, and `download_route` support them.
- `vidr`: host plane copies in `frame`, for a CPU codec. `upload_planes` interleaves I420 to NV12, writes I010 as `value << 6`, and packs to words on a word-form backend. `download_planes` reverses it into a `HostFrame`, whose buffers a same-size frame reuses. `HostPlanes` takes pitched sources, odd sizes round chroma up, and `FrameError` gains `HostFrameSize`, `HostStride`, `HostPlaneShort`, and `HostSampleRange`.

### Changed

- `vidr`: `Decoder::route` and `Encoder::route` are required methods with no default, so every implementor adds them.
- `vidr`: `OutputPlanes` has a private field. Build one with `OutputPlanes::new` in place of a struct literal. `y` and `uv` stay public.
- `vidr`: `DtsClock` moves from `encode::nvenc` to `encode`, behind `encode`, so another encoder can reuse it.
- `vidr-remaster`: `RunOutcome::Complete::report` is now `Option<PathBuf>`. `None` means the report did not write.

### Fixed

- `vidr-remaster`: a second signal removes the temporary directory before `process::exit`. `TempOutput` registers its directory, and `remove_temp_directories` removes every registered one. `ctrlc` calls the handler on its own thread, outside signal context, so the removal runs there.
- `vidr-remaster`: the provenance sidecar is written beside the temporary output before the publish, and renames into place after it. A failure to hash or write it now leaves no output. Only the rename can fail after the publish, and that gives `RunOutcome::OutputWithoutSidecar`, report status `output_without_sidecar`, exit code 1, and an error that names both paths. The sidecar bytes are unchanged, since a record holds no output path.
- `vidr-remaster`: a report that does not write after the publish keeps the run complete. The command prints a warning with the report path and the cause, and exits 0.
- `vidr-remaster`: `report.json` is written atomically: a temporary file in the same directory, a sync, then a rename. It keeps the mode a plain write gives, and so does the sidecar, which was owner-only before.
- `vidr`: NVDEC decode of VP9 reported matrix coefficients 0 (identity) and limited range for every stream, because NVDEC returns an all-zero video signal description for VP9. `ColorFacts::yuv_spec` returned `None`, so frame conversion refused a BT.709 VP9 stream. `NvdecDecoder` now reads `color_space` and `color_range` from the latest key frame header, maps `color_space` to H.273 `MatrixCoefficients`, and reports transfer and primaries as unspecified, because VP9 carries neither. A stream with no key frame header yet reports matrix 2, never 0.
- `vidr`: `frame::to_rgb_reusing` writes an even frame into a buffer the caller keeps, and `frame::reuse_tensor` keeps such a buffer. `FrameError::OutOfMemory` separates an allocation failure from other numr errors.
- `vidr-remaster`: the model catalog in `catalog`, in the default build, with no network code.
  - `catalog_entry` states each model's directory name, download URL, licence, network, and Cargo feature.
  - `models_root_from_env` reads `--models-dir` or `VIDR_MODELS_DIR`. `resolve_model_dir` returns the model's directory. A missing directory or checkpoint names the expected path and the URL.
  - `require_model` refuses a model by name when its feature is off. `licence_notice` states that `purevision` uses CC BY-NC-SA 4.0, for a run's report and for these errors.
- `vidr-remaster`: `RemasterTransform`, a `FrameTransform` over any numr backend with the `transform` feature, and the features `mp4`, `mkv`, `h264`, `hevc`, `model-srvgg`, `model-rrdb`, `f16`, and `cuda`.
  - `load` and `from_model` run `to_rgb` at F16 NHWC, `SrModel::forward_native`, a cast to F32, an antialiased bicubic resize to the target, and `from_rgb`. An equal size skips the cast and the resize. `from_rgb` clamps RGB to 0..1, as the PoC does.
  - `resize_only` runs the same steps without a model, for `--from-upscaled`.
  - The output format follows each frame's bit depth, and `ColorPlan` supplies both colour specs. RGB and resize buffers persist across frames of one size. Output planes are new per frame, because the encoder holds them.
  - An allocation failure names the frame size, the model, and the next action: `--model fast` or a smaller source.
- `vidr-remaster`: `coded_output_size` turns a display output size into the encoder's size, with the sides swapped for a 90 or 270 degree source.
