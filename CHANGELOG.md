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
- `vidr`: `check_output`, which compares an output stream with its source and reports every mismatch in frame count, display size, rotation, display aspect ratio, pts order, pts values, and total duration.
