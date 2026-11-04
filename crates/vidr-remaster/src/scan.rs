//! The decisions of a pre-scan, on the plain facts of each decoded frame.
//!
//! `NvdecDecoder` replaces its hardware decoder when a new sequence starts, and
//! says nothing. So a scan compares every frame with the first. The decode loop
//! lives in `run_frames`, and every decision lives here.

use std::fmt;
use std::path::PathBuf;

use serde_json::{Value, json};
use vidr::color::{check_sdr, merge_container};
use vidr::decode::{ColorFacts, PixelFormat};
use vidr::media::ContainerColour;
use vidr::{
    ColorError, Dimensions, Rational, Rotation, StartOffset, StreamFacts, TrackFacts,
    TrackPacketCount, TrackTiming,
};

use crate::error::RemasterError;
use crate::rate::frame_rate_hint;
use crate::reuse::VideoFacts;

/// What a scan reads from one decoded frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScanFrame {
    /// Presentation timestamp.
    pub pts: i64,
    /// Decoded size, after the stream's cropping.
    pub display: Dimensions,
    /// Sample layout.
    pub format: PixelFormat,
    /// Colour facts from the sequence header.
    pub color: ColorFacts,
    /// Whether the decoder reports the frame as progressive.
    pub progressive: bool,
}

/// Why a scan refuses a video. Frame numbers count from 1 in display order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScanRefusal {
    /// A frame is interlaced or MBAFF-coded.
    NotProgressive {
        /// The frame.
        frame: u64,
    },
    /// A frame differs in size from the first.
    SizeChange {
        /// The frame.
        frame: u64,
        /// The first frame's size.
        first: Dimensions,
        /// This frame's size.
        found: Dimensions,
    },
    /// A frame differs in sample layout from the first.
    FormatChange {
        /// The frame.
        frame: u64,
        /// The first frame's layout.
        first: PixelFormat,
        /// This frame's layout.
        found: PixelFormat,
    },
    /// A frame differs in colour facts from the first.
    ColourChange {
        /// The frame.
        frame: u64,
    },
    /// A frame's pts does not exceed the pts before it.
    PtsNotIncreasing {
        /// The frame.
        frame: u64,
        /// The pts before it.
        previous: i64,
        /// Its pts.
        pts: i64,
    },
    /// The colour marks HDR video.
    Hdr(ColorError),
    /// The decoder returns no frame.
    NoFrames {
        /// The packets read.
        packets: u64,
    },
    /// The decoder returns another number of frames than the track has packets.
    CountMismatch {
        /// The frames returned.
        frames: u64,
        /// The packets read.
        packets: u64,
    },
}

/// A scan's refusal of the video at `path`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScanError {
    /// The video.
    pub path: PathBuf,
    /// Why the scan refuses it.
    pub refusal: ScanRefusal,
}

impl fmt::Display for ScanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let path = self.path.display();
        match self.refusal {
            ScanRefusal::NotProgressive { frame } => write!(
                f,
                "Frame {frame} of {path} is interlaced or MBAFF-coded, and the frame transform \
                 needs progressive frames. A progressive source coded as MBAFF is refused too. \
                 Run a deinterlace step on the source first."
            ),
            ScanRefusal::SizeChange {
                frame,
                first,
                found,
            } => write!(
                f,
                "Frame dimensions change inside {path}: frame {frame} is {found}, and the first \
                 frame is {first}. Split the source into constant-resolution videos first."
            ),
            ScanRefusal::FormatChange {
                frame,
                first,
                found,
            } => write!(
                f,
                "Pixel format changes inside {path}: frame {frame} is {found}, and the first \
                 frame is {first}. Split the source at the change first."
            ),
            ScanRefusal::ColourChange { frame } => write!(
                f,
                "Colour facts change inside {path} at frame {frame}. Split the source at the \
                 change first."
            ),
            ScanRefusal::PtsNotIncreasing {
                frame,
                previous,
                pts,
            } => write!(
                f,
                "Missing or nonincreasing frame timestamp at frame {frame} in {path}: pts {pts} \
                 follows {previous}. Repair source timing first."
            ),
            ScanRefusal::Hdr(error) => write!(f, "HDR video is unsupported at {path}: {error}"),
            ScanRefusal::NoFrames { packets } => write!(
                f,
                "No decoded video frames in {path} from {packets} packet(s). Choose a playable \
                 video file."
            ),
            ScanRefusal::CountMismatch { frames, packets } => write!(
                f,
                "The decoder returns {frames} frame(s) from {packets} packet(s) of {path}, and \
                 the output check needs one frame per packet. Repair or remux the source first. \
                 A stream that sets `no_output_of_prior_pics_flag` needs a re-encode."
            ),
        }
    }
}

impl std::error::Error for ScanError {}

/// What a complete scan establishes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScanFacts {
    /// Frames the decoder returned. The edit list has not removed any.
    pub frames: u64,
    /// The decoded size of every frame.
    pub display: Dimensions,
    /// The sample layout of every frame.
    pub format: PixelFormat,
    /// The bitstream's colour facts, with container tags filling the gaps.
    pub color: ColorFacts,
}

/// The decisions of one scan, one frame at a time.
#[derive(Clone, Copy, Debug)]
pub struct ScanState {
    container: Option<ContainerColour>,
    first: Option<ScanFrame>,
    last_pts: i64,
    frames: u64,
}

impl ScanState {
    /// A scan of a track whose container states `container`.
    pub const fn new(container: Option<ContainerColour>) -> Self {
        Self {
            container,
            first: None,
            last_pts: 0,
            frames: 0,
        }
    }

    /// Takes the next frame in display order and returns the frames so far.
    ///
    /// The first frame answers the HDR question, so an HDR source stops at once.
    ///
    /// # Errors
    ///
    /// The first [`ScanRefusal`] this frame earns, in this order: HDR, size,
    /// format, colour, picture structure, pts.
    pub fn observe(&mut self, frame: ScanFrame) -> Result<u64, ScanRefusal> {
        let number = self.frames + 1;
        match &self.first {
            None => check_sdr(&merge_container(frame.color, self.container))
                .map_err(ScanRefusal::Hdr)?,
            Some(first) => {
                if frame.display != first.display {
                    return Err(ScanRefusal::SizeChange {
                        frame: number,
                        first: first.display,
                        found: frame.display,
                    });
                }
                if frame.format != first.format {
                    return Err(ScanRefusal::FormatChange {
                        frame: number,
                        first: first.format,
                        found: frame.format,
                    });
                }
                if frame.color != first.color {
                    return Err(ScanRefusal::ColourChange { frame: number });
                }
            }
        }
        if !frame.progressive {
            return Err(ScanRefusal::NotProgressive { frame: number });
        }
        if self.first.is_some() && frame.pts <= self.last_pts {
            return Err(ScanRefusal::PtsNotIncreasing {
                frame: number,
                previous: self.last_pts,
                pts: frame.pts,
            });
        }
        if self.first.is_none() {
            self.first = Some(frame);
        }
        self.last_pts = frame.pts;
        self.frames = number;
        Ok(number)
    }

    /// Ends the scan after `packets` packets.
    ///
    /// The output check counts one frame per packet on both sides. So a scan
    /// refuses a count mismatch here, ahead of a GPU pass the check then fails.
    ///
    /// # Errors
    ///
    /// [`ScanRefusal::NoFrames`] and [`ScanRefusal::CountMismatch`].
    pub fn finish(self, packets: u64) -> Result<ScanFacts, ScanRefusal> {
        let Some(first) = self.first else {
            return Err(ScanRefusal::NoFrames { packets });
        };
        if self.frames != packets {
            return Err(ScanRefusal::CountMismatch {
                frames: self.frames,
                packets,
            });
        }
        Ok(ScanFacts {
            frames: self.frames,
            display: first.display,
            format: first.format,
            color: merge_container(first.color, self.container),
        })
    }
}

/// `stream` with the decoded size in place of the size the container states.
#[must_use]
pub fn decoded_stream(mut stream: StreamFacts, scan: &ScanFacts) -> StreamFacts {
    stream.geometry.coded = scan.display;
    stream
}

/// The checked timing of `stream`, with no start offset.
///
/// # Errors
///
/// [`RemasterError::Timing`] for pts that do not increase, no frames, or a last
/// frame with no duration source.
pub fn track_timing(stream: &StreamFacts) -> Result<TrackTiming, RemasterError> {
    Ok(TrackTiming::new(
        stream.time_base,
        StartOffset::default(),
        stream.frames.clone(),
        stream.hints,
    )?)
}

/// The facts `find_upscaled` compares, from `stream`.
///
/// # Errors
///
/// [`RemasterError::Timing`] and [`RemasterError::Geometry`].
pub fn video_facts(stream: &StreamFacts) -> Result<VideoFacts, RemasterError> {
    Ok(VideoFacts {
        frames: stream.frame_count() as u64,
        duration: track_timing(stream)?.total_duration(),
        dimensions: stream.geometry.display_dimensions(),
        sample_aspect_ratio: stream.geometry.display_sample_aspect_ratio()?,
    })
}

/// The facts of an enhanced video, or the reason it cannot stand in for the
/// source.
///
/// The transform works on coded pixels, and the output keeps the source's
/// rotation. So a candidate must hold the source's rotation, which
/// `check_match` does not compare.
///
/// # Errors
///
/// The text the reuse messages quote.
pub fn candidate_facts(
    stream: &StreamFacts,
    source_rotation: Rotation,
) -> Result<VideoFacts, String> {
    let rotation = stream.geometry.rotation;
    if rotation != source_rotation {
        return Err(format!(
            "Rotation differs: {}/{} degrees clockwise.",
            rotation.degrees(),
            source_rotation.degrees()
        ));
    }
    video_facts(stream).map_err(|error| error.to_string())
}

/// The pts span from the first frame to the last, as exact time.
///
/// # Errors
///
/// [`RemasterError::Time`] when the span does not fit.
pub(crate) fn pts_span(
    stream: &StreamFacts,
    timing: &TrackTiming,
) -> Result<Rational, RemasterError> {
    Ok(Rational::from_ticks(
        timing.last_pts().saturating_sub(timing.first_pts()),
        stream.time_base,
    )?)
}

fn track_list(tracks: &[TrackPacketCount]) -> Vec<Value> {
    tracks
        .iter()
        .map(|track| json!({ "track_id": track.track_id, "packets": track.packets }))
        .collect()
}

fn seconds(value: Rational) -> f64 {
    value.num() as f64 / value.den() as f64
}

/// The report's `input_video` or `output_video` section, the PoC's fields where
/// vidr holds them. `output` adds the PoC's `output_width` and `output_height`.
///
/// # Errors
///
/// [`RemasterError::Timing`], [`RemasterError::Time`], and
/// [`RemasterError::Geometry`].
pub fn video_section(
    stream: &StreamFacts,
    scan: &ScanFacts,
    bytes: u64,
    tracks: &TrackFacts,
    output: Option<Dimensions>,
) -> Result<Value, RemasterError> {
    let timing = track_timing(stream)?;
    let geometry = &stream.geometry;
    let display = geometry.display_dimensions();
    let span = pts_span(stream, &timing)?;
    let rate = frame_rate_hint(stream.frame_count() as u64, span, timing.last_duration());
    let mut section = json!({
        "width": display.width(),
        "height": display.height(),
        "coded_width": geometry.coded.width(),
        "coded_height": geometry.coded.height(),
        "rotation": geometry.rotation.degrees(),
        "frames": stream.frame_count(),
        "decoded_frames": scan.frames,
        "fps": rate.to_string(),
        "sample_aspect_ratio": geometry.display_sample_aspect_ratio()?.to_string(),
        "duration": seconds(timing.total_duration()),
        "last_frame_duration": timing.last_duration().to_string(),
        "last_frame_timestamp": span.to_string(),
        "audio_streams": track_list(&tracks.audio),
        "subtitle_streams": track_list(&tracks.subtitles),
        "chapters": tracks.chapters,
        "bytes": bytes,
        "pixel_format": scan.format.to_string(),
        "matrix_coefficients": scan.color.matrix_coefficients,
        "transfer_characteristics": scan.color.transfer_characteristics,
        "colour_primaries": scan.color.colour_primaries,
        "full_range": scan.color.full_range,
    });
    if let (Some(output), Some(fields)) = (output, section.as_object_mut()) {
        fields.insert("output_width".to_owned(), output.width().into());
        fields.insert("output_height".to_owned(), output.height().into());
    }
    Ok(section)
}

#[cfg(test)]
mod tests {
    use vidr::decode::{MATRIX_BT709, MATRIX_UNSPECIFIED, TRANSFER_UNSPECIFIED};
    use vidr::{DurationHints, FrameTiming, Geometry, SampleAspectRatio};

    use super::*;

    fn dims(width: u32, height: u32) -> Dimensions {
        Dimensions::new(width, height).expect("size")
    }

    fn sdr() -> ColorFacts {
        ColorFacts {
            matrix_coefficients: MATRIX_BT709,
            transfer_characteristics: 1,
            colour_primaries: 1,
            full_range: false,
            chroma_location: None,
        }
    }

    fn frame(pts: i64) -> ScanFrame {
        ScanFrame {
            pts,
            display: dims(640, 360),
            format: PixelFormat::Nv12,
            color: sdr(),
            progressive: true,
        }
    }

    fn refused(path: &str, refusal: ScanRefusal) -> String {
        ScanError {
            path: PathBuf::from(path),
            refusal,
        }
        .to_string()
    }

    #[test]
    fn a_clean_stream_gives_its_facts() {
        let mut state = ScanState::new(None);
        for (index, pts) in [0, 40, 80].into_iter().enumerate() {
            assert_eq!(state.observe(frame(pts)), Ok(index as u64 + 1));
        }
        let facts = state.finish(3).expect("facts");
        assert_eq!(facts.frames, 3);
        assert_eq!(facts.display, dims(640, 360));
        assert_eq!(facts.format, PixelFormat::Nv12);
        assert_eq!(facts.color, sdr());
    }

    #[test]
    fn negative_pts_pass_when_they_increase() {
        let mut state = ScanState::new(None);
        state.observe(frame(-80)).expect("first");
        state.observe(frame(-40)).expect("second");
    }

    #[test]
    fn a_frame_that_is_not_progressive_is_refused_with_the_deinterlace_step() {
        let mut state = ScanState::new(None);
        state.observe(frame(0)).expect("first");
        let mut field = frame(40);
        field.progressive = false;
        let refusal = state.observe(field).expect_err("refused");
        assert_eq!(refusal, ScanRefusal::NotProgressive { frame: 2 });
        let message = refused("clip.mp4", refusal);
        assert!(message.contains("interlaced or MBAFF-coded"), "{message}");
        assert!(
            message.contains("A progressive source coded as MBAFF is refused too"),
            "{message}"
        );
        assert!(message.contains("deinterlace"), "{message}");
    }

    #[test]
    fn the_first_frame_is_checked_for_its_structure_too() {
        let mut state = ScanState::new(None);
        let mut field = frame(0);
        field.progressive = false;
        assert_eq!(
            state.observe(field),
            Err(ScanRefusal::NotProgressive { frame: 1 })
        );
    }

    #[test]
    fn a_size_change_is_refused_with_both_sizes() {
        let mut state = ScanState::new(None);
        state.observe(frame(0)).expect("first");
        let mut changed = frame(40);
        changed.display = dims(1280, 720);
        let refusal = state.observe(changed).expect_err("refused");
        assert_eq!(
            refusal,
            ScanRefusal::SizeChange {
                frame: 2,
                first: dims(640, 360),
                found: dims(1280, 720),
            }
        );
        let message = refused("clip.mp4", refusal);
        assert!(message.contains("Frame dimensions change inside clip.mp4"));
        assert!(message.contains("constant-resolution"), "{message}");
    }

    #[test]
    fn a_format_change_is_refused() {
        let mut state = ScanState::new(None);
        state.observe(frame(0)).expect("first");
        let mut changed = frame(40);
        changed.format = PixelFormat::P010;
        assert!(matches!(
            state.observe(changed),
            Err(ScanRefusal::FormatChange { frame: 2, .. })
        ));
    }

    #[test]
    fn a_colour_change_is_refused() {
        let mut state = ScanState::new(None);
        state.observe(frame(0)).expect("first");
        let mut changed = frame(40);
        changed.color.full_range = true;
        assert_eq!(
            state.observe(changed),
            Err(ScanRefusal::ColourChange { frame: 2 })
        );
    }

    #[test]
    fn pts_must_increase() {
        for pts in [40, 39] {
            let mut state = ScanState::new(None);
            state.observe(frame(0)).expect("first");
            state.observe(frame(40)).expect("second");
            let refusal = state.observe(frame(pts)).expect_err("refused");
            assert_eq!(
                refusal,
                ScanRefusal::PtsNotIncreasing {
                    frame: 3,
                    previous: 40,
                    pts,
                }
            );
            let message = refused("clip.mp4", refusal);
            assert!(message.contains("Repair source timing first"), "{message}");
        }
    }

    #[test]
    fn hdr_in_the_bitstream_stops_at_the_first_frame() {
        let mut hdr = frame(0);
        hdr.color.transfer_characteristics = 16;
        let refusal = ScanState::new(None).observe(hdr).expect_err("refused");
        assert!(matches!(refusal, ScanRefusal::Hdr(_)), "{refusal:?}");
        let message = refused("clip.mp4", refusal);
        assert!(message.contains("HDR video is unsupported at clip.mp4"));
    }

    #[test]
    fn hdr_in_the_container_counts_when_the_bitstream_says_nothing() {
        let mut quiet = frame(0);
        quiet.color.matrix_coefficients = MATRIX_UNSPECIFIED;
        quiet.color.transfer_characteristics = TRANSFER_UNSPECIFIED;
        let tagged = ContainerColour {
            transfer: Some(18),
            ..ContainerColour::default()
        };
        assert!(matches!(
            ScanState::new(Some(tagged)).observe(quiet),
            Err(ScanRefusal::Hdr(_))
        ));
    }

    #[test]
    fn container_tags_fill_the_gaps_in_the_facts() {
        let mut quiet = frame(0);
        quiet.color.matrix_coefficients = MATRIX_UNSPECIFIED;
        let tagged = ContainerColour {
            matrix: Some(MATRIX_BT709),
            ..ContainerColour::default()
        };
        let mut state = ScanState::new(Some(tagged));
        state.observe(quiet).expect("first");
        let facts = state.finish(1).expect("facts");
        assert_eq!(facts.color.matrix_coefficients, MATRIX_BT709);
    }

    #[test]
    fn no_frames_and_a_count_mismatch_are_refused() {
        let refusal = ScanState::new(None).finish(4).expect_err("refused");
        assert_eq!(refusal, ScanRefusal::NoFrames { packets: 4 });
        assert!(refused("clip.mp4", refusal).contains("No decoded video frames in clip.mp4"));
        let mut state = ScanState::new(None);
        state.observe(frame(0)).expect("first");
        assert_eq!(
            state.finish(2),
            Err(ScanRefusal::CountMismatch {
                frames: 1,
                packets: 2,
            })
        );
    }

    fn stream(coded: Dimensions, rotation: Rotation) -> StreamFacts {
        StreamFacts::new(
            Geometry::new(coded, rotation, SampleAspectRatio::new(4, 3)),
            Rational::new(1, 25).expect("time base"),
            (0..3).map(|pts| FrameTiming::new(pts, None, 1)).collect(),
            DurationHints::default(),
        )
    }

    fn scan_of(display: Dimensions) -> ScanFacts {
        ScanFacts {
            frames: 3,
            display,
            format: PixelFormat::Nv12,
            color: sdr(),
        }
    }

    #[test]
    fn the_decoded_size_replaces_the_stated_size() {
        let fixed = decoded_stream(
            stream(dims(1920, 1088), Rotation::Upright),
            &scan_of(dims(1920, 1080)),
        );
        assert_eq!(fixed.geometry.coded, dims(1920, 1080));
        assert_eq!(
            fixed.geometry.sample_aspect_ratio,
            SampleAspectRatio::new(4, 3)
        );
    }

    #[test]
    fn video_facts_describe_the_upright_picture() {
        let facts = video_facts(&stream(dims(640, 360), Rotation::Clockwise90)).expect("facts");
        assert_eq!(facts.frames, 3);
        assert_eq!(facts.dimensions, dims(360, 640));
        assert_eq!(facts.duration, Rational::new(3, 25).expect("duration"));
        assert_eq!(facts.sample_aspect_ratio, SampleAspectRatio::new(3, 4));
    }

    #[test]
    fn a_candidate_with_another_rotation_is_refused() {
        let candidate = stream(dims(640, 360), Rotation::Upright);
        let reason = candidate_facts(&candidate, Rotation::Clockwise90).expect_err("refused");
        assert_eq!(reason, "Rotation differs: 0/90 degrees clockwise.");
        assert!(candidate_facts(&candidate, Rotation::Upright).is_ok());
    }

    #[test]
    fn the_report_section_holds_the_poc_fields() {
        let tracks = TrackFacts {
            audio: vec![TrackPacketCount {
                track_id: 2,
                packets: 50,
            }],
            subtitles: Vec::new(),
            chapters: 1,
        };
        let section = video_section(
            &stream(dims(640, 360), Rotation::Clockwise90),
            &scan_of(dims(640, 360)),
            1234,
            &tracks,
            Some(dims(480, 854)),
        )
        .expect("section");
        assert_eq!(section["width"], 360);
        assert_eq!(section["height"], 640);
        assert_eq!(section["coded_width"], 640);
        assert_eq!(section["rotation"], 90);
        assert_eq!(section["frames"], 3);
        assert_eq!(section["fps"], "25/1");
        assert_eq!(section["sample_aspect_ratio"], "3:4");
        assert_eq!(section["last_frame_duration"], "1/25");
        assert_eq!(section["audio_streams"][0]["packets"], 50);
        assert_eq!(section["chapters"], 1);
        assert_eq!(section["bytes"], 1234);
        assert_eq!(section["pixel_format"], PixelFormat::Nv12.to_string());
        assert_eq!(section["output_width"], 480);
        assert_eq!(section["output_height"], 854);
    }
}
