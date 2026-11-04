//! What a run decides from the scan facts: output sizes, the encoder config, and
//! the output's place on the source's timeline.

use vidr::decode::{ColorFacts, PixelFormat};
use vidr::encode::{EncodeCodec, EncodeConfig, Preset, RateControl};
use vidr::{
    Dimensions, EditTimeline, EncodeError, Geometry, Rational, SampleAspectRatio, StartOffset,
    StreamFacts, TrackTiming,
};

use crate::error::RemasterError;
use crate::execution::EncoderBackend;
use crate::rate::frame_rate_hint;
use crate::request::{Codec, RequestError};
use crate::resolution::{ResolutionError, ResolutionMode, choose_resolution, coded_output_size};
use crate::scan::{pts_span, track_timing};

/// The output's two sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputSizes {
    /// The upright picture, as `check_output` compares it.
    pub display: Dimensions,
    /// What the transform writes and the encoder takes.
    pub coded: Dimensions,
}

/// Picks the output's sizes from the source's upright size and `mode`.
///
/// # Errors
///
/// [`ResolutionError::TooLarge`] when a side exceeds 32 bits.
pub fn output_sizes(
    source: &Geometry,
    mode: ResolutionMode,
) -> Result<OutputSizes, ResolutionError> {
    let display = choose_resolution(source.display_dimensions(), mode)?;
    Ok(OutputSizes {
        display,
        coded: coded_output_size(display, source.rotation),
    })
}

/// The encoder's name for `codec`.
pub const fn encode_codec(codec: Codec) -> EncodeCodec {
    match codec {
        Codec::Hevc => EncodeCodec::Hevc,
        Codec::H264 => EncodeCodec::H264,
        Codec::Av1 => EncodeCodec::Av1,
    }
}

/// The request's and the scan's inputs to [`encode_config`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodePlan {
    /// The codec.
    pub codec: Codec,
    /// The encoder, which sets the effort.
    pub encoder: EncoderBackend,
    /// Encoder quality, 1 to 51.
    pub quality: u32,
    /// The size the encoder takes.
    pub coded: Dimensions,
    /// The sample layout of the decoded frames. It sets the coded bit depth.
    pub format: PixelFormat,
    /// The hint from [`stream_rate`].
    pub frame_rate: Rational,
    /// The time base of the packets the pump reads.
    pub time_base: Rational,
    /// The source's sample aspect ratio, of the coded frame.
    pub sample_aspect_ratio: SampleAspectRatio,
    /// The colour facts of `ColorPlan::tags`.
    pub tags: ColorFacts,
}

/// The effort of `encoder`. NVENC, Vulkan Video, and OpenH264 take `P5`. `P3` gives rav1r speed 7,
/// because its slower speeds take far longer per frame than a whole video allows.
pub const fn encoder_preset(encoder: EncoderBackend) -> Preset {
    match encoder {
        EncoderBackend::Rav1r => Preset::P3,
        EncoderBackend::Nvenc | EncoderBackend::Vulkan | EncoderBackend::Openh264(_) => Preset::P5,
    }
}

/// The encoder config of a run: [`encoder_preset`], `VbrQuality` at the request's
/// quality, and no B-frames.
///
/// # Errors
///
/// [`RemasterError::Request`] for a quality above 8 bits, and
/// [`RemasterError::Encode`] for a config no GPU takes.
pub fn encode_config(plan: &EncodePlan) -> Result<EncodeConfig, RemasterError> {
    let cq = u8::try_from(plan.quality).map_err(|_| RequestError::QualityOutOfRange {
        quality: plan.quality,
    })?;
    let mut config = EncodeConfig::new(
        encode_codec(plan.codec),
        plan.coded.width(),
        plan.coded.height(),
        plan.format,
        plan.frame_rate,
        plan.time_base,
    );
    config.preset = encoder_preset(plan.encoder);
    config.rate_control = RateControl::VbrQuality { cq };
    config.b_frames = 0;
    config.color = plan.tags;
    config.sample_aspect_ratio = plan.sample_aspect_ratio;
    config.validate()?;
    Ok(config)
}

/// Names the fix when the encoder refuses to open: the hardware encoder's own message asks
/// for "a codec the GPU encodes at this depth", and the command line has one. OpenH264
/// is the only encoder that refuses the `format` field.
pub fn encoder_open_error(error: EncodeError) -> RemasterError {
    match error {
        EncodeError::BitDepthUnsupported {
            codec: EncodeCodec::H264,
            format: PixelFormat::P010,
        } => RemasterError::H264TenBit,
        EncodeError::InvalidConfig {
            field: "format", ..
        } => RemasterError::Openh264TenBit,
        other => RemasterError::Encode(other),
    }
}

/// The encoder's frame-rate hint for `stream`. See [`frame_rate_hint`].
///
/// # Errors
///
/// [`RemasterError::Timing`] and [`RemasterError::Time`].
pub fn stream_rate(stream: &StreamFacts) -> Result<Rational, RemasterError> {
    let timing = track_timing(stream)?;
    let span = pts_span(stream, &timing)?;
    Ok(frame_rate_hint(
        stream.frame_count() as u64,
        span,
        timing.last_duration(),
    ))
}

/// When the first frame shows on the source's timeline, for
/// `VideoOutput::start_offset`. Copied tracks keep that timeline.
///
/// # Errors
///
/// [`RemasterError::Timing`] and [`RemasterError::Time`].
pub fn start_offset(
    stream: &StreamFacts,
    timeline: Option<EditTimeline>,
) -> Result<Rational, RemasterError> {
    let start = StartOffset {
        edit: timeline.map_or(Rational::ZERO, |timeline| timeline.start_offset()),
        container_start: Rational::ZERO,
    };
    let timing = TrackTiming::new(stream.time_base, start, stream.frames.clone(), stream.hints)?;
    Ok(timing.video_offset()?)
}

#[cfg(test)]
mod tests {
    use vidr::{DurationHints, EditEntry, FrameTiming, Rotation};

    use super::*;

    fn dims(width: u32, height: u32) -> Dimensions {
        Dimensions::new(width, height).expect("size")
    }

    fn base() -> Rational {
        Rational::new(1, 1000).expect("time base")
    }

    fn tags() -> ColorFacts {
        ColorFacts {
            matrix_coefficients: 1,
            transfer_characteristics: 1,
            colour_primaries: 1,
            full_range: false,
            chroma_location: None,
        }
    }

    fn plan() -> EncodePlan {
        EncodePlan {
            codec: Codec::Hevc,
            encoder: EncoderBackend::Nvenc,
            quality: 24,
            coded: dims(1280, 720),
            format: PixelFormat::P010,
            frame_rate: Rational::new(30000, 1001).expect("rate"),
            time_base: base(),
            sample_aspect_ratio: SampleAspectRatio::new(4, 3),
            tags: tags(),
        }
    }

    fn stream(pts: &[i64]) -> StreamFacts {
        let square = SampleAspectRatio::SQUARE;
        StreamFacts::new(
            Geometry::new(dims(640, 360), Rotation::Upright, square),
            base(),
            pts.iter()
                .map(|&pts| FrameTiming::new(pts, None, 40))
                .collect(),
            DurationHints::default(),
        )
    }

    #[test]
    fn a_rotated_source_swaps_the_coded_size() {
        let source = Geometry::new(
            dims(1280, 720),
            Rotation::Clockwise90,
            SampleAspectRatio::SQUARE,
        );
        let sizes = output_sizes(&source, ResolutionMode::Auto).expect("sizes");
        assert_eq!(sizes.display, dims(720, 1280));
        assert_eq!(sizes.coded, dims(1280, 720));
    }

    #[test]
    fn the_config_follows_the_request() {
        let config = encode_config(&plan()).expect("config");
        assert_eq!(config.codec, EncodeCodec::Hevc);
        assert_eq!((config.width, config.height), (1280, 720));
        assert_eq!(config.format, PixelFormat::P010);
        assert_eq!(config.preset, Preset::P5);
        assert_eq!(config.rate_control, RateControl::VbrQuality { cq: 24 });
        assert_eq!(config.b_frames, 0);
        assert_eq!(config.color, tags());
        assert_eq!(config.sample_aspect_ratio, SampleAspectRatio::new(4, 3));
        assert_eq!(config.time_base, base());
    }

    #[test]
    fn rav1r_takes_a_faster_preset_and_av1_takes_its_own_codec() {
        let mut av1 = plan();
        av1.codec = Codec::Av1;
        av1.encoder = EncoderBackend::Rav1r;
        let config = encode_config(&av1).expect("config");
        assert_eq!(config.codec, EncodeCodec::Av1);
        assert_eq!(config.preset, Preset::P3);
    }

    #[test]
    fn a_frame_rate_outside_32_bits_is_refused_by_the_config_check() {
        let mut wide = plan();
        wide.frame_rate = Rational::new(1 << 40, 7).expect("rate");
        assert!(matches!(
            encode_config(&wide),
            Err(RemasterError::Encode(_))
        ));
    }

    #[test]
    fn nvenc_ten_bit_h264_names_hevc() {
        let error = encoder_open_error(EncodeError::BitDepthUnsupported {
            codec: EncodeCodec::H264,
            format: PixelFormat::P010,
        });
        assert!(matches!(error, RemasterError::H264TenBit));
        assert!(error.to_string().contains("--codec hevc"), "{error}");
    }

    #[test]
    fn openh264_ten_bit_names_av1() {
        let error = encoder_open_error(EncodeError::InvalidConfig {
            field: "format",
            value: "P010".to_owned(),
            reason: "8 bits only",
        });
        assert!(matches!(error, RemasterError::Openh264TenBit));
        assert!(error.to_string().contains("--codec av1"), "{error}");
    }

    #[test]
    fn other_encoder_errors_pass_through() {
        let error = encoder_open_error(EncodeError::BitDepthUnsupported {
            codec: EncodeCodec::Hevc,
            format: PixelFormat::P010,
        });
        assert!(matches!(error, RemasterError::Encode(_)));
    }

    #[test]
    fn the_rate_comes_from_the_span() {
        let rate = stream_rate(&stream(&[0, 40, 80, 120])).expect("rate");
        assert_eq!(rate, Rational::new(25, 1).expect("rate"));
    }

    #[test]
    fn one_frame_takes_its_rate_from_its_duration() {
        let rate = stream_rate(&stream(&[0])).expect("rate");
        assert_eq!(rate, Rational::new(25, 1).expect("rate"));
    }

    #[test]
    fn the_start_offset_is_the_first_pts_with_no_edit_list() {
        let offset = start_offset(&stream(&[80, 120]), None).expect("offset");
        assert_eq!(offset, Rational::new(2, 25).expect("offset"));
    }

    #[test]
    fn the_start_offset_adds_an_empty_edit() {
        let empty = EditEntry::empty(500);
        let media = EditEntry::media(1000, 0);
        let timeline = EditTimeline::resolve(&[empty, media], base(), base()).expect("timeline");
        let offset = start_offset(&stream(&[0, 40]), Some(timeline)).expect("offset");
        assert_eq!(offset, Rational::new(1, 2).expect("offset"));
    }
}
