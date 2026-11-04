//! The encoded video track of an output.

use containr::{ContainerColour, Language, TrackKind};

use crate::decode::{ColorFacts, MATRIX_UNSPECIFIED, PRIMARIES_UNSPECIFIED, TRANSFER_UNSPECIFIED};
use crate::encode::EncodeCodec;
use crate::error::MediaError;
use crate::geometry::{Rotation, SampleAspectRatio};
use crate::time::Rational;

/// What the output states about the encoded video track.
///
/// The encoder's first packet supplies [`VideoOutput::config_record`], so an
/// output starts after that packet, and a single pass writes it.
///
/// # Timing
///
/// MP4 stores every pts, dts, and duration exactly at `1 / time_base.den`
/// ticks per second. Matroska rounds each pts half-even to its
/// `TimestampScale`, 1 ms unless a copied Matroska track needs finer:
///
/// | Value | Tolerance |
/// | --- | --- |
/// | MP4 pts, dts, duration | exact |
/// | MP4 start offset | half a movie tick, in the empty edit |
/// | MKV pts relative to the first frame | half a tick, the value `check_output` compares |
/// | MKV pts on the absolute time line | one tick: half from the offset, half of its own |
/// | MKV total duration | half a tick |
///
/// Two frames that round onto one Matroska tick give
/// [`MediaError::VideoPacket`], which names MP4.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoOutput {
    /// The codec.
    pub codec: EncodeCodec,
    /// The encoder's `avcC`, `hvcC`, or `av1C` body.
    pub config_record: Vec<u8>,
    /// Coded width in pixels.
    pub width: u32,
    /// Coded height in pixels.
    pub height: u32,
    /// The ratio written as `pasp`, or as the Matroska display size.
    pub sample_aspect_ratio: SampleAspectRatio,
    /// The rotation written as the `tkhd` matrix, or the Matroska roll.
    pub rotation: Rotation,
    /// The unit of [`EncodedPacket`](crate::encode::EncodedPacket) timestamps,
    /// in seconds per tick.
    pub time_base: Rational,
    /// [`ColorPlan::tags`](crate::color::ColorPlan::tags). A code of 2 writes
    /// no tag. All three codes at 2 with limited range write no colour box.
    pub colour: ColorFacts,
    /// The language. Each container stores its own written form of it.
    pub language: Language,
    /// Seconds from the output's time zero to the first frame, 0 or more.
    ///
    /// Copied tracks keep their timeline, so pass the first frame's time on the
    /// source's timeline: `TrackTiming::video_offset` with a `container_start`
    /// of 0. MP4 writes it as an empty edit, and Matroska adds it to each pts.
    pub start_offset: Rational,
}

impl VideoOutput {
    /// A video track with square pixels, no rotation, no colour tags, language
    /// `und`, and no start offset. Set the public fields after it.
    pub fn new(
        codec: EncodeCodec,
        config_record: Vec<u8>,
        width: u32,
        height: u32,
        time_base: Rational,
    ) -> Self {
        Self {
            codec,
            config_record,
            width,
            height,
            sample_aspect_ratio: SampleAspectRatio::SQUARE,
            rotation: Rotation::Upright,
            time_base,
            colour: ColorFacts {
                matrix_coefficients: MATRIX_UNSPECIFIED,
                transfer_characteristics: TRANSFER_UNSPECIFIED,
                colour_primaries: PRIMARIES_UNSPECIFIED,
                full_range: false,
                chroma_location: None,
            },
            language: Language::Iso639(*b"und"),
            start_offset: Rational::ZERO,
        }
    }

    /// Refuses what no output holds.
    pub(super) fn check(&self) -> Result<(), MediaError> {
        let refuse = |field: &'static str, value: String, reason: &'static str| {
            Err(MediaError::VideoOutput {
                field,
                value,
                reason,
            })
        };
        if self.width == 0 || self.height == 0 {
            return refuse(
                "width, height",
                format!("{}x{}", self.width, self.height),
                "both sides must be above zero. Pass the encoder's size.",
            );
        }
        if self.config_record.is_empty() {
            return refuse(
                "config_record",
                "empty".to_owned(),
                "a decoder needs the record. Start the output after the encoder's first packet.",
            );
        }
        if !self.time_base.is_positive() {
            return refuse(
                "time_base",
                self.time_base.to_string(),
                "a tick must last above zero. Pass the encoder's time base.",
            );
        }
        if self.start_offset.is_negative() {
            return refuse(
                "start_offset",
                self.start_offset.to_string(),
                "no output holds a frame before time zero. Pass the offset from the first \
                 stream's start.",
            );
        }
        Ok(())
    }

    /// The MP4 track spec.
    #[cfg(feature = "mp4")]
    pub(super) fn mp4_spec(&self, timescale: u32) -> Result<containr::mp4::TrackSpec, MediaError> {
        use containr::PixelAspect;
        use containr::mp4::{SampleEntrySpec, TrackSpec, VisualCodec};

        let codec = match self.codec {
            EncodeCodec::H264 => VisualCodec::Avc,
            EncodeCodec::Hevc => VisualCodec::Hevc,
            EncodeCodec::Av1 => VisualCodec::Av1,
        };
        let size_error = || MediaError::VideoOutput {
            field: "width, height",
            value: format!("{}x{}", self.width, self.height),
            reason: "MP4 holds 16 bits per side. Write MKV output (`.mkv`), which holds it.",
        };
        let width = u16::try_from(self.width).map_err(|_| size_error())?;
        let height = u16::try_from(self.height).map_err(|_| size_error())?;
        let (num, den) = self.aspect_parts();
        let mut spec = TrackSpec::new(
            TrackKind::Video,
            timescale,
            SampleEntrySpec::Visual {
                codec,
                config: self.config_record.clone(),
                width,
                height,
            },
        );
        spec.language = self.language.mdhd_form();
        spec.matrix = self.rotation.display_matrix();
        spec.colour = container_colour(&self.colour);
        // The `tkhd` width states the display width in 16.16 fixed point.
        let display = u128::from(self.width) * u128::from(num) * (1 << 16) / u128::from(den.max(1));
        spec.display_width = u32::try_from(display).map_err(|_| self.aspect_error())?;
        spec.display_height = u32::from(height) << 16;
        if self.sample_aspect_ratio != SampleAspectRatio::SQUARE {
            spec.pixel_aspect = Some(PixelAspect {
                h_spacing: u32::try_from(num).map_err(|_| self.aspect_error())?,
                v_spacing: u32::try_from(den).map_err(|_| self.aspect_error())?,
            });
        }
        Ok(spec)
    }

    /// The Matroska track spec. `uid` keeps a Matroska source's `TrackUID`.
    #[cfg(feature = "mkv")]
    pub(super) fn mkv_spec(
        &self,
        timebase: containr::Timebase,
        uid: Option<u64>,
    ) -> Result<containr::mkv::TrackSpec, MediaError> {
        use containr::matroska::{MatroskaVideo, Projection};
        use containr::mkv::TrackSpec;

        let codec_id = match self.codec {
            EncodeCodec::H264 => "V_MPEG4/ISO/AVC",
            EncodeCodec::Hevc => "V_MPEGH/ISO/HEVC",
            EncodeCodec::Av1 => "V_AV1",
        };
        let pose_roll = match self.rotation {
            Rotation::Upright => None,
            Rotation::Clockwise90 => Some(-90.0),
            Rotation::Clockwise180 => Some(180.0),
            Rotation::Clockwise270 => Some(90.0),
        };
        let projection = pose_roll.map(|pose_roll| Projection {
            projection_type: 0,
            private: None,
            pose_yaw: 0.0,
            pose_pitch: 0.0,
            pose_roll,
        });
        let (display_width, display_height, display_unit) = self.display_size()?;
        let mut spec = TrackSpec::new(TrackKind::Video, codec_id, timebase);
        spec.matroska.uid = uid;
        spec.matroska.codec_private = Some(self.config_record.clone());
        spec.matroska.language = self.language.matroska_code().to_owned();
        spec.matroska.video = Some(MatroskaVideo {
            pixel_width: u64::from(self.width),
            pixel_height: u64::from(self.height),
            display_width,
            display_height,
            display_unit,
            crop: [0; 4],
            interlaced: 0,
            projection,
            colour: container_colour(&self.colour),
            extra: Vec::new(),
        });
        Ok(spec)
    }

    /// The Matroska display size for the sample aspect ratio. A width that
    /// lands on a whole pixel stays in pixels. Any other stores the exact
    /// ratio with `DisplayUnit` 3.
    #[cfg(feature = "mkv")]
    fn display_size(
        &self,
    ) -> Result<(Option<u64>, Option<u64>, containr::matroska::DisplayUnit), MediaError> {
        use containr::matroska::DisplayUnit;

        if self.sample_aspect_ratio == SampleAspectRatio::SQUARE {
            return Ok((None, None, DisplayUnit::Pixels));
        }
        let (num, den) = self.aspect_parts();
        let wide = u128::from(self.width) * u128::from(num);
        let divisor = u128::from(den.max(1));
        let narrow = |value: u128| u64::try_from(value).map_err(|_| self.aspect_error());
        if wide.is_multiple_of(divisor) {
            Ok((
                Some(narrow(wide / divisor)?),
                Some(u64::from(self.height)),
                DisplayUnit::Pixels,
            ))
        } else {
            Ok((
                Some(narrow(wide)?),
                Some(narrow(u128::from(self.height) * u128::from(den))?),
                DisplayUnit::AspectRatio,
            ))
        }
    }

    /// The sample aspect ratio's parts. Its numerator is positive.
    fn aspect_parts(&self) -> (u64, u64) {
        let ratio = self.sample_aspect_ratio.ratio();
        (ratio.num().unsigned_abs(), ratio.den())
    }

    fn aspect_error(&self) -> MediaError {
        MediaError::VideoOutput {
            field: "sample_aspect_ratio",
            value: self.sample_aspect_ratio.to_string(),
            reason: "the container stores it in 32 bits. Use a ratio with smaller parts.",
        }
    }
}

/// The container tags of `facts`. Code 2 is unspecified, so it writes no tag.
/// The output claims nothing the plan does not state.
fn container_colour(facts: &ColorFacts) -> Option<ContainerColour> {
    let tag = |code: u8, unspecified: u8| (code != unspecified).then_some(code);
    let colour = ContainerColour {
        primaries: tag(facts.colour_primaries, PRIMARIES_UNSPECIFIED),
        transfer: tag(facts.transfer_characteristics, TRANSFER_UNSPECIFIED),
        matrix: tag(facts.matrix_coefficients, MATRIX_UNSPECIFIED),
        full_range: None,
    };
    if colour.is_empty() && !facts.full_range {
        return None;
    }
    Some(ContainerColour {
        full_range: Some(facts.full_range),
        ..colour
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(primaries: u8, transfer: u8, matrix: u8, full_range: bool) -> ColorFacts {
        ColorFacts {
            matrix_coefficients: matrix,
            transfer_characteristics: transfer,
            colour_primaries: primaries,
            full_range,
            chroma_location: None,
        }
    }

    #[test]
    fn unspecified_codes_write_no_tag() {
        assert_eq!(container_colour(&facts(2, 2, 2, false)), None);
        assert_eq!(
            container_colour(&facts(2, 2, 2, true)),
            Some(ContainerColour {
                full_range: Some(true),
                ..ContainerColour::default()
            })
        );
        assert_eq!(
            container_colour(&facts(1, 2, 1, false)),
            Some(ContainerColour {
                primaries: Some(1),
                transfer: None,
                matrix: Some(1),
                full_range: Some(false),
            })
        );
    }

    fn output() -> VideoOutput {
        VideoOutput::new(
            EncodeCodec::H264,
            vec![1],
            64,
            48,
            Rational::new(1, 1000).unwrap(),
        )
    }

    #[test]
    fn check_refuses_each_field_by_name() {
        assert_eq!(output().check(), Ok(()));
        let mut empty = output();
        empty.config_record.clear();
        assert!(matches!(
            empty.check(),
            Err(MediaError::VideoOutput {
                field: "config_record",
                ..
            })
        ));
        let mut early = output();
        early.start_offset = Rational::new(-1, 2).unwrap();
        assert!(matches!(
            early.check(),
            Err(MediaError::VideoOutput {
                field: "start_offset",
                ..
            })
        ));
    }

    #[cfg(feature = "mkv")]
    #[test]
    fn display_size_is_whole_pixels_or_the_exact_ratio() {
        use containr::matroska::DisplayUnit;

        let mut video = output();
        assert_eq!(
            video.display_size().unwrap(),
            (None, None, DisplayUnit::Pixels)
        );
        video.sample_aspect_ratio = SampleAspectRatio::new(4, 3);
        video.width = 96;
        video.height = 64;
        assert_eq!(
            video.display_size().unwrap(),
            (Some(128), Some(64), DisplayUnit::Pixels)
        );
        video.sample_aspect_ratio = SampleAspectRatio::new(8, 9);
        video.width = 64;
        video.height = 48;
        assert_eq!(
            video.display_size().unwrap(),
            (Some(512), Some(432), DisplayUnit::AspectRatio)
        );
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn a_side_above_16_bits_names_mkv() {
        let mut video = output();
        video.width = 70_000;
        let error = video.mp4_spec(1000).unwrap_err();
        assert!(error.to_string().contains("MKV"), "{error}");
    }
}
