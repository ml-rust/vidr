//! What to encode and how: the backend-neutral encoder config.

use super::codec::{EncodeCodec, Preset, RateControl, Tuning};
use crate::decode::{
    ColorFacts, MATRIX_UNSPECIFIED, PRIMARIES_UNSPECIFIED, PixelFormat, TRANSFER_UNSPECIFIED,
};
use crate::error::EncodeError;
use crate::geometry::SampleAspectRatio;
use crate::time::Rational;

/// Everything an encoder needs at open.
///
/// [`EncodeConfig::new`] fills every optional field with its default. Set the
/// public fields after it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodeConfig {
    /// The codec.
    pub codec: EncodeCodec,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The sample layout of the frames, which sets the coded bit depth.
    pub format: PixelFormat,
    /// Frames per second, as an exact rational. Numerator and denominator must
    /// fit 32 bits.
    pub frame_rate: Rational,
    /// The unit of the `pts` and `duration` values, in seconds per tick.
    pub time_base: Rational,
    /// Default `P4`.
    pub preset: Preset,
    /// Default [`Tuning::HighQuality`].
    pub tuning: Tuning,
    /// Default `cq` 23.
    pub rate_control: RateControl,
    /// Consecutive B-frames. Default 0, which matches ffmpeg's `-bf 0`.
    pub b_frames: u32,
    /// Frames from one keyframe to the next, above `b_frames`. Default 250,
    /// which matches ffmpeg's NVENC default.
    pub gop_length: u32,
    /// Colour facts the sequence header carries. Default: every code point
    /// unspecified, limited range, no chroma location.
    pub color: ColorFacts,
    /// Sample aspect ratio the sequence header carries. Default square.
    pub sample_aspect_ratio: SampleAspectRatio,
}

impl EncodeConfig {
    /// Creates a config with the default for every other field.
    pub const fn new(
        codec: EncodeCodec,
        width: u32,
        height: u32,
        format: PixelFormat,
        frame_rate: Rational,
        time_base: Rational,
    ) -> Self {
        Self {
            codec,
            width,
            height,
            format,
            frame_rate,
            time_base,
            preset: Preset::P4,
            tuning: Tuning::HighQuality,
            rate_control: RateControl::VbrQuality { cq: 23 },
            b_frames: 0,
            gop_length: 250,
            color: ColorFacts {
                matrix_coefficients: MATRIX_UNSPECIFIED,
                transfer_characteristics: TRANSFER_UNSPECIFIED,
                colour_primaries: PRIMARIES_UNSPECIFIED,
                full_range: false,
                chroma_location: None,
            },
            sample_aspect_ratio: SampleAspectRatio::SQUARE,
        }
    }

    /// Checks everything that needs no GPU.
    ///
    /// # Errors
    ///
    /// - [`EncodeError::OddSize`] for an odd H.264 or HEVC size.
    /// - [`EncodeError::InvalidConfig`] for any other value out of range.
    pub fn validate(&self) -> Result<(), EncodeError> {
        let invalid = |field: &'static str, value: String, reason: &'static str| {
            Err(EncodeError::InvalidConfig {
                field,
                value,
                reason,
            })
        };
        if self.width == 0 || self.height == 0 {
            return invalid(
                "width, height",
                format!("{}x{}", self.width, self.height),
                "both sides must be above zero",
            );
        }
        let crops_in_twos = matches!(self.codec, EncodeCodec::H264 | EncodeCodec::Hevc);
        if crops_in_twos && (self.width % 2 == 1 || self.height % 2 == 1) {
            return Err(EncodeError::OddSize {
                codec: self.codec,
                width: self.width,
                height: self.height,
            });
        }
        self.frame_rate_parts()?;
        if !self.time_base.is_positive() {
            return invalid(
                "time_base",
                self.time_base.to_string(),
                "the time base must be above zero",
            );
        }
        if self.gop_length == 0 || self.gop_length <= self.b_frames {
            return invalid(
                "gop_length",
                format!("{} with b_frames {}", self.gop_length, self.b_frames),
                "the GOP must be above zero and longer than the B-frame run",
            );
        }
        self.validate_rate_control()?;
        if self.color.matrix_coefficients == 0 {
            return invalid(
                "color.matrix_coefficients",
                "0".to_owned(),
                "the identity matrix means RGB, and 4:2:0 cannot carry it",
            );
        }
        if self.codec == EncodeCodec::Av1
            && let Some(location) = self.color.chroma_location
            && location.to_av1().is_none()
        {
            return invalid(
                "color.chroma_location",
                format!("{location:?}"),
                "AV1 signals Left and TopLeft only",
            );
        }
        self.display_aspect_ratio()?;
        Ok(())
    }

    fn validate_rate_control(&self) -> Result<(), EncodeError> {
        let invalid = |value: String, reason: &'static str| {
            Err(EncodeError::InvalidConfig {
                field: "rate_control",
                value,
                reason,
            })
        };
        let is_av1 = self.codec == EncodeCodec::Av1;
        match self.rate_control {
            RateControl::VbrQuality { cq } => {
                let top = if is_av1 { 63 } else { 51 };
                if cq == 0 || cq > top {
                    return invalid(
                        format!("cq {cq}"),
                        "cq must be 1 to 51 for H.264 and HEVC, 1 to 63 for AV1",
                    );
                }
            }
            RateControl::ConstQp {
                intra,
                inter_p,
                inter_b,
            } => {
                if !is_av1 && intra.max(inter_p).max(inter_b) > 51 {
                    return invalid(
                        format!("qp {intra}, {inter_p}, {inter_b}"),
                        "H.264 and HEVC quantisers must be at most 51",
                    );
                }
            }
            RateControl::Bitrate { average, maximum } => {
                if average == 0 || maximum.is_some_and(|maximum| maximum < average) {
                    return invalid(
                        format!("average {average}, maximum {maximum:?}"),
                        "the average must be above zero and the maximum at least the average",
                    );
                }
            }
        }
        Ok(())
    }

    /// The frame rate as 32-bit `(numerator, denominator)`.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidConfig`] when the rate is not above zero, or when
    /// a part does not fit.
    pub fn frame_rate_parts(&self) -> Result<(u32, u32), EncodeError> {
        let parts = u32::try_from(self.frame_rate.num())
            .ok()
            .zip(u32::try_from(self.frame_rate.den()).ok());
        match parts {
            Some((num, den)) if num > 0 => Ok((num, den)),
            _ => Err(EncodeError::InvalidConfig {
                field: "frame_rate",
                value: self.frame_rate.to_string(),
                reason: "the rate must be above zero, with both parts fitting 32 bits",
            }),
        }
    }

    /// The display aspect ratio in lowest terms: the frame size times the
    /// sample aspect ratio.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidConfig`] when a part does not fit 32 bits.
    pub fn display_aspect_ratio(&self) -> Result<(u32, u32), EncodeError> {
        let sar = self.sample_aspect_ratio.ratio();
        let num = u128::from(self.width) * u128::from(sar.num().unsigned_abs());
        let den = u128::from(self.height) * u128::from(sar.den());
        let divisor = greatest_common_divisor(num, den).max(1);
        let parts = u32::try_from(num / divisor)
            .ok()
            .zip(u32::try_from(den / divisor).ok());
        parts.ok_or_else(|| EncodeError::InvalidConfig {
            field: "sample_aspect_ratio",
            value: self.sample_aspect_ratio.to_string(),
            reason: "the display aspect ratio of this frame size does not fit 32 bits",
        })
    }
}

fn greatest_common_divisor(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::ChromaLocation;

    fn rate() -> Rational {
        Rational::new(30000, 1001).unwrap_or(Rational::ONE)
    }

    fn base() -> Rational {
        Rational::new(1, 30000).unwrap_or(Rational::ONE)
    }

    fn config(codec: EncodeCodec) -> EncodeConfig {
        EncodeConfig::new(codec, 640, 360, PixelFormat::Nv12, rate(), base())
    }

    fn field_of(config: &EncodeConfig) -> Option<&'static str> {
        match config.validate() {
            Err(EncodeError::InvalidConfig { field, .. }) => Some(field),
            _ => None,
        }
    }

    #[test]
    fn defaults_pass_and_match_the_poc() {
        let config = config(EncodeCodec::H264);
        assert_eq!(config.validate(), Ok(()));
        assert_eq!(config.b_frames, 0);
        assert_eq!(config.frame_rate_parts(), Ok((30000, 1001)));
    }

    #[test]
    fn odd_sizes_are_refused_for_h264_and_hevc_only() {
        for codec in [EncodeCodec::H264, EncodeCodec::Hevc] {
            let mut odd = config(codec);
            odd.width = 641;
            assert_eq!(
                odd.validate(),
                Err(EncodeError::OddSize {
                    codec,
                    width: 641,
                    height: 360,
                })
            );
        }
        let mut av1 = config(EncodeCodec::Av1);
        av1.height = 361;
        assert_eq!(av1.validate(), Ok(()));
    }

    #[test]
    fn rate_control_ranges_follow_the_codec() {
        let mut h264 = config(EncodeCodec::H264);
        h264.rate_control = RateControl::VbrQuality { cq: 52 };
        assert_eq!(field_of(&h264), Some("rate_control"));
        h264.rate_control = RateControl::VbrQuality { cq: 0 };
        assert_eq!(field_of(&h264), Some("rate_control"));
        let mut av1 = config(EncodeCodec::Av1);
        av1.rate_control = RateControl::VbrQuality { cq: 63 };
        assert_eq!(av1.validate(), Ok(()));
        h264.rate_control = RateControl::Bitrate {
            average: 5_000_000,
            maximum: Some(1_000_000),
        };
        assert_eq!(field_of(&h264), Some("rate_control"));
    }

    #[test]
    fn gop_must_outlast_the_b_frame_run() {
        let mut short = config(EncodeCodec::H264);
        short.gop_length = 2;
        short.b_frames = 2;
        assert_eq!(field_of(&short), Some("gop_length"));
    }

    #[test]
    fn frame_rate_and_time_base_must_be_positive() {
        let mut zero = config(EncodeCodec::H264);
        zero.frame_rate = Rational::ZERO;
        assert_eq!(field_of(&zero), Some("frame_rate"));
        let mut base = config(EncodeCodec::H264);
        base.time_base = Rational::ZERO;
        assert_eq!(field_of(&base), Some("time_base"));
    }

    #[test]
    fn av1_refuses_a_location_it_cannot_signal() {
        let mut av1 = config(EncodeCodec::Av1);
        av1.color.chroma_location = Some(ChromaLocation::Center);
        assert_eq!(field_of(&av1), Some("color.chroma_location"));
    }

    #[test]
    fn display_aspect_ratio_applies_the_sample_ratio() {
        let mut anamorphic = config(EncodeCodec::H264);
        anamorphic.width = 720;
        anamorphic.height = 576;
        anamorphic.sample_aspect_ratio = SampleAspectRatio::new(16, 15);
        assert_eq!(anamorphic.display_aspect_ratio(), Ok((4, 3)));
    }
}
