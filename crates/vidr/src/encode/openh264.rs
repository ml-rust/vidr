//! H.264 Constrained Baseline software encode on OpenH264, from numr tensors
//! through host memory. Each frame takes one copy through host memory, and
//! [`Encoder::route`] declares it.
//!
//! # Licence position
//!
//! OpenH264's code is BSD-2-Clause. H.264 patents are a separate matter:
//!
//! - The `openh264` feature compiles OpenH264 from source. A source build gets no
//!   Cisco patent cover. Whoever ships it answers for the H.264 patent licence.
//! - The `openh264-cisco` feature loads Cisco's binary from a path the caller
//!   supplies, after the `openh264` crate checks its hash. Cisco's cover applies
//!   only to its own binary. Cisco's FAQ requires the binary to download when the
//!   product installs on the user's device. vidr holds no download code.
//!
//! # Rate control
//!
//! | `RateControl` | OpenH264 |
//! | --- | --- |
//! | `VbrQuality { cq }` | Quality mode with QP range `cq..=cq`, so each picture codes at QP `cq`. No quality target adapts it |
//! | `ConstQp { intra, inter_p, .. }` | The same, for `intra == inter_p`. `inter_b` has no use with no B-frames. A pair that differs is refused: the crate has one QP range |
//! | `Bitrate { average, maximum }` | Bitrate mode at `average`. A `maximum` other than `average` is refused: the crate sets the peak to the target |
//!
//! Frame skip stays off, so each frame gives one picture.

use std::collections::VecDeque;
use std::marker::PhantomData;

use ::openh264::encoder::{
    BitRate, ColorPrimaries, Complexity, Encoder as Library, EncoderConfig, FrameRate, FrameType,
    IntraFramePeriod, MatrixCoefficients, Profile, QpRange, RateControlMode,
    TransferCharacteristics, VuiConfig,
};
use ::openh264::formats::YUVSource;
use ::openh264::{OpenH264API, Timestamp};
use numr::dtype::DType;
use numr::runtime::Runtime;

use super::dts::DtsClock;
use super::{
    ChromaLocation, EncodeCodec, EncodeConfig, EncodeFrame, EncodedPacket, Encoder, PixelFormat,
    Preset, RateControl, Tuning,
};
use crate::bitstream::{FromAnnexB, LengthSize, NalCodec};
use crate::decode::{MATRIX_UNSPECIFIED, PRIMARIES_UNSPECIFIED, TRANSFER_UNSPECIFIED};
use crate::error::EncodeError;
use crate::frame::{HostFrame, download_planes, download_route};
use crate::geometry::SampleAspectRatio;
use crate::route::FrameRoute;

const ENCODER: &str = "OpenH264";
/// The wrapper's limit on the longer side of a frame.
const MAX_LONG_SIDE: u32 = 3840;
/// The wrapper's limit on the shorter side of a frame.
const MAX_SHORT_SIDE: u32 = 2160;

/// H.264 Constrained Baseline software encode from numr tensors on any backend.
///
/// # Output
///
/// Each `send` returns one packet, in display order, with `dts` equal to `pts`.
/// See [`EncodedPacket`]. [`Encoder::config_record`] returns the `avcC` payload
/// once the first packet exists, the same as NVENC.
///
/// # Refusals
///
/// `new` returns [`EncodeError::InvalidConfig`], with the field's name, for each
/// of these:
///
/// | Field | Refused |
/// | --- | --- |
/// | `codec` | HEVC, AV1 |
/// | `format` | P010 |
/// | `b_frames` | above 0 |
/// | `width, height` | above 3840x2160 or 2160x3840 |
/// | `tuning` | `Lossless`, `UltraHighQuality` |
/// | `rate_control` | See the module documentation |
/// | `color` | a code point the crate has no name for |
/// | `color.chroma_location` | anything but `Left` or unset: OpenH264 writes no `chroma_loc_info` |
/// | `sample_aspect_ratio` | anything but square: the crate sets no aspect ratio |
///
/// # Errors after the first
///
/// After an encode error, every later `send` and `flush` returns the same error.
/// A refused frame leaves the encoder usable.
pub struct Openh264Encoder<R: Runtime> {
    config: EncodeConfig,
    library: Library,
    filter: FromAnnexB,
    clock: DtsClock,
    /// Reused for each frame, so a frame of the same size allocates nothing here.
    host: HostFrame,
    annexb: Vec<u8>,
    config_record: Option<Vec<u8>>,
    packets: VecDeque<EncodedPacket>,
    frame_rate: (u32, u32),
    frame_index: u64,
    last_pts: Option<i64>,
    flushed: bool,
    error: Option<EncodeError>,
    runtime: PhantomData<fn() -> R>,
}

/// A host frame as the wrapper reads it. `EncodeConfig::validate` makes the size
/// even, so a chroma row is half a luma row.
struct Source<'a>(&'a HostFrame);

impl YUVSource for Source<'_> {
    fn dimensions(&self) -> (usize, usize) {
        (self.0.width(), self.0.height())
    }

    fn strides(&self) -> (usize, usize, usize) {
        let chroma = self.0.width() / 2;
        (self.0.width(), chroma, chroma)
    }

    fn y(&self) -> &[u8] {
        self.0.y()
    }

    fn u(&self) -> &[u8] {
        self.0.u()
    }

    fn v(&self) -> &[u8] {
        self.0.v()
    }
}

impl<R> Openh264Encoder<R>
where
    R: Runtime<DType = DType>,
{
    /// Opens an encoder on OpenH264 compiled from source. See the module
    /// documentation for the licence position.
    ///
    /// # Errors
    ///
    /// [`EncodeError::OddSize`] or [`EncodeError::InvalidConfig`] for a config it
    /// refuses, and [`EncodeError::Software`] when OpenH264 does not open.
    #[cfg(feature = "openh264")]
    pub fn new(config: EncodeConfig) -> Result<Self, EncodeError> {
        Self::open(config, OpenH264API::from_source())
    }

    /// Opens an encoder on Cisco's binary at `library`. The `openh264` crate reads
    /// the file and checks its hash against the list it holds. See the module
    /// documentation for the licence position.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Openh264Library`] for a binary that does not load or a hash
    /// not on the list, and the errors of `new`.
    #[cfg(feature = "openh264-cisco")]
    pub fn with_cisco_binary(
        config: EncodeConfig,
        library: impl AsRef<std::path::Path>,
    ) -> Result<Self, EncodeError> {
        let settings = library_settings(&config)?;
        let path = library.as_ref();
        let api =
            OpenH264API::from_blob_path(path).map_err(|error| EncodeError::Openh264Library {
                path: path.display().to_string(),
                reason: error.to_string(),
            })?;
        Self::with_settings(config, settings, api)
    }

    #[cfg(feature = "openh264")]
    fn open(config: EncodeConfig, api: OpenH264API) -> Result<Self, EncodeError> {
        let settings = library_settings(&config)?;
        Self::with_settings(config, settings, api)
    }

    fn with_settings(
        config: EncodeConfig,
        settings: EncoderConfig,
        api: OpenH264API,
    ) -> Result<Self, EncodeError> {
        let library = Library::with_api_config(api, settings).map_err(|error| {
            software(
                "WelsCreateSVCEncoder",
                format!("{}x{}", config.width, config.height),
                &error,
            )
        })?;
        Ok(Self {
            frame_rate: config.frame_rate_parts()?,
            config,
            library,
            filter: FromAnnexB::new(NalCodec::H264, LengthSize::Four),
            clock: DtsClock::new(0),
            host: HostFrame::new(),
            annexb: Vec::new(),
            config_record: None,
            packets: VecDeque::new(),
            frame_index: 0,
            last_pts: None,
            flushed: false,
            error: None,
            runtime: PhantomData,
        })
    }

    /// The config the encoder opened with.
    pub const fn config(&self) -> &EncodeConfig {
        &self.config
    }

    fn stored_error(&self) -> Result<(), EncodeError> {
        match &self.error {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    fn keep_error(&mut self, result: Result<(), EncodeError>) -> Result<(), EncodeError> {
        if let Err(error) = &result {
            self.error = Some(error.clone());
        }
        result
    }

    /// Refuses a frame whose size or format differs from the config.
    fn check_host_frame(&self) -> Result<(), EncodeError> {
        let (width, height) = (self.host.width(), self.host.height());
        let matches = width == self.config.width as usize
            && height == self.config.height as usize
            && self.host.format() == PixelFormat::Nv12;
        if matches {
            return Ok(());
        }
        Err(EncodeError::SurfaceMismatch {
            plane: "luma",
            expected: format!("[{}, {}] in NV12", self.config.height, self.config.width),
            found: format!("[{height}, {width}] in {}", self.host.format()),
        })
    }

    /// The frame's time in whole milliseconds at the config's rate. Only the
    /// library's rate control reads it. Packets keep the exact pts.
    fn timestamp(&self) -> Timestamp {
        let (num, den) = self.frame_rate;
        let millis = u128::from(self.frame_index) * 1000 * u128::from(den) / u128::from(num);
        Timestamp::from_millis(u64::try_from(millis.min(i64::MAX as u128)).unwrap_or(0))
    }

    fn encode(&mut self, pts: i64, duration: i64) -> Result<(), EncodeError> {
        let stamp = self.timestamp();
        let stream = self
            .library
            .encode_at(&Source(&self.host), stamp)
            .map_err(|error| software("EncodeFrame", format!("the frame at pts {pts}"), &error))?;
        let frame_type = stream.frame_type();
        if matches!(frame_type, FrameType::Invalid | FrameType::Skip) {
            return Err(EncodeError::NoPicture {
                encoder: ENCODER,
                pts,
                frame_type: format!("{frame_type:?}"),
            });
        }
        self.annexb.clear();
        stream.write_vec(&mut self.annexb);
        let data = self.filter.filter(&self.annexb)?;
        if self.config_record.is_none() {
            let record = self.filter.config().map(|config| config.to_bytes());
            self.config_record = record.transpose()?;
        }
        self.clock.push(pts, duration);
        let dts = self.clock.next(pts)?;
        self.frame_index += 1;
        self.packets.push_back(EncodedPacket {
            data,
            pts,
            dts,
            duration,
            keyframe: frame_type == FrameType::IDR,
        });
        Ok(())
    }
}

fn software(call: &'static str, input: String, error: &::openh264::Error) -> EncodeError {
    EncodeError::Software {
        encoder: ENCODER,
        call,
        input,
        reason: error.to_string(),
    }
}

fn refused(field: &'static str, value: String, reason: &'static str) -> EncodeError {
    EncodeError::InvalidConfig {
        field,
        value,
        reason,
    }
}

/// Checks the config, and returns the wrapper's settings for it.
fn library_settings(config: &EncodeConfig) -> Result<EncoderConfig, EncodeError> {
    config.validate()?;
    if config.codec != EncodeCodec::H264 {
        return Err(refused(
            "codec",
            config.codec.to_string(),
            "OpenH264 encodes H.264 only",
        ));
    }
    if config.format != PixelFormat::Nv12 {
        return Err(refused(
            "format",
            config.format.to_string(),
            "OpenH264 encodes 8-bit NV12 only",
        ));
    }
    if config.b_frames > 0 {
        return Err(refused(
            "b_frames",
            config.b_frames.to_string(),
            "OpenH264 writes no B-frames, so `b_frames` must be 0",
        ));
    }
    check_size(config)?;
    check_tuning(config.tuning)?;
    if config.sample_aspect_ratio != SampleAspectRatio::SQUARE {
        return Err(refused(
            "sample_aspect_ratio",
            config.sample_aspect_ratio.to_string(),
            "the openh264 crate sets no aspect ratio, so the output must use square samples",
        ));
    }
    if !matches!(
        config.color.chroma_location,
        None | Some(ChromaLocation::Left)
    ) {
        return Err(refused(
            "color.chroma_location",
            format!("{:?}", config.color.chroma_location),
            "OpenH264 writes no chroma location, so only the H.264 default `Left` holds",
        ));
    }
    let (num, den) = config.frame_rate_parts()?;
    let mut settings = rate_settings(config.rate_control)?
        .profile(Profile::Baseline)
        .complexity(complexity(config.preset))
        .max_frame_rate(FrameRate::from_hz(num as f32 / den as f32))
        .intra_frame_period(IntraFramePeriod::from_num_frames(config.gop_length))
        // Keyframes fall on the config's period, and each frame gives one picture.
        .scene_change_detect(false)
        .skip_frames(false)
        .background_detection(false);
    if let Some(vui) = vui(config)? {
        settings = settings.vui(vui);
    }
    Ok(settings)
}

fn check_size(config: &EncodeConfig) -> Result<(), EncodeError> {
    let long = config.width.max(config.height);
    let short = config.width.min(config.height);
    if long > MAX_LONG_SIDE || short > MAX_SHORT_SIDE {
        return Err(refused(
            "width, height",
            format!("{}x{}", config.width, config.height),
            "the openh264 crate encodes at most 3840x2160 or 2160x3840",
        ));
    }
    Ok(())
}

fn check_tuning(tuning: Tuning) -> Result<(), EncodeError> {
    match tuning {
        Tuning::HighQuality | Tuning::LowLatency | Tuning::UltraLowLatency => Ok(()),
        Tuning::Lossless | Tuning::UltraHighQuality => Err(refused(
            "tuning",
            format!("{tuning:?}"),
            "OpenH264 has no such mode, so pick `HighQuality`, `LowLatency`, or `UltraLowLatency`",
        )),
    }
}

const fn complexity(preset: Preset) -> Complexity {
    match preset {
        Preset::P1 | Preset::P2 => Complexity::Low,
        Preset::P3 | Preset::P4 | Preset::P5 => Complexity::Medium,
        Preset::P6 | Preset::P7 => Complexity::High,
    }
}

/// The rate settings. `EncodeConfig::validate` holds each QP at 51 or below, which
/// `QpRange::new` asserts.
fn rate_settings(rate_control: RateControl) -> Result<EncoderConfig, EncodeError> {
    let base = EncoderConfig::new();
    let fixed = |qp: u8| {
        base.rate_control_mode(RateControlMode::Quality)
            .qp(QpRange::new(qp, qp))
            .adaptive_quantization(false)
    };
    match rate_control {
        RateControl::VbrQuality { cq } => Ok(fixed(cq)),
        RateControl::ConstQp { intra, inter_p, .. } if intra == inter_p => Ok(fixed(intra)),
        RateControl::ConstQp {
            intra,
            inter_p,
            inter_b,
        } => Err(refused(
            "rate_control",
            format!("qp {intra}, {inter_p}, {inter_b}"),
            "the openh264 crate has one QP range, so `intra` and `inter_p` must be equal",
        )),
        RateControl::Bitrate { average, maximum } => {
            if average > i32::MAX as u32 {
                return Err(refused(
                    "rate_control",
                    format!("average {average}"),
                    "OpenH264 takes a bitrate that fits 31 bits",
                ));
            }
            if maximum.is_some_and(|maximum| maximum != average) {
                return Err(refused(
                    "rate_control",
                    format!("average {average}, maximum {maximum:?}"),
                    "the openh264 crate sets the peak to the target, so `maximum` must be `None` \
                     or equal `average`",
                ));
            }
            Ok(base
                .rate_control_mode(RateControlMode::Bitrate)
                .bitrate(BitRate::from_bps(average)))
        }
    }
}

/// The VUI for the config's colour, or `None` when it states nothing.
fn vui(config: &EncodeConfig) -> Result<Option<VuiConfig>, EncodeError> {
    let color = config.color;
    let states_nothing = color.colour_primaries == PRIMARIES_UNSPECIFIED
        && color.transfer_characteristics == TRANSFER_UNSPECIFIED
        && color.matrix_coefficients == MATRIX_UNSPECIFIED
        && !color.full_range;
    if states_nothing {
        return Ok(None);
    }
    let unnamed = |field: &'static str, value: u8| {
        refused(
            field,
            value.to_string(),
            "the openh264 crate has no name for this H.273 code point",
        )
    };
    let primaries = primaries(color.colour_primaries)
        .ok_or_else(|| unnamed("color.colour_primaries", color.colour_primaries))?;
    let transfer = transfer(color.transfer_characteristics).ok_or_else(|| {
        unnamed(
            "color.transfer_characteristics",
            color.transfer_characteristics,
        )
    })?;
    let matrix = matrix(color.matrix_coefficients)
        .ok_or_else(|| unnamed("color.matrix_coefficients", color.matrix_coefficients))?;
    Ok(Some(
        VuiConfig::new()
            .color_primaries(primaries)
            .transfer_characteristics(transfer)
            .matrix_coefficients(matrix)
            .full_range(color.full_range),
    ))
}

const fn primaries(code: u8) -> Option<ColorPrimaries> {
    Some(match code {
        1 => ColorPrimaries::Bt709,
        2 => ColorPrimaries::Unspecified,
        4 => ColorPrimaries::Bt470M,
        5 => ColorPrimaries::Bt470BG,
        6 => ColorPrimaries::Smpte170M,
        7 => ColorPrimaries::Smpte240M,
        8 => ColorPrimaries::Film,
        9 => ColorPrimaries::Bt2020,
        _ => return None,
    })
}

const fn transfer(code: u8) -> Option<TransferCharacteristics> {
    Some(match code {
        1 => TransferCharacteristics::Bt709,
        2 => TransferCharacteristics::Unspecified,
        4 => TransferCharacteristics::Bt470M,
        5 => TransferCharacteristics::Bt470Bg,
        6 => TransferCharacteristics::Smpte170M,
        7 => TransferCharacteristics::Smpte240M,
        8 => TransferCharacteristics::Linear,
        13 => TransferCharacteristics::Srgb,
        14 => TransferCharacteristics::Bt2020_10,
        15 => TransferCharacteristics::Bt2020_12,
        16 => TransferCharacteristics::Smpte2084,
        18 => TransferCharacteristics::Hlg,
        _ => return None,
    })
}

const fn matrix(code: u8) -> Option<MatrixCoefficients> {
    Some(match code {
        1 => MatrixCoefficients::Bt709,
        2 => MatrixCoefficients::Unspecified,
        4 => MatrixCoefficients::Fcc,
        5 => MatrixCoefficients::Bt470Bg,
        6 => MatrixCoefficients::Smpte170M,
        7 => MatrixCoefficients::Smpte240M,
        8 => MatrixCoefficients::Ycgco,
        9 => MatrixCoefficients::Bt2020Ncl,
        10 => MatrixCoefficients::Bt2020Cl,
        _ => return None,
    })
}

impl<R> Encoder<R> for Openh264Encoder<R>
where
    R: Runtime<DType = DType>,
{
    fn send(
        &mut self,
        frame: &EncodeFrame<'_, R>,
        pts: i64,
        duration: i64,
    ) -> Result<(), EncodeError> {
        if self.flushed {
            return Err(EncodeError::AfterFlush { pts });
        }
        self.stored_error()?;
        if duration <= 0 {
            return Err(EncodeError::NonPositiveDuration { pts, duration });
        }
        if let Some(previous) = self.last_pts
            && pts <= previous
        {
            return Err(EncodeError::PtsNotIncreasing { pts, previous });
        }
        download_planes(frame, &mut self.host)?;
        self.check_host_frame()?;
        let result = self.encode(pts, duration);
        if result.is_ok() {
            self.last_pts = Some(pts);
        }
        self.keep_error(result)
    }

    fn flush(&mut self) -> Result<(), EncodeError> {
        // OpenH264 holds no picture: each `send` returns its own.
        self.flushed = true;
        self.stored_error()
    }

    fn receive(&mut self) -> Option<EncodedPacket> {
        self.packets.pop_front()
    }

    fn config_record(&self) -> Option<&[u8]> {
        self.config_record.as_deref()
    }

    fn route(&self) -> FrameRoute {
        download_route::<R>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::Rational;

    fn config() -> EncodeConfig {
        let rate = Rational::new(30000, 1001).unwrap_or(Rational::ONE);
        let base = Rational::new(1, 30000).unwrap_or(Rational::ONE);
        EncodeConfig::new(EncodeCodec::H264, 640, 360, PixelFormat::Nv12, rate, base)
    }

    fn refused_field(config: &EncodeConfig) -> Option<&'static str> {
        match library_settings(config) {
            Err(EncodeError::InvalidConfig { field, .. }) => Some(field),
            _ => None,
        }
    }

    #[test]
    fn the_defaults_pass() {
        assert!(library_settings(&config()).is_ok());
    }

    #[test]
    fn hevc_and_av1_are_refused() {
        for codec in [EncodeCodec::Hevc, EncodeCodec::Av1] {
            let mut other = config();
            other.codec = codec;
            assert_eq!(refused_field(&other), Some("codec"));
        }
    }

    #[test]
    fn p010_and_b_frames_are_refused_by_name() {
        let mut p010 = config();
        p010.format = PixelFormat::P010;
        assert_eq!(refused_field(&p010), Some("format"));
        let mut b_frames = config();
        b_frames.b_frames = 1;
        assert_eq!(refused_field(&b_frames), Some("b_frames"));
    }

    #[test]
    fn sizes_past_the_crate_limit_are_refused() {
        let mut wide = config();
        (wide.width, wide.height) = (4096, 2160);
        assert_eq!(refused_field(&wide), Some("width, height"));
        let mut limit = config();
        (limit.width, limit.height) = (2160, 3840);
        assert!(library_settings(&limit).is_ok());
    }

    #[test]
    fn rate_control_maps_or_refuses() {
        let with = |rate_control| {
            let mut config = config();
            config.rate_control = rate_control;
            config
        };
        let same = RateControl::ConstQp {
            intra: 30,
            inter_p: 30,
            inter_b: 40,
        };
        assert!(library_settings(&with(same)).is_ok());
        let split = RateControl::ConstQp {
            intra: 30,
            inter_p: 31,
            inter_b: 31,
        };
        assert_eq!(refused_field(&with(split)), Some("rate_control"));
        let exact = RateControl::Bitrate {
            average: 1_000_000,
            maximum: Some(1_000_000),
        };
        assert!(library_settings(&with(exact)).is_ok());
        let peak = RateControl::Bitrate {
            average: 1_000_000,
            maximum: Some(2_000_000),
        };
        assert_eq!(refused_field(&with(peak)), Some("rate_control"));
    }

    #[test]
    fn a_fact_the_stream_cannot_carry_is_refused() {
        let mut location = config();
        location.color.chroma_location = Some(ChromaLocation::Center);
        assert_eq!(refused_field(&location), Some("color.chroma_location"));
        let mut aspect = config();
        aspect.sample_aspect_ratio = SampleAspectRatio::new(4, 3);
        assert_eq!(refused_field(&aspect), Some("sample_aspect_ratio"));
        let mut tuning = config();
        tuning.tuning = Tuning::Lossless;
        assert_eq!(refused_field(&tuning), Some("tuning"));
        let mut primaries = config();
        primaries.color.colour_primaries = 12;
        assert_eq!(refused_field(&primaries), Some("color.colour_primaries"));
    }

    #[test]
    fn an_unstated_colour_writes_no_vui() {
        assert!(matches!(vui(&config()), Ok(None)));
        let mut bt709 = config();
        bt709.color.colour_primaries = 1;
        bt709.color.transfer_characteristics = 1;
        bt709.color.matrix_coefficients = 1;
        assert!(matches!(vui(&bt709), Ok(Some(_))));
    }
}
