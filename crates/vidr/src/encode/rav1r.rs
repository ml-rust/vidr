//! AV1 software encode on rav1r, from numr tensors through host memory.
//!
//! Each frame takes one copy through host memory, and [`Encoder::route`] declares
//! it. rav1r allocates its own frame for each input, because its lookahead holds
//! the frame.
//!
//! # Licence position
//!
//! rav1r's code is BSD-2-Clause, under the Alliance for Open Media Patent License
//! 1.0. A binary distribution reproduces that patent licence and the BSD text in
//! its documentation, legal notices, or other written materials.
//!
//! # Config
//!
//! | `EncodeConfig` | rav1r |
//! | --- | --- |
//! | `preset` | Speed: `P1` 10, `P2` 8, `P3` 7, `P4` 5, `P5` 3, `P6` 2, `P7` 0 |
//! | `rate_control` `VbrQuality { cq }` | `quantizer` = `round(cq * 255 / 63)`, with no target bitrate |
//! | `rate_control` `Bitrate { average, .. }` | `bitrate` = `average`, `quantizer` 255, which sets no ceiling |
//! | `b_frames` 0 | `low_latency` true: no reorder |
//! | `b_frames` 3 | `low_latency` false: rav1r's reorder group of 4 frames |
//! | `tuning` `LowLatency`, `UltraLowLatency` | `rdo_lookahead_frames` 1. They need `b_frames` 0 |
//! | `gop_length` | `min_key_frame_interval` and `max_key_frame_interval` |
//! | `frame_rate` | `time_base`, the reciprocal, for rate control |
//! | `color` | `color_description`, `pixel_range`, `chroma_sample_position` |
//! | `sample_aspect_ratio` | `sample_aspect_ratio`, which sets the header's render size |
//!
//! Tiles stay at rav1r's automatic choice, and threads use rayon's global pool.
//!
//! # Keyframes
//!
//! `scene_detection_mode` is `None`. rav1r has no `no_scene_detection` or
//! `enable_keyframe_detection` field, and this one is its switch. So a keyframe
//! falls only on the config's period, and a scene cut changes no frame's type. A
//! keyframe never changes the frame count.
//!
//! In reorder mode rav1r places no keyframe in the last 5 frames of the stream.
//!
//! # Timestamps
//!
//! rav1r hides reordered frames inside temporal units, and a temporal unit shows
//! one frame. So packets come out in display order, and `dts` equals `pts` for
//! every `b_frames`, with no negative `dts`.
//!
//! # Aspect ratio
//!
//! AV1 states a sample aspect ratio only through a whole-pixel render size. A
//! ratio with no whole-pixel render size rounds in the bitstream. An MP4 or
//! Matroska output keeps the exact ratio in its track header.

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::Arc;

use ::rav1r::prelude::{
    ChromaSamplePosition, ChromaSampling, ColorDescription, ColorPrimaries, Config, Context,
    EncoderConfig, EncoderStatus, Frame, FrameType, MatrixCoefficients, Pixel, PixelRange,
    Rational as LibraryRational, SceneDetectionSpeed, TransferCharacteristics,
};
use numr::dtype::DType;
use numr::runtime::Runtime;

use super::dts::DtsClock;
use super::{
    ChromaLocation, EncodeCodec, EncodeConfig, EncodeFrame, EncodedPacket, Encoder, PixelFormat,
    Preset, RateControl, Tuning,
};
use crate::bitstream::av1::Av1FromEncoder;
use crate::decode::{MATRIX_UNSPECIFIED, PRIMARIES_UNSPECIFIED, TRANSFER_UNSPECIFIED};
use crate::error::EncodeError;
use crate::frame::{HostFrame, download_planes, download_route};
use crate::route::FrameRoute;

const ENCODER: &str = "rav1r";
/// rav1r's minimum side outside still pictures.
const MIN_SIDE: u32 = 16;
/// AV1's maximum side.
const MAX_SIDE: u32 = 65535;
/// The frames rav1r reorders between two anchors, with its group of 4.
const REORDER_B_FRAMES: u32 = 3;
/// The quantizer that sets no ceiling in bitrate mode.
const NO_QUANTIZER_CEILING: usize = 255;

/// AV1 software encode from numr tensors on any backend, at 8 or 10 bits.
///
/// # Output
///
/// `receive` can return `None` after a `send`, because rav1r's lookahead holds
/// frames. `flush` encodes the rest. Packets come in display order, one per frame,
/// with `dts` equal to `pts`. See the module documentation.
/// [`Encoder::config_record`] returns the `av1C` payload once the first packet
/// exists.
///
/// # Refusals
///
/// `new` returns [`EncodeError::InvalidConfig`], with the field's name, for each
/// of these:
///
/// | Field | Refused |
/// | --- | --- |
/// | `codec` | H.264, HEVC |
/// | `width, height` | below 16 or above 65535 |
/// | `b_frames` | anything but 0 or 3: rav1r's reorder group has one size |
/// | `tuning` | `Lossless`, `UltraHighQuality`, and a low-latency tuning with `b_frames` 3 |
/// | `rate_control` | `ConstQp`: rav1r scales its one quantizer per frame type. `Bitrate` with a peak other than `average`, or an `average` above 31 bits |
/// | `color.*` | a code point rav1r has no name for |
/// | `rav1r config` | anything else `Config::validate` refuses |
///
/// # Errors after the first
///
/// After an encode error, every later `send` and `flush` returns the same error.
/// A refused frame leaves the encoder usable.
pub struct Rav1rEncoder<R: Runtime> {
    config: EncodeConfig,
    session: Session,
    filter: Av1FromEncoder,
    clock: DtsClock,
    /// Reused for each frame, so a frame of the same size allocates nothing here.
    host: HostFrame,
    /// `(pts, duration)` of each frame with no packet yet, in input order.
    pending: VecDeque<(i64, i64)>,
    /// The input number the next packet must show.
    next_input: u64,
    config_record: Option<Vec<u8>>,
    packets: VecDeque<EncodedPacket>,
    last_pts: Option<i64>,
    flushed: bool,
    error: Option<EncodeError>,
    runtime: PhantomData<fn() -> R>,
}

/// The rav1r context of the config's bit depth.
enum Session {
    Eight(Box<Context<u8>>),
    Ten(Box<Context<u16>>),
}

/// One temporal unit from rav1r.
struct Output {
    data: Vec<u8>,
    input_frameno: u64,
    keyframe: bool,
}

impl Session {
    fn open(library: &Config, format: PixelFormat) -> Result<Self, EncodeError> {
        let invalid = |error: ::rav1r::prelude::InvalidConfig| invalid_config(&error);
        Ok(match format {
            PixelFormat::Nv12 => Self::Eight(Box::new(library.new_context().map_err(invalid)?)),
            PixelFormat::P010 => Self::Ten(Box::new(library.new_context().map_err(invalid)?)),
        })
    }

    /// Copies the host frame to a new rav1r frame, and sends it.
    fn send(&mut self, host: &HostFrame) -> Result<(), EncoderStatus> {
        match self {
            Self::Eight(context) => send_frame(context, host),
            Self::Ten(context) => send_frame(context, host),
        }
    }

    /// Signals the end of the stream. This avoids `Context::flush`, which panics
    /// on an error.
    fn finish(&mut self) -> Result<(), EncoderStatus> {
        match self {
            Self::Eight(context) => context.send_frame(None::<Arc<Frame<u8>>>),
            Self::Ten(context) => context.send_frame(None::<Arc<Frame<u16>>>),
        }
    }

    fn receive(&mut self) -> Result<Output, EncoderStatus> {
        match self {
            Self::Eight(context) => receive_output(context),
            Self::Ten(context) => receive_output(context),
        }
    }
}

fn send_frame<T: Pixel>(context: &mut Context<T>, host: &HostFrame) -> Result<(), EncoderStatus> {
    let planes = host.planes();
    let sample = host.format().bytes_per_sample();
    let mut frame = context.new_frame();
    frame.planes[0].copy_from_raw_u8(planes.y, planes.y_stride, sample);
    frame.planes[1].copy_from_raw_u8(planes.u, planes.u_stride, sample);
    frame.planes[2].copy_from_raw_u8(planes.v, planes.v_stride, sample);
    context.send_frame(frame)
}

fn receive_output<T: Pixel>(context: &mut Context<T>) -> Result<Output, EncoderStatus> {
    let packet = context.receive_packet()?;
    Ok(Output {
        keyframe: packet.frame_type == FrameType::KEY,
        input_frameno: packet.input_frameno,
        data: packet.data,
    })
}

impl<R> Rav1rEncoder<R>
where
    R: Runtime<DType = DType>,
{
    /// Opens an encoder.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidConfig`] for a config it refuses. See the type's
    /// documentation.
    pub fn new(config: EncodeConfig) -> Result<Self, EncodeError> {
        let library = library_config(&config)?;
        Ok(Self {
            session: Session::open(&library, config.format)?,
            config,
            filter: Av1FromEncoder::new(),
            // A temporal unit shows its frames in display order, so no packet
            // decodes after it displays.
            clock: DtsClock::new(0),
            host: HostFrame::new(),
            pending: VecDeque::new(),
            next_input: 0,
            config_record: None,
            packets: VecDeque::new(),
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
            && self.host.format() == self.config.format;
        if matches {
            return Ok(());
        }
        Err(EncodeError::SurfaceMismatch {
            plane: "luma",
            expected: format!(
                "[{}, {}] in {}",
                self.config.height, self.config.width, self.config.format
            ),
            found: format!("[{height}, {width}] in {}", self.host.format()),
        })
    }

    fn encode(&mut self, pts: i64, duration: i64) -> Result<(), EncodeError> {
        self.session.send(&self.host).map_err(|status| {
            software(
                "Context::send_frame",
                format!("the frame at pts {pts}"),
                status,
            )
        })?;
        self.pending.push_back((pts, duration));
        self.clock.push(pts, duration);
        self.drain()
    }

    /// Takes every packet rav1r can give now.
    fn drain(&mut self) -> Result<(), EncodeError> {
        loop {
            match self.session.receive() {
                Ok(output) => self.accept(output)?,
                Err(EncoderStatus::Encoded) => {}
                Err(EncoderStatus::NeedMoreData | EncoderStatus::LimitReached) => return Ok(()),
                Err(status) => {
                    return Err(software(
                        "Context::receive_packet",
                        format!("the packet for input frame {}", self.next_input),
                        status,
                    ));
                }
            }
        }
    }

    /// Filters one temporal unit, and queues its packet with the input's timing.
    fn accept(&mut self, output: Output) -> Result<(), EncodeError> {
        let in_order = output.input_frameno == self.next_input;
        let Some((pts, duration)) = self.pending.pop_front().filter(|_| in_order) else {
            return Err(EncodeError::Software {
                encoder: ENCODER,
                call: "Context::receive_packet",
                input: format!("the packet for input frame {}", output.input_frameno),
                reason: format!(
                    "the next frame in display order is input frame {}",
                    self.next_input
                ),
            });
        };
        let dts = self.clock.next(pts)?;
        let data = self.filter.filter(&output.data)?;
        if self.config_record.is_none() {
            let record = self.filter.config().map(|config| config.to_bytes());
            self.config_record = record.transpose()?;
        }
        self.next_input += 1;
        self.packets.push_back(EncodedPacket {
            data,
            pts,
            dts,
            duration,
            keyframe: output.keyframe,
        });
        Ok(())
    }

    fn finish(&mut self) -> Result<(), EncodeError> {
        self.session.finish().map_err(|status| {
            software(
                "Context::send_frame",
                "the end of the stream".to_owned(),
                status,
            )
        })?;
        self.drain()?;
        match self.pending.front() {
            Some(&(pts, _)) => Err(EncodeError::NoPicture {
                encoder: ENCODER,
                pts,
                frame_type: "none".to_owned(),
            }),
            None => Ok(()),
        }
    }
}

fn software(call: &'static str, input: String, status: EncoderStatus) -> EncodeError {
    EncodeError::Software {
        encoder: ENCODER,
        call,
        input,
        reason: status.to_string(),
    }
}

fn refused(field: &'static str, value: String, reason: &'static str) -> EncodeError {
    EncodeError::InvalidConfig {
        field,
        value,
        reason,
    }
}

fn invalid_config(error: &::rav1r::prelude::InvalidConfig) -> EncodeError {
    refused(
        "rav1r config",
        error.to_string(),
        "rav1r refuses this value",
    )
}

/// Checks the config, and returns rav1r's config for it.
fn library_config(config: &EncodeConfig) -> Result<Config, EncodeError> {
    config.validate()?;
    if config.codec != EncodeCodec::Av1 {
        return Err(refused(
            "codec",
            config.codec.to_string(),
            "rav1r encodes AV1 only",
        ));
    }
    check_size(config)?;
    let low_latency = check_b_frames(config)?;
    let low_lookahead = check_tuning(config.tuning, low_latency)?;
    let (num, den) = config.frame_rate_parts()?;
    let (quantizer, bitrate) = rate(config.rate_control)?;
    let ratio = config.sample_aspect_ratio.ratio();

    let mut settings = EncoderConfig::with_speed_preset(speed(config.preset));
    settings.width = config.width as usize;
    settings.height = config.height as usize;
    settings.time_base = LibraryRational::new(u64::from(den), u64::from(num));
    settings.bit_depth = usize::from(config.format.bit_depth());
    settings.chroma_sampling = ChromaSampling::Cs420;
    settings.chroma_sample_position = chroma_position(config.color.chroma_location);
    settings.pixel_range = if config.color.full_range {
        PixelRange::Full
    } else {
        PixelRange::Limited
    };
    settings.color_description = color_description(config)?;
    settings.sample_aspect_ratio = LibraryRational::new(ratio.num().unsigned_abs(), ratio.den());
    settings.low_latency = low_latency;
    settings.quantizer = quantizer;
    settings.bitrate = bitrate;
    settings.min_key_frame_interval = u64::from(config.gop_length);
    settings.max_key_frame_interval = u64::from(config.gop_length);
    settings.speed_settings.scene_detection_mode = SceneDetectionSpeed::None;
    if low_lookahead {
        settings.speed_settings.rdo_lookahead_frames = 1;
    }

    let library = Config::new().with_encoder_config(settings);
    library.validate().map_err(|error| invalid_config(&error))?;
    Ok(library)
}

fn check_size(config: &EncodeConfig) -> Result<(), EncodeError> {
    let fits = |side: u32| (MIN_SIDE..=MAX_SIDE).contains(&side);
    if fits(config.width) && fits(config.height) {
        return Ok(());
    }
    Err(refused(
        "width, height",
        format!("{}x{}", config.width, config.height),
        "rav1r encodes sides from 16 to 65535",
    ))
}

/// Returns whether the stream has no reorder.
fn check_b_frames(config: &EncodeConfig) -> Result<bool, EncodeError> {
    match config.b_frames {
        0 => Ok(true),
        REORDER_B_FRAMES => Ok(false),
        other => Err(refused(
            "b_frames",
            other.to_string(),
            "rav1r has no reorder or a group of 4 frames, so `b_frames` must be 0 or 3",
        )),
    }
}

/// Returns whether the tuning asks for the shortest lookahead.
fn check_tuning(tuning: Tuning, low_latency: bool) -> Result<bool, EncodeError> {
    match tuning {
        Tuning::HighQuality => Ok(false),
        Tuning::LowLatency | Tuning::UltraLowLatency if low_latency => Ok(true),
        Tuning::LowLatency | Tuning::UltraLowLatency => Err(refused(
            "tuning",
            format!("{tuning:?} with b_frames 3"),
            "reorder holds frames, so a low-latency tuning needs `b_frames` 0",
        )),
        Tuning::Lossless | Tuning::UltraHighQuality => Err(refused(
            "tuning",
            format!("{tuning:?}"),
            "rav1r has no such mode, so pick `HighQuality`, `LowLatency`, or `UltraLowLatency`",
        )),
    }
}

/// `P1`, the fastest, is speed 10, and `P7`, the best quality, is speed 0.
const fn speed(preset: Preset) -> u8 {
    match preset {
        Preset::P1 => 10,
        Preset::P2 => 8,
        Preset::P3 => 7,
        Preset::P4 => 5,
        Preset::P5 => 3,
        Preset::P6 => 2,
        Preset::P7 => 0,
    }
}

/// `round(cq * 255 / 63)` in integers. `EncodeConfig::validate` holds `cq` at 1 to
/// 63, and 63 is odd, so no value falls on a half.
const fn quantizer_of(cq: u8) -> usize {
    (cq as usize * 255 + 31) / 63
}

/// The `quantizer` and `bitrate` fields.
fn rate(rate_control: RateControl) -> Result<(usize, i32), EncodeError> {
    match rate_control {
        RateControl::VbrQuality { cq } => Ok((quantizer_of(cq), 0)),
        RateControl::ConstQp {
            intra,
            inter_p,
            inter_b,
        } => Err(refused(
            "rate_control",
            format!("qp {intra}, {inter_p}, {inter_b}"),
            "rav1r scales its one quantizer per frame type, so use `VbrQuality` or `Bitrate`",
        )),
        RateControl::Bitrate { average, maximum } => {
            let Ok(bitrate) = i32::try_from(average) else {
                return Err(refused(
                    "rate_control",
                    format!("average {average}"),
                    "rav1r takes a bitrate that fits 31 bits",
                ));
            };
            if maximum.is_some_and(|maximum| maximum != average) {
                return Err(refused(
                    "rate_control",
                    format!("average {average}, maximum {maximum:?}"),
                    "rav1r has no peak control, so `maximum` must be `None` or equal `average`",
                ));
            }
            Ok((NO_QUANTIZER_CEILING, bitrate))
        }
    }
}

/// `EncodeConfig::validate` refuses every other location for AV1.
const fn chroma_position(location: Option<ChromaLocation>) -> ChromaSamplePosition {
    match location {
        Some(ChromaLocation::Left) => ChromaSamplePosition::Vertical,
        Some(ChromaLocation::TopLeft) => ChromaSamplePosition::Colocated,
        _ => ChromaSamplePosition::Unknown,
    }
}

/// The colour description, or `None` when every code point is unspecified.
fn color_description(config: &EncodeConfig) -> Result<Option<ColorDescription>, EncodeError> {
    let color = config.color;
    let states_nothing = color.colour_primaries == PRIMARIES_UNSPECIFIED
        && color.transfer_characteristics == TRANSFER_UNSPECIFIED
        && color.matrix_coefficients == MATRIX_UNSPECIFIED;
    if states_nothing {
        return Ok(None);
    }
    let unnamed = |field: &'static str, value: u8| {
        refused(
            field,
            value.to_string(),
            "rav1r has no name for this H.273 code point",
        )
    };
    Ok(Some(ColorDescription {
        color_primaries: primaries(color.colour_primaries)
            .ok_or_else(|| unnamed("color.colour_primaries", color.colour_primaries))?,
        transfer_characteristics: transfer(color.transfer_characteristics).ok_or_else(|| {
            unnamed(
                "color.transfer_characteristics",
                color.transfer_characteristics,
            )
        })?,
        matrix_coefficients: matrix(color.matrix_coefficients)
            .ok_or_else(|| unnamed("color.matrix_coefficients", color.matrix_coefficients))?,
    }))
}

const fn primaries(code: u8) -> Option<ColorPrimaries> {
    Some(match code {
        1 => ColorPrimaries::BT709,
        2 => ColorPrimaries::Unspecified,
        4 => ColorPrimaries::BT470M,
        5 => ColorPrimaries::BT470BG,
        6 => ColorPrimaries::BT601,
        7 => ColorPrimaries::SMPTE240,
        8 => ColorPrimaries::GenericFilm,
        9 => ColorPrimaries::BT2020,
        10 => ColorPrimaries::XYZ,
        11 => ColorPrimaries::SMPTE431,
        12 => ColorPrimaries::SMPTE432,
        22 => ColorPrimaries::EBU3213,
        _ => return None,
    })
}

const fn transfer(code: u8) -> Option<TransferCharacteristics> {
    Some(match code {
        1 => TransferCharacteristics::BT709,
        2 => TransferCharacteristics::Unspecified,
        4 => TransferCharacteristics::BT470M,
        5 => TransferCharacteristics::BT470BG,
        6 => TransferCharacteristics::BT601,
        7 => TransferCharacteristics::SMPTE240,
        8 => TransferCharacteristics::Linear,
        9 => TransferCharacteristics::Log100,
        10 => TransferCharacteristics::Log100Sqrt10,
        11 => TransferCharacteristics::IEC61966,
        12 => TransferCharacteristics::BT1361,
        13 => TransferCharacteristics::SRGB,
        14 => TransferCharacteristics::BT2020_10Bit,
        15 => TransferCharacteristics::BT2020_12Bit,
        16 => TransferCharacteristics::SMPTE2084,
        17 => TransferCharacteristics::SMPTE428,
        18 => TransferCharacteristics::HLG,
        _ => return None,
    })
}

const fn matrix(code: u8) -> Option<MatrixCoefficients> {
    Some(match code {
        1 => MatrixCoefficients::BT709,
        2 => MatrixCoefficients::Unspecified,
        4 => MatrixCoefficients::FCC,
        5 => MatrixCoefficients::BT470BG,
        6 => MatrixCoefficients::BT601,
        7 => MatrixCoefficients::SMPTE240,
        8 => MatrixCoefficients::YCgCo,
        9 => MatrixCoefficients::BT2020NCL,
        10 => MatrixCoefficients::BT2020CL,
        11 => MatrixCoefficients::SMPTE2085,
        12 => MatrixCoefficients::ChromatNCL,
        13 => MatrixCoefficients::ChromatCL,
        14 => MatrixCoefficients::ICtCp,
        _ => return None,
    })
}

impl<R> Encoder<R> for Rav1rEncoder<R>
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
        if self.flushed {
            return self.stored_error();
        }
        self.flushed = true;
        self.stored_error()?;
        let result = self.finish();
        self.keep_error(result)
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
    use numr::runtime::cpu::CpuRuntime;

    use super::*;
    use crate::geometry::SampleAspectRatio;
    use crate::time::Rational;

    fn config() -> EncodeConfig {
        let rate = Rational::new(30000, 1001).unwrap_or(Rational::ONE);
        let base = Rational::new(1, 30000).unwrap_or(Rational::ONE);
        EncodeConfig::new(EncodeCodec::Av1, 64, 48, PixelFormat::Nv12, rate, base)
    }

    fn refused_field(config: &EncodeConfig) -> Option<&'static str> {
        match library_config(config) {
            Err(EncodeError::InvalidConfig { field, .. }) => Some(field),
            _ => None,
        }
    }

    #[test]
    fn the_defaults_pass() {
        assert!(library_config(&config()).is_ok());
    }

    #[test]
    fn cq_maps_to_the_rav1r_quantizer_range() {
        assert_eq!(quantizer_of(1), 4);
        assert_eq!(quantizer_of(20), 81);
        assert_eq!(quantizer_of(23), 93);
        assert_eq!(quantizer_of(62), 251);
        assert_eq!(quantizer_of(63), 255);
    }

    #[test]
    fn presets_cover_the_speed_range_in_order() {
        let speeds = [
            Preset::P1,
            Preset::P2,
            Preset::P3,
            Preset::P4,
            Preset::P5,
            Preset::P6,
            Preset::P7,
        ]
        .map(speed);
        assert_eq!(speeds, [10, 8, 7, 5, 3, 2, 0]);
    }

    #[test]
    fn h264_and_hevc_are_refused() {
        for codec in [EncodeCodec::H264, EncodeCodec::Hevc] {
            let mut other = config();
            other.codec = codec;
            assert_eq!(refused_field(&other), Some("codec"));
        }
    }

    #[test]
    fn sizes_outside_the_rav1r_range_are_refused() {
        let mut small = config();
        small.width = 15;
        assert_eq!(refused_field(&small), Some("width, height"));
        let mut large = config();
        large.height = 65536;
        assert_eq!(refused_field(&large), Some("width, height"));
        let mut odd = config();
        (odd.width, odd.height) = (17, 33);
        assert!(library_config(&odd).is_ok());
    }

    #[test]
    fn b_frames_map_to_no_reorder_or_the_group_of_four() {
        assert_eq!(check_b_frames(&config()), Ok(true));
        let mut reorder = config();
        reorder.b_frames = 3;
        assert_eq!(check_b_frames(&reorder), Ok(false));
        assert!(library_config(&reorder).is_ok());
        for count in [1, 2, 4, 7] {
            let mut other = config();
            other.b_frames = count;
            assert_eq!(refused_field(&other), Some("b_frames"), "{count}");
        }
    }

    #[test]
    fn tunings_map_or_refuse() {
        assert_eq!(check_tuning(Tuning::HighQuality, false), Ok(false));
        assert_eq!(check_tuning(Tuning::LowLatency, true), Ok(true));
        assert_eq!(check_tuning(Tuning::UltraLowLatency, true), Ok(true));
        for tuning in [Tuning::Lossless, Tuning::UltraHighQuality] {
            let mut other = config();
            other.tuning = tuning;
            assert_eq!(refused_field(&other), Some("tuning"));
        }
        let mut reorder = config();
        reorder.b_frames = 3;
        reorder.tuning = Tuning::LowLatency;
        assert_eq!(refused_field(&reorder), Some("tuning"));
    }

    #[test]
    fn rate_control_maps_or_refuses() {
        assert_eq!(rate(RateControl::VbrQuality { cq: 20 }), Ok((81, 0)));
        let exact = RateControl::Bitrate {
            average: 500_000,
            maximum: Some(500_000),
        };
        assert_eq!(rate(exact), Ok((255, 500_000)));
        let with = |rate_control| {
            let mut config = config();
            config.rate_control = rate_control;
            config
        };
        let qp = RateControl::ConstQp {
            intra: 30,
            inter_p: 30,
            inter_b: 30,
        };
        assert_eq!(refused_field(&with(qp)), Some("rate_control"));
        let peak = RateControl::Bitrate {
            average: 500_000,
            maximum: Some(900_000),
        };
        assert_eq!(refused_field(&with(peak)), Some("rate_control"));
        let wide = RateControl::Bitrate {
            average: u32::MAX,
            maximum: None,
        };
        assert_eq!(refused_field(&with(wide)), Some("rate_control"));
    }

    #[test]
    fn every_named_colour_code_maps_to_its_own_number() {
        for code in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22] {
            assert_eq!(primaries(code).map(|name| name as u8), Some(code));
        }
        for code in (1..=18).filter(|code| *code != 3) {
            assert_eq!(transfer(code).map(|name| name as u8), Some(code));
        }
        for code in (1..=14).filter(|code| *code != 3) {
            assert_eq!(matrix(code).map(|name| name as u8), Some(code));
        }
    }

    #[test]
    fn a_reserved_colour_code_is_refused_by_field() {
        let mut primaries = config();
        primaries.color.colour_primaries = 3;
        assert_eq!(refused_field(&primaries), Some("color.colour_primaries"));
        let mut transfer = config();
        transfer.color.transfer_characteristics = 3;
        assert_eq!(
            refused_field(&transfer),
            Some("color.transfer_characteristics")
        );
        let mut matrix = config();
        matrix.color.matrix_coefficients = 3;
        assert_eq!(refused_field(&matrix), Some("color.matrix_coefficients"));
    }

    #[test]
    fn an_unstated_colour_writes_no_description() {
        assert!(matches!(color_description(&config()), Ok(None)));
        let mut bt709 = config();
        bt709.color.colour_primaries = 1;
        assert!(matches!(color_description(&bt709), Ok(Some(_))));
    }

    #[test]
    fn a_render_size_past_the_av1_limit_is_refused() {
        let mut wide = config();
        (wide.width, wide.height) = (60000, 16);
        wide.sample_aspect_ratio = SampleAspectRatio::new(2, 1);
        assert_eq!(refused_field(&wide), Some("rav1r config"));
    }

    #[test]
    fn the_encoder_moves_between_threads() {
        const fn assert_send<T: Send>() {}
        assert_send::<Rav1rEncoder<CpuRuntime>>();
    }
}
