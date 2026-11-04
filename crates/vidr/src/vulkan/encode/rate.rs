//! Rate control from the config and the capabilities, and the chains that state it.
//!
//! A quality target prefers the driver's disabled mode, which gives a constant quantiser. A driver
//! that lists no such mode takes VBR, then CBR, at a bitrate mapped from the frame size, the frame
//! rate, and the target. [`RatePlan::name`] states which one a stream runs.

use std::ffi::c_void;
use std::ptr;

use ash::vk;

use crate::encode::{EncodeCodec, EncodeConfig, RateControl};
use crate::error::EncodeError;
use crate::vulkan::gap::{self, VideoEncodeAv1RateControlInfo, VideoEncodeAv1RateControlLayerInfo};
use crate::vulkan::retained::{Retained, RetainedList};

use super::caps::EncodeCaps;

/// How long a virtual buffer holds, in milliseconds.
const VIRTUAL_BUFFER_MS: u32 = 1000;
/// How full a virtual buffer starts, in milliseconds.
const INITIAL_BUFFER_MS: u32 = 500;
/// Bits per luma sample at the reference target, which a target of 6 more halves.
const BITS_PER_SAMPLE: f64 = 0.25;
/// The target that gives [`BITS_PER_SAMPLE`].
const REFERENCE_TARGET: f64 = 18.0;

/// What a stream's rate control does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RatePlan {
    /// The disabled mode, with one quantiser per picture type.
    Constant { intra: i32, inter: i32 },
    /// Variable bitrate.
    Vbr { average: u64, maximum: u64 },
    /// Constant bitrate.
    Cbr { bitrate: u64 },
}

impl RatePlan {
    /// The Vulkan mode.
    pub(super) const fn mode(self) -> vk::VideoEncodeRateControlModeFlagsKHR {
        match self {
            Self::Constant { .. } => vk::VideoEncodeRateControlModeFlagsKHR::DISABLED,
            Self::Vbr { .. } => vk::VideoEncodeRateControlModeFlagsKHR::VBR,
            Self::Cbr { .. } => vk::VideoEncodeRateControlModeFlagsKHR::CBR,
        }
    }

    /// The name of the mode.
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Constant { .. } => "constant quantiser",
            Self::Vbr { .. } => "VBR",
            Self::Cbr { .. } => "CBR",
        }
    }

    /// The quantiser a picture states: its value in the disabled mode, and 0 in any other mode.
    pub(super) const fn quantiser(self, idr: bool) -> i32 {
        match self {
            Self::Constant { intra, inter } => {
                if idr {
                    intra
                } else {
                    inter
                }
            }
            Self::Vbr { .. } | Self::Cbr { .. } => 0,
        }
    }

    /// The peak bitrate a stream states, or `None` for a constant quantiser.
    pub(super) const fn top_bitrate(self) -> Option<u64> {
        match self.average_and_maximum() {
            Some((_, maximum)) => Some(maximum),
            None => None,
        }
    }

    const fn average_and_maximum(self) -> Option<(u64, u64)> {
        match self {
            Self::Constant { .. } => None,
            Self::Vbr { average, maximum } => Some((average, maximum)),
            Self::Cbr { bitrate } => Some((bitrate, bitrate)),
        }
    }
}

/// A target of 1 to 51, or 1 to 63 for AV1, as the codec's quantiser, inside what the driver lists.
fn target_quantiser(codec: EncodeCodec, target: u8, range: (i32, i32)) -> i32 {
    let value = match codec {
        EncodeCodec::Av1 => (i32::from(target) * 255 + 31) / 63,
        EncodeCodec::H264 | EncodeCodec::Hevc => i32::from(target),
    };
    value.clamp(range.0, range.1.max(range.0))
}

/// A bitrate for a target on a driver with no disabled mode: `BITS_PER_SAMPLE`, halved for each 6
/// of the target above the reference, on the frame's samples per second.
fn mapped_bitrate(config: &EncodeConfig, target: u8, maximum: u64) -> Result<u64, EncodeError> {
    let (rate_num, rate_den) = config.frame_rate_parts()?;
    let per_second = f64::from(config.width) * f64::from(config.height) * f64::from(rate_num)
        / f64::from(rate_den.max(1));
    let bits = BITS_PER_SAMPLE * ((REFERENCE_TARGET - f64::from(target)) / 6.0).exp2();
    Ok(((per_second * bits) as u64).clamp(1, maximum.max(1)))
}

/// VBR at this pair, after each clamp. An equal pair takes CBR where the driver lists it, because
/// NVIDIA's driver refuses VBR with an equal maximum. A driver with no CBR keeps VBR, which
/// `VUID-VkVideoEncodeRateControlInfoKHR-rateControlMode-08278` allows.
const fn vbr_or_cbr(average: u64, maximum: u64, cbr_listed: bool) -> RatePlan {
    if maximum == average && cbr_listed {
        RatePlan::Cbr { bitrate: average }
    } else {
        RatePlan::Vbr { average, maximum }
    }
}

/// The plan of `config` on `caps`.
///
/// # Errors
///
/// [`EncodeError::VulkanRefused`] when the driver lists no mode for the request, or no rate control
/// layer for a bitrate mode, and [`EncodeError::InvalidConfig`] for a bitrate above what it lists.
pub(super) fn plan_rate(config: &EncodeConfig, caps: &EncodeCaps) -> Result<RatePlan, EncodeError> {
    let modes = caps.rate_modes;
    let disabled = modes.contains(vk::VideoEncodeRateControlModeFlagsKHR::DISABLED);
    let vbr = modes.contains(vk::VideoEncodeRateControlModeFlagsKHR::VBR);
    let cbr = modes.contains(vk::VideoEncodeRateControlModeFlagsKHR::CBR);
    let refused = |what: &str| EncodeError::VulkanRefused {
        codec: config.codec,
        what: what.to_owned(),
        reason: format!("the driver lists rate control modes {modes:?}"),
    };
    let plan = match config.rate_control {
        RateControl::VbrQuality { cq } => {
            if disabled {
                let quantiser = target_quantiser(config.codec, cq, caps.quantiser);
                RatePlan::Constant {
                    intra: quantiser,
                    inter: quantiser,
                }
            } else if vbr || cbr {
                let bitrate = mapped_bitrate(config, cq, caps.max_bitrate)?;
                if vbr {
                    let maximum = bitrate.saturating_mul(2).min(caps.max_bitrate.max(1));
                    vbr_or_cbr(bitrate, maximum, cbr)
                } else {
                    RatePlan::Cbr { bitrate }
                }
            } else {
                return Err(refused("a quality target"));
            }
        }
        RateControl::ConstQp { intra, inter_p, .. } => {
            if !disabled {
                return Err(refused("constant quantisers"));
            }
            let range = (caps.quantiser.0, caps.quantiser.1.max(caps.quantiser.0));
            RatePlan::Constant {
                intra: i32::from(intra).clamp(range.0, range.1),
                inter: i32::from(inter_p).clamp(range.0, range.1),
            }
        }
        RateControl::Bitrate { average, maximum } => {
            let average = u64::from(average);
            let top = maximum.map_or(average, u64::from);
            if top > caps.max_bitrate {
                return Err(EncodeError::InvalidConfig {
                    field: "rate_control",
                    value: format!("maximum {top} bits per second"),
                    reason: "the driver lists a lower maximum bitrate",
                });
            }
            if vbr {
                let maximum = maximum
                    .map_or_else(|| average.saturating_mul(2), u64::from)
                    .min(caps.max_bitrate);
                vbr_or_cbr(average, maximum, cbr)
            } else if cbr {
                RatePlan::Cbr { bitrate: average }
            } else {
                return Err(refused("a target bitrate"));
            }
        }
    };
    let rated = !matches!(plan, RatePlan::Constant { .. });
    if rated && caps.max_rate_layers < 1 {
        return Err(EncodeError::VulkanRefused {
            codec: config.codec,
            what: format!("{} rate control", plan.name()),
            reason: "that mode states one rate control layer, and the driver lists \
                     maxRateControlLayers 0"
                .to_owned(),
        });
    }
    if rated && caps.requires_gop_remaining {
        return Err(EncodeError::VulkanRefused {
            codec: config.codec,
            what: format!("{} rate control", plan.name()),
            reason: "the driver requires each frame's remaining GOP, and vidr states none"
                .to_owned(),
        });
    }
    Ok(plan)
}

/// The codec's rate control layer structure, with every limit off. The specification makes it
/// optional, and NVIDIA's driver refuses a rate-controlled encode without it.
enum CodecLayer {
    H264(Retained<vk::VideoEncodeH264RateControlLayerInfoKHR<'static>>),
    Hevc(Retained<vk::VideoEncodeH265RateControlLayerInfoKHR<'static>>),
    Av1(Retained<VideoEncodeAv1RateControlLayerInfo>),
}

impl CodecLayer {
    fn new(codec: EncodeCodec) -> Self {
        match codec {
            EncodeCodec::H264 => Self::H264(Retained::new(Default::default())),
            EncodeCodec::Hevc => Self::Hevc(Retained::new(Default::default())),
            EncodeCodec::Av1 => Self::Av1(Retained::new(VideoEncodeAv1RateControlLayerInfo::new())),
        }
    }

    fn as_ptr(&self) -> *const c_void {
        match self {
            Self::H264(layer) => layer.as_ptr().cast(),
            Self::Hevc(layer) => layer.as_ptr().cast(),
            Self::Av1(layer) => layer.as_ptr().cast(),
        }
    }
}

/// The rate control and quality structures of one session, at stable addresses. Each coding scope
/// after the first states the same rate control, so the begin chain and the control chain share it.
pub(super) struct RateState {
    plan: RatePlan,
    quality_level: u32,
    _layers: RetainedList<vk::VideoEncodeRateControlLayerInfoKHR<'static>>,
    _codec_layer: CodecLayer,
    _h264: Option<Retained<vk::VideoEncodeH264RateControlInfoKHR<'static>>>,
    _hevc: Option<Retained<vk::VideoEncodeH265RateControlInfoKHR<'static>>>,
    _av1: Option<Retained<VideoEncodeAv1RateControlInfo>>,
    info: Retained<vk::VideoEncodeRateControlInfoKHR<'static>>,
    quality: Retained<vk::VideoEncodeQualityLevelInfoKHR<'static>>,
}

impl RateState {
    /// The disabled mode states no layer and no codec structure.
    pub(super) fn new(
        config: &EncodeConfig,
        plan: RatePlan,
        quality_level: u32,
    ) -> Result<Self, EncodeError> {
        let (rate_num, rate_den) = config.frame_rate_parts()?;
        let codec_layer = CodecLayer::new(config.codec);
        let layers: Vec<_> = plan
            .average_and_maximum()
            .map(
                |(average, maximum)| vk::VideoEncodeRateControlLayerInfoKHR {
                    p_next: codec_layer.as_ptr(),
                    average_bitrate: average,
                    max_bitrate: maximum,
                    frame_rate_numerator: rate_num,
                    frame_rate_denominator: rate_den,
                    ..Default::default()
                },
            )
            .into_iter()
            .collect();
        let layers = RetainedList::new(layers);
        let has_layers = !layers.is_empty();
        let gop = config.gop_length;
        let (mut h264, mut hevc, mut av1) = (None, None, None);
        let mut next: *const c_void = ptr::null();
        if has_layers {
            match config.codec {
                EncodeCodec::H264 => {
                    let info = Retained::new(vk::VideoEncodeH264RateControlInfoKHR {
                        flags: vk::VideoEncodeH264RateControlFlagsKHR::REGULAR_GOP
                            | vk::VideoEncodeH264RateControlFlagsKHR::REFERENCE_PATTERN_FLAT,
                        gop_frame_count: gop,
                        idr_period: gop,
                        consecutive_b_frame_count: 0,
                        temporal_layer_count: 1,
                        ..Default::default()
                    });
                    next = info.as_ptr().cast();
                    h264 = Some(info);
                }
                EncodeCodec::Hevc => {
                    let info = Retained::new(vk::VideoEncodeH265RateControlInfoKHR {
                        flags: vk::VideoEncodeH265RateControlFlagsKHR::REGULAR_GOP
                            | vk::VideoEncodeH265RateControlFlagsKHR::REFERENCE_PATTERN_FLAT,
                        gop_frame_count: gop,
                        idr_period: gop,
                        consecutive_b_frame_count: 0,
                        sub_layer_count: 1,
                        ..Default::default()
                    });
                    next = info.as_ptr().cast();
                    hevc = Some(info);
                }
                EncodeCodec::Av1 => {
                    let info = Retained::new(VideoEncodeAv1RateControlInfo {
                        s_type: gap::TYPE_AV1_ENCODE_RATE,
                        p_next: ptr::null(),
                        flags: gap::AV1_RATE_REGULAR_GOP | gap::AV1_RATE_PATTERN_FLAT,
                        gop_frame_count: gop,
                        key_frame_period: gop,
                        consecutive_bipredictive_frame_count: 0,
                        temporal_layer_count: 1,
                    });
                    next = info.as_ptr().cast();
                    av1 = Some(info);
                }
            }
        }
        let info = Retained::new(vk::VideoEncodeRateControlInfoKHR {
            p_next: next,
            rate_control_mode: plan.mode(),
            layer_count: layers.as_slice().len() as u32,
            p_layers: if has_layers {
                layers.as_ptr()
            } else {
                ptr::null()
            },
            virtual_buffer_size_in_ms: if has_layers { VIRTUAL_BUFFER_MS } else { 0 },
            initial_virtual_buffer_size_in_ms: if has_layers { INITIAL_BUFFER_MS } else { 0 },
            ..Default::default()
        });
        let quality = Retained::new(vk::VideoEncodeQualityLevelInfoKHR {
            p_next: info.as_ptr().cast(),
            quality_level,
            ..Default::default()
        });
        Ok(Self {
            plan,
            quality_level,
            _layers: layers,
            _codec_layer: codec_layer,
            _h264: h264,
            _hevc: hevc,
            _av1: av1,
            info,
            quality,
        })
    }

    pub(super) const fn plan(&self) -> RatePlan {
        self.plan
    }

    pub(super) const fn quality_level(&self) -> u32 {
        self.quality_level
    }

    /// The chain of a begin after the first: the rate control the session already holds.
    pub(super) fn begin_chain(&self) -> *const c_void {
        self.info.as_ptr().cast()
    }

    /// The chain of the first control command: the quality level, then the rate control.
    pub(super) fn control_chain(&self) -> *const c_void {
        self.quality.as_ptr().cast()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::PixelFormat;
    use crate::time::Rational;
    use crate::vulkan::encode::caps::test_caps::{av1, h264};

    fn config(codec: EncodeCodec, control: RateControl) -> EncodeConfig {
        let rate = Rational::new(30, 1).unwrap_or(Rational::ONE);
        let base = Rational::new(1, 30).unwrap_or(Rational::ONE);
        let mut config = EncodeConfig::new(codec, 1920, 1080, PixelFormat::Nv12, rate, base);
        config.rate_control = control;
        config
    }

    fn with_modes(modes: vk::VideoEncodeRateControlModeFlagsKHR) -> EncodeCaps {
        EncodeCaps {
            rate_modes: modes,
            ..h264()
        }
    }

    #[test]
    fn a_quality_target_prefers_the_disabled_mode() {
        let plan = plan_rate(
            &config(EncodeCodec::H264, RateControl::VbrQuality { cq: 24 }),
            &h264(),
        );
        assert_eq!(
            plan,
            Ok(RatePlan::Constant {
                intra: 24,
                inter: 24
            })
        );
        let plan = plan.expect("the plan");
        assert_eq!(
            plan.mode(),
            vk::VideoEncodeRateControlModeFlagsKHR::DISABLED
        );
        assert_eq!(plan.quantiser(true), 24);
        assert_eq!(plan.name(), "constant quantiser");
    }

    #[test]
    fn a_quality_target_takes_the_driver_quantiser_range() {
        let narrow = EncodeCaps {
            quantiser: (10, 30),
            ..h264()
        };
        for (cq, expected) in [(1, 10), (20, 20), (51, 30)] {
            let plan = plan_rate(
                &config(EncodeCodec::Hevc, RateControl::VbrQuality { cq }),
                &narrow,
            );
            assert_eq!(
                plan,
                Ok(RatePlan::Constant {
                    intra: expected,
                    inter: expected
                })
            );
        }
    }

    #[test]
    fn an_av1_target_maps_to_the_q_index_range() {
        let plan = |cq| {
            plan_rate(
                &config(EncodeCodec::Av1, RateControl::VbrQuality { cq }),
                &av1(),
            )
            .expect("the plan")
            .quantiser(true)
        };
        assert_eq!(plan(1), 4);
        assert_eq!(plan(32), 130);
        assert_eq!(plan(63), 255);
    }

    #[test]
    fn a_driver_with_no_disabled_mode_takes_a_mapped_rate() {
        let quality = config(EncodeCodec::H264, RateControl::VbrQuality { cq: 18 });
        let vbr = with_modes(
            vk::VideoEncodeRateControlModeFlagsKHR::VBR
                | vk::VideoEncodeRateControlModeFlagsKHR::CBR,
        );
        let expected = (1920.0_f64 * 1080.0 * 30.0 * 0.25) as u64;
        assert_eq!(
            plan_rate(&quality, &vbr),
            Ok(RatePlan::Vbr {
                average: expected,
                maximum: expected * 2,
            })
        );
        let cbr = with_modes(vk::VideoEncodeRateControlModeFlagsKHR::CBR);
        let plan = plan_rate(&quality, &cbr).expect("the plan");
        assert_eq!(plan, RatePlan::Cbr { bitrate: expected });
        assert_eq!(plan.quantiser(true), 0);
        let none = with_modes(vk::VideoEncodeRateControlModeFlagsKHR::empty());
        let text = plan_rate(&quality, &none).expect_err("refused").to_string();
        assert!(text.contains("a quality target"), "{text}");
    }

    #[test]
    fn a_higher_target_maps_to_a_lower_rate_up_to_the_driver_maximum() {
        let at = |cq: u8| {
            mapped_bitrate(
                &config(EncodeCodec::H264, RateControl::VbrQuality { cq }),
                cq,
                u64::MAX,
            )
            .expect("a rate")
        };
        assert_eq!(at(24) * 2, at(18));
        assert!(at(30) < at(24));
        let capped = mapped_bitrate(
            &config(EncodeCodec::H264, RateControl::VbrQuality { cq: 1 }),
            1,
            1000,
        );
        assert_eq!(capped, Ok(1000));
    }

    #[test]
    fn constant_quantisers_need_the_disabled_mode() {
        let request = config(
            EncodeCodec::H264,
            RateControl::ConstQp {
                intra: 20,
                inter_p: 24,
                inter_b: 26,
            },
        );
        assert_eq!(
            plan_rate(&request, &h264()),
            Ok(RatePlan::Constant {
                intra: 20,
                inter: 24
            })
        );
        let text = plan_rate(
            &request,
            &with_modes(vk::VideoEncodeRateControlModeFlagsKHR::VBR),
        )
        .expect_err("refused")
        .to_string();
        assert!(text.contains("constant quantisers"), "{text}");
    }

    #[test]
    fn a_target_bitrate_takes_vbr_then_cbr_and_stays_under_the_driver_maximum() {
        let request = config(
            EncodeCodec::H264,
            RateControl::Bitrate {
                average: 5_000_000,
                maximum: None,
            },
        );
        assert_eq!(
            plan_rate(&request, &h264()),
            Ok(RatePlan::Vbr {
                average: 5_000_000,
                maximum: 10_000_000,
            })
        );
        assert_eq!(
            plan_rate(
                &request,
                &with_modes(vk::VideoEncodeRateControlModeFlagsKHR::CBR)
            ),
            Ok(RatePlan::Cbr { bitrate: 5_000_000 })
        );
        let high = config(
            EncodeCodec::H264,
            RateControl::Bitrate {
                average: 200_000_000,
                maximum: None,
            },
        );
        assert!(matches!(
            plan_rate(&high, &h264()),
            Err(EncodeError::InvalidConfig { .. })
        ));
    }

    /// `VUID-VkVideoEncodeRateControlInfoKHR-rateControlMode-08356` needs an equal pair under CBR.
    #[test]
    fn a_pair_the_clamp_makes_equal_takes_cbr_where_listed_and_else_stays_vbr() {
        let top = h264().max_bitrate;
        let vbr_only = with_modes(vk::VideoEncodeRateControlModeFlagsKHR::VBR);
        let average = |top_u32: u32| {
            config(
                EncodeCodec::H264,
                RateControl::Bitrate {
                    average: top_u32,
                    maximum: None,
                },
            )
        };
        let at_top = average(top as u32);
        assert_eq!(
            plan_rate(&at_top, &h264()),
            Ok(RatePlan::Cbr { bitrate: top })
        );
        assert_eq!(
            plan_rate(&at_top, &vbr_only),
            Ok(RatePlan::Vbr {
                average: top,
                maximum: top,
            })
        );
        // The doubled maximum clamps above the average, so the pair stays VBR.
        let below = average(top as u32 - 1);
        assert_eq!(
            plan_rate(&below, &h264()),
            Ok(RatePlan::Vbr {
                average: top - 1,
                maximum: top,
            })
        );
        let stated = config(
            EncodeCodec::H264,
            RateControl::Bitrate {
                average: 5_000_000,
                maximum: Some(5_000_000),
            },
        );
        assert_eq!(
            plan_rate(&stated, &h264()),
            Ok(RatePlan::Cbr { bitrate: 5_000_000 })
        );
        let no_disabled = with_modes(
            vk::VideoEncodeRateControlModeFlagsKHR::VBR
                | vk::VideoEncodeRateControlModeFlagsKHR::CBR,
        );
        let low_top = EncodeCaps {
            max_bitrate: 1000,
            ..no_disabled
        };
        let quality = config(EncodeCodec::H264, RateControl::VbrQuality { cq: 1 });
        assert_eq!(
            plan_rate(&quality, &low_top),
            Ok(RatePlan::Cbr { bitrate: 1000 })
        );
    }

    #[test]
    fn a_driver_with_no_rate_layer_refuses_a_bitrate_and_keeps_the_disabled_mode() {
        let none = EncodeCaps {
            max_rate_layers: 0,
            ..h264()
        };
        let rated = config(
            EncodeCodec::H264,
            RateControl::Bitrate {
                average: 5_000_000,
                maximum: None,
            },
        );
        let text = plan_rate(&rated, &none).expect_err("refused").to_string();
        assert!(text.contains("maxRateControlLayers"), "{text}");
        let quality = config(EncodeCodec::H264, RateControl::VbrQuality { cq: 24 });
        assert_eq!(
            plan_rate(&quality, &none),
            Ok(RatePlan::Constant {
                intra: 24,
                inter: 24
            })
        );
        let no_disabled = EncodeCaps {
            rate_modes: vk::VideoEncodeRateControlModeFlagsKHR::VBR,
            ..none
        };
        assert!(matches!(
            plan_rate(&quality, &no_disabled),
            Err(EncodeError::VulkanRefused { .. })
        ));
    }

    #[test]
    fn the_top_bitrate_is_the_peak_of_a_plan() {
        let vbr = RatePlan::Vbr {
            average: 1000,
            maximum: 2000,
        };
        assert_eq!(vbr.top_bitrate(), Some(2000));
        assert_eq!(RatePlan::Cbr { bitrate: 7 }.top_bitrate(), Some(7));
        let constant = RatePlan::Constant { intra: 1, inter: 1 };
        assert_eq!(constant.top_bitrate(), None);
    }

    #[test]
    fn a_driver_that_requires_the_remaining_gop_refuses_rate_control() {
        let request = config(
            EncodeCodec::H264,
            RateControl::Bitrate {
                average: 5_000_000,
                maximum: None,
            },
        );
        let caps = EncodeCaps {
            requires_gop_remaining: true,
            ..h264()
        };
        let text = plan_rate(&request, &caps).expect_err("refused").to_string();
        assert!(text.contains("remaining GOP"), "{text}");
    }

    #[test]
    fn the_disabled_mode_states_no_layer() {
        let request = config(EncodeCodec::H264, RateControl::VbrQuality { cq: 24 });
        let plan = RatePlan::Constant {
            intra: 24,
            inter: 24,
        };
        let state = RateState::new(&request, plan, 2).expect("the state");
        // SAFETY: `state` owns the structure, and it still lives.
        let info = unsafe {
            *state
                .begin_chain()
                .cast::<vk::VideoEncodeRateControlInfoKHR>()
        };
        assert_eq!(info.layer_count, 0);
        assert!(info.p_layers.is_null());
        assert!(info.p_next.is_null());
        assert_eq!(info.virtual_buffer_size_in_ms, 0);
        assert_eq!(info.rate_control_mode, plan.mode());
    }

    #[test]
    fn a_rate_mode_states_one_layer_and_the_gop() {
        let request = config(EncodeCodec::Hevc, RateControl::VbrQuality { cq: 24 });
        let plan = RatePlan::Vbr {
            average: 1000,
            maximum: 2000,
        };
        let state = RateState::new(&request, plan, 3).expect("the state");
        assert_eq!(state.quality_level(), 3);
        assert_eq!(state.plan(), plan);
        // SAFETY: `state` owns each structure of the chain, and it still lives.
        let (quality, info, layer, codec) = unsafe {
            let quality = *state
                .control_chain()
                .cast::<vk::VideoEncodeQualityLevelInfoKHR>();
            let info = *state
                .begin_chain()
                .cast::<vk::VideoEncodeRateControlInfoKHR>();
            (
                quality,
                info,
                *info.p_layers,
                *info.p_next.cast::<vk::VideoEncodeH265RateControlInfoKHR>(),
            )
        };
        assert_eq!(quality.quality_level, 3);
        assert_eq!(quality.p_next, state.begin_chain());
        assert_eq!(info.layer_count, 1);
        assert_eq!(info.virtual_buffer_size_in_ms, VIRTUAL_BUFFER_MS);
        assert_eq!(layer.average_bitrate, 1000);
        assert_eq!(layer.max_bitrate, 2000);
        assert_eq!(layer.frame_rate_numerator, 30);
        assert_eq!(codec.gop_frame_count, 250);
        assert_eq!(codec.idr_period, 250);
        assert_eq!(codec.sub_layer_count, 1);
    }
}
