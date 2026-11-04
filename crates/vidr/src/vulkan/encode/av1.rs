//! AV1 sequence and frame structures for encode: key frames at `gop_length`, and inter frames that
//! read the frame before them through one reference name. No GPU has run this path.
//!
//! The parser's `SequenceHeader` is the source of the session parameters' sequence header.

use std::ptr;

use ash::vk;
use ash::vk::native::{
    StdVideoAV1CDEF, StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_INTER,
    StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_KEY, StdVideoAV1GlobalMotion,
    StdVideoAV1InterpolationFilter_STD_VIDEO_AV1_INTERPOLATION_FILTER_EIGHTTAP,
    StdVideoAV1LoopFilter, StdVideoAV1LoopFilterFlags, StdVideoAV1LoopRestoration,
    StdVideoAV1Quantization, StdVideoAV1TxMode_STD_VIDEO_AV1_TX_MODE_SELECT,
};

use crate::bitstream::av1::seq_header::{
    ColorConfig, OperatingPoint, SELECT_INTEGER_MV, SequenceHeader, bit_depth,
};
use crate::decode::PixelFormat;
use crate::encode::{EncodeCodec, EncodeConfig};
use crate::error::EncodeError;
use crate::vulkan::av1::std_params::StdSequence;
use crate::vulkan::gap::{
    self, AV1_PICTURE_SHOW_FRAME, AV1_PRIMARY_REF_NONE, MAX_AV1_REFERENCES,
    StdVideoEncodeAv1OperatingPointInfo, StdVideoEncodeAv1PictureInfo,
    StdVideoEncodeAv1ReferenceInfo, VideoEncodeAv1DpbSlotInfo, VideoEncodeAv1PictureInfo,
    VideoEncodeAv1SessionParametersCreateInfo,
};
use crate::vulkan::retained::Retained;
use crate::vulkan::zeroed::zeroed;

use super::caps::{CodecCaps, EncodeCaps};
use super::engine::{EncodeEngine, EncodeRequest, SlotInfo};
use super::gop::PicturePlan;

/// `OrderHintBits`.
const ORDER_HINT_BITS: u8 = 7;
/// `seq_level_idx` 31: no level limit.
const LEVEL_MAX_PARAMETERS: u8 = 31;
/// `RefFrame` slots of the AV1 frame header.
const NUM_REF_FRAMES: usize = 8;
/// `Default_Ref_Deltas` of the AV1 specification, by reference name with intra first.
const DEFAULT_REF_DELTAS: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];

/// The fewest bits that hold `value`, at least 1.
const fn bits_for(value: u32) -> u8 {
    let used = (u32::BITS - value.leading_zeros()) as u8;
    if used == 0 { 1 } else { used }
}

/// The sequence header of `config`, in the parser's structure.
pub(super) fn sequence_header(config: &EncodeConfig, superblock_128: bool) -> SequenceHeader {
    let color = &config.color;
    let high = config.format == PixelFormat::P010;
    SequenceHeader {
        operating_points: vec![OperatingPoint {
            seq_level_idx: LEVEL_MAX_PARAMETERS,
            ..OperatingPoint::default()
        }],
        frame_width_bits_minus_1: bits_for(config.width - 1) - 1,
        frame_height_bits_minus_1: bits_for(config.height - 1) - 1,
        max_frame_width_minus_1: (config.width - 1) as u16,
        max_frame_height_minus_1: (config.height - 1) as u16,
        use_128x128_superblock: superblock_128,
        enable_order_hint: true,
        order_hint_bits: ORDER_HINT_BITS,
        // AV1 5.5.1: with `seq_force_screen_content_tools` 0, this takes its select value.
        seq_force_integer_mv: SELECT_INTEGER_MV,
        color: ColorConfig {
            high_bitdepth: high,
            bit_depth: bit_depth(high, false),
            color_description_present_flag: true,
            color_primaries: color.colour_primaries,
            transfer_characteristics: color.transfer_characteristics,
            matrix_coefficients: color.matrix_coefficients,
            color_range: color.full_range,
            subsampling_x: true,
            subsampling_y: true,
            chroma_sample_position: color
                .chroma_location
                .and_then(|location| location.to_av1())
                .unwrap_or(0),
            ..ColorConfig::default()
        },
        ..SequenceHeader::default()
    }
}

/// The reference name a single-reference picture reads: the lowest the driver lists.
pub(super) fn reference_name(mask: u32) -> Option<usize> {
    let name = mask.trailing_zeros() as usize;
    (name < MAX_AV1_REFERENCES).then_some(name)
}

/// The order hints of each `RefFrame` slot. A key frame fills all of them, and an inter frame
/// fills slot 0, the one every reference name reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RefHints([u8; NUM_REF_FRAMES]);

impl RefHints {
    fn after(self, idr: bool, order_hint: u8) -> Self {
        if idr {
            return Self([order_hint; NUM_REF_FRAMES]);
        }
        let mut hints = self.0;
        hints[0] = order_hint;
        Self(hints)
    }
}

/// `OrderHint` of the picture `since_idr` pictures after a key frame.
const fn order_hint(since_idr: u32) -> u8 {
    (since_idr % (1 << ORDER_HINT_BITS)) as u8
}

/// The frame header fields of one picture. Its pointers stay null for the caller to set.
fn picture_info(
    config: &EncodeConfig,
    plan: &PicturePlan,
    name: usize,
    hints: RefHints,
) -> StdVideoEncodeAv1PictureInfo {
    StdVideoEncodeAv1PictureInfo {
        flags: AV1_PICTURE_SHOW_FRAME,
        frame_type: if plan.idr {
            StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_KEY
        } else {
            StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_INTER
        },
        frame_presentation_time: 0,
        current_frame_id: 0,
        order_hint: order_hint(plan.since_idr),
        primary_ref_frame: if plan.idr {
            AV1_PRIMARY_REF_NONE
        } else {
            name as u8
        },
        refresh_frame_flags: if plan.idr { 0xFF } else { 0x01 },
        coded_denom: 0,
        render_width_minus_1: (config.width - 1) as u16,
        render_height_minus_1: (config.height - 1) as u16,
        interpolation_filter:
            StdVideoAV1InterpolationFilter_STD_VIDEO_AV1_INTERPOLATION_FILTER_EIGHTTAP,
        tx_mode: StdVideoAV1TxMode_STD_VIDEO_AV1_TX_MODE_SELECT,
        delta_q_res: 0,
        delta_lf_res: 0,
        ref_order_hint: hints.0,
        // Every reference name reads `RefFrame` slot 0, which holds the frame before this one.
        ref_frame_idx: [0; 7],
        reserved1: [0; 3],
        delta_frame_id_minus_1: [0; 7],
        p_tile_info: ptr::null(),
        p_quantization: ptr::null(),
        p_segmentation: ptr::null(),
        p_loop_filter: ptr::null(),
        p_cdef: ptr::null(),
        p_loop_restoration: ptr::null(),
        p_global_motion: ptr::null(),
        p_extension_header: ptr::null(),
        p_buffer_removal_times: ptr::null(),
    }
}

/// The name slots of a picture: its one reference at `name`, and -1 at every other.
fn name_slots(name: usize, reference: Option<u32>) -> [i32; MAX_AV1_REFERENCES] {
    let mut slots = [-1; MAX_AV1_REFERENCES];
    if let Some(slot) = reference {
        slots[name] = slot as i32;
    }
    slots
}

/// The structure of the picture `since_idr` pictures after a key frame, for its slot.
fn reference_info(since_idr: u32) -> StdVideoEncodeAv1ReferenceInfo {
    StdVideoEncodeAv1ReferenceInfo {
        flags: 0,
        ref_frame_id: 0,
        frame_type: if since_idr == 0 {
            StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_KEY
        } else {
            StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_INTER
        },
        order_hint: order_hint(since_idr),
        reserved1: [0; 3],
        p_extension_header: ptr::null(),
    }
}

/// The frame header's quantisation.
fn quantization(q_index: i32) -> StdVideoAV1Quantization {
    StdVideoAV1Quantization {
        flags: zeroed(),
        base_q_idx: q_index.clamp(0, 255) as u8,
        DeltaQYDc: 0,
        DeltaQUDc: 0,
        DeltaQUAc: 0,
        DeltaQVDc: 0,
        DeltaQVAc: 0,
        qm_y: 0,
        qm_u: 0,
        qm_v: 0,
    }
}

/// The frame header's loop filter: a strength that follows the quantiser, on luma and chroma.
fn loop_filter(q_index: i32) -> StdVideoAV1LoopFilter {
    let level = (q_index.clamp(0, 255) / 8).min(63) as u8;
    let mut flags: StdVideoAV1LoopFilterFlags = zeroed();
    flags.set_loop_filter_delta_enabled(1);
    StdVideoAV1LoopFilter {
        flags,
        loop_filter_level: [level; 4],
        loop_filter_sharpness: 0,
        update_ref_delta: 0,
        loop_filter_ref_deltas: DEFAULT_REF_DELTAS,
        update_mode_delta: 0,
        loop_filter_mode_deltas: [0; 2],
    }
}

/// One stream's sequence header, at stable addresses for the life of the encoder.
pub(super) struct Av1Stream {
    sequence: StdSequence,
    operating_point: Retained<StdVideoEncodeAv1OperatingPointInfo>,
    /// The stated operating points: 1, or 0 on a driver that lists none.
    operating_points: u32,
    name: usize,
    hints: RefHints,
    config: EncodeConfig,
}

impl Av1Stream {
    pub(super) fn new(config: &EncodeConfig, caps: &EncodeCaps) -> Result<Self, EncodeError> {
        let CodecCaps::Av1 {
            reference_name_mask,
            superblock_sizes,
            max_operating_points,
        } = caps.codec
        else {
            return Err(EncodeError::VulkanCore {
                reason: "AV1 structures read capabilities of another codec".to_owned(),
            });
        };
        let refused = |what: &str, reason: String| EncodeError::VulkanRefused {
            codec: EncodeCodec::Av1,
            what: what.to_owned(),
            reason,
        };
        let name = reference_name(reference_name_mask).ok_or_else(|| {
            refused(
                "a single-reference picture",
                format!("the driver lists reference names {reference_name_mask:#x}"),
            )
        })?;
        if config.width > u32::from(u16::MAX) + 1 || config.height > u32::from(u16::MAX) + 1 {
            return Err(refused(
                "this frame size",
                format!("{}x{} is past the AV1 limit", config.width, config.height),
            ));
        }
        let superblock_128 = superblock_sizes & gap::AV1_SUPERBLOCK_64 == 0;
        let header = sequence_header(config, superblock_128);
        Ok(Self {
            sequence: StdSequence::new(&header),
            operating_point: Retained::new(StdVideoEncodeAv1OperatingPointInfo {
                seq_level_idx: LEVEL_MAX_PARAMETERS,
                ..StdVideoEncodeAv1OperatingPointInfo::default()
            }),
            operating_points: max_operating_points.min(1),
            name,
            hints: RefHints::default(),
            config: *config,
        })
    }

    /// Creates the engine's session parameters and returns the driver's sequence header OBU.
    pub(super) fn parameters(&self, engine: &mut EncodeEngine) -> Result<Vec<u8>, EncodeError> {
        let create = VideoEncodeAv1SessionParametersCreateInfo {
            s_type: gap::TYPE_AV1_ENCODE_PARAMETERS,
            p_next: ptr::null(),
            p_std_sequence_header: self.sequence.as_ptr(),
            p_std_decoder_model_info: ptr::null(),
            std_operating_point_count: self.operating_points,
            p_std_operating_points: if self.operating_points > 0 {
                self.operating_point.as_ptr()
            } else {
                ptr::null()
            },
        };
        // SAFETY: `create` lives through the call, and `self` holds what it points to.
        unsafe { engine.create_parameters(ptr::from_ref(&create).cast()) }?;
        // SAFETY: AV1 takes no codec structure in this call.
        unsafe { engine.encoded_parameters(ptr::null()) }
    }

    /// Encodes one picture from the engine's input image.
    pub(super) fn encode(
        &mut self,
        engine: &mut EncodeEngine,
        plan: &PicturePlan,
        pts: i64,
    ) -> Result<Vec<u8>, EncodeError> {
        let q_index = engine.rate().plan().quantiser(plan.idr);
        let quantization = quantization(q_index);
        let filter = loop_filter(q_index);
        let cdef = StdVideoAV1CDEF {
            cdef_damping_minus_3: 0,
            cdef_bits: 0,
            cdef_y_pri_strength: [0; 8],
            cdef_y_sec_strength: [0; 8],
            cdef_uv_pri_strength: [0; 8],
            cdef_uv_sec_strength: [0; 8],
        };
        let restoration = StdVideoAV1LoopRestoration {
            FrameRestorationType: [0; 3],
            LoopRestorationSize: [0; 3],
        };
        let motion = StdVideoAV1GlobalMotion {
            GmType: [0; 8],
            gm_params: [[0; 6]; 8],
        };
        let mut picture = picture_info(&self.config, plan, self.name, self.hints);
        picture.p_quantization = &quantization;
        picture.p_loop_filter = &filter;
        picture.p_cdef = &cdef;
        picture.p_loop_restoration = &restoration;
        picture.p_global_motion = &motion;
        let info = VideoEncodeAv1PictureInfo {
            s_type: gap::TYPE_AV1_ENCODE_PICTURE,
            p_next: ptr::null(),
            prediction_mode: if plan.idr {
                gap::AV1_PREDICTION_INTRA_ONLY
            } else {
                gap::AV1_PREDICTION_SINGLE_REFERENCE
            },
            rate_control_group: if plan.idr {
                gap::AV1_GROUP_INTRA
            } else {
                gap::AV1_GROUP_PREDICTIVE
            },
            constant_q_index: q_index.max(0) as u32,
            p_std_picture_info: &picture,
            reference_name_slot_indices: name_slots(self.name, plan.reference_slot),
            primary_reference_cdf_only: vk::FALSE,
            generate_obu_extension_header: vk::FALSE,
        };
        let setup_std = reference_info(plan.since_idr);
        let setup = VideoEncodeAv1DpbSlotInfo {
            s_type: gap::TYPE_AV1_ENCODE_SLOT,
            p_next: ptr::null(),
            p_std_reference_info: &setup_std,
        };
        let reference_std = reference_info(plan.since_idr.wrapping_sub(1));
        let reference = VideoEncodeAv1DpbSlotInfo {
            s_type: gap::TYPE_AV1_ENCODE_SLOT,
            p_next: ptr::null(),
            p_std_reference_info: &reference_std,
        };
        let request = EncodeRequest {
            picture_info: ptr::from_ref(&info).cast(),
            setup: SlotInfo {
                slot: plan.setup_slot,
                codec_info: ptr::from_ref(&setup).cast(),
            },
            reference: plan.reference_slot.map(|slot| SlotInfo {
                slot,
                codec_info: ptr::from_ref(&reference).cast(),
            }),
            pts,
        };
        // SAFETY: each chain above is whole and lives through the call, and the encoder loads the
        // frame first.
        let bytes = unsafe { engine.encode(&request) }?;
        self.hints = self.hints.after(plan.idr, order_hint(plan.since_idr));
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{ChromaLocation, ColorFacts};
    use crate::time::Rational;
    use crate::vulkan::encode::caps::test_caps::{av1, h264};
    use crate::vulkan::encode::gop::GopState;

    fn config() -> EncodeConfig {
        let rate = Rational::new(30, 1).unwrap_or(Rational::ONE);
        let base = Rational::new(1, 30).unwrap_or(Rational::ONE);
        EncodeConfig::new(EncodeCodec::Av1, 640, 360, PixelFormat::P010, rate, base)
    }

    #[test]
    fn the_sequence_header_states_the_size_depth_and_colour() {
        let mut source = config();
        source.color = ColorFacts {
            matrix_coefficients: 9,
            transfer_characteristics: 16,
            colour_primaries: 9,
            full_range: true,
            chroma_location: Some(ChromaLocation::TopLeft),
        };
        let header = sequence_header(&source, false);
        assert_eq!(header.max_frame_width_minus_1, 639);
        assert_eq!(header.max_frame_height_minus_1, 359);
        assert_eq!(header.frame_width_bits_minus_1, 9);
        assert_eq!(header.frame_height_bits_minus_1, 8);
        assert_eq!(header.order_hint_bits, 7);
        assert_eq!(header.color.bit_depth, 10);
        assert!(header.color.high_bitdepth);
        assert_eq!(header.color.color_primaries, 9);
        assert_eq!(header.color.transfer_characteristics, 16);
        assert_eq!(header.color.matrix_coefficients, 9);
        assert!(header.color.color_range);
        assert_eq!(header.color.chroma_sample_position, 2);
        assert!(!header.film_grain_params_present);
        assert_eq!(header.operating_points.len(), 1);
    }

    #[test]
    fn bits_hold_each_value() {
        assert_eq!(bits_for(0), 1);
        assert_eq!(bits_for(1), 1);
        assert_eq!(bits_for(255), 8);
        assert_eq!(bits_for(256), 9);
    }

    #[test]
    fn the_reference_name_is_the_lowest_the_driver_lists() {
        assert_eq!(reference_name(0b0100_0101), Some(0));
        assert_eq!(reference_name(0b0100_0000), Some(6));
        assert_eq!(reference_name(0), None);
        assert_eq!(reference_name(1 << 7), None);
    }

    #[test]
    fn a_driver_with_no_name_is_refused_by_name() {
        let none = EncodeCaps {
            codec: CodecCaps::Av1 {
                reference_name_mask: 0,
                superblock_sizes: gap::AV1_SUPERBLOCK_64,
                max_operating_points: 1,
            },
            ..av1()
        };
        let text = Av1Stream::new(&config(), &none)
            .err()
            .expect("refused")
            .to_string();
        assert!(text.contains("single-reference"), "{text}");
        assert!(Av1Stream::new(&config(), &h264()).is_err());
        assert!(Av1Stream::new(&config(), &av1()).is_ok());
    }

    #[test]
    fn a_key_frame_refreshes_all_slots_and_an_inter_frame_slot_0() {
        let source = config();
        let mut gop = GopState::new(3);
        let key = gop.next();
        let picture = picture_info(&source, &key, 2, RefHints::default());
        assert_eq!(picture.frame_type, 0);
        assert_eq!(picture.refresh_frame_flags, 0xFF);
        assert_eq!(picture.primary_ref_frame, 7);
        assert_eq!(picture.flags, AV1_PICTURE_SHOW_FRAME);
        assert_eq!(name_slots(2, key.reference_slot), [-1; 7]);

        let hints = RefHints::default().after(true, 0);
        let inter = gop.next();
        let picture = picture_info(&source, &inter, 2, hints);
        assert_eq!(picture.frame_type, 1);
        assert_eq!(picture.refresh_frame_flags, 0x01);
        assert_eq!(picture.primary_ref_frame, 2);
        assert_eq!(picture.order_hint, 1);
        assert_eq!(picture.ref_frame_idx, [0; 7]);
        assert_eq!(picture.render_width_minus_1, 639);
        assert_eq!(
            name_slots(2, inter.reference_slot),
            [-1, -1, 0, -1, -1, -1, -1]
        );
    }

    #[test]
    fn order_hints_track_each_refresh_and_wrap() {
        let hints = RefHints::default().after(true, 5);
        assert_eq!(hints.0, [5; 8]);
        let hints = hints.after(false, 6);
        assert_eq!(hints.0, [6, 5, 5, 5, 5, 5, 5, 5]);
        assert_eq!(order_hint(127), 127);
        assert_eq!(order_hint(128), 0);
    }

    #[test]
    fn the_filters_follow_the_quantiser_and_stay_in_range() {
        assert_eq!(quantization(300).base_q_idx, 255);
        assert_eq!(quantization(100).base_q_idx, 100);
        assert_eq!(loop_filter(255).loop_filter_level, [31; 4]);
        assert_eq!(loop_filter(0).loop_filter_level, [0; 4]);
        assert_eq!(reference_info(0).frame_type, 0);
        assert_eq!(reference_info(1).frame_type, 1);
        assert_eq!(reference_info(1).order_hint, 1);
    }
}
