//! SPS and PPS parsing through `h264-reader`, the refusals of this decoder, and what
//! the decoder needs from a sequence.

use h264_reader::Context;
use h264_reader::nal::pps::{PicParameterSet, SliceGroup};
use h264_reader::nal::sps::{FrameMbsFlags, PicOrderCntType, SeqParameterSet};
use h264_reader::rbsp::BitReader;

use super::poc::PocConfig;
use super::std_video::std_level;
use crate::bitstream::rbsp::unescape;
use crate::decode::{ChromaFormat, DecodeCodec};
use crate::error::DecodeError;
use crate::vulkan::copy::FrameArea;

/// The decoder's name in errors.
pub(super) const DECODER: &str = "Vulkan Video";
/// The profiles the driver's H.264 decode names: Baseline, Main, High.
const PROFILES: [u8; 3] = [66, 77, 100];
const SUPPORTED_PROFILES: &str = "66 (Baseline), 77 (Main), and 100 (High)";

/// Parses a whole SPS NAL unit, header byte included.
pub(super) fn parse_sps(nal: &[u8]) -> Result<SeqParameterSet, DecodeError> {
    let rbsp = unescape(nal.get(1..).unwrap_or_default());
    SeqParameterSet::from_bits(BitReader::new(&rbsp[..])).map_err(|error| DecodeError::H264Header {
        decoder: DECODER,
        unit: "SPS",
        reason: format!("{error:?}"),
    })
}

/// Parses a whole PPS NAL unit against the sequence parameter sets in `context`.
pub(super) fn parse_pps(context: &Context, nal: &[u8]) -> Result<PicParameterSet, DecodeError> {
    let rbsp = unescape(nal.get(1..).unwrap_or_default());
    PicParameterSet::from_bits(context, BitReader::new(&rbsp[..])).map_err(|error| {
        DecodeError::H264Header {
            decoder: DECODER,
            unit: "PPS",
            reason: format!("{error:?}"),
        }
    })
}

/// Refuses an SPS by name, most specific first.
pub(super) fn check_sps(sps: &SeqParameterSet) -> Result<(), DecodeError> {
    let chroma = &sps.chroma_info;
    let (luma, chroma_depth) = (
        chroma.bit_depth_luma_minus8 + 8,
        chroma.bit_depth_chroma_minus8 + 8,
    );
    if luma != 8 || chroma_depth != 8 {
        return Err(DecodeError::H264BitDepth {
            decoder: DECODER,
            luma,
            chroma: chroma_depth,
        });
    }
    let format = match chroma.chroma_format.to_u32() {
        0 => Some(ChromaFormat::Monochrome),
        2 => Some(ChromaFormat::Yuv422),
        3 => Some(ChromaFormat::Yuv444),
        _ => None,
    };
    if let Some(chroma) = format {
        return Err(DecodeError::UnsupportedLayout {
            codec: DecodeCodec::H264,
            bit_depth_luma: luma,
            bit_depth_chroma: chroma_depth,
            chroma,
        });
    }
    if !matches!(sps.frame_mbs_flags, FrameMbsFlags::Frames) {
        return Err(DecodeError::H264Interlaced { decoder: DECODER });
    }
    let profile_idc: u8 = sps.profile_idc.into();
    if !PROFILES.contains(&profile_idc) {
        return Err(DecodeError::H264Tool {
            decoder: DECODER,
            field: "profile_idc",
            value: u64::from(profile_idc),
            supported: SUPPORTED_PROFILES,
        });
    }
    Ok(())
}

/// Refuses a PPS by name. Vulkan's `StdVideoH264PictureParameterSet` has no slice groups.
pub(super) fn check_pps(pps: &PicParameterSet) -> Result<(), DecodeError> {
    if let Some(groups) = &pps.slice_groups {
        return Err(DecodeError::H264Tool {
            decoder: DECODER,
            field: "num_slice_groups_minus1",
            value: slice_groups_minus1(groups),
            supported: "0, because Vulkan Video has no slice groups",
        });
    }
    Ok(())
}

/// `num_slice_groups_minus1`, which each map type holds in its own form.
fn slice_groups_minus1(groups: &SliceGroup) -> u64 {
    match groups {
        SliceGroup::Interleaved { run_length_minus1 } => {
            run_length_minus1.len().saturating_sub(1) as u64
        }
        SliceGroup::ForegroundAndLeftover { rectangles } => rectangles.len() as u64,
        SliceGroup::Dispersed {
            num_slice_groups_minus1,
        }
        | SliceGroup::Changing {
            num_slice_groups_minus1,
            ..
        }
        | SliceGroup::ExplicitAssignment {
            num_slice_groups_minus1,
            ..
        } => u64::from(*num_slice_groups_minus1),
    }
}

/// Refuses a level H.264 does not name, or one above `max_level`, the driver's highest
/// `StdVideoH264LevelIdc`.
pub(super) fn check_level(level_idc: u8, max_level: u32) -> Result<(), DecodeError> {
    let refused = |reason: String| DecodeError::VulkanRefused {
        codec: DecodeCodec::H264,
        what: format!("level_idc {level_idc}"),
        reason,
    };
    match std_level(level_idc) {
        None => Err(refused("H.264 names no such level".to_owned())),
        Some(level) if level > max_level => Err(refused(format!(
            "the driver's highest level value is {max_level}, and this level has value {level}"
        ))),
        Some(_) => Ok(()),
    }
}

/// What the decoder needs from one sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SequenceFacts {
    pub(super) profile_idc: u8,
    pub(super) level_idc: u8,
    /// Coded `(width, height)`.
    pub(super) coded: (u32, u32),
    /// The display area inside the coded image.
    pub(super) area: FrameArea,
    /// `Max(max_num_ref_frames, 1)`.
    pub(super) max_references: u32,
    /// `MaxFrameNum`.
    pub(super) max_frame_num: u32,
    pub(super) poc: PocConfig,
    /// `max_num_reorder_frames`, stated or inferred.
    pub(super) reorder_frames: usize,
    /// `max_dec_frame_buffering`, stated or inferred, at least 1.
    pub(super) buffer_frames: usize,
}

impl SequenceFacts {
    /// DPB slots: each reference, and the picture in decode.
    pub(super) const fn slots(&self) -> u32 {
        self.max_references + 1
    }
}

/// Reads the facts of an SPS that `check_sps` passed.
pub(super) fn sequence_facts(sps: &SeqParameterSet) -> Result<SequenceFacts, DecodeError> {
    let width = (sps.pic_width_in_mbs_minus1 + 1).saturating_mul(16);
    let height = (sps.pic_height_in_map_units_minus1 + 1).saturating_mul(16);
    let invalid = || DecodeError::DisplayArea {
        rect: (0, 0, width as i32, height as i32),
        coded: (width, height),
    };
    // 7.4.2.1.1: CropUnitX and CropUnitY are 2 at 4:2:0 with frame_mbs_only_flag 1.
    let (left, right, top, bottom) = sps.frame_cropping.as_ref().map_or((0, 0, 0, 0), |crop| {
        (
            crop.left_offset.saturating_mul(2),
            crop.right_offset.saturating_mul(2),
            crop.top_offset.saturating_mul(2),
            crop.bottom_offset.saturating_mul(2),
        )
    });
    let area_width = width
        .checked_sub(left)
        .and_then(|rest| rest.checked_sub(right))
        .filter(|&rest| rest > 0)
        .ok_or_else(invalid)?;
    let area_height = height
        .checked_sub(top)
        .and_then(|rest| rest.checked_sub(bottom))
        .filter(|&rest| rest > 0)
        .ok_or_else(invalid)?;
    let (reorder_frames, buffer_frames) = output_limits(sps);
    Ok(SequenceFacts {
        profile_idc: sps.profile_idc.into(),
        level_idc: sps.level_idc,
        coded: (width, height),
        area: FrameArea {
            left,
            top,
            width: area_width,
            height: area_height,
        },
        max_references: sps.max_num_ref_frames.max(1),
        max_frame_num: 1 << sps.log2_max_frame_num(),
        poc: poc_config(&sps.pic_order_cnt),
        reorder_frames,
        buffer_frames,
    })
}

fn poc_config(poc: &PicOrderCntType) -> PocConfig {
    match poc {
        PicOrderCntType::TypeZero {
            log2_max_pic_order_cnt_lsb_minus4,
        } => PocConfig::Type0 {
            max_lsb: 1 << (u32::from(*log2_max_pic_order_cnt_lsb_minus4) + 4),
        },
        PicOrderCntType::TypeOne {
            offset_for_non_ref_pic,
            offset_for_top_to_bottom_field,
            offsets_for_ref_frame,
            ..
        } => PocConfig::Type1 {
            offset_for_non_ref_pic: *offset_for_non_ref_pic,
            offset_for_top_to_bottom_field: *offset_for_top_to_bottom_field,
            offsets_for_ref_frame: offsets_for_ref_frame.clone(),
        },
        PicOrderCntType::TypeTwo => PocConfig::Type2,
    }
}

/// `(max_num_reorder_frames, max_dec_frame_buffering)` from the VUI, or as E.2.1 infers.
fn output_limits(sps: &SeqParameterSet) -> (usize, usize) {
    let restrictions = sps
        .vui_parameters
        .as_ref()
        .and_then(|vui| vui.bitstream_restrictions.as_ref());
    if let Some(restrictions) = restrictions {
        return (
            restrictions.max_num_reorder_frames as usize,
            (restrictions.max_dec_frame_buffering as usize).max(1),
        );
    }
    let profile_idc: u8 = sps.profile_idc.into();
    // Intra profiles with constraint_set3_flag hold no reordering.
    let intra_only =
        matches!(profile_idc, 44 | 86 | 100 | 110 | 122 | 244) && sps.constraint_flags.flag3();
    if intra_only {
        return (0, 1);
    }
    let frames = max_dpb_frames(sps);
    // Type 2 output order equals decode order.
    if matches!(sps.pic_order_cnt, PicOrderCntType::TypeTwo) {
        return (0, frames);
    }
    (frames, frames)
}

/// `MaxDpbFrames` of A.3.1, or 16 for a level with no table row.
fn max_dpb_frames(sps: &SeqParameterSet) -> usize {
    let frame_mbs = u64::from(sps.pic_width_in_mbs_minus1 + 1)
        * u64::from(sps.pic_height_in_map_units_minus1 + 1);
    let Some(limits) = sps.level().limits() else {
        return 16;
    };
    (u64::from(limits.max_dpb_mbs) / frame_mbs.max(1)).clamp(1, 16) as usize
}

#[cfg(test)]
mod tests {
    use super::super::test_nal::{REAL_SPS, TestSps, pps_nal, sps_nal};
    use super::*;

    #[test]
    fn a_real_sps_gives_its_coded_size_crop_and_output_limits() {
        let sps = parse_sps(&REAL_SPS).expect("parses");
        check_sps(&sps).expect("passes");
        let facts = sequence_facts(&sps).expect("facts");
        assert_eq!(facts.profile_idc, 100);
        assert_eq!(facts.level_idc, 12);
        assert_eq!(facts.coded, (320, 192));
        assert_eq!(
            facts.area,
            FrameArea {
                left: 0,
                top: 0,
                width: 320,
                height: 180,
            }
        );
        assert_eq!(facts.max_references, 4);
        assert_eq!(facts.slots(), 5);
        assert_eq!(facts.max_frame_num, 16);
        assert_eq!(facts.poc, PocConfig::Type0 { max_lsb: 64 });
        assert_eq!((facts.reorder_frames, facts.buffer_frames), (2, 4));
    }

    #[test]
    fn output_limits_are_inferred_without_a_vui() {
        // 320x192 is 240 macroblocks, and level 3.0 holds 8100, so 33 frames cap at 16.
        let sps = parse_sps(&sps_nal(&TestSps::default())).expect("parses");
        let facts = sequence_facts(&sps).expect("facts");
        assert_eq!((facts.reorder_frames, facts.buffer_frames), (16, 16));

        let type2 = TestSps {
            poc_type: 2,
            ..TestSps::default()
        };
        let sps = parse_sps(&sps_nal(&type2)).expect("parses");
        let facts = sequence_facts(&sps).expect("facts");
        assert_eq!((facts.reorder_frames, facts.buffer_frames), (0, 16));
        assert_eq!(facts.poc, PocConfig::Type2);

        let intra = TestSps {
            constraint_flags: 0x10,
            ..TestSps::default()
        };
        let sps = parse_sps(&sps_nal(&intra)).expect("parses");
        let facts = sequence_facts(&sps).expect("facts");
        assert_eq!((facts.reorder_frames, facts.buffer_frames), (0, 1));
    }

    #[test]
    fn zero_reference_frames_still_take_one_reference_and_two_slots() {
        let none = TestSps {
            refs: 0,
            ..TestSps::default()
        };
        let sps = parse_sps(&sps_nal(&none)).expect("parses");
        let facts = sequence_facts(&sps).expect("facts");
        assert_eq!((facts.max_references, facts.slots()), (1, 2));
    }

    #[test]
    fn each_refused_stream_is_named() {
        let refused = |sps: TestSps| {
            let sps = parse_sps(&sps_nal(&sps)).expect("parses");
            check_sps(&sps).expect_err("refused")
        };
        assert!(matches!(
            refused(TestSps {
                bit_depth_minus8: 2,
                profile_idc: 110,
                ..TestSps::default()
            }),
            DecodeError::H264BitDepth { luma: 10, .. }
        ));
        assert!(matches!(
            refused(TestSps {
                chroma_format: 2,
                profile_idc: 122,
                ..TestSps::default()
            }),
            DecodeError::UnsupportedLayout {
                chroma: ChromaFormat::Yuv422,
                ..
            }
        ));
        assert!(matches!(
            refused(TestSps {
                frame_mbs_only: false,
                ..TestSps::default()
            }),
            DecodeError::H264Interlaced { .. }
        ));
        assert!(matches!(
            refused(TestSps {
                profile_idc: 88,
                ..TestSps::default()
            }),
            DecodeError::H264Tool {
                field: "profile_idc",
                value: 88,
                ..
            }
        ));
    }

    #[test]
    fn a_level_is_checked_against_the_driver_and_the_standard() {
        let level_3_1 = std_level(31).expect("a level");
        assert_eq!(check_level(31, level_3_1), Ok(()));
        assert_eq!(check_level(30, level_3_1), Ok(()));
        let text = check_level(40, level_3_1).expect_err("above").to_string();
        assert!(text.contains("level_idc 40"), "{text}");
        assert!(text.contains("highest level value"), "{text}");
        let text = check_level(99, u32::MAX).expect_err("unknown").to_string();
        assert!(text.contains("level_idc 99"), "{text}");
        assert!(text.contains("no such level"), "{text}");
    }

    #[test]
    fn slice_groups_are_refused_by_name() {
        let mut context = Context::new();
        context.put_seq_param_set(parse_sps(&sps_nal(&TestSps::default())).expect("parses"));
        let plain = parse_pps(&context, &pps_nal(0, 0, 0)).expect("parses");
        check_pps(&plain).expect("passes");
        let grouped = parse_pps(&context, &pps_nal(1, 0, 1)).expect("parses");
        assert!(matches!(
            check_pps(&grouped),
            Err(DecodeError::H264Tool {
                field: "num_slice_groups_minus1",
                value: 1,
                ..
            })
        ));
    }

    #[test]
    fn damaged_units_are_named() {
        let cut = &REAL_SPS[..4];
        assert!(matches!(
            parse_sps(cut),
            Err(DecodeError::H264Header { unit: "SPS", .. })
        ));
        let context = Context::new();
        assert!(matches!(
            parse_pps(&context, &pps_nal(0, 0, 0)),
            Err(DecodeError::H264Header { unit: "PPS", .. })
        ));
    }
}
