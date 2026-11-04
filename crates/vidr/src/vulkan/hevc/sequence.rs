//! What this backend refuses in a sequence, and what a sequence asks of a session.

use crate::bitstream::hevc_sps_full::SeqParameterSet;
use crate::decode::hevc_order::buffer::DpbLimits;
use crate::decode::{ChromaFormat, DecodeCodec, PixelFormat};
use crate::error::DecodeError;
use crate::vulkan::copy::FrameArea;
pub(super) use crate::vulkan::error::DECODER;

/// `STD_VIDEO_H265_PROFILE_IDC_MAIN`, `_MAIN_10`, and `_MAIN_STILL_PICTURE`.
const PROFILES: [u8; 3] = [1, 2, 3];
const SUPPORTED_PROFILES: &str = "1 (Main), 2 (Main 10), and 3 (Main Still Picture)";
/// `STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE`, which also bounds `NumPicTotalCurr`.
pub(super) const MAX_CURRENT_REFERENCES: u32 = 8;

/// What a sequence asks of a session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SequenceFacts {
    /// `StdVideoH265ProfileIdc`.
    pub(super) std_profile_idc: u32,
    /// `StdVideoH265LevelIdc`.
    pub(super) std_level: u32,
    pub(super) format: PixelFormat,
    pub(super) coded: (u32, u32),
    pub(super) area: FrameArea,
    pub(super) limits: DpbLimits,
    pub(super) slots: u32,
    pub(super) active_references: u32,
}

/// The `StdVideoH265LevelIdc` of `general_level_idc`, which is 30 times the level.
pub(super) const fn std_level(level_idc: u8) -> Option<u32> {
    Some(match level_idc {
        30 => 0,
        60 => 1,
        63 => 2,
        90 => 3,
        93 => 4,
        120 => 5,
        123 => 6,
        150 => 7,
        153 => 8,
        156 => 9,
        180 => 10,
        183 => 11,
        186 => 12,
        _ => return None,
    })
}

/// The profile a stream decodes under: `general_profile_idc`, or the first profile its
/// `general_profile_compatibility_flag` lists.
fn stream_profile(profile_idc: u8, compatibility_flags: u32) -> Option<u8> {
    if PROFILES.contains(&profile_idc) {
        return Some(profile_idc);
    }
    // Flag `j` sits `j` bits below the top.
    PROFILES
        .into_iter()
        .find(|&profile| compatibility_flags & (1 << (31 - u32::from(profile))) != 0)
}

/// Refuses a sequence by name, most specific first, and states what it asks of a session.
pub(super) fn sequence_facts(sps: &SeqParameterSet) -> Result<SequenceFacts, DecodeError> {
    let format = check_layout(sps)?;
    let general = &sps.profile_tier_level.general;
    let Some(profile) = stream_profile(general.profile_idc, general.compatibility_flags) else {
        return Err(DecodeError::HevcTool {
            decoder: DECODER,
            field: "general_profile_idc",
            value: u64::from(general.profile_idc),
            supported: SUPPORTED_PROFILES,
        });
    };
    let Some(std_level) = std_level(general.level_idc) else {
        return Err(DecodeError::HevcTool {
            decoder: DECODER,
            field: "general_level_idc",
            value: u64::from(general.level_idc),
            supported: "30 times a level from 1 to 6.2",
        });
    };
    let limits = DpbLimits::from_sps(sps);
    let buffering = limits.max_dec_buffering as u32;
    Ok(SequenceFacts {
        std_profile_idc: u32::from(profile),
        std_level,
        format,
        coded: (
            sps.pic_width_in_luma_samples,
            sps.pic_height_in_luma_samples,
        ),
        area: conformance_area(sps),
        limits,
        // Each stored picture, and the current one.
        slots: buffering + 1,
        active_references: buffering.saturating_sub(1).clamp(1, MAX_CURRENT_REFERENCES),
    })
}

/// Refuses all but 4:2:0 at equal 8 or 10 bits.
fn check_layout(sps: &SeqParameterSet) -> Result<PixelFormat, DecodeError> {
    let depth = |bits: u32| u8::try_from(bits).unwrap_or(u8::MAX);
    let (luma, chroma) = (depth(sps.bit_depth_y()), depth(sps.bit_depth_c()));
    let layout = match sps.chroma_format_idc {
        0 => ChromaFormat::Monochrome,
        1 => ChromaFormat::Yuv420,
        2 => ChromaFormat::Yuv422,
        _ => ChromaFormat::Yuv444,
    };
    match PixelFormat::for_bit_depth(luma) {
        Some(format) if layout == ChromaFormat::Yuv420 && luma == chroma => Ok(format),
        _ => Err(DecodeError::UnsupportedLayout {
            codec: DecodeCodec::Hevc,
            bit_depth_luma: luma,
            bit_depth_chroma: chroma,
            chroma: layout,
        }),
    }
}

/// The conformance window of 7.4.3.2.1 in luma samples. An oversized window gives a size of
/// 0, which the display size check refuses.
fn conformance_area(sps: &SeqParameterSet) -> FrameArea {
    let (unit_x, unit_y) = (sps.sub_width_c(), sps.sub_height_c());
    let left = sps.conf_win_left_offset.saturating_mul(unit_x);
    let right = sps.conf_win_right_offset.saturating_mul(unit_x);
    let top = sps.conf_win_top_offset.saturating_mul(unit_y);
    let bottom = sps.conf_win_bottom_offset.saturating_mul(unit_y);
    FrameArea {
        left,
        top,
        width: sps
            .pic_width_in_luma_samples
            .saturating_sub(left.saturating_add(right)),
        height: sps
            .pic_height_in_luma_samples
            .saturating_sub(top.saturating_add(bottom)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sps() -> SeqParameterSet {
        use crate::bitstream::hevc_sps_full::tests as sps_tests;
        sps_tests::valid(&sps_tests::Spec::default())
    }

    #[test]
    fn levels_follow_the_header_values() {
        assert_eq!(std_level(30), Some(0));
        assert_eq!(std_level(93), Some(4));
        assert_eq!(std_level(186), Some(12));
        assert_eq!(std_level(33), None);
    }

    #[test]
    fn a_profile_comes_from_the_idc_or_its_compatibility_flags() {
        assert_eq!(stream_profile(1, 0), Some(1));
        assert_eq!(stream_profile(3, 0), Some(3));
        assert_eq!(stream_profile(4, 0), None);
        // Flag 2 on an idc of 0.
        assert_eq!(stream_profile(0, 0x2000_0000), Some(2));
    }

    #[test]
    fn layouts_are_refused_by_name() {
        let mut stream = sps();
        stream.bit_depth_luma_minus8 = 2;
        stream.bit_depth_chroma_minus8 = 0;
        let error = check_layout(&stream).expect_err("unequal depth");
        assert!(matches!(
            error,
            DecodeError::UnsupportedLayout {
                bit_depth_luma: 10,
                bit_depth_chroma: 8,
                ..
            }
        ));
        let mut stream = sps();
        stream.bit_depth_luma_minus8 = 4;
        stream.bit_depth_chroma_minus8 = 4;
        assert!(check_layout(&stream).is_err(), "12 bits");
        let mut stream = sps();
        stream.chroma_format_idc = 3;
        assert!(check_layout(&stream).is_err(), "4:4:4");
        let mut stream = sps();
        stream.bit_depth_luma_minus8 = 2;
        stream.bit_depth_chroma_minus8 = 2;
        assert_eq!(check_layout(&stream), Ok(PixelFormat::P010));
    }

    #[test]
    fn other_profiles_are_refused_by_name() {
        let mut stream = sps();
        stream.profile_tier_level.general.profile_idc = 4;
        stream.profile_tier_level.general.compatibility_flags = 0;
        let text = sequence_facts(&stream).expect_err("refused").to_string();
        assert!(text.contains("general_profile_idc equal to 4"), "{text}");
    }

    #[test]
    fn a_conformance_window_counts_in_chroma_units() {
        let mut stream = sps();
        stream.pic_width_in_luma_samples = 128;
        stream.pic_height_in_luma_samples = 96;
        stream.conf_win_left_offset = 1;
        stream.conf_win_right_offset = 2;
        stream.conf_win_top_offset = 3;
        stream.conf_win_bottom_offset = 4;
        assert_eq!(
            conformance_area(&stream),
            FrameArea {
                left: 2,
                top: 6,
                width: 122,
                height: 82,
            }
        );
        stream.conf_win_right_offset = u32::MAX;
        assert_eq!(conformance_area(&stream).width, 0);
    }

    #[test]
    fn a_session_holds_the_buffer_and_the_current_picture() {
        let mut stream = sps();
        stream.sps_max_dec_pic_buffering_minus1 = [15; 7];
        stream.profile_tier_level.general.profile_idc = 1;
        stream.profile_tier_level.general.level_idc = 93;
        let facts = sequence_facts(&stream).expect("facts");
        assert_eq!(facts.slots, 17);
        assert_eq!(facts.active_references, MAX_CURRENT_REFERENCES);
        stream.sps_max_dec_pic_buffering_minus1 = [0; 7];
        let facts = sequence_facts(&stream).expect("facts");
        assert_eq!(facts.slots, 2);
        assert_eq!(facts.active_references, 1);
    }
}
