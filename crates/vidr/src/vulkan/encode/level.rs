//! The lowest level that holds a frame size, a frame rate, and a bitrate. The tables come from
//! Annex A of ITU-T H.264 V16 and ITU-T H.265 V11, Main tier.
//!
//! A target bitrate counts every bit of the stream, and each VCL limit sits below its NAL limit. So
//! a target under the VCL limit meets both. A constant quantiser states no target, and only the
//! size and rate choose its level.

use crate::encode::EncodeCodec;
use crate::error::EncodeError;

/// `(level_idc, MaxMBPS, MaxFS, MaxDpbMbs, MaxBR)` of ITU-T H.264 Table A-1. Level 1b differs from
/// level 1 only in MaxBR and MaxCPB, and the next row covers it, so it takes no row.
const H264_LEVELS: [(u8, u64, u64, u64, u64); 19] = [
    (10, 1_485, 99, 396, 64),
    (11, 3_000, 396, 900, 192),
    (12, 6_000, 396, 2_376, 384),
    (13, 11_880, 396, 2_376, 768),
    (20, 11_880, 396, 2_376, 2_000),
    (21, 19_800, 792, 4_752, 4_000),
    (22, 20_250, 1_620, 8_100, 4_000),
    (30, 40_500, 1_620, 8_100, 10_000),
    (31, 108_000, 3_600, 18_000, 14_000),
    (32, 216_000, 5_120, 20_480, 20_000),
    (40, 245_760, 8_192, 32_768, 20_000),
    (41, 245_760, 8_192, 32_768, 50_000),
    (42, 522_240, 8_704, 34_816, 50_000),
    (50, 589_824, 22_080, 110_400, 135_000),
    (51, 983_040, 36_864, 184_320, 240_000),
    (52, 2_073_600, 36_864, 184_320, 240_000),
    (60, 4_177_920, 139_264, 696_320, 240_000),
    (61, 8_355_840, 139_264, 696_320, 480_000),
    (62, 16_711_680, 139_264, 696_320, 800_000),
];
/// `cpbBrVclFactor` of ITU-T H.264 Table A-2 for High, in bits per second of MaxBR.
const H264_HIGH_BR_VCL_FACTOR: u64 = 1_250;
/// The first `level_idc` with `fR` of 1 / 300 in ITU-T H.264 A.3.1.
const H264_FAST_LEVEL: u8 = 60;
/// Frames per second at `fR` of 1 / 172, for a frame below level 6.
const H264_RATE_LIMIT: u64 = 172;
/// Frames per second at `fR` of 1 / 300, for levels 6, 6.1, and 6.2.
const H264_FAST_RATE_LIMIT: u64 = 300;

/// `(general_level_idc, MaxLumaPs, MaxLumaSr, MaxBR)` of ITU-T H.265 Tables A.8 and A.9, Main tier.
/// The rows stop at level 6.2, the top of `StdVideoH265LevelIdc`. That enum has no value for levels
/// 6.3, 7, 7.1, and 7.2 of those tables.
const HEVC_LEVELS: [(u8, u64, u64, u64); 13] = [
    (30, 36_864, 552_960, 128),
    (60, 122_880, 3_686_400, 1_500),
    (63, 245_760, 7_372_800, 3_000),
    (90, 552_960, 16_588_800, 6_000),
    (93, 983_040, 33_177_600, 10_000),
    (120, 2_228_224, 66_846_720, 12_000),
    (123, 2_228_224, 133_693_440, 20_000),
    (150, 8_912_896, 267_386_880, 25_000),
    (153, 8_912_896, 534_773_760, 40_000),
    (156, 8_912_896, 1_069_547_520, 60_000),
    (180, 35_651_584, 1_069_547_520, 60_000),
    (183, 35_651_584, 2_139_095_040, 120_000),
    (186, 35_651_584, 4_278_190_080, 240_000),
];
/// `BrVclFactor` of ITU-T H.265 A.4.2 for Main and Main 10: `CpbVclFactor` 1000 of Table A.10, with
/// `HbrFactor` 1.
const HEVC_BR_VCL_FACTOR: u64 = 1_000;
/// Frames per second at `fR` of 1 / 300, which ITU-T H.265 A.4.2 states for the Main tier.
const HEVC_RATE_LIMIT: u64 = 300;

/// Whether `rate`, as numerator and denominator, stays at or under `limit` frames per second.
fn rate_within(rate: (u32, u32), limit: u64) -> bool {
    u64::from(rate.0) <= limit * u64::from(rate.1.max(1))
}

/// The refusal of a stream past each level of `codec`.
fn past_every_level(
    codec: EncodeCodec,
    coded: (u32, u32),
    rate: (u32, u32),
    top_bitrate: Option<u64>,
    reason: &str,
) -> EncodeError {
    let bitrate = top_bitrate.map_or_else(String::new, |top| format!(" and {top} bits per second"));
    EncodeError::VulkanRefused {
        codec,
        what: format!(
            "a frame of {}x{} at {}/{} frames per second{bitrate}",
            coded.0, coded.1, rate.0, rate.1
        ),
        reason: reason.to_owned(),
    }
}

/// `level_idc` for a frame of `coded` samples at `rate` frames per second, with one reference, up to
/// `top_bitrate` bits per second for High.
///
/// # Errors
///
/// [`EncodeError::VulkanRefused`] for a stream past each row of Table A-1.
pub(super) fn h264_level(
    coded: (u32, u32),
    rate: (u32, u32),
    top_bitrate: Option<u64>,
) -> Result<u8, EncodeError> {
    let columns = u64::from(coded.0.div_ceil(16));
    let rows = u64::from(coded.1.div_ceil(16));
    let frame = columns * rows;
    let per_second = (frame * u64::from(rate.0)).div_ceil(u64::from(rate.1.max(1)));
    let top = top_bitrate.unwrap_or(0);
    H264_LEVELS
        .iter()
        .find(|&&(level, mbps, fs, dpb, max_br)| {
            let rate_limit = if level < H264_FAST_LEVEL {
                H264_RATE_LIMIT
            } else {
                H264_FAST_RATE_LIMIT
            };
            // MaxFS also bounds each side: width and height at most sqrt(8 * MaxFS).
            frame <= fs
                && columns * columns <= 8 * fs
                && rows * rows <= 8 * fs
                && per_second <= mbps
                && frame <= dpb
                && rate_within(rate, rate_limit)
                && top <= max_br * H264_HIGH_BR_VCL_FACTOR
        })
        .map(|&(level, ..)| level)
        .ok_or_else(|| {
            past_every_level(
                EncodeCodec::H264,
                coded,
                rate,
                top_bitrate,
                "ITU-T H.264 Table A-1 stops at level 6.2, and that level holds no such stream",
            )
        })
}

/// `general_level_idc` for a frame of `coded` samples at `rate` frames per second, up to
/// `top_bitrate` bits per second for Main or Main 10, Main tier.
///
/// # Errors
///
/// [`EncodeError::VulkanRefused`] for a stream past level 6.2, the top of `StdVideoH265LevelIdc`.
/// That covers a rate past 300 frames per second at any size.
pub(super) fn hevc_level(
    coded: (u32, u32),
    rate: (u32, u32),
    top_bitrate: Option<u64>,
) -> Result<u8, EncodeError> {
    let (width, height) = (u64::from(coded.0), u64::from(coded.1));
    let picture = width * height;
    let per_second = (picture * u64::from(rate.0)).div_ceil(u64::from(rate.1.max(1)));
    let top = top_bitrate.unwrap_or(0);
    HEVC_LEVELS
        .iter()
        .find(|&&(_, luma_ps, luma_sr, max_br)| {
            // A.4.1: width and height at most sqrt(8 * MaxLumaPs).
            picture <= luma_ps
                && width * width <= 8 * luma_ps
                && height * height <= 8 * luma_ps
                && per_second <= luma_sr
                && rate_within(rate, HEVC_RATE_LIMIT)
                && top <= max_br * HEVC_BR_VCL_FACTOR
        })
        .map(|&(level, ..)| level)
        .ok_or_else(|| {
            past_every_level(
                EncodeCodec::Hevc,
                coded,
                rate,
                top_bitrate,
                "StdVideoH265LevelIdc stops at level 6.2, and the Main tier of that level holds no \
                 such stream. ITU-T H.265 A.4.2 also stops at 300 frames per second",
            )
        })
}

/// Refuses a level past what the driver lists. `std_level` and `driver_max` are values of the codec's
/// Std enum, and `level_idc` is the same level as the bitstream states it.
///
/// # Errors
///
/// [`EncodeError::VulkanRefused`] when `std_level` sits above `driver_max`.
pub(super) fn check_driver_level(
    codec: EncodeCodec,
    level_idc: u8,
    std_level: u32,
    driver_max: u32,
) -> Result<(), EncodeError> {
    if std_level <= driver_max {
        return Ok(());
    }
    Err(EncodeError::VulkanRefused {
        codec,
        what: format!("level_idc {level_idc}, which this size, rate, and bitrate need"),
        reason: format!("the driver lists at most Std level value {driver_max}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h264_takes_the_lowest_level_that_fits() {
        assert_eq!(h264_level((320, 240), (30, 1), None), Ok(13));
        assert_eq!(h264_level((640, 368), (30000, 1001), None), Ok(30));
        assert_eq!(h264_level((1920, 1088), (30, 1), None), Ok(40));
        assert_eq!(h264_level((1920, 1088), (60, 1), None), Ok(42));
        assert_eq!(h264_level((3840, 2176), (30, 1), None), Ok(51));
    }

    #[test]
    fn a_narrow_h264_frame_takes_a_level_its_longest_side_fits() {
        // 240 macroblocks across need 8 * MaxFS of at least 57600, which MaxFS 5120 lacks.
        assert_eq!(h264_level((3840, 256), (1, 1), None), Ok(40));
    }

    /// 1920x1088 at 30 is 8160 macroblocks and 244800 per second, which level 4 holds. High takes
    /// MaxBR times 1250: 25 Mb/s at level 4, 62.5 Mb/s at levels 4.1 and 4.2, 168.75 Mb/s at level 5.
    #[test]
    fn an_h264_bitrate_past_maxbr_takes_a_higher_level() {
        let at = |top| h264_level((1920, 1088), (30, 1), Some(top));
        assert_eq!(at(25_000_000), Ok(40));
        assert_eq!(at(25_000_001), Ok(41));
        assert_eq!(at(62_500_000), Ok(41));
        assert_eq!(at(62_500_001), Ok(50));
    }

    /// 320x240 is 300 macroblocks. At 172 that is 51600 per second, which level 3 at 40500 lacks and
    /// level 3.1 holds. `fR` shuts each level below 6 past 172 frames per second, and level 6
    /// and up past 300.
    #[test]
    fn an_h264_frame_rate_past_fr_takes_level_6_and_then_a_refusal() {
        assert_eq!(h264_level((320, 240), (172, 1), None), Ok(31));
        assert_eq!(h264_level((320, 240), (173, 1), None), Ok(60));
        assert_eq!(h264_level((320, 240), (300, 1), None), Ok(60));
        let text = h264_level((320, 240), (301, 1), None)
            .expect_err("refused")
            .to_string();
        assert!(text.contains("301/1 frames per second"), "{text}");
        assert!(text.contains("Table A-1"), "{text}");
    }

    #[test]
    fn hevc_takes_the_lowest_level_that_fits() {
        assert_eq!(hevc_level((320, 256), (30, 1), None), Ok(60));
        assert_eq!(hevc_level((640, 384), (30000, 1001), None), Ok(63));
        assert_eq!(hevc_level((1920, 1088), (30, 1), None), Ok(120));
        assert_eq!(hevc_level((3840, 2176), (30, 1), None), Ok(150));
        assert_eq!(hevc_level((3840, 2176), (60, 1), None), Ok(153));
    }

    /// Main tier MaxBR times 1000: 12 Mb/s at level 4, 20 Mb/s at 4.1, 25 Mb/s at 5, 240 Mb/s at 6.2.
    #[test]
    fn an_hevc_bitrate_past_maxbr_takes_a_higher_level_then_a_refusal() {
        let at = |top| hevc_level((1920, 1088), (30, 1), Some(top));
        assert_eq!(at(12_000_000), Ok(120));
        assert_eq!(at(12_000_001), Ok(123));
        assert_eq!(at(20_000_001), Ok(150));
        assert_eq!(at(240_000_000), Ok(186));
        let text = at(240_000_001).expect_err("refused").to_string();
        assert!(text.contains("240000001 bits per second"), "{text}");
        assert!(text.contains("StdVideoH265LevelIdc"), "{text}");
    }

    /// 8192x4320 is 35389440 samples. At 120 that is 4246732800 per second, under 4278190080 of
    /// level 6.2. At 121 it is 4282122240, past that row.
    #[test]
    fn an_hevc_sample_rate_past_level_6_2_is_refused() {
        assert_eq!(hevc_level((8192, 4320), (120, 1), None), Ok(186));
        assert!(matches!(
            hevc_level((8192, 4320), (121, 1), None),
            Err(EncodeError::VulkanRefused { .. })
        ));
    }

    /// 320x256 is 81920 samples. At 300 that is 24576000 per second, which level 3.1 holds. `fR`
    /// shuts the Main tier past 300 frames per second at any size.
    #[test]
    fn an_hevc_frame_rate_past_300_is_refused() {
        assert_eq!(hevc_level((320, 256), (300, 1), None), Ok(93));
        let text = hevc_level((320, 256), (301, 1), None)
            .expect_err("refused")
            .to_string();
        assert!(text.contains("300 frames per second"), "{text}");
    }

    #[test]
    fn a_frame_past_every_level_is_refused_by_name() {
        for error in [
            h264_level((16384, 16384), (240, 1), None),
            hevc_level((16384, 16384), (240, 1), None),
        ] {
            let text = error.expect_err("refused").to_string();
            assert!(text.contains("16384x16384"), "{text}");
        }
        assert!(h264_level((1920, 1088), (30, 1), Some(1_000_000_001)).is_err());
        assert_eq!(
            h264_level((1920, 1088), (30, 1), Some(1_000_000_000)),
            Ok(62)
        );
    }

    #[test]
    fn a_level_past_the_driver_is_refused_by_name() {
        assert_eq!(check_driver_level(EncodeCodec::H264, 40, 10, 10), Ok(()));
        let text = check_driver_level(EncodeCodec::Hevc, 186, 12, 9)
            .expect_err("refused")
            .to_string();
        assert!(text.contains("level_idc 186"), "{text}");
        assert!(text.contains("value 9"), "{text}");
    }
}
