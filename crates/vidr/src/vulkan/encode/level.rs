//! The lowest level that holds a frame size and rate. The tables come from Annex A of
//! ITU-T H.264 V16 and ITU-T H.265 V11, Main tier.

/// `(level_idc, MaxMBPS, MaxFS, MaxDpbMbs)` of ITU-T H.264 Table A-1. Level 1b holds the values of
/// level 1, so it takes no row.
const H264_LEVELS: [(u8, u64, u64, u64); 19] = [
    (10, 1_485, 99, 396),
    (11, 3_000, 396, 900),
    (12, 6_000, 396, 2_376),
    (13, 11_880, 396, 2_376),
    (20, 11_880, 396, 2_376),
    (21, 19_800, 792, 4_752),
    (22, 20_250, 1_620, 8_100),
    (30, 40_500, 1_620, 8_100),
    (31, 108_000, 3_600, 18_000),
    (32, 216_000, 5_120, 20_480),
    (40, 245_760, 8_192, 32_768),
    (41, 245_760, 8_192, 32_768),
    (42, 522_240, 8_704, 34_816),
    (50, 589_824, 22_080, 110_400),
    (51, 983_040, 36_864, 184_320),
    (52, 2_073_600, 36_864, 184_320),
    (60, 4_177_920, 139_264, 696_320),
    (61, 8_355_840, 139_264, 696_320),
    (62, 16_711_680, 139_264, 696_320),
];

/// `(general_level_idc, MaxLumaPs, MaxLumaSr)` of ITU-T H.265 Tables A.8 and A.9, Main tier. The
/// rows stop at level 6.2, the top of `StdVideoH265LevelIdc`. That enum has no value for levels
/// 6.3, 7, 7.1, and 7.2 of those tables.
const HEVC_LEVELS: [(u8, u64, u64); 13] = [
    (30, 36_864, 552_960),
    (60, 122_880, 3_686_400),
    (63, 245_760, 7_372_800),
    (90, 552_960, 16_588_800),
    (93, 983_040, 33_177_600),
    (120, 2_228_224, 66_846_720),
    (123, 2_228_224, 133_693_440),
    (150, 8_912_896, 267_386_880),
    (153, 8_912_896, 534_773_760),
    (156, 8_912_896, 1_069_547_520),
    (180, 35_651_584, 1_069_547_520),
    (183, 35_651_584, 2_139_095_040),
    (186, 35_651_584, 4_278_190_080),
];

/// `level_idc` for a frame of `coded` samples at `rate` frames per second, with one reference. A
/// frame past every level takes the top level.
pub(super) fn h264_level(coded: (u32, u32), rate: (u32, u32)) -> u8 {
    let columns = u64::from(coded.0.div_ceil(16));
    let rows = u64::from(coded.1.div_ceil(16));
    let frame = columns * rows;
    let per_second = (frame * u64::from(rate.0)).div_ceil(u64::from(rate.1.max(1)));
    H264_LEVELS
        .iter()
        .find(|&&(_, mbps, fs, dpb)| {
            // MaxFS also bounds each side: width and height at most sqrt(8 * MaxFS).
            frame <= fs
                && columns * columns <= 8 * fs
                && rows * rows <= 8 * fs
                && per_second <= mbps
                && frame <= dpb
        })
        .map_or(62, |&(level, ..)| level)
}

/// `general_level_idc` for a frame of `coded` samples at `rate` frames per second. A frame past
/// every level takes the top level.
pub(super) fn hevc_level(coded: (u32, u32), rate: (u32, u32)) -> u8 {
    let (width, height) = (u64::from(coded.0), u64::from(coded.1));
    let picture = width * height;
    let per_second = (picture * u64::from(rate.0)).div_ceil(u64::from(rate.1.max(1)));
    HEVC_LEVELS
        .iter()
        .find(|&&(_, luma_ps, luma_sr)| {
            // A.4.1: width and height at most sqrt(8 * MaxLumaPs).
            picture <= luma_ps
                && width * width <= 8 * luma_ps
                && height * height <= 8 * luma_ps
                && per_second <= luma_sr
        })
        .map_or(186, |&(level, ..)| level)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h264_takes_the_lowest_level_that_fits() {
        assert_eq!(h264_level((320, 240), (30, 1)), 13);
        assert_eq!(h264_level((640, 368), (30000, 1001)), 30);
        assert_eq!(h264_level((1920, 1088), (30, 1)), 40);
        assert_eq!(h264_level((1920, 1088), (60, 1)), 42);
        assert_eq!(h264_level((3840, 2176), (30, 1)), 51);
    }

    #[test]
    fn a_narrow_h264_frame_takes_a_level_its_longest_side_fits() {
        // 240 macroblocks across need 8 * MaxFS of at least 57600, which MaxFS 5120 lacks.
        assert_eq!(h264_level((3840, 256), (1, 1)), 40);
    }

    #[test]
    fn hevc_takes_the_lowest_level_that_fits() {
        assert_eq!(hevc_level((320, 256), (30, 1)), 60);
        assert_eq!(hevc_level((640, 384), (30000, 1001)), 63);
        assert_eq!(hevc_level((1920, 1088), (30, 1)), 120);
        assert_eq!(hevc_level((3840, 2176), (30, 1)), 150);
        assert_eq!(hevc_level((3840, 2176), (60, 1)), 153);
    }

    #[test]
    fn a_frame_past_every_level_takes_the_top_level() {
        assert_eq!(h264_level((16384, 16384), (240, 1)), 62);
        assert_eq!(hevc_level((16384, 16384), (240, 1)), 186);
    }
}
