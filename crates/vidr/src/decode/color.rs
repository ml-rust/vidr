//! Colour facts a decoder reads from the sequence header.
//!
//! Code points follow ITU-T H.273, which H.264 VUI, HEVC VUI, and the AV1
//! `color_config` share.

use numr::ops::{ChromaSiting, ColorMatrix, ColorRange, YuvSpec};

/// H.273 `MatrixCoefficients` 1: ITU-R BT.709.
pub const MATRIX_BT709: u8 = 1;
/// H.273 `MatrixCoefficients` 2: unspecified.
pub const MATRIX_UNSPECIFIED: u8 = 2;
/// H.273 `MatrixCoefficients` 5: ITU-R BT.470 System B, G. BT.601 625-line.
pub const MATRIX_BT470BG: u8 = 5;
/// H.273 `MatrixCoefficients` 6: SMPTE 170M. BT.601 525-line.
pub const MATRIX_SMPTE170M: u8 = 6;

/// H.273 `TransferCharacteristics` 2: unspecified.
pub const TRANSFER_UNSPECIFIED: u8 = 2;
/// H.273 `ColourPrimaries` 2: unspecified.
pub const PRIMARIES_UNSPECIFIED: u8 = 2;

/// Position of a 4:2:0 chroma sample relative to the luma samples.
///
/// The variants are H.273 `ChromaSampleLocType` 0 to 5.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ChromaLocation {
    /// Type 0: co-sited with luma column `2j`, midway between rows. The
    /// H.264 and HEVC default.
    Left,
    /// Type 1: midway between columns and between rows.
    Center,
    /// Type 2: co-sited with luma column `2j` and row `2i`. AV1
    /// `CSP_COLOCATED`.
    TopLeft,
    /// Type 3: midway between columns, co-sited with row `2i`.
    Top,
    /// Type 4: co-sited with luma column `2j` and row `2i + 1`.
    BottomLeft,
    /// Type 5: midway between columns, co-sited with row `2i + 1`.
    Bottom,
}

impl ChromaLocation {
    /// The location for AV1 `chroma_sample_position`: 1 is `Left`
    /// (`CSP_VERTICAL`), 2 is `TopLeft` (`CSP_COLOCATED`). 0 (`CSP_UNKNOWN`)
    /// and the reserved 3 give `None`.
    pub const fn from_av1(chroma_sample_position: u8) -> Option<Self> {
        match chroma_sample_position {
            1 => Some(Self::Left),
            2 => Some(Self::TopLeft),
            _ => None,
        }
    }

    /// The location for H.264 and HEVC `chroma_sample_loc_type_*_field`,
    /// which is ITU-T H.273 `ChromaSampleLocType`: 0 to 5 map to `Left`,
    /// `Center`, `TopLeft`, `Top`, `BottomLeft`, and `Bottom`. A value above
    /// 5 gives `None`.
    pub const fn from_h273(chroma_sample_loc_type: u8) -> Option<Self> {
        match chroma_sample_loc_type {
            0 => Some(Self::Left),
            1 => Some(Self::Center),
            2 => Some(Self::TopLeft),
            3 => Some(Self::Top),
            4 => Some(Self::BottomLeft),
            5 => Some(Self::Bottom),
            _ => None,
        }
    }

    /// The H.273 `ChromaSampleLocType`, 0 to 5, that H.264 and HEVC VUI carry.
    pub const fn to_h273(self) -> u8 {
        match self {
            Self::Left => 0,
            Self::Center => 1,
            Self::TopLeft => 2,
            Self::Top => 3,
            Self::BottomLeft => 4,
            Self::Bottom => 5,
        }
    }

    /// The AV1 `chroma_sample_position`: 1 for `Left`, 2 for `TopLeft`. AV1
    /// has no code for the other locations, so they give `None`.
    pub const fn to_av1(self) -> Option<u8> {
        match self {
            Self::Left => Some(1),
            Self::TopLeft => Some(2),
            Self::Center | Self::Top | Self::BottomLeft | Self::Bottom => None,
        }
    }

    /// The horizontal siting numr's colour conversion takes.
    ///
    /// | Type | Variant | Horizontal | Vertical |
    /// | --- | --- | --- | --- |
    /// | 0 | `Left` | `Left` | midway between rows |
    /// | 1 | `Center` | `Center` | midway between rows |
    /// | 2 | `TopLeft` | `Left` | on row `2i` |
    /// | 3 | `Top` | `Center` | on row `2i` |
    /// | 4 | `BottomLeft` | `Left` | on row `2i + 1` |
    /// | 5 | `Bottom` | `Center` | on row `2i + 1` |
    ///
    /// numr places chroma midway between luma rows for every siting. This
    /// function keeps the horizontal class only. The vertical offset of
    /// types 2 to 5 is not modelled: the conversion treats them as midway,
    /// which is half a luma row off.
    pub const fn horizontal(self) -> ChromaSiting {
        match self {
            Self::Left | Self::TopLeft | Self::BottomLeft => ChromaSiting::Left,
            Self::Center | Self::Top | Self::Bottom => ChromaSiting::Center,
        }
    }
}

/// Colour description of a decoded frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorFacts {
    /// H.273 `MatrixCoefficients`. 2 means unspecified.
    pub matrix_coefficients: u8,
    /// H.273 `TransferCharacteristics`. 2 means unspecified.
    pub transfer_characteristics: u8,
    /// H.273 `ColourPrimaries`. 2 means unspecified.
    pub colour_primaries: u8,
    /// `video_full_range_flag`: true for full range, false for limited.
    pub full_range: bool,
    /// Chroma sample position, or `None` when the stream signals none.
    /// NVDEC reports none, so vidr reads it from `av1C` for AV1, and from
    /// the SPS VUI for H.264 and HEVC: the top field value of
    /// `chroma_sample_loc_type_top_field`, which also covers frames. It is
    /// `None` for VP9, for an H.264 or HEVC SPS without
    /// `chroma_loc_info_present_flag` or that vidr cannot parse, and for
    /// Annex-B input before the first SPS.
    ///
    /// The full location is kept here. numr models the horizontal class only
    /// (see [`ChromaLocation::horizontal`]), so the vertical offset of
    /// `TopLeft`, `Top`, `BottomLeft`, and `Bottom` is not applied by
    /// [`ColorFacts::yuv_spec`].
    pub chroma_location: Option<ChromaLocation>,
}

impl ColorFacts {
    /// The numr matrix for `matrix_coefficients`, or `None` for a matrix
    /// numr does not convert: unspecified, BT.2020, and the others.
    pub const fn matrix(&self) -> Option<ColorMatrix> {
        match self.matrix_coefficients {
            MATRIX_BT709 => Some(ColorMatrix::Bt709),
            MATRIX_BT470BG | MATRIX_SMPTE170M => Some(ColorMatrix::Bt601),
            _ => None,
        }
    }

    /// The numr code range.
    pub const fn range(&self) -> ColorRange {
        if self.full_range {
            ColorRange::Full
        } else {
            ColorRange::Limited
        }
    }

    /// The numr spec for `ColorOps::yuv_to_rgb`, or `None` when
    /// [`ColorFacts::matrix`] is `None`.
    ///
    /// `default_siting` applies when `chroma_location` is `None`. Pass
    /// `ChromaSiting::Left` for H.264, HEVC, and AV1: a stream that signals
    /// nothing has chroma location type 0, which is `Left`. A signalled
    /// location sets the horizontal siting only, as
    /// [`ChromaLocation::horizontal`] states.
    pub fn yuv_spec(&self, default_siting: ChromaSiting) -> Option<YuvSpec> {
        let siting = match self.chroma_location {
            Some(location) => location.horizontal(),
            None => default_siting,
        };
        self.matrix()
            .map(|matrix| YuvSpec::new(matrix, self.range(), siting))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(matrix: u8, full_range: bool, location: Option<ChromaLocation>) -> ColorFacts {
        ColorFacts {
            matrix_coefficients: matrix,
            transfer_characteristics: 1,
            colour_primaries: 1,
            full_range,
            chroma_location: location,
        }
    }

    #[test]
    fn bt709_limited_maps_to_numr() {
        let spec = facts(MATRIX_BT709, false, None).yuv_spec(ChromaSiting::Left);
        assert_eq!(
            spec,
            Some(YuvSpec::new(
                ColorMatrix::Bt709,
                ColorRange::Limited,
                ChromaSiting::Left
            ))
        );
    }

    #[test]
    fn bt601_codes_share_a_matrix() {
        assert_eq!(
            facts(MATRIX_BT470BG, true, None).matrix(),
            Some(ColorMatrix::Bt601)
        );
        assert_eq!(
            facts(MATRIX_SMPTE170M, true, None).matrix(),
            Some(ColorMatrix::Bt601)
        );
        assert_eq!(
            facts(MATRIX_SMPTE170M, true, None).range(),
            ColorRange::Full
        );
    }

    #[test]
    fn unspecified_and_bt2020_have_no_spec() {
        assert_eq!(
            facts(MATRIX_UNSPECIFIED, false, None).yuv_spec(ChromaSiting::Left),
            None
        );
        assert_eq!(facts(9, false, None).yuv_spec(ChromaSiting::Left), None);
    }

    #[test]
    fn signalled_location_wins_over_the_default() {
        let spec =
            facts(MATRIX_BT709, false, Some(ChromaLocation::Center)).yuv_spec(ChromaSiting::Left);
        assert_eq!(spec.map(|spec| spec.siting), Some(ChromaSiting::Center));
    }

    #[test]
    fn h273_types_map_to_locations_and_sitings() {
        let expected = [
            (ChromaLocation::Left, ChromaSiting::Left),
            (ChromaLocation::Center, ChromaSiting::Center),
            (ChromaLocation::TopLeft, ChromaSiting::Left),
            (ChromaLocation::Top, ChromaSiting::Center),
            (ChromaLocation::BottomLeft, ChromaSiting::Left),
            (ChromaLocation::Bottom, ChromaSiting::Center),
        ];
        for (kind, (location, siting)) in expected.into_iter().enumerate() {
            assert_eq!(ChromaLocation::from_h273(kind as u8), Some(location));
            assert_eq!(location.to_h273(), kind as u8);
            assert_eq!(location.horizontal(), siting);
        }
        assert_eq!(ChromaLocation::from_h273(6), None);
    }

    #[test]
    fn av1_positions_map_to_locations() {
        assert_eq!(ChromaLocation::from_av1(0), None);
        assert_eq!(ChromaLocation::from_av1(1), Some(ChromaLocation::Left));
        assert_eq!(ChromaLocation::from_av1(2), Some(ChromaLocation::TopLeft));
        assert_eq!(ChromaLocation::from_av1(3), None);
        assert_eq!(ChromaLocation::Left.to_av1(), Some(1));
        assert_eq!(ChromaLocation::TopLeft.to_av1(), Some(2));
        assert_eq!(ChromaLocation::Center.to_av1(), None);
        assert_eq!(ChromaLocation::TopLeft.horizontal(), ChromaSiting::Left);
        assert_eq!(ChromaLocation::Top.horizontal(), ChromaSiting::Center);
    }
}
