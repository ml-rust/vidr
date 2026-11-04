//! The rules that turn stream colour facts into a conversion plan.
//!
//! The rules come from the PoC (`full_video_io.py`, `ai_upscale.py`). Code
//! points follow ITU-T H.273.

use numr::ops::{ChromaSiting, ColorMatrix, YuvSpec};

use crate::decode::{ColorFacts, MATRIX_UNSPECIFIED, PRIMARIES_UNSPECIFIED, TRANSFER_UNSPECIFIED};
use crate::error::ColorError;

/// H.273 `TransferCharacteristics` 16: SMPTE ST 2084, the HDR10 transfer.
const TRANSFER_PQ: u8 = 16;
/// H.273 `TransferCharacteristics` 18: ARIB STD-B67, the HLG transfer.
const TRANSFER_HLG: u8 = 18;
/// H.273 `MatrixCoefficients` 9: BT.2020 non-constant luminance.
const MATRIX_BT2020_NCL: u8 = 9;
/// H.273 `MatrixCoefficients` 10: BT.2020 constant luminance.
const MATRIX_BT2020_CL: u8 = 10;
/// H.273 `ColourPrimaries` 9: BT.2020.
const PRIMARIES_BT2020: u8 = 9;

/// Fills the unspecified fields of `bitstream` from the container tags.
///
/// A bitstream code other than 2 wins. Code 2 takes the container value.
/// The range is a flag with no unspecified state: `true` always stays, and
/// `false` takes a container value of full. The chroma location stays, since
/// [`ContainerColour`](containr::ContainerColour) holds none.
///
/// ffprobe has two answers, and this function follows neither in every case:
///
/// | SPS | Container | ffprobe stream | ffprobe frame | This function |
/// | --- | --- | --- | --- | --- |
/// | code A | code B | B | A | A |
/// | code 2 | code B | B | 2 | B |
/// | no signal | code B | B | B | B |
/// | full | limited | limited | full | full |
/// | limited | full | full | limited | full |
///
/// The last row is a limit of [`ColorFacts::full_range`]: `false` is both
/// "signalled limited" and "not signalled".
#[cfg(any(feature = "mp4", feature = "mkv"))]
pub fn merge_container(
    bitstream: ColorFacts,
    container: Option<containr::ContainerColour>,
) -> ColorFacts {
    let Some(container) = container else {
        return bitstream;
    };
    let fill = |code: u8, unspecified: u8, tag: Option<u8>| {
        if code == unspecified {
            tag.unwrap_or(code)
        } else {
            code
        }
    };
    ColorFacts {
        matrix_coefficients: fill(
            bitstream.matrix_coefficients,
            MATRIX_UNSPECIFIED,
            container.matrix,
        ),
        transfer_characteristics: fill(
            bitstream.transfer_characteristics,
            TRANSFER_UNSPECIFIED,
            container.transfer,
        ),
        colour_primaries: fill(
            bitstream.colour_primaries,
            PRIMARIES_UNSPECIFIED,
            container.primaries,
        ),
        full_range: bitstream.full_range || container.full_range == Some(true),
        chroma_location: bitstream.chroma_location,
    }
}

/// Refuses HDR colour, with the PoC's rule: transfer 16 or 18, matrix 9 or
/// 10, or primaries 9. Transfers 14 and 15 alone pass, since they are the
/// BT.2020 SDR transfers.
///
/// # Errors
///
/// [`ColorError::Hdr`] for the first HDR code, in the order transfer,
/// matrix, primaries.
pub fn check_sdr(facts: &ColorFacts) -> Result<(), ColorError> {
    let code = facts.transfer_characteristics;
    if code == TRANSFER_PQ || code == TRANSFER_HLG {
        return Err(ColorError::Hdr {
            field: "TransferCharacteristics",
            code,
        });
    }
    let code = facts.matrix_coefficients;
    if code == MATRIX_BT2020_NCL || code == MATRIX_BT2020_CL {
        return Err(ColorError::Hdr {
            field: "MatrixCoefficients",
            code,
        });
    }
    let code = facts.colour_primaries;
    if code == PRIMARIES_BT2020 {
        return Err(ColorError::Hdr {
            field: "ColourPrimaries",
            code,
        });
    }
    Ok(())
}

/// The conversions and output tags of one stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorPlan {
    /// The spec from decoded planes to RGB.
    pub decode: YuvSpec,
    /// The spec from RGB to the planes the encoder takes. It equals
    /// `decode`, so the output keeps the source's matrix and range.
    pub encode: YuvSpec,
    /// The colour facts the output carries. An unspecified or reserved
    /// code stays unspecified, so the output claims nothing the source did
    /// not.
    pub tags: ColorFacts,
}

/// Builds the plan for `facts`, after [`check_sdr`].
///
/// An unspecified matrix converts with BT.601 both ways, as the swscale
/// default the PoC relies on. Its tag stays unspecified. The range is limited
/// unless `facts` says full. A chroma location passes through to the tags,
/// and its horizontal class sets the siting. No location gives
/// [`ChromaSiting::Left`], the default of H.264, HEVC and AV1.
///
/// # Errors
///
/// [`ColorError::Hdr`] from [`check_sdr`], and
/// [`ColorError::UnsupportedMatrix`] for a matrix outside BT.709, the BT.601
/// variants, FCC, SMPTE 240M and unspecified.
pub fn color_plan(facts: &ColorFacts) -> Result<ColorPlan, ColorError> {
    check_sdr(facts)?;
    let matrix = if facts.matrix_coefficients == MATRIX_UNSPECIFIED {
        ColorMatrix::Bt601
    } else {
        facts.matrix().ok_or(ColorError::UnsupportedMatrix {
            code: facts.matrix_coefficients,
        })?
    };
    let siting = match facts.chroma_location {
        Some(location) => location.horizontal(),
        None => ChromaSiting::Left,
    };
    let spec = YuvSpec::new(matrix, facts.range(), siting);
    Ok(ColorPlan {
        decode: spec,
        encode: spec,
        tags: ColorFacts {
            transfer_characteristics: tag(facts.transfer_characteristics, TRANSFER_UNSPECIFIED),
            colour_primaries: tag(facts.colour_primaries, PRIMARIES_UNSPECIFIED),
            ..*facts
        },
    })
}

/// H.273 reserves 0 and 3 for transfer and primaries. The PoC writes no tag
/// for a reserved value.
const fn tag(code: u8, unspecified: u8) -> u8 {
    match code {
        0 | 3 => unspecified,
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use numr::ops::ColorRange;

    use super::*;
    use crate::decode::{
        ChromaLocation, MATRIX_BT470BG, MATRIX_BT709, MATRIX_FCC, MATRIX_SMPTE170M,
        MATRIX_SMPTE240M,
    };

    fn facts(matrix: u8, transfer: u8, primaries: u8) -> ColorFacts {
        ColorFacts {
            matrix_coefficients: matrix,
            transfer_characteristics: transfer,
            colour_primaries: primaries,
            full_range: false,
            chroma_location: None,
        }
    }

    #[test]
    fn hdr_codes_are_refused_with_their_code() {
        let cases = [
            (facts(1, 16, 1), "TransferCharacteristics", 16),
            (facts(1, 18, 1), "TransferCharacteristics", 18),
            (facts(9, 1, 1), "MatrixCoefficients", 9),
            (facts(10, 1, 1), "MatrixCoefficients", 10),
            (facts(1, 1, 9), "ColourPrimaries", 9),
        ];
        for (facts, field, code) in cases {
            let error = check_sdr(&facts);
            assert_eq!(error, Err(ColorError::Hdr { field, code }));
            let message = error.map_or_else(|error| error.to_string(), |()| String::new());
            assert!(message.contains(&format!("{field} {code}")), "{message}");
            assert!(message.contains("HDR"), "{message}");
            assert!(!message.contains(';'), "{message}");
        }
    }

    #[test]
    fn the_transfer_is_named_before_the_matrix_and_primaries() {
        assert_eq!(
            check_sdr(&facts(9, 16, 9)),
            Err(ColorError::Hdr {
                field: "TransferCharacteristics",
                code: 16,
            })
        );
        assert_eq!(
            check_sdr(&facts(9, 1, 9)),
            Err(ColorError::Hdr {
                field: "MatrixCoefficients",
                code: 9,
            })
        );
    }

    #[test]
    fn bt2020_sdr_transfers_alone_pass() {
        assert_eq!(check_sdr(&facts(1, 14, 1)), Ok(()));
        assert_eq!(check_sdr(&facts(1, 15, 1)), Ok(()));
        assert_eq!(check_sdr(&facts(2, 2, 2)), Ok(()));
    }

    #[test]
    fn each_supported_matrix_maps_both_ways() {
        let cases = [
            (MATRIX_BT709, ColorMatrix::Bt709),
            (MATRIX_BT470BG, ColorMatrix::Bt601),
            (MATRIX_SMPTE170M, ColorMatrix::Bt601),
            (MATRIX_FCC, ColorMatrix::Fcc),
            (MATRIX_SMPTE240M, ColorMatrix::Smpte240m),
        ];
        for (code, matrix) in cases {
            let source = facts(code, 1, 1);
            let plan = color_plan(&source);
            let spec = YuvSpec::new(matrix, ColorRange::Limited, ChromaSiting::Left);
            assert_eq!(
                plan,
                Ok(ColorPlan {
                    decode: spec,
                    encode: spec,
                    tags: source,
                })
            );
        }
    }

    #[test]
    fn an_unspecified_matrix_converts_as_bt601_and_stays_untagged() {
        let source = facts(MATRIX_UNSPECIFIED, 1, 5);
        let spec = YuvSpec::new(ColorMatrix::Bt601, ColorRange::Limited, ChromaSiting::Left);
        assert_eq!(
            color_plan(&source),
            Ok(ColorPlan {
                decode: spec,
                encode: spec,
                tags: source,
            })
        );
    }

    #[test]
    fn full_range_reaches_both_specs_and_the_tags() {
        let source = ColorFacts {
            full_range: true,
            ..facts(MATRIX_BT709, 1, 1)
        };
        let plan = color_plan(&source);
        assert_eq!(plan.map(|plan| plan.decode.range), Ok(ColorRange::Full));
        assert_eq!(plan.map(|plan| plan.encode.range), Ok(ColorRange::Full));
        assert_eq!(plan.map(|plan| plan.tags.full_range), Ok(true));
    }

    #[test]
    fn other_matrices_are_errors_that_name_the_code() {
        for code in [0, 3, 8, 11, 14, 255] {
            let error = color_plan(&facts(code, 1, 1));
            assert_eq!(error, Err(ColorError::UnsupportedMatrix { code }));
            let message = error.map_or_else(|error| error.to_string(), |_| String::new());
            assert!(
                message.contains(&format!("MatrixCoefficients {code}")),
                "{message}"
            );
        }
    }

    #[test]
    fn hdr_is_refused_before_the_matrix_check() {
        assert_eq!(
            color_plan(&facts(9, 1, 1)),
            Err(ColorError::Hdr {
                field: "MatrixCoefficients",
                code: 9,
            })
        );
        assert_eq!(
            color_plan(&facts(MATRIX_BT709, 16, 1)),
            Err(ColorError::Hdr {
                field: "TransferCharacteristics",
                code: 16,
            })
        );
    }

    #[test]
    fn chroma_location_passes_through() {
        let source = ColorFacts {
            chroma_location: Some(ChromaLocation::Top),
            ..facts(MATRIX_BT709, 1, 1)
        };
        let plan = color_plan(&source);
        assert_eq!(
            plan.map(|plan| plan.decode.siting),
            Ok(ChromaSiting::Center)
        );
        assert_eq!(
            plan.map(|plan| plan.encode.siting),
            Ok(ChromaSiting::Center)
        );
        assert_eq!(
            plan.map(|plan| plan.tags.chroma_location),
            Ok(Some(ChromaLocation::Top))
        );
    }

    #[test]
    fn reserved_transfer_and_primaries_stay_untagged() {
        for code in [0, 3] {
            let plan = color_plan(&facts(MATRIX_BT709, code, code));
            assert_eq!(
                plan.map(|plan| plan.tags.transfer_characteristics),
                Ok(TRANSFER_UNSPECIFIED)
            );
            assert_eq!(
                plan.map(|plan| plan.tags.colour_primaries),
                Ok(PRIMARIES_UNSPECIFIED)
            );
        }
    }

    #[cfg(any(feature = "mp4", feature = "mkv"))]
    mod merge {
        use containr::ContainerColour;

        use super::*;

        const TAGS: ContainerColour = ContainerColour {
            primaries: Some(5),
            transfer: Some(6),
            matrix: Some(7),
            full_range: Some(true),
        };

        #[test]
        fn no_container_tags_leave_the_bitstream_unchanged() {
            let bitstream = facts(2, 2, 2);
            assert_eq!(merge_container(bitstream, None), bitstream);
            assert_eq!(
                merge_container(bitstream, Some(ContainerColour::default())),
                bitstream
            );
        }

        #[test]
        fn a_specified_bitstream_value_wins() {
            let merged = merge_container(facts(1, 1, 1), Some(TAGS));
            assert_eq!(merged.matrix_coefficients, 1);
            assert_eq!(merged.transfer_characteristics, 1);
            assert_eq!(merged.colour_primaries, 1);
        }

        #[test]
        fn an_unspecified_bitstream_value_takes_the_container_value() {
            let merged = merge_container(facts(2, 2, 2), Some(TAGS));
            assert_eq!(merged.matrix_coefficients, 7);
            assert_eq!(merged.transfer_characteristics, 6);
            assert_eq!(merged.colour_primaries, 5);
        }

        #[test]
        fn each_field_fills_on_its_own() {
            let merged = merge_container(facts(1, 2, 2), Some(TAGS));
            assert_eq!(merged.matrix_coefficients, 1);
            assert_eq!(merged.transfer_characteristics, 6);
            let partial = ContainerColour {
                matrix: Some(1),
                ..ContainerColour::default()
            };
            let merged = merge_container(facts(2, 2, 2), Some(partial));
            assert_eq!(merged.matrix_coefficients, 1);
            assert_eq!(merged.transfer_characteristics, 2);
            assert_eq!(merged.colour_primaries, 2);
        }

        #[test]
        fn a_container_value_of_two_changes_nothing() {
            let unspecified = ContainerColour {
                primaries: Some(2),
                transfer: Some(2),
                matrix: Some(2),
                full_range: None,
            };
            let bitstream = facts(2, 2, 2);
            assert_eq!(merge_container(bitstream, Some(unspecified)), bitstream);
        }

        #[test]
        fn full_range_from_either_side_gives_full() {
            let range = |bitstream: bool, container: Option<bool>| {
                let source = ColorFacts {
                    full_range: bitstream,
                    ..facts(1, 1, 1)
                };
                let tags = ContainerColour {
                    full_range: container,
                    ..ContainerColour::default()
                };
                merge_container(source, Some(tags)).full_range
            };
            assert!(range(true, Some(false)));
            assert!(range(true, None));
            assert!(range(false, Some(true)));
            assert!(!range(false, Some(false)));
            assert!(!range(false, None));
        }

        #[test]
        fn the_chroma_location_stays() {
            let source = ColorFacts {
                chroma_location: Some(ChromaLocation::Center),
                ..facts(2, 2, 2)
            };
            assert_eq!(
                merge_container(source, Some(TAGS)).chroma_location,
                Some(ChromaLocation::Center)
            );
        }
    }
}
