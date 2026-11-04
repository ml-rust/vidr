//! HEVC `scaling_list_data()` (ITU-T H.265 7.3.4) and the lists 7.4.5 derives.

use super::bits::BitReader;
use crate::error::BitstreamError;

/// Table 7-6 at `matrixId` 0 to 2.
const DEFAULT_INTRA: [u8; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 16, 17, 16, 17, 18, 17, 18, 18, 17, 18, 21, 19, 20,
    21, 20, 19, 21, 24, 22, 22, 24, 24, 22, 22, 24, 25, 25, 27, 30, 27, 25, 25, 29, 31, 35, 35, 31,
    29, 36, 41, 44, 41, 36, 47, 54, 54, 47, 65, 70, 65, 88, 88, 115,
];
/// Table 7-6 at `matrixId` 3 to 5.
const DEFAULT_INTER: [u8; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 20, 20, 20,
    20, 20, 20, 20, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 28, 28, 28, 28, 28,
    28, 33, 33, 33, 33, 33, 41, 41, 41, 41, 54, 54, 54, 71, 71, 91,
];
/// The inferred `scaling_list_dc_coef_minus8` of a default list (7.4.5).
const DEFAULT_DC_MINUS8: i16 = 8;

/// One `scaling_list_data()`.
///
/// Size 3 holds `matrixId` 0 and 3 only. (7-50) derives its other matrices for
/// `ChromaArrayType` 3 from size 2.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ScalingListData {
    /// Indexed by `sizeId` and `matrixId`.
    pub(crate) scaling_list_pred_mode_flag: [[bool; 6]; 4],
    /// Indexed by `sizeId` and `matrixId`.
    pub(crate) scaling_list_pred_matrix_id_delta: [[u8; 6]; 4],
    /// Indexed by `sizeId - 2` and `matrixId`, with the 7.4.5 inference.
    pub(crate) scaling_list_dc_coef_minus8: [[i16; 6]; 2],
    /// `ScalingList`, indexed by `sizeId`, `matrixId`, and `i`. Size 0 uses 16
    /// coefficients and the others use 64.
    pub(crate) scaling_list: [[[u8; 64]; 6]; 4],
}

/// The coefficients of `sizeId`.
fn coefficient_count(size_id: usize) -> usize {
    if size_id == 0 { 16 } else { 64 }
}

/// The default list of Tables 7-5 and 7-6.
fn default_list(size_id: usize, matrix_id: usize) -> [u8; 64] {
    if size_id == 0 {
        let mut list = [0u8; 64];
        list[..16].fill(16);
        list
    } else if matrix_id < 3 {
        DEFAULT_INTRA
    } else {
        DEFAULT_INTER
    }
}

impl ScalingListData {
    /// The lists that apply with `scaling_list_enabled_flag` 1 and no
    /// `scaling_list_data()` in the SPS or PPS (7.4.3.3.1).
    pub(crate) fn default_lists() -> Self {
        let mut data = Self {
            scaling_list_pred_mode_flag: [[false; 6]; 4],
            scaling_list_pred_matrix_id_delta: [[0; 6]; 4],
            scaling_list_dc_coef_minus8: [[DEFAULT_DC_MINUS8; 6]; 2],
            scaling_list: [[[0; 64]; 6]; 4],
        };
        for size_id in 0..4 {
            for matrix_id in 0..6 {
                data.scaling_list[size_id][matrix_id] = default_list(size_id, matrix_id);
            }
        }
        data
    }

    /// Reads `scaling_list_data()`. The matrix loop takes steps of 3 at size 3
    /// whatever the chroma format (7.3.4).
    pub(crate) fn parse(reader: &mut BitReader<'_>) -> Result<Self, BitstreamError> {
        let mut data = Self {
            scaling_list_pred_mode_flag: [[false; 6]; 4],
            scaling_list_pred_matrix_id_delta: [[0; 6]; 4],
            scaling_list_dc_coef_minus8: [[0; 6]; 2],
            scaling_list: [[[0; 64]; 6]; 4],
        };
        for size_id in 0..4usize {
            let step = if size_id == 3 { 3 } else { 1 };
            for matrix_id in (0..6usize).step_by(step) {
                if reader.bit("scaling_list_pred_mode_flag")? {
                    data.scaling_list_pred_mode_flag[size_id][matrix_id] = true;
                    data.read_explicit(reader, size_id, matrix_id)?;
                } else {
                    data.read_predicted(reader, size_id, matrix_id, step)?;
                }
            }
        }
        Ok(data)
    }

    fn read_predicted(
        &mut self,
        reader: &mut BitReader<'_>,
        size_id: usize,
        matrix_id: usize,
        step: usize,
    ) -> Result<(), BitstreamError> {
        let delta = reader.ue_max(
            "scaling_list_pred_matrix_id_delta",
            (matrix_id / step) as u32,
            "0 to matrixId, or 0 to matrixId / 3 at sizeId 3",
        )? as usize;
        self.scaling_list_pred_matrix_id_delta[size_id][matrix_id] = delta as u8;
        if delta == 0 {
            self.scaling_list[size_id][matrix_id] = default_list(size_id, matrix_id);
            if size_id > 1 {
                self.scaling_list_dc_coef_minus8[size_id - 2][matrix_id] = DEFAULT_DC_MINUS8;
            }
        } else {
            // (7-42) and (7-43).
            let reference = matrix_id - delta * step;
            self.scaling_list[size_id][matrix_id] = self.scaling_list[size_id][reference];
            if size_id > 1 {
                self.scaling_list_dc_coef_minus8[size_id - 2][matrix_id] =
                    self.scaling_list_dc_coef_minus8[size_id - 2][reference];
            }
        }
        Ok(())
    }

    fn read_explicit(
        &mut self,
        reader: &mut BitReader<'_>,
        size_id: usize,
        matrix_id: usize,
    ) -> Result<(), BitstreamError> {
        let mut next_coef = 8i64;
        if size_id > 1 {
            let dc = reader.se_range("scaling_list_dc_coef_minus8", -7, 247, "-7 to 247")?;
            self.scaling_list_dc_coef_minus8[size_id - 2][matrix_id] = dc as i16;
            next_coef = dc + 8;
        }
        for index in 0..coefficient_count(size_id) {
            let delta = reader.se_range("scaling_list_delta_coef", -128, 127, "-128 to 127")?;
            next_coef = (next_coef + delta + 256) % 256;
            if next_coef == 0 {
                return Err(reader.invalid("ScalingList", 0, "1 to 255"));
            }
            self.scaling_list[size_id][matrix_id][index] = next_coef as u8;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_bits::BitWriter;
    use super::*;

    fn parse(writer: &BitWriter) -> Result<ScalingListData, BitstreamError> {
        ScalingListData::parse(&mut BitReader::new(&writer.bytes(), "test"))
    }

    /// Writes the matrices from a start point with `pred_mode` 0 and delta 0. A
    /// test then states only the matrices it cares about.
    fn defaults_from(writer: &mut BitWriter, first_size: usize, first_matrix: usize) {
        for size_id in first_size..4 {
            let step = if size_id == 3 { 3 } else { 1 };
            for matrix_id in (0..6).step_by(step) {
                if size_id == first_size && matrix_id < first_matrix {
                    continue;
                }
                writer.flag(false).ue(0);
            }
        }
    }

    #[test]
    fn all_predicted_from_default_equals_the_default_lists() {
        let mut writer = BitWriter::new();
        defaults_from(&mut writer, 0, 0);
        let data = parse(&writer).unwrap_or_else(|error| panic!("{error}"));
        let mut expected = ScalingListData::default_lists();
        // Size 3 reads matrices 0 and 3 only.
        for matrix_id in [1, 2, 4, 5] {
            expected.scaling_list[3][matrix_id] = [0; 64];
            expected.scaling_list_dc_coef_minus8[1][matrix_id] = 0;
        }
        assert_eq!(data, expected);
        assert_eq!(data.scaling_list[1][0], DEFAULT_INTRA);
        assert_eq!(data.scaling_list[1][3], DEFAULT_INTER);
        assert_eq!(data.scaling_list[0][0][15], 16);
        assert_eq!(data.scaling_list_dc_coef_minus8[0][2], 8);
    }

    #[test]
    fn explicit_list_accumulates_and_wraps() {
        let mut writer = BitWriter::new();
        // Size 0, matrix 0: 8 + 2 = 10, then 10 - 3 = 7, then a wrap at 256.
        writer.flag(true).se(2).se(-3).se(127).se(127).se(1);
        for _ in 5..16 {
            writer.se(0);
        }
        defaults_from(&mut writer, 0, 1);
        let data = parse(&writer).unwrap_or_else(|error| panic!("{error}"));
        assert!(data.scaling_list_pred_mode_flag[0][0]);
        assert_eq!(data.scaling_list[0][0][..5], [10, 7, 134, 5, 6]);
        assert_eq!(data.scaling_list[0][0][15], 6);
    }

    #[test]
    fn explicit_dc_starts_the_list_at_size_2() {
        let mut writer = BitWriter::new();
        defaults_from_sizes_below(&mut writer, 2);
        // Size 2, matrix 0: dc 20 + 8 = 28. The first coefficient is 28 + 1.
        writer.flag(true).se(20).se(1);
        for _ in 1..64 {
            writer.se(0);
        }
        // Matrix 1 copies matrix 0 through delta 1, dc included.
        writer.flag(false).ue(1);
        defaults_from(&mut writer, 2, 2);
        let data = parse(&writer).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(data.scaling_list_dc_coef_minus8[0][0], 20);
        assert_eq!(data.scaling_list[2][0][0], 29);
        assert_eq!(data.scaling_list[2][0][63], 29);
        assert_eq!(data.scaling_list_pred_matrix_id_delta[2][1], 1);
        assert_eq!(data.scaling_list[2][1], data.scaling_list[2][0]);
        assert_eq!(data.scaling_list_dc_coef_minus8[0][1], 20);
    }

    /// Writes all sizes below `size_id` as defaults.
    fn defaults_from_sizes_below(writer: &mut BitWriter, size_id: usize) {
        for _ in 0..size_id * 6 {
            writer.flag(false).ue(0);
        }
    }

    #[test]
    fn size_3_steps_by_three_and_predicts_in_threes() {
        let mut writer = BitWriter::new();
        defaults_from_sizes_below(&mut writer, 3);
        writer.flag(true).se(0);
        for _ in 0..64 {
            writer.se(0);
        }
        // Matrix 3, delta 1: reference matrix 0.
        writer.flag(false).ue(1);
        let data = parse(&writer).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(data.scaling_list[3][3], data.scaling_list[3][0]);
        assert_eq!(data.scaling_list[3][3][0], 8);
    }

    #[test]
    fn out_of_range_values_are_refused() {
        // Matrix 0 can only predict from the default list.
        let mut writer = BitWriter::new();
        writer.flag(false).ue(1);
        assert!(matches!(
            parse(&writer),
            Err(BitstreamError::InvalidField {
                field: "scaling_list_pred_matrix_id_delta",
                value: 1,
                ..
            })
        ));
        // Size 3, matrix 3: delta above matrixId / 3.
        let mut writer = BitWriter::new();
        defaults_from_sizes_below(&mut writer, 3);
        writer.flag(false).ue(0).flag(false).ue(2);
        assert!(matches!(
            parse(&writer),
            Err(BitstreamError::InvalidField {
                field: "scaling_list_pred_matrix_id_delta",
                value: 2,
                ..
            })
        ));
        // A delta coefficient outside the range.
        let mut writer = BitWriter::new();
        writer.flag(true).se(128);
        assert!(matches!(
            parse(&writer),
            Err(BitstreamError::InvalidSignedField {
                field: "scaling_list_delta_coef",
                value: 128,
                ..
            })
        ));
        // A coefficient of 0.
        let mut writer = BitWriter::new();
        writer.flag(true).se(-8);
        assert!(matches!(
            parse(&writer),
            Err(BitstreamError::InvalidField {
                field: "ScalingList",
                ..
            })
        ));
        // A dc outside the range.
        let mut writer = BitWriter::new();
        defaults_from_sizes_below(&mut writer, 2);
        writer.flag(true).se(-8);
        assert!(matches!(
            parse(&writer),
            Err(BitstreamError::InvalidSignedField {
                field: "scaling_list_dc_coef_minus8",
                value: -8,
                ..
            })
        ));
    }

    #[test]
    fn truncation_at_every_byte_is_an_error() {
        let mut writer = BitWriter::new();
        writer.flag(true);
        for _ in 0..16 {
            writer.se(1);
        }
        defaults_from(&mut writer, 0, 1);
        let bytes = writer.bytes();
        for end in 0..bytes.len() - 1 {
            let result = ScalingListData::parse(&mut BitReader::new(&bytes[..end], "test"));
            assert!(result.is_err(), "cut at byte {end}");
        }
    }
}
