//! `film_grain_params()` (AV1 specification 5.9.30), with `load_grain_params()`.

use super::ref_table::RefTable;
use super::types::{FilmGrain, FrameHeader, INTER_FRAME};
use crate::bitstream::av1::seq_header::SequenceHeader;
use crate::bitstream::bits::BitReader;
use crate::error::BitstreamError;

/// Reads `film_grain_params()` into `header.film_grain`.
///
/// # Errors
///
/// [`BitstreamError`] for a payload that ends early or a count past its array. The same goes for a
/// `film_grain_params_ref_idx` that names no reference of the frame.
pub(super) fn film_grain_params(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    refs: &RefTable,
    header: &mut FrameHeader,
) -> Result<(), BitstreamError> {
    header.film_grain = FilmGrain::default();
    if !sequence.film_grain_params_present || (!header.show_frame && !header.showable_frame) {
        return Ok(());
    }
    if !reader.bit("apply_grain")? {
        return Ok(());
    }
    let grain_seed = reader.bits(16, "grain_seed")? as u16;
    let update_grain = header.frame_type != INTER_FRAME || reader.bit("update_grain")?;
    if !update_grain {
        let name = reader.u8(3, "film_grain_params_ref_idx")?;
        if !header.ref_frame_idx.contains(&name) {
            return Err(reader.invalid(
                "film_grain_params_ref_idx",
                name.into(),
                "a value of ref_frame_idx",
            ));
        }
        let saved = refs.get(name);
        if !saved.valid {
            return Err(reader.invalid(
                "film_grain_params_ref_idx",
                name.into(),
                "a reference name that holds a frame",
            ));
        }
        header.film_grain = FilmGrain {
            grain_seed,
            update_grain: false,
            film_grain_params_ref_idx: name,
            ..saved.film_grain
        };
        return Ok(());
    }
    let mono = sequence.color.mono_chrome;
    let subsampled = sequence.color.subsampling_x && sequence.color.subsampling_y;
    let mut grain = FilmGrain {
        apply_grain: true,
        grain_seed,
        update_grain: true,
        ..FilmGrain::default()
    };
    grain.num_y_points = read_points(
        reader,
        14,
        "num_y_points",
        "point_y_value",
        "point_y_scaling",
        &mut grain.point_y_value,
        &mut grain.point_y_scaling,
    )?;
    grain.chroma_scaling_from_luma = !mono && reader.bit("chroma_scaling_from_luma")?;
    let no_chroma_points =
        mono || grain.chroma_scaling_from_luma || (subsampled && grain.num_y_points == 0);
    if !no_chroma_points {
        grain.num_cb_points = read_points(
            reader,
            10,
            "num_cb_points",
            "point_cb_value",
            "point_cb_scaling",
            &mut grain.point_cb_value,
            &mut grain.point_cb_scaling,
        )?;
        grain.num_cr_points = read_points(
            reader,
            10,
            "num_cr_points",
            "point_cr_value",
            "point_cr_scaling",
            &mut grain.point_cr_value,
            &mut grain.point_cr_scaling,
        )?;
    }
    grain.grain_scaling_minus_8 = reader.u8(2, "grain_scaling_minus_8")?;
    grain.ar_coeff_lag = reader.u8(2, "ar_coeff_lag")?;
    read_ar_coeffs(reader, &mut grain)?;
    grain.ar_coeff_shift_minus_6 = reader.u8(2, "ar_coeff_shift_minus_6")?;
    grain.grain_scale_shift = reader.u8(2, "grain_scale_shift")?;
    if grain.num_cb_points > 0 {
        grain.cb_mult = reader.u8(8, "cb_mult")?;
        grain.cb_luma_mult = reader.u8(8, "cb_luma_mult")?;
        grain.cb_offset = reader.bits(9, "cb_offset")? as u16;
    }
    if grain.num_cr_points > 0 {
        grain.cr_mult = reader.u8(8, "cr_mult")?;
        grain.cr_luma_mult = reader.u8(8, "cr_luma_mult")?;
        grain.cr_offset = reader.bits(9, "cr_offset")? as u16;
    }
    grain.overlap_flag = reader.bit("overlap_flag")?;
    grain.clip_to_restricted_range = reader.bit("clip_to_restricted_range")?;
    header.film_grain = grain;
    Ok(())
}

/// Reads a point count and its pairs.
fn read_points(
    reader: &mut BitReader<'_>,
    limit: u8,
    count_field: &'static str,
    value_field: &'static str,
    scaling_field: &'static str,
    values: &mut [u8],
    scalings: &mut [u8],
) -> Result<u8, BitstreamError> {
    let count = reader.u8(4, count_field)?;
    if count > limit {
        return Err(reader.invalid(count_field, count.into(), "at most the size of its array"));
    }
    for index in 0..usize::from(count) {
        values[index] = reader.u8(8, value_field)?;
        scalings[index] = reader.u8(8, scaling_field)?;
    }
    Ok(count)
}

fn read_ar_coeffs(reader: &mut BitReader<'_>, grain: &mut FilmGrain) -> Result<(), BitstreamError> {
    let lag = usize::from(grain.ar_coeff_lag);
    let num_pos_luma = 2 * lag * (lag + 1);
    let num_pos_chroma = if grain.num_y_points > 0 {
        for coefficient in &mut grain.ar_coeffs_y_plus_128[..num_pos_luma] {
            *coefficient = reader.u8(8, "ar_coeffs_y_plus_128")?;
        }
        num_pos_luma + 1
    } else {
        num_pos_luma
    };
    if grain.chroma_scaling_from_luma || grain.num_cb_points > 0 {
        for coefficient in &mut grain.ar_coeffs_cb_plus_128[..num_pos_chroma] {
            *coefficient = reader.u8(8, "ar_coeffs_cb_plus_128")?;
        }
    }
    if grain.chroma_scaling_from_luma || grain.num_cr_points > 0 {
        for coefficient in &mut grain.ar_coeffs_cr_plus_128[..num_pos_chroma] {
            *coefficient = reader.u8(8, "ar_coeffs_cr_plus_128")?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::av1::frame_header::SWITCH_FRAME;
    use crate::bitstream::av1::seq_header::ColorConfig;
    use crate::bitstream::test_bits::BitWriter;

    fn grain_sequence() -> SequenceHeader {
        SequenceHeader {
            film_grain_params_present: true,
            color: ColorConfig {
                subsampling_x: true,
                subsampling_y: true,
                ..ColorConfig::default()
            },
            ..SequenceHeader::default()
        }
    }

    fn shown_inter() -> FrameHeader {
        FrameHeader {
            frame_type: INTER_FRAME,
            show_frame: true,
            ref_frame_idx: [1, 1, 1, 1, 2, 2, 2],
            ..FrameHeader::default()
        }
    }

    /// `update_grain` 0 loads every field from the name, and only the seed stays coded.
    #[test]
    fn a_load_takes_the_saved_parameters_with_a_new_seed() {
        let mut refs = RefTable::default();
        refs.names[2].valid = true;
        refs.names[2].film_grain = FilmGrain {
            apply_grain: true,
            grain_seed: 11,
            update_grain: true,
            num_y_points: 3,
            cb_mult: 90,
            overlap_flag: true,
            ..FilmGrain::default()
        };
        let mut w = BitWriter::new();
        w.flag(true).u(16, 777).flag(false).u(3, 2);
        let bytes = w.bytes();
        let mut reader = BitReader::new(&bytes, "test");
        let mut header = shown_inter();
        film_grain_params(&mut reader, &grain_sequence(), &refs, &mut header).expect("parses");
        let grain = header.film_grain;
        assert!(grain.apply_grain);
        assert_eq!(grain.grain_seed, 777);
        assert!(!grain.update_grain);
        assert_eq!(grain.film_grain_params_ref_idx, 2);
        assert_eq!(grain.num_y_points, 3);
        assert_eq!(grain.cb_mult, 90);
        assert!(grain.overlap_flag);
    }

    #[test]
    fn a_load_from_a_name_the_frame_does_not_list_is_refused_by_name() {
        let mut refs = RefTable::default();
        refs.names[5].valid = true;
        let mut w = BitWriter::new();
        w.flag(true).u(16, 1).flag(false).u(3, 5);
        let bytes = w.bytes();
        let mut reader = BitReader::new(&bytes, "test");
        let mut header = shown_inter();
        let error = film_grain_params(&mut reader, &grain_sequence(), &refs, &mut header)
            .expect_err("refused");
        assert!(
            error.to_string().contains("film_grain_params_ref_idx"),
            "{error}"
        );
    }

    /// A key frame reads no `update_grain`. A count of 15 luma points is past the array.
    #[test]
    fn a_luma_count_past_the_array_is_refused_by_name() {
        let mut w = BitWriter::new();
        w.flag(true).u(16, 1).u(4, 15);
        let bytes = w.bytes();
        let mut reader = BitReader::new(&bytes, "test");
        let mut header = FrameHeader {
            show_frame: true,
            ..FrameHeader::default()
        };
        let error = film_grain_params(
            &mut reader,
            &grain_sequence(),
            &RefTable::default(),
            &mut header,
        )
        .expect_err("refused");
        assert!(error.to_string().contains("num_y_points"), "{error}");
    }

    /// Every part of the syntax, on a switch frame, which reads no `update_grain`.
    #[test]
    fn a_full_set_of_parameters_keeps_each_field() {
        let mut w = BitWriter::new();
        w.flag(true).u(16, 500);
        w.u(4, 2).u(8, 10).u(8, 20).u(8, 200).u(8, 40);
        w.flag(false);
        w.u(4, 1).u(8, 30).u(8, 31);
        w.u(4, 1).u(8, 50).u(8, 51);
        w.u(2, 3).u(2, 1);
        // Lag 1: 4 luma positions, 5 chroma positions.
        for value in 0..4 {
            w.u(8, 128 + value);
        }
        for value in 0..5 {
            w.u(8, 100 + value);
        }
        for value in 0..5 {
            w.u(8, 60 + value);
        }
        w.u(2, 2).u(2, 1);
        w.u(8, 128).u(8, 192).u(9, 256);
        w.u(8, 127).u(8, 191).u(9, 255);
        w.flag(true).flag(true);
        let bytes = w.bytes();
        let mut reader = BitReader::new(&bytes, "test");
        let mut header = FrameHeader {
            frame_type: SWITCH_FRAME,
            show_frame: true,
            ..FrameHeader::default()
        };
        film_grain_params(
            &mut reader,
            &grain_sequence(),
            &RefTable::default(),
            &mut header,
        )
        .expect("parses");
        let grain = header.film_grain;
        assert_eq!(grain.num_y_points, 2);
        assert_eq!(grain.point_y_scaling[1], 40);
        assert_eq!(grain.point_cb_value[0], 30);
        assert_eq!(grain.point_cr_scaling[0], 51);
        assert_eq!(grain.ar_coeff_lag, 1);
        assert_eq!(grain.ar_coeffs_y_plus_128[3], 131);
        assert_eq!(grain.ar_coeffs_cb_plus_128[4], 104);
        assert_eq!(grain.ar_coeffs_cr_plus_128[4], 64);
        assert_eq!(grain.ar_coeff_shift_minus_6, 2);
        assert_eq!(grain.cb_offset, 256);
        assert_eq!(grain.cr_offset, 255);
        assert!(grain.overlap_flag && grain.clip_to_restricted_range);
    }

    #[test]
    fn a_hidden_frame_that_never_shows_holds_no_grain() {
        let mut reader = BitReader::new(&[], "test");
        let mut header = FrameHeader::default();
        film_grain_params(
            &mut reader,
            &grain_sequence(),
            &RefTable::default(),
            &mut header,
        )
        .expect("parses");
        assert!(!header.film_grain.apply_grain);
    }
}
