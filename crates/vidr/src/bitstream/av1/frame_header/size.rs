//! `frame_size()`, `render_size()`, `frame_size_with_refs()`, `superres_params()`, and
//! `compute_image_size()` (AV1 specification 5.9.5 to 5.9.9).

use super::ref_table::RefTable;
use super::types::{FrameHeader, REFS_PER_FRAME};
use crate::bitstream::av1::seq_header::SequenceHeader;
use crate::bitstream::bits::BitReader;
use crate::error::BitstreamError;

/// `SUPERRES_NUM`.
const SUPERRES_NUM: u32 = 8;
/// `SUPERRES_DENOM_MIN`.
const SUPERRES_DENOM_MIN: u32 = 9;
/// `SUPERRES_DENOM_BITS`.
const SUPERRES_DENOM_BITS: u32 = 3;

fn superres_params(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    header: &mut FrameHeader,
) -> Result<(), BitstreamError> {
    header.use_superres = sequence.enable_superres && reader.bit("use_superres")?;
    let denom = if header.use_superres {
        header.coded_denom = reader.u8(SUPERRES_DENOM_BITS, "coded_denom")?;
        u32::from(header.coded_denom) + SUPERRES_DENOM_MIN
    } else {
        SUPERRES_NUM
    };
    header.upscaled_width = header.frame_width;
    header.frame_width = (header.upscaled_width * SUPERRES_NUM + denom / 2) / denom;
    Ok(())
}

const fn compute_image_size(header: &mut FrameHeader) {
    header.mi_cols = 2 * ((header.frame_width + 7) >> 3);
    header.mi_rows = 2 * ((header.frame_height + 7) >> 3);
}

/// Reads `frame_size()`.
pub(super) fn frame_size(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    header: &mut FrameHeader,
) -> Result<(), BitstreamError> {
    let (max_width, max_height) = (
        u32::from(sequence.max_frame_width_minus_1) + 1,
        u32::from(sequence.max_frame_height_minus_1) + 1,
    );
    if header.frame_size_override_flag {
        let width_bits = u32::from(sequence.frame_width_bits_minus_1) + 1;
        let height_bits = u32::from(sequence.frame_height_bits_minus_1) + 1;
        header.frame_width = reader.bits(width_bits, "frame_width_minus_1")? as u32 + 1;
        header.frame_height = reader.bits(height_bits, "frame_height_minus_1")? as u32 + 1;
        header.frame_size_coded = true;
        if header.frame_width > max_width {
            return Err(reader.invalid(
                "frame_width_minus_1",
                u64::from(header.frame_width - 1),
                "at most max_frame_width_minus_1",
            ));
        }
        if header.frame_height > max_height {
            return Err(reader.invalid(
                "frame_height_minus_1",
                u64::from(header.frame_height - 1),
                "at most max_frame_height_minus_1",
            ));
        }
    } else {
        header.frame_width = max_width;
        header.frame_height = max_height;
    }
    superres_params(reader, sequence, header)?;
    compute_image_size(header);
    Ok(())
}

/// Reads `render_size()`.
pub(super) fn render_size(
    reader: &mut BitReader<'_>,
    header: &mut FrameHeader,
) -> Result<(), BitstreamError> {
    header.render_and_frame_size_different = reader.bit("render_and_frame_size_different")?;
    if header.render_and_frame_size_different {
        header.render_width = reader.bits(16, "render_width_minus_1")? as u32 + 1;
        header.render_height = reader.bits(16, "render_height_minus_1")? as u32 + 1;
    } else {
        header.render_width = header.upscaled_width;
        header.render_height = header.frame_height;
    }
    Ok(())
}

/// Reads `frame_size_with_refs()`.
pub(super) fn frame_size_with_refs(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    refs: &RefTable,
    header: &mut FrameHeader,
) -> Result<(), BitstreamError> {
    for index in 0..REFS_PER_FRAME {
        let found = reader.bit("found_ref")?;
        header.found_ref = Some((index as u8 + 1, found));
        if found {
            let name = header.ref_frame_idx[index];
            let saved = refs.get(name);
            if !saved.valid {
                return Err(reader.invalid(
                    "found_ref",
                    u64::from(name),
                    "a reference name that holds a frame",
                ));
            }
            header.frame_width = saved.upscaled_width;
            header.frame_height = saved.frame_height;
            header.render_width = saved.render_width;
            header.render_height = saved.render_height;
            superres_params(reader, sequence, header)?;
            compute_image_size(header);
            return Ok(());
        }
    }
    frame_size(reader, sequence, header)?;
    render_size(reader, header)
}
