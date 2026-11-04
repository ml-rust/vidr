//! `tile_info()` (AV1 specification 5.9.15), and the `ns(n)` code it reads.

use super::types::{FrameHeader, TileInfo};
use crate::bitstream::av1::seq_header::SequenceHeader;
use crate::bitstream::bits::BitReader;
use crate::error::BitstreamError;

/// `MAX_TILE_WIDTH`.
const MAX_TILE_WIDTH: u32 = 4096;
/// `MAX_TILE_AREA`.
const MAX_TILE_AREA: u32 = 4096 * 2304;
/// `MAX_TILE_COLS`.
const MAX_TILE_COLS: u32 = 64;
/// `MAX_TILE_ROWS`.
const MAX_TILE_ROWS: u32 = 64;

/// `tile_log2()`: the smallest `k` with `blk_size << k` at least `target`.
const fn tile_log2(blk_size: u32, target: u32) -> u32 {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

/// Reads an `ns(n)` value (4.10.7) for an `n` of 1 or more.
pub(super) fn read_ns(
    reader: &mut BitReader<'_>,
    n: u32,
    field: &'static str,
) -> Result<u32, BitstreamError> {
    let w = 32 - n.leading_zeros();
    let m = (1u32 << w) - n;
    let v = reader.bits(w - 1, field)? as u32;
    if v < m {
        return Ok(v);
    }
    let extra = u32::from(reader.bit(field)?);
    Ok((v << 1) - m + extra)
}

/// Reads the unary increments of one uniform log2 value.
fn read_increments(
    reader: &mut BitReader<'_>,
    min: u32,
    max: u32,
    field: &'static str,
) -> Result<u32, BitstreamError> {
    let mut value = min;
    while value < max && reader.bit(field)? {
        value += 1;
    }
    Ok(value)
}

/// `MiColStarts` or `MiRowStarts` and the sizes of a uniform spacing.
fn uniform_starts(count_sb: u32, log2: u32, sb_shift: u32, mi: u32) -> (Vec<u16>, Vec<u16>) {
    let size_sb = (count_sb + (1 << log2) - 1) >> log2;
    let mut starts = Vec::new();
    let mut sizes = Vec::new();
    let mut start_sb = 0;
    while start_sb < count_sb {
        starts.push((start_sb << sb_shift) as u16);
        sizes.push((size_sb.min(count_sb - start_sb) - 1) as u16);
        start_sb += size_sb;
    }
    starts.push(mi as u16);
    (starts, sizes)
}

/// Reads `tile_info()` for the frame size `header` holds.
pub(super) fn tile_info(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    header: &FrameHeader,
) -> Result<TileInfo, BitstreamError> {
    let (mi_cols, mi_rows) = (header.mi_cols, header.mi_rows);
    let (sb_cols, sb_rows, sb_shift) = if sequence.use_128x128_superblock {
        ((mi_cols + 31) >> 5, (mi_rows + 31) >> 5, 5)
    } else {
        ((mi_cols + 15) >> 4, (mi_rows + 15) >> 4, 4)
    };
    let sb_size = sb_shift + 2;
    let max_tile_width_sb = MAX_TILE_WIDTH >> sb_size;
    let max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size);
    let min_log2_cols = tile_log2(max_tile_width_sb, sb_cols);
    let max_log2_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS));
    let max_log2_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS));
    let min_log2_tiles = min_log2_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));
    let mut info = TileInfo {
        uniform_tile_spacing_flag: reader.bit("uniform_tile_spacing_flag")?,
        ..TileInfo::default()
    };
    if info.uniform_tile_spacing_flag {
        let cols_log2 = read_increments(reader, min_log2_cols, max_log2_cols, "tile_cols_log2")?;
        (info.mi_col_starts, info.width_in_sbs_minus_1) =
            uniform_starts(sb_cols, cols_log2, sb_shift, mi_cols);
        let min_log2_rows = min_log2_tiles.saturating_sub(cols_log2);
        let rows_log2 = read_increments(reader, min_log2_rows, max_log2_rows, "tile_rows_log2")?;
        (info.mi_row_starts, info.height_in_sbs_minus_1) =
            uniform_starts(sb_rows, rows_log2, sb_shift, mi_rows);
        info.tile_cols_log2 = cols_log2 as u8;
        info.tile_rows_log2 = rows_log2 as u8;
    } else {
        explicit_spacing(
            reader,
            &mut info,
            (sb_cols, sb_rows, sb_shift),
            (mi_cols, mi_rows),
            (max_tile_width_sb, min_log2_tiles),
        )?;
    }
    info.tile_cols = (info.mi_col_starts.len() - 1) as u16;
    info.tile_rows = (info.mi_row_starts.len() - 1) as u16;
    if info.tile_cols_log2 > 0 || info.tile_rows_log2 > 0 {
        let bits = u32::from(info.tile_cols_log2) + u32::from(info.tile_rows_log2);
        info.context_update_tile_id = reader.bits(bits, "context_update_tile_id")? as u16;
        info.tile_size_bytes_minus_1 = reader.u8(2, "tile_size_bytes_minus_1")?;
        if u32::from(info.context_update_tile_id) >= info.num_tiles() {
            return Err(reader.invalid(
                "context_update_tile_id",
                info.context_update_tile_id.into(),
                "less than TileCols * TileRows",
            ));
        }
    }
    Ok(info)
}

/// Reads the explicit sizes of a non-uniform spacing.
fn explicit_spacing(
    reader: &mut BitReader<'_>,
    info: &mut TileInfo,
    (sb_cols, sb_rows, sb_shift): (u32, u32, u32),
    (mi_cols, mi_rows): (u32, u32),
    (max_tile_width_sb, min_log2_tiles): (u32, u32),
) -> Result<(), BitstreamError> {
    let mut widest = 0;
    let mut start_sb = 0;
    while start_sb < sb_cols {
        if info.mi_col_starts.len() as u32 >= MAX_TILE_COLS {
            return Err(reader.invalid(
                "width_in_sbs_minus_1",
                u64::from(start_sb),
                "sizes that give at most 64 tile columns",
            ));
        }
        info.mi_col_starts.push((start_sb << sb_shift) as u16);
        let max_width = (sb_cols - start_sb).min(max_tile_width_sb);
        let width = read_ns(reader, max_width, "width_in_sbs_minus_1")?;
        info.width_in_sbs_minus_1.push(width as u16);
        widest = widest.max(width + 1);
        start_sb += width + 1;
    }
    info.mi_col_starts.push(mi_cols as u16);
    info.tile_cols_log2 = tile_log2(1, info.width_in_sbs_minus_1.len() as u32) as u8;
    let area = sb_rows * sb_cols;
    let max_area = if min_log2_tiles > 0 {
        area >> (min_log2_tiles + 1)
    } else {
        area
    };
    let max_height_sb = (max_area / widest.max(1)).max(1);
    start_sb = 0;
    while start_sb < sb_rows {
        if info.mi_row_starts.len() as u32 >= MAX_TILE_ROWS {
            return Err(reader.invalid(
                "height_in_sbs_minus_1",
                u64::from(start_sb),
                "sizes that give at most 64 tile rows",
            ));
        }
        info.mi_row_starts.push((start_sb << sb_shift) as u16);
        let max_height = (sb_rows - start_sb).min(max_height_sb);
        let height = read_ns(reader, max_height, "height_in_sbs_minus_1")?;
        info.height_in_sbs_minus_1.push(height as u16);
        start_sb += height + 1;
    }
    info.mi_row_starts.push(mi_rows as u16);
    info.tile_rows_log2 = tile_log2(1, info.height_in_sbs_minus_1.len() as u32) as u8;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::test_bits::BitWriter;

    #[test]
    fn ns_reads_the_short_and_the_long_forms() {
        // n = 5: w = 3, m = 3. Values 0, 1, 2 take 2 bits. Values 3, 4 take 3 bits.
        let mut w = BitWriter::new();
        w.u(2, 0).u(2, 2).u(3, 0b110).u(3, 0b111);
        let bytes = w.bytes();
        let mut reader = BitReader::new(&bytes, "test");
        assert_eq!(read_ns(&mut reader, 5, "v"), Ok(0));
        assert_eq!(read_ns(&mut reader, 5, "v"), Ok(2));
        assert_eq!(read_ns(&mut reader, 5, "v"), Ok(3));
        assert_eq!(read_ns(&mut reader, 5, "v"), Ok(4));
        // n = 1 reads nothing.
        assert_eq!(read_ns(&mut reader, 1, "v"), Ok(0));
    }

    #[test]
    fn tile_log2_holds_its_definition() {
        assert_eq!(tile_log2(1, 1), 0);
        assert_eq!(tile_log2(1, 2), 1);
        assert_eq!(tile_log2(1, 3), 2);
        assert_eq!(tile_log2(64, 65), 1);
    }

    fn sized(width: u32, height: u32) -> FrameHeader {
        FrameHeader {
            mi_cols: 2 * ((width + 7) >> 3),
            mi_rows: 2 * ((height + 7) >> 3),
            ..FrameHeader::default()
        }
    }

    /// 640x360 at 64x64 superblocks: 10 by 6 superblocks, so one increment gives two columns of 5
    /// superblocks. MiCols is 160, and MiRows is 90.
    #[test]
    fn a_uniform_spacing_gives_starts_in_mode_info_units() {
        let mut w = BitWriter::new();
        // Uniform, one column increment then stop, one row increment then stop, tile id, size.
        w.flag(true).flag(true).flag(false).flag(true).flag(false);
        w.u(2, 3).u(2, 1);
        let bytes = w.bytes();
        let mut reader = BitReader::new(&bytes, "test");
        let sequence = SequenceHeader::default();
        let info = tile_info(&mut reader, &sequence, &sized(640, 360)).expect("parses");
        assert_eq!((info.tile_cols, info.tile_rows), (2, 2));
        assert_eq!(info.mi_col_starts, [0, 80, 160]);
        assert_eq!(info.mi_row_starts, [0, 48, 90]);
        assert_eq!(info.width_in_sbs_minus_1, [4, 4]);
        assert_eq!(info.height_in_sbs_minus_1, [2, 2]);
        assert_eq!(info.context_update_tile_id, 3);
        assert_eq!(info.tile_size_bytes_minus_1, 1);
    }

    /// 10 superblocks across: tile widths 3 and 7. `ns(10)` reads 3 bits below 6 and 4 bits from
    /// 6 up, and the second width stops at the end.
    #[test]
    fn an_explicit_spacing_keeps_each_size() {
        let mut w = BitWriter::new();
        w.flag(false);
        // Width 3: value 2 in `ns(10)`. m = 6, so 3 bits.
        w.u(3, 2);
        // Width 7: value 6 in `ns(7)`. w = 3, m = 1, v = 3 (0b11) + extra 1 gives 6.
        w.u(2, 0b11).u(1, 1);
        // Rows: 6 superblocks, one row of all 6. Value 5 in `ns(6)`. w = 3, m = 2, v = 3, extra 1.
        w.u(2, 0b11).u(1, 1);
        // Two columns: 1 bit of tile id, then 2 bits of size.
        w.u(1, 1).u(2, 0);
        let bytes = w.bytes();
        let mut reader = BitReader::new(&bytes, "test");
        let sequence = SequenceHeader::default();
        let info = tile_info(&mut reader, &sequence, &sized(640, 360)).expect("parses");
        assert_eq!(info.mi_col_starts, [0, 48, 160]);
        assert_eq!(info.width_in_sbs_minus_1, [2, 6]);
        assert_eq!(info.mi_row_starts, [0, 90]);
        assert_eq!(info.height_in_sbs_minus_1, [5]);
        assert_eq!(info.tile_cols_log2, 1);
        assert_eq!(info.tile_rows_log2, 0);
        assert_eq!(info.context_update_tile_id, 1);
    }

    #[test]
    fn a_tile_id_past_the_last_tile_is_refused_by_name() {
        // 192x64 at 64x64: 3 by 1 superblocks. Two increments give log2 2 and tiles of 1. So the
        // frame holds 3 columns and 2 id bits, and id 3 is past them.
        let mut w = BitWriter::new();
        w.flag(true).flag(true).flag(true);
        w.u(2, 3).u(2, 0);
        let bytes = w.bytes();
        let mut reader = BitReader::new(&bytes, "test");
        let sequence = SequenceHeader::default();
        let error = tile_info(&mut reader, &sequence, &sized(192, 64)).expect_err("refused");
        assert!(
            error.to_string().contains("context_update_tile_id"),
            "{error}"
        );
    }
}
