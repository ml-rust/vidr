//! `tile_group_obu()` (AV1 specification 5.11.1): where each tile's bytes sit.

use super::frame_header::TileInfo;
use crate::bitstream::bits::BitReader;
use crate::error::BitstreamError;

const STRUCTURE: &str = "AV1 tile group";

/// One tile's bytes, within the bytes of one frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TileRange {
    pub(crate) offset: u32,
    pub(crate) size: u32,
}

/// What one tile group holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TileGroupFacts {
    /// `tile_start_and_end_present_flag`.
    pub(crate) start_and_end_present: bool,
    /// `tg_start`.
    pub(crate) tg_start: u32,
    /// `tg_end`.
    pub(crate) tg_end: u32,
}

fn invalid(field: &'static str, value: u64, allowed: &'static str) -> BitstreamError {
    BitstreamError::InvalidField {
        structure: STRUCTURE,
        field,
        value,
        allowed,
    }
}

/// Parses one tile group, and adds each tile to `tiles`. `payload` starts at the tile group
/// syntax, and `base` is its offset within the bytes of the frame. `next_tile` is the `TileNum`
/// this group must start at.
///
/// # Errors
///
/// [`BitstreamError`] for each of these:
///
/// - a group that ends inside a tile or its size field
/// - a tile with no bytes
/// - a `tg_start` other than `next_tile`
pub(crate) fn parse_tile_group(
    payload: &[u8],
    base: u32,
    info: &TileInfo,
    next_tile: u32,
    tiles: &mut Vec<TileRange>,
) -> Result<TileGroupFacts, BitstreamError> {
    let num_tiles = info.num_tiles();
    let mut reader = BitReader::new(payload, STRUCTURE);
    let start_and_end_present = num_tiles > 1 && reader.bit("tile_start_and_end_present_flag")?;
    let (tg_start, tg_end) = if start_and_end_present {
        let bits = u32::from(info.tile_cols_log2) + u32::from(info.tile_rows_log2);
        (
            reader.bits(bits, "tg_start")? as u32,
            reader.bits(bits, "tg_end")? as u32,
        )
    } else {
        (0, num_tiles - 1)
    };
    if tg_start != next_tile {
        return Err(invalid(
            "tg_start",
            tg_start.into(),
            "the tile after the last one of the previous group",
        ));
    }
    if tg_end < tg_start || tg_end >= num_tiles {
        return Err(invalid(
            "tg_end",
            tg_end.into(),
            "from tg_start up to NumTiles - 1",
        ));
    }
    let size_bytes = usize::from(info.tile_size_bytes_minus_1) + 1;
    let mut position = reader.position().div_ceil(8) as usize;
    for tile in tg_start..=tg_end {
        let remaining = payload.len().saturating_sub(position);
        let size = if tile == tg_end {
            remaining
        } else {
            let field = payload
                .get(position..position + size_bytes)
                .ok_or_else(|| truncated("tile_size_minus_1", position, payload.len()))?;
            position += size_bytes;
            let minus_1 = field
                .iter()
                .rev()
                .fold(0_u64, |value, &byte| (value << 8) | u64::from(byte));
            (minus_1 as usize).saturating_add(1)
        };
        if size == 0 {
            return Err(truncated("tile data", position, payload.len()));
        }
        if size > payload.len().saturating_sub(position) {
            return Err(truncated("tile data", payload.len(), payload.len()));
        }
        tiles.push(TileRange {
            offset: base.saturating_add(position as u32),
            size: size as u32,
        });
        position += size;
    }
    Ok(TileGroupFacts {
        start_and_end_present,
        tg_start,
        tg_end,
    })
}

const fn truncated(field: &'static str, offset: usize, size: usize) -> BitstreamError {
    BitstreamError::Truncated {
        structure: STRUCTURE,
        field,
        offset,
        size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_by_two(size_bytes_minus_1: u8) -> TileInfo {
        TileInfo {
            tile_cols_log2: 1,
            tile_rows_log2: 1,
            tile_cols: 2,
            tile_rows: 2,
            tile_size_bytes_minus_1: size_bytes_minus_1,
            ..TileInfo::default()
        }
    }

    #[test]
    fn one_tile_takes_every_byte() {
        let info = TileInfo {
            tile_cols: 1,
            tile_rows: 1,
            ..TileInfo::default()
        };
        let mut tiles = Vec::new();
        let facts = parse_tile_group(&[1, 2, 3], 100, &info, 0, &mut tiles).expect("parses");
        assert_eq!(
            tiles,
            [TileRange {
                offset: 100,
                size: 3
            }]
        );
        assert_eq!((facts.tg_start, facts.tg_end), (0, 0));
    }

    /// A group of all four tiles, flag 0 in a byte with its alignment. Sizes are little-endian in
    /// 2 bytes: tile 0 holds 3 bytes, tile 1 holds 1, tile 2 holds 258, and tile 3 the rest.
    #[test]
    fn size_fields_are_little_endian_and_the_last_tile_takes_the_rest() {
        let mut data = vec![0x00];
        data.extend([2, 0]);
        data.extend([0xa0; 3]);
        data.extend([0, 0]);
        data.push(0xa1);
        data.extend([1, 1]);
        data.extend([0xa2; 258]);
        data.extend([0xa3; 5]);
        let mut tiles = Vec::new();
        parse_tile_group(&data, 10, &two_by_two(1), 0, &mut tiles).expect("parses");
        assert_eq!(
            tiles,
            [
                TileRange {
                    offset: 13,
                    size: 3
                },
                TileRange {
                    offset: 18,
                    size: 1
                },
                TileRange {
                    offset: 21,
                    size: 258
                },
                TileRange {
                    offset: 279,
                    size: 5
                },
            ]
        );
    }

    /// Flag 1, then 2 bits of start and end: tiles 2 to 3, as the second group of a frame.
    #[test]
    fn a_later_group_starts_where_the_last_one_stopped() {
        // 1 10 11 000: flag, start 2, end 3, alignment.
        let mut data = vec![0b1101_1000];
        data.extend([0, 0xaa, 0xbb]);
        let mut tiles = Vec::new();
        let facts = parse_tile_group(&data, 0, &two_by_two(0), 2, &mut tiles).expect("parses");
        assert!(facts.start_and_end_present);
        assert_eq!((facts.tg_start, facts.tg_end), (2, 3));
        assert_eq!(
            tiles,
            [
                TileRange { offset: 2, size: 1 },
                TileRange { offset: 3, size: 1 }
            ]
        );
    }

    #[test]
    fn a_wrong_start_or_end_is_refused_by_name() {
        let mut tiles = Vec::new();
        // Start 2 where tile 1 comes next.
        let error = parse_tile_group(&[0b1100_1000, 1, 2], 0, &two_by_two(0), 1, &mut tiles)
            .expect_err("refused");
        assert!(error.to_string().contains("tg_start"), "{error}");
        // End 1 before start 2.
        let error = parse_tile_group(&[0b1100_1000, 1, 2], 0, &two_by_two(0), 2, &mut tiles)
            .expect_err("refused");
        assert!(error.to_string().contains("tg_end"), "{error}");
    }

    #[test]
    fn a_truncation_at_every_byte_is_a_named_error() {
        let mut data = vec![0x00];
        data.extend([2, 0xa0, 0xa0, 0xa0, 0, 0xa1, 0, 0xa2, 0xa3]);
        for end in 0..data.len() {
            let mut tiles = Vec::new();
            let result = parse_tile_group(&data[..end], 0, &two_by_two(0), 0, &mut tiles);
            assert!(result.is_err(), "end {end}");
        }
    }
}
