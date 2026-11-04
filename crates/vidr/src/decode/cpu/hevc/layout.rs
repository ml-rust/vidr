//! CTB address conversion and tile membership (6.5.1), and z-scan addresses
//! (6.5.2) for one picture.

use super::error::HevcError;

/// `spread( v )` has bit `i` of `v` at bit `2 * i`, for `p` in (6-10). A CTB holds
/// at most 16 minimum blocks per side.
const SPREAD: [u16; 16] = [0, 1, 4, 5, 16, 17, 20, 21, 64, 65, 68, 69, 80, 81, 84, 85];

/// Address tables of one picture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PicLayout {
    width: u32,
    height: u32,
    ctb_log2: u32,
    min_tb_log2: u32,
    width_in_ctbs: u32,
    /// `CtbAddrRsToTs`.
    rs_to_ts: Vec<u32>,
    /// `CtbAddrTsToRs`.
    ts_to_rs: Vec<u32>,
    /// `TileId`, indexed by raster address.
    tile_id_rs: Vec<u32>,
}

/// The sizes [`PicLayout::new`] reads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayoutSizes<'a> {
    /// `pic_width_in_luma_samples`.
    pub(crate) width: u32,
    /// `pic_height_in_luma_samples`.
    pub(crate) height: u32,
    /// `CtbLog2SizeY`.
    pub(crate) ctb_log2: u32,
    /// `MinTbLog2SizeY`.
    pub(crate) min_tb_log2: u32,
    /// `colWidth`, in CTBs.
    pub(crate) column_widths: &'a [u32],
    /// `rowHeight`, in CTBs.
    pub(crate) row_heights: &'a [u32],
}

impl PicLayout {
    /// Builds the tables. Tile sizes that do not cover the picture are refused.
    ///
    /// # Errors
    ///
    /// - [`HevcError::InvalidSyntax`] for tile sizes that do not cover the picture.
    /// - [`HevcError::Allocation`] when a table does not fit in memory.
    pub(crate) fn new(sizes: &LayoutSizes<'_>) -> Result<Self, HevcError> {
        let width_in_ctbs = sizes.width.div_ceil(1 << sizes.ctb_log2);
        let height_in_ctbs = sizes.height.div_ceil(1 << sizes.ctb_log2);
        check_tile_total("colWidth", sizes.column_widths, width_in_ctbs)?;
        check_tile_total("rowHeight", sizes.row_heights, height_in_ctbs)?;
        let total = u64::from(width_in_ctbs) * u64::from(height_in_ctbs);
        let count = usize::try_from(total).map_err(|_| HevcError::Allocation {
            what: "CTB addresses",
            count: usize::MAX,
        })?;
        let mut rs_to_ts = zeroed(count)?;
        let mut ts_to_rs = zeroed(count)?;
        let mut tile_id_rs = zeroed(count)?;

        // Tiles run in raster order, and CTBs in raster order inside each tile.
        // That order gives CtbAddrRsToTs of (6-7) and TileId of (6-9).
        let mut ts = 0u32;
        let mut tile = 0u32;
        let mut row_start = 0u32;
        for &tile_rows in sizes.row_heights {
            let mut column_start = 0u32;
            for &tile_columns in sizes.column_widths {
                for y in row_start..row_start + tile_rows {
                    for x in column_start..column_start + tile_columns {
                        let rs = (y * width_in_ctbs + x) as usize;
                        rs_to_ts[rs] = ts;
                        ts_to_rs[ts as usize] = rs as u32;
                        tile_id_rs[rs] = tile;
                        ts += 1;
                    }
                }
                column_start += tile_columns;
                tile += 1;
            }
            row_start += tile_rows;
        }
        Ok(Self {
            width: sizes.width,
            height: sizes.height,
            ctb_log2: sizes.ctb_log2,
            min_tb_log2: sizes.min_tb_log2,
            width_in_ctbs,
            rs_to_ts,
            ts_to_rs,
            tile_id_rs,
        })
    }

    /// `pic_width_in_luma_samples`.
    pub(crate) const fn width(&self) -> u32 {
        self.width
    }

    /// `pic_height_in_luma_samples`.
    pub(crate) const fn height(&self) -> u32 {
        self.height
    }

    /// `CtbLog2SizeY`.
    pub(crate) const fn ctb_log2(&self) -> u32 {
        self.ctb_log2
    }

    /// `PicWidthInCtbsY`.
    pub(crate) const fn width_in_ctbs(&self) -> u32 {
        self.width_in_ctbs
    }

    /// `PicSizeInCtbsY`.
    pub(crate) fn ctb_count(&self) -> usize {
        self.rs_to_ts.len()
    }

    /// `CtbAddrRsToTs[ rs ]`, or `None` outside the picture.
    pub(crate) fn rs_to_ts(&self, rs: usize) -> Option<u32> {
        self.rs_to_ts.get(rs).copied()
    }

    /// `CtbAddrTsToRs[ ts ]`, or `None` outside the picture.
    pub(crate) fn ts_to_rs(&self, ts: usize) -> Option<u32> {
        self.ts_to_rs.get(ts).copied()
    }

    /// The raster address of the CTB covering a luma location.
    pub(crate) fn ctb_rs_at(&self, x: u32, y: u32) -> usize {
        let row = (y >> self.ctb_log2) as usize;
        let column = (x >> self.ctb_log2) as usize;
        row * self.width_in_ctbs as usize + column
    }

    /// `TileId` of the CTB at raster address `rs`.
    pub(crate) fn tile_id(&self, rs: usize) -> Option<u32> {
        self.tile_id_rs.get(rs).copied()
    }

    /// `MinTbAddrZs` of the minimum block covering a luma location (6-10).
    pub(crate) fn min_tb_addr_zs(&self, x: u32, y: u32) -> u64 {
        let ctb_bits = self.ctb_log2 - self.min_tb_log2;
        let ts = self.rs_to_ts(self.ctb_rs_at(x, y)).unwrap_or(0);
        (u64::from(ts) << (2 * ctb_bits)) + self.min_tb_addr_in_ctb(x, y)
    }

    /// The part of `MinTbAddrZs` from the location inside its CTB. Two locations in one
    /// CTB compare the same through this part alone.
    pub(crate) fn min_tb_addr_in_ctb(&self, x: u32, y: u32) -> u64 {
        let ctb_bits = self.ctb_log2 - self.min_tb_log2;
        let mask = (1u32 << ctb_bits) - 1;
        let x_in_ctb = ((x >> self.min_tb_log2) & mask) as usize;
        let y_in_ctb = ((y >> self.min_tb_log2) & mask) as usize;
        u64::from(SPREAD[x_in_ctb & 15]) + 2 * u64::from(SPREAD[y_in_ctb & 15])
    }
}

fn check_tile_total(element: &'static str, sizes: &[u32], total: u32) -> Result<(), HevcError> {
    let sum: u64 = sizes.iter().map(|&size| u64::from(size)).sum();
    if sum != u64::from(total) {
        return Err(HevcError::InvalidSyntax {
            element,
            value: sum as i64,
            allowed: "tile sizes that add up to the picture size in CTBs",
        });
    }
    Ok(())
}

fn zeroed(count: usize) -> Result<Vec<u32>, HevcError> {
    let mut table = Vec::new();
    table
        .try_reserve_exact(count)
        .map_err(|_| HevcError::Allocation {
            what: "CTB addresses",
            count,
        })?;
    table.resize(count, 0);
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(width: u32, height: u32, columns: &[u32], rows: &[u32]) -> PicLayout {
        PicLayout::new(&LayoutSizes {
            width,
            height,
            ctb_log2: 4,
            min_tb_log2: 2,
            column_widths: columns,
            row_heights: rows,
        })
        .unwrap()
    }

    #[test]
    fn one_tile_keeps_raster_order() {
        let layout = layout(40, 32, &[3], &[2]);
        assert_eq!(layout.ctb_count(), 6);
        for rs in 0..6 {
            assert_eq!(layout.rs_to_ts(rs), Some(rs as u32));
            assert_eq!(layout.ts_to_rs(rs), Some(rs as u32));
            assert_eq!(layout.tile_id(rs), Some(0));
        }
        assert_eq!(layout.rs_to_ts(6), None);
    }

    /// Two columns of widths 1 and 2 over two rows, by hand from (6-7). The
    /// grid in raster order is `0 2 3` over `1 4 5`.
    #[test]
    fn tiles_reorder_the_ctbs() {
        let layout = layout(48, 32, &[1, 2], &[2]);
        let expected = [0, 2, 3, 1, 4, 5];
        for (rs, &ts) in expected.iter().enumerate() {
            assert_eq!(layout.rs_to_ts(rs), Some(ts));
            assert_eq!(layout.ts_to_rs(ts as usize), Some(rs as u32));
        }
        let tiles = [0, 1, 1, 0, 1, 1];
        for (rs, &tile) in tiles.iter().enumerate() {
            assert_eq!(layout.tile_id(rs), Some(tile));
        }
    }

    #[test]
    fn tile_sizes_must_cover_the_picture() {
        let sizes = LayoutSizes {
            width: 48,
            height: 32,
            ctb_log2: 4,
            min_tb_log2: 2,
            column_widths: &[1, 1],
            row_heights: &[2],
        };
        assert!(matches!(
            PicLayout::new(&sizes),
            Err(HevcError::InvalidSyntax {
                element: "colWidth",
                value: 2,
                ..
            })
        ));
    }

    /// Values from (6-10), with four minimum blocks per CTB side.
    #[test]
    fn z_scan_addresses_interleave_x_and_y() {
        let layout = layout(32, 32, &[2], &[2]);
        assert_eq!(layout.min_tb_addr_zs(0, 0), 0);
        assert_eq!(layout.min_tb_addr_zs(4, 0), 1);
        assert_eq!(layout.min_tb_addr_zs(0, 4), 2);
        assert_eq!(layout.min_tb_addr_zs(4, 4), 3);
        assert_eq!(layout.min_tb_addr_zs(8, 0), 4);
        assert_eq!(layout.min_tb_addr_zs(12, 12), 15);
        assert_eq!(layout.min_tb_addr_zs(16, 0), 16);
        assert_eq!(layout.min_tb_addr_zs(0, 16), 32);
        assert_eq!(layout.min_tb_addr_zs(31, 31), 63);
    }

    #[test]
    fn z_scan_addresses_follow_the_tile_scan() {
        let layout = layout(48, 32, &[1, 2], &[2]);
        // The CTB at raster address 3 has tile scan address 1.
        assert_eq!(layout.min_tb_addr_zs(0, 16), 16);
        // The CTB at raster address 1 has tile scan address 2.
        assert_eq!(layout.min_tb_addr_zs(16, 0), 32);
    }
}
