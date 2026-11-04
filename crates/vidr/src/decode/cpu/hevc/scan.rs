//! Scan order arrays (6.5.3 to 6.5.5). Each entry is `[ x, y ]`.

/// `scanIdx` of the up-right diagonal scan.
pub(crate) const SCAN_DIAGONAL: usize = 0;
/// `scanIdx` of the horizontal scan.
pub(crate) const SCAN_HORIZONTAL: usize = 1;
/// `scanIdx` of the vertical scan.
pub(crate) const SCAN_VERTICAL: usize = 2;

/// The up-right diagonal scan of a block of `SIZE` positions (6-11).
const fn diagonal<const SIZE: usize>(block: i32) -> [[u8; 2]; SIZE] {
    let mut scan = [[0u8; 2]; SIZE];
    let mut index = 0;
    let mut x = 0;
    let mut y = 0;
    while index < SIZE {
        while y >= 0 {
            if x < block && y < block {
                scan[index] = [x as u8, y as u8];
                index += 1;
            }
            y -= 1;
            x += 1;
        }
        y = x;
        x = 0;
    }
    scan
}

/// The horizontal scan (6-12) with `vertical` false, or the vertical scan (6-13)
/// with `vertical` true.
const fn raster<const SIZE: usize>(block: usize, vertical: bool) -> [[u8; 2]; SIZE] {
    let mut scan = [[0u8; 2]; SIZE];
    let mut index = 0;
    while index < SIZE {
        let outer = (index / block) as u8;
        let inner = (index % block) as u8;
        scan[index] = if vertical {
            [outer, inner]
        } else {
            [inner, outer]
        };
        index += 1;
    }
    scan
}

static SCAN_1: [[[u8; 2]; 1]; 3] = [diagonal(1), raster(1, false), raster(1, true)];
static SCAN_2: [[[u8; 2]; 4]; 3] = [diagonal(2), raster(2, false), raster(2, true)];
static SCAN_4: [[[u8; 2]; 16]; 3] = [diagonal(4), raster(4, false), raster(4, true)];
static SCAN_8: [[[u8; 2]; 64]; 3] = [diagonal(8), raster(8, false), raster(8, true)];

/// `ScanOrder[ log2_block ][ scan_idx ]`. A `log2_block` above 3 selects 3, and a
/// `scan_idx` above 2 selects 2.
pub(crate) fn scan_order(log2_block: u32, scan_idx: usize) -> &'static [[u8; 2]] {
    let scan = if scan_idx > 2 { 2 } else { scan_idx };
    match log2_block {
        0 => &SCAN_1[scan],
        1 => &SCAN_2[scan],
        2 => &SCAN_4[scan],
        _ => &SCAN_8[scan],
    }
}

/// The scan position of `[ x, y ]` in `scan`, or 0 for a location outside it.
pub(crate) fn scan_position(scan: &[[u8; 2]], x: u32, y: u32) -> usize {
    scan.iter()
        .position(|&entry| u32::from(entry[0]) == x && u32::from(entry[1]) == y)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_4x4_goes_up_and_right() {
        assert_eq!(
            scan_order(2, SCAN_DIAGONAL),
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [0, 2],
                [1, 1],
                [2, 0],
                [0, 3],
                [1, 2],
                [2, 1],
                [3, 0],
                [1, 3],
                [2, 2],
                [3, 1],
                [2, 3],
                [3, 2],
                [3, 3],
            ]
        );
    }

    #[test]
    fn horizontal_fills_rows_and_vertical_fills_columns() {
        assert_eq!(
            scan_order(1, SCAN_HORIZONTAL),
            [[0, 0], [1, 0], [0, 1], [1, 1]]
        );
        assert_eq!(
            scan_order(1, SCAN_VERTICAL),
            [[0, 0], [0, 1], [1, 0], [1, 1]]
        );
        assert_eq!(scan_order(2, SCAN_HORIZONTAL)[5], [1, 1]);
        assert_eq!(scan_order(2, SCAN_HORIZONTAL)[6], [2, 1]);
        assert_eq!(scan_order(2, SCAN_VERTICAL)[6], [1, 2]);
    }

    #[test]
    fn every_scan_visits_every_position_once() {
        for log2_block in 0..4 {
            let size = 1u32 << log2_block;
            for scan_idx in 0..3 {
                let scan = scan_order(log2_block, scan_idx);
                assert_eq!(scan.len() as u32, size * size);
                for y in 0..size {
                    for x in 0..size {
                        let position = scan_position(scan, x, y);
                        assert_eq!(scan[position], [x as u8, y as u8]);
                    }
                }
            }
        }
    }

    #[test]
    fn the_8x8_diagonal_ends_at_the_far_corner() {
        let scan = scan_order(3, SCAN_DIAGONAL);
        assert_eq!(scan[1], [0, 1]);
        assert_eq!(scan[2], [1, 0]);
        assert_eq!(scan[63], [7, 7]);
    }
}
