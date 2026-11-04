//! `INTRA_ANGULAR2` to `INTRA_ANGULAR34` (8.4.4.2.6).

use super::refs::{PredBlock, RefSamples};

/// `intraPredAngle` (Table 8-5), by `predModeIntra`.
const INTRA_PRED_ANGLE: [i32; 35] = [
    0, 0, 32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26, -32, -26, -21, -17, -13,
    -9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32,
];
/// `invAngle` (Table 8-6), by `predModeIntra`. Other modes hold 0.
const INV_ANGLE: [i32; 35] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4096, -1638, -910, -630, -482, -390, -315, -256, -315, -390,
    -482, -630, -910, -1638, -4096, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];
/// The first mode that predicts from the row above.
const FIRST_VERTICAL_MODE: usize = 18;
const INTRA_ANGULAR10: u8 = 10;
const INTRA_ANGULAR26: u8 = 26;
/// `ref[ 0 ]` sits here, so `ref[ -nTbS ]` to `ref[ 2 * nTbS + 1 ]` fit.
const REF_ORIGIN: i32 = 32;
const REF_LEN: usize = 98;

/// What the angular prediction of one block reads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AngularMode {
    /// `predModeIntra`, 2 to 34. Above 34 acts as 34.
    pub(crate) mode: u8,
    /// True for `cIdx` 0, which takes the boundary filter below an `nTbS` of 32.
    pub(crate) luma: bool,
    /// `( 1 << BitDepth ) - 1`.
    pub(crate) max_value: i32,
}

/// Writes `predSamples`. A mode below 18 predicts the block as its transpose: the
/// equations of the two branches differ only in the roles of `x` and `y`.
pub(crate) fn predict_angular(refs: &RefSamples, mode: AngularMode, out: &mut PredBlock) {
    let index = usize::from(mode.mode).min(34);
    let vertical = index >= FIRST_VERTICAL_MODE;
    let size = refs.size();
    let angle = INTRA_PRED_ANGLE[index];
    let reference = reference_array(refs, index, vertical);
    for (line, row) in out.rows_mut(size).enumerate() {
        let position = (line as i32 + 1) * angle;
        let start = (REF_ORIGIN + (position >> 5) + 1) as usize;
        let fraction = (position & 31) as u32;
        if fraction == 0 {
            row.copy_from_slice(&reference[start..start + size]);
            continue;
        }
        let near = &reference[start..start + size];
        let far = &reference[start + 1..start + 1 + size];
        for ((sample, &a), &b) in row.iter_mut().zip(near).zip(far) {
            let value = (32 - fraction) * u32::from(a) + fraction * u32::from(b) + 16;
            *sample = (value >> 5) as u16;
        }
    }
    if !vertical {
        out.transpose(size);
    }
    if mode.luma && size < 32 {
        boundary_filter(refs, mode, out);
    }
}

/// `ref` of (8-53) to (8-55) and (8-61) to (8-63), offset by [`REF_ORIGIN`].
fn reference_array(refs: &RefSamples, index: usize, vertical: bool) -> [u16; REF_LEN] {
    let size = refs.size() as i32;
    let corner = refs.corner_index() as i32;
    // The main side grows along the line in one direction, the other side in the
    // other direction.
    let direction = if vertical { 1 } else { -1 };
    let main = |position: i32| refs.at((corner + direction * position) as usize);
    let side = |position: i32| refs.at((corner - direction * position) as usize);
    let angle = INTRA_PRED_ANGLE[index];
    let mut reference = [0u16; REF_LEN];
    let top = if angle < 0 { size } else { 2 * size };
    for position in 0..=top {
        reference[(REF_ORIGIN + position) as usize] = main(position);
    }
    let last = (size * angle) >> 5;
    if last < -1 {
        for position in last..0 {
            let distance = (position * INV_ANGLE[index] + 128) >> 8;
            reference[(REF_ORIGIN + position) as usize] = side(distance);
        }
    }
    reference
}

/// (8-60) and (8-68).
fn boundary_filter(refs: &RefSamples, mode: AngularMode, out: &mut PredBlock) {
    let size = refs.size();
    let corner = i32::from(refs.corner());
    let clip = |value: i32| value.clamp(0, mode.max_value) as u16;
    if mode.mode == INTRA_ANGULAR26 {
        let base = i32::from(refs.top(0));
        for y in 0..size {
            let value = base + ((i32::from(refs.left(y)) - corner) >> 1);
            out.set(size, 0, y, clip(value));
        }
    } else if mode.mode == INTRA_ANGULAR10 {
        let base = i32::from(refs.left(0));
        for x in 0..size {
            let value = base + ((i32::from(refs.top(x)) - corner) >> 1);
            out.set(size, x, 0, clip(value));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::refs::tests::filled;
    use super::*;

    const fn chroma(mode: u8) -> AngularMode {
        AngularMode {
            mode,
            luma: false,
            max_value: 255,
        }
    }

    /// Left rows hold 100 + y and top columns hold 200 + x. The corner holds 50.
    fn ramp(log2_size: u32) -> RefSamples {
        let mut refs = filled(log2_size, 0, 50, 0);
        for index in 0..2 * refs.size() {
            refs.set_left(index, 100 + index as u16);
            refs.set_top(index, 200 + index as u16);
        }
        refs
    }

    #[test]
    fn mode_26_copies_the_row_above() {
        let refs = ramp(2);
        let mut out = PredBlock::new();
        predict_angular(&refs, chroma(26), &mut out);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(out.get(4, x, y), 200 + x as u16);
            }
        }
    }

    #[test]
    fn mode_10_copies_the_left_column() {
        let refs = ramp(2);
        let mut out = PredBlock::new();
        predict_angular(&refs, chroma(10), &mut out);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(out.get(4, x, y), 100 + y as u16);
            }
        }
    }

    /// Angle 32: each row moves one sample along the top: ( x + y + 2 )th reference.
    #[test]
    fn mode_34_follows_the_top_diagonal() {
        let refs = ramp(2);
        let mut out = PredBlock::new();
        predict_angular(&refs, chroma(34), &mut out);
        assert_eq!(out.get(4, 0, 0), 201);
        assert_eq!(out.get(4, 3, 3), 207);
        assert_eq!(out.get(4, 1, 2), 204);
    }

    /// Angle 32 in the horizontal branch: the same along the left column.
    #[test]
    fn mode_2_follows_the_left_diagonal() {
        let refs = ramp(2);
        let mut out = PredBlock::new();
        predict_angular(&refs, chroma(2), &mut out);
        assert_eq!(out.get(4, 0, 0), 101);
        assert_eq!(out.get(4, 3, 3), 107);
        assert_eq!(out.get(4, 2, 1), 104);
    }

    /// Angle -32: ref[ -1 ] and below come from the left column through invAngle
    /// -256, so ref[ x ] = p[ -1 ][ -1 - x ].
    #[test]
    fn mode_18_extends_the_reference_through_the_left_column() {
        let refs = ramp(2);
        let mut out = PredBlock::new();
        predict_angular(&refs, chroma(18), &mut out);
        // y = 0: iIdx -1, so x = 0 reads ref[ 0 ], the corner.
        assert_eq!(out.get(4, 0, 0), 50);
        assert_eq!(out.get(4, 1, 0), 200);
        // y = 1: x = 0 reads ref[ -1 ] = p[ -1 ][ 0 ].
        assert_eq!(out.get(4, 0, 1), 100);
        // y = 3: x = 0 reads ref[ -3 ] = p[ -1 ][ 2 ].
        assert_eq!(out.get(4, 0, 3), 102);
        assert_eq!(out.get(4, 3, 3), 50);
    }

    /// Angle 5: y = 0 gives iIdx 0 and iFact 5.
    /// ( 27 * 200 + 5 * 201 + 16 ) >> 5 = 200.
    /// y = 3: position 20, ( 12 * 200 + 20 * 201 + 16 ) >> 5 = 201.
    #[test]
    fn a_fractional_angle_interpolates() {
        let refs = ramp(2);
        let mut out = PredBlock::new();
        predict_angular(&refs, chroma(28), &mut out);
        assert_eq!(out.get(4, 0, 0), 200);
        assert_eq!(out.get(4, 0, 3), 201);
        // y = 3, x = 3: ( 12 * 203 + 20 * 204 + 16 ) >> 5 = 204.
        assert_eq!(out.get(4, 3, 3), 204);
    }

    /// (8-60): the first column adds half the left gradient, with clipping.
    #[test]
    fn mode_26_filters_the_first_luma_column() {
        let refs = ramp(2);
        let mut out = PredBlock::new();
        let mode = AngularMode {
            mode: 26,
            luma: true,
            max_value: 255,
        };
        predict_angular(&refs, mode, &mut out);
        // 200 + ( ( 100 - 50 ) >> 1 ) = 225.
        assert_eq!(out.get(4, 0, 0), 225);
        // 200 + ( ( 103 - 50 ) >> 1 ) = 226.
        assert_eq!(out.get(4, 0, 3), 226);
        assert_eq!(out.get(4, 1, 0), 201);
        let low = AngularMode {
            max_value: 210,
            ..mode
        };
        predict_angular(&refs, low, &mut out);
        assert_eq!(out.get(4, 0, 0), 210);
    }

    /// (8-68): the first row adds half the top gradient.
    #[test]
    fn mode_10_filters_the_first_luma_row() {
        let refs = ramp(2);
        let mut out = PredBlock::new();
        let mode = AngularMode {
            mode: 10,
            luma: true,
            max_value: 255,
        };
        predict_angular(&refs, mode, &mut out);
        // 100 + ( ( 200 - 50 ) >> 1 ) = 175.
        assert_eq!(out.get(4, 0, 0), 175);
        // 100 + ( ( 203 - 50 ) >> 1 ) = 176.
        assert_eq!(out.get(4, 3, 0), 176);
        assert_eq!(out.get(4, 0, 1), 101);
    }

    #[test]
    fn a_32x32_block_takes_no_boundary_filter() {
        let refs = ramp(5);
        let mut out = PredBlock::new();
        let mode = AngularMode {
            mode: 26,
            luma: true,
            max_value: 255,
        };
        predict_angular(&refs, mode, &mut out);
        assert_eq!(out.get(32, 0, 0), 200);
    }

    /// Every mode at every size stays inside the reference array.
    #[test]
    fn every_mode_and_size_runs() {
        for log2_size in 2..=5 {
            let refs = ramp(log2_size);
            let mut out = PredBlock::new();
            for mode in 2..=34 {
                predict_angular(&refs, chroma(mode), &mut out);
            }
        }
    }
}
