//! The residual of one transform block from its parsed levels (8.6.2).

use super::scaling::{FLAT_FACTOR, ScalingFactors, matrix_id};
use super::syntax::records::Coeff;
use crate::decode::cpu::dsp::scale::ScaleStep;
use crate::decode::cpu::dsp::transform::{
    CoeffExtent, TbBlock, inverse_dct, inverse_dst, round_residual, transform_skip,
};

/// What 8.6.2 reads for one transform block.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ResidualInput<'a> {
    /// `Log2( nTbS )`.
    pub(crate) log2_size: u32,
    /// `cIdx`.
    pub(crate) c_idx: usize,
    /// True when `CuPredMode` is `MODE_INTRA`.
    pub(crate) intra: bool,
    /// `cu_transquant_bypass_flag`.
    pub(crate) transquant_bypass: bool,
    /// `transform_skip_flag`.
    pub(crate) transform_skip: bool,
    /// `qP` (8-291 to 8-293).
    pub(crate) qp: u32,
    /// `BitDepth` of the component.
    pub(crate) bit_depth: u32,
    /// The levels that are not zero.
    pub(crate) coeffs: &'a [Coeff],
    /// `None` with `scaling_list_enabled_flag` 0.
    pub(crate) factors: Option<&'a ScalingFactors>,
}

/// Writes the residual `r` of (8.6.2) to `block`.
pub(crate) fn residual_samples(input: &ResidualInput<'_>, block: &mut TbBlock) {
    let log2_size = input.log2_size;
    block.clear(log2_size);
    if input.transquant_bypass {
        for coeff in input.coeffs {
            let (x, y) = (usize::from(coeff.x), usize::from(coeff.y));
            block.set(log2_size, x, y, i32::from(coeff.level));
        }
        return;
    }
    let extent = scale_levels(input, block);
    if input.transform_skip {
        transform_skip(log2_size, block);
    } else if input.intra && log2_size == 2 && input.c_idx == 0 {
        inverse_dst(block);
    } else {
        inverse_dct(log2_size, block, extent);
    }
    round_residual(log2_size, input.bit_depth, block);
}

/// The scaling process (8.6.3). Returns the extent of the levels it wrote.
fn scale_levels(input: &ResidualInput<'_>, block: &mut TbBlock) -> CoeffExtent {
    let log2_size = input.log2_size;
    let step = ScaleStep::new(input.qp, input.bit_depth, log2_size);
    // A transform skip block above 4x4 takes the flat factor. Main has no such
    // block, because Log2MaxTransformSkipSize is 2.
    let flat = input.transform_skip && log2_size > 2;
    let matrix = input
        .factors
        .filter(|_| !flat)
        .map(|factors| factors.matrix(log2_size, matrix_id(input.intra, input.c_idx)));
    let mut extent = CoeffExtent { max_x: 0, max_y: 0 };
    for coeff in input.coeffs {
        let (x, y) = (usize::from(coeff.x), usize::from(coeff.y));
        extent.max_x = extent.max_x.max(x);
        extent.max_y = extent.max_y.max(y);
        let m = matrix
            .and_then(|matrix| matrix.get((y << log2_size) + x))
            .map_or(FLAT_FACTOR, |&factor| i32::from(factor));
        block.set(log2_size, x, y, step.scale(i32::from(coeff.level), m));
    }
    extent
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::hevc_scaling::ScalingListData;

    fn input(coeffs: &[Coeff]) -> ResidualInput<'_> {
        ResidualInput {
            log2_size: 3,
            c_idx: 0,
            intra: false,
            transquant_bypass: false,
            transform_skip: false,
            qp: 30,
            bit_depth: 8,
            coeffs,
            factors: None,
        }
    }

    fn rows(log2_size: u32, block: &TbBlock) -> Vec<i32> {
        let size = 1usize << log2_size;
        (0..size * size)
            .map(|index| block.get(log2_size, index % size, index / size))
            .collect()
    }

    /// qP 30: levelScale[ 0 ] << 5 = 1280. bdShift 6 at 8x8.
    /// d = ( 4 * 16 * 1280 + 32 ) >> 6 = 1280.
    /// Column pass: ( 1280 * 64 + 64 ) >> 7 = 640. Row pass: 640 * 64 = 40960.
    /// (8-299): ( 40960 + 2048 ) >> 12 = 10.
    #[test]
    fn a_dc_level_gives_a_flat_residual() {
        let coeffs = [Coeff {
            x: 0,
            y: 0,
            level: 4,
        }];
        let mut block = TbBlock::new();
        residual_samples(&input(&coeffs), &mut block);
        assert_eq!(rows(3, &block), vec![10; 64]);
    }

    /// Bypass copies the levels, with no scaling and no rounding (8.6.2).
    #[test]
    fn bypass_copies_the_levels() {
        let coeffs = [
            Coeff {
                x: 3,
                y: 1,
                level: -7,
            },
            Coeff {
                x: 0,
                y: 2,
                level: 300,
            },
        ];
        let mut block = TbBlock::new();
        let mut request = input(&coeffs);
        request.log2_size = 2;
        request.transquant_bypass = true;
        residual_samples(&request, &mut block);
        let mut expected = vec![0; 16];
        expected[7] = -7;
        expected[8] = 300;
        assert_eq!(rows(2, &block), expected);
    }

    /// qP 30, bdShift 5 at 4x4. d = ( 2 * 16 * 1280 + 16 ) >> 5 = 1280.
    /// r = ( 1280 << 7 + 2048 ) >> 12 = 40.
    #[test]
    fn transform_skip_takes_no_transform() {
        let coeffs = [Coeff {
            x: 2,
            y: 3,
            level: 2,
        }];
        let mut block = TbBlock::new();
        let mut request = input(&coeffs);
        request.log2_size = 2;
        request.transform_skip = true;
        residual_samples(&request, &mut block);
        let mut expected = vec![0; 16];
        expected[14] = 40;
        assert_eq!(rows(2, &block), expected);
    }

    /// An intra 4x4 luma block takes the DST, and any other block the DCT.
    #[test]
    fn only_intra_4x4_luma_takes_the_dst() {
        let coeffs = [Coeff {
            x: 0,
            y: 0,
            level: 8,
        }];
        let mut dst_request = input(&coeffs);
        dst_request.log2_size = 2;
        dst_request.intra = true;
        let mut block = TbBlock::new();
        residual_samples(&dst_request, &mut block);
        // The first DST basis rises along each axis.
        assert!(block.get(2, 0, 0) < block.get(2, 3, 3));
        let mut chroma = dst_request;
        chroma.c_idx = 1;
        residual_samples(&chroma, &mut block);
        assert_eq!(block.get(2, 0, 0), block.get(2, 3, 3));
        let mut inter = dst_request;
        inter.intra = false;
        residual_samples(&inter, &mut block);
        assert_eq!(block.get(2, 0, 0), block.get(2, 3, 3));
    }

    /// The scaling list changes `m` at the coefficient's own location.
    /// d = ( 4 * 32 * 1280 + 32 ) >> 6 = 2560 at DC with factor 32.
    #[test]
    fn a_scaling_list_sets_the_factor() {
        let mut data = ScalingListData::default_lists();
        data.scaling_list[1][3][0] = 32;
        let factors = ScalingFactors::new(&data);
        let coeffs = [Coeff {
            x: 0,
            y: 0,
            level: 4,
        }];
        let mut block = TbBlock::new();
        let mut request = input(&coeffs);
        request.factors = Some(&factors);
        residual_samples(&request, &mut block);
        // Column pass: ( 2560 * 64 + 64 ) >> 7 = 1280. Row pass: 81920.
        // ( 81920 + 2048 ) >> 12 = 20.
        assert_eq!(rows(3, &block), vec![20; 64]);
    }

    /// A stale block from an earlier call leaves no trace.
    #[test]
    fn the_block_clears_first() {
        let mut block = TbBlock::new();
        block.set(3, 5, 5, 999);
        let mut request = input(&[]);
        request.transquant_bypass = true;
        residual_samples(&request, &mut block);
        assert_eq!(rows(3, &block), vec![0; 64]);
    }
}
