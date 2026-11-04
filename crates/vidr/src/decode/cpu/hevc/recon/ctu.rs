//! Reconstruction of one CTU from its records (8.4.4.1, 8.6.7).

use crate::decode::cpu::dsp::intra::angular::{AngularMode, predict_angular};
use crate::decode::cpu::dsp::intra::planar_dc::{predict_dc, predict_planar};
use crate::decode::cpu::dsp::intra::refs::{PredBlock, RefSamples};
use crate::decode::cpu::dsp::sample::Sample;
use crate::decode::cpu::dsp::transform::TbBlock;
use crate::decode::cpu::hevc::layout::PicLayout;
use crate::decode::cpu::hevc::params::{CodingParams, PcmParams};
use crate::decode::cpu::hevc::picture::{Picture, Plane};
use crate::decode::cpu::hevc::qp::{qp_prime_c, qp_prime_y};
use crate::decode::cpu::hevc::residual::{ResidualInput, residual_samples};
use crate::decode::cpu::hevc::scaling::ScalingFactors;
use crate::decode::cpu::hevc::syntax::maps::ParseMaps;
use crate::decode::cpu::hevc::syntax::records::{
    CtuRecords, CuRecord, PredMode, TbRecord, TuRecord,
};

use super::neighbours::{Availability, BlockAt, read_neighbours};

const INTRA_PLANAR: u8 = 0;
const INTRA_DC: u8 = 1;

/// What reconstruction reads besides the records.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReconContext<'a> {
    pub(crate) params: &'a CodingParams,
    pub(crate) layout: &'a PicLayout,
    pub(crate) maps: &'a ParseMaps,
    /// `None` with `scaling_list_enabled_flag` 0.
    pub(crate) factors: Option<&'a ScalingFactors>,
}

/// One block of 8.4.4.1 in its plane.
#[derive(Clone, Copy, Debug)]
struct IntraBlock<'a> {
    c_idx: usize,
    x: usize,
    y: usize,
    log2_size: u32,
    /// `predModeIntra`.
    mode: u8,
    /// The residual of this block, when one is coded.
    tb: Option<&'a TbRecord>,
}

/// The working memory of the reconstruct stage.
#[derive(Clone, Debug)]
pub(crate) struct Reconstructor {
    refs: RefSamples,
    prediction: PredBlock,
    residual: TbBlock,
}

impl Reconstructor {
    /// Working memory for any block size.
    pub(crate) const fn new() -> Self {
        Self {
            refs: RefSamples::new(),
            prediction: PredBlock::new(),
            residual: TbBlock::new(),
        }
    }

    /// Writes the samples of every intra coding unit in `records` to `picture`.
    /// A later unit adds the stage for inter coding units.
    pub(crate) fn reconstruct_ctu<S: Sample>(
        &mut self,
        context: &ReconContext<'_>,
        records: &CtuRecords,
        picture: &mut Picture<S>,
    ) {
        for (index, cu) in records.cus.iter().enumerate() {
            if cu.pred_mode != PredMode::Intra {
                continue;
            }
            if cu.pcm {
                if let Some(pcm) = context.params.pcm {
                    write_pcm(cu, records, pcm, context.params.bit_depth, picture);
                }
                continue;
            }
            let end = records
                .cus
                .get(index + 1)
                .map_or(records.tus.len(), |next| next.first_tu);
            let tus = records.tus.get(cu.first_tu..end).unwrap_or_default();
            for tu in tus {
                self.intra_transform_unit(context, records, cu, tu, picture);
            }
        }
    }

    /// One transform unit: luma, then Cb and Cr where this unit holds them.
    fn intra_transform_unit<S: Sample>(
        &mut self,
        context: &ReconContext<'_>,
        records: &CtuRecords,
        cu: &CuRecord,
        tu: &TuRecord,
        picture: &mut Picture<S>,
    ) {
        let tbs = records
            .tbs
            .get(tu.first_tb..tu.first_tb + tu.tb_count)
            .unwrap_or_default();
        let find = |c_idx: usize| tbs.iter().find(|tb| tb.c_idx == c_idx);
        let half = (1u32 << cu.log2_size) >> 1;
        let part = usize::from(tu.x - cu.x >= half) + 2 * usize::from(tu.y - cu.y >= half);
        let luma = IntraBlock {
            c_idx: 0,
            x: tu.x as usize,
            y: tu.y as usize,
            log2_size: tu.log2_size,
            mode: cu.intra_luma[part],
            tb: find(0),
        };
        self.intra_block(context, records, cu, &luma, picture.luma_mut());
        // A 4x4 luma unit shares one chroma block with its siblings (7.3.8.10).
        let chroma_at = if tu.log2_size > 2 {
            Some((tu.x, tu.y, tu.log2_size - 1))
        } else if tu.blk_idx == 3 {
            Some((tu.x_base, tu.y_base, 2))
        } else {
            None
        };
        let Some((x, y, log2_size)) = chroma_at else {
            return;
        };
        for c_idx in [1, 2] {
            let block = IntraBlock {
                c_idx,
                x: x as usize / 2,
                y: y as usize / 2,
                log2_size,
                mode: cu.intra_chroma,
                tb: find(c_idx),
            };
            self.intra_block(context, records, cu, &block, picture.plane_mut(c_idx));
        }
    }

    /// Prediction (8.4.4.2), residual (8.6.2), and picture construction (8.6.7).
    fn intra_block<S: Sample>(
        &mut self,
        context: &ReconContext<'_>,
        records: &CtuRecords,
        cu: &CuRecord,
        block: &IntraBlock<'_>,
        plane: &mut Plane<S>,
    ) {
        self.predict(context, block, plane);
        let size = 1usize << block.log2_size;
        let max_value = (1i32 << context.params.bit_depth) - 1;
        let mut residual = None;
        if let Some(tb) = block.tb {
            let coeffs = records
                .coeffs
                .get(tb.first_coeff..tb.first_coeff + tb.coeff_count)
                .unwrap_or_default();
            residual_samples(
                &ResidualInput {
                    log2_size: block.log2_size,
                    c_idx: block.c_idx,
                    intra: true,
                    transquant_bypass: cu.transquant_bypass,
                    transform_skip: tb.transform_skip,
                    qp: component_qp(context.params, cu.qp_y, block.c_idx),
                    bit_depth: context.params.bit_depth,
                    coeffs,
                    factors: context.factors,
                },
                &mut self.residual,
            );
            residual = Some(&self.residual);
        }
        for (line, row) in self.prediction.rows(size).enumerate() {
            let Some(target) = plane
                .row_mut(block.y + line)
                .and_then(|samples| samples.get_mut(block.x..block.x + size))
            else {
                return;
            };
            if let Some(block_residual) = residual {
                let added = block_residual.row(block.log2_size, line);
                for ((out, &sample), &value) in target.iter_mut().zip(row).zip(added) {
                    *out = S::clipped(i32::from(sample) + value, max_value);
                }
            } else {
                for (out, &sample) in target.iter_mut().zip(row) {
                    *out = S::clipped(i32::from(sample), max_value);
                }
            }
        }
    }

    /// The general intra sample prediction (8.4.4.2.1).
    fn predict<S: Sample>(
        &mut self,
        context: &ReconContext<'_>,
        block: &IntraBlock<'_>,
        plane: &Plane<S>,
    ) {
        let params = context.params;
        let luma = block.c_idx == 0;
        read_neighbours(
            plane,
            &Availability {
                params,
                layout: context.layout,
                maps: context.maps,
            },
            BlockAt {
                x: block.x,
                y: block.y,
                log2_size: block.log2_size,
                sub: if luma { 1 } else { 2 },
            },
            &mut self.refs,
        );
        if luma {
            self.refs
                .filter_for_mode(block.mode, params.strong_intra_smoothing, params.bit_depth);
        }
        match block.mode {
            INTRA_PLANAR => predict_planar(&self.refs, block.log2_size, &mut self.prediction),
            INTRA_DC => predict_dc(&self.refs, block.log2_size, luma, &mut self.prediction),
            mode => predict_angular(
                &self.refs,
                AngularMode {
                    mode,
                    luma,
                    max_value: (1 << params.bit_depth) - 1,
                },
                &mut self.prediction,
            ),
        }
    }
}

/// `qP` of (8-291) to (8-293) for `cIdx`.
fn component_qp(params: &CodingParams, qp_y: i32, c_idx: usize) -> u32 {
    let bd_offset = params.qp_bd_offset();
    match c_idx {
        0 => qp_prime_y(qp_y, bd_offset),
        1 => qp_prime_c(qp_y, params.cb_qp_offset, bd_offset),
        _ => qp_prime_c(qp_y, params.cr_qp_offset, bd_offset),
    }
}

/// The PCM samples of one coding unit (8-12, and the chroma form after it).
fn write_pcm<S: Sample>(
    cu: &CuRecord,
    records: &CtuRecords,
    pcm: PcmParams,
    bit_depth: u32,
    picture: &mut Picture<S>,
) {
    let size = 1usize << cu.log2_size;
    let chroma_size = size / 2;
    let luma_count = size * size;
    let chroma_count = chroma_size * chroma_size;
    let start = cu.first_pcm_sample;
    let Some(samples) = records
        .pcm_samples
        .get(start..start + luma_count + 2 * chroma_count)
    else {
        return;
    };
    let (luma, chroma) = samples.split_at(luma_count);
    let (cb, cr) = chroma.split_at(chroma_count);
    let max_value = (1i32 << bit_depth) - 1;
    let (x, y) = (cu.x as usize, cu.y as usize);
    let luma_shift = bit_depth.saturating_sub(pcm.bit_depth_luma);
    let chroma_shift = bit_depth.saturating_sub(pcm.bit_depth_chroma);
    let write = |plane: &mut Plane<S>, source: &[u16], at: (usize, usize), n: usize, shift: u32| {
        for (line, values) in source.chunks_exact(n).enumerate() {
            let target = plane
                .row_mut(at.1 + line)
                .and_then(|row| row.get_mut(at.0..at.0 + n));
            if let Some(target) = target {
                for (out, &value) in target.iter_mut().zip(values) {
                    *out = S::clipped(i32::from(value) << shift, max_value);
                }
            }
        }
    };
    write(picture.luma_mut(), luma, (x, y), size, luma_shift);
    write(
        picture.cb_mut(),
        cb,
        (x / 2, y / 2),
        chroma_size,
        chroma_shift,
    );
    write(
        picture.cr_mut(),
        cr,
        (x / 2, y / 2),
        chroma_size,
        chroma_shift,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::cpu::hevc::layout::LayoutSizes;
    use crate::decode::cpu::hevc::syntax::records::{Coeff, PartMode};

    fn cu(log2_size: u32) -> CuRecord {
        CuRecord {
            x: 0,
            y: 0,
            log2_size,
            pred_mode: PredMode::Intra,
            part_mode: PartMode::TwoNxTwoN,
            transquant_bypass: false,
            pcm: false,
            qp_y: 26,
            intra_luma: [INTRA_DC; 4],
            intra_chroma: INTRA_DC,
            first_pu: 0,
            first_tu: 0,
            first_pcm_sample: 0,
        }
    }

    fn tu(x: u32, y: u32, log2_size: u32, blk_idx: u32) -> TuRecord {
        TuRecord {
            x,
            y,
            x_base: 0,
            y_base: 0,
            log2_size,
            depth: 0,
            blk_idx,
            cbf_luma: false,
            cbf_cb: false,
            cbf_cr: false,
            cu: 0,
            first_tb: 0,
            tb_count: 0,
        }
    }

    struct Fixture {
        params: CodingParams,
        layout: PicLayout,
        maps: ParseMaps,
    }

    impl Fixture {
        /// A 16x16 picture of one CTB, in slice 0.
        fn new() -> Self {
            let params = CodingParams::for_tests();
            let layout = PicLayout::new(&LayoutSizes {
                width: 16,
                height: 16,
                ctb_log2: 4,
                min_tb_log2: 2,
                column_widths: &[1],
                row_heights: &[1],
            })
            .unwrap();
            let mut maps = ParseMaps::new(&params, &layout).unwrap();
            maps.set_ctb_slice(0, 0);
            Self {
                params,
                layout,
                maps,
            }
        }

        fn context(&self) -> ReconContext<'_> {
            ReconContext {
                params: &self.params,
                layout: &self.layout,
                maps: &self.maps,
                factors: None,
            }
        }
    }

    /// With no neighbours, DC predicts mid-grey in each plane.
    #[test]
    fn a_first_block_with_no_residual_is_mid_grey() {
        let fixture = Fixture::new();
        let records = CtuRecords {
            cus: vec![cu(4)],
            tus: vec![tu(0, 0, 4, 0)],
            ..CtuRecords::default()
        };
        let mut picture = Picture::<u8>::new(16, 16).unwrap();
        Reconstructor::new().reconstruct_ctu(&fixture.context(), &records, &mut picture);
        assert!(picture.luma().row(15).unwrap().iter().all(|&s| s == 128));
        assert!(picture.cb().row(7).unwrap().iter().all(|&s| s == 128));
        assert!(picture.cr().row(0).unwrap().iter().all(|&s| s == 128));
    }

    #[test]
    fn ten_bit_mid_grey_is_512() {
        let mut fixture = Fixture::new();
        fixture.params.bit_depth = 10;
        let records = CtuRecords {
            cus: vec![cu(4)],
            tus: vec![tu(0, 0, 4, 0)],
            ..CtuRecords::default()
        };
        let mut picture = Picture::<u16>::new(16, 16).unwrap();
        Reconstructor::new().reconstruct_ctu(&fixture.context(), &records, &mut picture);
        assert_eq!(picture.luma().row(3).unwrap()[3], 512);
        assert_eq!(picture.cr().row(3).unwrap()[3], 512);
    }

    /// (8-299) with a bypass residual adds one level to one sample.
    #[test]
    fn a_bypass_residual_adds_to_the_prediction() {
        let fixture = Fixture::new();
        let mut unit = cu(4);
        unit.transquant_bypass = true;
        let mut transform = tu(0, 0, 4, 0);
        transform.tb_count = 1;
        let records = CtuRecords {
            cus: vec![unit],
            tus: vec![transform],
            tbs: vec![TbRecord {
                x: 0,
                y: 0,
                log2_size: 4,
                c_idx: 0,
                transform_skip: false,
                first_coeff: 0,
                coeff_count: 1,
            }],
            coeffs: vec![Coeff {
                x: 5,
                y: 2,
                level: -100,
            }],
            ..CtuRecords::default()
        };
        let mut picture = Picture::<u8>::new(16, 16).unwrap();
        Reconstructor::new().reconstruct_ctu(&fixture.context(), &records, &mut picture);
        assert_eq!(picture.luma().row(2).unwrap()[5], 28);
        assert_eq!(picture.luma().row(2).unwrap()[6], 128);
    }

    /// Four 4x4 luma units in an 8x8 unit. Chroma waits for the fourth, and each
    /// luma unit reads the mode of its own quadrant.
    #[test]
    fn nxn_units_use_their_own_modes_and_one_chroma_block() {
        let fixture = Fixture::new();
        let mut unit = cu(3);
        unit.part_mode = PartMode::NxN;
        unit.intra_luma = [INTRA_DC, 26, 10, INTRA_PLANAR];
        let records = CtuRecords {
            cus: vec![unit],
            tus: vec![
                tu(0, 0, 2, 0),
                tu(4, 0, 2, 1),
                tu(0, 4, 2, 2),
                tu(4, 4, 2, 3),
            ],
            ..CtuRecords::default()
        };
        let mut picture = Picture::<u8>::new(16, 16).unwrap();
        fill(picture.cb_mut(), 7);
        Reconstructor::new().reconstruct_ctu(&fixture.context(), &records, &mut picture);
        // The first block is mid-grey, and each later block predicts from it.
        assert_eq!(picture.luma().row(0).unwrap()[0], 128);
        assert_eq!(picture.luma().row(3).unwrap()[7], 128);
        assert_eq!(picture.luma().row(7).unwrap()[3], 128);
        // One 4x4 chroma block, at the base location.
        assert_eq!(picture.cb().row(3).unwrap()[3], 128);
        assert_eq!(picture.cb().row(4).unwrap()[3], 7);
        assert_eq!(picture.cb().row(3).unwrap()[4], 7);
    }

    fn fill(plane: &mut Plane<u8>, value: u8) {
        for y in 0..plane.height() {
            plane.row_mut(y).unwrap().fill(value);
        }
    }

    /// (8-12) shifts each PCM sample up to the bit depth.
    #[test]
    fn pcm_samples_shift_to_the_bit_depth() {
        let mut fixture = Fixture::new();
        fixture.params.pcm = Some(PcmParams {
            log2_min_size: 3,
            log2_max_size: 3,
            bit_depth_luma: 5,
            bit_depth_chroma: 7,
        });
        let mut unit = cu(3);
        unit.pcm = true;
        let mut samples: Vec<u16> = vec![3; 64];
        samples.extend(vec![10u16; 16]);
        samples.extend(vec![20u16; 16]);
        let records = CtuRecords {
            cus: vec![unit],
            pcm_samples: samples,
            ..CtuRecords::default()
        };
        let mut picture = Picture::<u8>::new(16, 16).unwrap();
        Reconstructor::new().reconstruct_ctu(&fixture.context(), &records, &mut picture);
        assert_eq!(picture.luma().row(7).unwrap()[7], 3 << 3);
        assert_eq!(picture.luma().row(8).unwrap()[7], 0);
        assert_eq!(picture.cb().row(3).unwrap()[3], 10 << 1);
        assert_eq!(picture.cr().row(0).unwrap()[0], 20 << 1);
    }

    /// With `constrained_intra_pred_flag` 1, a block that is not intra gives no
    /// reference samples (8.4.4.2.1).
    #[test]
    fn constrained_intra_pred_ignores_other_blocks() {
        let mut fixture = Fixture::new();
        fixture.params.constrained_intra_pred = true;
        fixture.maps.set_cu_kind(0, 0, 8, false, false);
        let mut unit = cu(3);
        unit.x = 8;
        let records = CtuRecords {
            cus: vec![unit],
            tus: vec![TuRecord {
                x_base: 8,
                ..tu(8, 0, 3, 0)
            }],
            ..CtuRecords::default()
        };
        let mut picture = Picture::<u8>::new(16, 16).unwrap();
        fill(picture.luma_mut(), 40);
        Reconstructor::new().reconstruct_ctu(&fixture.context(), &records, &mut picture);
        assert_eq!(picture.luma().row(0).unwrap()[8], 128);
        // The same block reads the left neighbour when that neighbour is intra.
        fixture.maps.set_cu_kind(0, 0, 8, true, false);
        fill(picture.luma_mut(), 40);
        Reconstructor::new().reconstruct_ctu(&fixture.context(), &records, &mut picture);
        assert_eq!(picture.luma().row(0).unwrap()[8], 40);
    }
}
