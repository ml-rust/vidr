//! The reconstruct stage of one inter coding unit: motion (8.5.3.2), prediction
//! samples (8.5.3.3), and the residual (8.6.7).

use crate::decode::cpu::dsp::inter::interp::{
    RefBlock, RefPlane, interpolate_chroma, interpolate_luma,
};
use crate::decode::cpu::dsp::inter::weight::{
    TargetBlock, TargetPlane, put_bi, put_uni, put_weighted_bi, put_weighted_uni,
};
use crate::decode::cpu::dsp::residual_add::add_residual_row;
use crate::decode::cpu::dsp::sample::Sample;
use crate::decode::cpu::dsp::simd::{Cpu, with_simd};
use crate::decode::cpu::hevc::dpb::ref_lists::SliceRefs;
use crate::decode::cpu::hevc::motion::context::{MotionContext, PbGeometry};
use crate::decode::cpu::hevc::motion::field::MotionField;
use crate::decode::cpu::hevc::motion::temporal::TemporalContext;
use crate::decode::cpu::hevc::motion::types::PbMotion;
use crate::decode::cpu::hevc::picture::Picture;
use crate::decode::cpu::hevc::residual::{ResidualInput, residual_samples};
use crate::decode::cpu::hevc::syntax::records::{CtuRecords, CuRecord, PuRecord};

use super::ctu::{ReconContext, Reconstructor, component_qp};

/// What the inter stage reads and writes besides the records and the picture.
#[derive(Debug)]
pub(crate) struct InterContext<'a, S> {
    /// The lists of the current slice.
    pub(crate) refs: &'a SliceRefs<S>,
    /// The motion of the current picture.
    pub(crate) field: &'a mut MotionField,
    /// `PicOrderCntVal` of the current picture.
    pub(crate) poc: i32,
}

impl Reconstructor {
    /// Predicts each prediction block in order, then adds the residual.
    pub(super) fn inter_coding_unit<S: Sample>(
        &mut self,
        context: &ReconContext<'_>,
        inter: &mut InterContext<'_, S>,
        records: &CtuRecords,
        index: usize,
        picture: &mut Picture<S>,
    ) {
        let Some(cu) = records.cus.get(index) else {
            return;
        };
        let end = records
            .cus
            .get(index + 1)
            .map_or(records.pus.len(), |next| next.first_pu);
        let pus = records.pus.get(cu.first_pu..end).unwrap_or_default();
        for (part_idx, pu) in pus.iter().enumerate() {
            let pb = PbGeometry {
                x_cb: cu.x,
                y_cb: cu.y,
                n_cb: 1 << cu.log2_size,
                x_pb: pu.x,
                y_pb: pu.y,
                width: pu.width,
                height: pu.height,
                part_idx: part_idx as u32,
                part_mode: cu.part_mode,
            };
            let motion = motion_context(context, inter).pb_motion(&pb, pu);
            inter.field.fill(pu.x, pu.y, pu.width, pu.height, motion);
            self.predict_block(context.params.bit_depth, inter.refs, pu, &motion, picture);
        }
        self.add_residuals(context, records, index, cu, picture);
    }

    /// 8.5.3.3.1 for the three components of one prediction block.
    fn predict_block<S: Sample>(
        &mut self,
        bit_depth: u32,
        refs: &SliceRefs<S>,
        pu: &PuRecord,
        motion: &PbMotion,
        picture: &mut Picture<S>,
    ) {
        for c_idx in 0..3 {
            for list in 0..2 {
                if !motion.uses(list) {
                    continue;
                }
                let Some(frame) = refs.frames[list].get(motion.ref_idx[list] as usize) else {
                    // A zero prediction keeps output free of earlier blocks and thread count.
                    self.inter_samples[list].clear();
                    continue;
                };
                let source = frame.picture.plane(c_idx).view();
                self.interpolate_block(&source, c_idx, pu, motion, list, bit_depth);
            }
            let sub = if c_idx == 0 { 1 } else { 2 };
            let target = TargetBlock {
                x: pu.x as usize / sub,
                y: pu.y as usize / sub,
                width: pu.width as usize / sub,
                height: pu.height as usize / sub,
            };
            let mut plane = picture.plane_mut(c_idx).target();
            self.write_prediction(&mut plane, target, refs, motion, c_idx, bit_depth);
        }
    }

    /// (8-214) to (8-221) for one component and list. The chroma vector of 4:2:0 equals
    /// the luma vector, in units of eighth samples (8-210), (8-211).
    fn interpolate_block<S: Sample>(
        &mut self,
        source: &RefPlane<'_, S>,
        c_idx: usize,
        pu: &PuRecord,
        motion: &PbMotion,
        list: usize,
        bit_depth: u32,
    ) {
        let mv = motion.mv[list];
        let (mv_x, mv_y) = (i32::from(mv.x), i32::from(mv.y));
        let out = &mut self.inter_samples[list];
        if c_idx == 0 {
            let block = RefBlock {
                x: pu.x as i32 + (mv_x >> 2),
                y: pu.y as i32 + (mv_y >> 2),
                width: pu.width as usize,
                height: pu.height as usize,
            };
            let frac = ((mv_x & 3) as usize, (mv_y & 3) as usize);
            interpolate_luma(source, block, frac, bit_depth, &mut self.interp, out);
        } else {
            let block = RefBlock {
                x: pu.x as i32 / 2 + (mv_x >> 3),
                y: pu.y as i32 / 2 + (mv_y >> 3),
                width: pu.width as usize / 2,
                height: pu.height as usize / 2,
            };
            let frac = ((mv_x & 7) as usize, (mv_y & 7) as usize);
            interpolate_chroma(source, block, frac, bit_depth, &mut self.interp, out);
        }
    }

    /// 8.5.3.3.4: the default process, or the explicit one with a weight table.
    fn write_prediction<S: Sample>(
        &self,
        plane: &mut TargetPlane<'_, S>,
        target: TargetBlock,
        refs: &SliceRefs<S>,
        motion: &PbMotion,
        c_idx: usize,
        bit_depth: u32,
    ) {
        let [first, second] = &self.inter_samples;
        let weights = refs
            .weights
            .as_deref()
            .map(|weights| weights.block(c_idx, motion.ref_idx));
        match (motion.uses(0), motion.uses(1), weights) {
            (true, true, Some(weights)) => {
                put_weighted_bi(plane, target, [first, second], &weights, bit_depth);
            }
            (true, true, None) => put_bi(plane, target, [first, second], bit_depth),
            (true, false, Some(weights)) => {
                put_weighted_uni(plane, target, first, &weights, 0, bit_depth);
            }
            (true, false, None) => put_uni(plane, target, first, bit_depth),
            (false, true, Some(weights)) => {
                put_weighted_uni(plane, target, second, &weights, 1, bit_depth);
            }
            (false, true, None) => put_uni(plane, target, second, bit_depth),
            (false, false, _) => {}
        }
    }

    /// 8.6.7 for every transform block with residual in one coding unit.
    fn add_residuals<S: Sample>(
        &mut self,
        context: &ReconContext<'_>,
        records: &CtuRecords,
        index: usize,
        cu: &CuRecord,
        picture: &mut Picture<S>,
    ) {
        let end = records
            .cus
            .get(index + 1)
            .map_or(records.tus.len(), |next| next.first_tu);
        let params = context.params;
        let max_value = (1i32 << params.bit_depth) - 1;
        let cpu = Cpu::detected();
        for tu in records.tus.get(cu.first_tu..end).unwrap_or_default() {
            let tbs = records
                .tbs
                .get(tu.first_tb..tu.first_tb + tu.tb_count)
                .unwrap_or_default();
            for tb in tbs {
                let coeffs = records
                    .coeffs
                    .get(tb.first_coeff..tb.first_coeff + tb.coeff_count)
                    .unwrap_or_default();
                residual_samples(
                    &ResidualInput {
                        log2_size: tb.log2_size,
                        c_idx: tb.c_idx,
                        intra: false,
                        transquant_bypass: cu.transquant_bypass,
                        transform_skip: tb.transform_skip,
                        qp: component_qp(params, cu.qp_y, tb.c_idx),
                        bit_depth: params.bit_depth,
                        coeffs,
                        factors: context.factors,
                    },
                    &mut self.residual,
                );
                let sub = if tb.c_idx == 0 { 1 } else { 2 };
                let (x, y) = (tb.x as usize / sub, tb.y as usize / sub);
                let plane = picture.plane_mut(tb.c_idx);
                let size = 1usize << tb.log2_size;
                let residual = &self.residual;
                with_simd!(cpu, simd => {
                    for line in 0..size {
                        let target = plane
                            .row_mut(y + line)
                            .and_then(|samples| samples.get_mut(x..x + size));
                        let Some(target) = target else {
                            continue;
                        };
                        let added = residual.row(tb.log2_size, line);
                        add_residual_row(simd, target, added, max_value);
                    }
                });
            }
        }
    }
}

/// The inputs of motion derivation for the current slice.
fn motion_context<'a, S>(
    context: &ReconContext<'a>,
    inter: &'a InterContext<'_, S>,
) -> MotionContext<'a> {
    let col = inter.refs.col.as_ref();
    MotionContext {
        params: context.params,
        layout: context.layout,
        maps: context.maps,
        field: &*inter.field,
        refs: &inter.refs.lists,
        temporal: TemporalContext {
            col: col.map(|frame| &frame.col),
            col_poc: col.map_or(0, |frame| frame.poc),
            no_backward_pred: inter.refs.no_backward_pred,
            collocated_from_l0: inter.refs.collocated_from_l0,
            ctb_log2: context.params.ctb_log2,
            width: context.params.width,
            height: context.params.height,
            poc: inter.poc,
        },
        poc: inter.poc,
    }
}

#[cfg(test)]
#[path = "inter_tests.rs"]
mod tests;
