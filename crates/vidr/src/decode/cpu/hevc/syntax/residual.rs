//! `residual_coding()` (7.3.8.11) with sign data hiding and transform skip.

use crate::decode::cpu::dsp::scale::{COEFF_MAX, COEFF_MIN};
use crate::decode::cpu::hevc::cabac::context as ctx;
use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::scan::{SCAN_VERTICAL, scan_order, scan_position};

use super::parser::CtuParser;
use super::records::{Coeff, TbRecord, push_record};
use super::residual_ctx::{
    SigCtx, coded_sub_block_ctx_inc, greater1_ctx_inc, greater1_ctx_set, greater2_ctx_inc,
    last_prefix_ctx,
};

/// `Log2MaxTransformSkipSize`: `log2_max_transform_skip_block_size_minus2` is 0
/// in Main and Main 10 (A.3.2, A.3.3).
const LOG2_MAX_TRANSFORM_SKIP_SIZE: u32 = 2;
/// Coefficients per sub-block.
const SUB_BLOCK_COEFFS: usize = 16;
/// The most `coeff_abs_level_greater1_flag` of one sub-block (7.3.8.11).
const MAX_GREATER1_FLAGS: usize = 8;
/// The largest `cRiceParam` (9-24).
const MAX_RICE_PARAM: u32 = 4;

/// The arguments of `residual_coding()`.
#[derive(Clone, Copy, Debug)]
pub(super) struct TbArgs {
    pub(super) x0: u32,
    pub(super) y0: u32,
    pub(super) log2_size: u32,
    pub(super) c_idx: usize,
    pub(super) scan_idx: usize,
    /// `cu_transquant_bypass_flag`.
    pub(super) transquant_bypass: bool,
}

/// `coded_sub_block_flag`, one bit per sub-block.
#[derive(Clone, Copy)]
struct CodedSubBlocks {
    bits: u64,
    /// The largest `xS` and `yS`.
    last: u32,
}

impl CodedSubBlocks {
    const fn get(self, x: u32, y: u32) -> bool {
        (self.bits >> ((y << 3) + x)) & 1 == 1
    }

    const fn set(&mut self, x: u32, y: u32) {
        self.bits |= 1 << ((y << 3) + x);
    }

    /// The flag of the right neighbour, and of the lower neighbour.
    const fn neighbours(self, x: u32, y: u32) -> (bool, bool) {
        (
            x < self.last && self.get(x + 1, y),
            y < self.last && self.get(x, y + 1),
        )
    }
}

/// The significant positions of one sub-block, in parsing order: highest `n`
/// first.
struct SigList {
    positions: [u8; SUB_BLOCK_COEFFS],
    count: usize,
}

impl SigList {
    const fn new() -> Self {
        Self {
            positions: [0; SUB_BLOCK_COEFFS],
            count: 0,
        }
    }

    /// `count` never exceeds 16, because each `n` enters at most once.
    fn push(&mut self, position: usize) {
        if let Some(slot) = self.positions.get_mut(self.count) {
            *slot = position as u8;
            self.count += 1;
        }
    }

    fn as_slice(&self) -> &[u8] {
        self.positions.get(..self.count).unwrap_or_default()
    }
}

impl CtuParser<'_, '_> {
    /// Parses one transform block, and appends its record and levels.
    pub(super) fn residual_coding(&mut self, args: &TbArgs) -> Result<(), HevcError> {
        let (log2_size, c_idx, scan_idx) = (args.log2_size, args.c_idx, args.scan_idx);
        let transform_skip = self.params.transform_skip
            && !args.transquant_bypass
            && log2_size <= LOG2_MAX_TRANSFORM_SKIP_SIZE
            && self
                .cabac
                .decision(ctx::TRANSFORM_SKIP + usize::from(c_idx > 0));
        let (x_last, y_last) = self.last_significant_position(log2_size, c_idx, scan_idx);
        let sub_scan = scan_order(log2_size - 2, scan_idx);
        let position_scan = scan_order(2, scan_idx);
        let last_sub_block = scan_position(sub_scan, x_last >> 2, y_last >> 2);
        let last_position = scan_position(position_scan, x_last & 3, y_last & 3);
        let first_coeff = self.records.coeffs.len();
        let mut coded = CodedSubBlocks {
            bits: 0,
            last: (1 << (log2_size - 2)) - 1,
        };
        let mut last_greater1_ctx = 1;
        for index in (0..=last_sub_block).rev() {
            let [x_s, y_s] = sub_scan[index];
            let (x_s, y_s) = (u32::from(x_s), u32::from(y_s));
            let (right, below) = coded.neighbours(x_s, y_s);
            let mut sub_block_coded = true;
            let mut infer_dc = false;
            if index < last_sub_block && index > 0 {
                let inc = coded_sub_block_ctx_inc(right, below, c_idx);
                sub_block_coded = self.cabac.decision(ctx::CODED_SUB_BLOCK + inc);
                infer_dc = true;
            }
            if sub_block_coded {
                coded.set(x_s, y_s);
            }
            let prev_csbf = usize::from(right) + 2 * usize::from(below);
            let sig_ctx = SigCtx::new(log2_size, c_idx, scan_idx, (x_s, y_s), prev_csbf);
            let mut sig = SigList::new();
            let first_parsed = if index == last_sub_block {
                sig.push(last_position);
                last_position
            } else {
                SUB_BLOCK_COEFFS
            };
            if sub_block_coded {
                self.sig_coeff_flags(position_scan, &sig_ctx, first_parsed, infer_dc, &mut sig);
            }
            if sig.count == 0 {
                continue;
            }
            let ctx_set = greater1_ctx_set(index, c_idx, last_greater1_ctx);
            self.sub_block_levels(
                &SubBlock {
                    origin: (x_s << 2, y_s << 2),
                    position_scan,
                    positions: sig.as_slice(),
                    ctx_set,
                    c_idx,
                    transquant_bypass: args.transquant_bypass,
                },
                &mut last_greater1_ctx,
            )?;
        }
        let record = TbRecord {
            x: args.x0,
            y: args.y0,
            log2_size,
            c_idx,
            transform_skip,
            first_coeff,
            coeff_count: self.records.coeffs.len() - first_coeff,
        };
        push_record(&mut self.records.tbs, record, "transform block records")
    }

    /// The last significant coefficient's location (7-74 to 7-78).
    fn last_significant_position(
        &mut self,
        log2_size: u32,
        c_idx: usize,
        scan_idx: usize,
    ) -> (u32, u32) {
        let x_prefix = self.last_prefix(ctx::LAST_X_PREFIX, log2_size, c_idx);
        let y_prefix = self.last_prefix(ctx::LAST_Y_PREFIX, log2_size, c_idx);
        let x = self.last_position(x_prefix);
        let y = self.last_position(y_prefix);
        if scan_idx == SCAN_VERTICAL {
            (y, x)
        } else {
            (x, y)
        }
    }

    /// `last_sig_coeff_x_prefix` or `last_sig_coeff_y_prefix`: TR with `cMax`
    /// `( log2TrafoSize << 1 ) - 1`.
    fn last_prefix(&mut self, first_context: usize, log2_size: u32, c_idx: usize) -> u32 {
        let (offset, shift) = last_prefix_ctx(log2_size, c_idx);
        let c_max = (log2_size << 1) - 1;
        let mut prefix = 0;
        while prefix < c_max {
            let inc = (prefix >> shift) + offset;
            if !self.cabac.decision(first_context + inc as usize) {
                break;
            }
            prefix += 1;
        }
        prefix
    }

    /// `LastSignificantCoeffX` or `LastSignificantCoeffY` from a prefix, reading
    /// the suffix when the prefix is above 3 (7-74, 7-75).
    fn last_position(&mut self, prefix: u32) -> u32 {
        if prefix <= 3 {
            return prefix;
        }
        let suffix_bits = (prefix >> 1) - 1;
        (1 << suffix_bits) * (2 + (prefix & 1)) + self.cabac.bypass_bits(suffix_bits)
    }

    /// `sig_coeff_flag` for the positions below `first_parsed`, highest first.
    fn sig_coeff_flags(
        &mut self,
        scan: &[[u8; 2]],
        sig_ctx: &SigCtx,
        first_parsed: usize,
        mut infer_dc: bool,
        sig: &mut SigList,
    ) {
        for position in (0..first_parsed).rev() {
            if position > 0 || !infer_dc {
                let [x_p, y_p] = scan[position];
                let inc = sig_ctx.inc(usize::from(x_p), usize::from(y_p));
                if self.cabac.decision(ctx::SIG_COEFF + inc) {
                    sig.push(position);
                    infer_dc = false;
                }
            } else {
                // inferSbDcSigCoeffFlag is still 1, so this flag is inferred as 1.
                sig.push(position);
            }
        }
    }

    /// Parses the levels of one sub-block with significant coefficients.
    fn sub_block_levels(
        &mut self,
        sub_block: &SubBlock<'_>,
        last_greater1_ctx: &mut u32,
    ) -> Result<(), HevcError> {
        let (positions, c_idx) = (sub_block.positions, sub_block.c_idx);
        let count = positions.len();
        let mut base_levels = [1u32; SUB_BLOCK_COEFFS];
        let mut greater1_ctx = 1u32;
        let mut first_greater1 = None;
        for (index, base_level) in base_levels
            .iter_mut()
            .enumerate()
            .take(count.min(MAX_GREATER1_FLAGS))
        {
            let inc = greater1_ctx_inc(sub_block.ctx_set, greater1_ctx, c_idx);
            if self.cabac.decision(ctx::GREATER1 + inc) {
                *base_level = 2;
                first_greater1 = first_greater1.or(Some(index));
                greater1_ctx = 0;
            } else if greater1_ctx > 0 {
                greater1_ctx += 1;
            }
        }
        *last_greater1_ctx = greater1_ctx;
        if let Some(index) = first_greater1 {
            let inc = greater2_ctx_inc(sub_block.ctx_set, c_idx);
            if self.cabac.decision(ctx::GREATER2 + inc) {
                base_levels[index] = 3;
            }
        }
        let first_position = positions.last().copied().unwrap_or(0);
        let last_position = positions.first().copied().unwrap_or(0);
        let sign_hidden = self.params.sign_data_hiding
            && !sub_block.transquant_bypass
            && last_position - first_position > 3;
        let sign_count = count - usize::from(sign_hidden);
        // Sign bins in the top bits, in parsing order. A hidden sign reads as 0.
        let signs = self.cabac.bypass_bits(sign_count as u32) << (u32::BITS as usize - sign_count);
        let mut sum_abs_level = 0u64;
        let mut rice = 0;
        for (index, &position) in positions.iter().enumerate() {
            let base_level = base_levels[index];
            // The condition on baseLevel in 7.3.8.11.
            let threshold = if index >= MAX_GREATER1_FLAGS {
                1
            } else if first_greater1 == Some(index) {
                3
            } else {
                2
            };
            let abs_level = if base_level == threshold {
                let remaining = self.coeff_abs_level_remaining(rice)?;
                let abs_level = base_level + remaining;
                // (9-24): the next call reads this level.
                if abs_level > 3 << rice {
                    rice = (rice + 1).min(MAX_RICE_PARAM);
                }
                abs_level
            } else {
                base_level
            };
            sum_abs_level += u64::from(abs_level);
            let mut negative = (signs >> (u32::BITS as usize - 1 - index)) & 1 == 1;
            // Sign data hiding: the parity gives the sign of the last level.
            if sign_hidden && index == count - 1 && sum_abs_level & 1 == 1 {
                negative = true;
            }
            let level = if negative {
                -i64::from(abs_level)
            } else {
                i64::from(abs_level)
            };
            if !(i64::from(COEFF_MIN)..=i64::from(COEFF_MAX)).contains(&level) {
                return Err(HevcError::InvalidSyntax {
                    element: "TransCoeffLevel",
                    value: level,
                    allowed: "-32768 to 32767",
                });
            }
            let [x_p, y_p] = sub_block.position_scan[usize::from(position)];
            let coeff = Coeff {
                x: sub_block.origin.0 as u8 + x_p,
                y: sub_block.origin.1 as u8 + y_p,
                level: level as i16,
            };
            push_record(&mut self.records.coeffs, coeff, "coefficients")?;
        }
        Ok(())
    }

    /// `coeff_abs_level_remaining` (9.3.3.11) without extended precision.
    fn coeff_abs_level_remaining(&mut self, rice: u32) -> Result<u32, HevcError> {
        let prefix = self.bypass_unary(4);
        if prefix < 4 {
            return Ok((prefix << rice) + self.cabac.bypass_bits(rice));
        }
        let suffix = self.exp_golomb(rice + 1, "coeff_abs_level_remaining")?;
        Ok((4 << rice) + suffix)
    }
}

/// What the level syntax of one sub-block reads.
struct SubBlock<'a> {
    /// `( xS << 2, yS << 2 )`.
    origin: (u32, u32),
    position_scan: &'a [[u8; 2]],
    /// The significant positions, highest first.
    positions: &'a [u8],
    ctx_set: usize,
    c_idx: usize,
    transquant_bypass: bool,
}
